import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matrix,FULL_CONFIGS,viewMatrix} from '../src/full-config.mjs';
import {attentionFull} from '../src/full-engine.mjs';
import {simpleAttentionData,attentionMixPrefix,simpleAttentionMarkup} from '../src/simple-attention.mjs';
import {ATTENTION_WINDOW_SIZE,attentionWindow,attentionWindowForToken,windowFromScroll,attentionWindowData,attentionWindowKey,windowTokenLabel,attentionWindowMarkup} from '../src/attention-window.mjs';
function fixture(model='pixtral',n=48){
 const c=FULL_CONFIGS[model],make=seed=>matrix(n,c.width,Float32Array.from({length:n*c.width},(_,i)=>Math.sin((i+seed)*.053)));
 const q=make(7),k=make(19),v=make(123),{heads}=attentionFull(q,k,v,c);
 const l={input:make(4),rawQ:q,rawK:k,q,k,values:v,heads,afterAttention:make(9)},run={config:c,layers:[l]},s={selected:21,key:22,head:5,layer:0,expanded:3,attentionStart:19,attentionPanels:{},headFeature:0,feature:0};
 return {run,s,data:simpleAttentionData(run,s)};
}
test('window size is six tokens and clamps cleanly at both ends',()=>{
 assert.equal(ATTENTION_WINDOW_SIZE,6);
 assert.deepEqual(attentionWindow(48,-20).indices,[0,1,2,3,4,5]);assert.deepEqual(attentionWindow(48,999).indices,[42,43,44,45,46,47]);assert.equal(attentionWindow(48,42).after,0);assert.equal(attentionWindow(48,0).before,0);
 for(const total of [1,2,5,6,12,37,48,64,192])for(const start of [-1,0,3,total-1,total+99]){const w=attentionWindow(total,start);assert.equal(w.count,Math.min(6,total));assert.equal(w.before+w.count+w.after,total);assert.ok(w.indices.every(i=>i>=0&&i<total));}
});
test('manual patch selection stays in its current window unless it leaves it',()=>{
 assert.equal(attentionWindowForToken(48,19,21).start,19);assert.equal(attentionWindowForToken(48,19,24).start,19);
 assert.equal(attentionWindowForToken(48,19,47).start,42);assert.equal(attentionWindowForToken(48,19,0).start,0);assert.equal(attentionWindowForToken(48,null,21).start,19);
});
test('scroll position maps to token windows without changing or selecting a token',()=>{
 assert.equal(windowFromScroll(0,46,48).start,0);assert.equal(windowFromScroll(46*12,46,48).start,12);assert.equal(windowFromScroll(46*99,46,48).start,42);assert.equal(windowFromScroll(-40,46,48).start,0);
});
for(const model of ['pixtral','vit']){
 test(`${model}: displayed cells retain the original full-matrix softmax values, never re-normalized`,()=>{
  const {data}=fixture(model),before=data.head.attention.data.slice();
  for(const start of [0,19,42]){
   const w=attentionWindowData(data,start);assert.equal(w.weights.length,6);assert.ok(w.weights.every(row=>row.length===6));
   for(let i=0;i<6;i++)for(let j=0;j<6;j++)assert.equal(w.weights[i][j],data.head.attention.data[w.indices[i]*data.n+w.indices[j]]);
   for(const row of w.weights)assert.ok(row.reduce((a,b)=>a+b,0)<1);
   let visible=0,total=0;for(let k=0;k<data.n;k++){const value=data.head.attention.data[data.query*data.n+k];total+=value;if(k>=w.start&&k<w.end)visible+=value;}
   assert.equal(w.visibleMass,visible);assert.equal(w.hiddenMass,total-visible);assert.equal(w.totalMass,total);assert.ok(w.hiddenMass>0);
  }
  assert.deepEqual(data.head.attention.data,before);
 });
 test(`${model}: full output and automatic value mixing still include off-window keys`,()=>{
  const {data}=fixture(model),before=data.head.out.data.slice();const sums=[];
  for(const start of [0,19,42]){attentionWindowData(data,start);sums.push(attentionMixPrefix(data,data.n,63));}
  assert.ok(sums.every(v=>v===sums[0]));assert.ok(Math.abs(sums[0]-data.head.out.data[data.query*64+63])<1e-6);assert.deepEqual(data.head.out.data,before);
 });
 test(`${model}: magnified grid contains exactly six visible token paths and 36 clickable cells`,()=>{
  const {data,s}=fixture(model),html=attentionWindowMarkup(data,s);
  const cells=[...html.matchAll(/data-window-pair="(\d+):(\d+)" data-window-weight="([^"]+)"/g)];assert.equal(cells.length,36);
  for(const [,q,k,v] of cells){assert.ok(+q>=19&&+q<25);assert.ok(+k>=19&&+k<25);assert.equal(+v,data.head.attention.data[+q*48+ +k]);}
  for(const kind of ['q','k','v','out'])assert.equal((html.match(new RegExp(`data-window-kind="${kind}"`,'g'))||[]).length,6);
  for(const wire of ['q','k','v','out'])assert.equal((html.match(new RegExp(`data-window-wire="${wire}-`,'g'))||[]).length,6);
  assert.match(html,/Full 48 × 48/);assert.match(html,/sum ALL 48 keys/);assert.match(html,/not masked/);assert.match(html,/hover a cell for its weight/);assert.equal((html.match(/class="aw-faded-context"/g)||[]).length,2);
 });
 test(`${model}: p0 and CLS retain the right identity rather than being confused with row zero`,()=>{
  const {data}=fixture(model),html=attentionWindowMarkup(data,{attentionStart:0});
  assert.equal(windowTokenLabel(0,data.c),model==='vit'?'CLS':'p00');assert.equal(windowTokenLabel(1,data.c),model==='vit'?'p00':'p01');assert.match(html,new RegExp(`Inspect Q for ${windowTokenLabel(0,data.c)}`));
 });
}
test('off-window query remains selected and is explicitly labeled, with a way to follow it',()=>{
 const {data}=fixture(),html=attentionWindowMarkup(data,{attentionStart:0});
 assert.match(html,/Query <b>p21<\/b> \(off-window\)/);assert.match(html,/data-window-follow/);assert.match(html,/data-window-pair="0:0"/);
 assert.match(html,/tabindex="0" data-window-pair="0:0"/);
});
test('last window reaches the last token and disables only forward window movement',()=>{
 const {data}=fixture(),html=attentionWindowMarkup(data,{attentionStart:999});assert.match(html,/data-window-start="42"/);assert.match(html,/data-window-pair="47:47"/);assert.match(html,/data-window-step="6" disabled/);assert.doesNotMatch(html,/data-window-step="-6" disabled/);
});
test('grid keyboard navigation uses global coordinates and respects full token bounds',()=>{
 assert.deepEqual(attentionWindowKey(19,23,'ArrowRight',48),{query:19,key:24});assert.deepEqual(attentionWindowKey(19,23,'ArrowDown',48),{query:20,key:23});assert.deepEqual(attentionWindowKey(0,0,'ArrowUp',48),{query:0,key:0});assert.deepEqual(attentionWindowKey(47,47,'ArrowRight',48),{query:47,key:47});assert.equal(attentionWindowKey(19,23,'Tab',48),null);
});
test('small image sequences show every real token without invented padding',()=>{
 const {data}=fixture();for(const n of [1,3,6]){const small={...data,n,query:0,key:0,score:0,weight:.1,head:{...data.head,attention:matrix(n,n,new Float32Array(n*n).fill(1/n))}};const html=attentionWindowMarkup(small,{attentionStart:50});assert.equal((html.match(/data-window-pair=/g)||[]).length,n*n);assert.doesNotMatch(html,/aw-faded-context|NaN|undefined/);}
});
test('window does not recreate dense score and attention canvases on the main flow',()=>{
 const {run,s}=fixture(),seen=[],html=simpleAttentionMarkup({markup(...args){seen.push(args);return ''; }},run,s,()=>{});
 assert.equal(seen.length,0);assert.equal((html.match(/data-attention-overview/g)||[]).length,1);assert.equal((html.match(/data-window-pair=/g)||[]).length,36);assert.match(html,/data-attention-window/);assert.doesNotMatch(html,/class="sa-score"/);
});
test('scroll binding is local and disposable; offline and active page use the same window',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),[source,app,bundle]=await Promise.all(['src/attention-window.mjs','src/full-app.mjs','standalone.html'].map(read));
 assert.match(source,/rail.addEventListener\('scroll',scroll/);assert.match(source,/rail.removeEventListener\('scroll',scroll/);assert.match(source,/win.removeEventListener\('resize',resize/);
 assert.match(app,/patchWindowController\?\.dispose\(\)/);assert.match(app,/onStart:start=>\{state.attentionStart=start;\}/);assert.match(bundle,/function mountAttentionWindow/);assert.match(bundle,/Grey context is out of view, not masked/);
});
