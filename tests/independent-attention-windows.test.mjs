import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matrix,FULL_CONFIGS} from '../src/full-config.mjs';
import {attentionFull} from '../src/full-engine.mjs';
import {simpleAttentionData,attentionMixPrefix} from '../src/simple-attention.mjs';
import {attentionWindowData,attentionOutputWindow,attentionWindowMarkup,drawAttentionOverview} from '../src/attention-window.mjs';
import {withCausalAttention,attentionStages} from '../src/attention-stages.mjs';
import {attentionPanOffset} from '../src/attention-window-motion.mjs';
function fixture(model='pixtral'){
 const c=FULL_CONFIGS[model],n=c.cls?37:48,make=seed=>matrix(n,c.width,Float32Array.from({length:n*c.width},(_,i)=>Math.sin((i+seed)*.027))),q=make(7),k=make(31),v=make(101),{heads}=attentionFull(q,k,v,c),l={input:make(3),q,k,values:v,heads};
 return simpleAttentionData({config:c,layers:[l]},{selected:21,key:22,layer:0,head:2});
}
for(const model of ['pixtral','vit'])test(`${model}: query and key windows form an off-diagonal crop with unmodified full-row weights`,()=>{
 const data=fixture(model),original=data.head.attention.data.slice(),w=attentionWindowData(data,2,25);
 assert.equal(w.start,2);assert.equal(w.keys.start,25);assert.equal(w.weights.length,6);assert.ok(w.weights.every(row=>row.length===6));
 for(let i=0;i<6;i++)for(let j=0;j<6;j++)assert.equal(w.weights[i][j],data.head.attention.data[(i+2)*data.n+j+25]);
 const queryMoved=attentionWindowData(data,8,25),keyMoved=attentionWindowData(data,2,19);assert.deepEqual(queryMoved.keys,w.keys);assert.equal(keyMoved.start,w.start);
 let shownMass=0;for(let k=25;k<31;k++)shownMass+=data.head.attention.data[data.query*data.n+k];assert.equal(w.visibleMass,shownMass);assert.equal(queryMoved.visibleMass,shownMass);assert.ok(w.visibleMass<1);assert.deepEqual(data.head.attention.data,original);
});
test('Q/output rows and K/V rows use their respective windows and select global cell coordinates',()=>{
 const data=fixture(),html=attentionWindowMarkup(data,{attentionStart:2,attentionKeyStart:25,attentionOutputFeature:56});
 for(const [kind,start] of [['q',2],['out',2],['k',25],['v',25]]){
  const rows=[...html.matchAll(/data-window-token="(\d+)" data-window-kind="([a-z]+)"/g)].filter(m=>m[2]===kind).map(m=>+m[1]);assert.deepEqual(rows,Array.from({length:6},(_,i)=>i+start));
 }
 const cells=[...html.matchAll(/data-window-pair="(\d+):(\d+)" data-window-weight="([^"]+)"/g)];assert.equal(cells.length,36);for(const [,q,k,value] of cells){assert.ok(+q>=2&&+q<8);assert.ok(+k>=25&&+k<31);assert.equal(+value,data.head.attention.data[+q*data.n+ +k]);}
 assert.match(html,/Query <b>p21<\/b> \(off-window\)/);assert.match(html,/key <b>p22<\/b> \(off-window\)/);
});
test('the output snippet displays eight actual stored features and discloses full dimensions',()=>{
 const data=fixture(),before=data.head.out.data.slice(),window=attentionOutputWindow(data,19,56),html=attentionWindowMarkup(data,{attentionStart:19,attentionKeyStart:0,attentionOutputFeature:56});
 assert.deepEqual(window.features.indices,[56,57,58,59,60,61,62,63]);assert.equal(window.rows.start,19);assert.ok(window.values.every(row=>row.length===8));
 const cells=[...html.matchAll(/data-output-cell="(\d+):(\d+)" data-output-value="([^"]+)"/g)];assert.equal(cells.length,48);for(const [,r,c,v] of cells)assert.equal(+v,data.head.out.data[+r*64+ +c]);
 assert.match(html,/view 6 × 8/);assert.match(html,/full 48 × 64/);assert.match(html,/features 56–63/);assert.match(html,/rows follow Query/);assert.match(html,/Output\[19, 56\]/);assert.deepEqual(data.head.out.data,before);
});
test('output feature and row offsets clamp to valid tensor ranges',()=>{
 const data=fixture();assert.equal(attentionOutputWindow(data,999,999).rows.start,42);assert.equal(attentionOutputWindow(data,999,999).features.start,56);assert.equal(attentionOutputWindow(data,-4,-10).features.start,0);
 const small={...data,n:2,c:{...data.c,headWidth:3},head:{...data.head,out:matrix(2,3,new Float32Array([1,2,3,4,5,6]))}},w=attentionOutputWindow(small,50,50);assert.deepEqual(w.values,[[1,2,3],[4,5,6]]);assert.equal(w.features.count,3);assert.equal(w.rows.count,2);
});
test('changing key-window visibility does not change the output or omit off-window values',()=>{
 const data=fixture(),output=data.head.out.data.slice(),sums=[];
 for(const start of [0,12,25,42]){attentionWindowData(data,19,start);sums.push(attentionMixPrefix(data,data.n,63));}
 assert.ok(sums.every(x=>x===sums[0]));assert.ok(Math.abs(sums[0]-data.head.out.data[data.query*64+63])<1e-6);assert.deepEqual(output,data.head.out.data);
});
test('all three calculation stages support independent offsets and retain exact stored arithmetic',()=>{
 const data=fixture(),stages=attentionStages(data),html=attentionWindowMarkup(data,{attentionStart:2,attentionKeyStart:25,attentionBreakdown:true});
 const cells=[...html.matchAll(/data-window-pair="(\d+):(\d+)" data-window-weight="([^"]+)" data-attention-stage="([a-z]+)" data-stage-value="([^"]+)"/g)];assert.equal(cells.length,108);
 for(const [,q,k,weight,stage,value] of cells){const at=+q*data.n+ +k;assert.equal(+value,stages[stage].data[at]);assert.equal(+weight,data.head.attention.data[at]);}
 assert.match(html,/Attention_weights × V/);assert.match(html,/weights = softmax probabilities/);assert.match(html,/sum ALL 48 keys/);
});
test('causal comparison uses global query/key coordinates, not window-local offsets',()=>{
 const data=withCausalAttention(fixture()),html=attentionWindowMarkup(data,{attentionStart:2,attentionKeyStart:25,attentionBreakdown:true});
 const weights=[...html.matchAll(/data-window-weight="([^"]+)" data-attention-stage="softmax" data-stage-value="([^"]+)" data-stage-blocked="([^"]+)"/g)];assert.equal(weights.length,36);for(const [,w,value,blocked] of weights){assert.equal(+w,0);assert.equal(+value,0);assert.equal(blocked,'true');}
 assert.match(html,/Comparison output/);assert.match(html,/view 6 × 8/);
});
test('independent scroll animations move only rows or only columns, and honor the output row pitch',()=>{
 assert.deepEqual(attentionPanOffset('cell',{query:1,key:0}),{x:0,y:29});assert.deepEqual(attentionPanOffset('cell',{query:0,key:2}),{x:58,y:0});
 assert.deepEqual(attentionPanOffset('q',{query:0,key:2}),{x:0,y:0});assert.deepEqual(attentionPanOffset('k',{query:2,key:0}),{x:0,y:0});assert.deepEqual(attentionPanOffset('v',{query:0,key:-1}),{x:0,y:-17});assert.deepEqual(attentionPanOffset('out',{query:1,key:8},17),{x:0,y:17});
});
test('the minimap highlights the correct off-diagonal rectangle',()=>{
 const data=fixture(),calls=[],context={clearRect(){},drawImage(){},save(){},beginPath(){},rect(...args){calls.push(['rect',...args]);},clip(){},restore(){},strokeRect(...args){calls.push(['stroke',...args]);}},canvas={width:84,dataset:{},getContext:()=>context},doc={createElement:()=>({getContext:()=>({putImageData(){}})})},root={querySelector:()=>canvas,ownerDocument:doc};
 const previous=globalThis.ImageData;globalThis.ImageData=class{constructor(...args){this.args=args;}};
 try{drawAttentionOverview(root,data,2,25);}finally{if(previous===undefined)delete globalThis.ImageData;else globalThis.ImageData=previous;}
 assert.deepEqual(calls[0],['rect',25/48*84,2/48*84,6/48*84,6/48*84]);assert.equal(canvas.dataset.keyStart,25);assert.equal(canvas.dataset.windowStart,2);
});
test('controls remain mounted outside the redraw region and all scroll lifecycles are cleaned up',async()=>{
 const source=await readFile(new URL('../src/attention-window.mjs',import.meta.url),'utf8');
 assert.match(source,/root.querySelector\('\.aw-workbench'\).append\(controls\)/);assert.match(source,/target.innerHTML=attentionWindowMarkup/);assert.match(source,/root.addEventListener\('scroll',nativeScroll,true\)/);assert.match(source,/root.removeEventListener\('scroll',nativeScroll,true\)/);assert.match(source,/root.removeEventListener\('wheel',wheel\)/);assert.match(source,/observer.disconnect\(\);controls.remove\(\)/);assert.match(source,/for\(const id of nativeFrames.values\(\)\)win.cancelAnimationFrame\(id\)/);
 assert.match(source,/if\(first!==renderedStart\)update\(first,first\)/);assert.match(source,/else align\(start,false\)/);
});
test('global selection recenters only the changed query/key and the offline build contains the same controls',async()=>{
 const app=await readFile(new URL('../src/full-app.mjs',import.meta.url),'utf8'),bundle=await readFile(new URL('../standalone.html',import.meta.url),'utf8');
 assert.match(app,/if\(query!==previous\)state.attentionStart=/);assert.match(app,/if\(key!==previousKey\)state.attentionKeyStart=/);assert.match(app,/attentionOutputFeature=0/);assert.match(bundle,/Scroll query patches independently/);assert.match(bundle,/Scroll key patches independently/);assert.match(bundle,/Scroll head-output feature columns/);assert.match(bundle,/Attention_weights × V/);
});
