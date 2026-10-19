import {test} from 'node:test';
import assert from 'node:assert/strict';
import {runEncoder,rng,patchify,linear,HEAD_DIM,gelu} from '../src/engine.mjs';
import {projectionTrace,scalarAddress,visibleRows,expansionMarkup,patchRailMarkup,nextRailPatch,railScrollOffset,STAGE_NAMES} from '../src/expansion.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
function fixture(size=16,model='vit'){const w=size*2,h=size*2,r=rng(71),pixels=Uint8ClampedArray.from({length:w*h*4},(_,i)=>i%4===3?255:Math.floor(r()*256));const patches=patchify(pixels,w,h,size);return {patches,result:runEncoder(patches,2,model)};}
test('flattened scalar indices map to the correct pixel and RGB channel',()=>{assert.deepEqual(scalarAddress(0,16),{row:0,col:0,channel:0});assert.deepEqual(scalarAddress(413,16),{row:8,col:9,channel:2});assert.deepEqual(scalarAddress(767,16),{row:15,col:15,channel:2});assert.deepEqual(scalarAddress(3071,32),{row:31,col:31,channel:2});});
test('matrix cutaway always includes selected input, including boundaries',()=>{for(const n of [8,24,48,768,1728,3072])for(const i of [0,1,Math.floor(n/2),n-1]){const rows=visibleRows(n,i);assert.ok(rows.includes(i));assert.ok(rows.every(r=>r>=0&&r<n));assert.ok(rows.length<=9);}});
for(const size of [16,24,32])test(`${size}px patch traces exactly reconstruct every embedding dimension`,()=>{const {patches,result}=fixture(size);for(let p=0;p<patches.length;p++)for(let j=0;j<24;j++){const t=projectionTrace(patches[p].pixels,result.projectionWeights,j);assert.equal(t.inputCount,size*size*3);assert.equal(t.sum,result.embeddings[p][j]);}assert.strictEqual(result.projectionWeights,runEncoder(patches,2).projectionWeights);});
for(const model of ['vit','pixtral']){
 test(`${model}: exposed QKV matrices reproduce every head at every layer`,()=>{const {result}=fixture(16,model);for(const l of result.layers)for(let h=0;h<3;h++)for(let t=0;t<l.input.length;t++)for(const op of ['q','k','v']){const w=l.weights[op].map(row=>row.slice(h*HEAD_DIM,(h+1)*HEAD_DIM)),expected=op==='q'?l.heads[h].rawQ[t]:op==='k'?l.heads[h].rawK[t]:l.heads[h].values[t];assert.deepEqual(linear(l.normalized[t],w),expected);}});
 test(`${model}: FFN weight traces, activation and residual readouts agree`,()=>{const {result}=fixture(16,model);for(const l of result.layers)for(let t=0;t<l.input.length;t++){assert.deepEqual(linear(l.ffnInput[t],l.weights.up),l.preActivation[t]);assert.deepEqual(linear(l.expanded[t],l.weights.down),l.mlpDelta[t]);for(let d=0;d<48;d++){if(model==='vit')near(gelu(l.preActivation[t][d]),l.expanded[t][d]);else near(l.preActivation[t][d]/(1+Math.exp(-l.preActivation[t][d]))*l.gate[t][d],l.expanded[t][d]);}}});
 test(`${model}: all seven stages render and every matrix cutaway matches its engine output`,()=>{const {patches,result}=fixture(16,model);for(let stage=0;stage<STAGE_NAMES.length;stage++){for(const ffn of model==='pixtral'?['up','gate','down']:['up','down']){const s={expanded:stage,selected:1,key:2,patch:16,scalar:408,dimension:23,model,cols:2,rows:2,width:32,height:32,layer:2,head:1,qkv:'k',ffn,adapter:'second'};const html=expansionMarkup(s,patches,result,'data:image/png;base64,');assert.ok(html.includes('data-collapse'));assert.ok(!/NaN|undefined/.test(html));for(const match of html.matchAll(/data-trace-sum="([^"]+)" data-trace-output="([^"]+)"/g))assert.equal(Number(match[1]),Number(match[2]));}}});
}
test('both Pixtral projector layers reconstruct the displayed vectors',()=>{const {result}=fixture(16,'pixtral');for(let t=0;t<result.final.length;t++){assert.deepEqual(linear(result.final[t],result.adapterWeights[0]),result.adapterPre[t]);assert.deepEqual(result.adapterPre[t].map(gelu),result.adapterHidden[t]);assert.deepEqual(linear(result.adapterHidden[t],result.adapterWeights[1]),result.projected[t]);}});
test('CLS embedding expansion does not pretend CLS comes from pixels',()=>{const {patches,result}=fixture();const html=expansionMarkup({expanded:2,selected:-1,model:'vit',key:0,layer:0,head:0,patch:16,cols:2,rows:2},patches,result,'');assert.match(html,/not a projected image patch/);assert.doesNotMatch(html,/data-weight=/);});
test('invalid trace dimensions are rejected',()=>{assert.throws(()=>projectionTrace([1],[[2]],2));assert.throws(()=>projectionTrace([1,2],[[2]],0));});
for(const model of ['vit','pixtral'])test(`${model}: Patches is weight-free and Embedding owns the learned projection`,()=>{
 const {patches,result}=fixture(16,model);
 const s={expanded:1,selected:1,key:2,patch:16,scalar:413,dimension:9,model,cols:2,rows:2,width:32,height:32,layer:0,head:0};
 const guarded=new Proxy(result,{get(target,key){if(key==='projectionWeights'||key==='embeddings')throw Error('Patchification must not access projection weights or embeddings');return target[key];}});
 const patchHTML=expansionMarkup(s,patches,guarded,'');
 assert.match(patchHTML,/flatten-pixel-groups/);assert.match(patchHTML,/no learned parameters/);
 assert.doesNotMatch(patchHTML,/data-weight=|data-trace-sum=|class="matrix-product"/);
 assert.equal(s.selected,1);assert.equal(s.scalar,413);assert.equal(s.dimension,9);
 s.expanded=2;const embedHTML=expansionMarkup(s,patches,result,'');
 assert.match(embedHTML,/data-weight="413,9"/);assert.match(embedHTML,/data-trace-sum=/);
 assert.equal(s.selected,1);assert.equal(s.scalar,413);assert.equal(s.dimension,9);
});
for(const size of [16,24,32])test(`${size}px flatten view preserves RGB values and indexing at both boundaries`,()=>{
 const {patches,result}=fixture(size);
 for(const scalar of [0,1,2,size*3-1,size*3,patches[1].pixels.length-1]){
  const s={expanded:1,selected:1,key:2,patch:size,scalar,dimension:0,model:'vit',cols:2,rows:2,width:size*2,height:size*2,layer:0,head:0};
  const html=expansionMarkup(s,patches,result,'');
  const channels=[...html.matchAll(/data-scalar="(\d+)" data-raw="(\d+)" data-normalized="([^"]+)" aria-pressed="(true|false)"/g)];
  assert.equal(channels.length,12);assert.equal(channels.filter(m=>m[4]==='true').length,1);
  assert.equal(Number(channels.find(m=>m[4]==='true')[1]),scalar);
  for(const [,index,raw,normalized] of channels){assert.equal(Number(normalized),patches[1].pixels[Number(index)]);near(Number(raw)/127.5-1,Number(normalized));}
  assert.equal(Number(html.match(/data-flat-value="([^"]+)"/)[1]),patches[1].pixels[scalar]);
 }
});
for(const model of ['vit','pixtral'])test(`${model}: every expanded stage retains every patch in the same order`,()=>{
 const {patches,result}=fixture(16,model);
 for(let expanded=0;expanded<STAGE_NAMES.length;expanded++){
  const s={expanded,selected:1,key:2,patch:16,scalar:413,dimension:0,model,cols:2,rows:2,width:32,height:32,layer:0,head:0};
  const html=expansionMarkup(s,patches,result,'data:image/png;base64,example');
  const indices=[...html.matchAll(/data-rail-patch="(-?\d+)"/g)].map(m=>+m[1]);
  assert.deepEqual(indices,[...(model==='vit'&&expanded>=2?[-1]:[]),0,1,2,3]);
  assert.equal((html.match(/class="token-rail"/g)||[]).length,1);
  assert.match(html,/data-rail-patch="1"[^>]*aria-pressed="true"[^>]*tabindex="0"/);
  assert.doesNotMatch(html,/id="expand-patch"/);
  assert.ok(html.indexOf('class="token-rail"')<html.indexOf('class="expansion-inspection"'));
 }
});
test('the thumbnail rail preserves 2D patch crop positions and works for one-column images',()=>{
 const {patches}=fixture();
 const base={model:'pixtral',expanded:2,selected:3,patch:16,cols:2,rows:2};
 const html=patchRailMarkup(base,patches,'data:image/png;base64,test');
 assert.match(html,/background-size:200% 200%/);assert.match(html,/background-position:100% 100%/);
 assert.equal((html.match(/class="token-rail-thumb"/g)||[]).length,4);
 const narrow=patches.map((p,i)=>({...p,col:0,row:i}));
 const portrait=patchRailMarkup({...base,cols:1,rows:4},narrow,'');
 assert.doesNotMatch(portrait,/NaN|Infinity/);assert.match(portrait,/background-position:0% 100%/);
});
test('CLS is selectable only once it exists, and never appears for Pixtral',()=>{
 const {patches}=fixture();
 const s={expanded:2,selected:-1,patch:16,model:'vit',cols:2,rows:2};
 assert.match(patchRailMarkup(s,patches,''),/data-rail-patch="-1"[^>]*aria-pressed="true"[^>]*tabindex="0"/);
 assert.doesNotMatch(patchRailMarkup({...s,expanded:1,selected:0},patches,''),/data-rail-patch="-1"/);
 assert.doesNotMatch(patchRailMarkup({...s,model:'pixtral',selected:0},patches,''),/data-rail-patch="-1"/);
});
test('rail keyboard navigation reaches every patch without going out of bounds',()=>{
 assert.equal(nextRailPatch(0,'ArrowUp',36,false),0);
 assert.equal(nextRailPatch(0,'ArrowUp',36,true),-1);
 assert.equal(nextRailPatch(-1,'ArrowDown',36,true),0);
 assert.equal(nextRailPatch(35,'ArrowRight',36,true),35);
 assert.equal(nextRailPatch(12,'Home',36,true),-1);
 assert.equal(nextRailPatch(12,'Home',36,false),0);
 assert.equal(nextRailPatch(12,'End',36,false),35);
 assert.equal(nextRailPatch(12,'Tab',36,false),null);
});
test('rail scrolling preserves visible context and moves only enough to reveal selection',()=>{
 assert.equal(railScrollOffset(100,300,160,202),100);
 assert.equal(railScrollOffset(100,300,70,112),70);
 assert.equal(railScrollOffset(100,300,380,422),122);
 assert.equal(railScrollOffset(0,300,0,42),0);
});
