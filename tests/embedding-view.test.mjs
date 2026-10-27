import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {patchify,runEncoder,rng,norm,linear} from '../src/engine.mjs';
import {outputColor} from '../src/readout.mjs';
import {embeddingRepresentation,embeddingMatrixMarkup,embeddingOverviewMarkup} from '../src/embedding-view.mjs';
import {expansionMarkup} from '../src/expansion.mjs';
function fixture(model='vit',patch=16){
 const random=rng(581),patches=patchify(Uint8ClampedArray.from({length:patch*3*patch*2*4},()=>Math.floor(random()*256)),patch*3,patch*2,patch);
 const s={model,patch,selected:2,cols:3,rows:2,width:patch*3,height:patch*2,layer:0,head:0,key:2,expanded:2,scalar:8,dimension:3};
 return {s,patches,result:runEncoder(patches,3,model)};
}
for(const model of ['vit','pixtral']){
 test(`${model}: enlarged embeddings use all of result.input, not raw projections or final output`,()=>{
  const {s,patches,result}=fixture(model),data=embeddingRepresentation(s,result),html=embeddingMatrixMarkup(s,patches,result);
  assert.strictEqual(data.rows,result.input);assert.strictEqual(data.selected,result.input[s.selected+(model==='vit'?1:0)]);
  assert.equal(data.count,patches.length+(model==='vit'?1:0));assert.equal(data.width,24);
  const cells=[...html.matchAll(/data-embedding-value="([^"]+)" style="background:([^"]+)"/g)].map(([,value,color])=>({value:+value,color}));
  assert.deepEqual(cells,result.input.flat().map(value=>({value,color:outputColor(value)})));
  assert.notDeepEqual(result.input,result.final);assert.notDeepEqual(result.input,result.embeddings);
  assert.equal((html.match(/data-embedding-patch=/g)||[]).length,data.count);
 });
 test(`${model}: selecting any token keeps the complete matrix and matching colored vector`,()=>{
  const {s,patches,result}=fixture(model);
  for(let selected=model==='vit'?-1:0;selected<patches.length;selected++){
   const state={...s,selected},html=embeddingOverviewMarkup(state,patches,result),matrix=embeddingMatrixMarkup(state,patches,result);
   assert.equal((matrix.match(/aria-pressed="true"/g)||[]).length,1);assert.equal((matrix.match(/tabindex="0"/g)||[]).length,1);
   assert.match(matrix,new RegExp(`data-embedding-patch="${selected}"[^>]*aria-pressed="true" tabindex="0"`));
   const vector=html.match(/data-embedding-vector="([^"]+)"/)[1].split(',').map(Number);
   assert.deepEqual(vector,result.input[selected+(model==='vit'?1:0)]);
   assert.equal((html.match(/data-embedding-value=/g)||[]).length,result.input.length*24);
   for(let d=0;d<24;d++)assert.ok(html.includes(`style="background:${outputColor(vector[d])}" title="Feature ${d}:`));
  }
 });
 test(`${model}: input matrix is independent of inspected block/head and decoder comparison`,()=>{
  const {s,patches,result}=fixture(model),html=embeddingMatrixMarkup(s,patches,result);
  for(let layer=0;layer<4;layer++)for(let head=0;head<3;head++)assert.equal(embeddingMatrixMarkup({...s,layer,head,compareCausal:true},patches,result),html);
 });
 test(`${model}: existing projection math stays available without displacing the matrix`,()=>{
  const {s,patches,result}=fixture(model),closed=expansionMarkup(s,patches,result,''),opened=expansionMarkup({...s,embeddingDetail:true},patches,result,'');
  assert.ok(closed.indexOf('id="expanded-embedding-matrix"')<closed.indexOf('id="embedding-arithmetic"'));
  assert.match(closed,/id="embedding-arithmetic" class="eb-arithmetic" >/);assert.match(opened,/id="embedding-arithmetic" class="eb-arithmetic" open/);
  assert.match(opened,/data-weight="8,3"/);assert.match(opened,/data-trace-sum="([^"]+)" data-trace-output="\1"/);
  assert.equal((closed.match(/data-multiply="embedding"/g)||[]).length,1);
  assert.match(closed,/data-rail-patch="2"[^>]*aria-pressed="true"/);
 });
 test(`${model}: embedding overview renders without modifying engine tensors or selection`,()=>{
  const {s,patches,result}=fixture(model),before=structuredClone(result),state={...s};embeddingOverviewMarkup(s,patches,result);embeddingMatrixMarkup(s,patches,result);assert.deepEqual(result,before);assert.deepEqual(s,state);
 });
}
test('ViT input rows include learned positions, while CLS is never presented as a patch projection',()=>{
 const {s,patches,result}=fixture();
 for(let p=0;p<patches.length;p++)assert.deepEqual(embeddingRepresentation({...s,selected:p},result).selected,result.embeddings[p].map((v,d)=>v+result.positions[p+1][d]));
 const cls=embeddingOverviewMarkup({...s,selected:-1},patches,result);assert.match(cls,/learned \[CLS\] \+ position\[0\]/);assert.match(cls,/not a projected image patch/);assert.doesNotMatch(cls,/data-multiply="embedding"/);
 assert.deepEqual(embeddingRepresentation({...s,selected:-1},result).selected,result.cls.map((v,d)=>v+result.positions[0][d]));
});
test('Pixtral matrix includes RMSNorm, not projection-only vectors or added RoPE',()=>{
 const {s,patches,result}=fixture('pixtral'),html=embeddingOverviewMarkup(s,patches,result);
 for(let p=0;p<patches.length;p++)assert.deepEqual(embeddingRepresentation({...s,selected:p},result).selected,norm(linear(patches[p].pixels,result.projectionWeights),true));
 assert.match(html,/RMSNorm\(flatten\(patch\) × Wₑ\)/);assert.match(html,/2D RoPE is applied later/);assert.doesNotMatch(html,/data-embedding-patch="-1"/);
});
test('patch-size and grid changes keep every token, including large and one-patch grids',()=>{
 for(const model of ['vit','pixtral'])for(const patch of [16,24,32]){
  const {s,patches,result}=fixture(model,patch);assert.equal((embeddingMatrixMarkup(s,patches,result).match(/data-embedding-value=/g)||[]).length,result.input.length*24);
 }
 const {s,patches,result}=fixture('pixtral'),largePatches=Array.from({length:192},(_,i)=>({row:Math.floor(i/16),col:i%16})),large={...result,input:Array.from({length:192},(_,i)=>result.input[i%6])};
 const html=embeddingMatrixMarkup({...s,selected:191,cols:16,rows:12},largePatches,large);assert.equal((html.match(/data-embedding-value=/g)||[]).length,192*24);assert.match(html,/data-embedding-patch="191"[^>]*aria-pressed="true"/);
 const one=runEncoder(patches.slice(0,1),1,'pixtral');assert.doesNotMatch(embeddingOverviewMarkup({...s,selected:0,cols:1,rows:1},patches.slice(0,1),one),/NaN|undefined|Infinity/);
});
test('full-dimension entry point uses prepared input tensors and a bounded canvas renderer',async()=>{
 const html=await readFile(new URL('../standalone.html',import.meta.url),'utf8');assert.match(html,/run.input/);assert.match(html,/function createMatrixRenderer/);assert.match(html,/pooled overview/);assert.match(html,/function featureMarkup/);
});
