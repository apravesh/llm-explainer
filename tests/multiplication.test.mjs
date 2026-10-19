import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {patchify,runEncoder,rng,linear} from '../src/engine.mjs';
import {multiplicationOperation,multiplicationTrace,multiplicationProgress,multiplicationRowStep,multiplicationMarkup,mlpLanesMarkup,patchConnectionPath} from '../src/multiplication.mjs';
function fixture(model='vit',patch=16){
 const random=rng(771),patches=patchify(Uint8ClampedArray.from({length:patch*patch*4*6},()=>Math.floor(random()*256)),patch*3,patch*2,patch);
 const s={model,selected:2,patch,layer:0,cols:3,rows:2};return {s,patches,result:runEncoder(patches,3,model)};
}
for(const model of ['vit','pixtral']){
 const kinds=['embedding','mlp-up','mlp-down',...(model==='pixtral'?['mlp-gate','projector-first','projector-second']:[])];
 for(const kind of kinds)test(`${model} ${kind}: every dot product exactly matches live engine output`,()=>{
  const {s,patches,result}=fixture(model),before=structuredClone(result);
  for(const layer of [0,3])for(const selected of [0,5]){
   const op=multiplicationOperation({...s,layer,selected},patches,result,kind);
   assert.deepEqual(linear(op.input,op.weights),op.output);
   for(let column=0;column<op.output.length;column++){
    const trace=multiplicationTrace(op,column);assert.equal(trace.total,op.output[column]);
    let sum=0;assert.equal(multiplicationProgress(trace,0).sum,0);
    for(let i=0;i<op.input.length;i++){
     sum+=op.input[i]*op.weights[i][column];
     const frame=multiplicationProgress(trace,i+1);
     assert.equal(frame.sum,sum);assert.equal(frame.index,i);assert.equal(frame.term,op.input[i]*op.weights[i][column]);assert.equal(frame.complete,i===op.input.length-1);
    }
   }
  }
  assert.deepEqual(result,before);
 });
 test(`${model}: all patches reuse the same matrices but have distinct input vectors`,()=>{
  const {s,patches,result}=fixture(model);
  for(const kind of kinds){const a=multiplicationOperation({...s,selected:0},patches,result,kind),b=multiplicationOperation({...s,selected:1},patches,result,kind);assert.strictEqual(a.weights,b.weights);assert.notStrictEqual(a.input,b.input);}
 });
 test(`${model}: all token lanes remain selectable in compact and expanded diagrams`,()=>{
  const {s,patches,result}=fixture(model),count=patches.length+(model==='vit'?1:0);
  for(const compact of [true,false]){
   const html=mlpLanesMarkup(s,patches,result,{compact});
   assert.equal((html.match(/data-mlp-lane=/g)||[]).length,count);
   assert.equal((html.match(/class="mlp-lane-wire"/g)||[]).length,count*2);
   assert.equal((html.match(/data-multiply-patch=/g)||[]).length,count*2);
   assert.equal((html.match(/data-mlp-patch="2" aria-pressed="true" tabindex="0"/g)||[]).length,1);
   assert.match(html,/Same W₁ \/ W₂/);assert.doesNotMatch(html,/NaN|undefined/);
  }
 });
 test(`${model}: overlay lists all tokens and columns without requiring another zoom`,()=>{
  const {s,patches,result}=fixture(model),op=multiplicationOperation(s,patches,result,'mlp-up'),html=multiplicationMarkup(op,s,patches,'sample.svg');
  assert.equal((html.match(/data-mm-patch=/g)||[]).length,patches.length+(model==='vit'?1:0));assert.equal((html.match(/data-mm-output=/g)||[]).length,48);
  assert.match(html,/data-mm-close/);assert.match(html,/data-mm-step/);assert.match(html,/data-mm-progress/);assert.match(html,/data-mm-sweep/);assert.match(html,/sums use full precision/);assert.doesNotMatch(html,/data-expand-step|data-step=|NaN|undefined/);
 });
}
test('MLP input, pre-activation, gate and residual are not confused with each other',()=>{
 const {s,patches,result}=fixture('pixtral'),l=result.layers[0];
 const up=multiplicationOperation(s,patches,result,'mlp-up'),gate=multiplicationOperation(s,patches,result,'mlp-gate'),down=multiplicationOperation(s,patches,result,'mlp-down');
 assert.strictEqual(up.input,l.ffnInput[2]);assert.strictEqual(up.output,l.preActivation[2]);assert.strictEqual(gate.output,l.gate[2]);assert.strictEqual(down.input,l.expanded[2]);assert.strictEqual(down.output,l.mlpDelta[2]);
 assert.notDeepEqual(down.output,l.output[2]);assert.match(up.after,/SiLU\(z\) ⊙ g/);assert.match(down.after,/add Δh/);
});
test('ViT CLS can use the MLP but never a pixel projector',()=>{
 const {s,patches,result}=fixture(),state={...s,selected:-1};
 const op=multiplicationOperation(state,patches,result,'mlp-up');assert.strictEqual(op.input,result.layers[0].ffnInput[0]);assert.equal(op.patch,'[CLS]');
 assert.throws(()=>multiplicationOperation(state,patches,result,'embedding'),/learned vector/);assert.throws(()=>multiplicationOperation(state,patches,result,'mlp-gate'),/Unavailable/);assert.throws(()=>multiplicationOperation(state,patches,result,'projector-first'),/Pixtral/);
});
test('large RGB products retain all scalar terms and all columns',()=>{
 const {s,patches,result}=fixture('pixtral',32),op=multiplicationOperation(s,patches,result,'embedding'),trace=multiplicationTrace(op,23);
 assert.equal(trace.terms.length,3072);assert.equal(trace.total,op.output[23]);assert.equal(multiplicationProgress(trace,9999).count,3072);assert.equal(multiplicationProgress(trace,-1).count,0);assert.throws(()=>multiplicationTrace(op,24),/Invalid/);
 const html=multiplicationMarkup(op,s,patches,'');assert.match(html,/3072 × 24/);assert.equal((html.match(/data-mm-patch=/g)||[]).length,patches.length);
});
test('token keyboard navigation clamps at both boundaries, including CLS',()=>{
 assert.equal(multiplicationRowStep(0,'ArrowUp',6,true),-1);assert.equal(multiplicationRowStep(-1,'ArrowUp',6,true),-1);assert.equal(multiplicationRowStep(5,'ArrowDown',6,false),5);assert.equal(multiplicationRowStep(4,'Home',6,true),-1);assert.equal(multiplicationRowStep(4,'Home',6,false),0);assert.equal(multiplicationRowStep(0,'End',6,false),5);assert.equal(multiplicationRowStep(0,'Enter',6,false),null);
});
test('patch wires connect source and target centers without crossing into another row',()=>{
 assert.equal(patchConnectionPath(10,20,60,90),'M10,20 C34,20 36,90 60,90');
});
test('offline bundle includes overlay and connected lanes without external modules',async()=>{
 const html=await readFile(new URL('../standalone.html',import.meta.url),'utf8');assert.match(html,/function createMultiplicationOverlay/);assert.match(html,/function mlpLanesMarkup/);assert.match(html,/mm-dialog/);assert.match(html,/patch-embedding-wires/);
});

for(const model of ['vit','pixtral'])test(`${model}: full output matrix includes every exact product with correct token mapping`,()=>{
 const {s,patches,result}=fixture(model),kinds=['embedding','mlp-up','mlp-down',...(model==='pixtral'?['mlp-gate','projector-first','projector-second']:[])];
 for(const kind of kinds){
  const op=multiplicationOperation(s,patches,result,kind),html=multiplicationMarkup(op,s,patches,'');
  assert.strictEqual(op.inputs[op.row],op.input);assert.strictEqual(op.outputs[op.row],op.output);
  assert.equal(op.outputs.length,patches.length+(kind!=='embedding'&&model==='vit'?1:0));
  assert.deepEqual(op.inputs.map(x=>linear(x,op.weights)),op.outputs);
  const cells=[...html.matchAll(/data-mm-result-value="([^"]+)"/g)].map(m=>+m[1]);assert.deepEqual(cells,op.outputs.flat());
  assert.equal((html.match(/data-mm-result-row=/g)||[]).length,op.outputs.length);
  assert.match(html,new RegExp(`data-mm-result-row="${op.row}"[^>]*aria-pressed="true" tabindex="0"`));
  assert.match(html,new RegExp(`${op.outputs.length} × ${op.output.length}`));
 }
});
