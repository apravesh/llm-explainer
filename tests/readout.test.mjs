import {test} from 'node:test';
import assert from 'node:assert/strict';
import {runEncoder,patchify,rng,linear} from '../src/engine.mjs';
import {STAGE_NAMES,expansionMarkup} from '../src/expansion.mjs';
import {finalRepresentation,outputMatrixMarkup,outputColor,decoderPrefix,compactNextMarkup,nextStageMarkup,outputReadoutMarkup} from '../src/readout.mjs';
function fixture(model='vit'){
 const r=rng(156),patches=patchify(Uint8ClampedArray.from({length:32*48*4},()=>Math.floor(r()*256)),48,32,16),result=runEncoder(patches,3,model);
 return {patches,result,s:{model,expanded:6,selected:1,key:2,patch:16,cols:3,rows:2,width:48,height:32,layer:0,head:0,scalar:7,dimension:0}};
}
test('the new handoff stage follows encoder Output, not a transformer block',()=>{
 assert.equal(STAGE_NAMES.length,7);assert.equal(STAGE_NAMES[5],'Output');assert.equal(STAGE_NAMES[6],'Projector');
});
for(const model of ['vit','pixtral']){
 test(`${model}: final output count and token index distinguish CLS from patch states`,()=>{
  const {s,result,patches}=fixture(model),data=finalRepresentation(s,result);
  assert.equal(data.count,patches.length+(model==='vit'?1:0));assert.equal(data.width,24);assert.equal(data.patchCount,patches.length);assert.equal(data.hasCls,model==='vit');assert.strictEqual(data.selected,result.final[s.selected+(model==='vit'?1:0)]);
 });
 test(`${model}: Output shows only encoder states, not decoder-ready vectors or predictions`,()=>{
  const {s,result,patches}=fixture(model),html=outputReadoutMarkup({...s,expanded:5},patches,result);
  const selected=html.match(/data-final-vector="([^"]+)"/)[1].split(',').map(Number);
  assert.deepEqual(selected,finalRepresentation(s,result).selected);assert.match(html,/not class labels or text/);assert.match(html,/data-expand-step="6"/);assert.doesNotMatch(html,/data-projector-vector|data-trace-sum/);
 });
 test(`${model}: show only the selected readout path and no second model picker`,()=>{
  const {s,result,patches}=fixture(model),html=nextStageMarkup(s,patches,result);
  assert.match(html,new RegExp(`data-readout-model="${model}"`));
  assert.equal((html.match(/class="ro-selected-path"/g)||[]).length,1);
  assert.equal((html.match(/id="readout-path-content"/g)||[]).length,1);
  assert.doesNotMatch(html,/ro-two-paths|data-next-model|ro-switch-path|Explore both paths/);
  if(model==='pixtral'){
   assert.match(html,/all patches → projector → text decoder/);assert.match(html,/class="ro-decoder"/);
   assert.doesNotMatch(html,/CLASSIFICATION ViT|class="ro-cls"|class="ro-classifier"/);
  }else{
   assert.match(html,/final \[CLS\] → classifier → class scores/);assert.match(html,/class="ro-cls"/);
   assert.doesNotMatch(html,/PIXTRAL|class="ro-decoder"|projector → text decoder/);
  }
 });
 test(`${model}: browsing the handoff does not mutate encoder tensors or selection`,()=>{
  const {s,result,patches}=fixture(model),snapshot=structuredClone(result),selected=s.selected;
  for(let step=0;step<3;step++)nextStageMarkup({...s,decoderStep:step,clsStep:step},patches,result);
  assert.deepEqual(result,snapshot);assert.equal(s.selected,selected);
 });
 test(`${model}: seventh-stage navigation works and stops at the last stage`,()=>{
  const {s,result,patches}=fixture(model),html=expansionMarkup(s,patches,result,'');
  assert.match(html,/data-expand-step="5" >← Output/);assert.match(html,/data-expand-step="7" disabled/);assert.match(html,/class="token-rail"/);assert.doesNotMatch(html,/NaN|undefined/);
 });
}
test('CLS walkthrough starts from a learned vector and ends at the exact normalized readout',()=>{
 const {s,result,patches}=fixture(),initial=nextStageMarkup({...s,clsStep:0},patches,result),last=nextStageMarkup({...s,clsStep:result.layers.length+2},patches,result);
 assert.deepEqual(initial.match(/data-cls-vector="([^"]+)"/)[1].split(',').map(Number),result.cls);
 assert.deepEqual(last.match(/data-cls-vector="([^"]+)"/)[1].split(',').map(Number),result.final[0]);
 assert.match(initial,/not an image crop/);assert.match(initial,/classifier not run/);assert.doesNotMatch(initial,/ro-joint-stream/);
 for(let i=0;i<result.layers.length;i++){const html=nextStageMarkup({...s,clsStep:i+2},patches,result);assert.deepEqual(html.match(/data-cls-vector="([^"]+)"/)[1].split(',').map(Number),result.layers[i].output[0]);}
});
test('decoder prefix includes every patch in row-major order with correct row boundaries',()=>{
 for(const [rows,cols] of [[1,1],[1,4],[3,1],[6,8]]){
  const prefix=decoderPrefix(rows,cols),images=prefix.filter(t=>t.kind==='image');
  assert.deepEqual(images.map(t=>t.patch),Array.from({length:rows*cols},(_,i)=>i));
  assert.equal(prefix.filter(t=>t.kind==='break').length,rows-1);assert.equal(prefix.filter(t=>t.kind==='end').length,1);
  assert.equal(prefix.length,rows*cols+rows+2);assert.equal(prefix.at(-1).kind,'text');
  for(let r=0;r<rows;r++)assert.equal(prefix[1+(r+1)*(cols+1)-1].kind,r===rows-1?'end':'break');
 }
});
test('autoregressive steps append symbolic text slots, never regenerate image vectors',()=>{
 const initial=decoderPrefix(2,3);
 for(let generated=1;generated<=2;generated++){
  const next=decoderPrefix(2,3,generated);assert.deepEqual(next.slice(0,initial.length),initial);assert.equal(next.filter(t=>t.kind==='generated').length,generated);assert.equal(next.at(-1).kind,'generated');
 }
 assert.throws(()=>decoderPrefix(0,2));assert.throws(()=>decoderPrefix(2,2,3));
});
test('Pixtral handoff shows actual projected vectors and labels decoder as symbolic, not executed',()=>{
 const {s,result,patches}=fixture('pixtral'),html=nextStageMarkup({...s,decoderStep:1},patches,result);
 assert.deepEqual(html.match(/data-projector-vector="([^"]+)"/)[1].split(',').map(Number),result.projected[s.selected]);
 assert.equal((html.match(/data-readout-patch=/g)||[]).length,patches.length);
 assert.match(html,/SCHEMATIC · NOT RUN/);assert.match(html,/not model predictions/);assert.match(html,/decoder’s own input sequence/);assert.match(html,/No \[CLS\] bottleneck/);
 assert.match(html,/t₂/);assert.equal((html.match(/is-query/g)||[]).length,1);
});
test('projector arithmetic remains inspectable at the new handoff stage',()=>{
 const {s,result,patches}=fixture('pixtral');
 for(const adapter of ['first','second']){
  const html=expansionMarkup({...s,projectorDetail:true,adapter,dimension:31},patches,result,'');const m=html.match(/data-trace-sum="([^"]+)" data-trace-output="([^"]+)"/);
  assert.ok(m);assert.equal(+m[1],+m[2]);assert.match(html,/id="projector-detail"/);assert.match(html,/ro-joint-stream/);
 }
});
test('compact preview distinguishes classification from text decoding',()=>{
 assert.match(compactNextMarkup('vit'),/classifier not run/);assert.match(compactNextMarkup('pixtral'),/Causal text decoder/);assert.match(compactNextMarkup('pixtral'),/decoder schematic only/);
});

for(const model of ['vit','pixtral']){
 test(`${model}: both output zoom states render every final activation with identical colors`,()=>{
  const {s,result,patches}=fixture(model),before=structuredClone(result);
  const cellData=html=>[...html.matchAll(/data-output-value="([^"]+)" style="background:([^"]+)"/g)].map(([,value,color])=>({value:+value,color}));
  const expanded=outputMatrixMarkup(s,patches,result),compact=outputMatrixMarkup(s,patches,result,{compact:true});
  const expected=result.final.flat().map(value=>({value,color:outputColor(value)}));
  assert.deepEqual(cellData(expanded),expected);assert.deepEqual(cellData(compact),expected);
  assert.equal((expanded.match(/data-output-patch=/g)||[]).length,result.final.length);
  assert.equal((compact.match(/class="ro-compact-row /g)||[]).length,result.final.length);
  assert.doesNotMatch(compact,/<button|id="output-matrix"/);assert.deepEqual(result,before);
 });
 test(`${model}: selecting any output row preserves the complete matrix and one keyboard tab stop`,()=>{
  const {s,result,patches}=fixture(model);
  for(let selected=model==='vit'?-1:0;selected<patches.length;selected++){
   const state={...s,expanded:5,selected},html=outputReadoutMarkup(state,patches,result),matrix=outputMatrixMarkup(state,patches,result);
   const row=matrix.match(/<button[^>]*data-output-patch="(-?\d+)"[^>]*aria-pressed="true" tabindex="0"/);
   assert.ok(row);assert.equal(+row[1],selected);
   assert.equal((matrix.match(/aria-pressed="true"/g)||[]).length,1);assert.equal((matrix.match(/tabindex="0"/g)||[]).length,1);
   const index=selected+(model==='vit'?1:0),vector=html.match(/data-final-vector="([^"]+)"/)[1].split(',').map(Number);
   assert.deepEqual(vector,result.final[index]);assert.equal((html.match(/data-output-value=/g)||[]).length,result.final.length*24);
   for(const v of vector)assert.ok(html.includes(`style="background:${outputColor(v)}" title="Dimension`));
  }
 });
 test(`${model}: output matrix always shows the final representation regardless of inspected layer`,()=>{
  const {s,result,patches}=fixture(model),expected=outputMatrixMarkup(s,patches,result);
  for(let layer=0;layer<result.layers.length;layer++)assert.equal(outputMatrixMarkup({...s,layer,compareCausal:true},patches,result),expected);
 });
}
test('output palette preserves sign, zero and explicit saturation at ±2',()=>{
 assert.equal(outputColor(0),'rgb(245,245,245)');assert.equal(outputColor(2),'rgb(107,143,207)');assert.equal(outputColor(-2),'rgb(204,143,115)');
 assert.equal(outputColor(20),outputColor(2));assert.equal(outputColor(-20),outputColor(-2));assert.notEqual(outputColor(.5),outputColor(-.5));
});
test('output matrix does not truncate large patch grids',()=>{
 const {s,result}=fixture('pixtral'),patches=Array.from({length:192},(_,i)=>({row:Math.floor(i/16),col:i%16}));
 const large={...result,final:Array.from({length:192},(_,i)=>result.final[i%result.final.length])};
 const html=outputMatrixMarkup({...s,rows:12,cols:16,selected:191},patches,large);
 assert.equal((html.match(/data-output-value=/g)||[]).length,192*24);assert.match(html,/data-output-patch="191"[^>]*aria-pressed="true" tabindex="0"/);
});

for(const model of ['vit','pixtral'])test(`${model}: compact preview and output destination do not advertise another path`,()=>{
 const {s,result,patches}=fixture(model),compact=compactNextMarkup(model),output=outputReadoutMarkup({...s,expanded:5},patches,result);
 assert.doesNotMatch(compact,/both paths|Need text instead/);
 if(model==='vit'){
  assert.doesNotMatch(compact,/decoder|LLM/);assert.match(output,/Follow \[CLS\] into the classification head/);assert.doesNotMatch(output,/into a text decoder/);
 }else{
  assert.doesNotMatch(compact,/classifier|Class scores|\[CLS\]/);assert.match(output,/Project these vectors into a text decoder/);assert.doesNotMatch(output,/classification head/);
 }
});
