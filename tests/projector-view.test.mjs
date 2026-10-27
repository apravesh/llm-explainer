import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {patchify,runEncoder,rng,linear,gelu} from '../src/engine.mjs';
import {stageNames,STAGE_NAMES,expansionMarkup} from '../src/expansion.mjs';
import {projectorTensors,projectorFlowMarkup,compactProjectorMarkup,PIXTRAL_WIDTHS} from '../src/projector-view.mjs';
import {outputColor} from '../src/readout.mjs';
function fixture(patch=16){const random=rng(91),patches=patchify(Uint8ClampedArray.from({length:patch*3*patch*2*4},()=>Math.floor(random()*256)),patch*3,patch*2,patch);return {patches,result:runEncoder(patches,3,'pixtral'),s:{model:'pixtral',expanded:6,selected:2,layer:0,head:0,patch,cols:3,rows:2,width:patch*3,height:patch*2,scalar:0,dimension:0}};}
test('real Pixtral dimensions are distinct from teaching dimensions',()=>{
 assert.deepEqual(PIXTRAL_WIDTHS,{patch:16,encoder:1024,decoder:5120,layers:24});
 const {s,patches,result}=fixture(),html=projectorFlowMarkup(s,patches,result);
 assert.match(html,/24 → 32 → 32/);assert.match(html,/1,024 → 5,120 → 5,120/);assert.match(html,/1,024 features per patch and 24 vision layers/);assert.match(html,/not slices of pretrained Pixtral weights/);
});
test('all four projector matrices are exact intermediate tensors',()=>{
 const {s,patches,result}=fixture(),before=structuredClone(result),tensors=projectorTensors(result),html=projectorFlowMarkup(s,patches,result);
 assert.deepEqual(tensors.map(t=>t.values),[result.final,result.adapterPre,result.adapterHidden,result.projected]);
 const cells=[...html.matchAll(/data-projector-cell="([^:]+):(\d+):(\d+)" data-projector-value="([^"]+)" style="background:([^"]+)"/g)];
 assert.equal(cells.length,patches.length*(24+32+32+32));
 for(const [,key,row,col,value,color] of cells){const v=tensors.find(t=>t.key===key).values[+row][+col];assert.equal(+value,v);assert.equal(color,outputColor(v));}
 assert.deepEqual(result.final.map(v=>linear(v,result.adapterWeights[0])),result.adapterPre);
 assert.deepEqual(result.adapterPre.map(v=>v.map(gelu)),result.adapterHidden);
 assert.deepEqual(result.adapterHidden.map(v=>linear(v,result.adapterWeights[1])),result.projected);assert.deepEqual(result,before);
});
test('each matrix keeps every patch selectable and one tab stop per matrix',()=>{
 const {s,patches,result}=fixture();
 for(let selected=0;selected<patches.length;selected++){
  const html=projectorFlowMarkup({...s,selected},patches,result);
  assert.equal((html.match(/data-projector-patch=/g)||[]).length,patches.length*4);
  assert.equal((html.match(/aria-pressed="true"/g)||[]).length,4);assert.equal((html.match(/tabindex="0"/g)||[]).length,4);
  for(const key of ['input','pre','hidden','output'])assert.match(html,new RegExp(`data-projector-patch="${selected}" data-projector-source="${key}"[^>]*aria-pressed="true" tabindex="0"`));
 }
});
test('both shared weight matrices open their own exact multiplication overlay',()=>{
 const {s,patches,result}=fixture(),html=projectorFlowMarkup(s,patches,result);
 assert.match(html,/data-multiply="projector-first"/);assert.match(html,/data-multiply="projector-second"/);assert.match(html,/24 × 32/);assert.match(html,/32 × 32/);assert.match(html,/Bias omitted/);
});
test('compact view retains the same input/output tensors without nested controls',()=>{
 const {s,result}=fixture(),html=compactProjectorMarkup(s,result),values=[...html.matchAll(/data-projector-value="([^"]+)"/g)].map(m=>+m[1]);
 assert.deepEqual(values,[...result.final.flat(),...result.projected.flat()]);assert.doesNotMatch(html,/<button/);assert.equal((html.match(/is-selected/g)||[]).length,2);
});
test('stage names and navigation follow the selected architecture',()=>{
 assert.equal(STAGE_NAMES[6],'Projector');assert.equal(stageNames('pixtral')[6],'Projector');assert.equal(stageNames('vit')[6],'Classification');assert.equal(STAGE_NAMES[6],'Projector');
 const {s,patches,result}=fixture();assert.match(expansionMarkup(s,patches,result,''),/ARCHITECTURE \/ PROJECTOR/);assert.match(expansionMarkup({...s,expanded:5},patches,result,''),/Projector →/);
 const vit=runEncoder(patches,3,'vit');const html=expansionMarkup({...s,model:'vit'},patches,vit,'');assert.match(html,/ARCHITECTURE \/ CLASSIFICATION/);assert.doesNotMatch(html,/class="pv-flow"|data-projector-matrix/);
});
test('projector matrices adapt to patch count and do not depend on block selection',()=>{
 for(const patch of [16,24,32]){const {s,patches,result}=fixture(patch),html=projectorFlowMarkup(s,patches,result);assert.equal(projectorFlowMarkup({...s,layer:3},patches,result),html);assert.doesNotMatch(html,/NaN|undefined/);}
});
test('hosted source and offline build include the dimensions notice and renamed stage',async()=>{
 for(const name of ['vision.html','standalone.html']){const html=await readFile(new URL('../'+name,import.meta.url),'utf8');assert.match(html,/id="next-stage-title">Projector/);assert.match(html,/id="model-scale-note"/);assert.match(html,/1,024 features per patch/);assert.doesNotMatch(html,/What next\?/);}
});
