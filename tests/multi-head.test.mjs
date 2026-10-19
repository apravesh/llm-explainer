import {test} from 'node:test';
import assert from 'node:assert/strict';
import {rng,patchify,runEncoder,linear,HEADS,HEAD_DIM,DIM} from '../src/engine.mjs';
import {multiHeadView,multiHeadMarkup} from '../src/multi-head.mjs';
import {expansionMarkup} from '../src/expansion.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
function fixture(model){const r=rng(938),p=patchify(Uint8ClampedArray.from({length:32*32*4},()=>Math.floor(r()*256)),32,32,16),result=runEncoder(p,2,model);return {patches:p,result,s:{model,expanded:3,layer:0,head:0,selected:1,key:3,patch:16,rows:2,cols:2,width:32,height:32,scalar:2,dimension:0,headDetailOpen:false}};}
for(const model of ['vit','pixtral']){
 test(`${model}: concatenation preserves head order and exactly reconstructs output projection and residual`,()=>{
  const {result}=fixture(model);for(const layer of result.layers)for(let q=0;q<layer.input.length;q++){
   const v=multiHeadView(layer,q);assert.equal(v.outputs.length,HEADS);assert.equal(v.concatenated.length,DIM);
   for(let h=0;h<HEADS;h++)assert.deepEqual(v.concatenated.slice(h*HEAD_DIM,(h+1)*HEAD_DIM),layer.heads[h].out[q]);
   assert.deepEqual(v.projected,layer.delta[q]);assert.deepEqual(v.residual,layer.afterAttention[q]);
  }
 });
 test(`${model}: splitting reduces QKV features, never the number of tokens`,()=>{
  const {result}=fixture(model);for(const l of result.layers){const n=l.input.length;
   for(const head of l.heads){for(const matrix of [head.q,head.k,head.values,head.out]){assert.equal(matrix.length,n);assert.ok(matrix.every(row=>row.length===HEAD_DIM));}assert.equal(head.attention.length,n);assert.ok(head.attention.every(row=>row.length===n));}
  }
 });
 test(`${model}: each head projects all hidden features, not just an input chunk`,()=>{
  const {result}=fixture(model);for(const l of result.layers)for(let h=0;h<HEADS;h++)for(const kind of ['q','k','v']){
   const w=l.weights[kind].map(row=>row.slice(h*HEAD_DIM,(h+1)*HEAD_DIM));assert.equal(w.length,DIM);
   const actual=kind==='q'?l.heads[h].rawQ:kind==='k'?l.heads[h].rawK:l.heads[h].values;
   l.normalized.forEach((x,i)=>assert.deepEqual(linear(x,w),actual[i]));
  }
 });
 test(`${model}: default overview shows all heads and correct dimensions without dumping a full head`,()=>{
  const {s,patches,result}=fixture(model),html=expansionMarkup(s,patches,result,''),n=result.input.length;
  assert.equal((html.match(/data-mh-card=/g)||[]).length,3);assert.equal((html.match(/data-mh-matrix=/g)||[]).length,12);
  for(let h=0;h<HEADS;h++)for(const type of ['Q','K','V','A'])assert.match(html,new RegExp(`data-mh-matrix="${h}:${type}"`));
  assert.match(html,new RegExp(`${n} × 8 each`));assert.match(html,new RegExp(`${n} × ${n}`));assert.match(html,/No patches are split between heads/);
  assert.match(html,/data-mh-concat=/);assert.match(html,/id="head-detail" hidden/);assert.doesNotMatch(html,/id="flow-scores"/);
  assert.equal((html.match(/id="compare-causal"/g)||[]).length,1);
 });
 test(`${model}: expanding any head preserves the other head previews and patch selection`,()=>{
  const {s,patches,result}=fixture(model);for(let head=0;head<3;head++){
   const html=expansionMarkup({...s,head,headDetailOpen:true},patches,result,'');assert.match(html,new RegExp(`INSIDE HEAD ${head+1}`));assert.match(html,/id="flow-scores"/);assert.match(html,/id="flow-weights"/);
   assert.equal((html.match(/data-mh-card=/g)||[]).length,3);assert.equal((html.match(/id="compare-causal"/g)||[]).length,1);assert.match(html,/data-rail-patch="1"[^>]*aria-pressed="true"/);
  }
 });
 test(`${model}: W_O arithmetic uses all 24 concatenated values`,()=>{
  const {s,patches,result}=fixture(model);for(const compareCausal of [false,true]){
   const html=expansionMarkup({...s,attentionInspect:'merge',compareCausal,dimension:23},patches,result,'');const match=html.match(/data-trace-sum="([^"]+)" data-trace-output="([^"]+)"/);assert.ok(match);near(+match[1],+match[2]);assert.match(html,/Concatenate heads → output projection → residual/);assert.doesNotMatch(html,/NaN|undefined/);
  }
 });
 test(`${model}: comparison applies causal masking to all heads and leaves every encoder tensor unchanged`,()=>{
  const {result}=fixture(model),copy=structuredClone(result);for(const l of result.layers)for(let q=0;q<l.input.length;q++){
   const v=multiHeadView(l,q,true);for(const head of v.views){near(head.weights[q].reduce((a,b)=>a+b,0),1);head.weights[q].forEach((w,k)=>{if(k>q)assert.equal(w,0);});}
   assert.deepEqual(linear(v.concatenated,l.weights.o),v.projected);if(q===l.input.length-1)assert.deepEqual(v.residual,l.afterAttention[q]);
  }assert.deepEqual(result,copy);
 });
}
