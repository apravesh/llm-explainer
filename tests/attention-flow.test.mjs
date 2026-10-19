import {test} from 'node:test';
import assert from 'node:assert/strict';
import {runEncoder,patchify,rng} from '../src/engine.mjs';
import {attentionView,attentionPrefix,attentionFlowMarkup,attentionScoreDetail,attentionTokenLabel} from '../src/attention-flow.mjs';
import {expansionMarkup} from '../src/expansion.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
function fixture(model='vit'){
 const rand=rng(319),data=Uint8ClampedArray.from({length:32*32*4},(_,i)=>i%4===3?255:Math.floor(rand()*256)),patches=patchify(data,32,32,16),result=runEncoder(patches,2,model);
 const s={expanded:3,selected:0,key:3,patch:16,model,layer:0,head:0,cols:2,rows:2,width:32,height:32,scalar:7,dimension:0};
 return {s,patches,result,head:result.layers[0].heads[0]};
}
for(const model of ['vit','pixtral']){
 test(`${model}: connected encoder attention exactly matches every original head`,()=>{
  const {result}=fixture(model);
  for(const layer of result.layers)for(const head of layer.heads){const v=attentionView(head);assert.deepEqual(v.scores,head.scores);assert.strictEqual(v.weights,head.attention);assert.deepEqual(v.output,head.out);}
 });
 test(`${model}: causal comparison masks future keys and renormalizes only the allowed prefix`,()=>{
  const {head}=fixture(model),v=attentionView(head,true);
  for(let q=0;q<v.weights.length;q++){
   near(v.weights[q].reduce((a,b)=>a+b,0),1);
   const denom=head.attention[q].slice(0,q+1).reduce((a,b)=>a+b,0);
   for(let k=0;k<v.weights.length;k++){if(k>q){assert.equal(v.scores[q][k],-Infinity);assert.equal(v.weights[q][k],0);}else near(v.weights[q][k],head.attention[q][k]/denom);}
  }
  assert.deepEqual(v.output[0],head.values[0]);
  assert.deepEqual(v.output.at(-1),head.out.at(-1));
 });
 test(`${model}: comparison and prefix animation never mutate encoder tensors`,()=>{
  const {head,result}=fixture(model),snapshot=structuredClone(result);
  for(const causal of [false,true,false]){const v=attentionView(head,causal);for(let q=0;q<v.weights.length;q++)for(let count=0;count<=v.weights.length;count++)attentionPrefix(v.weights[q],head.values,count);}
  assert.deepEqual(result,snapshot);
 });
 test(`${model}: animated prefixes add one weighted value vector and end at the exact output`,()=>{
  const {head}=fixture(model);
  for(const causal of [false,true]){const v=attentionView(head,causal);for(let q=0;q<v.weights.length;q++){
   let last=attentionPrefix(v.weights[q],head.values,0);assert.ok(last.every(x=>x===0));
   for(let count=1;count<=head.values.length;count++){const next=attentionPrefix(v.weights[q],head.values,count);next.forEach((x,d)=>near(x,last[d]+v.weights[q][count-1]*head.values[count-1][d]));last=next;}
   assert.deepEqual(last,v.output[q]);
  }}
 });
 test(`${model}: expanded head keeps all projections and both matrices visible with no arithmetic dump`,()=>{
  const {s,head,patches,result}=fixture(model),html=expansionMarkup({...s,headDetailOpen:true},patches,result,'');
  for(const type of ['q','k','v','scores','softmax','mix'])assert.match(html,new RegExp(`data-flow-node="${type}"`));
  assert.match(html,/id="flow-scores"/);assert.match(html,/id="flow-weights"/);assert.match(html,/id="compare-causal"/);
  assert.match(html,/Encoder attention · no causal mask/);assert.match(html,/data-attn-play/);
  assert.doesNotMatch(html,/data-weight=/);assert.match(html,/id="attention-calculation" hidden/);
  assert.equal((html.match(/data-attn-key=/g)||[]).length,head.values.length);
 });
 test(`${model}: every deeper calculation leaves the connected view and patch rail intact`,()=>{
  const {s,patches,result}=fixture(model);
  for(const kind of ['q','k','v','scores','softmax','values',...(model==='pixtral'?['rope']:[])]){
   const state={...s,attentionInspect:kind,qkv:['q','k','v'].includes(kind)?kind:'q'};
   const html=expansionMarkup(state,patches,result,'');
   assert.match(html,/id="attention-flow"/);assert.match(html,/class="token-rail"/);assert.match(html,/class="af-calculation"/);assert.doesNotMatch(html,/undefined|NaN/);
   if(['q','k','v'].includes(kind)){const match=html.match(/data-trace-sum="([^"]+)" data-trace-output="([^"]+)"/);assert.ok(match);near(+match[1],+match[2]);}
  }
 });
}
test('the decoder comparison is explicitly labeled and includes masked-key arithmetic',()=>{
 const {s,head,patches}=fixture(),state={...s,compareCausal:true,attentionInspect:'scores'};
 const html=attentionFlowMarkup(state,patches,head);assert.match(html,/Decoder-mask comparison · not the encoder/);assert.match(html,/Counterfactual at this head only/);assert.match(html,/blocked future position/);
 assert.match(attentionScoreDetail(state,patches,head),/masked to −∞/);
 assert.match(attentionScoreDetail({...state,attentionInspect:'softmax'},patches,head),/0\.0000000/);
});
test('only Pixtral inserts 2D RoPE on the Q/K route',()=>{
 for(const model of ['vit','pixtral']){const {s,head,patches}=fixture(model),html=attentionFlowMarkup(s,patches,head);assert.equal(html.includes('data-flow-node="rope"'),model==='pixtral');if(model==='pixtral')assert.match(html,/2D RoPE changes Q and K, never V/);}
});
test('token labels distinguish CLS from image patches',()=>{
 const {patches}=fixture();assert.equal(attentionTokenLabel(0,'vit',patches),'[CLS]');assert.equal(attentionTokenLabel(1,'vit',patches),'p00 (0, 0)');assert.equal(attentionTokenLabel(0,'pixtral',patches),'p00 (0, 0)');assert.equal(attentionTokenLabel(3,'pixtral',patches),'p03 (1, 1)');
});
