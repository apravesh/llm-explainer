import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {FULL_CONFIGS,matrix,matrixValue,rowOf} from '../src/full-config.mjs';
import {attentionFull,rotaryFull} from '../src/full-engine.mjs';
import {simpleAttentionData,attentionMixPrefix} from '../src/simple-attention.mjs';
import {attentionStages,attentionStageTrace,attentionStageValue,withCausalAttention,stageNumber,animateAttentionStages} from '../src/attention-stages.mjs';
import {attentionWindowMarkup,attentionWindowData} from '../src/attention-window.mjs';
import {attentionCellTooltipData} from '../src/attention-tooltip.mjs';
function fixture(model='pixtral',n=8){
 const c=FULL_CONFIGS[model],make=seed=>matrix(n,c.width,Float32Array.from({length:n*c.width},(_,i)=>Math.sin((i+seed)*.113)));
 const rawQ=make(7),rawK=make(53),values=make(97),positions=Array.from({length:n},(_,i)=>[Math.floor(i/4),i%4]);
 const q=c.rope?rotaryFull(rawQ,c,positions):rawQ,k=c.rope?rotaryFull(rawK,c,positions):rawK,{heads}=attentionFull(q,k,values,c);
 const layer={input:make(1),rawQ,rawK,q,k,values,heads,afterAttention:make(2)},run={config:c,layers:[layer]},s={model,head:c.heads-1,layer:0,selected:2,key:5,expanded:3,attentionStart:0,attentionPanels:{}};
 return {run,s,layer,data:simpleAttentionData(run,s)};
}
for(const model of ['pixtral','vit']){
 test(`${model}: raw QK dot products cover every head feature, scaling and softmax are real stored tensors`,()=>{
  const {data}=fixture(model),stages=attentionStages(data);assert.equal(stages.divisor,8);
  assert.strictEqual(stages.scaled,data.head.scores);assert.strictEqual(stages.softmax,data.head.attention);assert.strictEqual(stages.out,data.head.out);
  for(let q=0;q<data.n;q++)for(let k=0;k<data.n;k++){
   let dot=0;for(let d=0;d<64;d++)dot+=matrixValue(data.q,q,d)*matrixValue(data.k,k,d);
   assert.equal(stages.dot.data[q*data.n+k],dot);assert.equal(Math.fround(dot/8),data.head.scores.data[q*data.n+k]);
  }
  assert.equal(stages.causal,false);assert.strictEqual(attentionStages(data),stages);
 });
 test(`${model}: comparison applies only a causal mask and agrees with independent masked attention`,()=>{
  const {data,layer,run}=fixture(model),before=structuredClone(layer),compared=withCausalAttention(data),stages=attentionStages(compared),expected=attentionFull(layer.q,layer.k,layer.values,run.config,true).heads[data.headIndex];
  assert.strictEqual(compared.q,data.q);assert.strictEqual(compared.k,data.k);assert.strictEqual(compared.v,data.v);assert.strictEqual(compared.encoderHead,data.head);
  assert.deepEqual(compared.head.scores.data,expected.scores.data);assert.deepEqual(compared.head.attention.data,expected.attention.data);assert.deepEqual(compared.head.out.data,expected.out.data);
  assert.strictEqual(stages.dot,attentionStages(data).dot);assert.deepEqual(layer,before);
  for(let q=0;q<data.n;q++){
   const sum=rowOf(compared.head.attention,q).reduce((a,b)=>a+b,0);assert.ok(Math.abs(sum-1)<1e-6);
   for(let k=0;k<data.n;k++){if(k>q){assert.equal(compared.head.scores.data[q*data.n+k],-Infinity);assert.equal(compared.head.attention.data[q*data.n+k],0);}else assert.equal(compared.head.scores.data[q*data.n+k],data.head.scores.data[q*data.n+k]);}
  }
 });
 test(`${model}: selected computation uses the entire softmax row, not the six visible tokens`,()=>{
  const {data}=fixture(model),trace=attentionStageTrace(data);let denominator=0;for(let k=0;k<data.n;k++)denominator+=Math.exp(data.head.scores.data[data.query*data.n+k]-trace.max);
  assert.equal(trace.denominator,denominator);assert.equal(trace.total,data.n);assert.ok(Math.abs(trace.numerator/trace.denominator-trace.probability)<1e-7);
  const crop=attentionWindowData(data,0);assert.ok(crop.visibleMass<trace.probability+crop.totalMass);assert.equal(attentionStageTrace(data).denominator,trace.denominator);
 });
 test(`${model}: dot/scaling/softmax tooltips show their own values and correctly describe masking`,()=>{
  const {data}=fixture(model);const q=2,k=5,stages=attentionStages(data);
  const dot=attentionCellTooltipData(data,q,k,'dot'),scaled=attentionCellTooltipData(data,q,k,'scaled'),softmax=attentionCellTooltipData(data,q,k);
  assert.equal(Number(dot.decimal),stages.dot.data[q*data.n+k]);assert.equal(Number(scaled.decimal),stages.scaled.data[q*data.n+k]);assert.equal(Number(softmax.decimal),stages.softmax.data[q*data.n+k]);assert.match(dot.context,/before scaling or masking/);assert.match(scaled.context,/no causal mask/);
  const causal=withCausalAttention(data),masked=attentionCellTooltipData(causal,q,k,'scaled'),zero=attentionCellTooltipData(causal,q,k,'softmax');
  assert.equal(masked.decimal,'−∞');assert.match(masked.context,/blocked/);assert.equal(zero.percent,'0.0000%');assert.match(zero.context,/blocked/);assert.equal(attentionCellTooltipData(causal,q,k,'dot').decimal,dot.decimal);
 });
 test(`${model}: six-token window expands to three true matrices and defaults to an unmasked encoder`,()=>{
  const {data,s}=fixture(model),closed=attentionWindowMarkup(data,s),expanded=attentionWindowMarkup(data,{...s,attentionBreakdown:true});
  assert.equal((closed.match(/data-window-pair=/g)||[]).length,36);assert.equal((expanded.match(/data-window-pair=/g)||[]).length,108);
  assert.deepEqual([...expanded.matchAll(/data-attention-panel="([^"]+)"/g)].map(m=>m[1]),['dot','scaled','softmax']);
  assert.match(expanded,/Scaling · Mask/);assert.match(expanded,/no causal mask/);assert.match(expanded,/data-attention-causal /);assert.doesNotMatch(expanded,/data-attention-causal checked/);assert.match(expanded,/data-calculation-mode="encoder"/);
  for(const [,q,k,stage,value] of expanded.matchAll(/data-window-pair="(\d+):(\d+)"[^>]*data-attention-stage="([^"]+)" data-stage-value="([^"]+)"/g))assert.equal(+value,attentionStageValue(data,+q,+k,stage).value);
 });
 test(`${model}: comparison windows visibly distinguish blocked cells from off-window context`,()=>{
  const {data,s}=fixture(model),compared=withCausalAttention(data),html=attentionWindowMarkup(compared,{...s,attentionBreakdown:true});
  assert.match(html,/data-calculation-mode="causal-comparison"/);assert.match(html,/data-attention-causal checked/);assert.match(html,/Comparison only/);assert.match(html,/encoder and later-block results are unchanged/i);
  assert.equal((html.match(/data-stage-blocked="true"/g)||[]).length,30); // 15 future cells × scaling and softmax
  const trace=attentionStageTrace(compared,0,5);assert.equal(trace.scaled,-Infinity);assert.equal(trace.probability,0);assert.equal(trace.numerator,0);assert.equal(trace.denominator,1);
  const sum=attentionMixPrefix(compared,data.n,31);assert.ok(Math.abs(sum-compared.head.out.data[compared.query*64+31])<1e-6);
 });
}
test('global state enters comparison only when explicitly selected and retains original encoder output',()=>{
 const {run,s,data}=fixture(),before=run.layers[0].output,compared=simpleAttentionData(run,{...s,attentionCausal:true});
 assert.equal(data.causalComparison,undefined);assert.equal(compared.causalComparison,true);assert.strictEqual(compared.encoderHead,run.layers[0].heads[s.head]);assert.strictEqual(run.layers[0].output,before);
 assert.equal(simpleAttentionData(run,{...s,attentionCausal:false}).head,data.head);
});
test('raw dot ranges and masked numeric labels never confuse signed scores with probabilities',()=>{
 assert.equal(stageNumber(-Infinity),'−∞');assert.equal(stageNumber(-7.123456,3),'-7.123');
 const {data}=fixture();assert.equal(attentionStageValue(data,-1,0,'dot'),null);assert.equal(attentionStageValue(data,0,999,'scaled'),null);assert.equal(attentionStageValue(data,0,0,'unknown'),null);
 const stages=attentionStages(data);assert.ok(stages.ranges.dot.min<0);assert.ok(stages.ranges.dot.max>0);assert.ok(stages.ranges.softmax.min>=0);assert.ok(stages.ranges.softmax.max<=1);
});
test('expansion animation has one owner, supports cancellation, and honors reduced motion',()=>{
 const animations=[],root={querySelectorAll:()=>Array.from({length:3},()=>({animate(frames,options){const a={frames,options,cancelled:false,cancel(){this.cancelled=true;}};animations.push(a);return a;}}))};
 const mounted=animateAttentionStages(root);assert.equal(animations.length,3);assert.equal(animations[0].frames[0].transform,'translateX(140px)');assert.equal(animations[2].frames[0].transform,'translateX(-140px)');mounted.dispose();assert.ok(animations.every(a=>a.cancelled));animateAttentionStages(root,{reduced:true});assert.equal(animations.length,3);
});
test('both builds contain interactive expansion/collapse, replay, comparison and stage-aware tooltips',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),[html,window,app]=await Promise.all(['standalone.html','src/attention-window.mjs','src/full-app.mjs'].map(read));
 for(const pattern of [/function attentionStages/,/data-window-breakdown/,/data-attention-stage/,/data-attention-causal/,/attention-calculation-animation/])assert.match(html,pattern);
 assert.match(window,/if\(!wasExpanded\)present\(true\)/);assert.match(window,/attentionCausal:false/);assert.match(window,/calculationAnimation\?\.dispose\(\)/);assert.match(app,/attentionBreakdown:false,attentionCausal:false/);
});
