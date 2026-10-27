import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {FULL_CONFIGS,matrix,rowOf} from '../src/full-config.mjs';
import {attentionFull,rotaryFull} from '../src/full-engine.mjs';
import {simpleAttentionData,simpleAttentionMarkup,simpleAttentionDetails,attentionFlowPaths,attentionMixPrefix} from '../src/simple-attention.mjs';
import {createFullViews} from '../src/full-views.mjs';
function fixture(model='pixtral'){
 const c=FULL_CONFIGS[model],n=c.cls?4:3,make=offset=>matrix(n,c.width,Float32Array.from({length:n*c.width},(_,i)=>Math.sin((i+offset)*.17)));
 const rawQ=make(1),rawK=make(23),values=make(87),positions=Array.from({length:n},(_,i)=>[0,i]);
 const q=c.rope?rotaryFull(rawQ,c,positions):rawQ,k=c.rope?rotaryFull(rawK,c,positions):rawK;
 const {heads}=attentionFull(q,k,values,c),l={input:make(8),rawQ,rawK,q,k,values,heads,afterAttention:make(16)};
 return {run:{config:c,layers:[l]},s:{expanded:3,layer:0,selected:1,key:2,head:c.heads-1,feature:0,headFeature:63,attentionPanels:{}},c,l};
}
function capture(){const seen=[];return {seen,renderer:{markup(a,o){seen.push({a,o});return `<canvas data-label="${o.label}"></canvas>`;}}};}
for(const model of ['pixtral','vit']){
 test(`${model}: simple flow uses actual Q/K after positions, V, scores, weights and output`,()=>{
  const {run,s,l,c}=fixture(model),a=simpleAttentionData(run,s),{renderer,seen}=capture();
  const html=simpleAttentionMarkup(renderer,run,s,()=>{});
  assert.equal(seen.length,0);assert.strictEqual(a.q.source,l.q);assert.strictEqual(a.k.source,l.k);assert.strictEqual(a.v.source,l.values);
  assert.equal(a.q.colStart,(c.heads-1)*64);assert.equal(a.q.cols,64);
  assert.strictEqual(a.head,l.heads[s.head]);assert.equal((html.match(/data-window-pair=/g)||[]).length,l.input.rows**2);
  assert.equal(a.query,s.selected+(c.cls?1:0));assert.equal(a.key,s.key);
  assert.match(html,/softmax\(Q · Kᵀ \/ √64\)/);assert.match(html,/weights × V/);assert.match(html,/no causal mask/);assert.match(html,/sum ALL/);
  assert.doesNotMatch(html,/class="fm-features"|fa-key-strip|type="number"|Causal comparison/);
 });
 test(`${model}: all computation inspectors default closed and causal work is lazy`,()=>{
  const {run,s}=fixture(model),{renderer}=capture();let compared=0;const causal=()=>{compared++;return run.layers[0].heads[0];};
  const closed=simpleAttentionDetails(renderer,run,s,causal);assert.equal(compared,0);
  assert.equal((closed.match(/<details /g)||[]).length,5);assert.doesNotMatch(closed,/<details[^>]*\sopen(?:\s|>)/);
  const opened=simpleAttentionDetails(renderer,run,{...s,attentionPanels:{causal:true,vectors:true}},causal);assert.equal(compared,1);assert.match(opened,/data-attention-disclosure="causal" open/);assert.match(opened,/data-attention-disclosure="vectors" open/);
 });
 test(`${model}: window encodes full query/key coordinates and true cell weights`,()=>{
  const {run,s,c}=fixture(model),{renderer}=capture(),html=simpleAttentionMarkup(renderer,run,s,()=>{}),head=run.layers[0].heads[s.head];
  const cells=[...html.matchAll(/data-window-pair="(\d+):(\d+)" data-window-weight="([^"]+)"/g)];
  assert.equal(cells.length,head.attention.rows**2);
  for(const [,q,k,value] of cells)assert.equal(+value,head.attention.data[+q*head.attention.cols+ +k]);
  for(const kind of ['q','k','v','out'])assert.equal((html.match(new RegExp(`data-window-kind="${kind}"`,'g'))||[]).length,head.attention.rows);
 });
 test(`${model}: value mixing sums all tokens for the selected head without changing tensors`,()=>{
  const {run,s}=fixture(model),before=structuredClone(run),a=simpleAttentionData(run,s);
  for(const feature of [0,31,63]){
   assert.equal(attentionMixPrefix(a,0,feature),0);let expected=0;
   for(let k=0;k<a.n;k++){expected+=a.head.attention.data[a.query*a.n+k]*a.l.values.data[k*a.c.width+a.headIndex*64+feature];assert.equal(attentionMixPrefix(a,k+1,feature),expected);}
   assert.ok(Math.abs(expected-a.head.out.data[a.query*64+feature])<1e-6);
   assert.equal(attentionMixPrefix(a,999,feature),expected);
  }
  assert.deepEqual(run,before);
 });
 test(`${model}: one stacked calculation replaces the duplicate head-preview dashboard`,()=>{
  const {run,s}=fixture(model),{renderer,seen}=capture(),views=createFullViews(renderer,{select(){},selectPair(){}}),html=views.content(run,s);
  assert.equal((html.match(/data-head-preview=/g)||[]).length,1);assert.equal((html.match(/class="sa-workbench aw-workbench"/g)||[]).length,1);
  assert.equal((html.match(/data-full-head=/g)||[]).length,run.config.heads);
  assert.equal(seen.length,0);assert.doesNotMatch(html,/fa-stack-summary|fa-projection-nodes|fa-attention-matrices|fa-key-strip/);
  assert.match(html,/is-connected-flow/);assert.match(html,/data-detail-head=/);
 });
}
test('arrows connect Q and K to scores, weights and V to the sum, then to output',()=>{
 const box=(x,y,width=100,height=100)=>({x,y,width,height});
 const paths=attentionFlowPaths({q:box(0,0,80,32),k:box(0,60,80,32),v:box(0,120,80,32),scores:box(120,20),weights:box(260,20),mix:box(400,10,100,50),output:box(400,90,100,80)});
 assert.deepEqual(paths.map(p=>p.id),['q-scores','k-scores','scores-softmax','weights-mix','v-mix','mix-output']);
 assert.equal(paths.find(p=>p.id==='k-scores').kind,'k');assert.equal(paths.find(p=>p.id==='v-mix').kind,'v');
 assert.match(paths.find(p=>p.id==='v-mix').d,/M80,136/);assert.match(paths.find(p=>p.id==='v-mix').d,/202/); // below every matrix
 assert.equal(paths.find(p=>p.id==='mix-output').d,'M450,60 L450,85');
 assert.ok(paths.every(p=>!p.d.includes('NaN')));
});
test('new flow is bundled with arrows, opt-in inspectors, and exact projection operations',async()=>{
 const html=await readFile(new URL('../standalone.html',import.meta.url),'utf8');assert.match(html,/function simpleAttentionMarkup/);assert.match(html,/function drawSimpleAttentionWires/);assert.match(html,/data-window-inspect/);assert.match(html,/data-attention-disclosure/);assert.match(html,/sum ALL/);
});
