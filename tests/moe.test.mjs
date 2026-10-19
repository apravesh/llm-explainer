import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {MOE_DIM,MOE_HIDDEN,MOE_EXPERTS,moeLinear,moeSoftmax,moeNorm,createMoeModel,teachingTokens,routeToken,expertForward,runMoeToken,runMoeBatch,moeCost,moeMixPrefix} from '../src/moe-engine.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const fixture=()=>({model:createMoeModel(),tokens:teachingTokens('The little robot reads a book')});
test('MoE model has independent deterministic expert parameters and declared shapes',()=>{
 const m=createMoeModel();assert.deepEqual(m,createMoeModel());assert.notDeepEqual(m,createMoeModel(2));assert.equal(m.router.length,8);assert.equal(m.router[0].length,4);assert.equal(m.experts.length,4);
 for(const e of m.experts){assert.equal(e.up.length,8);assert.equal(e.up[0].length,16);assert.equal(e.down.length,16);assert.equal(e.down[0].length,8);}assert.notDeepEqual(m.experts[0],m.experts[1]);
});
test('teaching tokens are whitespace labels, bounded to 12, with deterministic synthetic vectors',()=>{
 assert.deepEqual(teachingTokens('hello   world'),teachingTokens('hello world'));assert.equal(teachingTokens('a '.repeat(20)).length,12);assert.equal(teachingTokens('  ').length,0);
 const t=teachingTokens('same same 🐦');assert.equal(t[2].label,'🐦');assert.notDeepEqual(t[0].hidden,t[1].hidden);assert.ok(t.every(x=>x.hidden.length===8&&x.hidden.every(Number.isFinite)));
});
test('stable softmax and normalization remain finite',()=>{
 const p=moeSoftmax([10000,10001,9999,9998]);near(p.reduce((a,b)=>a+b),1);assert.ok(p[1]>p[0]);assert.deepEqual(moeNorm(Array(8).fill(1)),Array(8).fill(0));near(moeNorm([1,2,3,4,5,6,7,8]).reduce((a,b)=>a+b),0);
});
for(const k of [1,2])for(const weighting of ['selected','full'])test(`top-${k} ${weighting}: route, expert execution, mixture, and residual are exact`,()=>{
 const {model,tokens}=fixture();for(const token of tokens){const tr=runMoeToken(token.hidden,model,{k,weighting});assert.equal(tr.selected.length,k);assert.equal(tr.expertEvaluations,k);assert.equal(tr.experts.filter(Boolean).length,k);near(tr.router.probabilities.reduce((a,b)=>a+b),1);
 assert.deepEqual(tr.router.logits,moeLinear(tr.normalized,model.router));assert.deepEqual(tr.selected,[0,1,2,3].sort((a,b)=>tr.router.logits[b]-tr.router.logits[a]||a-b).slice(0,k));
 for(let e=0;e<4;e++){if(!tr.selected.includes(e)){assert.equal(tr.gates[e],0);assert.equal(tr.experts[e],null);assert.equal(tr.contributions[e],null);}else{assert.deepEqual(tr.experts[e],expertForward(tr.normalized,model.experts[e]));tr.experts[e].output.forEach((v,d)=>near(tr.contributions[e][d],v*tr.gates[e]));}}
 if(weighting==='selected')near(tr.gates.reduce((a,b)=>a+b),1);else{tr.selected.forEach(e=>assert.equal(tr.gates[e],tr.router.probabilities[e]));assert.ok(tr.gates.reduce((a,b)=>a+b)<1);}
 for(let d=0;d<8;d++){near(tr.mixed[d],tr.selected.reduce((sum,e)=>sum+tr.gates[e]*tr.experts[e].output[d],0));near(tr.output[d],token.hidden[d]+tr.mixed[d]);}assert.deepEqual(moeMixPrefix(tr,k),tr.mixed);
 }
});
test('top-1 selected normalization is exactly one, not its original probability',()=>{
 const {tokens,model}=fixture(),tr=runMoeToken(tokens[0].hidden,model);assert.equal(tr.gates[tr.selected[0]],1);assert.ok(tr.router.probabilities[tr.selected[0]]<1);
});
test('unselected expert matrices are not evaluated',()=>{
 const {tokens,model}=fixture(),route=routeToken(moeNorm(tokens[0].hidden),model.router,{k:1});
 model.experts=model.experts.map((e,i)=>route.selected.includes(i)?e:new Proxy(e,{get(){throw Error('Unselected expert was executed');}}));
 const tr=runMoeToken(tokens[0].hidden,model);assert.equal(tr.expertEvaluations,1);
});
test('dense baseline skips router and all experts except expert 1',()=>{
 const {tokens,model}=fixture();Object.defineProperty(model,'router',{get(){throw Error('Dense baseline must not route');}});model.experts=model.experts.map((e,i)=>i===0?e:new Proxy(e,{get(){throw Error('Unused expert evaluated');}}));
 const tr=runMoeToken(tokens[0].hidden,model,{mode:'dense'});assert.equal(tr.router,null);assert.deepEqual(tr.selected,[0]);assert.deepEqual(tr.gates,[1,0,0,0]);assert.deepEqual(tr.mixed,expertForward(tr.normalized,model.experts[0]).output);
});
test('expert selection is deterministic in tied scores',()=>{
 const model=createMoeModel(),h=Array(8).fill(0);assert.deepEqual(routeToken(h,model.router,{k:2}).selected,[0,1]);
});
test('batch assignment counts equal tokens × k without token dropping',()=>{
 const {tokens,model}=fixture();for(const k of [1,2]){const b=runMoeBatch(tokens,model,{k});assert.equal(b.assignments,tokens.length*k);assert.equal(b.traces.length,tokens.length);assert.equal(b.loads.reduce((a,b)=>a+b),b.assignments);}
 const dense=runMoeBatch(tokens,model,{mode:'dense'});assert.deepEqual(dense.loads,[tokens.length,0,0,0]);
});
test('editing a hidden vector can change routing and output without changing model weights',()=>{
 const {tokens,model}=fixture(),before=structuredClone(model),tr1=runMoeToken(tokens[0].hidden,model),tr2=runMoeToken(tokens[0].hidden.map(v=>-v),model);assert.notDeepEqual(tr1.selected,tr2.selected);assert.notDeepEqual(tr1.output,tr2.output);assert.deepEqual(model,before);
});
test('prefix mixture animation shows exact partial sums and reaches the final mixture',()=>{
 const {tokens,model}=fixture(),tr=runMoeToken(tokens[0].hidden,model,{k:2});assert.deepEqual(moeMixPrefix(tr,0),Array(8).fill(0));assert.deepEqual(moeMixPrefix(tr,1),tr.contributions[tr.selected[0]]);assert.deepEqual(moeMixPrefix(tr,2),tr.mixed);
});
test('stored and active linear weights/MAC counts are correct and avoid a false speedup claim',()=>{
 const top1=moeCost(),top2=moeCost({k:2}),dense=moeCost({mode:'dense'});assert.equal(top1.stored,1056);assert.equal(top1.active,288);assert.equal(top2.active,544);assert.equal(top2.stored,1056);assert.equal(dense.active,256);assert.equal(dense.stored,256);assert.ok(top1.active>dense.active);assert.equal(moeCost({k:4}).active,1056);
});
test('invalid routing configuration and hidden states are rejected',()=>{
 const {model,tokens}=fixture();assert.throws(()=>runMoeToken([1],model));assert.throws(()=>runMoeToken(Array(8).fill(NaN),model));assert.throws(()=>runMoeToken(tokens[0].hidden,model,{k:0}));assert.throws(()=>runMoeToken(tokens[0].hidden,model,{weighting:'bogus'}));assert.throws(()=>runMoeToken(tokens[0].hidden,model,{mode:'bogus'}));
});
test('MoE UI discloses scope, synthetic inputs, router conventions, and inactive experts',async()=>{
 const html=await readFile(new URL('../moe.html',import.meta.url),'utf8');for(const id of ['moe-explorer','moe-k','moe-token-list','router-bars','expert-list','moe-inspector','dispatch-table','cost-comparison'])assert.match(html,new RegExp(`id="${id}"`));assert.match(html,/untrained/i);assert.match(html,/not a reproduction of Switch or Mixtral/);assert.match(html,/no capacity limit, token dropping/);assert.doesNotMatch(html,/Currently under construction/);
});
