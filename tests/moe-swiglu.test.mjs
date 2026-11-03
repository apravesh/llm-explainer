import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {createMoeModel,teachingTokens,runMoeToken,runMoeBatch,moeLinear,moeSilu,moeCost,routeToken,moeNorm} from '../src/moe-engine.mjs';
import {expertProcessMarkup,moeCostMarkup,routingWalkthrough} from '../src/moe-explanations.mjs';
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} ≠ ${b}`);
for(const k of [1,2])for(const weighting of ['selected','full'])test(`SwiGLU top-${k} ${weighting}: exact branches, expert selection and residual`,()=>{
 const model=createMoeModel(108,{activation:'swiglu'}),before=structuredClone(model),tokens=teachingTokens('The little robot reads a book');
 for(const token of tokens){const tr=runMoeToken(token.hidden,model,{k,weighting});assert.equal(tr.expertEvaluations,k);
  tr.experts.forEach((e,i)=>{if(!tr.selected.includes(i)){assert.equal(e,null);return;}assert.deepEqual(e.parallel,moeLinear(tr.normalized,model.experts[i].parallel));assert.deepEqual(e.gate,e.pre.map(moeSilu));e.activated.forEach((a,j)=>close(a,e.gate[j]*e.parallel[j]));assert.deepEqual(e.output,moeLinear(e.activated,model.experts[i].down));});
  for(let d=0;d<8;d++){close(tr.mixed[d],tr.selected.reduce((a,e)=>a+tr.gates[e]*tr.experts[e].output[d],0));close(tr.output[d],token.hidden[d]+tr.mixed[d]);}
 }assert.deepEqual(model,before);assert.equal(runMoeBatch(tokens,model,{k,weighting}).assignments,tokens.length*k);
});
test('changing expert architecture preserves router/input/common weights, but changes expert outputs',()=>{
 const a=createMoeModel(),b=createMoeModel(108,{activation:'swiglu'}),token=teachingTokens('cat')[0];assert.deepEqual(a.router,b.router);a.experts.forEach((e,i)=>{assert.deepEqual(e.up,b.experts[i].up);assert.deepEqual(e.down,b.experts[i].down);assert.equal(b.experts[i].parallel.length,8);assert.equal(b.experts[i].parallel[0].length,16);});
 const ta=runMoeToken(token.hidden,a),tb=runMoeToken(token.hidden,b);assert.deepEqual(ta.router,tb.router);assert.notDeepEqual(ta.output,tb.output);
 assert.throws(()=>createMoeModel(108,{activation:'bad'}));
});
test('SwiGLU still never evaluates unselected experts and dense mode skips the router',()=>{
 const model=createMoeModel(108,{activation:'swiglu'}),token=teachingTokens('cat')[0],route=routeToken(moeNorm(token.hidden),model.router);
 model.experts=model.experts.map((e,i)=>route.selected.includes(i)?e:new Proxy(e,{get(){throw Error('unused expert accessed');}}));runMoeToken(token.hidden,model);
 const dense=createMoeModel(108,{activation:'swiglu'});Object.defineProperty(dense,'router',{get(){throw Error('router accessed');}});const tr=runMoeToken(token.hidden,dense,{mode:'dense'});assert.deepEqual(tr.selected,[0]);assert.equal(tr.expertEvaluations,1);
});
test('stored and active counts match all actual matrices in either architecture',()=>{
 for(const activation of ['gelu','swiglu']){const m=createMoeModel(108,{activation}),size=w=>w.reduce((n,row)=>n+row.length,0),expert=size(m.experts[0].up)+size(m.experts[0].down)+(m.experts[0].parallel?size(m.experts[0].parallel):0);for(const k of [1,2,4]){const c=moeCost({activation,k});assert.equal(c.expert,expert);assert.equal(c.stored,size(m.router)+expert*4);assert.equal(c.active,size(m.router)+expert*k);}assert.equal(moeCost({activation,mode:'dense'}).stored,expert);}
 assert.equal(moeCost({activation:'swiglu'}).stored,1568);assert.equal(moeCost({activation:'swiglu',k:2}).active,800);assert.equal(moeCost({activation:'swiglu'}).active,416);
 assert.throws(()=>moeCost({activation:'bad'}));assert.throws(()=>moeCost({k:8}));
});
test('explanations distinguish the two gates, display accurate costs and no fake inactive activations',()=>{
 assert.match(expertProcessMarkup('swiglu'),/W₃/);assert.match(expertProcessMarkup('swiglu'),/⊙/);assert.doesNotMatch(expertProcessMarkup('gelu'),/W₃|SiLU/);
 const html=moeCostMarkup({activation:'swiglu',k:2,mode:'sparse'});assert.match(html,/1,568/);assert.match(html,/>800</);assert.match(html,/4d²/);assert.match(html,/3df/);assert.match(html,/not executed/);assert.match(html,/not wall-clock speed/);
 const tr=runMoeToken(teachingTokens('cat')[0].hidden,createMoeModel(),{k:2});assert.match(routingWalkthrough(tr),/top-2/);assert.match(routingWalkthrough(tr),/without running their MLPs/);assert.match(routingWalkthrough(tr),/entire same 8-feature x/);
});
test('MoE page keeps teaching scope, a single working animation, and opt-in math',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),html=await read('moe.html'),app=await read('src/moe-app.mjs');
 assert.match(html,/id="moe-activation"/);assert.match(html,/These are different gates/);assert.doesNotMatch(html,/id="moe-tour"|Not more work for every token/);
 assert.match(app,/moeCostMarkup\(state\)/);assert.match(app,/expertProcessMarkup\(state.activation\)/);assert.match(app,/numbers\('expert'/);assert.match(app,/data-numbers/);assert.match(app,/model=createMoeModel\(108,\{activation:state.activation\}\)/);
});
