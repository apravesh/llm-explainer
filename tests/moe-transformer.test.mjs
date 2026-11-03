import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {createMoeTransformer,runMoeTransformer,moeCausalAttention,moePosition,adjacentMoeBlock} from '../src/moe-transformer.mjs';
import {teachingTokens,moeNorm,moeLinear,runMoeToken} from '../src/moe-engine.mjs';
import {moeAttentionSummary,moeProjectionSummary,moeStackReadout} from '../src/moe-block-view.mjs';
import {moeLoopFrame} from '../src/moe-animation.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-11,`${a} != ${b}`),fixture=(activation='gelu',options={})=>{const model=createMoeTransformer({activation}),tokens=teachingTokens('one two three four');return {model,tokens,run:runMoeTransformer(tokens,model,{k:2,...options})};};
test('four distinct blocks own separate attention/router/experts, with one separate LM head',()=>{
 const m=createMoeTransformer();assert.equal(m.depth,4);assert.equal(m.heads,2);assert.deepEqual(m,createMoeTransformer());
 for(let i=1;i<4;i++){assert.notDeepEqual(m.blocks[0].attention,m.blocks[i].attention);assert.notDeepEqual(m.blocks[0].ffn.router,m.blocks[i].ffn.router);assert.notDeepEqual(m.blocks[0].ffn.experts,m.blocks[i].ffn.experts);}
 assert.notDeepEqual(m.lmHead,m.blocks[0].attention.o);assert.equal(m.lmHead.length,8);assert.equal(m.lmHead[0].length,8);
 const gated=createMoeTransformer({activation:'swiglu'});m.blocks.forEach((b,i)=>{assert.deepEqual(b.attention,gated.blocks[i].attention);assert.deepEqual(b.ffn.router,gated.blocks[i].ffn.router);});assert.deepEqual(m.lmHead,gated.lmHead);
});
for(const activation of ['gelu','swiglu'])for(const mode of ['sparse','dense'])for(const k of [1,2])test(`${activation} ${mode} top-${k}: full stack handoffs and final projection are actual computations`,()=>{
 const {tokens,model,run}=fixture(activation,{mode,k}),before=structuredClone({tokens,model});
 tokens.forEach((t,i)=>assert.deepEqual(run.input[i],t.hidden.map((v,j)=>v+moePosition(i)[j])));
 run.blocks.forEach((b,i)=>{assert.strictEqual(b.input,i?run.blocks[i-1].output:run.input);assert.strictEqual(b.attention.input,b.input);assert.equal(b.attention.heads.length,2);
  b.ffn.traces.forEach((tr,row)=>{assert.deepEqual(tr.hidden,b.attention.output[row]);assert.deepEqual(tr,runMoeToken(b.attention.output[row],model.blocks[i].ffn,{mode,k}));assert.strictEqual(b.output[row],tr.output);assert.equal(tr.expertEvaluations,mode==='dense'?1:k);});
 });
 run.final.forEach((row,i)=>{assert.deepEqual(row,moeNorm(run.blocks.at(-1).output[i]));assert.deepEqual(run.logits[i],moeLinear(row,model.lmHead));near(run.probabilities[i].reduce((a,b)=>a+b,0),1);assert.ok(row.every(Number.isFinite));});
 assert.deepEqual({tokens,model},before);
});
test('causal attention scores, head concatenation, WO and residual match independent scalar arithmetic',()=>{
 const model=createMoeTransformer(),input=teachingTokens('a b c').map(t=>t.hidden),a=moeCausalAttention(input,model.blocks[0].attention);
 a.heads.forEach((head,h)=>head.probabilities.forEach((p,i)=>{near(p.reduce((x,y)=>x+y,0),1);for(let j=0;j<p.length;j++){if(j>i){assert.equal(p[j],0);assert.equal(head.scores[i][j],-Infinity);}else{let sum=0;for(let d=0;d<4;d++)sum+=a.q[i][h*4+d]*a.k[j][h*4+d];near(head.scores[i][j],sum/2);}}
  for(let d=0;d<4;d++){let expected=0;for(let j=0;j<input.length;j++)expected+=p[j]*a.v[j][h*4+d];near(head.output[i][d],expected);}
 }));
 input.forEach((row,i)=>{assert.deepEqual(a.combined[i],a.heads.flatMap(h=>h.output[i]));assert.deepEqual(a.delta[i],moeLinear(a.combined[i],model.blocks[0].attention.o));a.output[i].forEach((v,j)=>near(v,row[j]+a.delta[i][j]));});
});
test('future tokens cannot change preceding states, routes or logits; past tokens can affect later outputs',()=>{
 const {tokens,model,run}=fixture('swiglu'),edited=structuredClone(tokens);edited[3].hidden=edited[3].hidden.map(x=>-x+2);const next=runMoeTransformer(edited,model,{k:2});
 run.blocks.forEach((b,l)=>{for(let i=0;i<3;i++){assert.deepEqual(b.output[i],next.blocks[l].output[i]);assert.deepEqual(b.ffn.traces[i].selected,next.blocks[l].ffn.traces[i].selected);}});
 for(let i=0;i<3;i++)assert.deepEqual(run.logits[i],next.logits[i]);assert.notDeepEqual(run.logits[3],next.logits[3]);
 edited[0].hidden=edited[0].hidden.map((x,j)=>x+(j%2?1:-1));const past=runMoeTransformer(edited,model,{k:2});assert.notDeepEqual(next.logits[3],past.logits[3]);
});
test('the same expert number in different blocks represents different outputs/routes',()=>{
 const {run}=fixture();assert.notDeepEqual(run.blocks[0].attention.output,run.blocks[1].attention.output);
 assert.notDeepEqual(run.blocks[0].ffn.traces.map(t=>t.router.logits),run.blocks[1].ffn.traces.map(t=>t.router.logits));
});
test('single-token and maximum-length sequences compute without invalid values',()=>{
 const model=createMoeTransformer();for(const text of ['one','a b c d e f g h i j k l']){const run=runMoeTransformer(teachingTokens(text),model,{k:2});assert.ok(run.logits.flat().every(Number.isFinite));assert.equal(run.blocks[0].attention.heads[0].probabilities[0][0],1);}
 for(const depth of [0,25,1.5])assert.throws(()=>createMoeTransformer({depth}));assert.throws(()=>runMoeTransformer([],model));assert.throws(()=>runMoeTransformer([{hidden:[NaN]}],model));assert.throws(()=>moeCausalAttention([[0]],model.blocks[0].attention));
});
test('final readout stays fixed when changing the inspected block; no fake vocabulary words',()=>{
 const {run,model}=fixture(),a=moeStackReadout(run,model,2,0),b=moeStackReadout(run,model,2,3);
 assert.equal(a.replace('Inspecting block 1','Inspecting block 4'),b);assert.match(a,/After all 4 blocks/);assert.match(a,/W_vocab · 8 × 8/);assert.match(a,/not real words/);assert.match(a,/position 2/);assert.equal((a.match(/logit =/g)||[]).length,8);
 const attention=moeAttentionSummary(run.blocks[1],2),projection=moeProjectionSummary(run.blocks[1],2);assert.match(attention,/block 2 input/);assert.match(projection,/× Wₒ/);assert.match(projection,/h after attention residual/);
});
test('block boundaries and keyboard navigation clamp; complete-block animation includes attention first',()=>{
 assert.equal(adjacentMoeBlock(0,'ArrowLeft',4),0);assert.equal(adjacentMoeBlock(3,'ArrowRight',4),3);assert.equal(adjacentMoeBlock(0,'End',4),3);assert.equal(adjacentMoeBlock(3,'Home',4),0);assert.equal(adjacentMoeBlock(1,'Tab',4),null);
 assert.equal(moeLoopFrame(0,{includeAttention:true}).phase,'attention');assert.equal(moeLoopFrame(750,{includeAttention:true}).phase,'projection');assert.equal(moeLoopFrame(1500,{includeAttention:true}).phase,'input');assert.equal(moeLoopFrame(5250,{includeAttention:true}).phase,'complete');assert.equal(moeLoopFrame(6750,{includeAttention:true}).cycle,1);
});
test('manual block inspection selects cached block results and preserves the user token',async()=>{
 const app=await readFile(new URL('../src/moe-app.mjs',import.meta.url),'utf8'),fn=app.slice(app.indexOf('function selectBlock('),app.indexOf('function render('));assert.match(fn,/state.layer=next;render\(\)/);assert.doesNotMatch(fn,/runMoeTransformer|state.selected=/);assert.match(app,/batch=stack.blocks\[state.layer\].ffn/);assert.match(app,/model=transformer.blocks\[state.layer\].ffn/);assert.match(app,/tokens\[state.selected\].hidden\[d\]=/);
});

test('unselected experts across the full stack are not evaluated',()=>{
 const tokens=teachingTokens('one'),model=createMoeTransformer({activation:'swiglu'}),run=runMoeTransformer(tokens,model,{k:2});
 model.blocks.forEach((block,l)=>{const selected=new Set(run.blocks[l].ffn.traces.flatMap(t=>t.selected));block.ffn.experts=block.ffn.experts.map((expert,e)=>selected.has(e)?expert:new Proxy(expert,{get(){throw Error('Unused expert accessed');}}));});
 assert.deepEqual(runMoeTransformer(tokens,model,{k:2}).logits,run.logits);
});
