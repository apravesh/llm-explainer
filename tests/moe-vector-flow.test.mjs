import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {moeVectorMarkup,expertVectorMarkup,contributionVectorMarkup,moeVectorEdges} from '../src/moe-vector-flow.mjs';
import {createMoeModel,teachingTokens,runMoeToken} from '../src/moe-engine.mjs';
test('vector strip retains feature count/order and exposes values only on hover',()=>{
 const values=[-2,1,.1234,0,3,4,-1,2],html=moeVectorMarkup(values,'h');assert.equal((html.match(/<i /g)||[]).length,8);assert.match(html,/1 by 8 vector/);assert.match(html,/h\[2\] = 0.1234000/);assert.doesNotMatch(html,/>0.1234</);assert.match(moeVectorMarkup([1],'<script>'),/&lt;script&gt;/);
});
for(const activation of ['gelu','swiglu'])for(const k of [1,2])test(`${activation} top-${k}: selected experts show full input and correct raw/weighted output vectors`,()=>{
 const tr=runMoeToken(teachingTokens('cat')[0].hidden,createMoeModel(108,{activation}),{k}),before=structuredClone(tr);
 for(let e=0;e<4;e++){const html=expertVectorMarkup(tr,e);if(tr.selected.includes(e)){assert.equal((html.match(/class="mv-vector"/g)||[]).length,2);assert.ok(html.includes(moeVectorMarkup(tr.normalized,'x')));assert.ok(html.includes(moeVectorMarkup(tr.experts[e].output,`y${e+1}`)));}else{assert.match(html,/no vector sent/);assert.doesNotMatch(html,/mv-vector/);}}
 const out=contributionVectorMarkup(tr);assert.equal((out.match(/data-contribution-node=/g)||[]).length,k);tr.selected.forEach(e=>assert.ok(out.includes(moeVectorMarkup(tr.contributions[e],`g${e+1} y${e+1}`))));assert.deepEqual(tr,before);
});
test('vector arrows originate at x, not router scores, and join weighted outputs with residual h',()=>{
 const r=(x,y,w=50,h=20)=>({x,y,w,h}),boxes={x:r(0,100),h:r(0,40),router:r(120,80),experts:{0:r(300,70),1:r(300,180)},expertOutputs:{0:r(390,95),1:r(390,205)},contributions:{0:r(500,90),1:r(500,200)},weighted:{0:r(590,90),1:r(590,200)},sum:r(550,290),add:r(740,250),output:r(740,340)};
 const {edges}=moeVectorEdges(boxes,[0,1]);assert.equal(edges.filter(e=>e.kind==='dispatch').length,2);edges.filter(e=>e.kind==='dispatch').forEach(e=>assert.ok(e.path.startsWith('M50 110')));assert.ok(edges.find(e=>e.id==='h-residual').path.startsWith('M0 50'));assert.ok(edges.some(e=>e.id==='weight-1-sum'));assert.ok(edges.every(e=>!e.path.includes('NaN')));
 const dense=moeVectorEdges(boxes,[0],{dense:true}).edges;assert.ok(!dense.some(e=>e.id==='x-router'));assert.equal(dense.filter(e=>e.kind==='dispatch').length,1);
});
test('overview and animation render vector strips, not raw hidden-value tables',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),app=await read('src/moe-app.mjs'),html=await read('moe.html');
 assert.doesNotMatch(app,/function hiddenStack|innerHTML=chips\(t\(\).mixed\)|innerHTML=chips\(moeMixPrefix/);
 assert.match(app,/numbers\('input'/);assert.match(app,/numbers\('combine'/);assert.match(app,/numbers\('output'/);assert.match(app,/moeVectorMarkup\(moeMixPrefix/);assert.match(html,/id="normalized-vector"/);assert.match(html,/id="moe-add-node"/);assert.match(html,/class="moe-flow-scroll"/);
});
