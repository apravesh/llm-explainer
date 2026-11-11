import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {INITIAL_TRAINING_STATE,cloneTrainingState,softmax,topK,forward,backward,trainStep} from '../src/moe-training.mjs';
const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8');
const lossAt=(s,mode,balance=0)=>{const f=forward(s,mode);return f.loss+(balance?backward(s,mode,{balance}).balanceLoss:0);};
function numericGrad(state,mode,balance){return state.logits.map((_,i)=>{const h=1e-6,a=cloneTrainingState(state),b=cloneTrainingState(state);a.logits[i]+=h;b.logits[i]-=h;return (lossAt(a,mode,balance)-lossAt(b,mode,balance))/(2*h);});}
test('softmax and top-k basics',()=>{const p=softmax([1,2,3]);assert.ok(Math.abs(p.reduce((a,b)=>a+b)-1)<1e-12);assert.deepEqual(topK([.1,.9,.5,.9],2),[1,3]);});
test('analytic router gradients match finite differences in every mode',()=>{
 for(const mode of ['top2','switch','top1'])for(const balance of [0,.05]){
  const s=cloneTrainingState(),g=backward(s,mode,{balance}),n=numericGrad(s,mode,balance);
  g.logits.forEach((v,i)=>assert.ok(Math.abs(v-n[i])<1e-6,`${mode} ${balance} logit ${i}: ${v} vs ${n[i]}`));
 }
});
test('task loss sends no gradient to unselected experts or, for renormalized top-k, their logits',()=>{
 const g=backward(cloneTrainingState(),'top2');assert.deepEqual(g.selected,[0,1]);
 assert.equal(g.taskLogit[2],0);assert.equal(g.taskLogit[3],0);assert.deepEqual(g.experts[2],[0,0]);assert.deepEqual(g.experts[3],[0,0]);
 const sw=backward(cloneTrainingState(),'switch');assert.notEqual(sw.taskLogit[2],0,'raw-probability gates touch every logit via the softmax denominator');
});
test('top-1 with a renormalized gate gives the router zero task gradient',()=>{const g=backward(cloneTrainingState(),'top1');assert.ok(g.logits.every(v=>v===0));assert.equal(g.gates[0],1);});
test('without balancing E3 is never discovered; with balancing it is and the loss falls',()=>{
 let a=cloneTrainingState(),b=cloneTrainingState();
 for(let i=0;i<120;i++){a=trainStep(a,'top2').state;b=trainStep(b,'top2',{balance:.05}).state;}
 assert.ok(!forward(a,'top2').selected.includes(2));assert.ok(forward(a,'top2').loss>.5);
 assert.ok(forward(b,'top2').selected.includes(2));assert.ok(forward(b,'top2').loss<.02);
});
test('training experts moves only the selected ones',()=>{const {state}=trainStep(cloneTrainingState(),'top2',{trainExperts:true});assert.notDeepEqual(state.experts[0],INITIAL_TRAINING_STATE.experts[0]);assert.deepEqual(state.experts[2],INITIAL_TRAINING_STATE.experts[2]);assert.deepEqual(state.experts[3],INITIAL_TRAINING_STATE.experts[3]);});
test('moe page includes the training section and its module',async()=>{const html=await read('moe.html');
 for(const id of ['moe-training','train-compare','tl-plane','tl-bars','tl-stepbtn','tl-balance'])assert.match(html,new RegExp(`id="${id}"`));
 assert.match(html,/src="src\/moe-training-app.mjs"/);assert.ok(html.indexOf('id="moe-training"')<html.indexOf('class="variants reading"'));assert.match(html,/router z-loss/);});
