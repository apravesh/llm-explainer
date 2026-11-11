import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile,access} from 'node:fs/promises';
import {ANSWERS,probabilities,draw,random,sampleGroup,groupAdvantages,clippedObjective,policyUpdate,initialRLState,advanceRL,inferSnapshot,LEARNING_RATE} from '../src/rl-engine.mjs';
const near=(a,b,t=1e-9)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
const steps=(s,n)=>{for(let i=0;i<n;i++)s=advanceRL(s);return s;};
test('RL sampling is seeded, uses normalized frozen probabilities and permits repeated draws',()=>{
 const p=probabilities([1001,1000,999]);near(p.reduce((a,b)=>a+b,0),1);assert.deepEqual(p,probabilities([1,0,-1]));assert.equal(draw([.2,.5,.3],.19),0);assert.equal(draw([.2,.5,.3],.2),1);assert.equal(draw([.2,.5,.3],.99),2);
 const a=sampleGroup([0,0,0],1);assert.deepEqual(a,sampleGroup([0,0,0],1));assert.deepEqual(a.samples.map(s=>s.answer),['4','3','4','5']);a.samples.forEach(s=>{near(s.oldProbability,1/3);assert.equal(s.answer,ANSWERS[s.action]);});assert.notEqual(a.seed,1);assert.ok(random(0).value>=0&&random(0).value<1);
 assert.throws(()=>draw([.2,.5,.3],1));assert.throws(()=>probabilities([NaN,0,0]));
});
test('group-relative advantages use population std and zero out equal rewards',()=>{
 const a=groupAdvantages([1,0,1,0]);near(a.mean,.5);near(a.std,.5);assert.deepEqual(a.advantages,[1,-1,1,-1]);
 for(const rewards of [[1,1,1,1],[0,0,0,0]]){const s=groupAdvantages(rewards);assert.equal(s.std,0);assert.deepEqual(s.advantages,[0,0,0,0]);}
 const b=groupAdvantages([1,0,0,0]);near(b.advantages.reduce((a,b)=>a+b,0),0);near(b.advantages.reduce((a,b)=>a+b*b,0)/4,1);
});
test('clipped GRPO gradient matches finite differences away from clipping boundaries',()=>{
 const samples=sampleGroup([0,0,0],1).samples,advantages=[1,-1,1,-1];
 for(const logits of [[0,0,0],[.01,.08,-.02],[-.8,.9,-.7],[.7,-.6,.8]]){
 const r=clippedObjective(logits,samples,advantages);near(r.gradient.reduce((a,b)=>a+b,0),0);
 for(let j=0;j<3;j++){const plus=[...logits],minus=[...logits];plus[j]+=1e-6;minus[j]-=1e-6;near(r.gradient[j],(clippedObjective(plus,samples,advantages).loss-clippedObjective(minus,samples,advantages).loss)/2e-6,1e-7);}
 }
 const initial=clippedObjective([0,0,0],samples,advantages);near(initial.loss,0);assert.ok(initial.gradient.some(g=>g!==0));
 const saturated=clippedObjective([-.8,.9,-.7],samples,advantages);assert.ok(saturated.terms.every(t=>t.clipped));assert.deepEqual(saturated.gradient,[0,0,0]);
 const wrongDirection=clippedObjective([.7,-.6,.8],samples,advantages);assert.ok(wrongDirection.terms.every(t=>!t.clipped));assert.ok(wrongDirection.gradient.some(g=>g!==0));
});
test('SGD updates actual logits; zero-advantage groups leave them unchanged',()=>{
 const samples=sampleGroup([0,0,0],1).samples,logits=[0,0,0],step=policyUpdate(logits,samples,[1,-1,1,-1]);
 for(let j=0;j<3;j++)near(step.logits[j],logits[j]-LEARNING_RATE*step.before.gradient[j]);assert.ok(probabilities(step.logits)[1]>1/3);assert.deepEqual(logits,[0,0,0]);assert.deepEqual(policyUpdate(logits,samples,[0,0,0,0]).logits,logits);
});
test('rollout state stays frozen through three updates and refreshes only at snapshot',()=>{
 let s=initialRLState();const original=structuredClone(s);s=advanceRL(s);assert.deepEqual(original,initialRLState());assert.deepEqual(s.learner,[0,0,0]);const group=structuredClone(s.group.samples);
 s=steps(s,2);assert.equal(s.phase,3);assert.deepEqual(s.group.rewards,[1,0,1,0]);
 for(let n=1;n<=3;n++){s=advanceRL(s);assert.equal(s.updates,n);assert.deepEqual(s.sampler,[0,0,0]);assert.deepEqual(s.group.samples,group);assert.equal(s.snapshots.length,1);}
 s=advanceRL(s);assert.equal(s.phase,7);assert.equal(s.round,1);assert.equal(s.snapshots.length,2);assert.deepEqual(s.sampler,s.learner);assert.notEqual(s.sampler,s.learner);assert.notEqual(s.snapshots[1].logits,s.learner);assert.deepEqual(s.snapshots[0].logits,[0,0,0]);
 const next=advanceRL(s);assert.equal(next.group.version,1);next.group.samples.forEach(r=>near(r.oldProbability,probabilities(s.sampler)[r.action]));assert.equal(next.phase,1);assert.equal(next.updates,3);
});
test('constant rewards create no improvement; inference never changes any snapshot or training state',()=>{
 const s=steps(initialRLState('constant'),7);assert.equal(s.updates,3);assert.deepEqual(s.learner,[0,0,0]);const before=structuredClone(s);for(const snapshot of s.snapshots)for(let seed=0;seed<5;seed++){const result=inferSnapshot(snapshot,seed);assert.ok(ANSWERS.includes(result.answer));assert.deepEqual(result,inferSnapshot(snapshot,seed));}assert.deepEqual(s,before);
 assert.throws(()=>initialRLState('bad'));
});
test('multiple rounds preserve every saved checkpoint and progress in groups of three updates',()=>{
 let s=initialRLState();for(let round=1;round<=10;round++){const old=structuredClone(s.snapshots);s=steps(s,7);assert.equal(s.round,round);assert.equal(s.updates,3*round);assert.deepEqual(s.snapshots.slice(0,-1),old);near(probabilities(s.learner).reduce((a,b)=>a+b,0),1);}
});
test('RL chapter replaces placeholders and all local navigation/assets resolve',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),html=await read('post-training-rl.html');assert.ok(!html.includes('training-placeholder'));for(const id of ['setup','loop','checkpoints','rl-next','rl-play','rl-reset','rl-reward','inference-checkpoint','inference-sample'])assert.ok(html.includes(`id="${id}"`));
 for(const [,href] of html.matchAll(/(?:href|src)="([^"]+)"/g))if(!/^(https?:|#|\.\/$)/.test(href))await access(new URL('../'+href.split('#')[0],import.meta.url));
 for(const page of ['index.html','pretraining.html','post-training-instruct.html','post-training-rl.html'])assert.ok(!(await read(page)).includes('Currently under construction'));
 assert.match(html,/three trainable logits/i);assert.match(html,/β = 0/);assert.match(html,/not a GRPO requirement/);assert.match(html,/not files or a production deployment/);assert.match(html,/not a hard bound/);
});
test('RL autoplay is finite and cancels stale callbacks on reset, visibility and motion changes',async()=>{
 const app=await readFile(new URL('../src/rl-app.mjs',import.meta.url),'utf8');assert.match(app,/ticket!==epoch\|\|!playing/);assert.match(app,/clearTimeout\(timer\)/);assert.match(app,/if\(state.phase===lastPhase\(\)\)\{cancel\(\);render\(\);return;/);assert.match(app,/grpo:\{last:7/);assert.match(app,/visibilitychange/);assert.match(app,/prefers-reduced-motion/);assert.match(app,/inferenceSeed=981/);assert.match(app,/servingVersion=\+e.target.value/);
});
test('REINFORCE: mean-baseline gradient matches finite differences and one update per group raises p(4)',async()=>{
 const m=await import('../src/rl-engine.mjs');let s=m.initialReinforceState();for(let i=0;i<3;i++)s=m.advanceReinforce(s);
 const {samples,stats}=s.group,g=m.reinforceObjective(s.learner,samples,stats.advantages),eps=1e-6;
 assert.ok(Math.abs(stats.advantages.reduce((a,b)=>a+b,0))<1e-12);
 for(let j=0;j<3;j++){const a=[...s.learner],b=[...s.learner];a[j]+=eps;b[j]-=eps;const num=(m.reinforceObjective(a,samples,stats.advantages).loss-m.reinforceObjective(b,samples,stats.advantages).loss)/(2*eps);assert.ok(Math.abs(num-g.gradient[j])<1e-6);}
 const before=m.probabilities(s.learner)[1];s=m.advanceReinforce(s);assert.equal(s.phase,4);assert.equal(s.updates,1);assert.equal(s.snapshots.length,2);assert.deepEqual(s.sampler,s.learner);
 for(let i=0;i<40;i++)s=m.advanceReinforce(s);assert.ok(m.probabilities(s.learner)[1]>before+.2);
 let flat=m.initialReinforceState('constant');for(let i=0;i<8;i++)flat=m.advanceReinforce(flat);assert.deepEqual(flat.learner,[0,0,0]);
});
test('DPO: margin starts at zero, gradient matches finite differences, reference stays frozen and loss falls',async()=>{
 const m=await import('../src/rl-engine.mjs'),ref=[0,0,0],theta=[.3,-.2,.1],g=m.dpoObjective(theta,ref),eps=1e-6;
 assert.ok(m.dpoObjective(ref,ref).terms.every(t=>Math.abs(t.margin)<1e-12&&Math.abs(t.loss-Math.log(2))<1e-12));
 for(let j=0;j<3;j++){const a=[...theta],b=[...theta];a[j]+=eps;b[j]-=eps;assert.ok(Math.abs((m.dpoObjective(a,ref).loss-m.dpoObjective(b,ref).loss)/(2*eps)-g.gradient[j])<1e-6);}
 let s=m.initialDpoState();const losses=[];for(let i=0;i<24;i++){s=m.advanceDpo(s);if(s.phase===4)losses.push(s.lastUpdate.before.loss);}
 assert.deepEqual(s.sampler,[0,0,0]);assert.ok(losses.every((l,i)=>i===0||l<losses[i-1]));assert.ok(m.probabilities(s.learner)[1]>.8);assert.throws(()=>m.dpoObjective(theta,ref,[{chosen:1,rejected:1}]));
});
test('RL page offers REINFORCE first, then GRPO and DPO, each with its own math',async()=>{
 const html=await readFile(new URL('../post-training-rl.html',import.meta.url),'utf8');
 assert.deepEqual([...html.matchAll(/data-rl-algo="(\w+)"/g)].map(x=>x[1]),['reinforce','grpo','dpo']);assert.match(html,/data-rl-algo="reinforce" aria-pressed="true"/);
 for(const key of ['reinforce','grpo','dpo'])assert.ok(html.includes(`data-only="${key}"`));assert.match(html,/Direct Preference Optimization/);
});
