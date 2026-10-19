import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {runEncoder,patchify,rng} from '../src/engine.mjs';
import {PATCH_TOUR_STEPS,createPatchTourPlan,patchTourFrame,createPatchTourClock} from '../src/patch-tour.mjs';
function fixture(model='vit',layer=0){const random=rng(164),patches=patchify(Uint8ClampedArray.from({length:48*32*4},()=>random()*256),48,32,16),result=runEncoder(patches,3,model),s={model,selected:2,patch:16,cols:3,rows:2,layer,head:1};return {s,patches,result,plan:createPatchTourPlan(s,patches,result)};}
for(const model of ['vit','pixtral']){
 test(`${model}: tour uses actual RGB and every scalar in the selected patch`,()=>{
  const {plan}=fixture(model),first=patchTourFrame(plan,0,0),last=patchTourFrame(plan,0,1);
  assert.deepEqual(first.rgb,plan.patch.pixels.slice(0,3).map(v=>Math.round((v+1)*127.5)));assert.equal(last.row,15);assert.equal(last.col,15);
  const flat=patchTourFrame(plan,1,1);assert.strictEqual(flat.values,plan.patch.pixels);assert.equal(flat.scalar,767);assert.equal(flat.start,756);assert.equal(flat.channel,2);
 });
 test(`${model}: embedding animation ends at the exact linear projection`,()=>{
  const {plan,result,s}=fixture(model),first=patchTourFrame(plan,2,0),last=patchTourFrame(plan,2,1);assert.equal(first.sum,0);assert.equal(last.terms,768);assert.equal(last.sum,result.embeddings[s.selected][0]);assert.equal(last.sum,last.expected);
 });
 test(`${model}: attention accumulates actual values to the exact selected head output`,()=>{
  const {plan}=fixture(model),first=patchTourFrame(plan,3,0),last=patchTourFrame(plan,3,1);
  assert.deepEqual(first.sum,Array(8).fill(0));assert.equal(last.terms,plan.head.values.length);assert.deepEqual(last.sum,plan.head.out[plan.index]);assert.deepEqual(last.expected,last.sum);
  const mid=patchTourFrame(plan,3,.5);for(let d=0;d<8;d++)assert.equal(mid.sum[d],plan.head.values.slice(0,mid.terms).reduce((sum,v,k)=>sum+plan.head.attention[plan.index][k]*v[d],0));
 });
 test(`${model}: MLP visibly completes expansion, activation, contraction and residual`,()=>{
  const {plan}=fixture(model),l=plan.layer,t=plan.index;
  const up=patchTourFrame(plan,4,.21);assert.equal(up.operation,'up');assert.equal(up.terms,24);assert.equal(up.sum,l.preActivation[t][0]);
  const activation=patchTourFrame(plan,4,.46);assert.equal(activation.kind,'activation');assert.strictEqual(activation.output,l.expanded[t]);assert.equal(activation.revealed,48);
  const down=patchTourFrame(plan,4,.71);assert.equal(down.operation,'down');assert.equal(down.terms,48);assert.equal(down.sum,l.mlpDelta[t][0]);
  const residual=patchTourFrame(plan,4,1);assert.equal(residual.kind,'residual');assert.strictEqual(residual.output,l.output[t]);assert.equal(residual.revealed,24);
 });
 test(`${model}: output continues through remaining blocks to the real final state`,()=>{
  for(let layer=0;layer<4;layer++){const {plan,result}=fixture(model,layer),end=patchTourFrame(plan,5,1);assert.ok(end.final);assert.strictEqual(end.output,result.final[plan.index]);assert.equal(end.block,4);if(layer<3){const first=patchTourFrame(plan,5,0);assert.equal(first.block,layer+2);assert.strictEqual(first.output,result.layers[layer+1].output[plan.index]);}}
 });
 test(`${model}: final handoff is explicitly symbolic and does not mistake a patch for CLS`,()=>{
  const {plan,result,s}=fixture(model),end=patchTourFrame(plan,6,1);assert.equal(end.symbolic,true);assert.equal(end.phase,2);assert.strictEqual(end.output,model==='pixtral'?result.projected[s.selected]:result.final[0]);
 });
 test(`${model}: tour never mutates selection, tensors, or weights`,()=>{
  const {plan,s,result}=fixture(model),before=structuredClone(result),state={...s};
  for(let step=0;step<7;step++)for(const p of [0,.21,.49,.71,1])patchTourFrame(plan,step,p);
  assert.deepEqual(result,before);assert.deepEqual(s,state);
 });
}
test('tour requires an actual patch and has seven finite playback durations',()=>{
 const {s,patches,result}=fixture();assert.throws(()=>createPatchTourPlan({...s,selected:-1},patches,result),/image patch/);assert.throws(()=>createPatchTourPlan({...s,selected:6},patches,result),/image patch/);assert.equal(PATCH_TOUR_STEPS.length,7);assert.ok(PATCH_TOUR_STEPS.every(s=>s.duration>0));
});
function clockHarness(){let id=0,now=0;const pending=new Map(),all=[],steps=[],frames=[],states=[];
 const clock=createPatchTourClock({request:cb=>{pending.set(++id,cb);all.push(cb);return id;},cancel:id=>pending.delete(id),onStep:s=>steps.push({...s}),onFrame:s=>frames.push({...s}),onState:s=>states.push({...s}),durations:Array(7).fill(100)});
 const advance=(ms=20)=>{now+=ms;const calls=[...pending.values()];pending.clear();calls.forEach(cb=>cb(now));};
 return {clock,pending,all,steps,frames,states,advance};
}
test('play immediately animates stage one and automatically advances to completion',()=>{
 const h=clockHarness();h.clock.start();assert.equal(h.steps[0].step,0);assert.equal(h.frames[0].progress,0);assert.equal(h.pending.size,1);
 for(let i=0;i<42;i++)h.advance();assert.deepEqual(h.steps.map(s=>s.step),[0,1,2,3,4,5,6]);assert.ok(h.frames.some(s=>s.step===0&&s.progress>0&&s.progress<1));assert.equal(h.clock.snapshot.completed,true);assert.equal(h.clock.snapshot.running,false);assert.equal(h.pending.size,0);
});
test('pause cancels timing and resume retains position without counting paused time',()=>{
 const h=clockHarness();h.clock.start();h.advance();h.advance(30);h.clock.pause();const saved=h.clock.snapshot;assert.equal(saved.elapsed,30);assert.equal(h.pending.size,0);h.advance(5000);assert.equal(h.clock.snapshot.elapsed,30);h.clock.resume();h.advance();assert.equal(h.clock.snapshot.elapsed,30);h.advance(20);assert.equal(h.clock.snapshot.elapsed,50);
});
test('manual stage navigation stays paused and clamps to valid bounds',()=>{
 const h=clockHarness();h.clock.start();h.clock.seek(4);assert.equal(h.clock.snapshot.step,4);assert.equal(h.clock.snapshot.running,false);assert.equal(h.pending.size,0);h.clock.seek(99);assert.equal(h.clock.snapshot.step,6);h.clock.seek(-2);assert.equal(h.clock.snapshot.step,0);
});
test('next or previous while playing starts that stage from the beginning',()=>{
 const h=clockHarness();h.clock.start();h.advance();h.advance(30);h.clock.seek(3,{play:true});assert.equal(h.clock.snapshot.step,3);assert.equal(h.clock.snapshot.elapsed,0);assert.equal(h.pending.size,1);h.advance();h.advance(100);assert.equal(h.clock.snapshot.step,4);
});
test('replay invalidates stale frames and creates exactly one playback loop',()=>{
 const h=clockHarness();h.clock.start();h.advance();h.advance(50);const stale=h.all.at(-1);h.clock.start();stale(9999);assert.equal(h.clock.snapshot.step,0);assert.equal(h.clock.snapshot.progress,0);assert.equal(h.pending.size,1);
});
test('stop prevents stale callbacks from advancing or reopening the tour',()=>{
 const h=clockHarness();h.clock.start();const stale=h.all.at(-1);h.clock.stop();const frames=h.frames.length;stale(9999);assert.equal(h.clock.snapshot.active,false);assert.equal(h.pending.size,0);assert.equal(h.frames.length,frames);
});
test('reduced-motion/manual playback shows end frames and schedules no motion',()=>{
 const h=clockHarness();h.clock.start({autoplay:false});assert.equal(h.clock.snapshot.progress,1);assert.equal(h.clock.snapshot.running,false);assert.equal(h.pending.size,0);h.clock.seek(4,{showEnd:true});assert.equal(h.frames.at(-1).progress,1);assert.equal(h.clock.snapshot.step,4);assert.equal(h.pending.size,0);
});
test('long frame gaps cannot skip unseen stages',()=>{
 const h=clockHarness();h.clock.start();h.advance();h.advance(100000);assert.equal(h.clock.snapshot.step,1);assert.equal(h.clock.snapshot.elapsed,0);assert.deepEqual(h.steps.map(s=>s.step),[0,1]);
});
test('the offline bundle includes the tour and no longer has the inert tour handler',async()=>{
 const html=await readFile(new URL('../standalone.html',import.meta.url),'utf8');assert.match(html,/function createPatchTour/);assert.match(html,/Play the animated walkthrough/);assert.doesNotMatch(html,/Use the right arrow to advance through the seven stages/);
});
