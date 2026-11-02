import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matrix,FULL_CONFIGS} from '../src/full-config.mjs';
import {adjacentBlock,blockKeyTarget,blockHandoff,createBlockPlayback,BLOCK_PLAYBACK_INTERVAL,BLOCK_TRANSITION_MS} from '../src/block-navigation.mjs';
function harness(count=24){
 let index=0,ready=true,id=0,now=0;const pending=new Map(),all=[],selected=[],states=[];
 const playback=createBlockPlayback({request:cb=>{pending.set(++id,cb);all.push(cb);return id;},cancel:i=>pending.delete(i),getCurrent:()=>index,getCount:()=>count,canPlay:()=>ready,onSelect:i=>{index=i;selected.push(i);},onState:s=>states.push({...s})});
 function advance(ms=1800){now+=ms;const callbacks=[...pending.values()];pending.clear();callbacks.forEach(cb=>cb(now));}
 return {playback,pending,all,selected,states,advance,setReady(v){ready=v;},setIndex(i){index=i;},setCount(n){count=n;},get index(){return index;}};
}
test('block motion is visible but quicker than its automatic dwell time',()=>{
 assert.equal(BLOCK_PLAYBACK_INTERVAL,1800);assert.equal(BLOCK_TRANSITION_MS,420);assert.ok(BLOCK_TRANSITION_MS<BLOCK_PLAYBACK_INTERVAL);
});
for(const count of [24,12])test(`${count} blocks: arrows clamp; Home/End reach first and last`,()=>{
 assert.equal(adjacentBlock(0,-1,count),0);assert.equal(adjacentBlock(count-1,1,count),count-1);assert.equal(blockKeyTarget(3,'ArrowRight',count),4);assert.equal(blockKeyTarget(3,'ArrowLeft',count),2);assert.equal(blockKeyTarget(3,'Home',count),0);assert.equal(blockKeyTarget(3,'End',count),count-1);assert.equal(blockKeyTarget(3,'Tab',count),null);
});
for(const count of [24,12])test(`${count} blocks: automatic playback reaches every block exactly once and stops`,()=>{
 const h=harness(count);h.playback.play();assert.equal(h.pending.size,1);h.advance(0);
 for(let i=0;i<count;i++)h.advance();
 assert.deepEqual(h.selected,Array.from({length:count-1},(_,i)=>i+1));assert.equal(h.index,count-1);assert.equal(h.playback.snapshot.playing,false);assert.equal(h.playback.snapshot.completed,true);assert.equal(h.pending.size,0);
});
test('pause cancels timing; resume retains dwell position without counting paused time',()=>{
 const h=harness();h.playback.play();h.advance(0);h.advance(700);h.playback.pause();assert.equal(h.pending.size,0);assert.equal(h.playback.snapshot.elapsed,700);h.advance(50000);assert.equal(h.index,0);h.playback.play();h.advance(0);h.advance(1000);assert.equal(h.index,0);h.advance(100);assert.equal(h.index,1);
});
test('manual seek pauses playback and invalidates its stale callback',()=>{
 const h=harness();h.playback.play();h.advance(0);const stale=h.all.at(-1);h.playback.seek(10);stale(99999);assert.equal(h.index,10);assert.equal(h.playback.snapshot.playing,false);assert.equal(h.pending.size,0);assert.deepEqual(h.selected,[10]);assert.equal(h.playback.snapshot.elapsed,0);
});
test('restart returns to block one and owns a single cancellable clock',()=>{
 const h=harness();h.setIndex(12);h.playback.play();h.advance(0);const stale=h.all.at(-1);h.playback.restart();stale(999999);assert.equal(h.index,0);assert.equal(h.pending.size,1);h.advance(0);h.advance();assert.equal(h.index,1);
});
test('play at the final block replays from the start rather than wrapping unexpectedly while stepping',()=>{
 const h=harness();h.setIndex(23);h.playback.play();assert.equal(h.index,0);assert.deepEqual(h.selected,[0]);h.playback.pause();h.playback.seek(23);assert.equal(h.index,23);assert.equal(adjacentBlock(h.index,1,24),23);
});
test('unready data cannot start automatic playback; loss of readiness pauses safely',()=>{
 const h=harness();h.setReady(false);h.playback.play();assert.equal(h.pending.size,0);h.playback.restart();assert.equal(h.pending.size,0);assert.deepEqual(h.selected,[]);h.setReady(true);h.playback.play();h.advance(0);h.setReady(false);h.advance();assert.equal(h.playback.snapshot.playing,false);assert.equal(h.index,0);assert.equal(h.pending.size,0);
});
test('reset on a model change invalidates old callbacks and uses the new depth',()=>{
 const h=harness(24);h.setIndex(18);h.playback.play();h.advance(0);const stale=h.all.at(-1);h.playback.reset();h.setCount(12);h.setIndex(0);stale(999999);assert.equal(h.index,0);assert.equal(h.playback.snapshot.count,12);assert.equal(h.pending.size,0);h.playback.play();h.advance(0);for(let i=0;i<12;i++)h.advance();assert.equal(h.index,11);
});
test('a long frame gap advances only one visible block, never skips unseen layers',()=>{
 const h=harness();h.playback.play();h.advance(0);h.advance(999999);assert.equal(h.index,1);assert.deepEqual(h.selected,[1]);assert.equal(h.playback.snapshot.elapsed,0);
});
test('pause from inside a selection callback cannot leave a queued animation',()=>{
 let index=0,id=0;const callbacks=new Map();let clock;clock=createBlockPlayback({getCount:()=>3,getCurrent:()=>index,request:cb=>{callbacks.set(++id,cb);return id;},cancel:i=>callbacks.delete(i),onSelect:i=>{index=i;clock.pause();}});clock.play();let cb=callbacks.get(1);callbacks.clear();cb(0);cb=callbacks.get(2);callbacks.clear();cb(1800);assert.equal(index,1);assert.equal(clock.snapshot.playing,false);assert.equal(callbacks.size,0);
});
for(const model of ['pixtral','vit'])test(`${model}: selected token's block output is the next block's actual input`,()=>{
 const config=FULL_CONFIGS[model],tokens=model==='vit'?3:2,make=v=>matrix(tokens,config.width,new Float32Array(tokens*config.width).fill(v));
 const layers=[{input:make(1),afterAttention:make(2),output:make(3)},{input:make(3),afterAttention:make(4),output:make(5)}];
 const run={config,layers},state={selected:1,head:7,layer:0,feature:0},before=structuredClone(run);
 const a=blockHandoff(run,state.selected,0),b=blockHandoff(run,state.selected,1);
 assert.equal(a.row,model==='vit'?2:1);assert.equal(a.previousOutput,null);assert.equal(a.output,b.input);assert.equal(b.previousOutput,a.output);assert.equal(a.nextInput,b.input);assert.equal(b.nextInput,null);assert.equal(blockHandoff(run,state.selected,2),null);assert.deepEqual(run,before);assert.deepEqual(state,{selected:1,head:7,layer:0,feature:0});
 if(model==='vit')assert.equal(blockHandoff(run,-1,0).row,0);
});
test('served and offline interfaces expose only inline block arrows, without the bar or slider',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
 const [html,vision,app,nav]=await Promise.all(['standalone.html','vision.html','src/full-app.mjs','src/block-navigation.mjs'].map(read));
 for(const page of [html,vision]){assert.match(page,/data-inline-block-step/);assert.doesNotMatch(page,/id="layer"|id="layer-number"|ENCODER DEPTH|id="block-navigator"/);}
 assert.match(html,/function createInlineBlockNavigation/);assert.doesNotMatch(html,/inlineBlockControlsMarkup/);
 assert.doesNotMatch(app,/createBlockNavigation|\$\('#layer'\)/);
 assert.match(nav,/visibilitychange/);assert.match(nav,/prefers-reduced-motion: reduce/);assert.match(nav,/win.removeEventListener\('resize'/);
 assert.doesNotMatch(nav,/createElement|setInterval|data-block-play|data-block-index/);
});

test('Q/K/V and MLP operations follow the selected block, while readouts remain final',async()=>{
 const {createCPUBackend}=await import('../src/full-backend.mjs'),{runFullEncoder}=await import('../src/full-engine.mjs'),{fullOperation}=await import('../src/full-operations.mjs');
 const config={...FULL_CONFIGS.pixtral,width:32,heads:2,headWidth:16,ffWidth:64,depth:3,decoderWidth:64},backend=await createCPUBackend(),run={config,layers:[],pixels:matrix(2,768,Float32Array.from({length:1536},(_,i)=>Math.sin(i*.17)))};
 await runFullEncoder({pixels:run.pixels,rows:1,cols:2,config,backend,onEmbedding:v=>Object.assign(run,v),onLayer:(v,i)=>run.layers[i]=v,onOutput:v=>run.final=v,onProjector:v=>Object.assign(run,v)});
 const final=run.final,projected=run.projected,state={selected:1,key:0,head:1,layer:0};
 for(let layer=0;layer<3;layer++){
  state.layer=layer;
  for(const kind of ['q','k','v','mlp-up','mlp-gate','mlp-down','attention-output'])assert.equal(fullOperation(run,state,kind).blockIndex,layer);
  assert.strictEqual(fullOperation(run,state,'q').input,run.layers[layer].normalized);
  assert.strictEqual(fullOperation(run,state,'mlp-up').weights,run.layers[layer].weights.up);
  assert.strictEqual(fullOperation(run,state,'mlp-down').output,run.layers[layer].mlpDelta);
  assert.strictEqual(fullOperation(run,state,'projector-first').input,final);assert.strictEqual(fullOperation(run,state,'projector-second').output,projected);
  assert.equal(fullOperation(run,state,'embedding').blockIndex,null);assert.equal(fullOperation(run,state,'projector-first').blockIndex,null);
 }
 backend.destroy();
});
