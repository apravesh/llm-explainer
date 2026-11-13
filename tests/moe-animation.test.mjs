import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {MOE_LOOP_TIMING,moeLoopFrame,mountMoeAnimation} from '../src/moe-animation.mjs';
class Events{constructor(){this.handlers=new Map();}addEventListener(k,f){if(!this.handlers.has(k))this.handlers.set(k,new Set());this.handlers.get(k).add(f);}removeEventListener(k,f){this.handlers.get(k)?.delete(f);}emit(k){this.handlers.get(k)?.forEach(f=>f());}}
function harness({dense=false,experts=2,reduced=false,hidden=false}={}){
 let id=0,now=0,modal=false,observer;const frames=new Map(),writes=[],media=Object.assign(new Events(),{matches:reduced});
 const win={matchMedia:()=>media,requestAnimationFrame:f=>{frames.set(++id,f);return id;},cancelAnimationFrame:id=>frames.delete(id),IntersectionObserver:class{constructor(fn){observer=this;this.fn=fn;}observe(){}disconnect(){this.disconnected=true;}show(value){this.fn([{isIntersecting:value}]);}}};
 const doc=Object.assign(new Events(),{defaultView:win,hidden,querySelector:()=>modal?{}:null}),classes=new Set(),nodes=['input','router','experts','combine','output'].map(phase=>({dataset:{moeNode:phase},classList:{toggle(_,v){this.active=v;},remove(){this.active=false;}}}));
 const root={ownerDocument:doc,isConnected:true,dataset:{},classList:{add:c=>classes.add(c),remove:c=>classes.delete(c)},querySelectorAll:()=>nodes};
 const controller=mountMoeAnimation(root,{dense,experts,onFrame:s=>writes.push(s)});
 return {root,doc,media,frames,writes,controller,classes,nodes,get observer(){return observer;},advance(ms){now+=ms;const fs=[...frames.values()];frames.clear();fs.forEach(f=>f(now));},modal(v){modal=v;controller.sync();}};
}
test('loop visits every phase, shows complete mixture, holds and restarts',()=>{
 assert.equal(MOE_LOOP_TIMING.hold,1500);
 for(const [time,phase] of [[0,'input'],[750,'router'],[1500,'experts'],[2250,'combine'],[3000,'output'],[3750,'complete'],[5249,'complete'],[5250,'input']])assert.equal(moeLoopFrame(time).phase,phase);
 assert.equal(moeLoopFrame(2250).count,1);assert.equal(moeLoopFrame(2625).count,2);assert.equal(moeLoopFrame(5250).cycle,1);
 assert.equal(moeLoopFrame(750,{dense:true}).phase,'experts');assert.equal(moeLoopFrame(3000,{dense:true}).phase,'complete');assert.equal(moeLoopFrame(4500,{dense:true}).cycle,1);
});
test('autostarts one frame clock, loops indefinitely, and only writes changed frames',()=>{
 const h=harness();assert.equal(h.root.dataset.playback,'playing');assert.equal(h.frames.size,1);assert.equal(h.writes[0].phase,'input');h.advance(0);h.advance(750);assert.equal(h.writes.at(-1).phase,'router');h.advance(3000);assert.equal(h.writes.at(-1).phase,'complete');const count=h.writes.length;h.advance(1000);assert.equal(h.writes.length,count);h.advance(500);assert.equal(h.writes.at(-1).cycle,1);assert.equal(h.writes.at(-1).phase,'input');assert.equal(h.frames.size,1);h.controller.dispose();assert.equal(h.frames.size,0);
});
test('hidden tabs and offscreen flow pause and resume elapsed time without skipping',()=>{
 const h=harness();h.advance(0);h.advance(800);h.doc.hidden=true;h.doc.emit('visibilitychange');assert.equal(h.frames.size,0);assert.equal(h.root.dataset.playback,'paused');h.advance(100000);h.doc.hidden=false;h.doc.emit('visibilitychange');h.advance(0);assert.equal(h.writes.at(-1).phase,'router');h.observer.show(false);assert.equal(h.frames.size,0);h.advance(100000);h.observer.show(true);h.advance(0);h.advance(700);assert.equal(h.writes.at(-1).phase,'experts');h.controller.dispose();
});
test('modal closes resume automatically and reduced motion has no repeating callbacks',()=>{
 const h=harness();h.modal(true);assert.equal(h.frames.size,0);h.modal(false);assert.equal(h.frames.size,1);h.media.matches=true;h.media.emit('change');assert.equal(h.frames.size,0);assert.equal(h.writes.at(-1).phase,'complete');assert.equal(h.root.dataset.playback,'reduced-motion');assert.ok(h.nodes.every(n=>!n.classList.active));h.media.matches=false;h.media.emit('change');assert.equal(h.writes.at(-1).phase,'input');assert.equal(h.frames.size,1);h.controller.dispose();
 const initial=harness({reduced:true});assert.equal(initial.frames.size,0);assert.equal(initial.writes.at(-1).count,2);initial.controller.dispose();
});
test('initially hidden and detached views are safe; disposal invalidates stale callbacks',()=>{
 const hidden=harness({hidden:true});assert.equal(hidden.frames.size,0);hidden.controller.dispose();
 const h=harness(),stale=[...h.frames.values()][0];h.controller.dispose();const count=h.writes.length;stale(10000);assert.equal(h.writes.length,count);assert.equal(h.frames.size,0);assert.ok(h.observer.disconnected);assert.ok([...h.doc.handlers.values(),...h.media.handlers.values()].every(s=>s.size===0));
 const detached=harness();detached.root.isConnected=false;detached.advance(0);assert.equal(detached.frames.size,0);assert.ok(detached.observer.disconnected);
});
test('page has no stop/replay control; rerenders restart the current trace and resize only adjusts wires',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),app=await read('src/moe-app.mjs'),html=await read('moe-live.html');
 assert.doesNotMatch(app+html,/moe-animate|Stop animation|Animate this token|Stop animation|Replay/);
 assert.match(app,/drawWires\(\);startAutomaticAnimation\(\)/);assert.match(app,/new ResizeObserver\(drawWires\)/);assert.match(app,/addEventListener\('close',\(\)=>animationController\?\.sync\(\)\)/);
 assert.match(html,/id="flow-progress" class="flow-progress" aria-live="off"/);assert.match(app,/moeVectorMarkup\(moeMixPrefix\(trace,count\)/);
});
