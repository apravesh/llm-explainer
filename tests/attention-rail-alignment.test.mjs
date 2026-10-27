import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {attentionRailOffset,mountAttentionRailAlignment} from '../src/attention-rail-alignment.mjs';
class Events{
 constructor(){this.handlers=new Map();}
 addEventListener(type,fn){if(!this.handlers.has(type))this.handlers.set(type,new Set());this.handlers.get(type).add(fn);}
 removeEventListener(type,fn){this.handlers.get(type)?.delete(fn);}
 emit(type){for(const fn of this.handlers.get(type)||[])fn();}
}
function harness(){
 const jobs=new Map(),properties=new Map();let sequence=0,targetTop=480,baseTop=100,observer;
 const media=Object.assign(new Events(),{matches:false}),win=Object.assign(new Events(),{
  matchMedia:()=>media,requestAnimationFrame:fn=>{jobs.set(++sequence,fn);return sequence;},cancelAnimationFrame:id=>jobs.delete(id),
  ResizeObserver:class{constructor(fn){observer=this;this.fn=fn;this.targets=[];this.disconnected=false;}observe(el){this.targets.push(el);}disconnect(){this.disconnected=true;}}
 });
 const aside={style:{setProperty:(k,v)=>properties.set(k,v),removeProperty:k=>properties.delete(k)},querySelector:()=>({})};
 const root={ownerDocument:{defaultView:win},isConnected:true,closest:()=>({}),querySelector:()=>({getBoundingClientRect:()=>({top:targetTop})})};
 const rail={isConnected:true,scrollTop:878,closest:()=>aside,getBoundingClientRect:()=>({top:baseTop+(parseFloat(properties.get('--attention-rail-offset'))||0)})};
 const controller=mountAttentionRailAlignment(root,rail);
 const flush=()=>{const pending=[...jobs.values()];jobs.clear();pending.forEach(fn=>fn());};
 return {controller,root,rail,win,media,jobs,properties,flush,get observer(){return observer;},setTarget(v){targetTop=v;},scrollPage(amount){baseTop-=amount;targetTop-=amount;}};
}
test('alignment offsets the list to the matrix without cumulative movement',()=>{
 assert.equal(attentionRailOffset(0,100,480),380);assert.equal(attentionRailOffset(380,480,480),380);
 assert.equal(attentionRailOffset(380,480,510),410);assert.equal(attentionRailOffset(380,480,450),350);
 assert.equal(attentionRailOffset(0,600,500),0);assert.equal(attentionRailOffset(380,480,510,true),0);
});
test('measurement lines up the list and preserves its patch scroll position',()=>{
 const h=harness();h.flush();assert.equal(h.properties.get('--attention-rail-offset'),'380px');assert.equal(h.rail.scrollTop,878);
 h.controller.schedule();h.flush();assert.equal(h.properties.get('--attention-rail-offset'),'380px');h.controller.dispose();
});
test('resizing recalculates the offset from actual geometry',()=>{
 const h=harness();h.flush();h.setTarget(550);h.observer.fn();h.flush();assert.equal(h.properties.get('--attention-rail-offset'),'450px');h.controller.dispose();
});
test('page scrolling does not drift the list away from the matrix',()=>{
 const h=harness();h.flush();h.scrollPage(300);h.controller.schedule();h.flush();assert.equal(h.properties.get('--attention-rail-offset'),'380px');h.controller.dispose();
});
test('horizontal mobile layout removes desktop spacing and restores it on return',()=>{
 const h=harness();h.flush();h.media.matches=true;h.media.emit('change');h.flush();assert.equal(h.properties.get('--attention-rail-offset'),'0px');
 h.media.matches=false;h.media.emit('change');h.flush();assert.equal(h.properties.get('--attention-rail-offset'),'380px');h.controller.dispose();
});
test('rapid measurements coalesce and disposal removes observers, frames and styles',()=>{
 const h=harness();h.controller.schedule();h.controller.schedule();assert.equal(h.jobs.size,1);h.flush();h.controller.schedule();h.controller.dispose();assert.equal(h.jobs.size,0);assert.equal(h.properties.size,0);assert.ok(h.observer.disconnected);
 assert.ok([...h.win.handlers.values()].every(set=>!set.size));assert.ok([...h.media.handlers.values()].every(set=>!set.size));h.controller.schedule();assert.equal(h.jobs.size,0);
});
test('removed views never get a stale alignment applied',()=>{
 const h=harness();h.root.isConnected=false;h.flush();assert.equal(h.properties.size,0);h.controller.dispose();
});
test('alignment is included in the standalone attention view and cleaned up with that view',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');const [window,bundle]=await Promise.all(['src/attention-window.mjs','standalone.html'].map(read));
 assert.match(window,/mountAttentionRailAlignment\(root,rail\)/);assert.match(window,/railAlignment.schedule\(\)/);assert.match(window,/railAlignment.dispose\(\)/);assert.match(bundle,/function mountAttentionRailAlignment/);
});
