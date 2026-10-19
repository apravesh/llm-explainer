import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ZOOM_TIMING,zoomDirection,boxTransform,createArchitectureMotion} from '../src/motion.mjs';

class Events {
 constructor(){this.listeners=new Map();}
 addEventListener(type,fn){if(!this.listeners.has(type))this.listeners.set(type,new Set());this.listeners.get(type).add(fn);}
 removeEventListener(type,fn){this.listeners.get(type)?.delete(fn);}
 emit(type,event={}){for(const fn of this.listeners.get(type)||[])fn({...event,type});}
}
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};}
function fakeStyle(){return {viewTransitionName:'',setProperty(key,value){this[key]=value;},removeProperty(key){delete this[key];}};}
function harness({native=true,animate=true,reduced=false,hidden=false}={}){
 const media=Object.assign(new Events(),{matches:reduced});
 const win=Object.assign(new Events(),{matchMedia:()=>media});
 const html={dataset:{},style:fakeStyle()};
 const doc=Object.assign(new Events(),{defaultView:win,documentElement:html,visibilityState:hidden?'hidden':'visible'});
 let expanded=null,mutations=0;const transitions=[],animations=[];
 function element(name,rect){return {name,style:fakeStyle(),dataset:{},getBoundingClientRect:()=>rect,querySelector:()=>null,querySelectorAll:()=>[],animate:!animate?undefined:frames=>{const done=deferred(),a={frames,finished:done.promise,cancelled:false,cancel(){this.cancelled=true;done.resolve();},end:done.resolve};animations.push(a);return a;}};}
 const overview={x:140,y:80,width:80,height:220},zoomed={x:80,y:80,width:820,height:700};
 const preview=element('preview',overview),panel=element('panel',zoomed),hero=element('hero',{x:90,y:100,width:16,height:16});
 panel.cloneNode=()=>({animate:panel.animate,style:fakeStyle(),className:'stage-expansion',removeAttribute(){},querySelectorAll(){return [];},setAttribute(){},remove(){}});
 const sequence=element('sequence',{x:80,y:40,width:100,height:380});
 const stage={querySelector:selector=>selector==='.stage-content'?preview:panel};
 const titles=Array.from({length:6},(_,i)=>element('title'+i,{x:30+i*50,y:10,width:40,height:20}));
 const root=element('root',{x:20,y:40,width:900,height:360});root.ownerDocument=doc;root.parentElement=new Events();
 root.querySelectorAll=()=>titles;
 root.querySelector=selector=>selector.startsWith('[data-stage=')?stage:['#patch-sequence','.token-rail'].includes(selector)?sequence:hero;
 doc.body={append(){}};
 if(native)doc.startViewTransition=callback=>{
  const ready=deferred(),finished=deferred();const tr={callback,ready:ready.promise,finished:finished.promise,skipped:false,
   capture(){callback();ready.resolve();},end(){finished.resolve();},skipTransition(){this.skipped=true;ready.reject(Error('intentionally skipped'));finished.resolve();}};
  transitions.push(tr);return tr;
 };
 const motion=createArchitectureMotion(root,()=>expanded);
 return {motion,root,html,win,doc,media,transitions,animations,titles,preview,panel,
   get expanded(){return expanded;},get mutations(){return mutations;},
   run(to){motion.run(to,()=>{expanded=to;mutations++;});}
 };
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));

test('zoom direction and timing distinguish opening, closing, and switching',()=>{
 assert.equal(zoomDirection(null,1),'in');assert.equal(zoomDirection(1,null),'out');assert.equal(zoomDirection(1,3),'switch');
 assert.deepEqual(ZOOM_TIMING,{in:1050,out:850,switch:750});
 assert.ok(ZOOM_TIMING.in>ZOOM_TIMING.out&&ZOOM_TIMING.out>ZOOM_TIMING.switch);
});
test('FLIP transform maps a target box back to its source',()=>{
 assert.equal(boxTransform({x:10,y:20,width:20,height:30},{x:100,y:60,width:200,height:150}),'translate(-90px, -40px) scale(0.1, 0.2)');
 assert.equal(boxTransform({x:0,y:0,width:0,height:5},{x:0,y:0,width:10,height:10}),null);
 assert.equal(boxTransform(null,{}),null);assert.equal(boxTransform({x:NaN},{}),null);
});
test('native snapshots name the same focus and patch before and after the update',async()=>{
 const h=harness();h.run(1);assert.equal(h.expanded,null);assert.equal(h.root.dataset.zoomPhase,'in');
 assert.equal(h.preview.style.viewTransitionName,'encoder-focus');assert.equal(h.root.dataset.zoomEngine,'view-transition');
 h.transitions[0].capture();assert.equal(h.expanded,1);assert.equal(h.preview.style.viewTransitionName,'');assert.equal(h.panel.style.viewTransitionName,'encoder-focus');
 h.transitions[0].end();await tick();assert.equal(h.motion.active,false);assert.deepEqual(h.html.dataset,{});assert.deepEqual(h.root.dataset,{});
 assert.equal(h.panel.style.viewTransitionName,'');assert.ok(h.titles.every(el=>el.style.viewTransitionName===''));
 h.motion.dispose();
});
test('rapid close commits a pending open exactly once and stale callbacks cannot reopen it',async()=>{
 const h=harness();h.run(1);h.run(null);assert.equal(h.expanded,1);assert.ok(h.transitions[0].skipped);
 h.transitions[1].capture();assert.equal(h.expanded,null);assert.equal(h.mutations,2);
 h.transitions[0].callback();assert.equal(h.expanded,null);assert.equal(h.mutations,2);
 h.transitions[1].end();await tick();assert.equal(h.motion.active,false);assert.deepEqual(h.root.dataset,{});
 h.motion.dispose();
});
test('a superseded transition cannot clear the active transition names',async()=>{
 const h=harness();h.run(1);h.transitions[0].capture();h.run(3);h.transitions[1].capture();await tick();
 assert.equal(h.root.dataset.zoomPhase,'switch');assert.equal(h.panel.style.viewTransitionName,'encoder-focus');
 h.transitions[1].end();await tick();assert.equal(h.panel.style.viewTransitionName,'');h.motion.dispose();
});
test('reduced motion and background tabs update immediately without animation',()=>{
 for(const options of [{reduced:true},{hidden:true}]){const h=harness(options);h.run(1);assert.equal(h.expanded,1);assert.equal(h.transitions.length,0);assert.equal(h.motion.active,false);h.motion.dispose();}
});
test('changing reduced-motion preference mid-zoom settles and cleans up',()=>{
 const h=harness();h.run(1);h.media.matches=true;h.media.emit('change');assert.equal(h.expanded,1);assert.equal(h.motion.active,false);assert.ok(h.transitions[0].skipped);h.run(null);assert.equal(h.transitions.length,1);h.motion.dispose();
});
test('resize, keyboard input, and controls safely finish an in-flight transition',()=>{
 for(const trigger of ['resize','keydown','input','click']){const h=harness();h.run(2);if(trigger==='resize')h.win.emit(trigger);else h.root.parentElement.emit(trigger);assert.equal(h.expanded,2);assert.equal(h.motion.active,false);assert.equal(h.root.style.viewTransitionName,'');h.motion.dispose();}
});
test('fallback performs a geometric zoom and cleans up animations',async()=>{
 const h=harness({native:false});h.run(1);assert.equal(h.expanded,1);assert.equal(h.root.dataset.zoomEngine,'flip');assert.ok(h.animations.length>0);
 assert.ok(h.animations.some(a=>a.frames[0].transform?.includes('scale(')));
 h.animations.forEach(a=>a.end());await tick();assert.equal(h.motion.active,false);assert.ok(h.animations.every(a=>a.cancelled));h.motion.dispose();
});
test('fallback collapse animates removed detail without delaying the state change',async()=>{
 const h=harness({native:false});h.run(1);h.motion.finish();h.run(null);assert.equal(h.expanded,null);assert.equal(h.root.dataset.zoomPhase,'out');
 assert.ok(h.animations.some(a=>a.frames[0].transform==='none'&&a.frames[1].opacity===0));h.motion.finish();await tick();assert.equal(h.motion.active,false);h.motion.dispose();
});
test('browsers without either animation API still retain a usable explainer',()=>{
 const h=harness({native:false,animate:false});h.run(1);assert.equal(h.expanded,1);assert.equal(h.motion.active,false);assert.deepEqual(h.root.dataset,{});h.motion.dispose();
});
test('native API failure updates exactly once and removes temporary names',()=>{
 const h=harness();h.doc.startViewTransition=()=>{throw Error('snapshot unavailable');};h.run(4);assert.equal(h.expanded,4);assert.equal(h.mutations,1);assert.equal(h.motion.active,false);assert.equal(h.root.style.viewTransitionName,'');h.motion.dispose();
});
test('ordinary same-stage updates do not replay the zoom',()=>{
 const h=harness();h.run(1);h.transitions[0].capture();h.motion.finish();h.run(1);assert.equal(h.transitions.length,1);assert.equal(h.mutations,2);assert.equal(h.motion.active,false);h.motion.dispose();
});
