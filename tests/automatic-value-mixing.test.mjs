import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matrix} from '../src/full-config.mjs';
import {VALUE_MIXING_TIMING,valueMixingFrame,mountAutomaticValueMixing} from '../src/automatic-value-mixing.mjs';
class Events{
 constructor(){this.listeners=new Map();}
 addEventListener(type,fn){if(!this.listeners.has(type))this.listeners.set(type,new Set());this.listeners.get(type).add(fn);}
 removeEventListener(type,fn){this.listeners.get(type)?.delete(fn);}
 emit(type){for(const fn of this.listeners.get(type)||[])fn();}
}
function harness({reduced=false,hidden=false,feature=1}={}){
 const pending=new Map(),all=[];let nextId=0,now=0;
 const media=Object.assign(new Events(),{matches:reduced}),win={matchMedia:()=>media,requestAnimationFrame(fn){pending.set(++nextId,fn);all.push(fn);return nextId;},cancelAnimationFrame(id){pending.delete(id);}};
 const doc=Object.assign(new Events(),{defaultView:win,hidden});
 const classes=new Set(),bench={classList:{add:c=>classes.add(c),remove:c=>classes.delete(c)}},box={textContent:'',dataset:{}},label={textContent:''};
 const root={ownerDocument:doc,isConnected:true,dataset:{},querySelector:s=>s==='.sa-workbench'?bench:s==='#full-value-mix'?box:label};
 // Head 2 uses columns 2..3, proving that the loop does not accidentally use head 1 values.
 const data={c:{width:4,headWidth:2},query:0,n:3,headIndex:1,l:{values:matrix(3,4,new Float32Array([99,99,1,2,99,99,3,4,99,99,5,6]))},head:{attention:matrix(3,3,new Float32Array([.25,.25,.5,0,0,0,0,0,0])),out:matrix(3,2,new Float32Array([3.5,4.5,0,0,0,0]))}};
 const original=structuredClone(data),controller=mountAutomaticValueMixing(root,{data,feature});
 function advance(ms=80){now+=ms;const frames=[...pending.values()];pending.clear();frames.forEach(fn=>fn(now));}
 return {controller,root,box,label,classes,pending,all,media,doc,data,original,advance};
}
test('loop reaches every key, holds the complete sum, and restarts at zero',()=>{
 const {perKey,hold}=VALUE_MIXING_TIMING,total=48*perKey;
 assert.deepEqual(valueMixingFrame(0,48),{cycle:0,count:0,complete:false});
 assert.equal(valueMixingFrame(perKey,48).count,1);
 assert.deepEqual(valueMixingFrame(total,48),{cycle:0,count:48,complete:true});
 assert.equal(valueMixingFrame(total+hold-1,48).count,48);
 assert.deepEqual(valueMixingFrame(total+hold,48),{cycle:1,count:0,complete:false});
 assert.deepEqual(valueMixingFrame(total+hold+perKey,48),{cycle:1,count:1,complete:false});
});
test('mount starts automatically, computes the correct head/feature, and loops without a button',()=>{
 const h=harness();assert.equal(h.root.dataset.valueMixing,'playing');assert.ok(h.classes.has('is-mixing'));assert.equal(h.pending.size,1);assert.equal(h.box.dataset.count,0);
 h.advance(0);h.advance();assert.equal(h.box.dataset.count,1);assert.equal(h.box.dataset.sum,.5);
 h.advance();assert.equal(h.box.dataset.sum,1.5);h.advance();assert.equal(h.box.dataset.sum,4.5);assert.equal(h.box.dataset.expected,4.5);assert.match(h.box.textContent,/complete/);
 h.advance(VALUE_MIXING_TIMING.hold);assert.equal(h.box.dataset.count,0);assert.equal(h.box.dataset.cycle,1);assert.equal(h.pending.size,1);assert.deepEqual(h.data,h.original);h.controller.dispose();
});
test('the loop also handles feature zero correctly',()=>{
 const h=harness({feature:0});h.advance(0);h.advance(240);assert.equal(h.box.dataset.sum,3.5);assert.equal(h.box.dataset.expected,3.5);h.controller.dispose();
});
test('background tabs cancel animation and resume without counting hidden time',()=>{
 const h=harness();h.advance(0);h.advance(80);h.doc.hidden=true;h.doc.emit('visibilitychange');assert.equal(h.pending.size,0);assert.ok(!h.classes.has('is-mixing'));assert.equal(h.root.dataset.valueMixing,'paused');
 h.advance(100000);assert.equal(h.box.dataset.count,1);h.doc.hidden=false;h.doc.emit('visibilitychange');assert.equal(h.pending.size,1);assert.ok(h.classes.has('is-mixing'));h.advance(0);assert.equal(h.box.dataset.count,1);h.advance(80);assert.equal(h.box.dataset.count,2);h.controller.dispose();
});
test('initially hidden view never schedules work until visible',()=>{
 const h=harness({hidden:true});assert.equal(h.pending.size,0);assert.equal(h.root.dataset.valueMixing,'paused');h.doc.hidden=false;h.doc.emit('visibilitychange');assert.equal(h.pending.size,1);assert.equal(h.box.dataset.count,0);h.controller.dispose();
});
test('reduced motion shows complete output without scheduling any animation',()=>{
 const h=harness({reduced:true});assert.equal(h.pending.size,0);assert.ok(!h.classes.has('is-mixing'));assert.equal(h.root.dataset.valueMixing,'reduced-motion');assert.equal(h.box.dataset.count,3);assert.equal(h.box.dataset.sum,4.5);assert.match(h.box.textContent,/reduced motion/);h.controller.dispose();
});
test('changing motion preference stops the running loop and restarts if re-enabled',()=>{
 const h=harness();h.advance(0);h.advance(80);h.media.matches=true;h.media.emit('change');assert.equal(h.pending.size,0);assert.equal(h.box.dataset.count,3);h.media.matches=false;h.media.emit('change');assert.equal(h.pending.size,1);assert.equal(h.box.dataset.count,0);h.controller.dispose();
});
test('disposal cancels pending work, prevents stale updates, and removes listeners',()=>{
 const h=harness();h.advance(0);h.advance(80);const stale=h.all.at(-1);h.controller.dispose();const text=h.box.textContent;assert.equal(h.pending.size,0);assert.ok(!h.classes.has('is-mixing'));stale(99999);assert.equal(h.box.textContent,text);assert.equal(h.pending.size,0);assert.equal(h.root.dataset.valueMixing,undefined);
 assert.ok([...h.doc.listeners.values()].every(s=>s.size===0));assert.ok([...h.media.listeners.values()].every(s=>s.size===0));h.doc.emit('visibilitychange');assert.equal(h.pending.size,0);
});
test('detached attention views clean up instead of leaving a background loop',()=>{
 const h=harness();h.root.isConnected=false;h.advance(80);assert.equal(h.pending.size,0);assert.ok(!h.classes.has('is-mixing'));assert.ok([...h.doc.listeners.values()].every(s=>s.size===0));
});
test('head/patch re-render can replace a loop without sharing progress or callbacks',()=>{
 const previous=harness(),stale=previous.all.at(-1);previous.advance(0);previous.advance(160);previous.controller.dispose();const next=harness({feature:0});stale(99999);assert.equal(next.box.dataset.count,0);assert.equal(previous.pending.size,0);assert.equal(next.pending.size,1);next.advance(0);next.advance(240);assert.equal(next.box.dataset.sum,3.5);next.controller.dispose();
});
test('active page and offline bundle contain automatic mixing but no value-mixing button',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8'),[view,app,overlay,bundle]=await Promise.all(['src/simple-attention.mjs','src/full-app.mjs','src/full-overlay.mjs','standalone.html'].map(read));
 for(const source of [view,app,bundle])assert.doesNotMatch(source,/data-full-mix|Animate value mixing/);
 assert.match(view,/data-value-mixing-label/);assert.match(view,/id="full-value-mix" aria-live="off"/);assert.match(app,/blockNavigation\?\.sync\(\);mountPatchWindow\(\);syncMix\(\)/);assert.match(app,/onClose:syncMix/);assert.match(app,/about.addEventListener\('close',syncMix\)/);assert.match(app,/document.querySelector\('dialog\[open\]'\)/);assert.match(overlay,/onClose\(\)/);assert.match(bundle,/function mountAutomaticValueMixing/);
});
