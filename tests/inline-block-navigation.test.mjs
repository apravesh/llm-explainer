import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createInlineBlockNavigation,inlineBlockControlsMarkup} from '../src/block-navigation.mjs';
class Events{
 constructor(){this.events=new Map();}
 addEventListener(type,fn){if(!this.events.has(type))this.events.set(type,new Set());this.events.get(type).add(fn);}
 removeEventListener(type,fn){this.events.get(type)?.delete(fn);}
 emit(type,event={}){for(const fn of this.events.get(type)||[])fn(event);}
}
function harness({depth=24,reduced=false,expanded=null,ready=depth}={}){
 const state={layer:0,head:5,selected:21,feature:1023,expanded},run={config:{depth},layers:Array.from({length:ready},()=>({}))};
 const preference=Object.assign(new Events(),{matches:reduced}),win=Object.assign(new Events(),{matchMedia:()=>preference}),doc=Object.assign(new Events(),{defaultView:win,hidden:false,visibilityState:'visible'});
 let created=0;doc.createElement=()=>{created++;throw Error('Inline controls must not create a new toolbar');};
 const messages=[],selections=[],animations=[],root=Object.assign(new Events(),{ownerDocument:doc,dataset:{},removeAttribute(name){if(name==='data-block-transition')delete this.dataset.blockTransition;}});
 const expandedMarker={};
 const buttons=[-1,1].map(delta=>({disabled:false,dataset:{inlineBlockStep:String(delta)},closest(selector){return selector==='[data-inline-block-step]'?this:selector==='.stage-expansion'&&state.expanded!==null?expandedMarker:null;},focus(){doc.activeElement=this;}}));
 const target={animate(frames,options){let resolve;const a={frames,options,cancelled:false,finished:new Promise(r=>resolve=r),cancel(){this.cancelled=true;resolve();},finish(){resolve();}};animations.push(a);return a;}};
 root.contains=b=>buttons.includes(b);
 root.querySelectorAll=selector=>selector.includes('[data-inline-block-step]')?buttons:selector==='.encoder-group .stage-content'?[target,target]:[];
 root.querySelector=selector=>selector==='.stage.is-expanded .expansion-inspection'?target:null;
 const controller=createInlineBlockNavigation({root,getRun:()=>run,getState:()=>state,beforeSelect(){},onSelect(index){state.layer=index;selections.push(index);},onAnnounce:m=>messages.push(m)});
 return {controller,state,run,buttons,root,doc,win,preference,messages,selections,animations,get created(){return created;},click(delta){root.emit('click',{target:buttons[delta===-1?0:1]});},key(key,delta=1){let prevented=false;root.emit('keydown',{target:buttons[delta===-1?0:1],key,preventDefault(){prevented=true;}});return prevented;}};
}
const tick=()=>new Promise(r=>setImmediate(r));
test('inline navigation does not create DOM controls or change the layout',()=>{
 const h=harness();assert.equal(h.created,0);assert.equal(h.buttons[0].disabled,true);assert.equal(h.buttons[1].disabled,false);h.controller.dispose();
});
for(const depth of [24,12])test(`${depth} blocks: retained arrows update one block and preserve patch, head and feature`,async()=>{
 const h=harness({depth});h.click(1);assert.equal(h.state.layer,1);assert.equal(h.state.selected,21);assert.equal(h.state.head,5);assert.equal(h.state.feature,1023);assert.equal(h.state.expanded,null);assert.equal(h.animations.length,2);assert.equal(h.root.dataset.blockTransition,'forward');assert.strictEqual(h.doc.activeElement,h.buttons[1]);
 h.animations.forEach(a=>a.finish());await tick();assert.equal(h.root.dataset.blockTransition,undefined);h.click(-1);assert.equal(h.state.layer,0);assert.equal(h.buttons[0].disabled,true);assert.strictEqual(h.doc.activeElement,h.buttons[1]);h.controller.seek(depth-1);assert.equal(h.buttons[1].disabled,true);assert.match(h.messages.at(-1),new RegExp(`block ${depth} of ${depth}`));h.controller.dispose();
});
test('arrows remain functional within the expanded attention or MLP heading',()=>{
 for(const expanded of [3,4]){const h=harness({expanded});h.click(1);assert.equal(h.state.expanded,expanded);assert.equal(h.animations.length,1);assert.equal(h.state.layer,1);assert.strictEqual(h.doc.activeElement,h.buttons[1]);h.controller.dispose();}
});
test('keyboard arrows and Home/End reach all blocks without a slider',()=>{
 const h=harness();assert.ok(h.key('End'));assert.equal(h.state.layer,23);assert.ok(h.key('Home',-1));assert.equal(h.state.layer,0);assert.ok(h.key('ArrowRight'));assert.equal(h.state.layer,1);assert.equal(h.key('Tab'),false);h.controller.dispose();
});
test('uncomputed neighbor is disabled and cannot be selected',()=>{
 const h=harness({ready:1});assert.ok(h.buttons[1].disabled);h.click(1);h.controller.seek(5);assert.equal(h.state.layer,0);assert.deepEqual(h.selections,[]);h.run.layers.push({});h.controller.sync();assert.equal(h.buttons[1].disabled,false);h.click(1);assert.equal(h.state.layer,1);h.controller.dispose();
});
test('reduced motion navigates immediately with no animations',()=>{
 const h=harness({reduced:true});h.click(1);assert.equal(h.state.layer,1);assert.equal(h.animations.length,0);assert.equal(h.root.dataset.blockTransition,undefined);h.controller.dispose();
});
test('rapid navigation cancels old transitions and does not lose the latest block',async()=>{
 const h=harness();h.click(1);const first=[...h.animations];h.click(1);assert.equal(h.state.layer,2);assert.ok(first.every(a=>a.cancelled));await tick();assert.equal(h.root.dataset.blockTransition,'forward');h.animations.slice(2).forEach(a=>a.finish());await tick();assert.equal(h.root.dataset.blockTransition,undefined);h.controller.dispose();
});
test('resize, hidden tab, preference changes, and disposal clean up transitions',()=>{
 for(const event of ['resize','visibilitychange','preference','dispose']){
  const h=harness();h.click(1);
  if(event==='resize')h.win.emit('resize');else if(event==='visibilitychange'){h.doc.hidden=true;h.doc.emit('visibilitychange');}else if(event==='preference'){h.preference.matches=true;h.preference.emit('change');}else h.controller.dispose();
  assert.ok(h.animations.every(a=>a.cancelled));assert.equal(h.root.dataset.blockTransition,undefined);h.controller.dispose();
  assert.ok([...h.root.events.values()].every(set=>set.size===0));assert.ok([...h.win.events.values()].every(set=>set.size===0));
 }
});
test('expanded heading markup contains just the two contextual arrows and block count',()=>{
 const html=inlineBlockControlsMarkup(5,24);assert.equal((html.match(/<button /g)||[]).length,2);assert.match(html,/Block 6 \/ 24/);assert.match(html,/aria-controls="architecture"/);assert.doesNotMatch(html,/range|ENCODER DEPTH|data-block-index|data-block-play/);
});
