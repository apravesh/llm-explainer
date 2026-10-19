import {test} from 'node:test';
import assert from 'node:assert/strict';
import {patchify,runEncoder,rng} from '../src/engine.mjs';
import {createMultiplicationOverlay} from '../src/multiplication.mjs';
/** Minimal DOM/canvas adapter: exercises real controller logic without browser dependencies. */
function harness({reduced=false}={}){
 const random=rng(20),patches=patchify(Uint8ClampedArray.from({length:32*32*4},()=>random()*256),32,32,16),result=runEncoder(patches,2,'vit'),s={model:'vit',selected:1,patch:16,rows:2,cols:2,layer:0};
 const callbacks=new Map();let next=0,now=0,dialog=null;
 const eventTarget=()=>{const listeners=new Map();return {addEventListener(k,f){if(!listeners.has(k))listeners.set(k,new Set());listeners.get(k).add(f);},removeEventListener(k,f){listeners.get(k)?.delete(f);},emit(k,e={}){for(const f of listeners.get(k)||[])f(e);}};};
 const media=Object.assign(eventTarget(),{matches:reduced});
 const win=Object.assign(eventTarget(),{matchMedia:()=>media,requestAnimationFrame:f=>{callbacks.set(++next,f);return next;},cancelAnimationFrame:id=>callbacks.delete(id)});
 const doc=Object.assign(eventTarget(),{defaultView:win,hidden:false});
 function element(tag='div',attrs={}){
  let html='',children=[],classes=new Set((attrs.class||'').split(' ').filter(Boolean));
  const e=Object.assign(eventTarget(),{tagName:tag.toUpperCase(),attrs:{...attrs},dataset:{},style:{},open:false,isConnected:true,scrollTop:0,scrollLeft:0,width:0,height:0,textContent:'',value:0});
  const dataKey=k=>k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase());
  for(const [k,v] of Object.entries(attrs))if(k.startsWith('data-'))e.dataset[dataKey(k)]=v;
  Object.defineProperty(e,'id',{get:()=>e.attrs.id,set:v=>e.attrs.id=v});
  Object.defineProperty(e,'className',{get:()=>[...classes].join(' '),set:v=>classes=new Set(v.split(' ').filter(Boolean))});
  e.classList={add(...a){a.forEach(x=>classes.add(x));},remove(...a){a.forEach(x=>classes.delete(x));},toggle(k,v){if(v)classes.add(k);else classes.delete(k);},contains:k=>classes.has(k)};
  e.setAttribute=(k,v)=>{e.attrs[k]=String(v);};e.getAttribute=k=>e.attrs[k]??null;e.hasAttribute=k=>Object.hasOwn(e.attrs,k);
  e.focus=()=>{doc.activeElement=e;};e.closest=selector=>matches(e,selector)?e:null;
  e.getBoundingClientRect=()=>({x:0,y:0,left:0,top:0,right:600,bottom:300,width:600,height:300});
  const context=new Proxy({},{get:()=>()=>{}});e.getContext=()=>context;
  function matches(node,selector){
   if(selector==='button')return node.tagName==='BUTTON';
   if(selector.startsWith('#'))return node.id===selector.slice(1);
   if(selector.startsWith('.'))return node.classList.contains(selector.slice(1));
   const attributes=[...selector.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)];
   return attributes.length>0&&attributes.every(([,k,v])=>node.hasAttribute(k)&&(v===undefined||node.getAttribute(k)===v));
  }
  e.querySelectorAll=selector=>children.filter(node=>matches(node,selector));e.querySelector=selector=>e.querySelectorAll(selector)[0]||null;
  Object.defineProperty(e,'innerHTML',{get:()=>html,set:value=>{html=value;children=[...value.matchAll(/<([a-z][\w-]*)\b([^>]*)>/g)].map(([,tag,attributes])=>{const attrs=Object.fromEntries([...attributes.matchAll(/([\w-]+)(?:="([^"]*)")?/g)].map(([,key,v])=>[key,v??'']));return element(tag,attrs);});}});
  e.showModal=()=>e.open=true;e.close=()=>{e.open=false;e.emit('close');};e.remove=()=>{e.isConnected=false;};
  return e;
 }
 doc.documentElement=element('html');doc.body={append(e){dialog=e;}};doc.createElement=tag=>element(tag);const origin=element('button',{id:'mlp-trigger'});doc.querySelector=()=>origin;
 const controller=createMultiplicationOverlay({document:doc,getData:()=>({s,patches,result,imageURL:''}),onPatch:i=>s.selected=i});
 const find=selector=>dialog.querySelector(selector);
 function click(selector){const target=find(selector);assert.ok(target,selector);dialog.emit('click',{target,clientX:0,clientY:0});}
 function advance(ticks=1){for(let i=0;i<ticks;i++){now+=240;const pending=[...callbacks.values()];callbacks.clear();pending.forEach(f=>f(now));}}
 return {controller,s,result,patches,dialog,doc,origin,media,callbacks,find,click,advance,open:()=>controller.open('mlp-up',origin)};
}
test('opening autoplay ends at exact result; no timer remains after completion',()=>{
 const h=harness();h.open();assert.ok(h.dialog.open);assert.ok(h.callbacks.size);h.advance(30);
 assert.equal(h.dialog.dataset.count,24);assert.equal(h.dialog.dataset.complete,true);assert.equal(h.find('[data-mm-sum]').dataset.value,h.find('[data-mm-target]').dataset.value);assert.equal(h.callbacks.size,0);h.controller.dispose();
});
test('pause and manual stepping do not leave an animation frame behind',()=>{
 const h=harness();h.open();h.advance(4);h.click('[data-mm-play]');const count=h.dialog.dataset.count;h.advance(20);assert.equal(h.dialog.dataset.count,count);assert.equal(h.callbacks.size,0);h.click('[data-mm-step]');assert.equal(h.dialog.dataset.count,count+1);h.click('[data-mm-reset]');assert.equal(h.dialog.dataset.count,0);assert.equal(h.find('[data-mm-sum]').dataset.value,0);h.controller.dispose();
});
test('choosing a new feature cancels autoplay and resets its partial sum',()=>{
 const h=harness();h.open();h.advance(4);h.click('[data-mm-output="3"]');assert.equal(h.dialog.dataset.column,3);assert.equal(h.dialog.dataset.count,0);assert.equal(h.callbacks.size,0);assert.equal(h.find('[data-mm-play]').textContent,'Play dot product');h.click('[data-mm-step]');assert.equal(h.find('[data-mm-sum]').dataset.value,h.result.layers[0].ffnInput[2][0]*h.result.layers[0].weights.up[0][3]);h.controller.dispose();
});
test('switching token resets the trace and reads that token, not the previous one',()=>{
 const h=harness();h.open();h.advance(4);h.click('[data-mm-patch="0"]');assert.equal(h.s.selected,0);assert.equal(h.dialog.dataset.count,0);assert.equal(h.callbacks.size,0);h.click('[data-mm-step]');assert.equal(h.find('[data-mm-sum]').dataset.value,h.result.layers[0].ffnInput[1][0]*h.result.layers[0].weights.up[0][0]);h.controller.dispose();
});
test('switching operation cancels the old animation and computes contraction correctly',()=>{
 const h=harness();h.open();h.advance(3);h.click('[data-mm-operation="mlp-down"]');assert.equal(h.callbacks.size,1);h.advance(55);assert.equal(h.dialog.dataset.count,48);assert.equal(h.find('[data-mm-sum]').dataset.value,h.result.layers[0].mlpDelta[2][0]);h.controller.dispose();
});
test('sweep computes every output and leaves state on the last visible column',()=>{
 const h=harness();h.open();h.click('[data-mm-sweep]');h.advance(55);assert.equal(h.dialog.dataset.column,47);assert.equal(h.dialog.dataset.count,24);assert.equal(h.find('[data-mm-sum]').dataset.value,h.result.layers[0].preActivation[2][47]);assert.equal(h.callbacks.size,0);h.controller.dispose();
});
test('close cancels animation, unlocks page scrolling and restores focus',()=>{
 const h=harness();h.open();h.advance(3);h.controller.close();assert.equal(h.callbacks.size,0);assert.equal(h.dialog.open,false);assert.equal(h.dialog.innerHTML,'');assert.equal(h.doc.documentElement.classList.contains('multiplication-open'),false);assert.strictEqual(h.doc.activeElement,h.origin);h.advance(50);assert.equal(h.dialog.innerHTML,'');h.controller.dispose();
});
test('reduced motion uses no animation but retains exact results and manual steps',()=>{
 const h=harness({reduced:true});h.open();assert.equal(h.callbacks.size,0);assert.equal(h.dialog.dataset.count,24);h.click('[data-mm-reset]');h.click('[data-mm-step]');assert.equal(h.dialog.dataset.count,1);h.click('[data-mm-sweep]');assert.equal(h.dialog.dataset.column,47);assert.equal(h.dialog.dataset.complete,true);assert.equal(h.callbacks.size,0);h.controller.dispose();
});
test('background tabs pause and do not resume unexpectedly',()=>{
 const h=harness();h.open();h.advance(3);h.doc.hidden=true;h.doc.emit('visibilitychange');assert.equal(h.callbacks.size,0);const count=h.dialog.dataset.count;h.advance(50);assert.equal(h.dialog.dataset.count,count);h.doc.hidden=false;h.doc.emit('visibilitychange');assert.equal(h.callbacks.size,0);h.controller.dispose();
});

test('clicking a full output row switches the patch and uses its exact product',()=>{
 const h=harness();h.open();h.click('[data-mm-result-row="4"]');assert.equal(h.s.selected,3);assert.equal(h.dialog.dataset.row,4);assert.equal(h.dialog.dataset.count,0);h.click('[data-mm-step]');assert.equal(h.find('[data-mm-sum]').dataset.value,h.result.layers[0].ffnInput[4][0]*h.result.layers[0].weights.up[0][0]);assert.equal(h.callbacks.size,0);h.controller.dispose();
});
test('arrow keys on the output matrix navigate token rows and feature columns',()=>{
 const h=harness();h.open();const press=(key,row)=>h.dialog.emit('keydown',{target:h.find(`[data-mm-result-row="${row}"]`),key,preventDefault(){}});
 press('ArrowRight',2);assert.equal(h.dialog.dataset.column,1);press('ArrowUp',2);assert.equal(h.s.selected,0);assert.equal(h.dialog.dataset.row,1);press('Home',1);assert.equal(h.s.selected,-1);assert.equal(h.dialog.dataset.row,0);press('End',0);assert.equal(h.s.selected,3);assert.equal(h.dialog.dataset.row,4);assert.equal(h.callbacks.size,0);h.controller.dispose();
});
