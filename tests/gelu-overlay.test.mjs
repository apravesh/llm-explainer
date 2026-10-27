import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matrix,FULL_CONFIGS} from '../src/full-config.mjs';
import {geluFull,activateFull} from '../src/full-engine.mjs';
import {projectorGeluData,geluCurveMarkup,geluScalarMarkup,animateGeluTrace,createGeluOverlay} from '../src/gelu-overlay.mjs';
const original=x=>.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x*x*x)));
function fixture(){const input=matrix(3,5120,Float32Array.from({length:3*5120},(_,i)=>Math.sin(i*.123)*3));return {config:FULL_CONFIGS.pixtral,patchCount:3,adapterPre:input,adapterHidden:activateFull(input)};}
test('the shared GELU helper preserves every existing FP32 activation exactly',()=>{
 const a=matrix(11,1024,Float32Array.from({length:11*1024},(_,i)=>Math.sin(i*.51)*10));const result=activateFull(a);
 assert.equal(result.rows,11);assert.equal(result.cols,1024);for(let i=0;i<a.data.length;i++){assert.equal(geluFull(a.data[i]),original(a.data[i]));assert.equal(result.data[i],Math.fround(original(a.data[i])));}
 const gate=matrix(11,1024,new Float32Array(11*1024).fill(.3)),gated=activateFull(a,gate);for(let i=0;i<a.data.length;i++)assert.equal(gated.data[i],Math.fround(a.data[i]/(1+Math.exp(-a.data[i]))*gate.data[i]));
});
test('GELU suppresses negatives smoothly, is not ReLU, and approaches identity for positives',()=>{
 assert.equal(geluFull(0),0);assert.ok(geluFull(-1)<0&&geluFull(-1)>-1);assert.ok(Math.abs(geluFull(-4))<.001);assert.ok(Math.abs(geluFull(4)-4)<.001);assert.ok(geluFull(.5)>0&&geluFull(.5)<.5);
 for(const x of [-3,-2,-1,-.1,.1,1,2,3]){assert.ok(geluFull(x)/x>=0&&geluFull(x)/x<=1);assert.ok(Math.abs(geluFull(x)-geluFull(-x)-x)<1e-12);}
});
test('the inspector reads real before/after cells for any patch and all 5120 features',()=>{
 const run=fixture(),before=structuredClone(run);
 for(const row of [0,1,2])for(const feature of [0,31,1023,4095,5119]){const d=projectorGeluData(run,row,feature);assert.strictEqual(d.input,run.adapterPre);assert.strictEqual(d.output,run.adapterHidden);assert.equal(d.x,run.adapterPre.data[row*5120+feature]);assert.equal(d.y,Math.fround(geluFull(d.x)));assert.equal(d.expected,geluFull(d.x));assert.ok(d.range.start<=feature&&d.range.end>feature);}
 assert.deepEqual(run,before);assert.equal(projectorGeluData(run,-20,99999).row,0);assert.equal(projectorGeluData(run,-20,99999).feature,5119);assert.equal(projectorGeluData(run,999,NaN).row,2);assert.equal(projectorGeluData(run,0,0).gate,.5);
});
test('no activation is fabricated for a pending projector or classification path',()=>{
 assert.equal(projectorGeluData(null),null);assert.equal(projectorGeluData({config:FULL_CONFIGS.pixtral}),null);assert.equal(projectorGeluData({...fixture(),config:FULL_CONFIGS.vit}),null);
});
test('curve and numeric explanation use the same approximation and distinguish sandbox values',()=>{
 const d=projectorGeluData(fixture(),1,63),actual=geluScalarMarkup(d),preview=geluScalarMarkup(d,-1,true);
 assert.match(actual,/Actual p1 · feature 63/);assert.match(actual,/Stored FP32 output/);assert.match(preview,/Sandbox value · matrices unchanged/);assert.match(preview,/-0.158808/);assert.match(preview,/does not hard-clip/);assert.doesNotMatch(preview,/Stored FP32/);
 for(const x of [-11,-2,0,.5,8]){const svg=geluCurveMarkup(x);assert.match(svg,new RegExp(`maps to ${geluFull(x).toFixed(4)}`));assert.match(svg,/pathLength="1"/);assert.doesNotMatch(svg,/NaN|undefined|Infinity/);}
});
test('trace animation is finite, cancellable, and respects reduced motion and hidden tabs',()=>{
 const animations=[],root={querySelector:()=>({animate:(frames,options)=>{const a={frames,options,cancelled:false,cancel(){this.cancelled=true;}};animations.push(a);return a;}})};
 const a=animateGeluTrace(root);assert.equal(animations.length,2);assert.ok(animations.every(a=>a.options.duration===950));a.dispose();assert.ok(animations.every(a=>a.cancelled));animateGeluTrace(root,{reduced:true});animateGeluTrace(root,{hidden:true});assert.equal(animations.length,2);
});
class Events{constructor(){this.listeners=new Map();}addEventListener(t,fn){if(!this.listeners.has(t))this.listeners.set(t,new Set());this.listeners.get(t).add(fn);}removeEventListener(t,fn){this.listeners.get(t)?.delete(fn);}emit(t,e={}){for(const fn of this.listeners.get(t)||[])fn(e);}}
function harness({reduced=false}={}){
 const run=fixture(),state={selected:1,projectorFeature:63},animations=[],media=Object.assign(new Events(),{matches:reduced});let dialog,closedCount=0;
 const win=Object.assign(new Events(),{matchMedia:()=>media}),doc=Object.assign(new Events(),{defaultView:win,hidden:false});
 const dataKey=k=>k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase());
 function element(tag='div',attrs={}){
  const e=Object.assign(new Events(),{tagName:tag.toUpperCase(),attrs:{...attrs},dataset:{},children:[],open:false,isConnected:true,scrollTop:0,value:attrs.value??'',textContent:''});let html='',classes=new Set((attrs.class||'').split(' '));
  for(const [k,v] of Object.entries(attrs))if(k.startsWith('data-'))e.dataset[dataKey(k)]=v;
  e.classList={add:k=>classes.add(k),remove:k=>classes.delete(k),contains:k=>classes.has(k)};
  e.setAttribute=(k,v)=>e.attrs[k]=String(v);e.getAttribute=k=>e.attrs[k]??null;e.hasAttribute=k=>Object.hasOwn(e.attrs,k);
  e.focus=()=>doc.activeElement=e;e.matches=selector=>selector.split(',').some(part=>{const s=part.trim();if(s==='button')return e.tagName==='BUTTON';if(s[0]==='.')return classes.has(s.slice(1));const keys=[...s.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)];return keys.length>0&&keys.every(([,k,v])=>e.hasAttribute(k)&&(v===undefined||e.getAttribute(k)===v));});
  e.closest=s=>e.matches(s)?e:null;e.querySelectorAll=s=>e.children.flatMap(c=>[c,...c.querySelectorAll('*')]).filter(c=>s==='*'||c.matches(s));e.querySelector=s=>e.querySelectorAll(s)[0]||null;
  Object.defineProperty(e,'innerHTML',{get:()=>html,set:value=>{html=value;e.children=[...value.matchAll(/<([a-z][\w-]*)\b([^>]*)>/g)].map(([,tag,a])=>element(tag,Object.fromEntries([...a.matchAll(/([\w-]+)(?:="([^"]*)")?/g)].map(([,k,v])=>[k,v??'']))));}});
  e.showModal=()=>e.open=true;e.close=()=>{e.open=false;e.emit('close');};e.remove=()=>e.isConnected=false;e.getBoundingClientRect=()=>({left:0,top:0,right:100,bottom:100});
  e.animate=(frames,options)=>{const a={cancelled:false,cancel(){this.cancelled=true;}};animations.push(a);return a;};return e;
 }
 doc.documentElement=element('html');doc.createElement=tag=>element(tag);doc.body={append:e=>dialog=e};const origin=element('button',{'data-full-gelu':''});doc.querySelector=()=>origin;
 const renderer={markup:(a,{label})=>`<canvas data-test-matrix="${label}"></canvas>`,reset(){},draw(){},pick:()=>false,hover(){}};
 const controller=createGeluOverlay({document:doc,getRun:()=>run,getState:()=>state,onSelect:(row,feature)=>{state.selected=row;state.projectorFeature=feature;},onClose:()=>closedCount++,renderer});
 const find=s=>dialog.querySelector(s),click=s=>{const target=find(s);assert.ok(target,s);dialog.emit('click',{target});};
 const number=(selector,value,type='change')=>{const target=find(selector);target.value=String(value);dialog.emit(type,{target});};
 return {controller,run,state,doc,dialog,media,win,origin,animations,find,click,number,get closedCount(){return closedCount;},open:()=>controller.open(origin)};
}
test('opening creates a modal with both matrices, selected feature, trace and focus trap entry',()=>{
 const h=harness();h.open();assert.ok(h.dialog.open);assert.equal(h.find('[data-gelu-feature]').value,'63');assert.equal(h.dialog.dataset.feature,63);assert.equal(h.doc.activeElement,h.find('[data-gelu-close]'));assert.ok(h.doc.documentElement.classList.contains('gelu-open'));assert.equal(h.dialog.querySelectorAll('[data-test-matrix]').length,2);assert.equal(h.animations.length,2);h.controller.dispose();
});
test('playground and reset never modify the projector tensors or selected feature',()=>{
 const h=harness(),before=structuredClone(h.run);h.open();h.click('[data-gelu-example="-1"]');assert.equal(h.dialog.dataset.preview,'true');assert.match(h.find('[data-gelu-scalar]').innerHTML,/Sandbox value/);assert.match(h.find('[data-gelu-scalar]').innerHTML,/-0.158808/);assert.equal(h.state.projectorFeature,63);
 h.number('[data-gelu-slider]',2,'input');assert.equal(h.find('[data-gelu-sandbox-value]').textContent,'2.000');h.click('[data-gelu-restore]');assert.equal(h.dialog.dataset.preview,'false');assert.match(h.find('[data-gelu-scalar]').innerHTML,/Actual p1 · feature 63/);assert.deepEqual(h.run,before);h.controller.dispose();
});
test('patch and feature controls clamp, read actual values and restore keyboard focus',()=>{
 const h=harness();h.open();h.number('[data-gelu-feature]',5119);assert.equal(h.state.projectorFeature,5119);assert.equal(h.doc.activeElement,h.find('[data-gelu-feature]'));
 h.number('[data-gelu-patch]',99);assert.equal(h.state.selected,2);assert.equal(h.dialog.dataset.row,2);h.number('[data-gelu-feature]',-10);assert.equal(h.state.projectorFeature,0);h.number('[data-gelu-feature]','');assert.equal(h.state.projectorFeature,0);assert.match(h.find('[data-gelu-scalar]').innerHTML,/Actual p2 · feature 0/);h.controller.dispose();
});
test('closing settles animations, unlocks scroll, and returns focus to GELU even after rerender',()=>{
 const h=harness();h.open();h.origin.isConnected=false;h.click('[data-gelu-close]');assert.ok(!h.dialog.open);assert.ok(h.animations.every(a=>a.cancelled));assert.equal(h.closedCount,1);assert.equal(h.doc.activeElement,h.origin);assert.ok(!h.doc.documentElement.classList.contains('gelu-open'));h.controller.dispose();
});
test('visibility, reduced motion, replay and disposal have no lingering animations or listeners',()=>{
 const h=harness();h.open();h.doc.hidden=true;h.doc.emit('visibilitychange');assert.ok(h.animations.every(a=>a.cancelled));h.click('[data-gelu-replay]');assert.equal(h.animations.length,2);h.doc.hidden=false;h.click('[data-gelu-replay]');assert.equal(h.animations.length,4);h.media.matches=true;h.media.emit('change');assert.ok(h.animations.every(a=>a.cancelled));h.click('[data-gelu-replay]');assert.equal(h.animations.length,4);h.controller.dispose();assert.ok([...h.doc.listeners.values(),...h.media.listeners.values(),...h.win.listeners.values()].every(s=>s.size===0));assert.ok(!h.dialog.isConnected);h.controller.open(h.origin);assert.ok(!h.dialog.open);
 const r=harness({reduced:true});r.open();assert.equal(r.animations.length,0);assert.match(r.find('[data-gelu-scalar]').innerHTML,/Stored FP32 output/);r.controller.dispose();
});
test('projector button and modal ship in the active and offline apps and recompute closes it',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');const [views,app,bundle]=await Promise.all(['src/full-views.mjs','src/full-app.mjs','standalone.html'].map(read));
 assert.match(views,/data-full-gelu aria-haspopup="dialog"/);assert.match(app,/geluOverlay.open\(b\)/);assert.match(app,/if\(geluOverlay.isOpen\)geluOverlay.close\(\)/);assert.match(bundle,/function createGeluOverlay/);assert.match(bundle,/function geluFull/);assert.match(bundle,/Playground only/);assert.match(bundle,/GELU paper/);
});
