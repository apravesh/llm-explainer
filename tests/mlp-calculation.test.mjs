import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {FULL_CONFIGS,matrix,weightDescriptor,matrixValue} from '../src/full-config.mjs';
import {activateFull,addFull} from '../src/full-engine.mjs';
import {createMatrixRenderer} from '../src/full-matrix.mjs';
import {createFullViews} from '../src/full-views.mjs';
import {MLP_CALCULATION_MS,mlpCalculationFrame,mlpCalculationData,mlpCalculationTraces,mlpActivationTrace,mlpCalculationMarkup,mlpCalculationNumbersMarkup,mountMlpCalculation} from '../src/mlp-calculation.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-5*(1+Math.abs(b)),`${a} != ${b}`);
function fixture(model='pixtral',real=false){
 const c=real?FULL_CONFIGS[model]:{...FULL_CONFIGS[model],width:8,ffWidth:12},make=(cols,seed)=>matrix(3,cols,Float32Array.from({length:3*cols},(_,i)=>Math.sin((i+seed)*.12)));
 const weights={up:weightDescriptor(c.width,c.ffWidth,11),down:weightDescriptor(c.ffWidth,c.width,23),...(c.gated?{gate:weightDescriptor(c.width,c.ffWidth,37)}:{})};
 const multiply=(a,w)=>{if(real)return make(w.cols,w.seed);const out=matrix(a.rows,w.cols);for(let r=0;r<a.rows;r++)for(let j=0;j<w.cols;j++){let sum=0;for(let k=0;k<a.cols;k++)sum+=matrixValue(a,r,k)*matrixValue(w,k,j);out.data[r*w.cols+j]=sum;}return out;};
 const input=make(c.width,3),ffnInput=make(c.width,7),preActivation=multiply(ffnInput,weights.up),gate=c.gated?multiply(ffnInput,weights.gate):null,expanded=activateFull(preActivation,gate),mlpDelta=multiply(expanded,weights.down),afterAttention=input,output=addFull(afterAttention,mlpDelta),l={input,ffnInput,preActivation,gate,expanded,mlpDelta,afterAttention,output,weights};
 return {run:{config:c,patchCount:c.cls?2:3,layers:[l,l]},state:{selected:1,layer:0,feature:c.width-1,wideFeature:c.ffWidth-1,mlpCalculation:true,expanded:4},l,c};
}
for(const model of ['pixtral','vit']){
 test(`${model}: all linear operations retain their true dependencies, complete outputs and selected row`,()=>{
  const {run,state,l,c}=fixture(model),before=structuredClone(run),data=mlpCalculationData(run,state);assert.deepEqual(data.ops.map(o=>o.key),c.gated?['up','gate','down']:['up','down']);assert.equal(data.row,c.cls?2:1);
  assert.strictEqual(data.ops[0].operation.input,l.ffnInput);assert.strictEqual(data.ops[0].operation.output,l.preActivation);
  if(c.gated){assert.strictEqual(data.ops[1].operation.input,l.ffnInput);assert.strictEqual(data.ops[1].operation.output,l.gate);}
  assert.strictEqual(data.ops.at(-1).operation.input,l.expanded);assert.strictEqual(data.ops.at(-1).operation.output,l.mlpDelta);assert.deepEqual(run,before);
 });
 test(`${model}: every dot-product prefix sums all input terms and agrees with stored FP32 output`,()=>{
  const {run,state}=fixture(model),data=mlpCalculationData(run,state),traces=mlpCalculationTraces(data);
  for(const {key,operation:op,feature} of data.ops){let sum=0;assert.equal(traces[key].prefix.length,op.input.cols+1);assert.equal(traces[key].prefix[0],0);for(let k=0;k<op.input.cols;k++){sum+=matrixValue(op.input,data.row,k)*matrixValue(op.weights,k,feature);assert.equal(traces[key].prefix[k+1],sum);}near(traces[key].total,traces[key].expected);}
  const activation=mlpActivationTrace(data);near(activation.computed,activation.expected);if(!data.c.gated)assert.equal(activation.b,null);
 });
 test(`${model}: one inline view includes every multiplication, full matrix dimensions, product and residual`,()=>{
  const {run,state,c,l}=fixture(model,true),seen=[],renderer={markup(m,opts){seen.push({m,opts});return `<canvas data-full-matrix="${seen.length}"></canvas>`;}},html=mlpCalculationMarkup(renderer,run,state);
  assert.equal((html.match(/data-mlp-operation=/g)||[]).length,c.gated?3:2);assert.doesNotMatch(html,/<dialog|showModal|data-full-op="mlp-/);
  assert.equal(seen.length,c.gated?11:8);for(const key of ['up',...(c.gated?['gate']:[]),'down']){const w=l.weights[key];assert.ok(seen.some(s=>s.m===w));}
  assert.ok(seen.some(s=>s.m===l.mlpDelta));assert.ok(seen.some(s=>s.m===l.output));assert.ok(seen.some(s=>s.m===l.afterAttention));assert.match(html,/output matrices always show their complete cached values/);
  assert.match(html,c.gated?/C = SiLU\(A\) ⊙ B/:/C = GELU\(A\)/);if(!c.gated)assert.doesNotMatch(html,/W₃|SiLU/);
  for(const {opts} of seen)assert.equal(typeof opts.click,'function');
 });
 test(`${model}: values and traces stay in the single closed-by-default disclosure`,()=>{
  const {run,state,c}=fixture(model),renderer=createMatrixRenderer('mi-test'),views=createFullViews(renderer,{select(){},selectMlpFeature(){}}),html=views.content(run,state);
  const disclosure=html.match(/<details class="fa-values-disclosure" data-mlp-values([^>]*)>([\s\S]*?)<\/details>/);assert.ok(disclosure);assert.doesNotMatch(disclosure[1],/open/);assert.match(disclosure[2],/Show actual values/);assert.equal((disclosure[2].match(/data-mlp-term=/g)||[]).length,c.gated?3:2);assert.doesNotMatch(html.replace(disclosure[0],''),/data-mlp-term=|class="fm-features"/);
  assert.match(views.content(run,{...state,mlpValuesOpen:true}),/data-mlp-values open/);
 });
}
test('CLS remains row zero and feature coordinates clamp at real model boundaries',()=>{
 const {run,state,c}=fixture('vit',true),data=mlpCalculationData(run,{...state,selected:-1,feature:999999,wideFeature:999999});assert.equal(data.row,0);assert.equal(data.feature,c.width-1);assert.equal(data.wide,c.ffWidth-1);assert.equal(mlpCalculationData({...run,layers:[]},state),null);
});
test('both branches animate together and contraction begins only after activation; timing is fixed',()=>{
 assert.equal(MLP_CALCULATION_MS,5500);for(const [time,phase,up,down] of [[0,'branches',0,0],[1200,'branches',512,0],[2400,'activation',1024,0],[3099,'activation',1024,0],[3100,'contraction',1024,0],[4300,'contraction',1024,2048],[5500,'complete',1024,4096],[9000,'complete',1024,4096]]){const f=mlpCalculationFrame(time,1024,4096);assert.equal(f.phase,phase);assert.equal(f.up,up);assert.equal(f.gate,up);assert.equal(f.down,down);}
});
class Events{constructor(){this.listeners=new Map();}addEventListener(t,fn){if(!this.listeners.has(t))this.listeners.set(t,new Set());this.listeners.get(t).add(fn);}removeEventListener(t,fn){this.listeners.get(t)?.delete(fn);}emit(t,e={}){for(const fn of this.listeners.get(t)||[])fn(e);}}
function harness({reduced=false,autoplay=true,hidden=false}={}){
 let now=0,seq=0;const pending=new Map(),allFrames=[],highlights=[],animations=[],selections=[],media=Object.assign(new Events(),{matches:reduced}),win=Object.assign(new Events(),{performance:{now:()=>now},matchMedia:()=>media,requestAnimationFrame:fn=>{pending.set(++seq,fn);allFrames.push(fn);return seq;},cancelAnimationFrame:id=>pending.delete(id)}),doc=Object.assign(new Events(),{defaultView:win,hidden,querySelector:()=>null});
 function el(attrs={}){const n=Object.assign(new Events(),{dataset:{},value:'',textContent:'',isConnected:true});let classes=new Set((attrs.class||'').split(' '));for(const [key,v] of Object.entries(attrs))if(key.startsWith('data-'))n.dataset[key.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=v;
  n.hasAttribute=k=>Object.hasOwn(attrs,k);n.matches=s=>s==='button'?attrs.tag==='button':s[0]==='.'?classes.has(s.slice(1)):[...s.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)].some(([,k,v])=>n.hasAttribute(k)&&(v===undefined||attrs[k]===v));n.closest=s=>n.matches(s)?n:null;n.classList={toggle(k,on){if(on)classes.add(k);else classes.delete(k);},remove:k=>classes.delete(k)};return n;}
 const {run,state}=fixture(),root=el(),host=el(),nodes=[];root.ownerDocument=doc;root.closest=()=>host;
 const add=attrs=>{const node=el(attrs);nodes.push(node);return node;};
 for(const key of ['up','gate','down']){
  for(const part of ['input','weights','output']){const n=add({'data-mlp-matrix':`${key}-${part}`}),canvas=el({'data-full-matrix':`${key}-${part}`});n.querySelector=()=>canvas;}
  add({'data-mlp-operation':key});for(const id of ['count','term','term-value','sum','expected','rounding'])add({[`data-mlp-${id}`]:key});
 }
 add({'data-mlp-status':''});add({'data-mlp-play':'',tag:'button'});add({'data-mlp-activation':''});add({'data-mlp-feature':'wide'});add({'data-mlp-feature':'feature'});
 root.querySelectorAll=host.querySelectorAll=s=>nodes.filter(n=>n.matches(s));root.querySelector=host.querySelector=s=>nodes.find(n=>n.matches(s))||null;
 root.animate=()=>{const a={cancelled:false,cancel(){this.cancelled=true;}};animations.push(a);return a;};
 const controller=mountMlpCalculation(root,{run,state,renderer:{highlight(...args){highlights.push(args.slice(1));}},onFeature:(...args)=>selections.push(args),autoplay});
 return {root,host,doc,win,media,controller,pending,allFrames,highlights,animations,selections,run,state,find:s=>host.querySelector(s),click(s){const target=host.querySelector(s);host.emit('click',{target});},input(s,value){const target=host.querySelector(s);target.value=String(value);host.emit('input',{target});},change(s,value){const target=host.querySelector(s);target.value=String(value);host.emit('change',{target});},advance(ms){now+=ms;const jobs=[...pending.values()];pending.clear();jobs.forEach(fn=>fn(now));}};
}
test('one playback traces parallel products, activation, contraction, and stops without modifying tensors',()=>{
 const h=harness(),before=structuredClone(h.run);assert.equal(h.root.dataset.phase,'branches');h.advance(1200);assert.equal(h.root.dataset.upCount,4);assert.equal(h.root.dataset.gateCount,4);assert.equal(h.root.dataset.downCount,0);h.advance(1200);assert.equal(h.root.dataset.phase,'activation');h.advance(700);assert.equal(h.root.dataset.phase,'contraction');h.advance(1200);assert.equal(h.root.dataset.downCount,6);h.advance(1200);assert.equal(h.root.dataset.phase,'complete');assert.equal(h.pending.size,0);assert.deepEqual(h.run,before);h.controller.dispose();
});
test('pause/resume and replay retain the prefix and do not accumulate animations',()=>{
 const h=harness();h.advance(1200);h.click('[data-mlp-play]');assert.equal(h.pending.size,0);h.advance(8000);assert.equal(h.root.dataset.upCount,4);h.click('[data-mlp-play]');h.advance(4300);assert.equal(h.root.dataset.phase,'complete');assert.equal(h.pending.size,0);h.click('[data-mlp-play]');assert.equal(h.root.dataset.upCount,0);assert.equal(h.pending.size,1);h.controller.dispose();
});
test('manual term scrub reveals the exact prefix and pauses the full animation',()=>{
 const h=harness(),trace=mlpCalculationTraces(mlpCalculationData(h.run,h.state));h.input('[data-mlp-term="gate"]',3);assert.equal(h.pending.size,0);assert.equal(h.root.dataset.phase,'inspection');assert.equal(h.find('[data-mlp-sum="gate"]').dataset.value,trace.gate.prefix[3]);h.win.emit('resize');assert.equal(h.root.dataset.phase,'inspection');assert.equal(h.find('[data-mlp-sum="gate"]').dataset.value,trace.gate.prefix[3]);h.input('[data-mlp-term="gate"]',0);assert.equal(h.find('[data-mlp-sum="gate"]').dataset.value,0);assert.match(h.find('[data-mlp-term-value="gate"]').textContent,/No terms/);h.controller.dispose();
});
test('numeric feature changes stop playback and report an update without mutating state inside controller',()=>{
 const h=harness();h.change('[data-mlp-feature="wide"]',5);assert.deepEqual(h.selections,[['wide',5]]);assert.equal(h.pending.size,0);h.change('[data-mlp-feature="feature"]','');assert.equal(h.selections.length,1);h.controller.dispose();
});
test('selection rerenders show completed results without replaying unless requested',()=>{
 const h=harness({autoplay:false});assert.equal(h.root.dataset.phase,'complete');assert.equal(h.pending.size,0);assert.equal(h.animations.length,0);h.controller.play();assert.equal(h.pending.size,1);h.controller.dispose();
});
test('reduced motion finishes immediately; hidden tabs and modal dialogs pause; disposal cleans up',()=>{
 const r=harness({reduced:true});assert.equal(r.root.dataset.phase,'complete');assert.equal(r.pending.size,0);assert.equal(r.animations.length,0);r.controller.dispose();
 const h=harness();h.advance(100);h.doc.hidden=true;h.doc.emit('visibilitychange');assert.equal(h.pending.size,0);h.doc.hidden=false;h.controller.play();h.doc.querySelector=()=>({open:true});h.advance(100);assert.equal(h.pending.size,0);h.doc.querySelector=()=>null;h.controller.play();h.media.matches=true;h.media.emit('change');assert.equal(h.root.dataset.phase,'complete');assert.equal(h.pending.size,0);h.controller.dispose();assert.ok(h.animations.every(a=>a.cancelled));assert.ok([...h.host.listeners.values(),...h.doc.listeners.values(),...h.media.listeners.values(),...h.win.listeners.values()].every(s=>s.size===0));
});
test('detached views cancel even a stale callback and cannot schedule additional frames',()=>{
 const h=harness(),stale=h.allFrames.at(-1);h.root.isConnected=false;h.advance(100);assert.equal(h.pending.size,0);stale(8000);assert.equal(h.pending.size,0);h.controller.play();assert.equal(h.pending.size,0);
});
test('every MLP trigger routes to one inline calculation before generic popup dispatch',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8'),app=await read('src/full-app.mjs'),bundle=await read('standalone.html');
 assert.match(app,/if\(\['mlp-up','mlp-gate','mlp-down'\].includes\(b.dataset.fullOp\)\)\{openMlpCalculation\(\);return;\}/);
 assert.ok(app.indexOf("includes(b.dataset.fullOp)")<app.indexOf('overlay.open(b.dataset.fullOp,b)'));
 assert.match(app,/if\(state.expanded===4&&state.mlpCalculation\)closeMlpCalculation\(\)/);assert.match(app,/mlpCalculationController\?\.dispose\(\)/);assert.match(app,/mountMlpPanel\(\)/);assert.match(bundle,/function mountMlpCalculation/);assert.match(bundle,/All three MLP multiplications/);assert.match(bundle,/Show actual values/);
});
