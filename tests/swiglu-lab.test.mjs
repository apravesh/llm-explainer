import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {SWIGLU_LAB_INPUTS,SWIGLU_TRACE_MS,swigluLabWeights,swigluLabTrace,swigluTraceStage,swigluDiagramMarkup,swigluValuesMarkup,swigluLabMarkup,mountSwigluLab} from '../src/swiglu-lab.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-12);
test('SwiGLU has two independent 4×6 projections and one 6×4 contraction',()=>{
 const w=swigluLabWeights();assert.deepEqual(Object.keys(w),['w1','w3','w2']);assert.equal(w.w1.length,4);assert.equal(w.w3.length,4);assert.ok(w.w1.every(r=>r.length===6));assert.ok(w.w3.every(r=>r.length===6));assert.equal(w.w2.length,6);assert.ok(w.w2.every(r=>r.length===4));assert.notDeepEqual(w.w1,w.w3);assert.deepEqual(w,swigluLabWeights());
});
test('each branch receives the original x, gating is elementwise, and contraction includes all terms',()=>{
 for(const x of SWIGLU_LAB_INPUTS){const t=swigluLabTrace(x),before=[...x];
  for(let j=0;j<6;j++){let a=0,v=0;for(let i=0;i<4;i++){a+=x[i]*t.weights.w1[i][j];v+=x[i]*t.weights.w3[i][j];}near(t.pre[j],a);near(t.gate[j],a/(1+Math.exp(-a)));near(t.values[j],v);near(t.mixed[j],t.gate[j]*v);}
  for(let j=0;j<4;j++){let y=0;for(let k=0;k<6;k++)y+=t.mixed[k]*t.weights.w2[k][j];near(t.output[j],y);}
  assert.deepEqual(x,before);assert.equal(t.mixed.length,6);assert.equal(t.output.length,4);
 }
});
test('changing the toy patch changes activations but never the weights or previous traces',()=>{
 const a=swigluLabTrace(),before=structuredClone(a),b=swigluLabTrace(SWIGLU_LAB_INPUTS[1]);assert.deepEqual(a.weights,b.weights);assert.notDeepEqual(a.output,b.output);assert.deepEqual(a,before);assert.deepEqual(swigluLabTrace([0,0,0,0]).output,[0,0,0,0]);
 for(const input of [[1,2],[1,2,3,NaN],[1,2,3,Infinity]])assert.throws(()=>swigluLabTrace(input));
});
test('desktop and narrow diagrams keep both branches, their product, and the output visible without numeric dumps',()=>{
 for(const compact of [false,true]){const html=swigluDiagramMarkup(swigluLabTrace(),compact);assert.equal((html.match(/data-sg-node=/g)||[]).length,6);assert.equal((html.match(/data-sg-edge="branches"/g)||[]).length,2);assert.equal((html.match(/data-sg-edge="mix"/g)||[]).length,2);assert.match(html,/data-sg-node="output"/);assert.match(html,/W₁ → SiLU/);assert.match(html,/g ⊙ v/);assert.match(html,/Parallel expansion/);assert.match(html,/1 × 4/);assert.match(html,/1 × 6/);assert.doesNotMatch(html,/NaN|undefined/);}
});
test('values disclosure is closed by default and preserves all exact feature products',()=>{
 const html=swigluLabMarkup(),t=swigluLabTrace(),values=swigluValuesMarkup(t);assert.match(html,/<details class="fa-values-disclosure sg-values"><summary>Show actual values<\/summary>/);assert.equal((values.match(/data-sg-product=/g)||[]).length,6);
 for(let i=0;i<6;i++)assert.ok(values.includes(`data-sg-product="${i}">${t.mixed[i].toFixed(5)}`));
 assert.match(html,/not a slice of the model/);assert.match(html,/not probabilities or binary switches/);assert.match(html,/before adding the residual/);assert.match(html,/1,024 → 4,096/);assert.match(html,/GELU\(xW₁\)W₂/);
});
test('animation stages put both expansions in the same phase and finish once',()=>{
 assert.equal(SWIGLU_TRACE_MS,2800);for(const [ms,stage] of [[-1,0],[0,0],[699,0],[700,1],[1400,2],[2100,3],[2800,4],[9999,4]])assert.equal(swigluTraceStage(ms),stage);
});
class Events{constructor(){this.handlers=new Map();}addEventListener(t,fn){if(!this.handlers.has(t))this.handlers.set(t,new Set());this.handlers.get(t).add(fn);}removeEventListener(t,fn){this.handlers.get(t)?.delete(fn);}emit(t,e={}){for(const fn of this.handlers.get(t)||[])fn(e);}}
function harness(reduced=false){
 let now=0,id=0,observer;const jobs=new Map(),media=Object.assign(new Events(),{matches:reduced}),win={performance:{now:()=>now},matchMedia:()=>media,requestAnimationFrame:fn=>{jobs.set(++id,fn);return id;},cancelAnimationFrame:id=>jobs.delete(id),IntersectionObserver:class{constructor(fn){this.fn=fn;observer=this;this.disconnected=false;}observe(target){this.target=target;}disconnect(){this.disconnected=true;}emit(visible){this.fn([{isIntersecting:visible,intersectionRatio:visible?1:0}]);}}};
 const doc=Object.assign(new Events(),{defaultView:win,hidden:false});
 function element(tag='div',attrs={}){
  const el=Object.assign(new Events(),{tag,attrs:{...attrs},dataset:{},children:[],open:false,isConnected:true,textContent:''});let html='';const classes=new Set((attrs.class||'').split(' '));
  for(const [k,v] of Object.entries(attrs))if(k.startsWith('data-'))el.dataset[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=v;
  el.hasAttribute=k=>Object.hasOwn(el.attrs,k);el.setAttribute=(k,v)=>el.attrs[k]=String(v);el.classList={contains:c=>classes.has(c),toggle(c,on){if(on)classes.add(c);else classes.delete(c);}};
  el.matches=s=>s==='button'?tag==='button':s[0]==='.'?classes.has(s.slice(1)):[...s.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)].some(([,k,v])=>el.hasAttribute(k)&&(v===undefined||el.attrs[k]===v));
  el.closest=s=>el.matches(s)?el:null;el.querySelectorAll=s=>el.children.flatMap(n=>[n,...n.querySelectorAll('*')]).filter(n=>s==='*'||n.matches(s));el.querySelector=s=>el.querySelectorAll(s)[0]||null;el.contains=n=>el===n||el.querySelectorAll('*').includes(n);
  Object.defineProperty(el,'innerHTML',{get:()=>html,set:value=>{html=value;el.children=[...value.matchAll(/<([a-z][\w-]*)\b([^>]*)>/g)].map(([,tag,rest])=>element(tag,Object.fromEntries([...rest.matchAll(/([\w-]+)(?:="([^"]*)")?/g)].map(([,k,v])=>[k,v??'']))));}});return el;
 }
 const root=element();root.ownerDocument=doc;const controller=mountSwigluLab(root),find=s=>root.querySelector(s),click=s=>root.emit('click',{target:find(s)}),advance=ms=>{now+=ms;const tasks=[...jobs.values()];jobs.clear();tasks.forEach(fn=>fn(now));};return {root,controller,find,click,advance,observer,jobs,media,doc};
}
test('entering view traces both branches together, once, without hiding any vectors',()=>{
 const h=harness();assert.equal(h.jobs.size,0);h.observer.emit(true);assert.equal(h.root.dataset.stage,0);h.advance(700);
 assert.ok(h.find('[data-sg-node="gate"]').classList.contains('is-active'));assert.ok(h.find('[data-sg-node="values"]').classList.contains('is-active'));assert.equal(h.root.querySelectorAll('[data-sg-node]').length,12);
 h.advance(700);assert.equal(h.root.dataset.stage,2);h.advance(700);assert.equal(h.root.dataset.stage,3);h.advance(700);assert.equal(h.root.dataset.stage,4);assert.equal(h.jobs.size,0);h.observer.emit(false);h.observer.emit(true);assert.equal(h.jobs.size,0);h.controller.dispose();
});
test('changing presets preserves disclosure state and uses the same actual computations',()=>{
 const h=harness();const disclosure=h.find('.sg-values');assert.equal(disclosure.open,false);disclosure.open=true;h.observer.emit(true);h.click('[data-sg-preset="1"]');assert.strictEqual(h.find('.sg-values'),disclosure);assert.equal(disclosure.open,true);assert.equal(h.root.dataset.preset,1);assert.equal(h.find('[data-sg-values]').innerHTML,swigluValuesMarkup(swigluLabTrace(SWIGLU_LAB_INPUTS[1])));assert.equal(h.jobs.size,1);h.controller.dispose();
});
test('offscreen and hidden pauses resume; replay cancels stale frames; disposal removes observers',()=>{
 const h=harness();h.observer.emit(true);h.advance(800);h.observer.emit(false);assert.equal(h.jobs.size,0);h.advance(5000);h.observer.emit(true);h.advance(700);assert.equal(h.root.dataset.stage,2);
 h.doc.hidden=true;h.doc.emit('visibilitychange');assert.equal(h.jobs.size,0);h.doc.hidden=false;h.doc.emit('visibilitychange');assert.equal(h.jobs.size,1);h.click('[data-sg-replay]');assert.equal(h.jobs.size,1);assert.equal(h.root.dataset.stage,0);h.controller.dispose();assert.equal(h.jobs.size,0);assert.ok(h.observer.disconnected);assert.ok([...h.root.handlers.values(),...h.doc.handlers.values(),...h.media.handlers.values()].every(s=>s.size===0));
});
test('reduced motion displays the complete flow without running a trace',()=>{
 const h=harness(true);h.observer.emit(true);assert.equal(h.root.dataset.stage,4);assert.equal(h.jobs.size,0);h.click('[data-sg-preset="2"]');assert.equal(h.root.dataset.stage,4);assert.equal(h.jobs.size,0);h.controller.dispose();
 const active=harness();active.observer.emit(true);active.advance(700);active.media.matches=true;active.media.emit('change');assert.equal(active.root.dataset.stage,4);assert.equal(active.jobs.size,0);active.controller.dispose();
});
test('standalone SwiGLU section is removed while RoPE, comparison, and main MLP remain',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
 for(const file of ['vision.html','standalone.html']){
  const html=await read(file);
  assert.doesNotMatch(html,/id="swiglu-lab"|id="swiglu-demo"|SwiGLU: a second/);
  assert.ok(html.indexOf('id="rope-lab"')<html.indexOf('id="comparison"'));
  assert.match(html,/<\/section>\s*<section id="comparison"/);
  assert.match(html,/class="stage mlp-stage"/);
 }
 const app=await read('src/full-app.mjs'),bundle=await read('standalone.html');
 assert.doesNotMatch(app,/mountSwigluLab|swiglu-demo/);assert.doesNotMatch(bundle,/function mountSwigluLab|SWIGLU_LAB_INPUTS/);
 assert.match(app,/mountRopeLab\(\$\('#rope-demo'\)\)/);
 assert.match(bundle,/function mlpProcessMarkup/);assert.match(bundle,/function mlpFlowMarkup/);assert.match(bundle,/function mountMlpCalculation/);
});
