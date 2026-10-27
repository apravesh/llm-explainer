import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {ROPE_LAB_Q,ROPE_LAB_K,ROPE_LAB_POSITIONS,ROPE_LAB_DURATION,ROPE_LAB_HOLD_MS,createRopeLabLoop,ropeLabPosition,ropeLabTurns,ropeTurnsMarkup,ropeWrapExplanationMarkup,ropeLabRotate,ropeLabTrace,ropeLabShift,ropeLabPairMarkup,ropePairTurnExplanation,ropeLabSumMarkup,ropeLabMatrixMarkup,ropeLabMarkup,ropeSingleCircleMarkup,ropeSingleValuesMarkup,ropeLabOverlayState,ropeMathOverlayMarkup,ropeComparisonOverlayMarkup,createRopeLabPlayback,mountRopeLab} from '../src/rope-lab.mjs';
const near=(a,b,epsilon=1e-12)=>assert.ok(Math.abs(a-b)<epsilon,`${a} != ${b}`);
const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
test('the teaching head has exactly two visible pairs, with explicit starting feature values',()=>{
 assert.deepEqual(ROPE_LAB_Q,[1,0,1,0]);assert.deepEqual(ROPE_LAB_K,[.8,.6,.6,.8]);const t=ropeLabTrace({progress:0});assert.equal(t.pairs.length,2);assert.deepEqual(t.products,[.8,0,.6,0]);near(t.total,1.4);
 assert.deepEqual(t.pairs[0].qRotated,[1,0]);assert.deepEqual(t.pairs[0].kRotated,[.8,.6]);assert.equal(t.pairs[0].axis,'row');assert.equal(t.pairs[1].axis,'column');
});
test('positive rotation turns counterclockwise, using the shown 2×2 matrix',()=>{
 const v=ropeLabRotate([1,0],Math.PI/2);near(v[0],0);near(v[1],1);
 for(const angle of [-3,-1,0,.2,Math.PI/2,5]){const [x,y]=ropeLabRotate([.8,.6],angle);near(x,Math.cos(angle)*.8-Math.sin(angle)*.6);near(y,Math.sin(angle)*.8+Math.cos(angle)*.6);near(x*x+y*y,1);}
});
test('default turns are explicit position × frequency: 30°/90° and 15°/30°',()=>{
 const t=ropeLabTrace();near(t.pairs[0].targetQ,Math.PI/6);near(t.pairs[0].targetK,Math.PI/2);near(t.pairs[1].targetQ,Math.PI/12);near(t.pairs[1].targetK,Math.PI/6);
 near(t.pairs[0].qRotated[0],Math.sqrt(3)/2);near(t.pairs[0].qRotated[1],.5);near(t.pairs[0].kRotated[0],-.6);near(t.pairs[0].kRotated[1],.8);near(t.pairs[0].dot,.4-.3*Math.sqrt(3));
});
test('the displayed score is exactly the sum of all four matching-coordinate products',()=>{
 for(const progress of [0,.1,.37,.5,.9,1]){const t=ropeLabTrace({progress});const q=t.pairs.flatMap(p=>p.qRotated),k=t.pairs.flatMap(p=>p.kRotated);near(t.total,dot(q,k));near(t.products.reduce((a,b)=>a+b),t.total);assert.equal(t.products.length,4);for(const p of t.pairs){near(Math.hypot(...p.qRotated),1);near(Math.hypot(...p.kRotated),1);}}
 const html=ropeLabSumMarkup(ropeLabTrace());assert.equal((html.match(/class="rl-product"/g)||[]).length,4);assert.equal((html.match(/Pair contribution/g)||[]).length,2);assert.doesNotMatch(html,/NaN|undefined/);
});
test('2D separates image axes without treating the feature plane as an image plane',()=>{
 const original=ropeLabTrace(),rows=ropeLabTrace({positions:{qr:2,qc:1,kr:4,kc:2}}),cols=ropeLabTrace({positions:{qr:1,qc:2,kr:3,kc:3}});
 assert.deepEqual(rows.pairs[1],original.pairs[1]);assert.deepEqual(cols.pairs[0],original.pairs[0]);assert.notDeepEqual(rows.pairs[0],original.pairs[0]);assert.notDeepEqual(cols.pairs[1],original.pairs[1]);
});
for(const mode of ['2d','1d'])for(const scale of [.25,1,2])test(`${mode}, scale ${scale}: RᵢᵀRⱼ equals Rⱼ₋ᵢ for each pair`,()=>{
 const t=ropeLabTrace({mode,scale});for(const p of t.pairs)near(p.dot,dot(p.q,ropeLabRotate(p.k,p.kAngle-p.qAngle)));
});
for(const mode of ['2d','1d'])test(`${mode}: joint translation preserves the dot product at EVERY animation frame`,()=>{
 const positions={...ROPE_LAB_POSITIONS},before={...positions},shift=ropeLabShift(positions,'both',mode);assert.deepEqual(positions,before);assert.deepEqual(shift.positions,{qr:2,qc:2,kr:4,kc:3});
 near(shift.before.total,shift.after.total);assert.equal(shift.before.deltaRow,shift.after.deltaRow);assert.equal(shift.before.deltaCol,shift.after.deltaCol);
 for(const progress of [0,.01,.25,.5,.75,.99,1]){const t=ropeLabTrace({positions:shift.positions,mode,progress,fromAngles:shift.fromAngles});near(t.total,shift.before.total);for(let i=0;i<2;i++)near(t.pairs[i].kAngle-t.pairs[i].qAngle,shift.before.pairs[i].kAngle-shift.before.pairs[i].qAngle);}
});
test('moving only K changes relative offsets and the default score without changing Q or base features',()=>{
 const shift=ropeLabShift(ROPE_LAB_POSITIONS,'key');assert.equal(shift.positions.qr,1);assert.equal(shift.positions.qc,1);assert.equal(shift.after.deltaRow,shift.before.deltaRow+1);assert.equal(shift.after.deltaCol,shift.before.deltaCol+1);assert.ok(Math.abs(shift.after.total-shift.before.total)>.1);
 assert.deepEqual(shift.before.pairs.map(p=>p.qRotated),shift.after.pairs.map(p=>p.qRotated));assert.deepEqual(shift.before.pairs.map(p=>p.k),shift.after.pairs.map(p=>p.k));
});
test('the translation experiment crosses the old grid boundary and only guards safe-integer overflow',()=>{
 assert.ok(ropeLabShift({qr:7,qc:1,kr:3,kc:2}));assert.ok(ropeLabShift({qr:1,qc:1,kr:7,kc:2},'key'));
 assert.equal(ropeLabShift({qr:Number.MAX_SAFE_INTEGER,qc:1,kr:3,kc:2}),null);
 assert.ok(ropeLabShift({qr:Number.MAX_SAFE_INTEGER,qc:1,kr:3,kc:2},'key'));
 let positions={...ROPE_LAB_POSITIONS},original=ropeLabTrace().total;for(let i=0;i<100;i++){const shift=ropeLabShift(positions);near(shift.after.total,original);positions=shift.positions;}assert.equal(positions.kr,103);assert.ok(ropeLabShift(positions));
});
test('1D comparison flattens the eight-column grid while keeping separate pair frequencies',()=>{
 const t=ropeLabTrace({mode:'1d'});assert.ok(t.pairs.every(p=>p.qPosition===9&&p.kPosition===26));near(t.pairs[0].frequency,2*t.pairs[1].frequency);assert.ok(t.pairs.every(p=>p.axis==='sequence'));
});
test('circle endpoints are the rotated feature values, with original dashed rays and text identities',()=>{
 for(const p of ropeLabTrace().pairs){const html=ropeLabPairMarkup(p);assert.equal((html.match(/class="rl-original/g)||[]).length,2);assert.match(html,/blue Q/);assert.match(html,/brown K/);assert.match(html,/not image coordinates/);assert.match(html,/Target turns from image positions/);
 for(const key of ['q','k']){const values=p[key+'Rotated'];assert.ok(html.includes(`x2="${176+83*values[0]}"`));assert.ok(html.includes(`y2="${127-83*values[1]}"`));assert.ok(html.includes(`data-angle="${p[key+'Angle']}"`));}}
});
test('R inspection gives actual numerical sine/cosine entries and component multiplications',()=>{
 const trace=ropeLabTrace();for(const pair of [0,1])for(const vector of ['q','k']){const html=ropeLabMatrixMarkup(trace,pair,vector);assert.match(html,/R\(θ\) is a 2 × 2 rotation matrix/);assert.match(html,/x′ = x cos θ − y sin θ/);assert.match(html,/y′ = x sin θ \+ y cos θ/);assert.match(html,/radians internally/);assert.doesNotMatch(html,/NaN|undefined/);}
 assert.match(ropeLabMatrixMarkup(trace,0,'q'),/0.8660/);assert.match(ropeLabMatrixMarkup(trace,0,'q'),/0.5000/);
});
function clock(){const jobs=new Map();let now=0,id=0;const values=[];let ended=0;const win={performance:{now:()=>now},requestAnimationFrame:fn=>{jobs.set(++id,fn);return id;},cancelAnimationFrame:i=>jobs.delete(i)};const controller=createRopeLabPlayback(win,p=>values.push(p),()=>ended++);return {win,controller,jobs,values,get ended(){return ended;},advance(ms){now+=ms;const pending=[...jobs.values()];jobs.clear();pending.forEach(fn=>fn(now));}};}
test('rotation playback reaches the target and leaves no animation frame after completion',()=>{
 const h=clock();h.controller.play();assert.equal(h.values[0],0);h.advance(ROPE_LAB_DURATION/2);near(h.values.at(-1),.5);h.advance(ROPE_LAB_DURATION/2);assert.equal(h.values.at(-1),1);assert.equal(h.ended,1);assert.equal(h.jobs.size,0);assert.equal(h.controller.playing,false);
});
test('pause, resume and rapid replay preserve progress without accumulating timers',()=>{
 const h=clock();h.controller.play();h.advance(650);near(h.controller.progress,.25);h.controller.pause();h.advance(5000);near(h.controller.progress,.25);assert.equal(h.jobs.size,0);h.controller.play(h.controller.progress);h.advance(650);near(h.controller.progress,.5);h.controller.play(0);assert.equal(h.jobs.size,1);h.advance(2600);assert.equal(h.ended,1);assert.equal(h.jobs.size,0);
});
test('manual/reduced-motion completion is immediate and disposal cancels current and future work',()=>{
 const h=clock();h.controller.play(0,{immediate:true});assert.deepEqual(h.values,[1]);assert.equal(h.ended,1);assert.equal(h.jobs.size,0);h.controller.play();h.controller.dispose();assert.equal(h.jobs.size,0);h.controller.play();assert.equal(h.jobs.size,0);
});
test('a reentrant initial frame cannot restart an animation that was cancelled by its callback',()=>{
 const h=clock();let controller;controller=createRopeLabPlayback(h.win,()=>controller.pause(),()=>{});controller.play();assert.equal(h.jobs.size,0);
});
test('the compact lab shows only one query patch and keeps derivation/comparison behind overlay buttons',()=>{
 const html=ropeLabMarkup();assert.match(html,/4 features/);assert.match(html,/not Pixtral’s frequency schedule/);assert.match(html,/real 64-feature heads/);assert.match(html,/not RGB values or image coordinates/);
 assert.match(html,/data-rl-single-circle/);assert.match(html,/data-rl-position="qr"/);assert.match(html,/data-rl-position="qc"/);assert.doesNotMatch(html,/data-rl-position="k[rc]"|data-rl-pairs|data-rl-sum|data-rl-matrix|data-rl-shift/);
 assert.match(html,/data-rl-overlay="math" aria-haspopup="dialog"/);assert.match(html,/data-rl-overlay="comparison" aria-haspopup="dialog"/);assert.match(html,/different feature pairs/);assert.match(html,/not two turns applied to the same vector/);
});
test('active and offline builds mount the new lab instead of the opaque eight-feature example',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');const [page,app,bundle]=await Promise.all(['vision.html','src/full-app.mjs','standalone.html'].map(read));assert.match(page,/id="rope-demo"/);assert.match(app,/mountRopeLab\(\$\('#rope-demo'\)\)/);assert.doesNotMatch(app,/function renderRope|const labQ=/);assert.match(bundle,/function mountRopeLab/);assert.match(bundle,/data-rl-progress/);assert.match(bundle,/No hidden pairs contribute/);assert.doesNotMatch(bundle,/Isolated 8D lab|separate 8D RoPE lab|first 2 of 4 pairs shown/);
 const source=await read('src/rope-lab.mjs');assert.match(source,/doc.addEventListener\('visibilitychange',visibility\)/);assert.match(source,/doc.removeEventListener\('visibilitychange',visibility\)/);assert.match(source,/media.removeEventListener\('change',reduced\)/);assert.match(source,/playback.dispose\(\)/);
});

test('the compact demo needs no long-page stepper, sticky controls, or scrolling anchor',()=>{
 const html=ropeLabMarkup();assert.doesNotMatch(html,/rl-stepper|data-rl-playback-anchor|rl-deeper|rl-score-section/);assert.match(html,/rl-loop-caption/);assert.doesNotMatch(html,/data-rl-play[ >]|data-rl-reset|data-rl-progress/);
});
test('one circle shows two Q pairs with their actual independent angles and endpoints',()=>{
 const trace=ropeLabTrace({positions:{qr:2,qc:1,kr:3,kc:2}}),html=ropeSingleCircleMarkup(trace);
 assert.equal((html.match(/<svg /g)||[]).length,1);assert.equal((html.match(/class="rl-unit-circle"/g)||[]).length,1);assert.equal((html.match(/data-rope-single-pair=/g)||[]).length,2);
 assert.match(html,/not consecutive turns of one vector/);assert.match(html,/blue row pair/);assert.match(html,/purple column pair/);assert.doesNotMatch(html,/data-rope-arrow="k/);
 for(const p of trace.pairs){assert.ok(html.includes(`data-angle="${p.qAngle}"`));assert.ok(html.includes(`x2="${176+83*p.qRotated[0]}"`));assert.ok(html.includes(`y2="${127-83*p.qRotated[1]}"`));}
 const values=ropeSingleValuesMarkup(trace);assert.match(values,/features 0, 1/);assert.match(values,/features 2, 3/);assert.match(values,/60° target/);assert.match(values,/15° target/);
});
test('overlays copy rather than mutate the main patch state; comparison starts before RoPE',()=>{
 const main={positions:{...ROPE_LAB_POSITIONS,qr:4},progress:.4},before=structuredClone(main),math=ropeLabOverlayState(main,'math'),comparison=ropeLabOverlayState(main,'comparison');
 assert.equal(math.progress,.4);assert.equal(comparison.progress,0);math.positions.qr=6;comparison.positions.kr=7;assert.deepEqual(main,before);assert.equal(comparison.positions.qr,4);
});
test('the math overlay combines R, numeric matrix multiplication and all four dot-product terms',()=>{
 const html=ropeMathOverlayMarkup(ropeLabOverlayState({positions:{...ROPE_LAB_POSITIONS},progress:1},'math'));
 assert.match(html,/R turns a pair into two new values/);assert.match(html,/data-rl-matrix/);assert.match(html,/Match features, multiply, and add/);assert.match(html,/data-rl-sum/);assert.match(html,/only here we introduce a key K/);assert.match(html,/No hidden pairs contribute/);assert.match(html,/not a probability/);assert.doesNotMatch(html,/data-rl-shift/);
});
test('the separate comparison overlay explains Q/K colors and keeps the relative identity optional',()=>{
 const html=ropeComparisonOverlayMarkup(ropeLabOverlayState({positions:{...ROPE_LAB_POSITIONS},progress:1},'comparison'));
 assert.match(html,/data-rl-pairs/);assert.match(html,/Blue = Q, not the row pair here/);assert.match(html,/Brown = K/);assert.match(html,/data-rl-shift="both"/);assert.match(html,/data-rl-shift="key"/);assert.match(html,/moving image content through a whole model/);assert.match(html,/<details class="rl-identity">/);
});
class RLEvents{constructor(){this.listeners=new Map();}addEventListener(type,fn){if(!this.listeners.has(type))this.listeners.set(type,new Set());this.listeners.get(type).add(fn);}removeEventListener(type,fn){this.listeners.get(type)?.delete(fn);}emit(type,e={}){for(const fn of this.listeners.get(type)||[])fn(e);}}
function labHarness({reduced=false}={}){
 let now=0,id=0,dialog,observer;const frames=new Map(),media=Object.assign(new RLEvents(),{matches:reduced}),win={performance:{now:()=>now},matchMedia:()=>media,requestAnimationFrame:fn=>{frames.set(++id,fn);return id;},cancelAnimationFrame:id=>frames.delete(id)};
 win.IntersectionObserver=class{constructor(callback,options){this.callback=callback;this.options=options;this.disconnected=false;observer=this;}observe(target){this.target=target;}disconnect(){this.disconnected=true;}emit(isIntersecting,intersectionRatio=isIntersecting?1:0){this.callback([{target:this.target,isIntersecting,intersectionRatio}]);}};
 const doc=Object.assign(new RLEvents(),{defaultView:win,hidden:false});const dataKey=k=>k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase());
 function element(tag='div',attrs={}){
  const e=Object.assign(new RLEvents(),{tagName:tag.toUpperCase(),attrs:{...attrs},dataset:{},children:[],open:false,isConnected:true,scrollTop:0,textContent:'',value:attrs.value??''});let html='',classes=new Set((attrs.class||'').split(' '));
  for(const [k,v] of Object.entries(attrs))if(k.startsWith('data-'))e.dataset[dataKey(k)]=v;
  e.setAttribute=(k,v)=>{e.attrs[k]=String(v);if(k.startsWith('data-'))e.dataset[dataKey(k)]=String(v);};e.getAttribute=k=>e.attrs[k]??null;e.hasAttribute=k=>Object.hasOwn(e.attrs,k);
  e.classList={add:k=>classes.add(k),remove:k=>classes.delete(k),contains:k=>classes.has(k)};e.focus=()=>doc.activeElement=e;
  e.matches=selector=>{if(selector==='button')return e.tagName==='BUTTON';if(selector[0]==='#')return e.attrs.id===selector.slice(1);if(selector[0]==='.')return classes.has(selector.slice(1));const tests=[...selector.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)];return tests.length>0&&tests.every(([,k,v])=>e.hasAttribute(k)&&(v===undefined||e.getAttribute(k)===v));};
  e.closest=s=>e.matches(s)?e:null;e.querySelectorAll=s=>e.children.flatMap(c=>[c,...c.querySelectorAll('*')]).filter(c=>s==='*'||c.matches(s));e.querySelector=s=>e.querySelectorAll(s)[0]||null;e.contains=n=>e===n||e.querySelectorAll('*').includes(n);
  Object.defineProperty(e,'innerHTML',{get:()=>html,set:value=>{html=value;e.children=[...value.matchAll(/<([a-z][\w-]*)\b([^>]*)>/g)].map(([,tag,a])=>element(tag,Object.fromEntries([...a.matchAll(/([\w-]+)(?:="([^"]*)")?/g)].map(([,k,v])=>[k,v??'']))));}});
  e.showModal=()=>e.open=true;e.close=()=>{e.open=false;e.emit('close');};e.remove=()=>e.isConnected=false;e.getBoundingClientRect=()=>({left:0,top:0,right:100,bottom:100});return e;
 }
 doc.documentElement=element('html');doc.createElement=tag=>element(tag);doc.body={append:e=>dialog=e};const root=element();root.ownerDocument=doc;
 const controller=mountRopeLab(root);
 const find=(sel,modal=false)=>(modal?dialog:root).querySelector(sel);
 const click=(sel,modal=false)=>{const target=find(sel,modal);assert.ok(target,sel);(modal?dialog:root).emit('click',{target});};
 const change=(sel,value,modal=false,type='change')=>{const target=find(sel,modal);target.value=String(value);(modal?dialog:root).emit(type,{target});};
 const advance=ms=>{now+=ms;const jobs=[...frames.values()];frames.clear();jobs.forEach(fn=>fn(now));};
 return {root,dialog,doc,media,frames,controller,find,click,change,advance,observer};
}
test('the mounted main lab loops one circle and two Q pairs without playback controls',()=>{
 const h=labHarness();assert.equal(h.root.querySelectorAll('[data-rope-single-pair]').length,2);assert.equal(h.root.querySelectorAll('.rl-unit-circle').length,1);
 for(const selector of ['[data-rl-play]','[data-rl-reset]','[data-rl-progress]'])assert.equal(h.find(selector),null);
 h.observer.emit(true);h.advance(1300);near(h.root.dataset.progress,.5);assert.equal(h.frames.size,1);h.advance(1300);assert.equal(h.root.dataset.progress,1);assert.equal(h.root.dataset.phase,'holding');assert.equal(h.frames.size,1);h.controller.dispose();
});
test('opening the math modal pauses main playback and preserves its current pose after close',()=>{
 const h=labHarness();h.observer.emit(true);h.advance(650);const before=h.root.dataset.progress;h.click('[data-rl-overlay="math"]');assert.ok(h.dialog.open);assert.equal(h.frames.size,0);assert.equal(h.dialog.dataset.progress,before);assert.equal(h.doc.activeElement,h.find('[data-rl-close]',true));assert.ok(h.doc.documentElement.classList.contains('rope-overlay-open'));
 h.click('[data-rl-play]',true);h.advance(2600);assert.equal(h.dialog.dataset.progress,1);assert.equal(h.root.dataset.progress,before);h.click('[data-rl-close]',true);assert.ok(!h.dialog.open);assert.equal(h.doc.activeElement,h.find('[data-rl-overlay="math"]'));assert.ok(!h.doc.documentElement.classList.contains('rope-overlay-open'));assert.equal(h.root.dataset.progress,before);h.controller.dispose();
});
test('comparison shifts animate inside their overlay and do not change the main Q position',()=>{
 const h=labHarness();h.change('[data-rl-position="qr"]',2);h.click('[data-rl-overlay="comparison"]');assert.equal(h.dialog.dataset.progress,0);h.advance(2600);const before=h.find('[data-rl-comparison]',true).innerHTML;
 h.click('[data-rl-shift="both"]',true);h.advance(1300);assert.match(h.find('[data-rl-comparison]',true).innerHTML,/Score change: 0.000000/);h.advance(1300);assert.equal(Number(h.find('[data-rl-position="qr"]').value),2);assert.equal(h.find('[data-rl-position="qr"]',true).value,3);
 h.click('[data-rl-shift="key"]',true);h.advance(2600);assert.match(h.find('[data-rl-comparison]',true).innerHTML,/Changed relative position/);assert.notEqual(h.find('[data-rl-comparison]',true).innerHTML,before);h.click('[data-rl-close]',true);assert.equal(Number(h.find('[data-rl-position="qr"]').value),2);h.controller.dispose();
});
test('matrix selection and progress slider update the combined explanation, never the main diagram',()=>{
 const h=labHarness();h.click('[data-rl-overlay="math"]');h.change('[data-rl-progress]',100,true,'input');h.click('[data-rl-vector="k"]',true);assert.match(h.find('[data-rl-matrix]',true).innerHTML,/R\(90.0°\)/);h.click('[data-rl-pair="1"]',true);assert.match(h.find('[data-rl-matrix]',true).innerHTML,/R\(30.0°\)/);assert.match(h.find('[data-rl-sum]',true).innerHTML,/0.252885/);assert.equal(h.root.dataset.progress,0);h.controller.dispose();
});
test('Escape, hidden tabs, reduced motion and disposal settle both playback owners',()=>{
 const h=labHarness();h.observer.emit(true);h.advance(100);h.doc.hidden=true;h.doc.emit('visibilitychange');assert.equal(h.frames.size,0);h.doc.hidden=false;h.click('[data-rl-overlay="comparison"]');h.advance(100);h.media.matches=true;h.media.emit('change');assert.equal(h.frames.size,0);assert.equal(h.dialog.dataset.progress,1);h.dialog.emit('cancel');assert.ok(!h.dialog.open);assert.equal(h.frames.size,0);assert.equal(h.doc.activeElement,h.find('[data-rl-overlay="comparison"]'));
 h.controller.dispose();assert.equal(h.frames.size,0);assert.ok([...h.doc.listeners.values(),...h.media.listeners.values(),...h.root.listeners.values()].every(v=>v.size===0));assert.ok(!h.dialog.isConnected);
 const r=labHarness({reduced:true});r.observer.emit(true);assert.equal(r.root.dataset.progress,1);assert.equal(r.frames.size,0);r.click('[data-rl-overlay="comparison"]');r.click('[data-rl-shift="both"]',true);assert.match(r.find('[data-rl-comparison]',true).innerHTML,/Score change: 0.000000/);assert.equal(r.frames.size,0);r.controller.dispose();
});

test('two-patch playback and move controls sit before the circles and scroll their own overlay, not the main page',async()=>{
 const html=ropeComparisonOverlayMarkup(ropeLabOverlayState({positions:{...ROPE_LAB_POSITIONS},progress:1},'comparison'));
 assert.ok(html.indexOf('data-rl-shift="both"')<html.indexOf('data-rl-pairs'));assert.match(html,/data-rl-motion-start/);
 const source=await readFile(new URL('../src/rope-lab.mjs',import.meta.url),'utf8');assert.match(source,/function showDetailMotion/);assert.match(source,/dialog.scrollTop\+=offset/);assert.match(source,/syncInputs\(dialog,s\);showDetailMotion\(\)/);
});

test('RoPE positions accept values above 7 and reject malformed, negative, fractional or unsafe numbers',()=>{
 for(const value of [0,7,8,16,64,1024,1000000,Number.MAX_SAFE_INTEGER])assert.equal(ropeLabPosition(String(value)),value);
 for(const value of ['', ' ', -1,1.5,'abc',Infinity,NaN,Number.MAX_SAFE_INTEGER+1])assert.equal(ropeLabPosition(value),null);
 assert.equal(ropeLabPosition('1e3'),1000);
 const main=ropeLabMarkup(),comparison=ropeComparisonOverlayMarkup(ropeLabOverlayState({positions:{...ROPE_LAB_POSITIONS},progress:0},'comparison'));
 assert.doesNotMatch(main,/max="7"/);assert.doesNotMatch(comparison,/max="7"/);assert.doesNotMatch(main,/Positions can exceed 7|rl-position-note/);
});
test('large position angles do not allocate SVG paths proportional to the number of rotations',()=>{
 for(const value of [8,1000,1000000,Number.MAX_SAFE_INTEGER]){
  const t=ropeLabTrace({positions:{qr:value,qc:value,kr:3,kc:2}}),main=ropeSingleCircleMarkup(t);
  assert.ok(main.length<11000);assert.doesNotMatch(main,/NaN|Infinity|undefined/);
  for(const p of t.pairs){assert.ok(ropeLabPairMarkup(p).length<14000);near(Math.hypot(...p.qRotated),1);assert.ok(main.includes(`data-angle="${p.qAngle}"`));}
 }
});
test('visible autoplay completes the turn, holds for exactly two seconds, and repeats',()=>{
 const h=labHarness();assert.equal(ROPE_LAB_HOLD_MS,2000);assert.equal(h.frames.size,0);assert.equal(h.root.dataset.progress,0);assert.equal(h.observer.target,h.find('.rl-single-visual'));
 h.observer.emit(false);h.observer.emit(true,.2);assert.equal(h.frames.size,0);
 h.observer.emit(true,.6);h.advance(2600);assert.equal(h.root.dataset.progress,1);assert.equal(h.root.dataset.phase,'holding');
 h.advance(1999);assert.equal(h.root.dataset.progress,1);h.observer.emit(true,.9);assert.equal(h.frames.size,1);
 h.advance(1);assert.equal(h.root.dataset.progress,0);assert.equal(h.root.dataset.phase,'rotating');h.advance(1300);near(h.root.dataset.progress,.5);
 h.advance(1300);assert.equal(h.root.dataset.phase,'holding');h.advance(2000);assert.equal(h.root.dataset.progress,0);assert.equal(h.frames.size,1);h.controller.dispose();assert.equal(h.frames.size,0);
});
test('scrolling away pauses unseen rotation and returning resumes rather than restarting',()=>{
 const h=labHarness();h.observer.emit(true);h.advance(650);h.observer.emit(false);assert.equal(h.frames.size,0);h.advance(5000);near(h.root.dataset.progress,.25);h.observer.emit(true);assert.equal(h.frames.size,1);h.advance(650);near(h.root.dataset.progress,.5);h.controller.dispose();
});
test('scrolling away during the hold preserves its remaining delay without background work',()=>{
 const h=labHarness();h.observer.emit(true);h.advance(2600);h.advance(750);h.observer.emit(false);assert.equal(h.frames.size,0);assert.equal(h.root.dataset.phase,'holding');
 h.advance(10000);assert.equal(h.root.dataset.progress,1);h.observer.emit(true);h.advance(1249);assert.equal(h.root.dataset.progress,1);h.advance(1);assert.equal(h.root.dataset.progress,0);h.controller.dispose();
});
test('editing row or column replays immediately and duplicate change/Enter does not restart playback',()=>{
 const h=labHarness();h.observer.emit(true);h.advance(2600);
 h.change('[data-rl-position="qr"]',16,false,'input');assert.equal(h.find('[data-rl-position="qr"]').value,'16');assert.equal(h.frames.size,1);assert.equal(h.root.dataset.progress,0);h.advance(650);near(h.root.dataset.progress,.25);
 h.change('[data-rl-position="qr"]',16);near(h.root.dataset.progress,.25);assert.equal(h.frames.size,1);
 h.change('[data-rl-position="qc"]',9,false,'input');assert.equal(h.root.dataset.progress,0);assert.equal(h.frames.size,1);h.advance(2600);assert.equal(h.root.dataset.progress,1);assert.match(h.find('[data-rl-single-values]').innerHTML,/480° target/);assert.match(h.find('[data-rl-single-values]').innerHTML,/135° target/);h.controller.dispose();
});
test('edits while out of view wait for visibility, while malformed inputs leave the last valid model state intact',()=>{
 const h=labHarness();h.change('[data-rl-position="qr"]',64,false,'input');assert.equal(h.frames.size,0);h.observer.emit(true);assert.equal(h.frames.size,1);h.advance(2600);assert.match(h.find('[data-rl-single-values]').innerHTML,/1920° target/);
 for(const bad of ['',-1,1.5,Infinity]){h.change('[data-rl-position="qr"]',bad,false,'input');assert.equal(h.frames.size,1);assert.match(h.find('[data-rl-single-values]').innerHTML,/1920° target/);}
 h.controller.dispose();
});
test('comparison-overlay coordinate changes autoplay without leaking into the main patch',()=>{
 const h=labHarness();h.observer.emit(true);h.advance(2600);h.click('[data-rl-overlay="comparison"]');h.change('[data-rl-position="kr"]',20,true,'input');assert.equal(h.frames.size,1);h.advance(2600);assert.equal(h.dialog.dataset.progress,1);assert.match(h.find('[data-rl-pairs]',true).innerHTML,/600.0°/);assert.equal(h.find('[data-rl-position="qr"]').value,'1');
 h.click('[data-rl-shift="both"]',true);h.advance(2600);assert.match(h.find('[data-rl-comparison]',true).innerHTML,/Score change: 0.000000/);h.controller.dispose();
});
test('hidden tabs and overlays pause the repeating loop and resume the same pose on return',()=>{
 const h=labHarness();h.doc.hidden=true;h.observer.emit(true);assert.equal(h.frames.size,0);h.doc.hidden=false;h.doc.emit('visibilitychange');assert.equal(h.frames.size,1);h.advance(650);
 h.doc.hidden=true;h.doc.emit('visibilitychange');assert.equal(h.frames.size,0);h.doc.hidden=false;h.doc.emit('visibilitychange');h.advance(650);near(h.root.dataset.progress,.5);
 h.click('[data-rl-overlay="math"]');h.observer.emit(false);h.observer.emit(true);assert.equal(h.frames.size,0);h.advance(5000);near(h.root.dataset.progress,.5);
 h.click('[data-rl-close]',true);assert.equal(h.frames.size,1);near(h.root.dataset.progress,.5);h.advance(1300);assert.equal(h.root.dataset.progress,1);h.advance(2000);assert.equal(h.root.dataset.progress,0);h.controller.dispose();
});
test('reduced motion completes first-view and edit replays instantly; disposal removes the observer',()=>{
 const h=labHarness({reduced:true});h.observer.emit(true);assert.equal(h.root.dataset.progress,1);assert.equal(h.frames.size,0);h.change('[data-rl-position="qr"]',100,false,'input');assert.equal(h.root.dataset.progress,1);assert.equal(h.frames.size,0);assert.match(h.find('[data-rl-single-values]').innerHTML,/3000° target/);h.controller.dispose();assert.ok(h.observer.disconnected);h.observer.emit(true);assert.equal(h.frames.size,0);
});

test('full-turn readouts distinguish rotation travel from the final remaining angle',()=>{
 const radians=degrees=>degrees*Math.PI/180;
 for(const [degrees,turns,remainder] of [[0,0,0],[1,0,1],[360,1,0],[361,1,1],[480,1,120],[720,2,0],[721,2,1],[10830,30,30]]){
  assert.deepEqual(ropeLabTurns(radians(degrees)),{degrees,turns,remainder});
 }
 assert.match(ropeTurnsMarkup(radians(480)),/480.0° = 1 full turn \+ 120.0°/);
 assert.match(ropeTurnsMarkup(radians(720)),/720.0° = 2 full turns \+ 0.0°/);
 assert.match(ropeSingleValuesMarkup(ropeLabTrace({positions:{qr:16,qc:9,kr:3,kc:2}})),/data-rl-turn-count="1"/);
});
test('361 degrees and 1 degree yield the same rotated vector and do not retain turn count',()=>{
 for(const vector of [[1,0],[.8,.6],[-2,.7]]){
  const a=ropeLabRotate(vector,Math.PI/180),b=ropeLabRotate(vector,361*Math.PI/180);
  a.forEach((value,i)=>near(value,b[i]));
 }
});
test('positions 361 and 1 happen to collide for the teaching speeds, not all frequencies',()=>{
 const a=ropeLabTrace({positions:{qr:1,qc:1,kr:3,kc:2}}),b=ropeLabTrace({positions:{qr:361,qc:361,kr:3,kc:2}});
 for(let i=0;i<2;i++)a.pairs[i].qRotated.forEach((v,d)=>near(v,b.pairs[i].qRotated[d]));
 const slow=[1,0],frequency=.1;assert.ok(Math.abs(ropeLabRotate(slow,frequency)[0]-ropeLabRotate(slow,361*frequency)[0])>.01);
});
test('the removed grid-history banner stays absent while wrap explanation is optional and precise',async()=>{
 const html=ropeLabMarkup(),explanation=ropeWrapExplanationMarkup();assert.doesNotMatch(html,/Positions can exceed 7|old 8×8 grid|rl-position-note/);
 assert.match(explanation,/<details class="rl-wrap-explanation">/);assert.doesNotMatch(explanation,/<details[^>]* open/);
 assert.match(explanation,/361° = 360° \+ 1°/);assert.match(explanation,/same final direction and the same two values/);assert.match(explanation,/does not store how many full turns/);assert.match(explanation,/Position indices are not degrees/);assert.match(explanation,/not a general 360-position reset/);
 const bundle=await readFile(new URL('../standalone.html',import.meta.url),'utf8');assert.doesNotMatch(bundle,/Positions can exceed 7 — that was only the old 8×8 grid/);assert.match(bundle,/function ropeLabTurns/);assert.match(bundle,/Beyond 360°: how is 361° different from 1°/);
});

test('editing a position cancels the old hold and immediately starts a fresh rotation',()=>{
 const h=labHarness();h.observer.emit(true);h.advance(2600);h.advance(1500);h.change('[data-rl-position="qr"]',8,false,'input');assert.equal(h.root.dataset.progress,0);assert.equal(h.root.dataset.phase,'rotating');assert.equal(h.frames.size,1);
 h.advance(500);near(h.root.dataset.progress,500/2600);h.advance(2100);assert.equal(h.root.dataset.progress,1);h.advance(1999);assert.equal(h.root.dataset.phase,'holding');h.advance(1);assert.equal(h.root.dataset.phase,'rotating');h.controller.dispose();
});
test('reduced-motion changes settle the result and re-enable the visible loop without duplicate owners',()=>{
 const h=labHarness();h.observer.emit(true);h.advance(650);h.media.matches=true;h.media.emit('change');assert.equal(h.root.dataset.progress,1);assert.equal(h.root.dataset.phase,'static');assert.equal(h.frames.size,0);
 h.observer.emit(false);h.observer.emit(true);assert.equal(h.frames.size,0);h.media.matches=false;h.media.emit('change');assert.equal(h.frames.size,1);assert.equal(h.root.dataset.progress,0);h.media.emit('change');assert.equal(h.frames.size,1);h.controller.dispose();assert.equal(h.frames.size,0);
});
test('loop holds do not repeatedly render unchanged values',()=>{
 const h=clock(),values=[],controller=createRopeLabLoop(h.win,(progress,phase)=>values.push({progress,phase}));controller.resume();h.advance(2600);assert.deepEqual(values.at(-1),{progress:1,phase:'holding'});
 const count=values.length;for(let i=0;i<19;i++)h.advance(100);assert.equal(values.length,count);h.advance(100);assert.deepEqual(values.at(-1),{progress:0,phase:'rotating'});controller.dispose();assert.equal(h.jobs.size,0);
});
test('cancelling or disposing from a loop callback cannot schedule a stale animation',()=>{
 const h=clock();let controller;controller=createRopeLabLoop(h.win,()=>controller.pause());controller.resume();assert.equal(h.jobs.size,0);controller.dispose();controller.resume();assert.equal(h.jobs.size,0);
});
test('main and two-patch comparison omit playback controls while the math overlay retains them',async()=>{
 const main=ropeLabMarkup();assert.doesNotMatch(main,/data-rl-play[ >]|data-rl-reset|data-rl-progress|Rotation progress/);assert.match(main,/holds for 2 seconds/);
 for(const kind of ['math','comparison']){const state=ropeLabOverlayState({positions:{...ROPE_LAB_POSITIONS},progress:.5},kind),html=kind==='math'?ropeMathOverlayMarkup(state):ropeComparisonOverlayMarkup(state);if(kind==='math'){assert.match(html,/data-rl-play/);assert.match(html,/data-rl-reset/);assert.match(html,/data-rl-progress/);}else{assert.doesNotMatch(html,/data-rl-play[ >]|data-rl-reset|data-rl-progress/);assert.doesNotMatch(html,/holds for 2 seconds|Both animations take|data-rl-loop-caption|rl-timing-note/);}}
 const bundle=await readFile(new URL('../standalone.html',import.meta.url),'utf8');assert.match(bundle,/function createRopeLabLoop/);assert.match(bundle,/ROPE_LAB_HOLD_MS=2000/);
});

test('two-patch overlay animates once on open and stops indefinitely with no timing captions or controls',()=>{
 const h=labHarness();h.click('[data-rl-overlay="comparison"]');assert.equal(h.frames.size,1);assert.equal(h.dialog.dataset.progress,0);assert.equal(h.dialog.dataset.phase,'rotating');
 for(const selector of ['[data-rl-play]','[data-rl-reset]','[data-rl-progress]','[data-rl-loop-caption]','.rl-timing-note'])assert.equal(h.find(selector,true),null);
 h.advance(1300);near(h.dialog.dataset.progress,.5);h.advance(1300);assert.equal(h.dialog.dataset.phase,'complete');assert.equal(h.dialog.dataset.playing,false);assert.equal(h.frames.size,0);assert.equal(h.find('[data-rl-shift="both"]',true).disabled,false);
 const rendered=h.find('[data-rl-pairs]',true).innerHTML;for(const time of [1999,1,2600,10000]){h.advance(time);assert.equal(h.dialog.dataset.progress,1);assert.equal(h.frames.size,0);assert.equal(h.find('[data-rl-pairs]',true).innerHTML,rendered);}
 h.controller.dispose();
});
test('a shared-position move animates once then keeps its before/after positions and result',()=>{
 const h=labHarness();h.click('[data-rl-overlay="comparison"]');h.advance(2600);h.click('[data-rl-shift="both"]',true);
 const coordinates=['qr','qc','kr','kc'].map(k=>h.find(`[data-rl-position="${k}"]`,true).value);
 h.advance(1300);assert.match(h.find('[data-rl-comparison]',true).innerHTML,/Score change: 0.000000/);h.advance(1300);assert.equal(h.dialog.dataset.phase,'complete');assert.equal(h.frames.size,0);
 const result=h.find('[data-rl-comparison]',true).innerHTML;h.advance(30000);assert.equal(h.find('[data-rl-comparison]',true).innerHTML,result);assert.deepEqual(['qr','qc','kr','kc'].map(k=>h.find(`[data-rl-position="${k}"]`,true).value),coordinates);assert.equal(h.frames.size,0);h.controller.dispose();
});
test('the key-only experiment runs once and preserves its final score',()=>{
 const h=labHarness();h.click('[data-rl-overlay="comparison"]');h.advance(2600);h.click('[data-rl-shift="key"]',true);h.advance(2600);const result=h.find('[data-rl-comparison]',true).innerHTML;
 assert.match(result,/Changed relative position/);assert.doesNotMatch(result,/Score change: 0.000000/);h.advance(2000);h.advance(2600);assert.equal(h.find('[data-rl-comparison]',true).innerHTML,result);h.controller.dispose();
});
test('hidden tabs resume an unfinished comparison but cannot restart a completed one',()=>{
 const h=labHarness();h.click('[data-rl-overlay="comparison"]');h.advance(650);h.doc.hidden=true;h.doc.emit('visibilitychange');assert.equal(h.frames.size,0);assert.equal(h.dialog.dataset.playing,false);h.advance(10000);near(h.dialog.dataset.progress,.25);
 h.doc.hidden=false;h.doc.emit('visibilitychange');assert.equal(h.frames.size,1);h.advance(1950);assert.equal(h.dialog.dataset.phase,'complete');assert.equal(h.frames.size,0);
 h.doc.hidden=true;h.doc.emit('visibilitychange');h.advance(2000);h.doc.hidden=false;h.doc.emit('visibilitychange');assert.equal(h.dialog.dataset.progress,1);assert.equal(h.frames.size,0);h.controller.dispose();
});
test('coordinate editing starts one new comparison without duplicate frames or automatic repetition',()=>{
 const h=labHarness();h.click('[data-rl-overlay="comparison"]');h.advance(2600);h.click('[data-rl-shift="both"]',true);h.advance(2600);
 h.change('[data-rl-position="kr"]',20,true,'input');assert.equal(h.frames.size,1);assert.equal(h.dialog.dataset.progress,0);h.advance(650);near(h.dialog.dataset.progress,.25);
 h.change('[data-rl-position="kr"]',20,true);near(h.dialog.dataset.progress,.25);h.advance(1950);assert.match(h.find('[data-rl-pairs]',true).innerHTML,/600.0°/);assert.equal(h.frames.size,0);h.advance(2000);assert.equal(h.dialog.dataset.progress,1);assert.equal(h.frames.size,0);h.controller.dispose();
});
test('reopening replays the completed comparison while closing restores the main-page loop',()=>{
 const h=labHarness();h.observer.emit(true);h.advance(650);const main=h.root.dataset.progress;h.click('[data-rl-overlay="comparison"]');h.advance(2600);h.click('[data-rl-close]',true);assert.equal(h.frames.size,1);near(h.root.dataset.progress,main);h.advance(650);near(h.root.dataset.progress,.5);
 h.click('[data-rl-overlay="comparison"]');assert.equal(h.dialog.dataset.progress,0);assert.equal(h.frames.size,1);h.dialog.emit('cancel');assert.equal(h.frames.size,1);h.observer.emit(false);assert.equal(h.frames.size,0);h.controller.dispose();
});
test('comparison reduced-motion toggles show the final answer and never restart a completed animation',()=>{
 const h=labHarness();h.click('[data-rl-overlay="comparison"]');h.advance(650);h.media.matches=true;h.media.emit('change');assert.equal(h.dialog.dataset.phase,'complete');assert.equal(h.dialog.dataset.progress,1);assert.equal(h.frames.size,0);
 h.click('[data-rl-shift="both"]',true);assert.equal(h.dialog.dataset.progress,1);assert.equal(h.frames.size,0);assert.match(h.find('[data-rl-comparison]',true).innerHTML,/Score change: 0.000000/);
 h.media.matches=false;h.media.emit('change');assert.equal(h.dialog.dataset.progress,1);assert.equal(h.frames.size,0);h.media.emit('change');assert.equal(h.frames.size,0);h.click('[data-rl-close]',true);h.click('[data-rl-overlay="comparison"]');assert.equal(h.dialog.dataset.progress,0);assert.equal(h.frames.size,1);h.controller.dispose();assert.equal(h.frames.size,0);
});
test('per-pair explanation separates a common angular frequency from each position-dependent turn and starting direction',()=>{
 const trace=ropeLabTrace(),row=trace.pairs[0],html=ropePairTurnExplanation(row);
 assert.match(html,/Same rate: 30.0° per row for Q and K/);assert.match(html,/Different row positions/);assert.match(html,/Q starts at 0.0°; K starts at 36.9°/);assert.match(html,/starting direction \+ the added RoPE turn/);
 near(row.targetQ,Math.PI/6);near(row.targetK,Math.PI/2);near(rlTestAngle(row.kRotated),126.86989764584402);
 const same=ropeLabTrace({positions:{qr:3,qc:1,kr:3,kc:2}}).pairs[0];near(same.qAngle,same.kAngle);assert.notDeepEqual(same.qRotated,same.kRotated);assert.match(ropePairTurnExplanation(same),/Same row → same added turn/);assert.match(ropePairTurnExplanation(same),/equal turns need not make the arrows overlap/);
 const overlay=ropeComparisonOverlayMarkup(ropeLabOverlayState({positions:{...ROPE_LAB_POSITIONS},progress:0},'comparison'));assert.doesNotMatch(overlay,/larger required turn looks faster|Both animations take|timing is illustrative/);
});
function rlTestAngle(v){return Math.atan2(v[1],v[0])*180/Math.PI;}

test('the comparison no longer contains either removed timing caption, including in the offline bundle',async()=>{
 const html=ropeComparisonOverlayMarkup(ropeLabOverlayState({positions:{...ROPE_LAB_POSITIONS},progress:0},'comparison'));
 assert.doesNotMatch(html,/Rotates automatically|holds for 2 seconds|Both animations take|2\.6 seconds|rl-timing-note|data-rl-loop-caption/);assert.match(html,/data-rl-narration/);
 const bundle=await readFile(new URL('../standalone.html',import.meta.url),'utf8');assert.doesNotMatch(bundle,/Both animations take 2\.6 seconds|Replays this same move|data-rl-loop-caption|comparisonLoop/);assert.match(bundle,/const comparisonPlayback=createRopeLabPlayback/);assert.match(bundle,/const playback=createRopeLabLoop/);
});
test('closing an unfinished comparison cancels it and reopening starts from zero with just one frame owner',()=>{
 const h=labHarness();h.click('[data-rl-overlay="comparison"]');h.advance(1300);near(h.dialog.dataset.progress,.5);h.dialog.emit('cancel');assert.equal(h.frames.size,0);h.advance(10000);h.click('[data-rl-overlay="comparison"]');assert.equal(h.dialog.dataset.progress,0);assert.equal(h.frames.size,1);h.advance(2600);assert.equal(h.frames.size,0);assert.equal(h.dialog.dataset.phase,'complete');h.controller.dispose();
});
