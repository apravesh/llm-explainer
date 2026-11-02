import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matrix,viewMatrix,weightDescriptor} from '../src/full-config.mjs';
import {createMatrixRenderer} from '../src/full-matrix.mjs';
import {patchWindow,patchWindowLabel,matrixPatchWindow,mountSharedPatchWindow,createPatchWindowMotion} from '../src/patch-window.mjs';
import {compactAttentionMarkup} from '../src/compact-attention.mjs';

test('six-patch windows clamp at both ends and label CLS separately',()=>{
 assert.deepEqual(patchWindow(48,19).indices,[19,20,21,22,23,24]);
 assert.equal(patchWindow(48,-5).start,0);assert.equal(patchWindow(48,99).start,42);
 assert.deepEqual(patchWindow(3,99).indices,[0,1,2]);assert.match(patchWindowLabel(patchWindow(37,0),true),/^CLS–p04 · 6 of 37$/);
 assert.match(patchWindowLabel(patchWindow(37,1),true),/^p00–p05/);
});
test('matrices stay full by default and single-row or weight calculations are never windowed',()=>{
 const a=matrix(48,1024),setting={start:19,total:48};
 assert.equal(matrixPatchWindow(a,null).count,48);assert.equal(matrixPatchWindow(a,setting).count,6);
 assert.equal(matrixPatchWindow(viewMatrix(a,{rowStart:21,rows:1}),setting).count,1);
 assert.equal(matrixPatchWindow(weightDescriptor(48,1024,1),setting).count,48);
});
function drawingHarness(){
 const renderer=createMatrixRenderer('test'),canvases=[],selected=[];
 const context=()=>({strokes:[],draws:[],putImageData(){},clearRect(){},drawImage(image){this.draws.push(image);},save(){},restore(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},strokeRect(...args){this.strokes.push(args);}});
 const doc={createElement(kind){if(kind==='canvas'){const ctx=context();return {width:0,height:0,getContext:()=>ctx};}return {className:'',remove(){}};}};
 function add(a,opts={}){
  const html=renderer.markup(a,opts),id=html.match(/data-full-matrix="([^"]+)"/)[1],ctx=context(),dimension={textContent:''};let note=null;
  const shell={querySelector:s=>s==='.fm-window-caption'?note:s==='.fm-label>span'?dimension:null,append(n){note=n;}};
  const canvas={dataset:{fullMatrix:id},width:0,height:0,ownerDocument:doc,parentElement:{querySelector:()=>null},getBoundingClientRect:()=>({left:0,top:0,width:96,height:60}),getContext:()=>ctx,closest:s=>s==='[data-full-matrix]'?canvas:s==='.fm-matrix'?shell:null,focus(){}};
  canvases.push(canvas);return {canvas,ctx,dimension,get note(){return note;}};
 }
 const root={querySelectorAll:()=>canvases,contains:c=>canvases.includes(c),querySelector:()=>canvases[0]};
 return {renderer,add,root,selected};
}
test('renderer opt-in crops MLP only; embedding/output stay full and hover/pick use global row identities',()=>{
 const previous=globalThis.ImageData;globalThis.ImageData=class{constructor(data,width,height){Object.assign(this,{data,width,height});}};
 try{
  const h=drawingHarness(),a=matrix(48,8,Float32Array.from({length:48*8},(_,i)=>i));
  const embedding=h.add(a,{label:'Embedding'}),output=h.add(a,{label:'Output'}),mlp=h.add(a,{label:'MLP',followPatchWindow:true,row:21,click:(r,c)=>h.selected.push([r,c])});
  h.renderer.setPatchWindow({start:19,total:48});h.renderer.draw(h.root);
  assert.equal(embedding.canvas.dataset.patchWindowCount,undefined);assert.equal(output.canvas.dataset.patchWindowCount,undefined);
  assert.equal(embedding.ctx.draws.at(-1).height,48);assert.equal(output.ctx.draws.at(-1).height,48);
  assert.equal(mlp.ctx.draws.at(-1).height,6);assert.equal(mlp.canvas.dataset.patchWindowStart,19);assert.equal(mlp.canvas.dataset.patchWindowCount,6);
  assert.equal(mlp.dimension.textContent,'full 48 × 8');assert.equal(mlp.note.textContent,'view 6 × 8');assert.match(mlp.note.title,/p19–p24/);
  h.renderer.hover({target:mlp.canvas,clientX:6,clientY:35});assert.match(mlp.canvas.title,/MLP\[22, 0\] = 176/);
  h.renderer.pick({type:'click',target:mlp.canvas,clientX:6,clientY:35},h.root);assert.deepEqual(h.selected,[[22,0]]);
  const old=mlp.ctx.strokes.length;h.renderer.setPatchWindow({start:42,total:48});h.renderer.draw(h.root);
  assert.equal(mlp.canvas.dataset.patchWindowStart,42);assert.equal(mlp.ctx.strokes.length,old,'off-window selected row is not falsely highlighted');
  assert.equal(embedding.canvas.dataset.patchWindowCount,undefined);assert.equal(output.canvas.dataset.patchWindowCount,undefined);
 }finally{if(previous===undefined)delete globalThis.ImageData;else globalThis.ImageData=previous;}
});
test('compact attention shows a 6×6 crop of full softmax, with no QKV mini-panels or renormalization',()=>{
 const n=48,attention=matrix(n,n,new Float32Array(n*n).fill(1/n)),data={n,c:{cls:false},head:{attention},query:21,key:21};
 const before=attention.data.slice(),html=compactAttentionMarkup(data,19);
 assert.equal((html.match(/data-window-pair=/g)||[]).length,36);assert.match(html,/data-window-pair="19:19"/);assert.match(html,/data-window-pair="24:24"/);assert.doesNotMatch(html,/data-window-pair="25:/);
 assert.match(html,/2.0833%/);assert.match(html,/full-row softmax over 48 keys/);assert.match(html,/Full 48 × 48 · no causal mask/);assert.doesNotMatch(html,/qkv|Query Q|Value V|16.6667%/);assert.deepEqual(attention.data,before);
 assert.match(compactAttentionMarkup({...data,c:{cls:true}},0),/CLS/);
});
class Events{constructor(){this.events=new Map();}addEventListener(t,f){if(!this.events.has(t))this.events.set(t,new Set());this.events.get(t).add(f);}removeEventListener(t,f){this.events.get(t)?.delete(f);}emit(t,e={}){for(const f of this.events.get(t)||[])f(e);}}
function scrollHarness({cls=false,horizontal=false,total=48}={}){
 let id=0,start=19;const frames=new Map(),changes=[],win=Object.assign(new Events(),{getComputedStyle:()=>({flexDirection:horizontal?'row':'column'}),requestAnimationFrame:f=>{frames.set(++id,f);return id;},cancelAnimationFrame:id=>frames.delete(id)});
 const rail=Object.assign(new Events(),{scrollTop:0,scrollLeft:0,getBoundingClientRect:()=>({top:100,left:20})});
 const items=Array.from({length:total},(_,i)=>({dataset:{fullPatch:String(i-(cls?1:0))},active:false,classList:{toggle(_,v){items[i].active=v;}},getBoundingClientRect:()=>({top:100+i*36-rail.scrollTop,bottom:134+i*36-rail.scrollTop,left:20+i*36-rail.scrollLeft,right:54+i*36-rail.scrollLeft})}));
 rail.querySelectorAll=()=>items;const root={ownerDocument:{defaultView:win}};
 const controller=mountSharedPatchWindow({root,rail,total,cls,getStart:()=>start,onStart:s=>{start=s;changes.push(s);}});
 const flush=()=>{const queue=[...frames.values()];frames.clear();queue.forEach(f=>f());};
 return {controller,rail,win,frames,items,changes,flush,get start(){return start;},key(key){let prevented=false;rail.emit('keydown',{target:rail,key,preventDefault(){prevented=true;}});return prevented;}};
}
for(const horizontal of [false,true])test(`shared ${horizontal?'horizontal':'vertical'} scroll preserves pane DOM and updates six-row window`,()=>{
 const h=scrollHarness({horizontal});assert.equal(horizontal?h.rail.scrollLeft:h.rail.scrollTop,684);assert.equal(h.items.filter(i=>i.active).length,6);
 if(horizontal)h.rail.scrollLeft=25*36;else h.rail.scrollTop=25*36;h.rail.emit('scroll');h.flush();assert.equal(h.start,25);assert.equal(h.items.filter(i=>i.active).length,6);
 assert.ok(h.key('End'));assert.equal(h.start,42);assert.ok(h.key('Home'));assert.equal(h.start,0);assert.ok(h.key('PageDown'));assert.equal(h.start,6);
 h.rail.emit('scroll');const stale=[...h.frames.values()];h.controller.dispose();stale.forEach(f=>f());assert.equal(h.start,6);assert.equal(h.frames.size,0);assert.ok([...h.rail.events.values()].every(s=>s.size===0));
});
test('shared rail includes CLS as token zero and clamps near the last patch',()=>{
 const h=scrollHarness({cls:true,total:37});h.key('Home');assert.equal(h.items[0].dataset.fullPatch,'-1');assert.ok(h.items[0].active);h.key('End');assert.equal(h.start,31);assert.equal(h.items.filter(i=>i.active).length,6);h.controller.dispose();
});
test('pan feedback respects reduced motion and cleans up on resize/dispose',()=>{
 let cancelled=0,started=0;const media=Object.assign(new Events(),{matches:false}),win=Object.assign(new Events(),{matchMedia:()=>media}),doc=Object.assign(new Events(),{defaultView:win,hidden:false}),node={getBoundingClientRect:()=>({width:100,height:60}),animate(){started++;return {cancel(){cancelled++;}};}};
 const root={ownerDocument:doc,querySelectorAll:()=>[node]},motion=createPatchWindowMotion(root);motion.play(1);assert.equal(started,1);win.emit('resize');assert.equal(cancelled,1);media.matches=true;motion.play(1);assert.equal(started,1);media.matches=false;motion.play(-1);motion.dispose();assert.equal(cancelled,2);assert.ok([...win.events.values()].every(s=>s.size===0));
});
test('current collapsed overview no longer crops matrices or advertises linked six-patch scrolling',async()=>{
 const read=n=>readFile(new URL('../'+n,import.meta.url),'utf8'),app=await read('src/full-app.mjs'),overview=await read('src/mlp-overview.mjs');
 assert.doesNotMatch(app,/mountSharedPatchWindow|updateOverviewWindow|setPatchWindow/);
 assert.doesNotMatch(overview,/followPatchWindow:true|data-full-matrix/);
 for(const file of ['vision.html','standalone.html']){const html=await read(file);assert.match(html,/id="attention-process-viz"/);assert.doesNotMatch(html,/class="ca-map"/);}
});
