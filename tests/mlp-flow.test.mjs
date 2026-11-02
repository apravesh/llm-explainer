import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matrix,matrixValue,viewMatrix,FULL_CONFIGS} from '../src/full-config.mjs';
import {activateFull,addFull} from '../src/full-engine.mjs';
import {createFullViews} from '../src/full-views.mjs';
import {activationRGB} from '../src/full-matrix.mjs';
import {mlpFlowData,mlpFlowPreview,mlpFlowDiagramMarkup,mlpFlowMarkup,mlpFlowFrame,mountMlpFlow} from '../src/mlp-flow.mjs';
function fixture(model='pixtral'){
 const c=FULL_CONFIGS[model],make=(cols,seed)=>matrix(3,cols,Float32Array.from({length:3*cols},(_,i)=>Math.sin((i+seed)*.73)));
 const preActivation=make(c.ffWidth,3),gate=c.gated?make(c.ffWidth,7):null,mlpDelta=make(c.width,11),afterAttention=make(c.width,29),l={input:make(c.width,19),ffnInput:make(c.width,13),preActivation,gate,expanded:activateFull(preActivation,gate),mlpDelta,afterAttention,output:addFull(afterAttention,mlpDelta)};
 return {c,l,run:{config:c,layers:[l],patchCount:c.cls?2:3},state:{expanded:4,layer:0,selected:1,wideFeature:0,feature:0}};
}
for(const model of ['pixtral','vit']){
 test(`${model}: flow uses full-width data from the actual selected encoder row without changing tensors`,()=>{
  const {run,state,l,c}=fixture(model),before=structuredClone(run),d=mlpFlowData(run,state);assert.equal(d.row,c.cls?2:1);assert.equal(d.patch,1);assert.equal(d.input.cols,c.width);assert.equal(d.mixed.cols,c.ffWidth);assert.equal(d.delta.cols,c.width);
  for(const [key,source] of [['input',l.ffnInput],['mixed',l.expanded],['delta',l.mlpDelta],['residual',l.afterAttention],['output',l.output]]){assert.strictEqual(d[key].source,source);assert.equal(d[key].rowStart,d.row);assert.equal(d[key].rows,1);}
  for(let j=0;j<c.ffWidth;j++){
   const a=matrixValue(l.preActivation,d.row,j);assert.equal(matrixValue(d.activated,0,j),c.gated?a/(1+Math.exp(-a)):matrixValue(l.expanded,d.row,j));
   if(c.gated)assert.equal(matrixValue(d.mixed,0,j),Math.fround(matrixValue(d.activated,0,j)*matrixValue(d.values,0,j)));
  }
  if(!c.gated)assert.equal(d.values,null);assert.deepEqual(run,before);
 });
 test(`${model}: responsive arrow diagrams retain the operation ordering, true dimensions, and separate residual`,()=>{
  const {run,state,c}=fixture(model),d=mlpFlowData(run,state);
  for(const compact of [false,true]){const html=mlpFlowDiagramMarkup({...d,calculationOpen:false},compact);assert.equal((html.match(/data-mf-node=/g)||[]).length,c.gated?8:6);assert.equal((html.match(/data-mf-operation=/g)||[]).length,c.gated?3:2);assert.match(html,/Residual bypass · not normalized x/);assert.match(html,/Before normalization/);assert.match(html,/H \+ Δ → H′/);assert.match(html,/Before the residual addition/);assert.match(html,new RegExp(`1 × ${c.width.toLocaleString()}`));assert.match(html,new RegExp(`1 × ${c.ffWidth.toLocaleString()}`));
   if(c.gated){assert.match(html,/g ⊙ v/);assert.match(html,/Same x → both branches/);assert.equal((html.match(/data-mf-edge="branches"/g)||[]).length,2);}else{assert.match(html,/GELU/);assert.doesNotMatch(html,/W₃|SiLU|g ⊙ v/);}
   assert.doesNotMatch(html,/NaN|undefined|4 → 6 → 4|toy patch/i);
  }
 });
}
test('CLS is row zero, not patch zero, and selecting another patch changes the real previews',()=>{
 const {run,state}=fixture('vit'),cls=mlpFlowData(run,{...state,selected:-1}),patch=mlpFlowData(run,{...state,selected:0});assert.equal(cls.row,0);assert.equal(cls.patch,-1);assert.equal(patch.row,1);assert.notEqual(matrixValue(cls.input,0,0),matrixValue(patch.input,0,0));assert.match(mlpFlowMarkup(run,{...state,selected:-1}),/Follow CLS/);assert.equal(mlpFlowData({...run,layers:[]},state),null);
});
test('previews pool real feature means and disclose an independent color scale without changing widths',()=>{
 const a=matrix(1,32,Float32Array.from({length:32},(_,i)=>(i%5-2)/100)),before=a.data.slice(),preview=mlpFlowPreview(a);assert.equal(preview.width,16);assert.equal(preview.height,1);assert.equal(a.cols,32);assert.match(preview.mode,/pooled/);
 const means=Array.from({length:16},(_,i)=>(a.data[2*i]+a.data[2*i+1])/2);assert.equal(preview.scale,Math.max(...means.map(Math.abs)));
 means.forEach((m,i)=>assert.deepEqual([...preview.data.slice(4*i,4*i+3)],activationRGB(m,preview.scale)));assert.deepEqual(a.data,before);
 const zero=mlpFlowPreview(matrix(1,64));assert.equal(zero.scale,1);assert.ok([...zero.data].every(Number.isFinite));
 const {run,state}=fixture();assert.match(mlpFlowMarkup(run,state),/auto-scale colors independently/);
});
test('full calculation is opt-in below the persistent flow and actual values remain collapsed',()=>{
 const {run,state,c,l}=fixture();l.weights={up:{kind:'weights',rows:c.width,cols:c.ffWidth,seed:11,scale:.1},gate:{kind:'weights',rows:c.width,cols:c.ffWidth,seed:13,scale:.1},down:{kind:'weights',rows:c.ffWidth,cols:c.width,seed:17,scale:.1}};
 const renderer={markup:()=>'<canvas></canvas>'},views=createFullViews(renderer,{select(){},selectMlpFeature(){}}),html=views.content(run,state);assert.match(html,/data-mlp-flow/);assert.doesNotMatch(html,/data-mlp-calculation data-/);assert.doesNotMatch(html,/data-token-lanes/);assert.match(html,/data-mlp-values ><summary>Show actual values/);
 const detailed=views.content(run,{...state,mlpCalculation:true});assert.match(detailed,/data-mlp-flow/);assert.match(detailed,/data-mlp-calculation data-/);assert.ok(detailed.indexOf('data-mlp-flow')<detailed.indexOf('data-mlp-calculation data-'));assert.match(detailed,/− Hide full calculation/);assert.equal((detailed.match(/data-mlp-calculation data-/g)||[]).length,1);
});
test('diagram geometry is bounded and each responsive variant has unique marker ids',()=>{
 const {run,state}=fixture(),d=mlpFlowData(run,state),wide=mlpFlowDiagramMarkup(d),compact=mlpFlowDiagramMarkup(d,true);assert.ok(wide.length<26000);assert.ok(compact.length<26000);assert.equal((wide.match(/class="mf-feature"/g)||[]).length,7*16);assert.match(wide,/mf-wide-swiglu/);assert.match(compact,/mf-compact-swiglu/);
});
test('flow tracing has the parallel branches, elementwise product, contraction and residual as distinct stages',()=>{
 for(const [ms,phase] of [[-1,'input'],[0,'input'],[650,'branches'],[1400,'mix'],[2100,'contraction'],[2800,'residual'],[3500,'complete'],[9000,'complete']])assert.equal(mlpFlowFrame(ms),phase);
});
class Events{constructor(){this.listeners=new Map();}addEventListener(t,f){if(!this.listeners.has(t))this.listeners.set(t,new Set());this.listeners.get(t).add(f);}removeEventListener(t,f){this.listeners.get(t)?.delete(f);}emit(t,e={}){for(const f of this.listeners.get(t)||[])f(e);}}
function harness({autoplay=true,reduced=false,gated=true}={}){
 const frames=new Map(),actions=[],media=Object.assign(new Events(),{matches:reduced});let now=0,id=0;
 const win={matchMedia:()=>media,performance:{now:()=>now},requestAnimationFrame:f=>{frames.set(++id,f);return id;},cancelAnimationFrame:id=>frames.delete(id)},doc=Object.assign(new Events(),{defaultView:win,hidden:false,querySelector:()=>null});
 const nodes=['input','up',...(gated?['gate','mix']:[]),'down','delta','residual','output'].map(key=>({dataset:{mfNode:key},active:false,classList:{toggle(_,v){nodes.find(n=>n.dataset.mfNode===key).active=v;}}})),edges=['branches','mix','contraction','residual'].map(key=>({dataset:{mfEdge:key},classList:{toggle(){}}})),status={textContent:''};
 const root=Object.assign(new Events(),{ownerDocument:doc,isConnected:true,dataset:{},querySelector:()=>status,querySelectorAll:s=>s==='[data-mf-node]'?nodes:edges});const controller=mountMlpFlow(root,{gated,autoplay,onCalculation:action=>actions.push(action)});
 function target(selector){return {closest:s=>s===selector?{}:null};}
 return {controller,root,doc,media,frames,actions,nodes,advance(ms){now+=ms;const fs=[...frames.values()];frames.clear();fs.forEach(f=>f(now));},click(selector){root.emit('click',{target:target(selector)});},key(key){let prevented=false;root.emit('keydown',{key,target:target('[data-mf-operation]'),preventDefault(){prevented=true;}});return prevented;}};
}
test('arrow tracing highlights both real expansion branches simultaneously and terminates once',()=>{
 const h=harness();h.advance(650);assert.deepEqual(h.nodes.filter(n=>n.active).map(n=>n.dataset.mfNode),['up','gate']);h.advance(750);assert.equal(h.root.dataset.phase,'mix');h.advance(700);assert.deepEqual(h.nodes.filter(n=>n.active).map(n=>n.dataset.mfNode),['down','delta']);h.advance(700);assert.deepEqual(h.nodes.filter(n=>n.active).map(n=>n.dataset.mfNode),['residual','output']);h.advance(700);assert.equal(h.root.dataset.phase,'complete');assert.equal(h.frames.size,0);h.controller.dispose();
});
test('all cards and keyboard activation open the same calculation and the footer toggles it',()=>{
 const h=harness();h.click('[data-mf-operation]');assert.deepEqual(h.actions,['open']);assert.equal(h.frames.size,0);assert.ok(h.key('Enter'));assert.ok(h.key(' '));assert.deepEqual(h.actions,['open','open','open']);h.click('[data-mf-calculation-toggle]');assert.equal(h.actions.at(-1),'toggle');h.controller.dispose();
});
test('replay, hidden views, reduced motion, modal interruptions and disposal leave no stale work',()=>{
 const h=harness();h.advance(800);h.click('[data-mf-replay]');assert.equal(h.frames.size,1);assert.equal(h.root.dataset.phase,'input');h.doc.hidden=true;h.doc.emit('visibilitychange');assert.equal(h.frames.size,0);h.doc.hidden=false;h.controller.replay();h.doc.querySelector=()=>({open:true});h.advance(300);assert.equal(h.frames.size,0);h.doc.querySelector=()=>null;h.controller.replay();h.media.matches=true;h.media.emit('change');assert.equal(h.root.dataset.phase,'complete');assert.equal(h.frames.size,0);h.controller.dispose();assert.ok([...h.root.listeners.values(),...h.doc.listeners.values(),...h.media.listeners.values()].every(v=>!v.size));h.controller.replay();assert.equal(h.frames.size,0);
 const staticView=harness({autoplay:false});assert.equal(staticView.root.dataset.phase,'complete');assert.equal(staticView.frames.size,0);staticView.controller.dispose();const r=harness({reduced:true});assert.equal(r.frames.size,0);assert.equal(r.root.dataset.phase,'complete');r.controller.dispose();
});
test('detached views dispose rather than continuing unseen animation',()=>{
 const h=harness();h.root.isConnected=false;h.advance(100);assert.equal(h.frames.size,0);h.controller.replay();assert.equal(h.frames.size,0);
});
test('active and offline apps keep arrow flow independent of the toy and of the optional full calculation',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');const app=await read('src/full-app.mjs'),bundle=await read('standalone.html'),flow=await read('src/mlp-flow.mjs');assert.match(app,/mlpFlowController\?\.dispose\(\)/);assert.match(app,/mountMlpFlow\(flow/);assert.match(app,/if\(i===4&&state.expanded!==4\)mlpFlowAutoplay=true/);assert.match(app,/data-mf-calculation-toggle/);assert.match(bundle,/function mlpFlowData/);assert.match(bundle,/function mountMlpFlow/);assert.doesNotMatch(flow,/swigluLabTrace|SWIGLU_LAB_INPUTS/);
});
