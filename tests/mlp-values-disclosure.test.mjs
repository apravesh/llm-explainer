import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {FULL_CONFIGS,matrix} from '../src/full-config.mjs';
import {createFullViews} from '../src/full-views.mjs';
import {featureMarkup} from '../src/full-matrix.mjs';
function fixture(model='pixtral'){
 const c=FULL_CONFIGS[model],rows=3,make=(cols,seed)=>matrix(rows,cols,Float32Array.from({length:rows*cols},(_,i)=>Math.sin((i+seed)*.031)));
 const layer={input:make(c.width,1),ffnInput:make(c.width,2),preActivation:make(c.ffWidth,3),expanded:make(c.ffWidth,4),output:make(c.width,5),mlpDelta:make(c.width,8),afterAttention:make(c.width,9),...(c.gated?{gate:make(c.ffWidth,6)}:{})},run={config:c,layers:[layer,layer]},state={expanded:4,layer:0,selected:1,wideFeature:c.ffWidth-1,feature:c.width-1};
 const rendered=[],renderer={markup(a,options){rendered.push({a,options});return `<canvas data-preview="${options.label}"></canvas>`;}},views=createFullViews(renderer,{select(){},selectPair(){},selectHead(){}});
 return {run,state,layer,rendered,views};
}
for(const model of ['pixtral','vit']){
 test(`${model}: MLP numeric tables are grouped into one native disclosure, closed by default`,()=>{
  const {run,state,views}=fixture(model),html=views.content(run,state),match=html.match(/<details class="fa-values-disclosure" data-mlp-values([^>]*)>([\s\S]*?)<\/details>/);
  assert.ok(match);assert.doesNotMatch(match[1],/\bopen\b/);assert.match(match[2],/^<summary>Show actual values<\/summary>/);
  assert.equal((match[2].match(/class="fm-features"/g)||[]).length,model==='pixtral'?4:3);
  const outside=html.replace(match[0],'');assert.doesNotMatch(outside,/class="fm-features"/);assert.match(outside,/data-mlp-flow/);assert.match(outside,/data-mf-operation="mlp-up"/);assert.match(outside,/data-mf-operation="mlp-down"/);
  assert.equal((outside.match(/data-preview=/g)||[]).length,0);assert.equal(outside.includes('data-mf-operation="mlp-gate"'),model==='pixtral');
 });
 test(`${model}: expanding the disclosure changes no values, dimensions, matrix interactions or selection`,()=>{
  const {run,state,views,layer,rendered}=fixture(model),before=structuredClone(run),closed=views.content(run,state),opened=views.content(run,{...state,mlpValuesOpen:true}),row=state.selected+(run.config.cls?1:0);
  assert.equal(opened.replace('data-mlp-values open','data-mlp-values '),closed);
  assert.ok(opened.includes(featureMarkup(layer.preActivation,row,state.wideFeature,{label:'Before activation',key:'wide'})));
  assert.ok(opened.includes(featureMarkup(layer.expanded,row,state.wideFeature,{label:'After activation / gating',key:'wide'})));
  assert.ok(opened.includes(featureMarkup(layer.output,row,state.feature,{label:'After contraction + residual'})));
  if(run.config.gated)assert.ok(opened.includes(featureMarkup(layer.gate,row,state.wideFeature,{label:'Parallel expansion (W₃)',key:'wide'})));
  assert.deepEqual(run,before);assert.equal(state.mlpValuesOpen,undefined);assert.ok(rendered.every(({options})=>typeof options.click==='function'));
 });
 test(`${model}: open/closed state can survive patch, feature and block changes`,()=>{
  const {run,state,views}=fixture(model);
  for(const change of [{selected:0},{feature:0,wideFeature:32},{layer:1}]){
   assert.match(views.content(run,{...state,...change,mlpValuesOpen:true}),/data-mlp-values open>/);
   assert.doesNotMatch(views.content(run,{...state,...change,mlpValuesOpen:false}),/data-mlp-values open>/);
  }
 });
}
test('the live app and offline bundle preserve the native toggle without rerendering its controls',async()=>{
 for(const file of ['src/full-app.mjs','standalone.html']){
  const source=await readFile(new URL('../'+file,import.meta.url),'utf8');assert.match(source,/mlpValuesOpen:false/);
  assert.match(source,/if\(e.target.hasAttribute\('data-mlp-values'\)\)\{if\(e.target.isConnected\)state.mlpValuesOpen=e.target.open;return;\}/);
 }
 const bundle=await readFile(new URL('../standalone.html',import.meta.url),'utf8');assert.match(bundle,/<summary>Show actual values<\/summary>/);
});
