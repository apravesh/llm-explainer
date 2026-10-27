import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {FULL_CONFIGS,matrix,weightDescriptor,weightAt,hash32,blockWeights,viewMatrix,matrixValue,featureWindow,estimatedParameters} from '../src/full-config.mjs';
import {createCPUBackend} from '../src/full-backend.mjs';
import {normalizeFull,addFull,activateFull,rotaryFull,attentionFull,runFullEncoder} from '../src/full-engine.mjs';
import {overviewPixels,selectedWindow} from '../src/full-matrix.mjs';
import {fullOperation} from '../src/full-operations.mjs';
import {fullDotTrace} from '../src/full-overlay.mjs';
const near=(a,b,tolerance=1e-4)=>assert.ok(Math.abs(a-b)<=tolerance*(1+Math.abs(b)),`${a} differs from ${b}`);
const synthetic=(r,c)=>matrix(r,c,Float32Array.from({length:r*c},(_,i)=>Math.sin(i*.31)));
function referenceMultiply(a,w){const out=matrix(a.rows,w.cols);for(let r=0;r<a.rows;r++)for(let j=0;j<w.cols;j++){let sum=0;for(let k=0;k<a.cols;k++)sum+=a.data[r*a.cols+k]*weightAt(w,k,j);out.data[r*w.cols+j]=sum;}return out;}
test('canonical Pixtral and ViT configs retain production widths, heads, depth, and patch size',()=>{
 const p=FULL_CONFIGS.pixtral,v=FULL_CONFIGS.vit;assert.deepEqual([p.width,p.heads,p.headWidth,p.ffWidth,p.depth,p.decoderWidth,p.patch],[1024,16,64,4096,24,5120,16]);assert.deepEqual([v.width,v.heads,v.headWidth,v.ffWidth,v.depth],[768,12,64,3072,12]);assert.equal(p.heads*p.headWidth,p.width);assert.equal(v.heads*v.headWidth,v.width);assert.ok(estimatedParameters(p)>400_000_000);
});
test('lazy FP32 weights are deterministic, dense, bounded, and independent between layers',()=>{
 const w=weightDescriptor(768,1024,71),again=weightDescriptor(768,1024,71);assert.deepEqual(w,again);assert.equal(w.kind,'weights');assert.ok(!w.data);for(let r=0;r<10;r++)for(let c=0;c<10;c++){assert.equal(weightAt(w,r,c),weightAt(again,r,c));assert.ok(Math.abs(weightAt(w,r,c))<=w.scale);}
 assert.notEqual(weightAt(blockWeights(FULL_CONFIGS.pixtral,0).q,10,20),weightAt(blockWeights(FULL_CONFIGS.pixtral,1).q,10,20));assert.notEqual(hash32(1),hash32(2));
});
for(const [rows,k,n] of [[2,19,17],[3,64,128],[1,768,1024],[1,1024,4096],[1,5120,5120]])test(`SIMD dense multiplication ${rows} × ${k} × ${n} agrees with an independent float64 reference`,async()=>{
 const backend=await createCPUBackend(),a=synthetic(rows,k),w=weightDescriptor(k,n,71);assert.match(backend.name,/SIMD/);const b=await backend.multiply(a,w);assert.deepEqual([b.rows,b.cols],[rows,n]);
 for(const row of [0,rows-1])for(const col of [0,Math.floor(n/2),n-1]){let expected=0;for(let i=0;i<k;i++)expected+=a.data[row*k+i]*weightAt(w,i,col);near(b.data[row*n+col],expected,8e-5);}
 backend.destroy();
});
test('backend safely reuses and grows its memory without stale weights or inputs',async()=>{
 const b=await createCPUBackend();for(const size of [32,512,16,64]){const a=synthetic(2,size),w=weightDescriptor(size,32,size),actual=await b.multiply(a,w),expected=referenceMultiply(a,w);actual.data.forEach((v,i)=>near(v,expected.data[i]));}b.destroy();
});
test('normalization, gating, GELU, and residuals preserve full tensor shapes and identities',()=>{
 const a=synthetic(3,1024),n=normalizeFull(a),r=normalizeFull(a,true);assert.equal(n.data.length,3072);for(let row=0;row<3;row++){const values=n.data.slice(row*1024,(row+1)*1024);near(values.reduce((x,y)=>x+y,0)/1024,0);}
 const g=synthetic(3,1024),out=activateFull(a,g),sum=addFull(a,g);for(let i=0;i<a.data.length;i++){near(out.data[i],a.data[i]/(1+Math.exp(-a.data[i]))*g.data[i]);near(sum.data[i],a.data[i]+g.data[i]);}assert.equal(activateFull(a).cols,1024);assert.notDeepEqual(n.data,r.data);
});
test('real 64-feature RoPE rotates all 32 pairs of every head and preserves norms',()=>{
 const c=FULL_CONFIGS.pixtral,a=synthetic(2,1024),out=rotaryFull(a,c,[[0,0],[2,3]]);assert.deepEqual(out.data.slice(0,1024),a.data.slice(0,1024));near(out.data.slice(1024).reduce((s,v)=>s+v*v,0),a.data.slice(1024).reduce((s,v)=>s+v*v,0));for(let h=0;h<16;h++)assert.notEqual(out.data[1024+h*64+62],a.data[1024+h*64+62]);
});
test('bidirectional and causal attention retain every head and compute the correct weighted sum',()=>{
 const c={width:64,heads:4,headWidth:16},q=synthetic(4,64),k=synthetic(4,64),v=synthetic(4,64),normal=attentionFull(q,k,v,c),causal=attentionFull(q,k,v,c,true);
 assert.equal(normal.heads.length,4);assert.deepEqual([normal.combined.rows,normal.combined.cols],[4,64]);
 for(let h=0;h<4;h++)for(let r=0;r<4;r++){near(normal.heads[h].attention.data.slice(r*4,r*4+4).reduce((a,b)=>a+b),1);for(let j=r+1;j<4;j++)assert.equal(causal.heads[h].attention.data[r*4+j],0);for(let d=0;d<16;d++){let sum=0;for(let j=0;j<4;j++)sum+=normal.heads[h].attention.data[r*4+j]*v.data[j*64+h*16+d];near(normal.heads[h].out.data[r*16+d],sum);assert.equal(normal.combined.data[r*64+h*16+d],normal.heads[h].out.data[r*16+d]);}}
});
test('streaming execution computes dependencies in order, no layer skipping, and exact projector shape',async()=>{
 const c={...FULL_CONFIGS.pixtral,width:64,heads:4,headWidth:16,ffWidth:128,depth:3,decoderWidth:128},backend=await createCPUBackend(),run={config:c,layers:[]},events=[];const pixels=synthetic(4,768);
 await runFullEncoder({pixels,rows:2,cols:2,config:c,backend,onEmbedding:v=>{Object.assign(run,v);events.push('embedding');},onLayer:(v,i)=>{run.layers[i]=v;events.push(i);},onOutput:v=>{run.final=v;events.push('output');},onProjector:v=>{Object.assign(run,v);events.push('projector');}});
 assert.deepEqual(events,['embedding',0,1,2,'output','projector']);assert.strictEqual(run.layers[0].input,run.input);assert.strictEqual(run.layers[1].input,run.layers[0].output);assert.strictEqual(run.final,run.layers[2].output);assert.deepEqual([run.projected.rows,run.projected.cols],[4,128]);
 for(const l of run.layers){assert.equal(l.expanded.cols,128);l.output.data.forEach((v,i)=>near(v,l.afterAttention.data[i]+l.mlpDelta.data[i]));}
 backend.destroy();
});
test('cancellation prevents any later layers or fabricated final output',async()=>{
 const c={...FULL_CONFIGS.pixtral,width:32,heads:2,headWidth:16,ffWidth:64,depth:3,decoderWidth:64},backend=await createCPUBackend();let cancelled=false,final=false,layers=0;
 await assert.rejects(runFullEncoder({pixels:synthetic(2,768),rows:1,cols:2,config:c,backend,onLayer:()=>{layers++;cancelled=true;},onOutput:()=>final=true,isCancelled:()=>cancelled}),/Cancelled/);assert.equal(layers,1);assert.equal(final,false);backend.destroy();
});
test('canvas overviews are bounded and disclose pooling instead of relabeling fewer values',()=>{
 const a=matrix(48,5120,Float32Array.from({length:48*5120},(_,i)=>i%2?1:-1)),p=overviewPixels(a);assert.deepEqual([p.width,p.height],[256,48]);assert.match(p.mode,/pooled/);assert.equal(p.data.length,256*48*4);const w=overviewPixels(weightDescriptor(5120,5120,71));assert.equal(w.width,256);assert.equal(w.height,160);assert.match(w.mode,/sampled weight/);
});
test('every feature including 1,023, 4,095 and 5,119 remains accessible in exact windows',()=>{
 for(const cols of [64,768,1024,4096,5120]){const a=synthetic(2,cols);for(const index of [0,31,32,cols-1]){const page=selectedWindow(a,1,index);assert.ok(page.values.length<=32);assert.equal(page.selected,index);assert.equal(page.values[index-page.start],a.data[cols+index]);}assert.equal(featureWindow(cols,99999).selected,cols-1);}
});
test('head views expose all input dimensions and only their correct output slice',()=>{
 const a=synthetic(2,1024),w=weightDescriptor(1024,1024,71),head=viewMatrix(a,{colStart:15*64,cols:64}),weights=viewMatrix(w,{colStart:15*64,cols:64});assert.deepEqual([weights.rows,weights.cols],[1024,64]);assert.equal(matrixValue(head,1,63),a.data[2047]);assert.equal(matrixValue(weights,1023,63),weightAt(w,1023,1023));
});
test('full-size UI and offline worker never call the reduced teaching engine',async()=>{
 const at=p=>new URL('../'+p,import.meta.url),html=await readFile(at('vision.html'),'utf8'),app=await readFile(at('src/full-app.mjs'),'utf8'),bundle=await readFile(at('standalone.html'),'utf8');assert.match(html,/src="src\/full-app.mjs"/);assert.doesNotMatch(app,/runEncoder\(/);assert.match(bundle,/full-worker-source/);assert.match(bundle,/WASM_KERNEL_BASE64/);assert.match(bundle,/width:1024,heads:16,headWidth:64,ffWidth:4096,depth:24,decoderWidth:5120/);assert.doesNotMatch(bundle,/DIM = 24/);
});

test('extreme image aspect ratios retain at least one whole patch, never a zero-sized canvas',async()=>{
 const {fullImageShape}=await import('../src/full-config.mjs');
 for(const [w,h] of [[1,10000],[10000,1],[640,480],[123,321],[1,1]]){
  const shape=fullImageShape(w,h,FULL_CONFIGS.pixtral);assert.ok(shape.width>=16&&shape.height>=16);assert.equal(shape.width%16,0);assert.equal(shape.height%16,0);assert.ok(shape.drawWidth>=1&&shape.drawHeight>=1);assert.ok(shape.drawWidth<=shape.width&&shape.drawHeight<=shape.height);
 }
 assert.deepEqual(fullImageShape(1,10000,FULL_CONFIGS.vit),{width:96,height:96,drawWidth:96,drawHeight:96});assert.throws(()=>fullImageShape(0,5,FULL_CONFIGS.pixtral),/Positive/);
});
test('full views use bounded canvas matrices and expose exact feature ranges at every stage',async()=>{
 const {createFullViews}=await import('../src/full-views.mjs'),c={...FULL_CONFIGS.pixtral,width:64,heads:4,headWidth:16,ffWidth:128,depth:2,decoderWidth:128},backend=await createCPUBackend();
 const run={config:c,layers:[],pixels:synthetic(2,768),patch:16,patchCount:2,rows:1,cols:2,width:32,height:16};
 await runFullEncoder({...run,backend,onEmbedding:v=>Object.assign(run,v),onLayer:(l,i)=>run.layers[i]=l,onOutput:v=>run.final=v,onProjector:v=>Object.assign(run,v)});
 const seen=[],renderer={markup(a,opts){seen.push({rows:a.rows,cols:a.cols,...opts});return `<canvas data-rows="${a.rows}" data-cols="${a.cols}"></canvas>`;}};
 const views=createFullViews(renderer,{select(){},selectPair(){},selectHead(){}}),state={selected:1,key:1,layer:1,head:3,feature:63,headFeature:15,wideFeature:127,projectorFeature:127,scalar:767};
 for(let expanded=0;expanded<7;expanded++){const html=views.content(run,{...state,expanded});assert.doesNotMatch(html,/NaN|undefined/);assert.ok(html.length<60000,`stage ${expanded} must not emit one HTML cell per model value`);}
 assert.ok(seen.some(x=>x.label==='Complete input embedding matrix'&&x.anchor==='embedding-expanded'));
 assert.ok(seen.some(x=>x.label==='Full encoder output'&&x.anchor==='output-expanded'));
 assert.ok(seen.some(x=>x.label==='Decoder-ready Y'&&x.cols===128));backend.destroy();
});
