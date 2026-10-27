import {test} from 'node:test';import assert from 'node:assert/strict';
import {FULL_CONFIGS,matrix,weightAt} from '../src/full-config.mjs';
import {createCPUBackend} from '../src/full-backend.mjs';
import {runFullEncoder} from '../src/full-engine.mjs';
import {fullOperation} from '../src/full-operations.mjs';import {fullDotTrace} from '../src/full-overlay.mjs';
test('actual Pixtral 48-token execution: 24 full 1024-wide blocks, 16 heads, 4096 FFN, 5120 projector',async()=>{
 const c=FULL_CONFIGS.pixtral,backend=await createCPUBackend(),run={config:c,layers:[],pixels:matrix(48,768,Float32Array.from({length:48*768},(_,i)=>Math.sin(i*.2)))},progress=[];
 await runFullEncoder({pixels:run.pixels,rows:6,cols:8,config:c,backend,onProgress:p=>progress.push(p),onEmbedding:v=>Object.assign(run,v),onLayer:(l,i)=>run.layers[i]=l,onOutput:v=>run.final=v,onProjector:v=>Object.assign(run,v)});
 assert.equal(run.layers.length,24);assert.equal(run.input.cols,1024);assert.equal(run.final.cols,1024);assert.deepEqual([run.projected.rows,run.projected.cols],[48,5120]);assert.equal(progress.at(-1).completed,progress.at(-1).total);
 for(let i=0;i<24;i++){const l=run.layers[i];assert.equal(l.heads.length,16);assert.equal(l.expanded.cols,4096);assert.equal(l.heads[15].out.cols,64);assert.strictEqual(l.input,i?run.layers[i-1].output:run.input);assert.ok(l.output.data.every(Number.isFinite));}
 const s={selected:47,layer:23,head:15,key:47};
 for(const kind of ['embedding','q','k','v','mlp-up','mlp-gate','mlp-down','attention-output','projector-first','projector-second']){const op=fullOperation(run,s,kind),trace=fullDotTrace(op,op.row,op.output.cols-1);assert.ok(Math.abs(trace.total-trace.expected)<2e-4*(1+Math.abs(trace.expected)),kind);}
 assert.equal(run.adapterWeights[1].rows*run.adapterWeights[1].cols,26_214_400);assert.ok(!run.adapterWeights[1].data);backend.destroy();
});
