import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {parameterFormulas,compactCount,PF_PRESETS} from '../src/param-formulas.mjs';
const part=(r,key)=>r.parts.find(p=>p.key===key);
test('dense block is 4d² attention + 8d² MLP and the totals add up',()=>{
 const d=256,L=8,V=8000,r=parameterFormulas({model:'dense',d,L,V});
 assert.equal(part(r,'attn').count,L*4*d*d);assert.equal(part(r,'mlp').count,L*8*d*d);
 assert.equal(part(r,'embed').count,V*d);assert.equal(part(r,'head').count,V*d);
 assert.equal(r.stored,2*V*d+L*12*d*d+L*2*d+d);assert.equal(r.stored,r.active);
 assert.equal(part(r,'mlp').count/(part(r,'attn').count+part(r,'mlp').count),2/3);
});
test('tying the LM head removes the second V·d table',()=>{const a=parameterFormulas({tied:false}),b=parameterFormulas({tied:true});assert.equal(a.stored-b.stored,a.config.V*a.config.d);});
test('MoE stores E experts but runs k, plus a d·E router',()=>{
 const d=128,L=4,E=8,k=2,r=parameterFormulas({model:'moe',d,L,E,k});
 assert.equal(part(r,'experts').count,L*E*8*d*d);assert.equal(part(r,'experts').active,L*k*8*d*d);assert.equal(part(r,'router').count,L*d*E);
 assert.equal(r.stored-r.active,L*(E-k)*8*d*d);assert.deepEqual(r.blockUnits,{attn:4,mlp:64,mlpActive:16});
});
test('vision encoder: patch embedding 3p²·d, no position weights, projector d·D + D·D',()=>{
 const d=256,r=parameterFormulas({model:'vision',d,p:16,Dmul:4});
 assert.equal(part(r,'patch').count,768*d);assert.equal(part(r,'pos').count,0);assert.equal(part(r,'proj').count,4*d*d+16*d*d);
 assert.ok(!part(r,'embed')&&!part(r,'head'));
});
test('compact counts',()=>{assert.equal(compactCount(0),'0');assert.equal(compactCount(950),'950');assert.equal(compactCount(2048000),'2.05M');});
test('analytics page includes the formulas-first section and module',async()=>{const html=await readFile(new URL('../analytics.html',import.meta.url),'utf8');
 for(const id of ['formula-models','pf-diagram','pf-bars','pf-table'])assert.match(html,new RegExp(`id="${id}"`));assert.match(html,/src="src\/param-formulas-app.mjs"/);assert.ok(html.indexOf('id="formula-models"')<html.indexOf('id="budget-form"'));});
test('real-model presets reproduce published parameter totals within 1%',()=>{
 const total=key=>parameterFormulas(PF_PRESETS[key].config).stored,close=(a,b)=>assert.ok(Math.abs(a-b)/b<.01,`${a} vs ${b}`);
 close(total('gpt2'),124.4e6);close(total('mistral'),7.24e9);close(total('llama3'),8.03e9);close(total('mixtral'),46.7e9);
 const mix=parameterFormulas(PF_PRESETS.mixtral.config);assert.ok(mix.active<mix.stored/3);assert.deepEqual(mix.blockUnits,{attn:2.5,mlp:84,mlpActive:21});
});
test('GQA and SwiGLU change the block formulas: 2d² + 2·d·d_kv and 3·d·f',()=>{
 const d=4096,r=parameterFormulas({d,L:1,fr:3.5,mlp:'swiglu',kvRatio:.25});
 assert.equal(part(r,'attn').count,2*d*d+2*d*1024);assert.equal(part(r,'mlp').count,3*d*14336);assert.equal(r.config.f,14336);
 const gpt=parameterFormulas(PF_PRESETS.gpt2.config);assert.equal(part(gpt,'positions').count,1024*768);assert.equal(part(gpt,'head').count,0);
});
