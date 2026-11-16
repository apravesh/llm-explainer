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
test('the page is two parts: weights on the Transformer-style diagram first, then the work',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8'),html=await read('analytics.html'),app=await read('src/param-formulas-app.mjs');
 assert.match(html,/<h1>Where do a model's weights <em>actually live\?<\/em><\/h1>/);
 assert.match(html,/01 \/ COUNT THE WEIGHTS/);assert.match(html,/02 \/ COUNT THE WORK/);
 assert.ok(html.indexOf('id="formula-models"')<html.indexOf('id="part-2"')&&html.indexOf('id="part-2"')<html.indexOf('id="calculator"'));
 assert.ok(html.indexOf('id="pf-symbols"')<html.indexOf('id="pf-diagram"'),'V, d and the other letters are defined before the diagram');assert.match(app,/vocabulary size: how many different tokens the model knows/);assert.match(app,/width: how many numbers describe one/);assert.match(app,/processed at once: no weights depend on it, only the work/);assert.ok(app.includes('id="pf-sym-n"'));
 for(const part of ["row('W_Q', 'd × d', c.d, c.d, 'query')","row('W_O', 'd × d', c.d, c.d, 'output')","row('W_in', 'f × d', c.f, c.d, 'widen')","row('W_out', 'd × f', c.d, c.f, 'narrow')",'pf-pop-sum'])assert.ok(app.includes(part),part);
 for(const id of ['pf-pop','pf-custom','pf-presets','pf-controls','pf-split'])assert.match(html,new RegExp(`id="${id}"`),id);
 assert.ok(html.indexOf('id="pf-custom"')<html.indexOf('data-pf-model="dense"'),'the dimension controls sit inside the collapsed section');
 assert.doesNotMatch(html,/Choose the budget you mean|class="takeaways"|class="design-grid"/,'no recap block and no card grid');
 assert.match(html,/What to notice/);assert.match(html,/href="moe.html">← MoE/);
 assert.match(app,/let preset = 'gpt2';/);assert.match(app,/function archFlow\(r, w = null\)/);assert.match(app,/function partCard\(r, key\)/);
 for(const part of ['TRANSFORMER BLOCK × ${c.L}','MULTI-HEAD SELF-ATTENTION','data-part="${keys[0]}"','LANGUAGE MODEL'])assert.ok(app.includes(part),part);
});
test('the figures quoted in "What to notice" follow from the formulas',()=>{
 const g=parameterFormulas(PF_PRESETS.gpt2.config),share=(r,cat)=>r.byCat[cat].stored/r.stored;
 assert.ok(Math.abs(g.parts.find(x=>x.key==='embed').count-38.6e6)<0.05e6);assert.ok(share(g,'embed')>0.30&&share(g,'embed')<0.34);
 assert.deepEqual([g.blockUnits.attn,g.blockUnits.mlp],[4,8]);
 assert.ok(share(parameterFormulas(PF_PRESETS.mistral.config),'embed')<0.05);
 const m=parameterFormulas(PF_PRESETS.mixtral.config);assert.ok(share(m,'mlp')>0.95);assert.equal(m.config.k/m.config.E,0.25);
 assert.equal(parameterFormulas(PF_PRESETS.pixtral.config).byCat.embed.stored,0);
});
test('work formulas: two operations per weight used, mixing without weights, and the cache makes writing cheap',async()=>{
 const {workFormulas,compactWork}=await import('../src/param-formulas.mjs');
 const g=PF_PRESETS.gpt2.config,w=workFormulas(g,{n:1024,mode:'read'}),part=(r,k)=>r.parts.find(x=>x.key===k);
 assert.equal(part(w,'embed').flops,0);
 assert.equal(part(w,'proj').flops,1024*12*2*4*768*768);assert.equal(part(w,'mlp').flops,1024*12*2*2*768*3072);
 assert.equal(part(w,'mix').flops,12*4*1024*1024*768);assert.equal(part(w,'head').flops,2*768*50257);
 assert.equal(w.total,w.parts.reduce((t,x)=>t+x.flops,0));
 const big=workFormulas(g,{n:8192,mode:'read'});
 assert.equal(part(big,'mlp').flops/part(w,'mlp').flops,8);assert.equal(part(big,'mix').flops/part(w,'mix').flops,64);
 assert.ok(big.byCat.mix/big.total>0.5,'mixing is most of the work on a long prompt');assert.ok(w.byCat.embed/w.total<0.01);
 const one=workFormulas(g,{n:1024,mode:'write'});
 assert.equal(part(one,'proj').flops,12*2*4*768*768);assert.equal(part(one,'mix').flops,12*4*1024*768);assert.ok(one.total<w.total/500);
 const m=workFormulas(PF_PRESETS.mixtral.config,{n:1024}),mx=part(m,'experts').flops;
 assert.equal(mx,1024*32*2*2*3*4096*14336,'only the k experts that run are paid for');assert.ok(part(m,'router').flops<mx/1000);
 const v=workFormulas(PF_PRESETS.pixtral.config,{n:48,mode:'write'});assert.equal(v.mode,'read');assert.ok(!part(v,'head')&&part(v,'proj-out').flops>0);
 assert.equal(compactWork(213e9),'213 billion');assert.equal(compactWork(3.87e12),'3.87 trillion');assert.equal(compactWork(0),'0');
});
test('Compute opens with the same diagram counting work, for the model picked in Capacity',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8'),html=await read('analytics.html'),app=await read('src/param-formulas-app.mjs');
 for(const id of ['pw-controls','pw-diagram','pw-bars','pw-insight'])assert.match(html,new RegExp(`id="${id}"`),id);
 assert.ok(html.indexOf('id="pw-diagram"')>html.indexOf('id="part-2"')&&html.indexOf('id="pw-diagram"')<html.indexOf('id="calculator"'));
 for(const part of ['Attention · mixing (no weights)','Writing is cheap per token because of the cache','Weights that are stored are not always weights that work'])assert.ok(html.includes(part),part);
 assert.match(app,/function archFlow\(r, w = null\)/);assert.match(app,/function workCard\(w, key\)/);assert.match(app,/renderWork\(\);\n\}/);
 for(const part of ['Read a prompt','Write one more token','2 · Mixing (no weights at all)','Model from Capacity'])assert.ok(app.includes(part),part);
});
test('weights that are shared are drawn once: one MLP, one MoE layer, one projector for all tokens',async()=>{
 const app=await readFile(new URL('../src/param-formulas-app.mjs',import.meta.url),'utf8');
 for(const part of ['one MLP · shared by every','one MLP · runs once per','one router and ${c.E} experts, shared by every token','each token runs its own ${c.k} experts','one shared projector'])assert.ok(app.includes(part),part);
 assert.equal((app.match(/class="pf-wide"/g)||[]).length,1,'the MLP shape is not drawn per token row');
});
test('the weights diagram follows a single token; the work diagram shows several',async()=>{
 const app=await readFile(new URL('../src/param-formulas-app.mjs',import.meta.url),'utf8'),html=await readFile(new URL('../analytics.html',import.meta.url),'utf8');
 assert.match(app,/rows = w \? \[128, 196, 264\] : \[196\]/);
 assert.match(html,/reduced to a single token, because every token goes through the same weights/);assert.match(html,/now with several tokens/);
});
