import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile,access} from 'node:fs/promises';
import {BUDGET_DEFAULTS,architectureBudget,budgetVariant,validateBudget} from '../src/architecture-budget.mjs';
const config=changes=>({...BUDGET_DEFAULTS,...changes});
test('ordinary and same-width SwiGLU match the 1:2 and 1:3 parameter splits',()=>{
 const b=architectureBudget(config()),{ordinary,swiglu}=b.variants;
 assert.equal(ordinary.attention,4194304);assert.equal(ordinary.expert,8388608);assert.equal(swiglu.expert,12582912);assert.equal(ordinary.mlpShare,2/3);assert.equal(swiglu.mlpShare,.75);
 assert.equal(ordinary.q+ordinary.key+ordinary.value+ordinary.o,ordinary.attention);
 assert.equal(ordinary.stored,ordinary.active);
});
test('GQA reduces K/V but not query, output, or expert matrices',()=>{
 const full=architectureBudget(config()),gqa=architectureBudget(config({kv:4}));
 assert.equal(gqa.variants.ordinary.attention,2*1024**2+2*1024*256);
 assert.equal(gqa.variants.ordinary.key,full.variants.ordinary.key/4);
 assert.equal(gqa.variants.ordinary.q,full.variants.ordinary.q);assert.equal(gqa.variants.swiglu.expert,full.variants.swiglu.expert);
 assert.equal(gqa.cacheBytes,full.cacheBytes/4);assert.equal(gqa.mixing,full.mixing);assert.equal(gqa.naiveScoreBytes,full.naiveScoreBytes);
});
test('head count is not double-counted in full MHA',()=>{
 const a=architectureBudget(config()),b=architectureBudget(config({h:8,kv:8}));assert.equal(a.variants.ordinary.attention,b.variants.ordinary.attention);assert.equal(a.cacheBytes,b.cacheBytes);
});
test('matched-budget width is rounded up to 64 and explicitly distinct from same-width MoE',()=>{
 const b=architectureBudget(config());assert.equal(b.variants.matched.f,2752);assert.ok(b.variants.matched.expert>=b.variants.ordinary.expert);assert.ok(b.variants.matched.expert-b.variants.ordinary.expert<3*1024*64);assert.equal(b.variants.moe.f,4096);
 const exact=architectureBudget(config({d:768,f:3072,h:12,kv:12}));assert.equal(exact.variants.matched.f,2048);assert.equal(exact.variants.matched.expert,exact.variants.ordinary.expert);
});
test('MoE separates total expert weights from per-token weights including the router',()=>{
 const a=architectureBudget(config()).variants.moe;assert.equal(a.ffnStored,8*12582912);assert.equal(a.ffnActive,2*12582912);assert.equal(a.router,8192);assert.equal(a.stored,a.attention+a.ffnStored+a.router);assert.equal(a.stackStored,a.stored*24);assert.equal(a.weightBytes,a.stackStored*2);
 const all=architectureBudget(config({k:8})).variants.moe;assert.equal(all.active,all.stored);
});
test('sequence length changes work/cache but not parameters; batch scales work linearly',()=>{
 const a=architectureBudget(config()),b=architectureBudget(config({n:4096})),c=architectureBudget(config({batch:2}));
 assert.equal(a.variants.moe.stored,b.variants.moe.stored);assert.equal(b.mixing,4*a.mixing);assert.equal(b.ffn,2*a.ffn);assert.equal(b.cacheBytes,2*a.cacheBytes);assert.equal(b.decodeMixingFlops,2*a.decodeMixingFlops);assert.equal(b.decodeLinearFlops,a.decodeLinearFlops);assert.equal(c.prefillFlops,2*a.prefillFlops);
});
test('cache and prefill/decode equations include batch, blocks, heads, and both K and V',()=>{
 const c=config({n:10,batch:3,layers:2,kv:4}),b=architectureBudget(c),a=b.variants.moe;
 assert.equal(b.cacheBytes,2*2*3*10*4*64*2);assert.equal(b.naiveScoreBytes,3*16*10*10*2);
 assert.equal(b.prefillFlops,2*(2*3*10*(a.attention+a.ffnActive+a.router)+4*3*100*1024));
 assert.equal(b.decodeLinearFlops,2*2*3*a.active);assert.equal(b.decodeMixingFlops,4*2*3*10*1024);
});
test('roofline is one GEMM and uses maximum lower bound, not summed times',()=>{
 const c=config({n:1,batch:1}),a=architectureBudget(c),g=a.gemm;
 assert.equal(g.flops,2*1024*4096);assert.equal(g.bytes,(1024+1024*4096+4096)*2);
 assert.equal(g.lowerBoundMs,Math.max(g.computeMs,g.memoryMs));assert.equal(g.bound,'memory bandwidth');
 const larger=architectureBudget(config({n:1,batch:128}));assert.ok(larger.gemm.intensity>g.intensity);assert.equal(larger.variants.moe.stored,a.variants.moe.stored);
});
test('precision changes memory only, not parameter/FLOP counts',()=>{
 const a=architectureBudget(config()),b=architectureBudget(config({bytes:1}));assert.equal(a.variants.moe.stored,b.variants.moe.stored);assert.equal(a.prefillFlops,b.prefillFlops);assert.equal(a.cacheBytes,b.cacheBytes*2);assert.equal(a.gemm.bytes,b.gemm.bytes*2);
});
test('invalid dimensions fail instead of yielding fractional heads or misleading counts',()=>{
 for(const bad of [{d:1025},{kv:3},{k:9},{n:0},{f:NaN},{bytes:3},{compute:0},{bandwidth:-5},{experts:1.5},{n:1e9}])assert.throws(()=>architectureBudget(config(bad)));
 assert.equal(validateBudget(config({k:1,experts:1})).experts,1);
 const before=config();architectureBudget(before);assert.deepEqual(before,BUDGET_DEFAULTS);
});
test('all valid extremes remain finite',()=>{
 const b=architectureBudget(config({d:16384,f:65536,layers:256,experts:256,k:256,n:131072,batch:256}));for(const value of [b.prefillFlops,b.cacheBytes,b.variants.moe.stackStored,b.gemm.lowerBoundMs])assert.ok(Number.isFinite(value));
});
test('separate analytical page is linked from all chapters and uses relative local assets',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),html=await read('analytics.html');
 for(const file of ['index.html','vision.html','moe.html','standalone.html'])assert.match(await read(file),/href="analytics.html"/);
 for(const [,path] of html.matchAll(/(?:href|src)="([^"#][^"]*)"/g))if(!/^(https?:|\.\/$)/.test(path)){assert.ok(!path.startsWith('/'));await access(new URL('../'+path,import.meta.url));}
 for(const id of ['calculator','design','work','gpu','sources','budget-form','budget-error','roofline-results'])assert.match(html,new RegExp(`id="${id}"`));
 assert.equal((html.match(/<form\b/g)||[]).length,(html.match(/<\/form>/g)||[]).length);
 assert.match(html,/not a benchmark/);assert.match(html,/not a GPU fit guarantee/);assert.match(await read('src/architecture-budget-app.mjs'),/full bidirectional square attention/);assert.match(html,/not a named GPU/);assert.match(html,/not wall-clock|Real latency/);
 for(const source of ['1706.03762','2002.05202','2305.13245','2401.04088','2205.14135','2211.15841','docs.nvidia.com'])assert.ok(html.includes(source));
});
