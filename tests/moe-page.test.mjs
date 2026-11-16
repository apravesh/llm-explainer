import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile,access} from 'node:fs/promises';
import {BLOCKS,HEADS,WIDTH,HEAD_DIM,FF,EXPERTS,TOP_K,MIXTRAL,STAND_IN,EXAMPLES,createModel,run,attentionRaw,attentionStages,layerCost} from '../src/moe-page-engine.mjs';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
const near=(a,b,eps=1e-9)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);

test('sizes shown are Mixtral 8x7B\'s; the numbers come from a stand-in with the same layout',()=>{
 assert.deepEqual([BLOCKS,HEADS,WIDTH,HEAD_DIM,FF,EXPERTS,TOP_K],[32,32,4096,128,14336,8,2]);
 assert.deepEqual([BLOCKS,HEADS,WIDTH,FF,EXPERTS,TOP_K],[MIXTRAL.blocks,MIXTRAL.heads,MIXTRAL.width,MIXTRAL.ff,MIXTRAL.experts,MIXTRAL.topK]);
 assert.deepEqual([STAND_IN.blocks,STAND_IN.heads,STAND_IN.experts,STAND_IN.topK],[BLOCKS,HEADS,EXPERTS,TOP_K],'same layout');
 assert.ok(STAND_IN.width<WIDTH&&STAND_IN.width===STAND_IN.heads*STAND_IN.headDim);
 const m=createModel();assert.deepEqual(m,createModel(),'deterministic');assert.equal(m.blocks.length,32);
 for(const b of m.blocks.slice(0,3)){assert.equal(b.ffn.router.length,STAND_IN.width);assert.equal(b.ffn.router[0].length,8);assert.equal(b.ffn.experts.length,8);for(const e of b.ffn.experts){assert.equal(e.up[0].length,STAND_IN.ff);assert.equal(e.gate[0].length,STAND_IN.ff);assert.equal(e.down.length,STAND_IN.ff);assert.equal(e.down[0].length,STAND_IN.width);}}
 assert.notDeepEqual(m.blocks[0].ffn.router,m.blocks[1].ffn.router);assert.notDeepEqual(m.blocks[0].ffn.experts[0],m.blocks[0].ffn.experts[1]);
});
test('three six-token examples with an illustrative next-token list',()=>{
 assert.equal(EXAMPLES.length,3);
 for(const ex of EXAMPLES){assert.equal(run(ex.text).tokens.length,6);assert.equal(ex.next.length,8);const total=ex.next.reduce((a,[,p])=>a+p,0);assert.ok(total>0.5&&total<=1);for(let i=1;i<ex.next.length;i++)assert.ok(ex.next[i-1][1]>=ex.next[i][1]);}
});
test('top-k really selects k experts, the highest-scoring ones, and only those are evaluated',()=>{
 for(const ex of EXAMPLES)for(const b of run(ex.text).blocks)for(const r of b.routes){
  assert.equal(r.scores.length,EXPERTS);assert.equal(r.chosen.length,TOP_K);
  const best=r.scores.map((s,e)=>[s,e]).sort((a,b)=>b[0]-a[0]).slice(0,TOP_K).map(x=>x[1]).sort((a,b)=>a-b);assert.deepEqual(r.chosen,best);
  r.experts.forEach((e,i)=>assert.equal(e!==null,r.chosen.includes(i),'only chosen experts run'));
 }
});
test('gate weights are a softmax over the chosen scores and sum to 1; the output is their weighted sum plus the residual',()=>{
 for(const b of run(EXAMPLES[0].text).blocks)for(const r of b.routes){
  near(r.gates.reduce((a,g)=>a+g,0),1);
  const z=r.chosen.reduce((a,e)=>a+Math.exp(r.scores[e]),0);
  r.gates.forEach((g,e)=>r.chosen.includes(e)?near(g,Math.exp(r.scores[e])/z):assert.equal(g,0));
  for(let d=0;d<STAND_IN.width;d++){const mix=r.chosen.reduce((a,e)=>a+r.gates[e]*r.experts[e].output[d],0);near(r.mixed[d],mix);near(r.output[d],r.input[d]+mix);}
 }
});
test('different blocks route differently; loads count every expert run; results are deterministic',()=>{
 for(const ex of EXAMPLES){
  const {blocks}=run(ex.text),routes=blocks.map(b=>b.routes.map(r=>r.chosen.join('')).join(' '));
  assert.ok(new Set(routes).size>=BLOCKS-1,'blocks route differently');
  for(const b of blocks){assert.equal(b.loads.length,EXPERTS);assert.equal(b.loads.reduce((a,v)=>a+v,0),6*TOP_K);}
  assert.strictEqual(run(ex.text),run(ex.text));
 }
 assert.notDeepEqual(run(EXAMPLES[0].text).blocks[0].routes.map(r=>r.chosen),run(EXAMPLES[1].text).blocks[0].routes.map(r=>r.chosen));
});
test('attention is the ordinary causal attention: scale by √128, mask later tokens, rows sum to 1',()=>{
 const {blocks}=run(EXAMPLES[0].text);
 for(const [b,h] of [[0,0],[1,1],[31,31]]){
  const raw=attentionRaw(blocks[b].attention,h),{scaled,weights}=attentionStages(raw);
  for(let i=0;i<6;i++){near(weights[i].reduce((a,v)=>a+v,0),1);for(let j=0;j<6;j++){assert.ok(Number.isFinite(raw[i][j]));near(scaled[i][j],raw[i][j]/Math.sqrt(128));if(j>i)assert.equal(weights[i][j],0);}}
 }
 assert.notDeepEqual(attentionRaw(blocks[0].attention,0),attentionRaw(blocks[0].attention,1),'heads differ');
});
test('stored and used weight counts are Mixtral\'s: three matrices per gated expert',()=>{
 const c=layerCost();
 assert.equal(c.expert,3*4096*14336);assert.equal(c.router,8*4096);assert.equal(c.stored,c.router+8*c.expert);assert.equal(c.active,c.router+2*c.expert);assert.equal(c.dense,c.expert);
 assert.ok(Math.abs(c.expert-176e6)<1e6,'about 176 million per expert, as the page says');
 // 32 such layers hold most of Mixtral's published 46.7 billion weights
 assert.ok(32*c.stored>0.95*MIXTRAL.stored&&32*c.stored<MIXTRAL.stored);
 assert.deepEqual([MIXTRAL.blocks,MIXTRAL.heads,MIXTRAL.width,MIXTRAL.ff,MIXTRAL.experts,MIXTRAL.topK,MIXTRAL.stored,MIXTRAL.active],[32,32,4096,14336,8,2,46.7e9,12.9e9]);
});
test('the page mirrors the text page and names what changes',async()=>{
 const html=await read('moe.html'),app=await read('src/moe-page-app.mjs'),css=await read('moe-page.css');
 for(const id of ['explorer','tx-lab','tx-example','tx-sentence','tx-flow','tx-tip','tx-formula','tx-result','differences','router','moe-math','top-k','who-lab','dispatch-table','expert-loads','experts','cost','cost-comparison','moe-training','train-compare','tl-plane','tl-bars','tl-stepbtn','tl-balance'])assert.match(html,new RegExp(`id="${id}"`),id);
 assert.match(html,/<link rel="stylesheet" href="training.css"><link rel="stylesheet" href="intro.css"><link rel="stylesheet" href="transformer.css"><link rel="stylesheet" href="moe-page.css"><link rel="stylesheet" href="site-header.css">/);
 assert.match(html,/src="src\/moe-page-app.mjs"/);assert.match(html,/src="src\/moe-training-app.mjs"/);
 assert.doesNotMatch(html,/moe\.css|src\/moe-app\.mjs|vision\.css/,'the earlier page\'s app and stylesheet are not loaded');
 assert.match(html,/ARCHITECTURE \/ 03 · MIXTURE OF EXPERTS/);assert.match(html,/<h1>How does a transformer use <em>only some of its weights for each token\?<\/em><\/h1>/);
 for(const part of ['a router picks experts','only the top-k experts run','more weights stored, same work per token','What changes from the text transformer','One MLP per block, shared by every token','A pool of expert MLPs per block','Mixtral 8x7B: 32 blocks · 32 heads · width 4,096 · 8 experts per block, 2 run per token','01 / THE ROUTER','02 / TOP-K','03 / EXPERTS','04 / STORED VS USED','05 / TRAINING','Who gets the work?','much smaller untrained stand-in with the same layout','The sizes shown are those of a real model, Mixtral 8x7B','The list of next tokens is illustrative','never trained'])assert.ok(html.includes(part),part);
 assert.equal((html.match(/class="mx-diff-row"/g)||[]).length,7);assert.equal((html.match(/class="mx-new">different</g)||[]).length,4);
 assert.equal((html.match(/<span role="cell" class="same">Unchanged<\/span>/g)||[]).length,2,'attention and output are unchanged');
 assert.doesNotMatch(html,/THREE IDEAS TO KEEP|lesson-grid|to remember/i,'no recap block');
 assert.match(html,/<details class="further"><summary>Where do the numbers on this page come from\?/);
 assert.match(html,/<a href="moe\.html" aria-current="page">MoE<\/a>/);assert.match(html,/<a href="transformer\.html">text page<\/a>/);
 assert.match(html,/<footer><a href="vision\.html">← Vision<\/a><span>LLM EXPLAINER · Mixture of experts<\/span><a href="analytics\.html">Parameters →<\/a><\/footer>/);
 for(const zone of ['attention','moe'])assert.ok(app.includes(`data-zone="${zone}"`),zone);
 assert.match(app,/const STAGE_NAMES=\['Dot product','Scaling · Mask','Softmax'\];/);
 assert.match(app,/const ROUTE_NAMES=\['Router scores',`Top-\$\{TOP_K\}`,'Softmax'\];/);
 for(const part of ['Experts run','Weighted sum','MoE layer','ATTENTION · UNCHANGED','UNCHANGED · ILLUSTRATIVE','class="ex run"','class="ex-col"','function setExpand(','function swap(','state.expand','xdim','data-dim=','late('])assert.ok(app.includes(part),part);
 assert.match(app,/chip\('block',fx\+18,fy,blockLabel\(\),176\)/);assert.match(app,/chip\('head',346,top,headLabel\(\),282\)/);
 assert.equal((app.match(/`<h4>/g)||[]).length,11,'eleven formula cards');
 for(const h of ['Router scores','Softmax over the chosen','Experts run','Weighted sum'])assert.ok(app.includes(`<h4>${h}</h4>`),h);
 assert.doesNotMatch(app,/temperature|top-p/i,'no sampling controls');
 for(const rule of ['.mx-chips','.mx-diff-row','.mx-new','.rcell','.ex.idle','.train-compare','.tl-bar','.mx-who-grid','.mx-pool'])assert.ok(css.includes(rule),rule);
});
test('every local link on the page resolves',async()=>{
 const html=await read('moe.html');
 for(const [,href] of html.matchAll(/(?:href|src)="([^"]+)"/g))if(!/^(https?:|#|\.\/$)/.test(href)){assert.ok(!href.startsWith('/'),href);await access(new URL('../'+href.split('#')[0],import.meta.url));}
 for(const [,page,id] of html.matchAll(/href="([a-z-]+\.html)#([a-z-]+)"/g))assert.match(await read(page),new RegExp(`id="${id}"`),`${page}#${id}`);
});
test('the earlier live-toy page is kept, unlinked, as moe-live.html',async()=>{
 const live=await read('moe-live.html');
 assert.match(live,/src="src\/moe-app.mjs"/);assert.match(live,/href="moe.css"/);assert.match(live,/id="moe-explorer"/);
 for(const page of ['index.html','moe.html','vision.html','transformer.html','analytics.html'])assert.doesNotMatch(await read(page),/moe-live\.html/,page);
});

test('src/moe-page-app.mjs: weights that exist once are drawn once (one Q/K/V block, one router and expert pool)',async()=>{
 const app=await (await import('node:fs/promises')).readFile(new URL('../src/moe-page-app.mjs',import.meta.url),'utf8');
 assert.match(app,/class="wblock"/);assert.equal((app.match(/class="wblock"/g)||[]).length,1);
 assert.match(app,/the same for every (token|patch)/);assert.doesNotMatch(app,/curve\(142,y,232,y\+dy\)/,'no per-token Q, K, V squares');
 assert.match(app,/class="ex-col"/);assert.match(app,/one router W_r, the same for every token/);
});
