import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {EXAMPLES} from '../src/text-data.mjs';
import {softmax,attentionStages,probabilities,sample,showToken,illustrativeVector,HEAD_DIM} from '../src/text-engine.mjs';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
const close=(a,b,eps=1e-3)=>assert.ok(Math.abs(a-b)<eps,`${a} vs ${b}`);
test('recorded data covers five prompts, 12 blocks and 12 heads',()=>{
 assert.equal(EXAMPLES.length,5);
 for(const ex of EXAMPLES){
  const n=ex.tokens.length;assert.equal(ex.tokenIds.length,n);assert.equal(ex.tokens.map(t=>t).join(''),ex.prompt);
  assert.equal(ex.top.length,50);for(let i=1;i<50;i++)assert.ok(ex.top[i-1][2]>=ex.top[i][2]);
  assert.equal(ex.attn.length,12);for(const b of ex.attn){assert.equal(b.length,12);for(const h of b){assert.equal(h.length,n);for(const r of h)assert.equal(r.length,n);}}
 }
 assert.deepEqual(EXAMPLES[0].tokenIds,[6601,32704,795,30132,2985,284]);
});
test('attention stages: scale by sqrt(64), mask the future, rows sum to 1',()=>{
 assert.equal(HEAD_DIM,64);
 const raw=EXAMPLES[0].attn[3][5],{scaled,masked,weights}=attentionStages(raw);
 close(scaled[2][0],raw[2][0]/8);assert.equal(masked[2][3],-Infinity);assert.equal(masked[2][2],scaled[2][2]);
 for(const [i,row] of weights.entries()){close(row.reduce((a,b)=>a+b,0),1);for(let j=i+1;j<row.length;j++)assert.equal(row[j],0);}
 // matches the softmax row the original recording stores for block 4, head 6, token 3
 close(weights[2][0],0.8251,2e-3);close(weights[2][1],0.1451,2e-3);close(weights[2][2],0.0298,2e-3);
 assert.equal(weights[0][0],1);
});
test('temperature and top-k reproduce the original page\'s default numbers',()=>{
 const rows=probabilities(EXAMPLES[0].top,{temperature:0.8,sampling:{type:'top-k',value:5}});
 assert.deepEqual(rows.filter(r=>r.kept).map(r=>showToken(r.token)),['visualize','create','see','make','easily']);
 close(rows[0].scaled,-169.891,2e-3);close(rows[0].probability,0.5467,1e-3);close(rows[1].probability,0.2087,1e-3);close(rows[3].probability,0.0626,1e-3);
 assert.equal(rows[5].probability,0);close(rows.reduce((a,r)=>a+r.probability,0),1,1e-9);
});
test('temperature sharpens or flattens; top-p keeps the smallest set reaching p',()=>{
 const p=t=>probabilities(EXAMPLES[1].top,{temperature:t,sampling:{type:'top-k',value:10}})[0].probability;
 assert.ok(p(0.3)>p(1)&&p(1)>p(2));
 const rows=probabilities(EXAMPLES[1].top,{temperature:1,sampling:{type:'top-p',value:0.5}}),kept=rows.filter(r=>r.kept);
 assert.ok(kept.length>=1&&kept.length<50);assert.deepEqual(kept.map(r=>r.rank),kept.map((_,i)=>i));
 const all=softmax(rows.map(r=>r.scaled));const mass=kept.reduce((a,r)=>a+all[r.rank],0);assert.ok(mass>=0.5);assert.ok(mass-all[kept.at(-1).rank]<0.5);
 assert.equal(probabilities(EXAMPLES[1].top,{temperature:1,sampling:{type:'top-p',value:1}}).filter(r=>r.kept).length,50);
 assert.equal(probabilities(EXAMPLES[1].top,{temperature:1,sampling:{type:'top-k',value:1}})[0].probability,1);
});
test('sampling only returns kept tokens and follows the random number',()=>{
 const rows=probabilities(EXAMPLES[2].top,{temperature:0.8,sampling:{type:'top-k',value:5}});
 assert.equal(sample(rows,0).rank,0);assert.equal(sample(rows,0.999999).rank,4);
 for(let i=0;i<50;i++)assert.ok(sample(rows,i/50).kept);
});
test('illustrative vectors are deterministic and bounded',()=>{
 assert.deepEqual(illustrativeVector(6601),illustrativeVector(6601));assert.notDeepEqual(illustrativeVector(6601),illustrativeVector(6602));
 for(const x of illustrativeVector(42,40))assert.ok(x>=-1&&x<=1);
});
test('text page: structure, honesty labels, credit and links',async()=>{
 const html=await read('transformer.html');
 for(const id of ['explorer','embedding','attention','mlp','output','tx-flow','tx-example','tx-generate','attn-lab','prob-table'])assert.match(html,new RegExp(`id="${id}"`));
 assert.match(html,/src="src\/text-app\.mjs"/);assert.match(html,/<a href="transformer\.html" aria-current="page">Transformer<\/a>/);
 assert.match(html,/Recorded from GPT-2/);assert.match(html,/does not run a model/i);assert.match(html,/illustrative/i);
 assert.match(html,/Transformer Explainer/);assert.match(html,/Polo Club/);assert.match(html,/MIT/);
 assert.doesNotMatch(html,/onnx|model-v2/i);
});
test('the Embedding section is the original article text with its figure and credit',async()=>{
 const html=await read('transformer.html'),sec=html.slice(html.indexOf('<section id="embedding"'),html.indexOf('<section id="attention"'));
 for(const part of ['Step 1: Tokenization','Step 2. Token Embedding','Step 3. Positional Encoding','Step 4. Final Embedding','Figure 1. Expanding the Embedding layer view','approximately 39 million parameters','assets/transformer-explainer/embedding.png','MIT licence'])assert.ok(sec.includes(part),part);
 assert.ok((await (await import('node:fs/promises')).stat(new URL('../assets/transformer-explainer/embedding.png',import.meta.url))).size>10000);
 assert.doesNotMatch(sec,/emb-lab/);
});
test('attention and MLP sections explain each step around their diagrams',async()=>{
 const html=await read('transformer.html'),att=html.slice(html.indexOf('<section id="attention"'),html.indexOf('<section id="mlp"')),mlp=html.slice(html.indexOf('<section id="mlp"'),html.indexOf('<section id="output"'));
 for(const part of ['Step 1 · Query, Key and Value','Step 2 · Twelve heads','Step 3 · Masked attention','Step 4 · Mix the Values','search bar','√64 = 8','must not peek ahead','id="attn-lab"','id="attn-stages"','qkv-lab'])assert.ok(att.includes(part),part);
 for(const part of ['Widen: 768 → 3,072','GELU: keep what fired','Narrow: 3,072 → 768','mlp-lab','4.7 million','residual connection','href="analytics.html"'])assert.ok(mlp.includes(part),part);
 // 2·768·3072 + 3072 + 768 weights and biases in the MLP; 4·768² + 4·768 in attention
 assert.equal(2*768*3072+3072+768,4722432);assert.equal(4*768*768+4*768,2362368);
});
test('temperature, sampling and Generate repaint only the probabilities',async()=>{
 const app=await read('src/text-app.mjs');
 assert.match(app,/tx-generate'\)\.addEventListener\('click',\(\)=>\{state\.rnd=Math\.random\(\);refreshProbs\(\);\}\)/);
 const input=app.slice(app.indexOf("document.addEventListener('input'"),app.indexOf("document.addEventListener('click'"));
 assert.match(input,/refreshProbs\(\);/);assert.doesNotMatch(input,/render\(/);
 const paint=app.slice(app.indexOf('function paintProbs'),app.indexOf('function refreshProbs'));
 assert.doesNotMatch(paint,/innerHTML=s|flow\(\)/,'the diagram is not rebuilt');
});
test('intro and text pages end without a recap block',async()=>{
 for(const page of ['what-is-an-llm.html','transformer.html']){const html=await read(page);assert.doesNotMatch(html,/TO REMEMBER|takeaways-intro|class="intro-cards"/);assert.match(html,/<section class="lesson-section notes-section"><details class="further">/);}
});
test('explorer is full width, and a block or head change redraws only that part',async()=>{
 const app=await read('src/text-app.mjs'),css=await read('transformer.css');
 assert.match(css,/\.lesson-section>\.tx-lab\{margin-inline:calc\(50% - 50vw\)/);
 assert.match(app,/function swap\(kind,dir\)/);assert.match(app,/swap\(kind,dir\);return;/);
 assert.doesNotMatch(app.slice(app.indexOf('function swap(kind,dir)'),app.indexOf('/* ---------- wiring')),/render\(|flow\(\)/,'swap must not rebuild the diagram');
 assert.match(css,/\.swap\.head-part\{animation:tx-head-in/);assert.match(css,/#tx-flow \.frame\{[^}]*stroke-dasharray/);assert.match(app,/chip\('block',/);assert.match(app,/chip\('head',/);assert.doesNotMatch(await read('transformer.html'),/id="tx-lab">[\s\S]*?class="tx-steppers"[\s\S]*?id="tx-result"/);assert.match(css,/#tx-flow \.cell\{transition:fill-opacity/);
});
test('every page header and the home page link to the text chapter',async()=>{
 for(const page of ['index.html','what-is-an-llm.html','llm-use-cases.html','vision.html','moe.html','analytics.html','pretraining.html','post-training-instruct.html','post-training-rl.html','transformer.html'])
  assert.match(await read(page),/<span class="nav-group">Architecture<\/span><a href="transformer\.html"[^>]*>Transformer<\/a><a href="vision\.html"/);
 assert.match(await read('index.html'),/class="chapter-link" href="transformer\.html"/);
});
