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
test('attention grids play as steps: multiply, scale, mask, then softmax row by row',async()=>{
 const app=await read('src/text-app.mjs'),css=await read('transformer.css'),html=await read('transformer.html');
 assert.match(html,/id="attn-play"/);for(const f of ['Q · K<sup>T</sup></span>','÷ √d<sub>k</sub>','e<sup>x</sup> ÷ Σ e<sup>x</sup>'])assert.ok(app.includes(f),f);assert.match(app,/const ATTN_STEPS=\[/);
 const steps=[...app.slice(app.indexOf('const ATTN_STEPS'),app.indexOf('function playAttention')).matchAll(/\[(\d+),(\d),'(show|mask)'/g)].map(m=>[+m[1],+m[2],m[3]]);
 assert.deepEqual(steps.map(s=>s.slice(1)),[[1,'show'],[2,'show'],[2,'mask'],[3,'show']]);
 for(let i=1;i<steps.length;i++)assert.ok(steps[i][0]-steps[i-1][0]>=1200,'each step has time to be read');
 assert.match(app,/class="was">\$\{scaled\[i\]\[j\]\.toFixed\(2\)\}<\/b><b class="now">−∞/,'future cells show their score before the mask');
 assert.match(app,/prefers-reduced-motion/);assert.match(app,/IntersectionObserver/);assert.match(app,/if\(c&&!attnBusy\)attnResult/);
 assert.match(css,/\.attn-stage\.mask \.c\.masked \.was\{display:none\}/);assert.match(css,/transition-delay:calc\(var\(--i\)\*26ms\)/);
});
test('attention expands in place into three stages, each with a formula on hover',async()=>{
 const app=await read('src/text-app.mjs'),css=await read('transformer.css'),html=await read('transformer.html');
 assert.match(app,/const STAGE_NAMES=\['Dot product','Scaling · Mask','Softmax · Dropout'\]/);
 assert.match(app,/data-expand="attention"/);assert.match(app,/function setExpand\(which\)/);
 assert.match(app,/if\(state\.expand\)return;\n  const el=e\.target\.closest\('\[data-t\]'\)/,'no row tracing while the stages are open');assert.match(app,/cell\.classList\.add\('hot'\)/);
 assert.doesNotMatch(app,/ATTENTION ⊕|CLOSE ×/,'no open/close buttons: the head itself is the target');
 assert.match(app,/if\(!state\.expand&&zone\)\{setExpand\(zone\);return;\}/);assert.match(app,/if\(state\.expand&&zone!==state\.expand&&!e\.target\.closest\('\.chip'\)\)\{setExpand\(null\);return;\}/);
 // opening timeline: lines draw, dots pop one by one, each later stage starts as the previous picture and turns into its own
 assert.match(app,/T1=t0\+QK\/n\+QK\+500,T2=T1\+500\+1000\+150,T3=T2\+450\+1000/);
 assert.match(app,/pathLength="1" class="draw"/);assert.match(app,/ pop" style="--at:/);assert.match(app,/look\(m-1,i,j\),b=look\(m,i,j\)/);
 assert.match(app,/class="fv\$\{opening\?' late':''\}"[^`]*H\$\{outX-74\}/,'Values stay connected to Out when the stages are open');
 for(const k of ['tx-draw','tx-pop','tx-turn','tx-grow'])assert.match(css,new RegExp(`@keyframes ${k}`));
 assert.match(css,/\.cell\.turn\{animation:tx-turn 1s ease-in-out backwards/,'colour changes slowly and then hands back to the live values');assert.match(app,/e\.key==='Escape'&&state\.expand/);
 const formulas=app.slice(app.indexOf('const FORMULAS=['),app.indexOf('function flow(){'));
 for(const part of ['Q<sub>i</sub> · K<sub>j</sub>','÷ √64','−∞','exp(scaled<sub>i,j</sub>)','Dropout randomly zeroes','weight<sub>i,j</sub> × V<sub>j</sub>','W<sub>2</sub> · GELU(W<sub>1</sub> · x + b<sub>1</sub>) + b<sub>2</sub>','logits = W<sub>out</sub> · h<sub>last</sub>','logit<sub>i</sub> ÷ T','keep the k highest scores','p<sub>i</sub> ≥ p'])assert.ok(formulas.includes(part),part);
 assert.match(app,/class="stage-name" data-formula="\$\{m\}"/);assert.match(app,/class="stage-name out-name" data-formula="3"/);assert.match(app,/class="stage-name mlp-name" data-formula="4"/);assert.match(app,/class="stage-name fan-hit" data-formula="5"/);for(const k of [6,7,8])assert.match(app,new RegExp(`has-formula\\\\?" data-formula=\\\\?"${k}`));assert.match(app,/closest\?\.\('\.tx-controls \.has-formula'\)/);assert.match(app,/formula\.innerHTML=FORMULAS\[\+name\.dataset\.formula\]/);
 assert.match(html,/id="tx-formula" role="tooltip" hidden/);assert.match(html,/id="tx-tip" hidden/);
 assert.match(css,/@keyframes tx-grow/);assert.match(css,/#tx-flow \.xdim\{opacity:\.12/);
});
test('output section explains logits, temperature and the two filters around its table',async()=>{
 const html=await read('transformer.html'),out=html.slice(html.indexOf('<section id="output"'),html.indexOf('</section>',html.indexOf('<section id="output"')));
 assert.match(out,/<h2>Output probabilities<\/h2>/);
 for(const part of ['From vector to logits to probabilities','50,257','<code>logits</code>','<h3>Temperature</h3>','Temperature = 1.','Temperature below 1.','Temperature above 1.','<h3>Top-k and top-p</h3>','id="prob-table"','id="prob-card"','data-controls'])assert.ok(out.includes(part),part);
});
test('embedding and probabilities also open in place, one part at a time',async()=>{
 const app=await read('src/text-app.mjs'),css=await read('transformer.css');
 for(const zone of ['embedding','attention','probabilities'])assert.match(app,new RegExp(`data-zone="${zone}"`));
 assert.match(app,/const X=state\.expand,E=X==='attention',EE=X==='embedding',EP=X==='probabilities'/);
 for(const part of ['Tokenization','Token embedding','Positional encoding','÷ temperature','Softmax','W<sub>E</sub>[ id<sub>i</sub> ] + W<sub>P</sub>[ i ]','exp(s<sub>i</sub>)'])assert.ok(app.includes(part),part);
 assert.match(app,/class="pl"[^`]*r\.logit\.toFixed\(2\)/);assert.match(app,/ps\.textContent=r\.scaled\.toFixed\(2\);g\.querySelector\('\.pk'\)\.textContent=r\.kept\?r\.scaled\.toFixed\(2\):'−∞'/);
 assert.match(app,/row&&state\.expand==='probabilities'/);assert.match(css,/#tx-flow:not\(\.expanded\) \[data-zone\]\{cursor:zoom-in\}/);
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
 assert.doesNotMatch(app.slice(app.indexOf('function swap(kind,dir)'),app.indexOf('// Open or close the three stages')),/render\(|flow\(\)/,'swap must not rebuild the diagram');
 assert.match(css,/\.swap\.head-part\{animation:tx-head-in/);assert.match(css,/tx-head-in \.95s/);assert.match(css,/@keyframes tx-block-in\{0%\{transform:scaleX\(0\)/);assert.match(css,/@keyframes tx-block-out\{[^}]*\}[^}]*\}100%\{transform:scaleX\(0\)/);assert.match(css,/\[data-dir=next\] \.ghost\{transform-origin:var\(--oxL\)/);assert.match(css,/\[data-dir=next\] \.swap:is\(\.head-part,\.block-part\)\{transform-origin:var\(--oxR\)/);assert.match(css,/#tx-flow \.frame\{[^}]*stroke-dasharray/);assert.match(app,/chip\('block',/);assert.match(app,/chip\('head',/);assert.doesNotMatch(await read('transformer.html'),/id="tx-lab">[\s\S]*?class="tx-steppers"[\s\S]*?id="tx-result"/);assert.match(css,/#tx-flow \.cell\{transition:fill-opacity/);
});
test('every page header and the home page link to the text chapter',async()=>{
 for(const page of ['index.html','what-is-an-llm.html','llm-use-cases.html','vision.html','moe.html','analytics.html','pretraining.html','post-training-instruct.html','post-training-rl.html','transformer.html'])
  assert.match(await read(page),/<span class="nav-group">Architecture<\/span><a href="transformer\.html"[^>]*>Transformer<\/a><a href="vision\.html"/);
 assert.match(await read('index.html'),/class="chapter-link" href="transformer\.html"/);
});
