import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {IMAGES} from '../src/vision-data.mjs';
import {BLOCKS,HEADS,WIDTH,HEAD_DIM,FF,LLM_WIDTH,COLS,ROWS,PATCHES,PIXELS,SHOWN,windowStart,patchPlace,patchName,attentionRaw,attentionStages} from '../src/vision-engine.mjs';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');

test('dimensions are those of Pixtral\'s vision encoder',()=>{
 assert.deepEqual([BLOCKS,HEADS,WIDTH,HEAD_DIM,FF,LLM_WIDTH],[24,16,1024,64,4096,5120]);
 assert.equal(WIDTH/HEADS,HEAD_DIM);assert.equal(COLS*ROWS,PATCHES);assert.equal(PATCHES,48);assert.equal(PIXELS,768);
});
test('three sample images, each measured as 48 patches',()=>{
 assert.deepEqual(IMAGES.map(i=>i.id),['bird','coast','poster']);
 for(const img of IMAGES){
  assert.equal(img.mean.length,48);assert.equal(img.stripe.length,48);assert.ok(img.sel>=0&&img.sel<48);
  for(const c of img.mean){assert.equal(c.length,3);for(const v of c)assert.ok(v>=0&&v<=255);}
  for(const s of img.stripe){assert.equal(s.length,10);for(const v of s)assert.ok(v>=0&&v<=1);}
 }
});
test('the six rows are a window of the patch sequence that holds the selected patch',()=>{
 for(let sel=0;sel<PATCHES;sel++){const s=windowStart(sel);assert.ok(s>=0&&s+SHOWN<=PATCHES);assert.ok(sel>=s&&sel<s+SHOWN);}
 assert.deepEqual(patchPlace(21),{row:2,col:5});assert.equal(patchName(0),'p1');
});
test('attention has no mask: every pair gets a finite score and every row sums to 1',()=>{
 const img=IMAGES[0],ps=[18,19,20,21,22,23];
 for(const [b,h] of [[0,0],[5,7],[23,15]]){
  const raw=attentionRaw(img,b,h,ps),{scaled,weights}=attentionStages(raw);
  assert.equal(raw.length,6);
  for(let i=0;i<6;i++){
   for(let j=0;j<6;j++){assert.ok(Number.isFinite(raw[i][j]));assert.ok(Math.abs(scaled[i][j]-raw[i][j]/8)<1e-9);assert.ok(weights[i][j]>0,'a later patch is not masked');}
   assert.ok(Math.abs(weights[i].reduce((a,b)=>a+b,0)-1)<1e-9);
  }
  assert.deepEqual(attentionRaw(img,b,h,ps),raw,'deterministic');
 }
 assert.notDeepEqual(attentionRaw(img,0,0,ps),attentionRaw(img,0,1,ps),'heads differ');
 assert.notDeepEqual(attentionRaw(img,0,0,ps),attentionRaw(img,12,0,ps),'blocks differ');
});
test('the page mirrors the text page and names what changes',async()=>{
 const html=await read('vision.html'),app=await read('src/vision-app.mjs'),css=await read('vision.css');
 for(const id of ['explorer','tx-lab','tx-flow','tx-tip','tx-formula','tx-result','differences','embedding','attention','attn-math','attn-lab','attn-stages','rope-lab','rope-demo','mlp','mlp-math','output','comparison','packing-grid'])assert.match(html,new RegExp(`id="${id}"`),id);
 assert.match(html,/<link rel="stylesheet" href="transformer.css"><link rel="stylesheet" href="vision.css">/);
 assert.match(html,/src="src\/vision-app.mjs"/);assert.match(html,/mountRopeLab\(document\.querySelector\('#rope-demo'\)\)/);
 assert.doesNotMatch(html,/styles\.css|full-app/,'the live model is not loaded');
 for(const part of ['patches, not tokens','no mask','2D RoPE','gated MLP','projector, not probabilities','What changes from the text transformer','Causal: a token cannot look at later tokens','None: every patch can look at every patch','W₂ · (SiLU(W₁ · x) ⊙ W₃ · x)','24 blocks · 16 heads · width 1,024','Every patch is kept','From 1,024 to 5,120','The values drawn are illustrative'])assert.ok(html.includes(part),part);
 assert.match(html,/<a href="transformer.html">← Transformer<\/a>/);assert.match(html,/<a href="vision.html" aria-current="page">Vision<\/a>/);
 for(const zone of ['embedding','attention','mlp','projector'])assert.ok(app.includes(`data-zone="${zone}"`),zone);
 assert.match(app,/const STAGE_NAMES=\['Dot product','Scaling','Softmax'\];/);assert.doesNotMatch(app,/class="cell[^"]*masked/,'no masked cells are drawn');
 for(const part of ['ATTENTION · NO MASK','↻ 2D RoPE','Gated MLP','Gate · W₃','1,024 → 5,120','LANGUAGE MODEL','matrix(1,024 × 768)','data-p="${p}"'])assert.ok(app.includes(part),part);
 assert.match(app,/chip\('block',fx\+18,fy,blockLabel\(\),176\)/);assert.match(app,/chip\('head',346,top,headLabel\(\),282\)/);
 assert.match(css,/\.rl-/,'RoPE lab styles are carried over');assert.match(css,/\.comparison-row/);
});
test('the earlier live-model page is kept, unlinked, as vision-live.html',async()=>{
 const live=await read('vision-live.html'),home=await read('index.html'),build=await read('build.mjs');
 assert.match(live,/src="src\/full-app.mjs"/);assert.doesNotMatch(home,/vision-live\.html/);assert.match(build,/at\('vision-live.html'\)/);
});
