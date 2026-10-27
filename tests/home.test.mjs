import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile,access} from 'node:fs/promises';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
const [home,vision,moe,build,standalone]=await Promise.all(['index.html','vision.html','moe.html','build.mjs','standalone.html'].map(read));
test('homepage links to text, vision, and MoE in the requested order',()=>{
 const list=home.match(/<ol class="chapter-list">([\s\S]*?)<\/ol>/)?.[1];assert.ok(list);
 assert.deepEqual([...list.matchAll(/class="chapter-link" href="([^"]+)"/g)].map(m=>m[1]),['https://poloclub.github.io/transformer-explainer/','vision.html','moe.html']);
 assert.match(home,/BY POLO CLUB/);assert.doesNotMatch(home,/<script/);
});
test('local chapter and stylesheet links resolve from the project subpath',async()=>{
 for(const html of [home,vision,moe])for(const [,href] of html.matchAll(/href="([^"]+)"/g))if(!/^(https?:|#|\.\/$)/.test(href)){
  assert.ok(!href.startsWith('/'),`${href} must preserve GitHub Pages project prefix`);await access(new URL('../'+href,import.meta.url));
 }
});
test('MoE is a working chapter with routing controls, disclosure, and a route home',()=>{
 assert.match(moe,/id="moe-explorer"/);assert.match(moe,/id="moe-k"/);assert.match(moe,/src="src\/moe-app.mjs"/);assert.match(moe,/aria-label="LLM Explainer home"/);assert.doesNotMatch(moe,/Currently under construction/);assert.match(moe,/untrained/i);
});
test('the complete vision chapter survives as a separate page with back navigation',()=>{
 for(const id of ['explorer','architecture','image-grid','rope-lab','comparison'])assert.match(vision,new RegExp(`id="${id}"`));
 assert.equal((vision.match(/data-stage="\d"/g)||[]).length,7);assert.match(vision,/href="\.\/" aria-label="LLM Explainer home"/);assert.match(vision,/src="src\/full-app.mjs"/);
});
test('offline builder reads the vision page, never the blog homepage',()=>{
 assert.match(build,/readFile\(at\('vision.html'\),'utf8'\)/);assert.doesNotMatch(build,/readFile\(at\('index.html'\)/);
 assert.match(standalone,/id="explorer"/);assert.doesNotMatch(standalone,/class="chapter-list"/);
});

test('vision chapter and offline bundle initialize Pixtral from the first render',async()=>{
 const app=await read('src/full-app.mjs');
 assert.match(app,/const state=\{[^\n]*model:'pixtral'/);
 for(const html of [vision,standalone]){
  const switchMarkup=html.match(/class="model-switch"[\s\S]*?<\/div>/)[0];
  assert.match(switchMarkup,/data-model="pixtral" aria-pressed="true"/);
  assert.match(switchMarkup,/data-model="vit" aria-pressed="false"/);
  assert.ok(switchMarkup.indexOf('data-model="pixtral"')<switchMarkup.indexOf('data-model="vit"'));
  assert.match(html,/id="output-shape">48 × 1,024/);assert.match(html,/id="ffn-title">Gated MLP/);
 }
 assert.doesNotMatch(app,/data-next-model|\[CLS\] → classifier; patches/);
});
