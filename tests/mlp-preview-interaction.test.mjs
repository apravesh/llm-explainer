import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
test('both compact MLP matrices are accessible expansion controls in source and standalone',async()=>{
 for(const file of ['src/full-app.mjs','standalone.html']){
  const source=await read(file);
  assert.match(source,/\$\$\('#ffn-viz \[data-full-matrix\]'\)\.forEach/);
  assert.match(source,/canvas.dataset.expandStep='4';canvas.tabIndex=0/);
  assert.match(source,/canvas.setAttribute\('role','button'\)/);
  assert.match(source,/canvas.setAttribute\('aria-controls','stage-expansion-4'\)/);
  assert.match(source,/canvas.setAttribute\('aria-expanded',state.expanded===4\)/);
  assert.match(source,/index===0\?'input':'activated'/);
  assert.match(source,/c.gated\?'Gated MLP':'MLP'/);
 }
});
test('pointer, Enter and Space use stage expansion, while multiplication dispatch is preserved',async()=>{
 const app=await read('src/full-app.mjs');
 assert.match(app,/closest\('button, canvas\[data-expand-step\]'\)/);
 assert.match(app,/openStage\(\+b.dataset.expandStep,b.matches\('canvas'\)\)/);
 assert.match(app,/matches\('canvas\[data-expand-step\]'\)\&\&\['Enter',' '\].includes\(e.key\)\)\{e.preventDefault\(\);openStage\(\+e.target.dataset.expandStep,true\)/);
 assert.match(app,/if\(b.dataset.fullOp\)\{stopMix\(\);overlay.open\(b.dataset.fullOp,b\);return;/);
 assert.match(app,/data-full-op="mlp-up"/);assert.match(app,/data-full-op="mlp-down"/);
});
test('MLP expansion retains the selected context and focuses its visible heading',async()=>{
 const app=await read('src/full-app.mjs'),fn=app.match(/function openStage\(i,focusTitle=false\)\{[^\n]+/)[0];
 assert.match(fn,/state.expanded=i;state.step=i/);assert.match(fn,/if\(focusTitle\)\$\(`#stage-trigger-\$\{i\}`\)\?\.focus\(\{preventScroll:true\}\)/);
 assert.doesNotMatch(fn,/state.layer=|state.head=|state.feature=/);assert.match(fn,/if\(i<2&&state.selected<0\)state.selected=0/);
 const css=await read('styles.css');assert.match(css,/\.fa-mini-mlp canvas\[data-expand-step\]\{cursor:pointer/);assert.match(css,/canvas\[data-expand-step\]:focus-visible/);
});
