import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
test('all expanded stages use the shared header without a duplicate Collapse button',async()=>{
 for(const file of ['src/full-app.mjs','standalone.html']){
  const source=await read(file);
  assert.doesNotMatch(source,/data-full-collapse|<button[^>]*class="collapse-expansion"/);
  assert.match(source,/<div class="expansion-topline"><span class="expansion-breadcrumb">ARCHITECTURE \/ \$\{stageTitle\(i,c\).toUpperCase\(\)\}<\/span><\/div>/);
  assert.match(source,/b.querySelector\('\.stage-plus'\).textContent=i===state.expanded\?'−':'\+'/);
  assert.match(source,/b.setAttribute\('aria-expanded',i===state.expanded\)/);
  assert.match(source,/b.setAttribute\('aria-controls',`stage-expansion-\$\{i\}`\)/);
 }
});
test('the existing minus toggle and Escape still collapse and restore heading focus',async()=>{
 const app=await read('src/full-app.mjs');
 assert.match(app,/state.expanded===\+b.dataset.step\?collapse\(\):openStage\(\+b.dataset.step\)/);
 assert.match(app,/e.key==='Escape'&&state.expanded!==null\)\{e.preventDefault\(\);collapse\(\);return;/);
 assert.match(app,/motion.run\(null,\(\)=>\{const i=state.expanded;state.expanded=null;render\(\);\$\(`#stage-trigger-\$\{i\}`\)\?\.focus/);
});
test('attention calculation collapse remains separate from collapsing its whole stage',async()=>{
 const attention=await read('src/attention-window.mjs');
 assert.match(attention,/data-window-breakdown="close"/);
 assert.match(attention,/Collapse attention calculation/);
});
