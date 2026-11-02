import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
test('compact MLP is a native button so pointer, Enter and Space open the same flow',async()=>{
 const markup=await read('src/mlp-overview.mjs'),app=await read('src/full-app.mjs');
 assert.match(markup,/<button type="button" class="mp-process" data-mlp-process data-expand-step="4"/);
 assert.match(markup,/aria-controls="stage-expansion-4"/);assert.match(markup,/aria-expanded="\$\{state.expanded===4\}"/);
 assert.match(app,/openStage\(\+b.dataset.expandStep,b.matches\('canvas'\)\|\|b.hasAttribute\('data-mlp-process'\)\|\|b.hasAttribute\('data-attention-process'\)\)/);
 assert.doesNotMatch(markup,/data-full-op|data-inline-block-step|canvas/);
});
test('expanded math dispatch remains in place and preserves patch/head/block selection',async()=>{
 const app=await read('src/full-app.mjs'),fn=app.match(/function openStage\(i,focusTitle=false\)\{[^\n]+/)[0];
 assert.match(fn,/state.expanded=i;state.step=i/);assert.match(fn,/if\(focusTitle\)\$\(`#stage-trigger-\$\{i\}`\)\?\.focus\(\{preventScroll:true\}\)/);
 assert.doesNotMatch(fn,/state.layer=|state.head=|state.feature=/);
 assert.match(app,/if\(\['mlp-up','mlp-gate','mlp-down'\].includes\(b.dataset.fullOp\)\)\{openMlpCalculation\(\);return;\}/);
 const css=await read('styles.css');assert.match(css,/\.mp-process:focus-visible\{outline:3px/);
});
