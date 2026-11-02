import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
for(const file of ['vision.html','src/full-app.mjs','standalone.html'])test(`${file}: removed walkthrough and manual backend toggle do not leave controls or handlers`,async()=>{
 const source=await read(file);
 assert.doesNotMatch(source,/id="tour"|data-tour-|startTour\(|stopTour\(|tourFrame|tourRenderer|tourBar|Follow a patch|data-compute-mode|Recompute on CPU|Use automatic backend/);
});
test('automatic startup, cancellation and error-only CPU retry remain functional',async()=>{
 const app=await read('src/full-app.mjs');
 assert.match(app,/loadImage\(sampleSource\('bird'\)\)\.catch\(showError\)/);
 assert.match(app,/preferCPU:backendOverride/);
 assert.match(app,/backendOverride=false/);
 assert.match(app,/querySelector\('\[data-compute-cancel\]'\).hidden=run.complete\|\|!!run.error/);
 assert.match(app,/querySelector\('\[data-compute-retry\]'\).hidden=!run.error/);
 assert.match(app,/if\(e.target.closest\('\[data-compute-retry\]'\)\)\{backendOverride=true;recompute\(\);\}/);
 assert.match(app,/ticket\+\+;disposeWorker\(\);run.error='Cancelled/);
});
test('other interactive animations and shared block navigation are unaffected',async()=>{
 const app=await read('src/full-app.mjs');
 for(const name of ['mountMlpFlow','mountMlpCalculation','mountAutomaticValueMixing','mountAttentionWindow','mountRopeLab','createArchitectureMotion','createInlineBlockNavigation'])assert.ok(app.includes(name));
 assert.match(app,/function openStage\(i,focusTitle=false\)/);
 assert.match(app,/if\(state.expanded===4&&state.mlpCalculation\)closeMlpCalculation\(\);else collapse\(\)/);
});
