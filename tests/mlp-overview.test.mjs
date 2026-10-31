import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {FULL_CONFIGS} from '../src/full-config.mjs';
import {mlpProcessMarkup} from '../src/mlp-overview.mjs';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
for(const model of ['pixtral','vit'])test(`${model}: compact MLP is a horizontal process, not a matrix dashboard`,()=>{
 const run={config:FULL_CONFIGS[model],layers:[]},state={layer:0,expanded:null},before=structuredClone({run,state});
 const html=mlpProcessMarkup(run,state);
 assert.match(html,/viewBox="0 0 490 /);assert.match(html,/data-mlp-process/);assert.equal((html.match(/<button /g)||[]).length,1);
 assert.doesNotMatch(html,/canvas|data-full-matrix|six-row|scroll patches|NaN|undefined/);
 assert.match(html,/W₂/);assert.match(html,/contract/);assert.match(html,/\+ residual H → H′/);assert.match(html,/Shared weights for every patch/);
 assert.match(html,new RegExp(`${run.config.width.toLocaleString()} → ${run.config.ffWidth.toLocaleString()} → ${run.config.width.toLocaleString()}`));
 if(model==='pixtral'){assert.match(html,/W₁ → SiLU/);assert.match(html,/W₃/);assert.match(html,/parallel expansion/);assert.match(html,/⊙/);assert.match(html,/M72 70V109/);assert.match(html,/M217 116/);}
 else{assert.match(html,/W₁ → GELU/);assert.doesNotMatch(html,/W₃|SiLU|⊙|parallel expansion/);}
 assert.deepEqual({run,state},before);
});
test('process uses one keyboard-accessible expansion control and stays attached to selected block',()=>{
 const html=mlpProcessMarkup({config:FULL_CONFIGS.pixtral},{layer:5,expanded:4});
 assert.match(html,/data-block="5"/);assert.match(html,/aria-label="Expand Gated MLP process flow"/);assert.match(html,/data-expand-step="4"/);assert.match(html,/aria-controls="stage-expansion-4"/);assert.match(html,/aria-expanded="true"/);
});
test('live and offline overviews use process-only attention and MLP',async()=>{
 for(const file of ['vision.html','standalone.html']){const html=await read(file);assert.match(html,/id="attention-process-viz"/);assert.doesNotMatch(html,/<div class="qkv-column">|<canvas id="attention-matrix"|class="ca-map"/);}
 const app=await read('src/full-app.mjs');assert.match(app,/attentionProcessMarkup\(run,state\)/);assert.doesNotMatch(app,/compactAttentionMarkup|mountOverviewWindow|createMlpOverview/);
 assert.match(app,/mlpProcessMarkup\(run,state\)/);assert.match(await read('standalone.html'),/function mlpProcessMarkup/);
});
test('projector is slim only in the collapsed overview; expanded math remains intact',async()=>{
 const css=await read('styles.css');assert.match(css,/#architecture:not\(\.has-expansion\) \.next-preview-button\{width:100%;max-width:var\(--compact-projector-width\)/);
 assert.match(css,/minmax\(460px,3.6fr\) var\(--compact-output-width\) var\(--compact-projector-width\)/);assert.match(css,/\.mp-process svg\{display:block;width:100%;height:auto/);
 const views=await read('src/full-views.mjs');assert.match(views,/function projector\(/);assert.match(views,/mat\(run.projected,'Decoder-ready Y'/);assert.match(views,/mlpFlowMarkup\(run,s\)/);assert.match(views,/mlpCalculationMarkup\(renderer,run,s/);
});
