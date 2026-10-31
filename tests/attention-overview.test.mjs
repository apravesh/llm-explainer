import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {FULL_CONFIGS} from '../src/full-config.mjs';
import {attentionProcessMarkup} from '../src/attention-overview.mjs';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
for(const model of ['pixtral','vit'])test(`${model}: attention overview shows projections, weights and the separate V path without matrices`,()=>{
 const run={config:FULL_CONFIGS[model],layers:[]},state={layer:3,head:2,expanded:null},before=structuredClone({run,state}),html=attentionProcessMarkup(run,state);
 assert.match(html,/viewBox="0 0 490 200"/);assert.equal((html.match(/data-ap-node=/g)||[]).length,5);
 for(const key of ['q','k','v','weights','mix'])assert.match(html,new RegExp(`data-ap-node="${key}"`));
 for(const value of ['QKᵀ / √64','softmax → weights','× V','mix patches','concat → Wₒ + residual H → H′','No causal mask','every patch can attend to every patch'])assert.ok(html.includes(value),value);
 assert.match(html,/data-ap-edge="q-weights"/);assert.match(html,/data-ap-edge="k-weights"/);assert.match(html,/data-ap-edge="weights-mix"/);assert.match(html,/data-ap-edge="v-mix"/);assert.doesNotMatch(html,/data-ap-edge="v-weights"/);
 assert.match(html,new RegExp(`${run.config.heads} heads × 64 features → ${run.config.width.toLocaleString()}`));
 if(model==='pixtral')assert.match(html,/2D RoPE on Q\/K/);else assert.doesNotMatch(html,/RoPE/);
 assert.doesNotMatch(html,/canvas|data-full-matrix|data-window-pair|NaN|undefined/);assert.deepEqual({run,state},before);
});
test('one native process button opens attention; context reflects the selected block/head',()=>{
 const html=attentionProcessMarkup({config:FULL_CONFIGS.pixtral},{layer:5,head:9,expanded:3});
 assert.equal((html.match(/<button /g)||[]).length,1);assert.match(html,/data-block="5" data-head="9"/);
 assert.match(html,/data-attention-process data-expand-step="3"/);assert.match(html,/aria-controls="stage-expansion-3" aria-expanded="true"/);
 assert.match(html,/aria-label="Expand Self-attention process flow"/);assert.doesNotMatch(html,/data-inline-block-step/);
});
test('active and offline apps mount the process without stale canvas handlers and retain detailed attention',async()=>{
 const app=await read('src/full-app.mjs'),bundle=await read('standalone.html');
 assert.match(app,/attentionProcessMarkup\(run,state\)/);assert.match(bundle,/function attentionProcessMarkup/);
 assert.doesNotMatch(app,/\$\('#attention-matrix'\)|\$\('#matrix-caption'\)|q-swatch/);
 assert.match(app,/b.hasAttribute\('data-attention-process'\)/);assert.match(app,/mountAttentionWindow\(/);assert.match(app,/mountAutomaticValueMixing\(/);
 assert.match(app,/data-head-step="-1"/);assert.match(app,/data-head-step="1"/);
 assert.match(await read('src/full-views.mjs'),/headStackMarkup/);
});
