import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8');
test('attention and output projection precede the computed MoE input in one block',async()=>{
 const html=await read('moe.html'),start=html.indexOf('id="moe-flow"'),attention=html.indexOf('class="context-stage context-attention"'),projection=html.indexOf('class="context-stage context-projection"'),input=html.indexOf('data-moe-node="input"');
 assert.ok(start<attention&&attention<projection&&projection<input);
 const context=html.slice(attention,input);assert.match(context,/Self-attention/);assert.match(context,/causal mask \+ softmax/);assert.match(context,/Attention_weights × V/);assert.match(context,/Concatenate heads/);assert.match(context,/× Wₒ/);assert.match(context,/\+ input residual H/);
 assert.match(context,/Schematic · not executed/);assert.match(context,/Synthetic vector, not computed attention output/);assert.match(context,/Not vocabulary logits/);
 assert.doesNotMatch(context,/<button|<details|data-inspect|data-moe-node|data-full-matrix/);
});
test('computed stages and animation remain separate from schematic context',async()=>{
 const html=await read('moe.html');assert.equal((html.match(/data-moe-node=/g)||[]).length,5);
 const engine=await read('src/moe-engine.mjs');assert.doesNotMatch(engine,/attentionForward|attentionFull/);
 const css=await read('moe.css');assert.match(css,/grid-template-columns:145px 150px 105px 160px/);
});
