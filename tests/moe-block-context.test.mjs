import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8');
test('attention and WO precede experts inside one shared block; final LM head is outside',async()=>{
 const html=await read('moe.html'),group=html.indexOf('id="moe-transformer-block"'),flow=html.indexOf('id="moe-flow"'),attention=html.indexOf('class="context-stage context-attention"'),projection=html.indexOf('class="context-stage context-projection"'),input=html.indexOf('data-moe-node="input"'),readout=html.indexOf('id="moe-stack-readout"');
 assert.ok(group<flow&&flow<attention&&attention<projection&&projection<input&&input<readout);
 const context=html.slice(attention,input);assert.match(context,/Attention projection Wₒ/);assert.match(context,/Computed · 2 heads × 4/);assert.match(await read('src/moe-block-view.mjs'),/Not vocabulary logits/);assert.doesNotMatch(context,/<button|<details|data-inspect|schematic/i);
 assert.equal((html.match(/data-moe-block-step=/g)||[]).length,2);assert.match(html,/TRANSFORMER BLOCK 1 \/ 4/);assert.match(html,/<\/div><section id="moe-stack-readout"/);
});
test('all seven computed stages participate in the flow; synthetic limits stay explicit',async()=>{
 const html=await read('moe.html');assert.equal((html.match(/data-moe-node=/g)||[]).length,7);assert.match(html,/sinusoidal positions/);assert.match(html,/no RoPE/);assert.match(html,/No pretrained tokenizer/);assert.doesNotMatch(html,/Schematic · not executed|Demo begins at h|runs only the FFN/);
});
