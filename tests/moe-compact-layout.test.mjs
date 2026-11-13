import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {moeProjectionSummary} from '../src/moe-block-view.mjs';
const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8');
test('attention projection is a non-expanding labeled arrow retaining WO and residual semantics',()=>{
 const html=moeProjectionSummary({index:1,attention:{output:[[1,2,3,4,5,6,7,8]]}},0);
 assert.match(html,/class="projection-arrow"/);assert.match(html,/× Wₒ/);assert.match(html,/\+ H/);assert.match(html,/Not vocabulary logits/);assert.match(html,/8 by 8/);assert.doesNotMatch(html,/<button|mv-vector|<details/);
});
test('laptop layout replaces the wide projection track and uses a clean dotted block',async()=>{
 const css=await read('moe.css'),html=await read('moe-live.html');
 assert.match(css,/border:1px dotted #a8bacf/);assert.match(css,/110px 54px 84px 126px minmax\(162px,1fr\) 166px 86px;gap:18px;min-width:896px/);
 assert.match(css,/@media\(min-width:941px\) and \(max-width:1199px\)/);
 const section=html.slice(html.indexOf('class="context-stage context-projection"'),html.indexOf('<svg id="moe-wires"'));
 assert.match(section,/class="projection-connector"/);assert.doesNotMatch(section,/class="context-process"|context-boundary/);
});
