import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
test('collapsed preview widths leave space for label, step number, and plus',async()=>{
 const css=await read('styles.css');
 assert.match(css,/--compact-embedding-width:100px/);assert.match(css,/--compact-output-width:82px/);assert.match(css,/--compact-projector-width:100px/);
 assert.match(css,/\.embed-stage>\.stage-title,#architecture:not\(\.has-expansion\) \.output-stage>\.stage-title\{white-space:nowrap;flex-wrap:nowrap/);
 assert.match(css,/\.next-stage>\.stage-title\{white-space:nowrap;flex-wrap:nowrap/);
 assert.match(css,/:is\(\.embed-stage,\.output-stage,\.next-stage\)>\.stage-title \.stage-plus\{flex-shrink:0\}/);
});
test('responsive grid and preview widths use the same values without changing expanded views',async()=>{
 for(const name of ['styles.css','standalone.html']){const css=await read(name);
  assert.match(css,/grid-template-columns:minmax\(90px,1fr\) 66px var\(--compact-embedding-width\)/);
  assert.match(css,/#architecture:not\(\.has-expansion\) \.next-preview-button\{width:100%;max-width:var\(--compact-projector-width\)/);
  assert.match(css,/#architecture:not\(\.has-expansion\) #output-vectors\{width:var\(--compact-output-width\)/);
 }
});
