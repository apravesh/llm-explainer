import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
for(const name of ['vision.html','standalone.html'])test(`${name}: one shared selector labels the container around both transformer stages`,async()=>{
 const html=await read(name);
 assert.equal((html.match(/data-inline-block-step="-1"/g)||[]).length,1);
 assert.equal((html.match(/data-inline-block-step="1"/g)||[]).length,1);
 assert.match(html,/<div class="encoder-group" role="group" aria-labelledby="block-caption">/);
 assert.match(html,/<span class="inline-block-controls" role="group" aria-label="Transformer block navigation">/);
 const group=html.slice(html.indexOf('<div class="encoder-group"'),html.indexOf('<section class="stage output-stage"'));
 assert.match(group,/class="stage attention-stage(?: active)?"/);assert.match(group,/class="stage mlp-stage"/);
 assert.doesNotMatch(html,/inlineBlockControlsMarkup|expanded-block-controls/);
});
test('expanded stage headings no longer render a separate block selector',async()=>{
 const app=await read('src/full-app.mjs');
 assert.match(app,/classList.toggle\('has-expanded-block',state.expanded===3\|\|state.expanded===4\)/);
 assert.match(app,/<h3 class="full-stage-heading">\$\{stageTitle\(i,c\)\} · full dimensions<\/h3>/);
 assert.doesNotMatch(app,/inlineBlockControlsMarkup/);
});
test('expanded layout retains a visible enclosing block and responsive shared headings',async()=>{
 const css=await read('styles.css');
 assert.match(css,/\.architecture.has-expansion \.encoder-group\{\s*display:grid;/);
 assert.match(css,/\.architecture.has-expansion \.encoder-group>\.block-label\{\s*position:static;display:flex;/);
 assert.doesNotMatch(css,/\.architecture.has-expansion \.encoder-group\{display:contents\}/);
 assert.doesNotMatch(css,/\.architecture.has-expansion \.block-label\{display:none\}/);
 assert.match(css,/\.encoder-group>\.stage>\.stage-expansion:not\(\[hidden\]\)\{grid-column:1\/-1;grid-row:3;min-width:0\}/);
 assert.match(css,/\.architecture.has-expansion \.encoder-group\{grid-column:1\/-1;grid-row:2;width:auto;/);
});
