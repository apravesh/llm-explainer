import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {canSee,pairs,cached,compare,layerKinds,PATTERNS} from '../src/long-context.mjs';
const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8');

test('a local layer sees itself and the w − 1 tokens before it; a global layer sees every earlier token',()=>{
 assert.ok(canSee(5,0));assert.ok(!canSee(2,3),'never a later token');
 assert.ok(canSee(5,2,4));assert.ok(!canSee(5,1,4));assert.ok(canSee(5,5,4));
 for(const [n,w] of [[16,4],[16,8],[10,3],[7,20]]){let count=0;for(let i=0;i<n;i++)for(let j=0;j<n;j++)if(canSee(i,j,w))count++;assert.equal(pairs(n,w),count);}
 assert.equal(pairs(16),136);assert.equal(pairs(16,4),58);
 assert.equal(cached(1000),1000);assert.equal(cached(1000,64),64);assert.equal(cached(10,64),10);
});
test('5 local : 1 global at 128K with a window of 1,024 keeps about a sixth of the cost',()=>{
 const g=compare({n:131072,w:1024,pattern:'global'}),m=compare({n:131072,w:1024,pattern:'mixed'}),l=compare({n:131072,w:1024,pattern:'local'});
 assert.equal(g.pairsShare,1);assert.equal(g.cacheShare,1);
 assert.equal(m.cachePerLayer,(5*1024+131072)/6);assert.ok(m.cacheShare>0.16&&m.cacheShare<0.19);assert.ok(m.pairsShare>0.16&&m.pairsShare<0.19);
 assert.equal(l.cachePerLayer,1024);assert.ok(l.pairsShare<0.02);
 assert.equal(compare({n:1024,w:4096,pattern:'local'}).pairsShare,1,'a window longer than the prompt saves nothing');
 assert.deepEqual(layerKinds('mixed',12),['local','local','local','local','local','global','local','local','local','local','local','global']);
 assert.deepEqual(Object.keys(PATTERNS),['global','mixed','local']);
});
test('the Parameters page has 03 Optimizations instead of the old calculator',async()=>{
 const html=await read('analytics.html');
 for(const part of ['01 / COUNT THE WEIGHTS','02 / COUNT THE WORK','03 / SPEND LESS','<h2>Optimizations</h2>','Long context: local and global attention layers','id="lc-lab"','src="src/long-context-app.mjs"','only the mask changes','Gemma 3','Mistral 7B','arxiv.org/abs/2503.19786','arxiv.org/abs/2310.06825','arithmetic, not measured speed or memory','href="#optimizations"'])assert.ok(html.includes(part),part);
 for(const gone of ['04 /','05 /','GQA','budget-form','architecture-budget','panel-gpu','id="calculator"'])assert.ok(!html.includes(gone),gone);
 assert.ok(html.indexOf('id="part-2"')<html.indexOf('id="optimizations"')&&html.indexOf('id="optimizations"')<html.indexOf('id="sources"'));
 for(const m of html.matchAll(/href="#([a-z0-9-]+)"/g))assert.ok(html.includes(`id="${m[1]}"`),m[1]);
 assert.match(await read('glossary.html'),/analytics\.html#long-context/);
});
