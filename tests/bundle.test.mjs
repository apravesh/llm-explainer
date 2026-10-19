import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
const html=await readFile(new URL('../standalone.html',import.meta.url),'utf8');
test('offline bundle has syntactically valid JavaScript, including both selector helpers',()=>{const scripts=[...html.matchAll(/<script type="module">([\s\S]*?)<\/script>/g)];assert.equal(scripts.length,1);const checked=spawnSync(process.execPath,['--input-type=module','--check'],{input:scripts[0][1],encoding:'utf8'});assert.equal(checked.status,0,checked.stderr);assert.match(scripts[0][1],/\$\$=s=>/);});
test('offline bundle embeds all required styles, logic, and sample assets',()=>{assert.match(html,/<style>/);assert.doesNotMatch(html,/<script[^>]*\bsrc=/);assert.doesNotMatch(html,/<link[^>]*rel="stylesheet"/);assert.doesNotMatch(html,/loadImage\(['"`]assets\//);assert.equal((html.match(/data:image\/svg\+xml;base64,/g)||[]).length,4);});
