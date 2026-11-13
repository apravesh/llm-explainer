import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile,access} from 'node:fs/promises';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
const html=await read('glossary.html');
const terms=[...html.matchAll(/<div class="gl-term" id="(term-[a-z0-9-]+)"><dt>([^<]+?)(?: <small>([^<]*)<\/small>)?<\/dt><dd>([\s\S]*?)<\/dd><\/div>/g)].map(m=>({id:m[1],term:m[2],exp:m[3]||'',def:m[4]}));
test('glossary defines the course terminology, once each, concisely',()=>{
 assert.ok(terms.length>=60,`${terms.length} terms`);
 assert.equal(new Set(terms.map(t=>t.id)).size,terms.length,'unique ids');
 const names=terms.map(t=>t.term);
 for(const t of ['LLM','Training','Inference','RAG','OCR','VLM','GPU','YOLO','MoE','Dense model','Transformer','Token','Tokenization','Token ID','Embedding','Positional embedding','Attention','Query, Key, Value','Output projection','Logits','Pretraining','Mid-training','Post-training','Special tokens','Curriculum','Learning rate','Epoch','Loss curve','Distillation','Vision encoder','Thinking','Agent','Harness','Quantization','Diffusion','Speculative decoding','W&amp;B','Inference engine','CUDA'])assert.ok(names.includes(t),t);
 for(const t of terms){const words=t.def.replace(/<[^>]+>/g,'').trim().split(/\s+/).length;assert.ok(words>=5&&words<=48,`${t.term}: ${words} words`);}
 assert.match(html,new RegExp(`${terms.length} common terms`));
});
test('glossary gets the easily confused facts right',()=>{
 const def=name=>terms.find(t=>t.term===name).def;
 assert.match(def('YOLO'),/find and box objects/);assert.doesNotMatch(def('YOLO'),/segmentation/);
 assert.match(def('Inference engine'),/vLLM/);assert.match(def('Inference'),/weights do not change/);
 assert.match(def('RAG'),/built around the model, not inside it/);assert.match(def('Diffusion'),/not an LLM/i);
 assert.match(def('Quantization'),/fewer bits/);assert.match(def('Vocabulary'),/50,257/);
});
test('every chapter link in the glossary resolves, including its section anchor',async()=>{
 const local=[...html.matchAll(/href="([^"#:]+\.html)(?:#([a-z-]+))?"/g)];assert.ok(local.length>30);
 for(const [,file,anchor] of local){const page=await read(file);if(anchor)assert.match(page,new RegExp(`id="${anchor}"`),`${file}#${anchor}`);}
 await access(new URL('../src/glossary-app.mjs',import.meta.url));
});
test('resources list the course links and open in a new tab',()=>{
 for(const url of ['https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi','https://poloclub.github.io/transformer-explainer/','https://www.youtube.com/watch?v=kCc8FmEb1nY','https://huggingface.co/spaces/Xenova/the-tokenizer-playground','https://www.youtube.com/watch?v=-W9F__D3oY4'])
  assert.match(html,new RegExp(`<li><a href="${url.replace(/[.?]/g,'\\$&')}" target="_blank" rel="noopener">`));
});
test('glossary is linked from every header and the home page',async()=>{
 for(const page of ['index.html','what-is-an-llm.html','llm-use-cases.html','transformer.html','vision.html','moe.html','analytics.html','pretraining.html','post-training-instruct.html','post-training-rl.html','glossary.html'])
  assert.match(await read(page),/<span class="nav-group">Reference<\/span><a href="glossary\.html"[^>]*>Glossary<\/a><\/nav>/);
 const home=await read('index.html');assert.match(home,/<section id="reference"[\s\S]*class="chapter-link" href="glossary\.html"/);assert.doesNotMatch(home,/<script/);
});
test('filter matches every typed word, in any order',async()=>{
 const src=await read('src/glossary-app.mjs');
 const matches=new Function('return '+src.match(/export const matches=([\s\S]*?);\n/)[1])();
 assert.ok(matches('rag retrieval-augmented generation search first','search rag'));assert.ok(matches('anything',''));assert.ok(!matches('token a word','rag'));
});
