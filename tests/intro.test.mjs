import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile,access} from 'node:fs/promises';
import {END,NEWLINE,CORPUS,CHAT_EXAMPLES,NEXT_WORD_PROMPTS,STORY_PROMPTS,tokenize,chatPrompt,predict,choose,generate,detokenize,nextRandom} from '../src/intro-engine.mjs';
test('predictions are a probability distribution drawn only from words the examples contain',()=>{
 const vocab=new Set(CORPUS.flat());
 for(const prompt of [...NEXT_WORD_PROMPTS,...STORY_PROMPTS]){const r=predict(tokenize(prompt));assert.ok(r.matched>=3);assert.ok(Math.abs(r.all.reduce((s,c)=>s+c.p,0)-1)<1e-9);assert.ok(r.all.every(c=>vocab.has(c.token)&&c.p>0));assert.ok(r.candidates.every((c,i,a)=>i===0||c.p<=a[i-1].p));}
 const cat=predict(tokenize('the cat sat on the'));assert.equal(cat.candidates[0].token,'mat');assert.ok(cat.candidates[0].p>.5&&cat.candidates[0].p<.9);assert.ok(cat.candidates.some(c=>c.token==='sofa'));
 assert.equal(predict(tokenize('the capital of france is')).candidates[0].token,'paris');assert.throws(()=>predict([]));assert.deepEqual(predict(['zebra']).candidates,[]);
});
test('repeating the prediction builds sentences and whole stories, then stops at the end marker',()=>{
 assert.equal(detokenize([...tokenize('the cat sat on the'),...generate(tokenize('the cat sat on the'))]),'The cat sat on the mat.');
 const story=generate(tokenize('once upon a time'),{limit:80});assert.equal(story.at(-1),END);assert.match(detokenize([...tokenize('once upon a time'),...story]),/^Once upon a time there was a small dragon .* And they lived happily ever after\.$/);
 const a=generate(tokenize('once upon a time'),{limit:80,seed:3}),b=generate(tokenize('once upon a time'),{limit:80,seed:3});assert.deepEqual(a,b);assert.equal(a.at(-1),END);
 const r=nextRandom(7);assert.ok(r.value>=0&&r.value<1);assert.equal(choose({candidates:[]}),null);
});
test('the chat script turns the same next-word loop into replies that end with a stop',()=>{
 for(const [question,answer] of CHAT_EXAMPLES){const prompt=chatPrompt(question);assert.deepEqual(prompt.slice(0,2),['user',':']);assert.deepEqual(prompt.slice(-3),[NEWLINE,'assistant',':']);
  const reply=generate(prompt);assert.equal(reply.at(-1),END);assert.deepEqual(reply.slice(0,-1),tokenize(answer));}
 assert.equal(detokenize(chatPrompt('what is the capital of france ?')),'User: What is the capital of France?\nAssistant:');
 assert.equal(detokenize(generate(chatPrompt('hello !'))),'Hello! How can I help you today?');
});
test('intro page is first on the home page and in every header, and its assets resolve',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),html=await read('what-is-an-llm.html'),home=await read('index.html');
 for(const id of ['next-word','loop','stories','chat','nw-machine','loop-machine','story-text','chat-raw'])assert.ok(html.includes(`id="${id}"`));
 assert.match(html,/Not a real LLM/);assert.match(html,/GPT-2/);assert.match(html,/PaLM/);assert.match(html,/<code>User:<\/code>/);
 for(const [,href] of html.matchAll(/(?:href|src)="([^"]+)"/g))if(!/^(https?:|#|\.\/)/.test(href)){assert.ok(!href.startsWith('/'));await access(new URL('../'+href.split('#')[0],import.meta.url));}
 assert.ok(home.indexOf('id="intro"')<home.indexOf('id="chapters"'));
 for(const page of ['index.html','vision.html','moe.html','analytics.html','pretraining.html','post-training-instruct.html','post-training-rl.html','standalone.html'])assert.match(await read(page),/<a href="what-is-an-llm.html">How does an LLM actually work\?<\/a>/);
 assert.match(html,/<a href="what-is-an-llm.html" aria-current="page">/);
});
test('random stories vary: many different stories per opening, and a new seed on every start',async()=>{
 const {generate,tokenize,detokenize,STORY_PROMPTS}=await import('../src/intro-engine.mjs');
 for(const p of STORY_PROMPTS){const seen=new Set();for(let s=1;s<=200;s++)seen.add(detokenize(generate(tokenize(p),{seed:s*7919,limit:90})));assert.ok(seen.size>=25,`${p}: ${seen.size} distinct stories`);}
 const app=await (await import('node:fs/promises')).readFile(new URL('../src/intro-app.mjs',import.meta.url),'utf8');
 assert.match(app,/story\.seed = freshSeed\(\)/);assert.doesNotMatch(app,/story\.seed = 3;/);
});
