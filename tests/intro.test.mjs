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
 for(const id of ['next-word','loop','uses','stories','chat','search','loop-machine','story-text','chat-raw'])assert.equal(html.split(`id="${id}"`).length,2,`one #${id}`);
 const ids=[...html.matchAll(/ id="([^"]+)"/g)].map(m=>m[1]);assert.equal(new Set(ids).size,ids.length,'no duplicate ids on the merged page');
 assert.match(html,/Not a real LLM/);assert.match(html,/GPT-2/);assert.match(html,/PaLM/);assert.match(html,/<code>User:<\/code>/);
 for(const [,href] of html.matchAll(/(?:href|src)="([^"]+)"/g))if(!/^(https?:|#|\.\/)/.test(href)){assert.ok(!href.startsWith('/'));await access(new URL('../'+href.split('#')[0],import.meta.url));}
 for(const [,anchor] of html.matchAll(/href="#([^"]+)"/g))assert.ok(html.includes(`id="${anchor}"`),`#${anchor}`);
 assert.ok(home.indexOf('id="intro"')<home.indexOf('id="chapters"'));
 for(const page of ['index.html','vision.html','moe.html','analytics.html','pretraining.html','post-training-instruct.html','post-training-rl.html','standalone.html'])assert.match(await read(page),/<a href="what-is-an-llm.html">Overview<\/a>/);
 assert.match(html,/<a href="what-is-an-llm.html" aria-current="page">/);
 for(const asset of ['intro.css','use-cases.css','src/intro-app.mjs','src/use-cases-app.mjs'])assert.ok(html.includes(`="${asset}"`),asset);
});
test('the overview has three numbered sections: next word, use cases, chat with web search',async()=>{
 const html=await readFile(new URL('../what-is-an-llm.html',import.meta.url),'utf8');
 assert.deepEqual([...html.matchAll(/<section id="([^"]+)" class="lesson-section"><div class="section-heading"><p class="eyebrow">([^<]+)<\/p>/g)].map(m=>[m[1],m[2]]),[['next-word','01 / THE SIMPLEST FORM'],['uses','02 / USE CASES'],['search','03 / CHAT WITH WEB SEARCH']]);
 assert.deepEqual([...html.match(/<p class="page-links">.*?<\/p>/)[0].matchAll(/href="([^"]+)"/g)].map(m=>m[1]),['#next-word','#uses','#search']);
 assert.doesNotMatch(html,/0[24] \/ (REPEAT|CHAT<)|03 \/ COMPLETION|POPULAR USES TODAY/);
 assert.ok(html.indexOf('<section id="search"')<html.indexOf('notes-section')&&html.indexOf('notes-section')<html.indexOf('<section class="chapter-next">'),'web search is the last lesson, then notes, then what is next');
 assert.equal(html.split('<section class="chapter-next">').length,2,'one next block');
 assert.match(html.slice(html.indexOf('<section class="chapter-next">')),/NEXT \/ OPEN THE BOX[\s\S]*<a href="transformer\.html">Architecture →/);
});
test('section 01 merges the simplest form and repeat: the first title, one description, the repeat lab',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),html=await read('what-is-an-llm.html'),app=await read('src/intro-app.mjs');
 const s=html.slice(html.indexOf('<section id="next-word"'),html.indexOf('<section id="uses"'));
 assert.match(s,/<h2>Given a stream of words, predict the next word<\/h2><p>What is an LLM in its simplest form\? Text goes in, a guess for the next word comes out\. Add that word to the end of the text and ask again\. [^<]*full sentences and more[^<]*<\/p><\/div>/);
 assert.equal(s.split('<p').length-1,5,'eyebrow, one description, the lab’s text and result lines, and the token note');
 assert.match(s,/<div id="loop"><div class="lab machine-lab" id="loop-lab">/);
 for(const part of ['id="loop-prompts"','data-loop-pick="top" aria-pressed="true">Take the top guess','data-loop-pick="dice" aria-pressed="false">Roll the dice','id="loop-text"','id="loop-machine"','class="loop-back"','id="loop-next">Add the next word →','id="loop-play"','id="loop-reset"','id="loop-count"','class="token-note"'])assert.ok(s.includes(part),part);
 assert.doesNotMatch(html,/nw-lab|nw-machine|nw-prompts|nw-result|Keep predicting the next word iteratively/);
 assert.doesNotMatch(app,/nw-|nwPrompt|renderNextWord/,'the one-guess lab is gone from the code too');
 assert.match(app,/\nloopStart\(3\); storyStart\(0\); renderChat\(\);\n/);
});
test('random stories vary: many different stories per opening, and a new seed on every start',async()=>{
 const {generate,tokenize,detokenize,STORY_PROMPTS}=await import('../src/intro-engine.mjs');
 for(const p of STORY_PROMPTS){const seen=new Set();for(let s=1;s<=200;s++)seen.add(detokenize(generate(tokenize(p),{seed:s*7919,limit:90})));assert.ok(seen.size>=25,`${p}: ${seen.size} distinct stories`);}
 const app=await (await import('node:fs/promises')).readFile(new URL('../src/intro-app.mjs',import.meta.url),'utf8');
 assert.match(app,/story\.seed = freshSeed\(\)/);assert.doesNotMatch(app,/story\.seed = 3;/);
});
test('the repeat lab points its return arrow at the end of the text, where the new word goes',async()=>{
 const fs=await import('node:fs/promises'),app=await fs.readFile(new URL('../src/intro-app.mjs',import.meta.url),'utf8'),css=await fs.readFile(new URL('../intro.css',import.meta.url),'utf8');
 assert.match(app,/function placeLoopBack\(\)/);assert.match(app,/word\.right \+ 7 - start\.left/);assert.match(app,/\n placeLoopBack\(\);\n\}/);
 assert.match(css,/\.loop-back::after\{[^}]*height:var\(--rise/);
});
test('adding a word in the repeat lab moves only what changed',async()=>{
 const fs=await import('node:fs/promises'),app=await fs.readFile(new URL('../src/intro-app.mjs',import.meta.url),'utf8'),css=await fs.readFile(new URL('../intro.css',import.meta.url),'utf8');
 assert.match(app,/renderLoop\(true, loop\.context\.length - 1, true\)/);assert.match(app,/function loopMotion\(box, before, from, oldBars, held\)/);
 assert.match(app,/el\.classList\.toggle\('fresh', fresh && !step && !reduced\.matches\)/,'no full replay when one word is added');
 assert.match(app,/data-i="\$\{hidden \+ i\}"/);assert.match(app,/word\.animate\(frames/);assert.match(app,/if \(!reduced\.matches && !loopDone\(\)\) loopMotion/);
 assert.doesNotMatch(css,/\.machine\.step \.mc-word\{animation/,'earlier words are not re-animated');
 for(const part of ['.machine.step .mc-arrow:not(.out) i','.machine.step .mc-box','.machine.step .mc-arrow.out i','.machine.step .mc-row'])assert.ok(css.includes(part),part);
});

test('the text going into the box stays on one line; earliest words give way to the ellipsis',async()=>{
 const app=await readFile(new URL('../src/intro-app.mjs',import.meta.url),'utf8'),css=await readFile(new URL('../intro.css',import.meta.url),'utf8');
 assert.match(app,/function fitWords\(el\)/);assert.match(app,/\n fitWords\(el\);\n\}/);
 assert.match(css,/@media\(min-width:901px\)\{\.mc-words\{flex-wrap:nowrap\}/);
});

test('there are always five guesses to show; the fill-ins are rare common words that are never drawn at random',()=>{
 for(const p of NEXT_WORD_PROMPTS){let c=tokenize(p);for(let i=0;i<30;i++){
  const pr=predict(c);assert.equal(pr.candidates.length,5,c.join(' '));
  assert.ok(Math.abs(pr.all.reduce((s,x)=>s+x.p,0)-1)<1e-9,'the sampled distribution is unchanged');
  for(const x of pr.candidates.filter(x=>x.rare)){assert.ok(x.p<0.005);assert.ok(!pr.all.some(a=>a.token===x.token));}
  const t=choose(pr);if(t===END)break;c.push(t);
 }}
});
