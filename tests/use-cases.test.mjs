import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile,access} from 'node:fs/promises';
import {USE_CASES,SEARCH_DEMO,typingUnits,queryPrompt,searchPrompt,segmentsText} from '../src/use-cases-data.mjs';
test('creative writing comes first, then chat; the scripted uses are one text stream that ends with the model writing',()=>{
 assert.deepEqual(USE_CASES.map(u=>u.key),['writing','chat','code','agent']);
 assert.deepEqual(USE_CASES.map(u=>u.label),['Creative writing','Chat','Code completion','Agentic workflow']);
 assert.deepEqual(USE_CASES.map(u=>u.lab),['stories','chat',undefined,undefined],'the first two are live labs on the page');
 for(const u of USE_CASES){for(const k of ['kicker','give','writes','where','note'])assert.ok(u[k].length>0,`${u.key}.${k}`);assert.match(u.note,u.lab?/toy box · not a real LLM/:/not live model output/);}
 for(const u of USE_CASES.filter(u=>u.lab))assert.equal(u.segments.length,0);
 for(const u of USE_CASES.filter(u=>!u.lab)){assert.ok(u.point.length>0);assert.ok(u.segments.every(s=>['app','user','model','tool'].includes(s.who)&&s.text.length>0));assert.equal(u.segments.at(-1).who,'model');assert.ok(u.segments.some(s=>s.who==='user'));for(const s of u.segments)assert.equal(typingUnits(s.text).join(''),s.text);}
 const agent=USE_CASES[3].segments;assert.equal(agent.filter(s=>s.who==='tool').length,agent.filter(s=>s.kind==='action').length);agent.forEach((s,i)=>{if(s.who==='tool')assert.equal(agent[i-2].kind,'action');});
 assert.match(segmentsText(agent),/^Task: .*\nTools: .*\nAssistant: /);
});
test('search prompt pastes numbered results above the question, and the answer only uses facts found there',()=>{
 const d=SEARCH_DEMO,withResults=searchPrompt(d.question,d.results),text=segmentsText(withResults);
 assert.ok(text.indexOf('[1] ')<text.indexOf('User: '));assert.ok(text.endsWith(`User: ${d.question}\nAssistant: `));assert.equal(withResults.filter(s=>s.who==='tool').length,d.results.length);
 for(const fact of ['Amara Okafor','2:31:07','40 seconds'])assert.ok(d.answer.includes(fact)&&text.includes(fact.replace('Amara ','')));
 for(const n of d.sources)assert.ok(n>=1&&n<=d.results.length);
 const without=segmentsText(searchPrompt(d.question,[]));assert.equal(without,`User: ${d.question}\nAssistant: `);assert.ok(!without.includes('Okafor'));assert.ok(!d.answerWithoutSearch.includes('Okafor'));
 assert.ok(segmentsText(queryPrompt(d.question)).endsWith('Search query: '));assert.throws(()=>searchPrompt(d.question,null));assert.ok(d.results.every(r=>r.site.includes('.example')));
});
const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8');
const PAGES=['index.html','what-is-an-llm.html','transformer.html','vision.html','vision-live.html','moe.html','analytics.html','pretraining.html','post-training-instruct.html','post-training-rl.html','glossary.html'];
test('use cases are section 02 of the overview: creative writing is first and selected, chat is there, each shows its lab',async()=>{
 const html=await read('what-is-an-llm.html'),app=await read('src/use-cases-app.mjs'),intro=await read('src/intro-app.mjs');
 const s=html.slice(html.indexOf('<section id="uses"'),html.indexOf('<section id="search"'));
 for(const id of ['uses-lab','use-tabs','use-label','use-facts','use-screen','use-raw','use-point','use-controls','use-replay','story-lab','chat-demo'])assert.ok(s.includes(`id="${id}"`),id);
 assert.deepEqual([...s.matchAll(/<button type="button" data-use="(\d)" aria-pressed="(true|false)"><small>[^<]+<\/small>([^<]+)<\/button>/g)].map(m=>m.slice(1)),[['0','true','Creative writing'],['1','false','Chat'],['2','false','Code completion'],['3','false','Agentic workflow']]);
 assert.deepEqual([...s.matchAll(/<div class="use-panel" id="([^"]+)" data-use-panel="[^"]+"( hidden)?>/g)].map(m=>[m[1],!m[2]]),[['stories',true],['chat',false],['use-scripted',false]],'only the creative writing panel starts visible');
 const panel=id=>{const a=s.indexOf(`<div class="use-panel" id="${id}"`),b=s.indexOf('<div class="use-panel"',a+1);return s.slice(a,b<0?s.length:b);};
 const story=panel('stories'),chat=panel('chat');
 assert.match(story,/<p class="use-desc">Give the LLM the start of a story and ask it to complete it\. [^<]*GPT-2 in 2019[^<]*PaLM/);
 for(const part of ['id="story-prompts"','data-story-pick="top"','data-story-pick="dice"','id="story-text"','id="story-sentence">Write the next sentence →','id="story-all"','id="story-reset"'])assert.ok(story.includes(part),part);
 assert.match(chat,/<p class="use-desc">A chatbot is the same loop, started with a script\./);assert.match(chat,/<code>User:<\/code>/);
 for(const part of ['id="chat-questions"','id="chat-window"','id="chat-raw"','id="chat-result"','id="chat-replay"','WHAT THE BOX SEES'])assert.ok(chat.includes(part),part);
 assert.doesNotMatch(html,/<section id="(stories|chat)"/,'no standalone completion or chat sections');
 assert.equal(html.split('Give the LLM the start of a story').length,2);assert.equal(html.split('A chatbot is the same loop').length,2);
 assert.match(app,/panel\.hidden = panel\.id !== \(c\.lab \|\| 'use-scripted'\)/);
 assert.match(app,/new CustomEvent\('usecase:shown', \{detail: useCase\(\)\.key\}\)/);assert.match(intro,/addEventListener\('usecase:shown'/);
 assert.match(app,/addEventListener\('hashchange', useFromHash\)/,'#stories and #chat select their use case');
 assert.match(html,/not a real LLM/);assert.match(app,/c\.note/);
 assert.match(await read('use-cases.css'),/\.use-panel\[hidden\],#use-controls\[hidden\]\{display:none\}/);
});
test('chat with web search is the last lesson section, with its lab and assets',async()=>{
 const html=await read('what-is-an-llm.html');
 const sections=[...html.matchAll(/<section id="([^"]+)" class="lesson-section">/g)].map(m=>m[1]);assert.equal(sections.at(-1),'search');
 const s=html.slice(html.indexOf('<section id="search"'));
 for(const id of ['search-lab','search-flow','search-chat','search-engine','search-raw','search-result','search-next','search-play','search-reset'])assert.ok(s.includes(`id="${id}"`),id);
 assert.match(s,/<p class="eyebrow">03 \/ CHAT WITH WEB SEARCH<\/p>/);assert.match(s,/retrieval-augmented generation, or RAG/);assert.match(s,/invented for this example/);
 for(const [,href] of html.matchAll(/(?:href|src)="([^"]+)"/g))if(!/^(https?:|#|\.\/)/.test(href)){assert.ok(!href.startsWith('/'));await access(new URL('../'+href.split('#')[0],import.meta.url));}
});
test('the old use-cases page is only a redirect to the overview section',async()=>{
 const html=await read('llm-use-cases.html');
 assert.match(html,/<meta http-equiv="refresh" content="0; url=what-is-an-llm\.html#uses">/);assert.match(html,/<link rel="canonical" href="what-is-an-llm\.html#uses">/);assert.match(html,/<a href="what-is-an-llm\.html#uses">/);
 assert.doesNotMatch(html,/<script|site-bar|class="lab/);assert.ok(html.length<900);
 assert.match(await read('what-is-an-llm.html'),/<section id="uses"/);
});
test('no header links to a Use cases page, the home page has no Use cases card, and nothing else links to the old page',async()=>{
 for(const page of [...PAGES,'standalone.html']){const html=await read(page),nav=html.match(/<nav aria-label="Chapters">.*?<\/nav>/)[0];
  assert.doesNotMatch(nav,/Use cases|llm-use-cases/,page);assert.match(nav,/<span class="nav-group">Intro<\/span><a href="what-is-an-llm\.html"[^>]*>Overview<\/a>/,page);
  assert.doesNotMatch(html,/llm-use-cases/,page);}
 const home=await read('index.html'),intro=home.slice(home.indexOf('<section id="intro"'),home.indexOf('<section id="chapters"'));
 const cards=[...intro.matchAll(/class="chapter-link" href="([^"]+)"/g)].map(m=>m[1]);assert.equal(cards[0],'what-is-an-llm.html');assert.ok(!cards.includes('llm-use-cases.html'));
 assert.doesNotMatch(intro,/What are LLMs used for\?/);assert.match(intro,/creative writing, chat, code completion and agents[^<]*searches the web/);
 assert.doesNotMatch(await read('README.md'),/llm-use-cases\.html` is the second/);
});
test('the three points after the search demo are plain text, not numbered cards',async()=>{
 const html=await read('what-is-an-llm.html');
 assert.doesNotMatch(html,/rag-cards|intro-cards/);
 const block=html.slice(html.indexOf('<div class="plain-points">'),html.indexOf('<details class="further">',html.indexOf('<div class="plain-points">')));
 for(const title of ['The model learned nothing new','The answer is only as good as the results','The same trick works on your own files'])assert.ok(block.includes(`<h3>${title}</h3>`),title);
 assert.doesNotMatch(block,/<span>0\d<\/span>/);
});
