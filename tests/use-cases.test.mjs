import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile,access} from 'node:fs/promises';
import {USE_CASES,SEARCH_DEMO,typingUnits,queryPrompt,searchPrompt,segmentsText} from '../src/use-cases-data.mjs';
test('every use case is one text stream that ends with the model writing',()=>{
 assert.deepEqual(USE_CASES.map(u=>u.key),['chat','writing','code','agent']);
 for(const u of USE_CASES){assert.ok(u.segments.every(s=>['app','user','model','tool'].includes(s.who)&&s.text.length>0));assert.equal(u.segments.at(-1).who,'model');assert.ok(u.segments.some(s=>s.who==='user'));for(const s of u.segments)assert.equal(typingUnits(s.text).join(''),s.text);}
 assert.ok(!USE_CASES[1].segments.some(s=>s.who==='app'),'plain completion needs no script');
 const agent=USE_CASES[3].segments;assert.equal(agent.filter(s=>s.who==='tool').length,agent.filter(s=>s.kind==='action').length);agent.forEach((s,i)=>{if(s.who==='tool')assert.equal(agent[i-2].kind,'action');});
 assert.match(segmentsText(USE_CASES[0].segments),/^User: .*\nAssistant: /);
});
test('search prompt pastes numbered results above the question, and the answer only uses facts found there',()=>{
 const d=SEARCH_DEMO,withResults=searchPrompt(d.question,d.results),text=segmentsText(withResults);
 assert.ok(text.indexOf('[1] ')<text.indexOf('User: '));assert.ok(text.endsWith(`User: ${d.question}\nAssistant: `));assert.equal(withResults.filter(s=>s.who==='tool').length,d.results.length);
 for(const fact of ['Amara Okafor','2:31:07','40 seconds'])assert.ok(d.answer.includes(fact)&&text.includes(fact.replace('Amara ','')));
 for(const n of d.sources)assert.ok(n>=1&&n<=d.results.length);
 const without=segmentsText(searchPrompt(d.question,[]));assert.equal(without,`User: ${d.question}\nAssistant: `);assert.ok(!without.includes('Okafor'));assert.ok(!d.answerWithoutSearch.includes('Okafor'));
 assert.ok(segmentsText(queryPrompt(d.question)).endsWith('Search query: '));assert.throws(()=>searchPrompt(d.question,null));assert.ok(d.results.every(r=>r.site.includes('.example')));
});
test('use-cases page is second in Intro, linked from every header, and its assets resolve',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),html=await read('llm-use-cases.html');
 for(const id of ['uses','search','use-tabs','use-raw','search-flow','search-engine','search-raw','search-next'])assert.ok(html.includes(`id="${id}"`));
 assert.match(html,/retrieval-augmented generation, or RAG/);assert.match(html,/not live model output/);assert.match(html,/invented for this example/);
 for(const [,href] of html.matchAll(/(?:href|src)="([^"]+)"/g))if(!/^(https?:|#|\.\/)/.test(href)){assert.ok(!href.startsWith('/'));await access(new URL('../'+href.split('#')[0],import.meta.url));}
 for(const page of ['index.html','vision.html','moe.html','analytics.html','pretraining.html','post-training-instruct.html','post-training-rl.html','what-is-an-llm.html','standalone.html'])assert.match(await read(page),/<a href="llm-use-cases.html">Use cases<\/a>/);
 assert.match(html,/<a href="llm-use-cases.html" aria-current="page">Use cases<\/a>/);
});
