import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {createTokenizer,VOCAB_SIZE,END_OF_TEXT} from '../src/tokenizer-gpt2.mjs';
import {EXAMPLES} from '../src/text-data.mjs';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
const tok=createTokenizer(await read('assets/gpt2-merges.txt'));

test('GPT-2 tokenizer: 50,000 merge rules give 50,257 token IDs',()=>{
 assert.equal(tok.mergeCount,50000);assert.equal(VOCAB_SIZE,256+50000+1);assert.equal(END_OF_TEXT,50256);
 assert.deepEqual(tok.merge(0),['Ġ','t'],'the first rule glues a space to t');
});
test('it reproduces the token IDs recorded from GPT-2 on the Transformer page',()=>{
 for(const e of EXAMPLES){assert.deepEqual(tok.ids(e.prompt),e.tokenIds,e.prompt);assert.deepEqual(tok.encode(e.prompt).map(t=>t.text),e.tokens);}
});
test('reference IDs from the Hugging Face tokenizer, including accents, CJK and emoji',()=>{
 const cases={
  'The cat sat on the mat.':[464,3797,3332,319,262,2603,13],
  'strawberry':[301,1831,8396],
  'strawberry strawberry':[301,1831,8396,41236],
 };
 for(const [text,ids] of Object.entries(cases))assert.deepEqual(tok.ids(text),ids,text);
 for(const text of ['naïve café 日本語 🙂','  two  spaces\n\nand\ttabs ','I\'m sure they\'ll say it\'s John\'s.'])assert.equal(tok.encode(text).map(t=>t.token).join('').length>0,true);
});
test('the differences between tokens and words that the page describes are true',()=>{
 const ids=s=>tok.ids(s);
 assert.equal(ids(' cat').length,1);assert.equal(ids('cat').length,1);assert.notEqual(ids(' cat')[0],ids('cat')[0],'the space is part of the token');
 assert.equal(new Set([ids('hello')[0],ids('Hello')[0],ids('HELLO')[0]]).size,3,'capitals change the token');
 assert.ok(ids('antidisestablishmentarianism').length>3,'rare words are several tokens');
 assert.ok(ids('1234567').length>1,'long numbers are chopped up');
 assert.ok(ids('日本語').length>3,'non-English text costs more tokens than characters');
 assert.equal(ids('.').length,1);
});
test('bytes survive: joining the tokens gives back the text',()=>{
 for(const text of ['Zürich 🙂 ok','a  b\nc','naïve café 日本語'])assert.equal(new TextDecoder().decode(Uint8Array.from(tok.encode(text).flatMap(t=>tok.bytesOf(t.token)))),text);
 assert.equal(tok.encode('tokenization').map(t=>t.text).join(''),'tokenization');assert.equal(tok.encode('The cat sat.').map(t=>t.text).join(''),'The cat sat.');
});
test('merge steps replay the rules earliest-first and end at the real tokens',()=>{
 const steps=tok.mergeSteps('tokenization');
 assert.deepEqual(steps[0].parts,[...'tokenization']);assert.equal(steps[0].rule,null);
 assert.deepEqual(steps.at(-1).parts.map(tok.text),['token','ization']);
 for(let k=1;k<steps.length;k++){assert.ok(steps[k].parts.length<steps[k-1].parts.length);assert.ok(Number.isInteger(steps[k].rule));}
 assert.equal(tok.mergeSteps(' the').at(-1).parts.length,1,'a common word collapses to one token');
});
test('page: live tokenizer, token-vs-word points, three ways, and a BPE lab with honest notes',async()=>{
 const html=await read('tokenization.html'),app=await read('src/tokens-app.mjs');
 for(const id of ['tokenizer','tk-lab','tk-input','tk-out','tk-chars','tk-words','tk-tokens','tk-ids','differences','why','tw-chars','tw-words','tw-tokens','bpe','bpe-build','bpe-ladder','bpe-next','bpe-play','bpe-reset','bpe-word'])assert.match(html,new RegExp(`id="${id}"`),id);
 for(const part of ['Why do LLMs read tokens, <em>not words?</em>','A token is not quite a word','The space belongs to the token','Capital letters change the token','Limits and prices are counted in tokens','100 tokens is about 75 words','Byte Pair Encoding','256 + 50,000 + 1 special end-of-text token = 50,257 tokens','pretraining.html#tokenizer','Everything here is computed by GPT-2\'s actual tokenizer','Other models have their own tokenizers'])assert.ok(html.includes(part),part);
 assert.equal((html.match(/data-try="/g)||[]).length,6);
 assert.match(html,/<a href="tokenization.html" aria-current="page">Tokens<\/a>/);assert.match(html,/src="src\/tokens-app.mjs"/);
 assert.match(app,/fetch\(new URL\('\.\.\/assets\/gpt2-merges\.txt',import\.meta\.url\)\)/);
 assert.doesNotMatch(html,/To remember|class="recap/i);
});
