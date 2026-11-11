import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile,access} from 'node:fs/promises';
import {CRAWL_EXAMPLE,exampleCorpus,corpusView} from '../src/pretraining-data.mjs';
import {BPE_WORDS,bpeTrace,encodeBpe,mergePair,pairCounts} from '../src/pretraining-bpe.mjs';
import {PRETRAIN_BATCH,PRETRAIN_VOCAB,createPretrainModel,pretrainForward,pretrainStep,applyPretrainSgd} from '../src/pretraining-engine.mjs';
const near=(a,b,t=1e-7)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
test('illustrative corpus filters sources/quality, deduplicates before splitting and never trains on held out data',()=>{
 const before=structuredClone(CRAWL_EXAMPLE),c=exampleCorpus();assert.equal(c.raw.length,7);assert.equal(c.eligible.length,5);assert.equal(c.unique.length,4);assert.equal(c.train.length,3);assert.equal(c.validation.length,1);assert.equal(c.shards.length,2);assert.deepEqual(c.shards.flat(),c.train);
 assert.ok(!c.train.some(t=>c.validation.some(v=>v.text===t.text)));assert.ok(!c.eligible.some(r=>!r.allowed||!r.quality));assert.equal(c.raw[0].text,'The cat sat on the mat.');assert.ok(!c.raw[0].text.includes('Home'));assert.deepEqual(CRAWL_EXAMPLE,before);
 assert.equal(corpusView(0).filter(r=>r.kept).length,7);assert.equal(corpusView(1).filter(r=>r.kept).length,5);assert.equal(corpusView(2).filter(r=>r.kept).length,4);assert.equal(corpusView(3).filter(r=>r.status.startsWith('validation')).length,1);assert.throws(()=>corpusView(9));
});
test('BPE counts pairs by frequency, replaces non-overlapping occurrences and keeps word boundaries',()=>{
 assert.deepEqual(mergePair(['a','a','a','a','a'],['a','a']),['aa','aa','a']);
 assert.deepEqual(pairCounts([{tokens:['a','a','a'],count:3}])[0],{pair:['a','a'],count:6});
 const t=bpeTrace(0);assert.deepEqual(t.pairs[0],{pair:['e','s'],count:9});assert.equal(t.tokenCount,BPE_WORDS.reduce((n,w)=>n+(w.word.length+1)*w.count,0));
 const next=bpeTrace(1);assert.equal(next.vocab.length,t.vocab.length+1);assert.equal(next.tokenCount,t.tokenCount-9);assert.ok(next.words.find(w=>w.word==='newest').tokens.includes('es'));
 for(let i=0;i<=20;i++){const s=bpeTrace(i);for(const w of s.words)assert.equal(w.tokens.join(''),w.word+'</w>');assert.ok(s.words.every(w=>w.tokens.every(x=>s.vocab.includes(x))));}
});
test('BPE encoding applies ranked rules without changing training; unknown demo alphabet is explicit',()=>{
 const t=bpeTrace(8),before=structuredClone(t);for(const word of ['lower','lowest','newest','widest']){const e=encodeBpe(word,t);assert.equal(e.tokens.join(''),word+'</w>');assert.ok(e.ids.every(id=>id>=0));assert.deepEqual(e.ids.map(id=>t.vocab[id]),e.tokens);}assert.deepEqual(t,before);assert.throws(()=>encodeBpe('zoo',t));assert.throws(()=>bpeTrace(-1));assert.throws(()=>bpeTrace(21));assert.throws(()=>encodeBpe('<script>',t));
});
test('mini-transformer computes shifted targets, normalized probabilities, causal attention and mean CE',()=>{
 const model=createPretrainModel(),before=structuredClone(model),r=pretrainForward(model);assert.equal(r.count,10);assert.equal(r.traces.length,10);assert.equal(r.summary.reduce((n,g)=>n+g.count,0),192);
 for(const tr of r.traces){assert.equal(tr.target,PRETRAIN_BATCH[tr.batch][tr.position+1]);assert.equal(tr.input,PRETRAIN_BATCH[tr.batch][tr.position]);assert.deepEqual(tr.context,PRETRAIN_BATCH[tr.batch].slice(0,tr.position+1));near(tr.probabilities.reduce((s,p)=>s+p,0),1);near(tr.loss,-Math.log(tr.probabilities[tr.target]));assert.equal(tr.attention.length,tr.position+1);near(tr.attention.reduce((s,p)=>s+p,0),1);assert.equal(tr.hidden.length,4);assert.equal(tr.logits.length,8);}
 near(r.loss,r.traces.reduce((s,t)=>s+t.loss,0)/10);assert.deepEqual(model,before);assert.deepEqual(model,createPretrainModel());
});
test('the batch loss and gradients are the average of both sequences, not their sum',()=>{
 const model=createPretrainModel(),a=pretrainForward(model,[PRETRAIN_BATCH[0]]),b=pretrainForward(model,[PRETRAIN_BATCH[1]]),both=pretrainForward(model);near(both.loss,(a.loss+b.loss)/2);
 for(const key of Object.keys(model))for(let i=0;i<model[key].length;i++)for(let j=0;j<model[key][i].length;j++)near(both.grads[key][i][j],(a.grads[key][i][j]+b.grads[key][i][j])/2);
 const unequal=pretrainForward(model,[PRETRAIN_BATCH[0],PRETRAIN_BATCH[1].slice(0,3)]);const short=pretrainForward(model,[PRETRAIN_BATCH[1].slice(0,3)]);near(unequal.loss,(a.loss*5+short.loss*2)/7);assert.equal(unequal.count,7);
});
test('autodiff gradients match finite differences through embeddings, attention, MLP and LM head',()=>{
 const model=createPretrainModel(),r=pretrainForward(model),eps=1e-5;
 for(const key of Object.keys(model))for(const [i,j] of [[0,0],[Math.floor(model[key].length/2),1],[model[key].length-1,model[key][0].length-1]]){const original=model[key][i][j];model[key][i][j]=original+eps;const plus=pretrainForward(model,PRETRAIN_BATCH,{gradients:false}).loss;model[key][i][j]=original-eps;const minus=pretrainForward(model,PRETRAIN_BATCH,{gradients:false}).loss;model[key][i][j]=original;near(r.grads[key][i][j],(plus-minus)/(2*eps),2e-6);}
});
test('future tokens cannot leak into earlier-position predictions',()=>{
 const model=createPretrainModel(),batch=structuredClone(PRETRAIN_BATCH),original=pretrainForward(model,batch);batch[0][4]=3;const changed=pretrainForward(model,batch);
 for(let i=0;i<4;i++)assert.deepEqual(original.traces[i].probabilities,changed.traces[i].probabilities);assert.notDeepEqual(original.traces[4].probabilities,changed.traces[4].probabilities);
 assert.deepEqual(original.traces[5].probabilities,changed.traces[5].probabilities);
});
test('SGD updates every parameter by the mean-loss gradient and repeated training fits this batch',()=>{
 const model=createPretrainModel(),before=structuredClone(model),step=pretrainStep(model);for(const key of Object.keys(model))for(let i=0;i<model[key].length;i++)for(let j=0;j<model[key][i].length;j++)near(step.model[key][i][j],model[key][i][j]-.1*step.before.grads[key][i][j],1e-12);
 assert.deepEqual(model,before);assert.ok(step.after.loss<step.before.loss);let updated=step.model;for(let i=0;i<9;i++)updated=pretrainStep(updated).model;assert.ok(pretrainForward(updated).loss<step.before.loss);assert.throws(()=>applyPretrainSgd(model,step.before.grads,NaN));assert.throws(()=>pretrainForward(model,[]));assert.throws(()=>pretrainForward(model,[[0,99]]));
});
test('architecture grouping remains intact and training has its own three pages',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),home=await read('index.html');assert.match(home,/<h2 id="chapters-title">Architecture<\/h2>/);assert.match(home,/<h2 id="training-title">Training<\/h2>/);
 const training=home.slice(home.indexOf('<section id="training"'));assert.deepEqual([...training.matchAll(/class="chapter-link" href="([^"]+)"/g)].map(m=>m[1]),['pretraining.html','post-training-instruct.html','post-training-rl.html']);
 for(const page of ['pretraining.html','post-training-instruct.html','post-training-rl.html']){const html=await read(page);for(const [,href] of html.matchAll(/(?:href|src)="([^"]+)"/g))if(!/^(https?:|#|\.\/$)/.test(href)){assert.ok(!href.startsWith('/'));await access(new URL('../'+href.split('#')[0],import.meta.url));}}
 assert.match(await read('post-training-rl.html'),/GRPO/);
 const html=await read('pretraining.html');for(const id of ['data','tokenizer','learn','train-batch','train-next','bpe-next','corpus-documents'])assert.ok(html.includes(`id="${id}"`));assert.match(html,/separate from the BPE demo/);assert.match(html,/not generalization/);assert.match(html,/not a pretrained LLM/);
});
test('UI keeps forward snapshots fixed during an update and cancels stale automatic timers',async()=>{
 const app=await readFile(new URL('../src/pretraining-app.mjs',import.meta.url),'utf8');assert.match(app,/strip\(trace.embedding,'Embedding lookup, before update'\)/);assert.match(app,/if\(ticket!==epoch\|\|!playing\)return/);assert.match(app,/clearTimeout\(timer\)/);assert.match(app,/if\(phase===4\)\{model=applyPretrainSgd/);assert.match(app,/document.hidden/);assert.match(app,/prefers-reduced-motion/);
});
