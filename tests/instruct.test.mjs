import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile,access} from 'node:fs/promises';
import {SPECIAL_TOKENS,INSTRUCT_VOCAB,INSTRUCT_EXAMPLES,chatTokens,targetDistribution,lossComparison,logitGradient,DATA_RECORDS} from '../src/instruct-engine.mjs';
const near=(a,b,eps=1e-10)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
test('fictional chat template has reserved IDs and ordinary role names',()=>{
 assert.equal(new Set(INSTRUCT_VOCAB).size,INSTRUCT_VOCAB.length);
 for(let b=0;b<INSTRUCT_EXAMPLES.length;b++){
 const tokens=chatTokens(b);assert.equal(tokens[0].text,'<bos>');assert.equal(tokens.at(-1).text,'<end>');assert.equal(tokens.at(-1).supervised,true);
 tokens.forEach(t=>{assert.equal(INSTRUCT_VOCAB[t.id],t.text);assert.equal(t.special,SPECIAL_TOKENS.includes(t.text));});
 for(const role of tokens.filter(t=>t.kind==='role'))assert.equal(role.special,false);
 assert.ok(tokens.filter(t=>t.kind==='prompt').every(t=>!t.supervised));
 const start=tokens.findIndex(t=>t.supervised);assert.ok(tokens.slice(start).every(t=>t.supervised));
 assert.deepEqual(chatTokens(b,{generation:true}),tokens.slice(0,start));
 assert.equal(chatTokens(b,{generation:true}).at(-1).text,'<text>');
 }
 assert.throws(()=>chatTokens(-1));assert.throws(()=>chatTokens(.5));
});
test('all valid targets are shifted: their context ends immediately before the target',()=>{
 for(const r of lossComparison('full').rows){const tokens=chatTokens(r.batch);assert.deepEqual(r.context,tokens.slice(0,r.position));assert.equal(r.id,tokens[r.position].id);assert.ok(r.position>0);}
 assert.equal(lossComparison('full').rows.length,INSTRUCT_EXAMPLES.reduce((n,_,b)=>n+chatTokens(b).length-1,0));
});
test('SFT masks prompts and headers, includes answer and stop tokens, keeps probabilities identical',()=>{
 const sft=lossComparison(),full=lossComparison('full');assert.equal(sft.count,6);assert.equal(full.count,27);assert.equal(sft.rows.length,full.rows.length);
 sft.rows.forEach((r,i)=>{assert.equal(r.included,r.supervised);assert.deepEqual(r.probabilities,full.rows[i].probabilities);assert.equal(r.p,full.rows[i].p);assert.equal(full.rows[i].included,true);});
 near(sft.loss,sft.rows.filter(r=>r.supervised).reduce((n,r)=>n-Math.log(r.p),0)/6);
 near(full.loss,full.sum/27);assert.throws(()=>lossComparison('invalid'));
});
test('illustrative distributions are normalized and mean-loss logit gradients match finite differences',()=>{
 for(const mode of ['full','sft']){const result=lossComparison(mode);for(const r of result.rows){near(r.probabilities.reduce((a,b)=>a+b,0),1);const grads=logitGradient(r,result.count);near(grads.reduce((a,b)=>a+b,0),0);const logits=r.probabilities.map(Math.log),eps=1e-5;
 for(const j of [0,r.id]){const loss=delta=>{const z=logits.map((v,k)=>v+(k===j?delta:0));return r.included?(Math.log(z.reduce((s,v)=>s+Math.exp(v),0))-z[r.id])/result.count:0;};near(grads[j],(loss(eps)-loss(-eps))/(2*eps),1e-8);}
 }}
 assert.throws(()=>targetDistribution(0,0));assert.throws(()=>targetDistribution(0,1));assert.throws(()=>targetDistribution(-1,.5));
});
test('changing a masked target leaves SFT loss unchanged but changing an answer changes it',()=>{
 const base=lossComparison(),prompt=base.rows.find(r=>!r.included),answer=base.rows.find(r=>r.included);
 near(lossComparison('sft',{batch:prompt.batch,position:prompt.position,p:.95}).loss,base.loss);
 assert.ok(lossComparison('sft',{batch:answer.batch,position:answer.position,p:.95}).loss<base.loss);
 near(lossComparison('sft',{batch:answer.batch,position:answer.position,p:.95}).count,base.count);
});
test('data pipeline rejects duplicates and weak answers before a disjoint illustrative split',()=>{
 const kept=DATA_RECORDS.filter(r=>r.kept);assert.equal(kept.length,2);assert.equal(new Set(kept.map(r=>r.prompt)).size,2);assert.equal(kept.filter(r=>r.split==='Train').length,1);assert.equal(kept.filter(r=>r.split==='Held-out evaluation').length,1);assert.equal(DATA_RECORDS.filter(r=>!r.kept).length,2);
});
test('Instruct chapter replaces its placeholder, links to the RL chapter and resolves all local assets',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8');const html=await read('post-training-instruct.html');assert.ok(!html.includes('training-placeholder'));for(const id of ['format','demonstrations','sft','chat-stream','loss-streams','target-probability'])assert.ok(html.includes(`id="${id}"`));
 assert.match(html,/Assistant-only loss is a choice/);assert.match(html,/no model training runs here/);assert.match(html,/loss mask is not an attention mask/);assert.match(html,/does not guarantee/i);
 for(const [,href] of html.matchAll(/(?:href|src)="([^"]+)"/g))if(!/^(https?:|#|\.\/$)/.test(href))await access(new URL('../'+href.split('#')[0],import.meta.url));
 assert.match(await read('index.html'),/Interactive chapter · Chat formatting &amp; SFT/);assert.match(await read('post-training-rl.html'),/GRPO/);
});
