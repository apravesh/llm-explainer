import {INSTRUCT_EXAMPLES,chatTokens,lossComparison,logitGradient,DATA_STAGES,DATA_RECORDS} from './instruct-engine.mjs';
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let example=0,generation=false,tokenIndex=1,dataStage=0,lossMode='sft',selected={batch:0,position:chatTokens(0).findIndex(t=>t.supervised)},override=null;
function renderChat(){
 const e=INSTRUCT_EXAMPLES[example],tokens=chatTokens(example,{generation});tokenIndex=Math.min(tokenIndex,tokens.length-1);
 $('#chat-preview').innerHTML=`<div><small>USER</small><p>${esc(e.prompt)}</p></div><span aria-hidden="true">→</span><div><small>ASSISTANT</small><p>${generation?'Waiting for generated tokens…':esc(e.answer)}</p></div>`;
 document.querySelectorAll('[data-chat-mode]').forEach(b=>b.setAttribute('aria-pressed',String((b.dataset.chatMode==='generation')===generation)));
 $('#chat-stream').innerHTML=tokens.map((t,i)=>`<button type="button" data-token="${i}" class="chat-token ${t.special?'special-token':''} ${t.kind==='answer'?'answer-token':''}" aria-pressed="${i===tokenIndex}" aria-label="Token ${i}: ${esc(t.text)}, ${t.special?'special':'ordinary'}, ID ${t.id}"><span>${t.special?'◆ ':''}${esc(t.text)}</span><small>ID ${t.id}</small></button>`).join('');
 const t=tokens[tokenIndex],meaning={'<bos>':'Beginning of this sequence.','<message>':'Start of a message; a role name follows.','<text>':'End of the role header; message content follows.','<end>':t.supervised?'End of the assistant message. In this toy setup, learn to emit it and stop the response.':'End of the user message. The assistant header follows.'};
 $('#chat-inspection').textContent=`${t.text} → ID ${t.id}. ${t.special?meaning[t.text]:t.kind==='role'?'An ordinary role-name token inside a delimited header—not a special token in this template.':t.kind==='answer'?'A token from the desired answer. SFT teaches the model to predict it from the preceding context.':'An ordinary prompt token, provided as context for the answer.'}`;
 $('#chat-mode-note').textContent=generation?'At generation time, supply the prompt and assistant header, then generate the continuation until an appropriate stop token. No answer is supplied and no weights are updated.':'During SFT, the correct assistant answer is already in the example. Predict each answer token from the prompt and the correct answer tokens before it.';
}
$('#chat-example').addEventListener('change',e=>{example=+e.target.value;renderChat();});
 document.querySelectorAll('[data-chat-mode]').forEach(b=>b.addEventListener('click',()=>{generation=b.dataset.chatMode==='generation';renderChat();}));
$('#chat-stream').addEventListener('click',e=>{const b=e.target.closest('[data-token]');if(!b)return;tokenIndex=+b.dataset.token;renderChat();$(`[data-token="${tokenIndex}"]`).focus({preventScroll:true});});
function renderData(){
 $('#data-stages').innerHTML=DATA_STAGES.map((s,i)=>`<button type="button" data-data-stage="${i}" aria-pressed="${dataStage===i}"><small>0${i+1}</small>${esc(s.title)}</button>`).join('');
 $('#data-note').textContent=DATA_STAGES[dataStage].note;
 $('#demo-records').innerHTML=DATA_RECORDS.map(r=>`<div class="demo-record ${dataStage>0&&!r.kept?'rejected':''} ${dataStage===2&&r.split==='Held-out evaluation'?'held-out':''}"><div><small>${esc(r.source)}</small><p>${esc(r.prompt)} <span aria-hidden="true">→</span> <strong>${esc(r.answer)}</strong></p></div><span>${dataStage===0?'Candidate':!r.kept?esc(r.reason):dataStage===1?'Accepted':esc(r.split)}</span></div>`).join('');
 $('#data-result').textContent=['4 candidates. A useful record demonstrates a response, not just a topic.','2 accepted · 1 duplicate removed · 1 incorrect answer rejected.','1 train record → template + token IDs + loss labels. 1 held-out record → evaluation only.'][dataStage];
}
$('#data-stages').addEventListener('click',e=>{const b=e.target.closest('[data-data-stage]');if(!b)return;dataStage=+b.dataset.dataStage;renderData();$(`[data-data-stage="${dataStage}"]`).focus({preventScroll:true});});
function renderLoss(){
 const result=lossComparison(lossMode,override),row=result.rows.find(r=>r.batch===selected.batch&&r.position===selected.position);
 document.querySelectorAll('[data-loss-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.lossMode===lossMode)));
 $('#loss-streams').innerHTML=INSTRUCT_EXAMPLES.map((_,batch)=>`<div class="loss-sequence"><small>SEQUENCE ${batch+1} · NEXT-TOKEN TARGETS</small><div class="loss-token-row"><span class="bos-context">&lt;bos&gt;<small>start context<br>not a target</small></span>${result.rows.filter(r=>r.batch===batch).map(r=>`<button type="button" data-loss-batch="${batch}" data-loss-position="${r.position}" class="loss-token ${r.included?'scored':''}" aria-pressed="${r.batch===selected.batch&&r.position===selected.position}" aria-label="Sequence ${batch+1}, target ${r.position}: ${esc(r.text)}, ${r.included?'scored':'context only'}"><span>${esc(r.text)}</span><small>m = ${r.included?1:0}</small></button>`).join('')}</div></div>`).join('');
 $('#selected-title').textContent=`Target: ${row.text} · sequence ${row.batch+1}`;
 $('#selected-context').textContent=`Prediction comes from the previous position, using: ${row.context.map(t=>t.text).join(' ')}`;
 $('#target-probability').value=Math.round(row.p*100);$('#target-probability-label').textContent=`${Math.round(row.p*100)}%`;
 $('#selected-loss').textContent=`${row.included?'Scored':'Masked'}: ${row.included?1:0} × −ln(${row.p.toFixed(2)}) = ${(row.included?row.loss:0).toFixed(4)}${row.included?'':' (raw token loss '+row.loss.toFixed(4)+')'}`;
 $('#sft-loss').textContent=`${result.count} scored / ${result.rows.length} possible targets → ${result.sum.toFixed(4)} / ${result.count} = ${result.loss.toFixed(4)} nats`;
 const grad=logitGradient(row,result.count)[row.id];
 $('#sft-gradient').textContent=row.included?`For this target’s logit: ∂L/∂z = (${row.p.toFixed(2)} − 1) / ${result.count} = ${grad.toFixed(4)}. Increasing this logit alone would reduce the loss. The optimizer instead updates shared weights via the chain rule.`:'This target has no direct loss or logit gradient (m = 0). Its token still provides context for later assistant predictions; those losses can backpropagate through its representations.';
}
 document.querySelectorAll('[data-loss-mode]').forEach(b=>b.addEventListener('click',()=>{lossMode=b.dataset.lossMode;renderLoss();}));
$('#loss-streams').addEventListener('click',e=>{const b=e.target.closest('[data-loss-position]');if(!b)return;selected={batch:+b.dataset.lossBatch,position:+b.dataset.lossPosition};override=null;renderLoss();$(`[data-loss-batch="${selected.batch}"][data-loss-position="${selected.position}"]`).focus({preventScroll:true});});
$('#target-probability').addEventListener('input',e=>{override={...selected,p:+e.target.value/100};renderLoss();});
document.addEventListener('click',e=>{if(e.target.closest('a[href^="#ref-"]'))$('#sources').open=true;});
renderChat();renderData();renderLoss();
