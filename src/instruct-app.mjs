import {INSTRUCT_EXAMPLES,chatTokens,lossComparison,logitGradient,DATA_STAGES,DATA_RECORDS} from './instruct-engine.mjs';
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)],esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
let example=0,generation=false,tokenIndex=1,dataStage=0,lossMode='sft',selected={batch:0,position:chatTokens(0).findIndex(t=>t.supervised)},override=null;

/* ---------- 01 · Messages → one string → token IDs ---------- */
const kindClass=t=>t.special?'k-special':t.kind==='role'?'k-role':t.kind==='answer'?'k-answer':'k-prompt';
// Which message a token belongs to: 0 = user, 1 = assistant, -1 = the sequence start.
function messageOf(tokens,index){let message=-1;for(let i=0;i<=index;i++)if(tokens[i].text==='<message>')message++;return message;}
function renderChat(fresh=false){
 const e=INSTRUCT_EXAMPLES[example],tokens=chatTokens(example,{generation});tokenIndex=Math.min(tokenIndex,tokens.length-1);
 const t=tokens[tokenIndex],lit=messageOf(tokens,tokenIndex);
 $('#chat-lab').classList.toggle('fresh',fresh&&!reduced.matches);
 $('#chat-preview').innerHTML=`<div class="bubble user ${lit===0?'lit':''}"><small>USER</small><p>${esc(e.prompt)}</p></div><div class="bubble assistant ${lit===1?'lit':''} ${generation?'pending':''}"><small>ASSISTANT</small><p>${generation?'<span class="typing" aria-label="not written yet"><i></i><i></i><i></i></span>':esc(e.answer)}</p></div>`;
 $$('[data-chat-mode]').forEach(b=>b.setAttribute('aria-pressed',String((b.dataset.chatMode==='generation')===generation)));
 $('#chat-string').innerHTML=tokens.map((tok,i)=>{const space=i>0&&!tok.special&&tok.kind===tokens[i-1].kind&&/^[A-Za-z]/.test(tok.text)?' ':'';
  return `${space}<span class="str ${kindClass(tok)} ${i===tokenIndex?'sel':''}" style="--d:${i*55}ms">${esc(tok.text)}</span>`;}).join('')+(generation?'<span class="caret" aria-hidden="true"></span>':'');
 $('#chat-stream').innerHTML=tokens.map((tok,i)=>`<button type="button" data-token="${i}" style="--d:${i*55+250}ms" class="chat-token ${kindClass(tok)}" aria-pressed="${i===tokenIndex}" aria-label="Token ${i}: ${esc(tok.text)}, ${tok.special?'special':'ordinary'}, ID ${tok.id}"><span>${esc(tok.text)}</span><small>${tok.id}</small></button>`).join('')+(generation?`<span class="chat-token next-slot" style="--d:${tokens.length*55+250}ms"><span>?</span><small>model writes here</small></span>`:'');
 const meaning={'<bos>':'Beginning of this sequence.','<message>':'Start of a message; a role name follows.','<text>':'End of the role header; message content follows.','<end>':t.supervised?'End of the assistant message. In this toy setup, the model learns to emit it and stop.':'End of the user message. The assistant header follows.'};
 $('#chat-inspection').textContent=`${t.text} → ID ${t.id}. ${t.special?meaning[t.text]:t.kind==='role'?'An ordinary role-name token inside a delimited header, not a special token in this template.':t.kind==='answer'?'A token from the desired answer. SFT teaches the model to predict it from everything before it.':'An ordinary prompt token: context for the answer.'}`;
 $('#chat-mode-note').textContent=generation?'At generation time the string stops right after the assistant header. The model writes the rest one token at a time until it emits a stop token. No weights are updated.':'In a training example the correct assistant answer is already in the string. The model predicts each answer token from the prompt and the correct answer tokens before it.';
}
$('#chat-example').addEventListener('change',e=>{example=+e.target.value;renderChat(true);});
$$('[data-chat-mode]').forEach(b=>b.addEventListener('click',()=>{generation=b.dataset.chatMode==='generation';renderChat(true);}));
$('#chat-replay').addEventListener('click',()=>{$('#chat-lab').classList.remove('fresh');void $('#chat-lab').offsetWidth;renderChat(true);});
$('#chat-stream').addEventListener('click',e=>{const b=e.target.closest('[data-token]');if(!b)return;tokenIndex=+b.dataset.token;renderChat();$(`[data-token="${tokenIndex}"]`).focus({preventScroll:true});});

/* ---------- 02 · Candidate answers → training set ---------- */
function renderData(fresh=false){
 $('#demo-lab').dataset.stage=dataStage;$('#demo-lab').classList.toggle('fresh',fresh&&!reduced.matches);
 $('#data-stages').innerHTML=DATA_STAGES.map((s,i)=>`<button type="button" data-data-stage="${i}" aria-pressed="${dataStage===i}" class="${i<dataStage?'done':''}"><small>${i+1}</small>${esc(s.title)}</button>`).join('<span aria-hidden="true">→</span>');
 $('#data-note').textContent=DATA_STAGES[dataStage].note;
 $('#demo-records').innerHTML=DATA_RECORDS.map((r,i)=>{const rejected=dataStage>0&&!r.kept,held=dataStage===2&&r.split==='Held-out evaluation';
  const stamp=dataStage===0?'<span class="stamp idle">candidate</span>':rejected?`<span class="stamp no">✗ ${esc(r.reason)}</span>`:dataStage===1?'<span class="stamp yes">✓ accepted</span>':`<span class="stamp ${held?'hold':'yes'}">${held?'held out · evaluation':'training set'}</span>`;
  return `<div class="demo-record ${rejected?'rejected':''} ${held?'held-out':''}" style="--d:${i*130}ms"><small>${esc(r.source)}</small><div class="mini-chat"><p class="mini user">${esc(r.prompt)}</p><p class="mini assistant">${esc(r.answer)}</p></div>${stamp}</div>`;}).join('');
 const format=$('#demo-format');format.hidden=dataStage<2;
 if(dataStage===2)format.innerHTML=`<p class="string-label">THE TRAINING RECORD, AS THE TRAINER SEES IT · template applied, graded tokens marked</p><div class="chat-stream">${chatTokens(0).map((t,i)=>`<span class="chat-token ${kindClass(t)} ${t.supervised?'graded':''}" style="--d:${i*55+300}ms"><span>${esc(t.text)}</span><small>${t.supervised?'graded':t.id}</small></span>`).join('')}</div>`;
 $('#data-result').textContent=['4 candidates. A useful record demonstrates a response, not just a topic.','2 accepted · 1 duplicate removed · 1 incorrect answer rejected.','1 train record → template + token IDs + loss labels. 1 held-out record → evaluation only.'][dataStage];
 $('#data-prev').disabled=dataStage===0;$('#data-next').disabled=dataStage===DATA_STAGES.length-1;$('#data-next').textContent=dataStage<DATA_STAGES.length-1?`${DATA_STAGES[dataStage+1].title} →`:'Training set ready ✓';
}
const setDataStage=stage=>{dataStage=Math.max(0,Math.min(DATA_STAGES.length-1,stage));renderData(true);};
$('#data-stages').addEventListener('click',e=>{const b=e.target.closest('[data-data-stage]');if(!b)return;setDataStage(+b.dataset.dataStage);$(`[data-data-stage="${dataStage}"]`).focus({preventScroll:true});});
$('#data-prev').addEventListener('click',()=>setDataStage(dataStage-1));$('#data-next').addEventListener('click',()=>setDataStage(dataStage+1));

/* ---------- 03 · The loss mask ---------- */
const LOSS_TOP=3,barHeight=loss=>Math.max(3,Math.round(Math.min(1,loss/LOSS_TOP)*30));
function lossCurve(p,included){const w=230,h=96,x=v=>26+(w-34)*v,y=v=>8+(h-30)*(1-Math.min(v,LOSS_TOP)/LOSS_TOP);let d='';for(let i=5;i<=100;i++){const v=i/100;d+=`${i===5?'M':'L'}${x(v).toFixed(1)} ${y(-Math.log(v)).toFixed(1)}`;}
 return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Loss is minus the logarithm of the probability: ${p.toFixed(2)} gives ${(-Math.log(p)).toFixed(2)}"><line class="axis" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(0)}"/><line class="axis" x1="${x(0)}" y1="${y(0)}" x2="${x(0)}" y2="${y(LOSS_TOP)}"/><path class="curve" d="${d}"/><line class="guide" x1="${x(p)}" y1="${y(0)}" x2="${x(p)}" y2="${y(-Math.log(p))}"/><line class="guide" x1="${x(0)}" y1="${y(-Math.log(p))}" x2="${x(p)}" y2="${y(-Math.log(p))}"/><circle class="${included?'':'off'}" cx="${x(p)}" cy="${y(-Math.log(p))}" r="5"/><text x="${x(.5)}" y="${h-3}" text-anchor="middle">probability given to the right token →</text><text x="8" y="${y(LOSS_TOP/2)}" text-anchor="middle" transform="rotate(-90 8 ${y(LOSS_TOP/2)})">loss</text></svg>`;}
function renderLoss(fresh=false){
 const result=lossComparison(lossMode,override),row=result.rows.find(r=>r.batch===selected.batch&&r.position===selected.position);
 $('#sft-lab').classList.toggle('fresh',fresh&&!reduced.matches);
 $$('[data-loss-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.lossMode===lossMode)));
 const flows=(batch,position)=>row.included&&batch===row.batch&&position<row.position;
 $('#loss-streams').innerHTML=INSTRUCT_EXAMPLES.map((_,batch)=>`<div class="loss-sequence"><small>SEQUENCE ${batch+1}</small><div class="loss-token-row"><span class="loss-token bos ${flows(batch,0)?'flow':''}" style="--d:${row.position*70}ms"><span>&lt;bos&gt;</span><small>start</small><i class="loss-bar none"></i></span>${result.rows.filter(r=>r.batch===batch).map(r=>`<button type="button" data-loss-batch="${batch}" data-loss-position="${r.position}" style="--d:${(row.position-r.position)*70}ms" class="loss-token ${r.included?'scored':''} ${flows(batch,r.position)?'flow':''}" aria-pressed="${r.batch===selected.batch&&r.position===selected.position}" aria-label="Sequence ${batch+1}, target ${r.position}: ${esc(r.text)}, ${r.included?`graded, loss ${r.loss.toFixed(2)}`:'context only'}"><span>${esc(r.text)}</span><small>${r.included?r.loss.toFixed(2):'not graded'}</small><i class="loss-bar ${r.included?'':'none'}" style="height:${r.included?barHeight(r.loss):3}px"></i></button>`).join('')}</div></div>`).join('');
 $('#selected-title').textContent=`Guessing “${row.text}” · sequence ${row.batch+1}`;
 $('#selected-context').textContent=`The model has read: ${row.context.map(t=>t.text).join(' ')}`;
 $('#loss-curve-plot').innerHTML=lossCurve(row.p,row.included);
 $('#target-probability').value=Math.round(row.p*100);$('#target-probability-label').textContent=`${Math.round(row.p*100)}%`;
 $('#selected-loss').textContent=row.included?`Graded: −ln(${row.p.toFixed(2)}) = ${row.loss.toFixed(3)}`:`Not graded: 0 × −ln(${row.p.toFixed(2)}) = 0 (it would have been ${row.loss.toFixed(3)})`;
 $('#sft-meter').innerHTML=result.rows.map(r=>`<i class="${r.included?'on':''} ${r===row?'sel':''}" style="height:${r.included?barHeight(r.loss)+6:4}px" title="${esc(r.text)}: ${r.included?r.loss.toFixed(2):'not graded'}"></i>`).join('');
 $('#sft-loss').textContent=`${result.count} graded of ${result.rows.length} guesses → ${result.sum.toFixed(3)} / ${result.count} = ${result.loss.toFixed(3)} nats`;
 const grad=logitGradient(row,result.count)[row.id];
 $('#sft-gradient').textContent=row.included?`This guess is graded, so its error flows backward through every token it could see, including the ungraded prompt (purple). At its own score: ∂L/∂z = (${row.p.toFixed(2)} − 1) / ${result.count} = ${grad.toFixed(4)}, a push to make “${row.text}” more likely next time.`:'This guess is not graded (m = 0), so no error starts here. The token is still read as context: errors from later assistant tokens flow back through it.';
}
$$('[data-loss-mode]').forEach(b=>b.addEventListener('click',()=>{lossMode=b.dataset.lossMode;renderLoss(true);}));
$('#loss-streams').addEventListener('click',e=>{const b=e.target.closest('[data-loss-position]');if(!b)return;selected={batch:+b.dataset.lossBatch,position:+b.dataset.lossPosition};override=null;renderLoss(true);$(`[data-loss-batch="${selected.batch}"][data-loss-position="${selected.position}"]`).focus({preventScroll:true});});
$('#target-probability').addEventListener('input',e=>{override={...selected,p:+e.target.value/100};renderLoss();});
document.addEventListener('click',e=>{if(e.target.closest('a[href^="#ref-"]'))$('#sources').open=true;});
renderChat(true);renderData();renderLoss(true);
