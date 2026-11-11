import {exampleCorpus,htmlSegments,corpusTape} from './pretraining-data.mjs';
import {bpeTrace,encodeBpe,encodeSteps,pairOccurrences} from './pretraining-bpe.mjs';
import {PRETRAIN_VOCAB,PRETRAIN_BATCH,PRETRAIN_PHASES,createPretrainModel,pretrainForward,applyPretrainSgd} from './pretraining-engine.mjs';
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const show=s=>s.replaceAll('</w>','⟂'),symbol=s=>esc(show(s)),fmt=(n,d=4)=>Number(n).toFixed(d);
const reduced=matchMedia('(prefers-reduced-motion: reduce)');

/* ---------- 01 · Data: HTML string → text → one corpus string ---------- */
const DATA_STEPS=[
 {name:'Crawl',note:'A crawler downloads each page. What arrives is not an article: it is one long string of HTML.'},
 {name:'Extract text',note:'Strip the tags, scripts, menus and footers. Only the main text of the page survives.'},
 {name:'Filter',note:'Not every page should be learned from. Drop spam and anything not approved for training.'},
 {name:'Deduplicate',note:'The same sentence was copied to another site. Keep one copy, so the model does not over-learn it.'},
 {name:'Join & split',note:'Glue the surviving documents into one long string, with a special token marking where each one ends. Hold one document back to test on later.'}
];
const corpus=exampleCorpus(),tape=corpusTape();
let dataStage=0,dataSelected='a',dataTimer=null,dataPlaying=false;
function recordState(r,stage){
 const duplicate=r.allowed&&r.quality&&!corpus.unique.some(x=>x.id===r.id);
 if(stage>=2&&!r.quality)return {kept:false,status:'removed · spam'};
 if(stage>=2&&!r.allowed)return {kept:false,status:'removed · not approved'};
 if(stage>=3&&duplicate)return {kept:false,status:'removed · duplicate of page 1'};
 if(stage===4){const doc=tape.docs.find(d=>d.id===r.id);return {kept:true,split:doc.split,status:doc.split==='validation'?'held out · validation':`training · shard ${doc.shard}`};}
 return {kept:true,status:stage===0?`${r.html.length} chars of HTML`:`${r.text.length} chars of text`};
}
function renderData(fresh=false){
 const lab=$('#data-lab'),record=corpus.raw.find(r=>r.id===dataSelected),state=recordState(record,dataStage);
 lab.dataset.stage=dataStage;lab.classList.toggle('fresh',fresh&&!reduced.matches);
 $('#corpus-stages').innerHTML=DATA_STEPS.map((s,i)=>`<button type="button" data-corpus-stage="${i}" aria-pressed="${i===dataStage}" class="${i<dataStage?'done':''}"><small>${i+1}</small>${s.name}</button>`).join('<span aria-hidden="true">→</span>');
 $('#corpus-explanation').textContent=DATA_STEPS[dataStage].note;
 $('#page-url').textContent='https://'+record.url;
 $('#page-label').textContent=dataStage===0?`RAW HTML · ONE STRING · ${record.html.length} CHARACTERS`:`EXTRACTED TEXT · ${record.text.length} CHARACTERS (WAS ${record.html.length})${state.kept?'':' · '+state.status.toUpperCase()}`;
 let delay=0;
 $('#html-string').className=`html-string${dataStage>=1?' stripped':''}${state.kept?'':' rejected'}`;
 $('#html-string').innerHTML=htmlSegments(record.html).map(seg=>`<span class="seg seg-${seg.kind}"${seg.kind==='content'?'':` style="--d:${(delay++)*45}ms"`}>${esc(seg.text)}</span>`).join('');
 $('#corpus-documents').innerHTML=corpus.raw.map((r,i)=>{const st=recordState(r,dataStage);
  return `<button type="button" data-record="${r.id}" aria-pressed="${r.id===dataSelected}" class="corpus-record ${st.kept?'':'excluded'} ${st.split||''}"><span class="doc-num">${i+1}</span><code>${esc(dataStage===0?r.html:r.text)}</code><span class="doc-status">${esc(st.status)}</span></button>`;}).join('');
 const tapeEl=$('#corpus-tape');tapeEl.hidden=dataStage<4;
 if(dataStage===4){
  const train=tape.docs.filter(d=>d.split==='train');
  $('#tape-train').innerHTML='<b>train</b>'+train.map((d,i)=>`${i>0&&d.shard!==train[i-1].shard?'<span class="cut" title="Shard boundary: a storage cut, not a batch">✂</span>':''}<span class="tape-doc shard-${d.shard}" style="--d:${i*320}ms">${esc(d.text)}</span><span class="tape-sep" style="--d:${i*320+160}ms">${esc(tape.separator)}</span>`).join('');
  $('#tape-validation').innerHTML='<b>validation</b>'+tape.docs.filter(d=>d.split==='validation').map(d=>`<span class="tape-doc" style="--d:${train.length*320}ms">${esc(d.text)}</span><span class="tape-sep" style="--d:${train.length*320+160}ms">${esc(tape.separator)}</span>`).join('');
 }
 const textChars=corpus.raw.reduce((n,r)=>n+r.text.length,0),kept=corpus.raw.filter(r=>recordState(r,dataStage).kept).length;
 $('#corpus-result').textContent=[
  `7 pages downloaded: ${tape.htmlChars} characters of HTML in total.`,
  `${tape.htmlChars} characters of HTML → ${textChars} characters of text. About ${Math.round(100*(1-textChars/tape.htmlChars))}% of the download was markup and page chrome.`,
  `${kept} of 7 pages remain. In this miniature the spam and approval flags are hand-assigned, not real classifiers.`,
  `${kept} of 7 pages remain. Real pipelines also catch near-duplicates, not only exact copies.`,
  `Training corpus: one string of ${tape.train.length} characters, stored in 2 shards. Validation: ${tape.validation.length} characters the model never trains on.`][dataStage];
 $('#corpus-prev').disabled=dataStage===0;$('#corpus-next').disabled=dataStage===DATA_STEPS.length-1;
 $('#corpus-next').textContent=dataStage<DATA_STEPS.length-1?`${DATA_STEPS[dataStage+1].name} →`:'Corpus ready ✓';
 $('#corpus-play').textContent=dataPlaying?'Pause':'▶ Play all';
}
function stopData(){if(dataTimer!==null)clearTimeout(dataTimer);dataTimer=null;dataPlaying=false;}
function setDataStage(stage){dataStage=Math.max(0,Math.min(DATA_STEPS.length-1,stage));renderData(true);}
function playData(){dataTimer=setTimeout(()=>{dataTimer=null;if(!dataPlaying||document.hidden||dataStage>=DATA_STEPS.length-1){stopData();renderData();return;}setDataStage(dataStage+1);if(dataStage>=DATA_STEPS.length-1){stopData();renderData();}else playData();},reduced.matches?600:2400);}
$('#corpus-stages').addEventListener('click',e=>{const b=e.target.closest('[data-corpus-stage]');if(!b)return;stopData();setDataStage(+b.dataset.corpusStage);$(`[data-corpus-stage="${dataStage}"]`).focus({preventScroll:true});});
$('#corpus-documents').addEventListener('click',e=>{const b=e.target.closest('[data-record]');if(!b)return;dataSelected=b.dataset.record;renderData();$(`[data-record="${dataSelected}"]`).focus({preventScroll:true});});
$('#corpus-prev').addEventListener('click',()=>{stopData();setDataStage(dataStage-1);});
$('#corpus-next').addEventListener('click',()=>{stopData();setDataStage(dataStage+1);});
$('#corpus-play').addEventListener('click',()=>{if(dataPlaying){stopData();renderData();return;}dataPlaying=true;setDataStage(dataStage>=DATA_STEPS.length-1?0:dataStage);playData();});

/* ---------- 02 · Tokenizer: animated BPE merges ---------- */
const BPE_START=bpeTrace(0),BPE_MAX=12,SAMPLE_WORDS=['lowest','slowest','wildest','newer'];
let merges=0,bpeAnimating=false,bpeTimer=null,bpePlaying=false;
const hue=rank=>(150+rank*53)%360;
function tile(token,ranks,extra='',note=''){const rank=ranks.get(token);
 return `<span class="token-piece ${rank===undefined?'':'learned'} ${extra}"${rank===undefined?'':` style="--h:${hue(rank)}"`}>${symbol(token)}${note}</span>`;}
function renderBpe(fresh=false){
 const trace=bpeTrace(merges),best=trace.pairs[0],ranks=new Map(trace.merges.map((m,i)=>[m.pair.join(''),i])),newest=fresh?trace.merges.at(-1)?.pair.join(''):null;
 $('#bpe-lab').classList.toggle('merging',bpeAnimating);
 $('#bpe-words').innerHTML=trace.words.map(w=>{const hits=bpeAnimating&&best?new Set(pairOccurrences(w.tokens,best.pair)):new Set();
  return `<div class="bpe-word"><strong>${w.word}<small>× ${w.count}</small></strong><div>${w.tokens.map((t,i)=>tile(t,ranks,hits.has(i)?'pair-a':hits.has(i-1)?'pair-b':t===newest?'fused':'')).join('')}</div></div>`;}).join('');
 $('#bpe-pairs').innerHTML=trace.pairs.slice(0,6).map((p,i)=>`<div class="bpe-pair ${i===0?'winning-pair':''}"><span class="pair-tiles">${tile(p.pair[0],ranks)}${tile(p.pair[1],ranks)}</span><span class="pair-track"><i style="width:${100*p.count/best.count}%"></i></span><strong>${p.count}×</strong></div>`).join('')||'<p class="small">Every word is a single token. Nothing left to merge.</p>';
 $('#bpe-count').textContent=trace.tokenCount;$('#bpe-count-bar').style.width=`${100*trace.tokenCount/BPE_START.tokenCount}%`;
 $('#bpe-vsize').textContent=trace.vocab.length;$('#bpe-vsize-bar').style.width=`${100*trace.vocab.length/(BPE_START.vocab.length+BPE_MAX)}%`;$('#bpe-mcount').textContent=trace.merges.length;
 const last=trace.merges.at(-1);
 $('#bpe-status').textContent=bpeAnimating?`Fusing every “${show(best.pair[0])}” “${show(best.pair[1])}” that sit next to each other…`:last?`Merge ${trace.merges.length}: “${show(last.pair[0])}” + “${show(last.pair[1])}” → “${show(last.pair.join(''))}”. The text is now ${trace.tokenCount} tokens instead of ${BPE_START.tokenCount}, using ${trace.vocab.length} vocabulary entries.`:`Start: every character is its own token. ${BPE_START.tokenCount} tokens, ${BPE_START.vocab.length} vocabulary entries.`;
 const more=!!best&&merges<BPE_MAX;
 $('#bpe-next').disabled=!more||bpeAnimating;$('#bpe-next').textContent=more?`Merge “${show(best.pair[0])}” + “${show(best.pair[1])}” → “${show(best.pair.join(''))}”`:'All merges done';
 $('#bpe-play').disabled=!more&&!bpePlaying;$('#bpe-play').textContent=bpePlaying?'Pause':'▶ Auto-merge';
 $('#bpe-vocab').innerHTML=trace.vocab.map((t,i)=>tile(t,ranks,t===newest?'fused':'',`<small>${i}</small>`)).join('');
 $('#bpe-merges').innerHTML=trace.merges.map(m=>`<li><code>${m.pair.map(symbol).join(' + ')} → ${symbol(m.pair.join(''))}</code></li>`).join('')||'<li>No merges yet: start from characters plus ⟂.</li>';
 $('#bpe-samples').innerHTML=SAMPLE_WORDS.map(w=>`<button type="button" class="quiet" data-sample="${w}">${w}</button>`).join('');
 renderEncode(trace,ranks);
}
function renderEncode(trace=bpeTrace(merges),ranks=new Map(trace.merges.map((m,i)=>[m.pair.join(''),i]))){
 const word=$('#bpe-word').value.trim().toLowerCase(),out=$('#bpe-encoded');
 if(!word){out.innerHTML='<p class="small">Type a word.</p>';return;}
 let steps,encoded;try{steps=encodeSteps(word,trace);encoded=encodeBpe(word,trace);}catch{out.innerHTML=`<p class="small">This toy tokenizer only knows the letters ${BPE_START.vocab.filter(c=>c.length===1).join(' ')}. Try one of the samples.</p>`;return;}
 out.innerHTML=steps.map((s,i)=>`<div class="encode-row" style="--d:${i*140}ms"><small>${s.rule===null?'characters':`rule ${s.rule+1}: ${symbol(s.pair[0])} + ${symbol(s.pair[1])}`}</small><div>${s.tokens.map(t=>tile(t,ranks)).join('')}</div></div>`).join('')
  +`<div class="encode-row ids" style="--d:${steps.length*140}ms"><small>token IDs</small><div>${encoded.ids.map(id=>`<span class="token-id">${id}</span>`).join('')}</div></div><p class="small">${word.length} characters → ${encoded.tokens.length} token${encoded.tokens.length===1?'':'s'}. ${steps.length-1} of ${trace.merges.length} learned rules applied, in the order they were learned.</p>`;
}
function stopBpe(){if(bpeTimer!==null)clearTimeout(bpeTimer);bpeTimer=null;bpeAnimating=false;bpePlaying=false;}
function mergeOnce(){
 if(bpeAnimating||merges>=BPE_MAX||!bpeTrace(merges).pairs[0])return;
 const finish=()=>{bpeTimer=null;bpeAnimating=false;merges++;renderBpe(true);
  if(bpePlaying){if(document.hidden||merges>=BPE_MAX||!bpeTrace(merges).pairs[0]){bpePlaying=false;renderBpe();}else bpeTimer=setTimeout(()=>{bpeTimer=null;mergeOnce();},reduced.matches?500:900);}};
 if(reduced.matches){finish();return;}
 bpeAnimating=true;renderBpe();bpeTimer=setTimeout(finish,850);
}
$('#bpe-next').addEventListener('click',()=>{bpePlaying=false;mergeOnce();});
$('#bpe-play').addEventListener('click',()=>{if(bpePlaying){stopBpe();renderBpe();return;}bpePlaying=true;mergeOnce();renderBpe();});
$('#bpe-reset').addEventListener('click',()=>{stopBpe();merges=0;renderBpe();});
$('#bpe-word').addEventListener('input',()=>renderEncode());
$('#bpe-samples').addEventListener('click',e=>{const b=e.target.closest('[data-sample]');if(!b)return;$('#bpe-word').value=b.dataset.sample;renderEncode();});

/* ---------- 03 · Learn: forward, loss, backward, update ---------- */
let model=createPretrainModel(),previousModel=null,batchSize=2,step=0,phase=0,selected=2,snapshot,after=null,timer=null,epoch=0,playing=false,history=[];
const batch=()=>PRETRAIN_BATCH.slice(0,batchSize),RATE=.1;
function prepare(){snapshot=pretrainForward(model,batch());after=null;if(!history.length)history=[snapshot.loss];}
function strip(values,label,{grad=false}={}){const scale=grad?Math.max(...values.map(Math.abs),1e-12):2;
 return `<span class="feature-strip ${grad?'grad':''}" role="img" aria-label="${esc(label)}"><span class="sr-only">${esc(label)}</span>${values.map((v,i)=>`<i style="--magnitude:${Math.min(1,Math.abs(v)/scale)};--sign:${v<0?'var(--negative)':grad?'var(--backward)':'var(--blue)'}" title="${esc(label)}[${i}] = ${v.toPrecision(4)}"></i>`).join('')}</span>`;}
function heat(matrix,scale,mark,label,positive='var(--blue)'){return `<div class="heat" role="img" aria-label="${esc(label)}" style="--cols:${matrix[0].length}">${matrix.flatMap((row,r)=>row.map((v,c)=>`<i class="${mark&&mark[0]===r&&mark[1]===c?'mark':''}" style="--magnitude:${Math.min(1,Math.abs(v)/scale)};--sign:${v<0?'var(--negative)':positive}" title="[${r},${c}] = ${v.toPrecision(4)}"></i>`)).join('')}</div>`;}
function lossCurve(){const w=520,h=64,max=Math.max(...history,1e-9),n=Math.max(history.length-1,1),x=i=>4+(w-8)*i/n,y=v=>h-5-(h-12)*v/max;
 return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Batch loss after each of ${history.length-1} updates"><line x1="4" y1="${h-5}" x2="${w-4}" y2="${h-5}"/>${history.length>1?`<polyline points="${history.map((v,i)=>`${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')}"/>`:''}<circle cx="${x(history.length-1).toFixed(1)}" cy="${y(history.at(-1)).toFixed(1)}" r="3.5"/></svg>`;}
function cancelPlayback(){epoch++;if(timer!==null)clearTimeout(timer);timer=null;playing=false;}
function renderTraining(fresh=false){
 const trace=snapshot.traces[selected],forward=phase>=1,lossShown=phase>=2,backward=phase>=3,updated=phase===4,root=$('#training-lab'),seq=batch()[trace.batch].slice(0,-1);
 root.dataset.phase=PRETRAIN_PHASES[phase];root.classList.toggle('fresh',fresh&&!reduced.matches);
 $('#training-phases').innerHTML=PRETRAIN_PHASES.map((p,i)=>`<span class="${i===phase?'current':i<phase?'done':''}"><small>${i+1}</small>${['Batch','Forward','Loss','Backward','Update'][i]}</span>`).join('');
 $('#training-batch').innerHTML=batch().map((ids,b)=>`<div class="batch-row"><span>Sequence ${b+1}</span><div class="sequence-line"><small>input</small>${ids.slice(0,-1).map((id,t)=>`<button type="button" data-train-select="${b*5+t}" class="${selected===b*5+t?'selected':''} ${b===trace.batch&&t<=trace.position?'context-visible':b===trace.batch?'masked':''}" aria-pressed="${selected===b*5+t}" aria-label="Inspect sequence ${b+1}, position ${t}: ${esc(PRETRAIN_VOCAB[id])} predicts ${esc(PRETRAIN_VOCAB[ids[t+1]])}">${esc(PRETRAIN_VOCAB[id])}</button>`).join('')}</div><div class="sequence-line shift" aria-hidden="true"><small></small>${ids.slice(1).map((_,t)=>`<span class="${selected===b*5+t?'target-selected':''}">↓</span>`).join('')}</div><div class="sequence-line targets"><small>target</small>${ids.slice(1).map((id,t)=>`<span class="${selected===b*5+t?'target-selected':''}">${esc(PRETRAIN_VOCAB[id])}</span>`).join('')}</div></div>`).join('');
 $('#training-context').innerHTML=`The target row is the input row shifted by one token. Here the model sees <strong>${trace.context.map(id=>esc(PRETRAIN_VOCAB[id])).join(' ')}</strong> and must guess <strong>${esc(PRETRAIN_VOCAB[trace.target])}</strong>. Click any token to inspect that position.`;
 $('#embedding-strip').innerHTML=forward?strip(trace.embedding,'Embedding lookup, before update'):'<span class="pending-math">waiting for tokens</span>';
 $('#attention-strip').innerHTML=`<span class="attn-chips">${seq.map((id,t)=>t<=trace.position?`<span class="attn-chip" style="--w:${forward?trace.attention[t]:0}" title="${forward?`attention weight ${(trace.attention[t]*100).toFixed(1)}%`:'visible to this position'}">${esc(PRETRAIN_VOCAB[id])}${forward?`<small>${Math.round(trace.attention[t]*100)}%</small>`:''}</span>`:`<span class="attn-chip masked" title="Future token: hidden by the causal mask">${esc(PRETRAIN_VOCAB[id])}</span>`).join('')}</span>`;
 $('#hidden-strip').innerHTML=forward?strip(trace.hidden,'Hidden state, before update'):'<span class="pending-math">mix and transform</span>';
 $('#logits-strip').innerHTML=forward?strip(trace.logits,'Vocabulary logits, before update'):'<span class="pending-math">one score per token</span>';
 for(const [id,key,name] of [['embedding-grad','embedding','embedding'],['attention-grad','attention','attention output'],['hidden-grad','mlp','MLP output'],['logits-grad','logits','logits']])$('#'+id).innerHTML=backward?`<small>gradient</small>${strip(trace.activationGradients[key],`Gradient of the batch loss at the ${name}`,{grad:true})}`:'';
 $('#loss-node-value').textContent=lossShown?fmt(trace.loss,3):'—';$('#loss-node-note').textContent=lossShown?`this token · batch mean ${fmt(snapshot.loss,3)}`:'how wrong was the guess?';
 const probs=updated?after.traces[selected].probabilities:trace.probabilities,logitGrad=trace.activationGradients.logits,gmax=Math.max(...logitGrad.map(Math.abs),1e-12);
 $('#prediction-title').textContent=`Guessing what follows “${trace.context.map(id=>PRETRAIN_VOCAB[id]).join(' ')}”`;
 $('#prediction-bars').innerHTML=PRETRAIN_VOCAB.map((token,i)=>{const delta=(after?.traces[selected].probabilities[i]-trace.probabilities[i])*100,up=logitGrad[i]<0;
  return `<div class="prediction-row ${i===trace.target?'correct-target':''}"><span>${esc(token)}${i===trace.target?' <small>target</small>':''}</span><span class="probability-track"><i style="width:${forward?probs[i]*100:0}%"></i>${updated?`<b class="before-tick" style="left:${trace.probabilities[i]*100}%" title="before the update"></b>`:''}</span><strong>${forward?fmt(probs[i]*100,1)+'%':'—'}</strong><em class="nudge ${up?'up':'down'}">${phase===3?`<span style="--s:${.55+.45*Math.abs(logitGrad[i])/gmax}">${up?'▲':'▼'}</span>`:updated?`${delta>=0?'+':'−'}${Math.abs(delta).toFixed(1)}`:''}</em></div>`;}).join('');
 $('#token-loss').textContent=!forward?'The forward pass will turn 8 scores into 8 probabilities.':!lossShown?`The model gave the correct token ${fmt(trace.probabilities[trace.target]*100,1)}%.`:phase===2?`ℓ = −ln p(target) = −ln ${fmt(trace.probabilities[trace.target],4)} = ${fmt(trace.loss,3)}. A confident correct guess would score near 0.`:phase===3?'Gradient at each score = p − 1 for the target, p for every other token: push the target up ▲, everything else down ▼.':`After one update the correct token went from ${fmt(trace.probabilities[trace.target]*100,1)}% to ${fmt(after.traces[selected].probabilities[trace.target]*100,1)}%. The tick marks the old value.`;
 $('#loss-grid').innerHTML=snapshot.traces.map((tr,i)=>`<button type="button" data-train-select="${i}" aria-pressed="${i===selected}" class="${i===selected?'selected':''}"><small>s${tr.batch+1} · t${tr.position}</small><span>${esc(PRETRAIN_VOCAB[tr.target])}</span><b>${lossShown?fmt(tr.loss,2):'—'}</b></button>`).join('');
 $('#batch-equation').textContent=lossShown?`L = average of ${snapshot.count} token losses = ${fmt(snapshot.loss,4)} nats`:`${batchSize} sequence${batchSize>1?'s':''} × 5 targets = ${snapshot.count} guesses, all scored together.`;
 $('#loss-curve').innerHTML=lossCurve();$('#curve-value').textContent=`${fmt(history[0],3)} → ${fmt(history.at(-1),3)}`;
 const first=snapshot.grads.lm.flat().map((g,i)=>({g,i})).sort((a,b)=>Math.abs(b.g)-Math.abs(a.g))[0],r=Math.floor(first.i/8),c=first.i%8,old=updated?previousModel.lm:model.lm,wScale=Math.max(...old.flat().map(Math.abs)),gScale=Math.max(...snapshot.grads.lm.flat().map(Math.abs),1e-12);
 $('#weight-update').innerHTML=!backward?'<span class="small">After the backward pass, every weight knows which way to move. The LM head’s 4 × 8 weights will appear here as an example.</span>':
  `<strong>${updated?'Every weight took one small step':'Each weight now has a gradient'} <small>LM head shown · 32 of 192 weights · one column per vocabulary token</small></strong><div class="update-row"><figure>${heat(old,wScale,[r,c],'LM head weights before the update')}<figcaption>W<small>weights</small></figcaption></figure><span class="op">−</span><span class="op eta">0.1 ×</span><figure class="grad">${heat(snapshot.grads.lm,gScale,[r,c],'Gradient of the batch loss for each LM head weight','var(--backward)')}<figcaption>∂L/∂W<small>gradient</small></figcaption></figure>${updated?`<span class="op">=</span><figure class="new">${heat(model.lm,wScale,[r,c],'LM head weights after the update')}<figcaption>W′<small>new weights</small></figcaption></figure>`:''}</div><code>outlined weight: ${fmt(old[r][c],4)} − 0.1 × ${fmt(first.g,4)} = ${fmt(old[r][c]-RATE*first.g,4)}</code>${updated?`<p>Same batch, new weights: loss <b>${fmt(snapshot.loss)} → ${fmt(after.loss)}</b>. This is training-batch fit, not a validation score.</p>`:'<small>Positive gradient: lowering the weight lowers the loss. Negative: raising it does. The step is the gradient times the learning rate 0.1.</small>'}`;
 $('#train-next').textContent=['Forward pass →','Measure the loss →','Backward pass ←','Update weights →','Run the batch again →'][phase];$('#train-next').disabled=playing;$('#train-fast').disabled=playing;$('#train-play').textContent=playing?'Pause':'▶ Play one update';$('#train-counter').textContent=`${step} update${step===1?'':'s'} · SGD, learning rate 0.1`;
 $('#train-status').textContent=['Batch ready. Each input token will be asked: what comes next?','Forward: tokens flow left to right. Each stage turns numbers into new numbers, ending in a probability for every vocabulary token.','Loss: one number for how surprised the model was by the right answers, averaged over the batch.','Backward: the loss flows right to left along the same path. Each stage learns how it contributed to the error.','Update: every weight moves a little against its gradient. Run the same batch again and the model is slightly less wrong.'][phase];
}
function advance(){if(phase===4){phase=0;prepare();}else{phase++;if(phase===4){previousModel=model;model=applyPretrainSgd(model,snapshot.grads,RATE);after=pretrainForward(model,batch(),{gradients:false});step++;history.push(after.loss);}}renderTraining(true);}
function queue(){const ticket=epoch;timer=setTimeout(()=>{timer=null;if(ticket!==epoch||!playing)return;if(document.hidden){cancelPlayback();renderTraining();return;}advance();if(phase===4){cancelPlayback();renderTraining();return;}queue();},reduced.matches?0:[1200,2300,1500,2300][phase]);}
function trainFast(count=30){cancelPlayback();if(phase===4)prepare();playing=true;const ticket=epoch;
 const one=()=>{model=applyPretrainSgd(model,snapshot.grads,RATE);step++;snapshot=pretrainForward(model,batch());after=null;history.push(snapshot.loss);phase=2;};
 const tick=left=>{if(ticket!==epoch||!playing)return;if(left===0||document.hidden){cancelPlayback();renderTraining();return;}one();renderTraining();timer=setTimeout(()=>tick(left-1),55);};
 if(reduced.matches){for(let i=0;i<count;i++)one();cancelPlayback();renderTraining();}else tick(count);}
$('#train-next').addEventListener('click',()=>{cancelPlayback();advance();});
$('#train-play').addEventListener('click',()=>{if(playing){cancelPlayback();renderTraining();return;}if(phase===4){phase=0;prepare();}playing=true;renderTraining();queue();});
$('#train-fast').addEventListener('click',()=>trainFast());
$('#train-reset').addEventListener('click',()=>{cancelPlayback();model=createPretrainModel();previousModel=null;step=0;phase=0;history=[];prepare();renderTraining();});
$('#train-batch').addEventListener('change',e=>{cancelPlayback();batchSize=+e.target.value;selected=Math.min(selected,batchSize*5-1);phase=0;history=[];prepare();renderTraining();});
$('#training-lab').addEventListener('click',e=>{const button=e.target.closest('[data-train-select]');if(!button)return;const scope=button.closest('#loss-grid')?'#loss-grid':'#training-batch';selected=+button.dataset.trainSelect;renderTraining();$(`${scope} [data-train-select="${selected}"]`).focus({preventScroll:true});});
document.addEventListener('visibilitychange',()=>{if(!document.hidden)return;if(playing){cancelPlayback();renderTraining();}if(dataPlaying){stopData();renderData();}if(bpePlaying){stopBpe();renderBpe();}});
reduced.addEventListener('change',()=>{if(playing){cancelPlayback();renderTraining();}});
// Internal citations open their otherwise collapsed source list.
document.addEventListener('click',e=>{if(e.target.closest('a[href^="#ref-"]'))$('#sources').open=true;});
renderData();renderBpe();prepare();renderTraining();
