// Text transformers (GPT-2): explorer and section labs.
// All numbers come from src/text-data.mjs (recorded from GPT-2) or are computed
// from them in src/text-engine.mjs. Nothing here runs a model.
import {EXAMPLES} from './text-data.mjs';
import {attentionStages,probabilities,sample,showToken,illustrativeVector,HEAD_DIM} from './text-engine.mjs';

const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=p=>p>=0.9995?'100%':p<=0?'0%':(p*100).toFixed(p<0.1?2:1)+'%';
const BLOCKS=12,HEADS=12;
const state={ex:0,block:0,head:0,temperature:0.8,type:'top-k',k:5,p:0.9,rnd:0.3,tok:0,sel:null};
const ex=()=>EXAMPLES[state.ex];
const sampling=()=>state.type==='top-k'?{type:'top-k',value:state.k}:{type:'top-p',value:state.p};
const rows=()=>probabilities(ex().top,{temperature:state.temperature,sampling:sampling()});

/* ---------- controls shared by the explorer and the probabilities lab ---------- */
function controlsMarkup(i){
 return `<label><span>Temperature</span><span class="row"><input type="range" min="0.1" max="2" step="0.1" data-c="temperature" aria-label="Temperature"><output data-o="temperature"></output></span></label>
<fieldset><legend>Sampling</legend><span class="row"><span class="radios"><label><input type="radio" name="tx-type-${i}" value="top-k" data-c="type">Top-k</label><label><input type="radio" name="tx-type-${i}" value="top-p" data-c="type">Top-p</label></span><input type="range" data-c="amount" aria-label="Top-k or top-p value"><output data-o="amount"></output></span></fieldset>`;
}
function syncControls(){
 $$('[data-c=temperature]').forEach(el=>{el.value=state.temperature;});
 $$('[data-o=temperature]').forEach(el=>{el.textContent=state.temperature.toFixed(1);});
 $$('[data-c=type]').forEach(el=>{el.checked=el.value===state.type;});
 $$('[data-c=amount]').forEach(el=>{
  if(state.type==='top-k'){el.min=1;el.max=10;el.step=1;el.value=state.k;}
  else{el.min=0.05;el.max=1;el.step=0.05;el.value=state.p;}
 });
 $$('[data-o=amount]').forEach(el=>{el.textContent=state.type==='top-k'?`k=${state.k}`:`p=${state.p.toFixed(2)}`;});
}
function syncSteppers(){
 $$('div[data-stepper]').forEach(el=>{
  const kind=el.dataset.stepper,label=kind==='block'?`Block ${state.block+1} of ${BLOCKS}`:`Head ${state.head+1} of ${HEADS}`;
  el.innerHTML=`<button type="button" data-step="-1" aria-label="Previous ${kind}">‹</button><b>${label}</b><button type="button" data-step="1" aria-label="Next ${kind}">›</button>`;
 });
}

/* ---------- explorer flow diagram ---------- */
const W=1230,Y0=106,PITCH=54,TP=13;
const blockLabel=()=>`TRANSFORMER BLOCK ${state.block+1} / ${BLOCKS}`,headLabel=()=>`MULTI-HEAD SELF-ATTENTION, HEAD ${state.head+1} / ${HEADS}`;
// A label with ‹ › buttons that sits on a frame's top edge (the Vision page's pattern).
const chip=(kind,x,y,label,w)=>{
 const btn=(d,bx)=>`<g class="step" data-step="${d}" role="button" tabindex="0" aria-label="${d>0?'Next':'Previous'} ${kind}"><rect x="${bx}" y="${y-9}" width="18" height="18" rx="4"/><text x="${bx+9}" y="${y+4.5}" text-anchor="middle">${d>0?'›':'‹'}</text></g>`;
 return `<g class="chip" data-stepper="${kind}"><rect class="chip-bg" x="${x-7}" y="${y-11}" width="${w+64}" height="22"/>${btn(-1,x)}<text class="chip-label ${kind}-label" x="${x+25+w/2}" y="${y+3.5}" text-anchor="middle">${label}</text>${btn(1,x+32+w)}</g>`;
};
const curve=(x1,y1,x2,y2)=>{const m=(x1+x2)/2;return `M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}`;};
function flow(){
 const {tokens,attn}=ex(),n=tokens.length,H=Y0+n*PITCH+48;
 const yc=i=>Y0+i*PITCH+PITCH/2;
 const {weights}=attentionStages(attn[state.block][state.head]);
 const list=rows(),picked=sample(list,state.rnd);
 const top=86,bottom=H-30,ch=bottom-top,fx=218,fy=48,fw=712,fh=H-fy-8;
 const gy={k:top+ch*0.2,q:top+ch*0.5,v:top+ch*0.82};
 const ty=(g,i)=>gy[g]+(i-(n-1)/2)*TP;
 const mx0=548,cx=j=>mx0+j*TP,mtop=ty('q',0)-TP/2-3,outX=716;
 const names={q:'Query',k:'Key',v:'Value'};
 let s=`<title id="tx-flow-title">A sentence flowing through embedding, transformer block ${state.block+1} and the output probabilities</title>`;
 // column titles
 s+=`<g class="stage" style="--d:0"><text class="col-title" x="20" y="30">Embedding</text></g>`;
 s+=`<g class="stage" style="--d:1"><rect class="frame" x="${fx}" y="${fy}" width="${fw}" height="${fh}" rx="10"/>${chip('block',fx+18,fy,blockLabel(),176)}</g>`;
 s+=`<g class="stage" style="--d:4"><text class="col-sub" x="806" y="${fy+34}">MLP</text><text class="col-sub" x="944" y="46">11 MORE</text><text class="col-sub" x="944" y="58">BLOCKS</text></g>`;
 s+=`<g class="stage" style="--d:6"><text class="col-title" x="1052" y="30">Probabilities</text></g>`;
 // embedding column + Q/K/V stubs
 s+=`<g class="stage" style="--d:0">`;
 tokens.forEach((t,i)=>{
  const y=yc(i);
  s+=`<g data-t="${i}"><text class="tok" x="120" y="${y+4}" text-anchor="end">${esc(showToken(t))}</text><rect x="130" y="${y-15}" width="12" height="30" rx="2" fill="#cdd6e2"/></g>`;
 });
 s+=`</g><g class="stage block-part" style="--d:1">`;
 tokens.forEach((t,i)=>{
  const y=yc(i);
  [['q',-12],['k',0],['v',12]].forEach(([g,dy])=>{
   s+=`<g data-t="${i}"><path class="fg" d="${curve(142,y,232,y+dy)}" stroke-width="1.2"/><rect class="${g}" x="232" y="${y+dy-4.5}" width="9" height="9" rx="1.5" opacity=".75"/><text class="mini" x="246" y="${y+dy+3}">${g.toUpperCase()}</text></g>`;
  });
 });
 s+=`</g>`;
 // head card with a stack behind it
 s+=`<g class="stage head-part" style="--d:2">`;
 for(let d=3;d>=1;d--)s+=`<rect class="card back" x="${330+d*5}" y="${top+d*5}" width="440" height="${ch}" rx="6"/>`;
 s+=`<rect class="card front" x="330" y="${top}" width="440" height="${ch}" rx="6"/>`;
 for(const g of ['k','q','v']){
  s+=`<text class="lbl-${g}" x="452" y="${ty(g,0)-12}">${names[g]}</text>`;
  tokens.forEach((t,i)=>{
   const y=ty(g,i),dy=g==='q'?-12:g==='k'?0:12;
   s+=`<g data-t="${i}"><path class="f${g}" d="${curve(254,yc(i)+dy,380,y)}" stroke-width="1" opacity=".55"/><text class="mini" x="446" y="${y+3}" text-anchor="end">${esc(showToken(t))}</text><rect class="${g}" x="452" y="${y-1.5}" width="12" height="3" rx="1"/></g>`;
  });
 }
 s+=`</g><g class="stage head-part" style="--d:3">`;
 // key lines down into matrix columns, query lines into rows
 tokens.forEach((t,j)=>{s+=`<path data-t="${j}" class="fk" d="M464,${ty('k',j)} H${cx(j)} V${mtop}" stroke-width="1" opacity=".5"/>`;});
 tokens.forEach((t,i)=>{s+=`<path data-t="${i}" class="fq" d="M464,${ty('q',i)} H${mx0-TP/2-2}" stroke-width="1" opacity=".6"/>`;});
 // attention matrix
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){
  const masked=j>i,w=weights[i][j];
  s+=`<circle class="cell${masked?' masked':''}" data-t="${i}" data-q="${i}" data-k="${j}" cx="${cx(j)}" cy="${ty('q',i)}" r="5"${masked?'':` fill="#7f58a3" fill-opacity="${(0.07+0.93*w).toFixed(3)}"`}/>`;
 }
 s+=`<text class="col-sub" x="${mx0-6}" y="${ty('q',n-1)+TP/2+16}">ATTENTION</text>`;
 // out: weighted mix of values
 const mr=cx(n-1)+8;
 s+=`<path class="band" style="fill:#e6dcf0" d="M${mr},${ty('q',0)-6} L${outX},${ty('q',0)-2} V${ty('q',n-1)+2} L${mr},${ty('q',n-1)+6} Z"/>`;
 s+=`<text class="lbl-o" x="${outX-4}" y="${ty('q',0)-12}">Out</text>`;
 tokens.forEach((t,i)=>{
  s+=`<g data-t="${i}"><path class="fv" d="${curve(464,ty('v',i),outX,ty('q',i))}" stroke-width="1" opacity=".4"/><rect class="o" x="${outX}" y="${ty('q',i)-1.5}" width="14" height="3" rx="1"/></g>`;
 });
 s+=`</g>`;
 // MLP and the remaining blocks
 s+=`<g class="stage" style="--d:2">${chip('head',346,top,headLabel(),282)}</g>`;
 s+=`<g class="stage block-part" style="--d:4">`;
 tokens.forEach((t,i)=>{
  const y=yc(i);
  s+=`<g data-t="${i}"><path class="fo" d="${curve(outX+14,ty('q',i),806,y)}" stroke-width="1.2" opacity=".7"/>`;
  s+=`<rect x="806" y="${y-15}" width="10" height="30" rx="2" fill="#d0bfe1"/><path class="band" style="fill:#d3dcee" d="M816,${y-15} L846,${y-22} V${y+22} L816,${y+15} Z"/><rect x="846" y="${y-22}" width="26" height="44" rx="3" fill="#b3c2e0"/><path class="band" style="fill:#d3dcee" d="M872,${y-22} L902,${y-15} V${y+15} L872,${y+22} Z"/><rect x="902" y="${y-15}" width="10" height="30" rx="2" fill="#a9c0f3"/>`;
  s+=`<path class="fg" d="M912,${y} H944" stroke-width="1.2"/>`;
  for(let b=0;b<3;b++)s+=`<rect x="${944+b*12}" y="${y-15}" width="7" height="30" rx="2" fill="#d3e0fa" opacity="${0.9-b*0.25}"/>`;
  s+=`</g>`;
 });
 s+=`</g>`;
 // probabilities for the last token: drawn once, then updated in place by refreshProbs()
 const shown=list.slice(0,10),py=r=>84+r*30,last=n-1;
 s+=`<g class="stage prob-part" style="--d:6">`;
 shown.forEach(r=>{s+=`<path class="fo pc" data-rank="${r.rank}" d="${curve(980,yc(last),1040,py(r.rank)-4)}"/>`;});
 shown.forEach(r=>{s+=`<g class="prob" data-rank="${r.rank}"><text class="pt" x="1124" y="${py(r.rank)}" text-anchor="end">${esc(showToken(r.token))}</text><rect class="pb" x="1132" y="${py(r.rank)-6}" width="56" height="4" rx="2"/><text class="pp" x="1138" y="${py(r.rank)}"></text></g>`;});
 s+=`</g>`;
 const svg=$('#tx-flow');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=s;
 paintProbs(list,picked);
 return picked;
}
// Temperature, top-k / top-p and Generate only touch the last step, so only that is repainted.
function paintProbs(list=rows(),picked=sample(list,state.rnd)){
 const svg=$('#tx-flow');
 for(const r of list.slice(0,10)){
  const bw=Math.max(2,r.probability*56),g=svg.querySelector(`.prob[data-rank="${r.rank}"]`),c=svg.querySelector(`.pc[data-rank="${r.rank}"]`);
  g.setAttribute('class',`prob${r.kept?' kept':''}${r===picked?' sampled':''}`);
  g.querySelector('.pb').style.transform=`scaleX(${(bw/56).toFixed(4)})`;
  const pp=g.querySelector('.pp');pp.textContent=pct(r.probability);pp.style.transform=`translateX(${bw.toFixed(1)}px)`;
  c.style.strokeWidth=(0.6+r.probability*7).toFixed(2);c.style.opacity=r.kept?(r===picked?0.95:0.45):0;
 }
 $('#tx-sentence').innerHTML=`${esc(ex().prompt)} <b>${esc(showToken(picked.token))}</b>`;
 defaultResult(picked);
 return picked;
}
function refreshProbs(){state.sel=null;syncControls();probTable(paintProbs());}
function defaultResult(picked=sample(rows(),state.rnd)){
 const kept=rows().filter(r=>r.kept).length;
 $('#tx-result').textContent=`Next token drawn: “${showToken(picked.token)}” (${pct(picked.probability)}), from ${kept} candidate${kept===1?'':'s'} left after ${state.type==='top-k'?`top-k (k=${state.k})`:`top-p (p=${state.p.toFixed(2)})`} at temperature ${state.temperature.toFixed(1)}.`;
}

/* ---------- 01 embedding ---------- */
const vec=vs=>`<span class="vec" aria-hidden="true">${vs.map(v=>`<i style="background:${v>=0?`rgba(79,123,232,${(0.12+0.8*v).toFixed(2)})`:`rgba(199,118,91,${(0.12-0.8*v).toFixed(2)})`}"></i>`).join('')}<i class="more">… 768</i></span>`;
function embedding(){
 if(!$('#emb-tokens'))return;   // the Embedding section is now the original article's text and figure
 const {tokens,tokenIds}=ex();if(state.tok>=tokens.length)state.tok=0;
 $('#emb-tokens').innerHTML=tokens.map((t,i)=>`<button type="button" data-tok="${i}" aria-pressed="${i===state.tok}">${esc(showToken(t))}</button>`).join('');
 const i=state.tok,id=tokenIds[i],te=illustrativeVector(id),pe=illustrativeVector(90000+i),sum=te.map((v,j)=>Math.max(-1,Math.min(1,(v+pe[j])/1.4)));
 $('#emb-flow').innerHTML=`<span class="what">Token</span><span class="val">“${esc(showToken(tokens[i]))}”</span><span></span>
<span class="what">Token ID</span><span class="val"><code>${id.toLocaleString('en-US')}</code> of 50,257</span><span></span>
<span class="what">Token embedding</span><span class="val">row ${id.toLocaleString('en-US')} of a learned table</span>${vec(te)}
<span class="op">+</span>
<span class="what">Position embedding</span><span class="val">position ${i+1} in the sentence</span>${vec(pe)}
<span class="op">=</span>
<span class="what">Input to block 1</span><span class="val">one vector per token</span>${vec(sum)}`;
}

/* ---------- 02 attention ---------- */
function attention(){
 const {tokens,attn}=ex(),n=tokens.length,raw=attn[state.block][state.head],{scaled,masked,weights}=attentionStages(raw);
 const maxRaw=Math.max(...raw.flat().map(Math.abs))||1,maxSc=Math.max(...scaled.flat().map(Math.abs))||1;
 const tint=(v,m)=>v>=0?`rgba(79,123,232,${(0.06+0.5*v/m).toFixed(2)})`:`rgba(199,118,91,${(0.06-0.5*v/m).toFixed(2)})`;
 const mat=(cell)=>{
  let h=`<div class="mat" style="grid-template-columns:minmax(52px,74px) repeat(${n},minmax(0,1fr))"><span></span>${tokens.map(t=>`<span class="cl">${esc(showToken(t))}</span>`).join('')}`;
  for(let i=0;i<n;i++){h+=`<span class="rl">${esc(showToken(tokens[i]))}</span>`;for(let j=0;j<n;j++)h+=cell(i,j);}
  return h+'</div>';
 };
 const c=(i,j,text,style,cls='')=>`<span class="c${cls}" data-q="${i}" data-k="${j}"${style?` style="background:${style}"`:''}>${text}</span>`;
 const stages=[
  ['1 · Dot product','Query of the row token · Key of the column token.',mat((i,j)=>c(i,j,raw[i][j].toFixed(1),tint(raw[i][j],maxRaw)))],
  [`2 · Scale and mask`,`Divide by √${HEAD_DIM} = 8. Later tokens are set to −∞.`,mat((i,j)=>j>i?c(i,j,'−∞','',' masked'):c(i,j,scaled[i][j].toFixed(2),tint(scaled[i][j],maxSc)))],
  ['3 · Softmax','Each row becomes weights that sum to 100%.',mat((i,j)=>j>i?c(i,j,'0','',' masked'):c(i,j,Math.round(weights[i][j]*100)+'%',`rgba(127,88,163,${(0.05+0.7*weights[i][j]).toFixed(2)})`))]
 ];
 $('#attn-stages').innerHTML=stages.map(([t,p,m])=>`<div class="attn-stage"><h3>${t}</h3><p>${p}</p>${m}</div>`).join('');
 attnResult(n-1,strongest(weights[n-1]));
 function strongest(row){return row.indexOf(Math.max(...row));}
}
function attnResult(i,j){
 const {tokens,attn}=ex(),raw=attn[state.block][state.head],{scaled,weights}=attentionStages(raw);
 $$('#attn-stages .c').forEach(el=>{el.classList.toggle('on',+el.dataset.q===i&&+el.dataset.k===j);el.classList.toggle('row',+el.dataset.q===i);});
 const q=showToken(tokens[i]),k=showToken(tokens[j]);
 $('#attn-result').textContent=j>i
  ?`“${q}” cannot look at “${k}”: it comes later in the sentence, so its score is masked to −∞ and its weight is 0.`
  :`“${q}” looking at “${k}”: dot product ${raw[i][j].toFixed(1)} → ÷8 = ${scaled[i][j].toFixed(2)} → softmax over the row = ${pct(weights[i][j])} of its attention (block ${state.block+1}, head ${state.head+1}).`;
}

/* ---------- 04 probabilities ---------- */
function probTable(picked){
 const list=rows(),shown=list.slice(0,12),sel=state.sel??picked.rank;
 const cut=state.type==='top-k'?'Top-k':'Top-p';
 let h=`<span class="h" style="text-align:right">Token</span><span class="h">Logit</span><span class="h">÷ ${state.temperature.toFixed(1)}</span><span class="h">${cut}</span><span class="h">Softmax</span>`;
 for(const r of shown){
  const c=`${r.kept?' row-kept':''}${r.rank===sel?' row-sel':''}`;
  h+=`<button type="button" class="tk${c}" data-rank="${r.rank}">${esc(showToken(r.token))}</button><span class="n${c}">${r.logit.toFixed(2)}</span><span class="n${c}">${r.scaled.toFixed(2)}</span><span class="n${c}">${r.kept?r.scaled.toFixed(2):'−∞'}</span><span class="bar${c}"><i style="width:${Math.max(2,r.probability*120).toFixed(0)}px"></i>${pct(r.probability)}</span>`;
 }
 $('#prob-table').innerHTML=h;
 const r=list[Math.min(sel,list.length-1)],kept=list.filter(x=>x.kept),best=kept[0].scaled;
 const terms=kept.slice(0,4).map(x=>`exp(${(x.scaled-best).toFixed(2)})`).join(' + ')+(kept.length>4?' + …':'');
 $('#prob-card').innerHTML=`<h3>Chance of “<b>${esc(showToken(r.token))}</b>” being drawn${r===picked?' · drawn this time':''}</h3><ol>
<li><span>1 · Temperature</span><code>${r.logit.toFixed(2)} ÷ ${state.temperature.toFixed(1)} = <b>${r.scaled.toFixed(2)}</b></code></li>
<li><span>2 · ${cut} filter</span><code>${r.kept?`kept (rank ${r.rank+1} of ${kept.length} kept)`:`dropped → −∞ (rank ${r.rank+1}, only ${kept.length} kept)`}</code></li>
<li><span>3 · Softmax, relative to the best score</span><code>${r.kept?`exp(${(r.scaled-best).toFixed(2)}) ÷ (${terms}) = <b>${pct(r.probability)}</b>`:`<b>0%</b>`}</code></li></ol>`;
}

/* Changing block or head redraws only that part: the dots fade to their new
   weights and the head card (plus the block's own columns, for a block change)
   replays a short entrance. Embedding and probabilities stay still. */
function swap(kind,dir){
 const svg=$('#tx-flow'),calm=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const sel=kind==='block'?'.head-part,.block-part':'.head-part',parts=[...svg.querySelectorAll(sel)].filter(g=>!g.classList.contains('ghost'));
 svg.querySelectorAll('.ghost').forEach(g=>g.remove());
 // the outgoing block/head leaves as a copy, so the old picture is still visible while the new one arrives
 const ghosts=calm?[]:parts.map(g=>{const c=g.cloneNode(true);c.classList.remove('swap','stage');c.classList.add('ghost');c.setAttribute('aria-hidden','true');g.before(c);return c;});
 const w=attentionStages(ex().attn[state.block][state.head]).weights;
 parts.forEach(g=>g.querySelectorAll('.cell:not(.masked)').forEach(c=>c.setAttribute('fill-opacity',(0.07+0.93*w[+c.dataset.q][+c.dataset.k]).toFixed(3))));
 svg.querySelector('.block-label').textContent=blockLabel();
 svg.querySelector('.head-label').textContent=headLabel();
 svg.querySelector('title').textContent=`A sentence flowing through embedding, transformer block ${state.block+1} and the output probabilities`;
 syncSteppers();attention();defaultResult();
 const moving=[...parts,svg.querySelector(`.${kind}-label`),svg.querySelector(kind==='block'?'.frame':'.card.front')];
 svg.dataset.swap=kind;svg.style.setProperty('--dir',dir);
 moving.forEach(g=>g.classList.remove('swap'));void svg.getBoundingClientRect();moving.forEach(g=>g.classList.add('swap'));
 setTimeout(()=>ghosts.forEach(g=>g.remove()),520);
}

/* ---------- wiring ---------- */
function render({fresh=false,keepSel=false}={}){
 if(!keepSel)state.sel=null;
 syncControls();syncSteppers();
 const picked=flow();embedding();attention();probTable(picked);
 if(fresh){const lab=$('#tx-lab');lab.classList.remove('fresh');void lab.offsetWidth;lab.classList.add('fresh');}
}
function init(){
 $$('[data-controls]').forEach((el,i)=>{el.innerHTML=controlsMarkup(i);});
 $('#tx-example').innerHTML=EXAMPLES.map((e,i)=>`<option value="${i}">${esc(e.prompt)} …</option>`).join('');
 $('#tx-example').addEventListener('change',e=>{state.ex=+e.target.value;state.rnd=0.3;state.tok=0;render({fresh:true});});
 $('#tx-generate').addEventListener('click',()=>{state.rnd=Math.random();refreshProbs();});
 document.addEventListener('input',e=>{
  const c=e.target.dataset?.c;if(!c)return;
  if(c==='temperature')state.temperature=+e.target.value;
  else if(c==='type'){if(e.target.checked)state.type=e.target.value;}
  else if(c==='amount'){if(state.type==='top-k')state.k=Math.round(+e.target.value);else state.p=+e.target.value;}
  refreshProbs();
 });
 document.addEventListener('click',e=>{
  const step=e.target.closest('[data-step]');
  if(step){const kind=step.closest('[data-stepper]').dataset.stepper,max=kind==='block'?BLOCKS:HEADS;const dir=+step.dataset.step;state[kind]=(state[kind]+dir+max)%max;swap(kind,dir);return;}
  const tok=e.target.closest('[data-tok]');if(tok){state.tok=+tok.dataset.tok;embedding();return;}
  const rank=e.target.closest('[data-rank]');if(rank){state.sel=+rank.dataset.rank;probTable(sample(rows(),state.rnd));}
 });
 document.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.closest?.('#tx-flow .step')){e.preventDefault();e.target.closest('.step').dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
 const svg=$('#tx-flow');
 svg.addEventListener('mouseover',e=>{
  const el=e.target.closest('[data-t]');if(!el)return;
  const t=el.dataset.t;svg.classList.add('tracing');
  svg.querySelectorAll('[data-t]').forEach(x=>x.classList.toggle('on',x.dataset.t===t));
  const cell=e.target.closest('.cell');
  if(cell){
   const i=+cell.dataset.q,j=+cell.dataset.k,{tokens,attn}=ex(),w=attentionStages(attn[state.block][state.head]).weights[i][j];
   $('#tx-result').textContent=j>i?`“${showToken(tokens[i])}” cannot look ahead at “${showToken(tokens[j])}”: masked.`:`“${showToken(tokens[i])}” gives ${pct(w)} of its attention to “${showToken(tokens[j])}” in block ${state.block+1}, head ${state.head+1}.`;
  }
 });
 svg.addEventListener('mouseleave',()=>{svg.classList.remove('tracing');svg.querySelectorAll('.on').forEach(x=>x.classList.remove('on'));defaultResult();});
 $('#attn-stages').addEventListener('mouseover',e=>{const c=e.target.closest('.c');if(c)attnResult(+c.dataset.q,+c.dataset.k);});
 render({fresh:true});
}
init();
