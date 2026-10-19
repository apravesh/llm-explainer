/** Connected attention explainer. Comparison masks never mutate the encoder. */
import {softmax, dot} from './engine.mjs';
const afmt=(value,digits=3)=>Number.isFinite(value)?value.toFixed(digits):'−∞';
const aescape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const acolor=(v,scale=2)=>{const t=Math.min(Math.abs(v)/scale,1),rgb=v<0?[204,143,115]:[98,135,209];return `rgb(${rgb.map(x=>Math.round(245+(x-245)*t)).join(',')})`;};
export function attentionView(head,causal=false){
 const scores=head.scores.map((row,q)=>row.map((value,k)=>causal&&k>q?-Infinity:value));
 const weights=causal?scores.map(softmax):head.attention;
 const output=weights.map(row=>head.values[0].map((_,d)=>row.reduce((sum,w,k)=>sum+w*head.values[k][d],0)));
 return {scores,weights,output,causal};
}
export function attentionPrefix(weights,values,count){
 const n=Math.min(weights.length,Math.max(0,Math.floor(count)));
 return values[0].map((_,d)=>weights.slice(0,n).reduce((sum,w,k)=>sum+w*values[k][d],0));
}
export function attentionTokenLabel(index,model,patches){
 if(model==='vit'&&index===0)return '[CLS]';
 const patchIndex=index-(model==='vit'?1:0),p=patches[patchIndex];
 return p?`p${String(patchIndex).padStart(2,'0')} (${p.row}, ${p.col})`:'—';
}
function tinyVector(values,kind){return `<span class="af-mini-vector ${kind}">${values.map((v,d)=>`<i style="--cell:${acolor(v)}" title="d${d}: ${afmt(v,5)}"></i>`).join('')}</span>`;}
export function attentionFlowMarkup(s,patches,head,{embedded=false}={}){
 const query=s.model==='vit'?s.selected+1:s.selected,key=s.key,n=head.values.length,causal=!!s.compareCausal;
 const view=attentionView(head,causal),ql=attentionTokenLabel(query,s.model,patches),kl=attentionTokenLabel(key,s.model,patches);
 const inspect=(type,title,extra='')=>`<button type="button" class="af-node-title" data-attn-inspect="${type}" aria-expanded="${s.attentionInspect===type}" aria-controls="attention-calculation">${title}<span aria-hidden="true">+</span></button>${extra}`;
 return `<section class="attention-flow ${causal?'is-causal':''}" id="attention-flow" aria-label="Connected self-attention computation" data-mode="${causal?'decoder-comparison':'encoder'}" data-query="${query}" data-key="${key}">
 ${embedded?'':` <div class="af-mode-banner"><div><strong>${causal?'Decoder-mask comparison · not the encoder':'Encoder attention · no causal mask'}</strong><p>${causal?'Same Q, K, V, with future sequence positions blocked. Remaining weights are renormalized.':'Every patch can attend to every other patch in this image, in both directions.'}</p></div><label class="af-compare"><input type="checkbox" id="compare-causal" ${causal?'checked':''}>Compare with decoder masking</label></div>
 ${causal?'<p class="af-comparison-note">Counterfactual at this head only. Later blocks, the MLP, and the encoder output still use the actual unmasked run.</p>':''}
`}
 <div class="af-workbench">
 <svg class="af-wires" aria-hidden="true"><defs><marker id="af-arrow-q" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="#7e9edd"/></marker><marker id="af-arrow-k" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="#d2a18e"/></marker><marker id="af-arrow-v" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="#8bbbab"/></marker></defs><g class="af-wire-paths"></g></svg>
 <div class="af-qkv-nodes"><div class="af-column-label">01 / PROJECT</div>
 ${['q','k','v'].map(type=>{const isQ=type==='q',vector=isQ?head.rawQ[query]:type==='k'?head.rawK[key]:head.values[key];return `<div class="af-projection af-${type}" data-flow-node="${type}">${inspect(type,`${type.toUpperCase()} · ${isQ?'Query':type==='k'?'Key':'Value'}`)}<div class="af-token-caption">${isQ?ql:kl}</div>${tinyVector(vector,type)}<code>h̄ × W${type.toUpperCase()}</code></div>`;}).join('')}
 <div class="af-pre-norm">${s.model==='pixtral'?'RMSNorm':'LayerNorm'} before projection<br>8 dimensions per head</div></div>
 <div class="af-position-slot">${s.model==='pixtral'?`<button class="af-rope" type="button" data-attn-inspect="rope" aria-expanded="${s.attentionInspect==='rope'}" aria-controls="attention-calculation" data-flow-node="rope"><span class="af-rope-glyph" aria-hidden="true">↻</span><strong>2D RoPE</strong><small>Rotate Q & K</small></button>`:'<div class="af-position-note">Q and K<br>meet here →</div>'}</div>
 <div class="af-score-node" data-flow-node="scores"><div class="af-column-label">02 / MATCH</div>${inspect('scores','QKᵀ / √8')}<div class="af-matrix" data-flow-anchor="score-matrix"><span class="af-axis-top">KEY →</span><span class="af-axis-left">QUERY ↓</span><canvas id="flow-scores" width="224" height="224" tabindex="0" role="img" aria-label="Scaled query-key scores. Click to choose a pair; arrow keys change query and key."></canvas></div><div class="af-matrix-ends"><span>${attentionTokenLabel(0,s.model,patches).split(' ')[0]}</span><span>${attentionTokenLabel(n-1,s.model,patches).split(' ')[0]}</span></div><div class="af-score-label">${causal?'upper triangle = −∞':'full matrix · no future-token restriction'}<span class="af-score-scale"><i></i> negative → positive</span></div></div>
 <div class="af-softmax-node" data-flow-node="softmax"><div class="af-column-label">03 / NORMALIZE</div>${inspect('softmax','Softmax')}<div class="af-matrix" data-flow-anchor="weight-matrix"><span class="af-axis-top">KEY →</span><span class="af-axis-left">QUERY ↓</span><canvas id="flow-weights" width="224" height="224" tabindex="0" role="img" aria-label="Attention weights. Every row sums to one. Click to choose a pair; arrow keys change query and key."></canvas></div><div class="af-matrix-ends"><span>${attentionTokenLabel(0,s.model,patches).split(' ')[0]}</span><span>${attentionTokenLabel(n-1,s.model,patches).split(' ')[0]}</span></div><div class="af-row-sum">selected row sums to <strong>1.000</strong></div></div>
 <div class="af-mix-node" data-flow-node="mix"><div class="af-column-label">04 / GATHER</div>${inspect('values','Weighted sum of V')}<div class="af-weight-scalar"><span>selected key’s weight</span><strong data-af-weight>${afmt(view.weights[query][key]*100,2)}%</strong></div><div class="af-values-strip"><span>×</span><span data-af-value-chips>${tinyVector(head.values[key],'v')}</span></div><div class="af-gather-arrow" aria-hidden="true">↓ <span>sum over all ${n} keys</span></div><div class="af-output-caption">${causal?'Comparison head output':'Head output'}</div><div class="af-output-chips" data-af-output>${view.output[query].map((v,d)=>`<span title="Output dimension ${d}"><small>${d}</small><strong>${afmt(v)}</strong></span>`).join('')}</div><button type="button" class="af-play" data-attn-play>▶ Animate value mixing</button><div class="af-play-status" data-af-progress aria-live="polite">Complete weighted sum · ${n} keys</div></div>
 </div>
 <div class="af-selected-pair"><span><i class="af-query-dot"></i> Query <strong>${ql}</strong></span><span><i class="af-key-dot"></i> Key <strong data-af-key-label>${kl}</strong></span><span>score <strong data-af-score>${afmt(view.scores[query][key],4)}</strong></span><span>weight <strong data-af-pair-weight>${afmt(view.weights[query][key]*100,3)}%</strong></span><span data-af-blocked>${causal&&key>query?'blocked future position':''}</span></div>
 <div class="af-key-strip-heading"><span>THIS QUERY’S ATTENTION · CLICK A KEY</span><span>Darker = larger weight in this row</span></div><div class="af-key-strip" role="group" aria-label="Keys for the selected query">${view.weights[query].map((w,k)=>`<button type="button" data-attn-key="${k}" aria-pressed="${k===key}" aria-label="Key ${attentionTokenLabel(k,s.model,patches)}, weight ${afmt(w*100,3)} percent" title="${attentionTokenLabel(k,s.model,patches)} · ${afmt(w*100,3)}%"><span>${attentionTokenLabel(k,s.model,patches).split(' ')[0]}</span><i style="--weight:${w/Math.max(...view.weights[query])}"></i></button>`).join('')}</div>
 <p class="af-footnote">${s.model==='pixtral'?'2D RoPE changes Q and K, never V. Packed images still require a block-diagonal boundary mask. ':'[CLS] participates like another token. '}This is one head before head concatenation, the output projection, and the residual addition. Live untrained weights—not semantic attention.</p>
 </section>`;
}

/** Canvas and animation lifetime is tied to one expanded rendering. */
export function mountAttentionFlow(root,{state:s,patches,head,previous,onPair,onInspect,onCompare}){
 const doc=root.ownerDocument,win=doc.defaultView,reduced=win.matchMedia('(prefers-reduced-motion: reduce)');
 const view=attentionView(head,!!s.compareCausal),query=s.model==='vit'?s.selected+1:s.selected,n=head.values.length;
 const $=selector=>root.querySelector(selector),canvases=[$('#flow-scores'),$('#flow-weights')];
 let key=s.key,frame=0,disposed=false,playing=false,phase='idle';
 const text=(selector,value)=>{const el=$(selector);if(el)el.textContent=value;};
 function drawMatrix(canvas,weights,isScore){
  const box=canvas.getBoundingClientRect(),size=Math.max(1,Math.round(box.width)),dpr=win.devicePixelRatio||1;
  if(canvas.width!==Math.round(size*dpr)||canvas.height!==Math.round(size*dpr)){canvas.width=Math.round(size*dpr);canvas.height=Math.round(size*dpr);}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,size,size);
  const cell=size/n,max=Math.max(...weights.flat());
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
   const blocked=view.causal&&j>i;const w=weights[i][j];
   ctx.fillStyle=blocked?'#eceff4':isScore?acolor(head.scores[i][j],3):`rgba(54,103,228,${.05+.86*Math.sqrt(w/(max||1))})`;
   ctx.fillRect(j*cell,i*cell,Math.max(.5,cell-.6),Math.max(.5,cell-.6));
   if(blocked&&cell>3){ctx.strokeStyle='#d4dae3';ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(j*cell,i*cell+cell-.6);ctx.lineTo(j*cell+cell-.6,i*cell);ctx.stroke();}
  }
  ctx.strokeStyle='#3667e4';ctx.lineWidth=1.6;ctx.strokeRect(.8,query*cell+.5,size-1.6,Math.max(1,cell-1));
  ctx.strokeStyle='#c7765b';ctx.lineWidth=1.3;ctx.strokeRect(key*cell+.5,query*cell+.5,Math.max(1,cell-1),Math.max(1,cell-1));
  canvas.setAttribute('aria-label',`${isScore?'Scaled scores':'Attention weights'}. Query ${attentionTokenLabel(query,s.model,patches)}, key ${attentionTokenLabel(key,s.model,patches)}: ${isScore?afmt(view.scores[query][key],4):afmt(weights[query][key]*100,3)+' percent'}. ${view.causal?'Decoder comparison; future positions blocked.':'Encoder; no causal mask.'} Arrow keys change the pair.`);
 }
 function draw(weights=view.weights){drawMatrix(canvases[0],weights,true);drawMatrix(canvases[1],weights,false);}
 function output(values){root.dataset.output=JSON.stringify(values);$('[data-af-output]').innerHTML=values.map((v,d)=>`<span title="Output dimension ${d}: ${afmt(v,7)}"><small>${d}</small><strong>${afmt(v)}</strong></span>`).join('');}
 function showKey(k,weights=view.weights){
  key=k;root.dataset.key=k;const w=weights[query][k];
  for(const type of ['k','v']){
   const node=$(`[data-flow-node="${type}"]`);node.querySelector('.af-token-caption').textContent=attentionTokenLabel(k,s.model,patches);
   node.querySelector('.af-mini-vector').outerHTML=tinyVector(type==='k'?head.rawK[k]:head.values[k],type);
  }
  text('[data-af-weight]',afmt(w*100,2)+'%');text('[data-af-key-label]',attentionTokenLabel(k,s.model,patches));text('[data-af-score]',afmt(view.scores[query][k],4));text('[data-af-pair-weight]',afmt(w*100,3)+'%');text('[data-af-blocked]',view.causal&&k>query?'blocked future position':'');
  $('[data-af-value-chips]').innerHTML=tinyVector(head.values[k],'v');
  root.querySelectorAll('[data-attn-key]').forEach(el=>el.setAttribute('aria-pressed',String(+el.dataset.attnKey===k)));
  draw(weights);
 }
 function connectors(){
  if(disposed)return;const svg=$('.af-wires'),bench=$('.af-workbench'),b=bench.getBoundingClientRect();svg.setAttribute('viewBox',`0 0 ${b.width} ${b.height}`);
  const rect=selector=>{const r=$(selector).getBoundingClientRect();return {x:r.left-b.left,y:r.top-b.top,w:r.width,h:r.height};};
  const q=rect('[data-flow-node="q"]'),k=rect('[data-flow-node="k"]'),v=rect('[data-flow-node="v"]'),score=rect('[data-flow-anchor="score-matrix"]'),weight=rect('[data-flow-anchor="weight-matrix"]'),mix=rect('[data-flow-node="mix"]');
  const connections=[];const path=(a,z,kind,id)=>{const dx=Math.max(15,(z.x-a.x)/2);connections.push(`<path class="af-wire af-wire-${kind}" data-wire="${id}" d="M${a.x} ${a.y}C${a.x+dx} ${a.y},${z.x-dx} ${z.y},${z.x} ${z.y}" marker-end="url(#af-arrow-${kind})"/>`);};
  const source=r=>({x:r.x+r.w+2,y:r.y+r.h/2});
  if(s.model==='pixtral'){
   const rp=rect('[data-flow-node="rope"]');path(source(q),{x:rp.x-3,y:rp.y+rp.h*.35},'q','q-rope');path(source(k),{x:rp.x-3,y:rp.y+rp.h*.7},'k','k-rope');path(source(rp),{x:score.x-4,y:score.y+score.h*.45},'q','rope-scores');
  }else{path(source(q),{x:score.x-4,y:score.y+score.h*.35},'q','q-scores');path(source(k),{x:score.x-4,y:score.y+score.h*.65},'k','k-scores');}
  path({x:score.x+score.w+2,y:score.y+score.h*.5},{x:weight.x-4,y:weight.y+weight.h*.5},'q','scores-softmax');
  if(mix.y>weight.y+weight.h){
   connections.push(`<path class="af-wire af-wire-q" data-wire="weights-mix" d="M${weight.x+weight.w/2} ${weight.y+weight.h+48}V${mix.y-7}" marker-end="url(#af-arrow-q)"/>`);
  }else path({x:weight.x+weight.w+2,y:weight.y+weight.h*.5},{x:mix.x-4,y:mix.y+100},'q','weights-mix');
  const baseline=Math.max(v.y+v.h,score.y+score.h,weight.y+weight.h)+67;
  connections.push(`<path class="af-wire af-wire-v af-value-bypass" data-wire="values-bypass" d="M${v.x+v.w+2} ${v.y+v.h/2}C${v.x+v.w+25} ${v.y+v.h/2},${v.x+v.w+10} ${baseline},${v.x+v.w+45} ${baseline}L${mix.x-22} ${baseline}Q${mix.x-8} ${baseline},${mix.x-8} ${baseline-14}L${mix.x-8} ${mix.y+160}L${mix.x} ${mix.y+160}" marker-end="url(#af-arrow-v)"/>`);
  $('.af-wire-paths').innerHTML=connections.join('');
 }
 function idle(){if(disposed)return;playing=false;phase='idle';root.dataset.phase=phase;root.classList.remove('is-mixing');text('[data-attn-play]','▶ Animate value mixing');text('[data-af-progress]',`Complete weighted sum · ${n} keys`);text('.af-output-caption',view.causal?'Comparison head output':'Head output');output(view.output[query]);showKey(s.key);}
 function stop(){win.cancelAnimationFrame(frame);idle();}
 function animateMix(){
  if(playing){stop();return;}
  win.cancelAnimationFrame(frame);
  if(reduced.matches){idle();text('[data-af-progress]','Reduced motion: complete weighted sum shown.');return;}
  playing=true;phase='mixing';root.dataset.phase=phase;root.classList.add('is-mixing');text('[data-attn-play]','■ Stop animation');text('.af-output-caption','Partial head output');
  let started=null,last=-1;output(Array(head.values[0].length).fill(0));
  function next(time){if(disposed)return;if(started===null)started=time;const count=Math.min(n,Math.floor((time-started)/100)+1);if(count!==last){last=count;showKey(count-1);output(attentionPrefix(view.weights[query],head.values,count));text('[data-af-progress]',`Partial sum · ${count} / ${n} keys`);}if(count<n){frame=win.requestAnimationFrame(next);}else idle();}
  frame=win.requestAnimationFrame(next);
 }
 function matrixPair(event,canvas){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(n-1,Math.floor((event.clientY-r.top)/r.height*n))),Math.max(0,Math.min(n-1,Math.floor((event.clientX-r.left)/r.width*n)))];}
 function onClick(event){const inspect=event.target.closest('[data-attn-inspect]');if(inspect){event.stopPropagation();stop();onInspect(inspect.dataset.attnInspect);return;}const selected=event.target.closest('[data-attn-key]');if(selected){stop();onPair(query,+selected.dataset.attnKey,`[data-attn-key="${selected.dataset.attnKey}"]`);return;}if(event.target.closest('[data-attn-play]'))animateMix();}
 const compare=()=>{stop();onCompare($('#compare-causal').checked);};
 const handlers=[];
 for(const canvas of canvases){
  const click=e=>{const [q,k]=matrixPair(e,canvas);stop();onPair(q,k,'#'+canvas.id);};
  const keydown=e=>{let q=query,k=key;if(e.key==='ArrowUp')q=Math.max(0,q-1);else if(e.key==='ArrowDown')q=Math.min(n-1,q+1);else if(e.key==='ArrowLeft')k=Math.max(0,k-1);else if(e.key==='ArrowRight')k=Math.min(n-1,k+1);else return;e.preventDefault();stop();onPair(q,k,'#'+canvas.id);};
  const hover=e=>{const [q,k]=matrixPair(e,canvas);canvas.title=`${attentionTokenLabel(q,s.model,patches)} → ${attentionTokenLabel(k,s.model,patches)} · score ${afmt(view.scores[q][k],4)} · weight ${afmt(view.weights[q][k]*100,3)}%`;};
  canvas.addEventListener('click',click);canvas.addEventListener('keydown',keydown);canvas.addEventListener('pointermove',hover);handlers.push(()=>{canvas.removeEventListener('click',click);canvas.removeEventListener('keydown',keydown);canvas.removeEventListener('pointermove',hover);});
 }
 root.addEventListener('click',onClick);$('#compare-causal')?.addEventListener('change',compare);
 const resize=new win.ResizeObserver(()=>{draw();connectors();});resize.observe($('.af-workbench'));
 const visibility=()=>{if(doc.visibilityState==='hidden')stop();};const preference=()=>{if(reduced.matches)stop();};
 doc.addEventListener('visibilitychange',visibility);reduced.addEventListener('change',preference);
 draw();connectors();output(view.output[query]);root.dataset.phase='idle';
 // Interpolate the comparison without touching the underlying Q/K/V or later blocks.
 if(previous&&previous.head===head&&previous.causal!==view.causal&&!reduced.matches){
  const old=previous.view,started=win.performance.now();root.dataset.phase='mask-transition';
  const morph=time=>{if(disposed)return;const t=Math.min(1,(time-started)/600),ease=1-(1-t)**3,weights=old.weights.map((row,q)=>row.map((v,k)=>v+(view.weights[q][k]-v)*ease));
   showKey(s.key,weights);output(old.output[query].map((v,d)=>v+(view.output[query][d]-v)*ease));
   if(t<1)frame=win.requestAnimationFrame(morph);else idle();};frame=win.requestAnimationFrame(morph);
 }
 return {snapshot:{head,causal:view.causal,view},dispose(){disposed=true;win.cancelAnimationFrame(frame);resize.disconnect();root.removeEventListener('click',onClick);$('#compare-causal')?.removeEventListener('change',compare);handlers.forEach(fn=>fn());doc.removeEventListener('visibilitychange',visibility);reduced.removeEventListener('change',preference);}};
}

export function attentionScoreDetail(s,patches,head){
 const q=s.model==='vit'?s.selected+1:s.selected,k=s.key,view=attentionView(head,!!s.compareCausal),products=head.q[q].map((v,d)=>v*head.k[k][d]);
 const row=(name,x)=>`<div class="af-arithmetic-row"><span>${name}</span>${x.map(v=>`<code>${afmt(v,4)}</code>`).join('')}</div>`;
 const max=Math.max(...view.scores[q]),exps=view.scores[q].map(x=>Math.exp(x-max)),den=exps.reduce((a,b)=>a+b,0);
 if(s.attentionInspect==='scores')return `${row('Q at query',head.q[q])}${row('K at key',head.k[k])}${row('Q ⊙ K',products)}<p class="af-equation">Σ(Q ⊙ K) = ${afmt(dot(head.q[q],head.k[k]),6)}; divide by √8 → ${afmt(head.scores[q][k],6)}${view.causal&&k>q?' → masked to −∞':''}</p>`;
 if(s.attentionInspect==='softmax')return `<div class="af-softmax-formula"><code>aₖ = exp(sₖ − max(s)) / Σⱼ exp(sⱼ − max(s))</code><code>max(row) = ${afmt(max,6)}</code><code>${afmt(exps[k],7)} / ${afmt(den,7)} = <strong>${afmt(view.weights[q][k],7)}</strong></code><p>All ${view.weights[q].length} weights sum to ${afmt(view.weights[q].reduce((a,b)=>a+b),6)}. ${view.causal?'Blocked positions contribute exp(−∞) = 0; all remaining positions are renormalized.':'No positions are removed in the encoder.'}</p></div>`;
 if(s.attentionInspect==='values')return `${row('V at key',head.values[k])}${row('weight × V',head.values[k].map(v=>v*view.weights[q][k]))}${row('sum all keys',view.output[q])}<p class="af-equation">Selected weight ${afmt(view.weights[q][k],7)} × this key’s V is one contribution. Sum ${view.weights[q].length} such vectors to obtain the complete head output.</p>`;
 if(s.attentionInspect==='rope')return `<p class="af-equation">Query ${aescape(attentionTokenLabel(q,s.model,patches))} and key ${aescape(attentionTokenLabel(k,s.model,patches))}. Each consecutive coordinate pair is rotated by row × θ or column × θ. This demo uses θ = 1, 0.1, 0.01, 0.001.</p>${row('Q before',head.rawQ[q])}${row('Q rotated',head.q[q])}${row('K before',head.rawK[k])}${row('K rotated',head.k[k])}<p class="af-equation">R(φ)[a, b] = [a cos φ − b sin φ, a sin φ + b cos φ]. V bypasses this operation.</p>`;
 return '';
}
