/** Multi-head overview. All heads see all tokens; only the feature dimension splits. */
import {linear} from './engine.mjs';
import {attentionView,attentionTokenLabel} from './attention-flow.mjs';
const mhfmt=(x,d=3)=>Number(x).toFixed(d);
const mhColors=[[84,125,205],[151,115,188],[80,148,126]];
export function multiHeadView(layer,query,causal=false){
 const views=layer.heads.map(head=>attentionView(head,causal));
 const outputs=views.map(view=>view.output[query]);
 const concatenated=outputs.flat(),projected=linear(concatenated,layer.weights.o);
 const residual=projected.map((v,i)=>v+layer.input[query][i]);
 return {views,outputs,concatenated,projected,residual,query,causal};
}
function mhVector(values,head,extra=''){
 return `<div class="mh-vector mh-color-${head}" ${extra}>${values.map((v,d)=>`<span title="d${d}: ${mhfmt(v,6)}" style="--strength:${Math.min(1,Math.abs(v)/3)*.6+.08}">${mhfmt(v,2)}</span>`).join('')}</div>`;
}
export function multiHeadMarkup(s,patches,layer){
 const query=s.model==='vit'?s.selected+1:s.selected,n=layer.input.length,heads=layer.heads.length,d=layer.heads[0].values[0].length,total=heads*d,view=multiHeadView(layer,query,!!s.compareCausal);
 const detail=!!s.headDetailOpen;
 return `<section id="multi-head" class="multi-head" data-mode="${view.causal?'decoder-comparison':'encoder'}" data-query="${query}" aria-label="Multi-head self-attention overview">
 <div class="af-mode-banner"><div><strong>${view.causal?'Decoder-mask comparison · not the encoder':'Encoder attention · no causal mask'}</strong><p>${view.causal?'Causal masking is applied independently to every head in this comparison.':'Every head sees every patch in this image. The heads have different learned projections.'}</p></div><label class="af-compare"><input type="checkbox" id="compare-causal" ${view.causal?'checked':''}>Compare with decoder masking</label></div>
 ${view.causal?'<p class="af-comparison-note">Counterfactual for this attention block only. Merged values below are comparison values; the actual encoder and all later layers are unchanged.</p>':''}
 <div class="mh-split-heading"><div><span class="af-column-label">01 / PROJECT & SPLIT FEATURES</span><h4>${total} features → ${heads} heads × ${d} features</h4></div><span class="mh-token-count">N = ${n} tokens${s.model==='vit'?' · includes [CLS]':''}</span></div>
 <div class="mh-projected-rows">${['Q','K','V'].map(type=>{const row=type==='Q'?query:s.key;return `<div class="mh-projected-row"><strong class="mh-type-${type}">${type}</strong><div class="mh-feature-groups">${layer.heads.map((head,h)=>{const values=type==='Q'?head.rawQ[row]:type==='K'?head.rawK[row]:head.values[row];return `<button type="button" class="mh-feature-group mh-color-${h}" data-mh-open="${h}" aria-label="Inspect head ${h+1}, ${type} projected features ${h*d} through ${(h+1)*d-1}">${values.map((v,i)=>`<i title="${type}[${row},${h*d+i}] = ${mhfmt(v,5)}" style="--strength:${.1+.65*Math.min(Math.abs(v)/3,1)}"></i>`).join('')}</button>`;}).join('')}</div><small>${n} × ${total}</small></div>`;}).join('')}</div>
 <p class="mh-input-note">Q shown for query <strong>${attentionTokenLabel(query,s.model,patches)}</strong>; K/V for key <strong>${attentionTokenLabel(s.key,s.model,patches)}</strong>. Each head projects from <em>all ${total} input features</em> into ${d} output features—not from a subset of the input.</p>
 <div class="mh-branch" aria-hidden="true"><span>↓ head 1</span><span>↓ head 2</span><span>↓ head 3</span></div>
 <div class="mh-heads">${layer.heads.map((head,h)=>`<article class="mh-head mh-color-${h} ${s.head===h?'is-selected':''}" data-mh-card="${h}">
 <button type="button" class="mh-head-heading" data-mh-open="${h}" aria-expanded="${detail&&s.head===h}" aria-controls="head-detail"><strong>Head ${h+1}</strong><span>features ${h*d}–${(h+1)*d-1}</span><i aria-hidden="true">${detail&&s.head===h?'−':'+'}</i></button>
 <div class="mh-head-operation">${s.model==='pixtral'?'2D RoPE on Q/K → ':''}softmax(QKᵀ / √${d})V</div>
 <button type="button" class="mh-matrix-preview" data-mh-open="${h}" aria-expanded="${detail&&s.head===h}" aria-controls="head-detail" aria-label="Expand head ${h+1} with ${n} tokens and ${d} features per token">
 <div class="mh-qkv-previews">${['Q','K','V'].map(type=>`<div><span>${type}</span><canvas data-mh-matrix="${h}:${type}" width="32" height="100" role="img" aria-label="Head ${h+1} ${type} matrix, ${n} by ${d}"></canvas></div>`).join('')}<small>${n} × ${d} each</small></div><span class="mh-matrix-arrow">→</span><div class="mh-attention-preview"><span>Attention</span><canvas data-mh-matrix="${h}:A" width="112" height="112" role="img" aria-label="Head ${h+1} attention weights, ${n} by ${n}"></canvas><small>${n} × ${n}</small></div></button>
 <div class="mh-head-weight"><span>selected query → key</span><strong>${mhfmt(view.views[h].weights[query][s.key]*100,2)}%</strong></div>
 <div class="mh-head-output-label">Head ${h+1} output · ${d} values</div>${mhVector(view.outputs[h],h,`data-mh-output="${h}"`)}
 <button type="button" class="mh-expand-link" data-mh-open="${h}" aria-expanded="${detail&&s.head===h}" aria-controls="head-detail">${detail&&s.head===h?'Collapse head detail ↑':'Inspect this head ↓'}</button></article>`).join('')}</div>
 <p class="mh-shape-note"><strong>What gets smaller?</strong> Q/K/V: N × ${total} → N × ${d} per head. <strong>Not the attention matrix:</strong> every head still has N × N scores; the previews are simply drawn smaller. No patches are split between heads. Attention shading is relative to each head’s maximum.</p>
 <div class="mh-merge-heading"><div><span class="af-column-label">02 / CONCATENATE → PROJECT → ADD RESIDUAL</span><h4>Bring the heads back together.</h4></div><button type="button" class="mh-merge-play" data-mh-animate>▶ Animate concatenation</button></div>
 <div class="mh-concat" data-mh-concat="${view.concatenated.join(',')}">${view.outputs.map((x,h)=>`<div class="mh-concat-piece"><span>Head ${h+1} · d${h*d}–${(h+1)*d-1}</span>${mhVector(x,h,`data-mh-target="${h}"`)}</div>`).join('')}</div>
 <div class="mh-merge-status" data-mh-progress aria-live="polite">${heads} × ${d} = ${total} values · concatenate, do not average</div>
 <div class="mh-projection-row"><span class="mh-merge-math">concat(h₁, h₂, h₃)</span><b>×</b><button type="button" data-mh-project aria-expanded="${s.attentionInspect==='merge'}" aria-controls="attention-calculation"><strong>Wₒ</strong><small>${total} × ${total}</small><span>inspect +</span></button><b>+</b><span class="mh-merge-math">input residual</span><b>=</b><span class="mh-merge-math">${view.causal?'comparison ':'attention '}output</span></div>
 <div class="mh-residual-values" data-mh-residual="${view.residual.join(',')}">${view.residual.map((v,i)=>`<span title="Residual output dimension ${i}: ${mhfmt(v,7)}"><small>${i}</small>${mhfmt(v,3)}</span>`).join('')}</div>
 <p class="mh-real-sizes">This teaching encoder: ${heads} heads × ${d}. ViT-B/16: 12 × 64 = 768. Pixtral-ViT (12B): 16 × 64 = 1,024. These are untrained teaching heads, not named semantic specialists.</p>
 </section>`;
}

export function mountMultiHead(root,{state:s,patches,layer,onHead,onInspect,onCompare,previous}){
 const win=root.ownerDocument.defaultView,doc=root.ownerDocument,reduced=win.matchMedia('(prefers-reduced-motion: reduce)');
 const q=s.model==='vit'?s.selected+1:s.selected,view=multiHeadView(layer,q,!!s.compareCausal);
 const canvases=[...root.querySelectorAll('[data-mh-matrix]')];let disposed=false,animations=[],frame=0,playing=false,generation=0;
 function draw(views=view.views){
  for(const canvas of canvases){
   const [h,type]=canvas.dataset.mhMatrix.split(':'),head=layer.heads[+h],values=type==='A'?views[+h].weights:type==='Q'?head.q:type==='K'?head.k:head.values;
   const rect=canvas.getBoundingClientRect(),width=Math.max(1,rect.width),height=Math.max(1,rect.height),ratio=win.devicePixelRatio||1;
   canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);const c=canvas.getContext('2d');c.scale(ratio,ratio);
   const rows=values.length,cols=values[0].length,w=width/cols,ht=height/rows,max=type==='A'?Math.max(...values.flat()):3,color=mhColors[+h%mhColors.length];
   for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){
    const blocked=type==='A'&&view.causal&&j>i,amount=type==='A'?Math.sqrt(values[i][j]/(max||1)):Math.min(Math.abs(values[i][j])/max,1);
    const target=type==='A'||values[i][j]>=0?color:[198,141,113];
    c.fillStyle=blocked?'#eaedf2':`rgb(${target.map(v=>Math.round(248+(v-248)*(.09+.86*amount))).join(',')})`;c.fillRect(j*w,i*ht,Math.max(.5,w-.4),Math.max(.5,ht-.4));
   }
   c.strokeStyle=`rgb(${color.join(',')})`;c.lineWidth=1.3;
   const selected=type==='K'||type==='V'?s.key:q;c.strokeRect(.7,selected*ht+.5,width-1.4,Math.max(1,ht-1));
   if(type==='A'){c.strokeStyle='#bd896e';c.strokeRect(s.key*w+.5,q*ht+.5,Math.max(1,w-1),Math.max(1,ht-1));}
  }
 }
 function stop(){generation++;win.cancelAnimationFrame(frame);animations.forEach(a=>a.cancel());animations=[];root.querySelectorAll('.mh-travel-vector').forEach(el=>el.remove());root.classList.remove('mh-is-merging');playing=false;root.dataset.phase='idle';root.querySelector('[data-mh-animate]').textContent='▶ Animate concatenation';}
 function animate(){
  if(playing){stop();root.querySelector('[data-mh-progress]').textContent='Concatenation stopped · complete 24-value vector shown';return;}
  stop();if(reduced.matches){root.querySelector('[data-mh-progress]').textContent='Reduced motion: all 3 outputs are shown concatenated.';return;}
  playing=true;const run=generation;root.dataset.phase='merging';root.classList.add('mh-is-merging');root.querySelector('[data-mh-animate]').textContent='■ Stop animation';root.querySelector('[data-mh-progress]').textContent='Moving each head’s 8 values into its own slot…';
  const base=root.getBoundingClientRect();
  view.outputs.forEach((_,h)=>{
   const source=root.querySelector(`[data-mh-output="${h}"]`),target=root.querySelector(`[data-mh-target="${h}"]`),a=source.getBoundingClientRect(),b=target.getBoundingClientRect(),ghost=source.cloneNode(true);
   ghost.removeAttribute('data-mh-output');ghost.classList.add('mh-travel-vector');ghost.setAttribute('aria-hidden','true');Object.assign(ghost.style,{position:'absolute',left:`${a.x-base.x}px`,top:`${a.y-base.y}px`,width:`${a.width}px`,height:`${a.height}px`,pointerEvents:'none',zIndex:'5',transformOrigin:'0 0'});root.append(ghost);
   const motion=ghost.animate([{transform:'none',opacity:1},{transform:`translate(${b.x-a.x}px,${b.y-a.y}px) scale(${b.width/a.width},${b.height/a.height})`,opacity:1}],{duration:900,delay:h*200,easing:'cubic-bezier(.22,.75,.18,1)',fill:'both'});animations.push(motion);
  });
  const projection=root.querySelector('.mh-projection-row').animate([{opacity:.35},{opacity:1}],{duration:350,delay:1300,fill:'both'});animations.push(projection);
  Promise.allSettled(animations.map(a=>a.finished)).then(()=>{if(!disposed&&playing&&run===generation){stop();root.querySelector('[data-mh-progress]').textContent='8 + 8 + 8 = 24 values → Wₒ → add the original input';}});
 }
 const click=e=>{
  const target=e.target.closest('button');if(!target)return;
  if(target.hasAttribute('data-mh-open')){stop();onHead(+target.dataset.mhOpen);}
  if(target.hasAttribute('data-mh-project')){stop();onInspect();}
  if(target.hasAttribute('data-mh-animate'))animate();
 };
 const compare=()=>{stop();onCompare(root.querySelector('#compare-causal').checked);};
 root.addEventListener('click',click);root.querySelector('#compare-causal').addEventListener('change',compare);
 const observer=new win.ResizeObserver(()=>{if(disposed)return;if(playing)stop();draw();});observer.observe(root.querySelector('.mh-heads'));
 const preference=()=>{if(reduced.matches)stop();};const visibility=()=>{if(doc.visibilityState==='hidden')stop();};reduced.addEventListener('change',preference);doc.addEventListener('visibilitychange',visibility);
 draw();root.dataset.phase='idle';
 if(previous?.layer===layer&&previous.causal!==view.causal&&!reduced.matches){
  const from=previous.view.views,started=win.performance.now();root.dataset.phase='mask-transition';
  const morph=time=>{if(disposed)return;const t=Math.min(1,(time-started)/650),ease=1-(1-t)**3;draw(view.views.map((v,h)=>({weights:v.weights.map((row,i)=>row.map((w,j)=>from[h].weights[i][j]+(w-from[h].weights[i][j])*ease))})));if(t<1)frame=win.requestAnimationFrame(morph);else root.dataset.phase='idle';};frame=win.requestAnimationFrame(morph);
 }
 return {snapshot:{layer,causal:view.causal,view},dispose(){disposed=true;stop();observer.disconnect();root.removeEventListener('click',click);root.querySelector('#compare-causal').removeEventListener('change',compare);reduced.removeEventListener('change',preference);doc.removeEventListener('visibilitychange',visibility);}};
}
