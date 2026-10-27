import {geluFull} from './full-engine.mjs';
import {matrixValue,featureWindow} from './full-config.mjs';
import {createMatrixRenderer,activationRGB} from './full-matrix.mjs';
/** Inspect the stored projector tensors; the playground never writes to them. */
export function projectorGeluData(run,row=0,feature=0){
 if(!run?.adapterPre||!run?.adapterHidden||!run.config.decoderWidth)return null;
 const input=run.adapterPre,output=run.adapterHidden;
 const clamp=(v,max)=>Math.max(0,Math.min(max,Math.trunc(v)||0));
 row=clamp(row,input.rows-1);feature=clamp(feature,input.cols-1);
 const x=matrixValue(input,row,feature),y=matrixValue(output,row,feature),expected=geluFull(x),gate=x===0?.5:expected/x;
 return {input,output,row,feature,x,y,gate,expected,range:featureWindow(input.cols,feature,8)};
}
export function geluCurveMarkup(x){
 const span=Math.max(4,Math.ceil(Math.abs(x))),left=48,right=448,top=26,bottom=222,ymin=-.6,ymax=span+.3;
 const px=v=>left+(v+span)/(2*span)*(right-left),py=v=>bottom-(v-ymin)/(ymax-ymin)*(bottom-top),y=geluFull(x);
 const path=Array.from({length:241},(_,i)=>{const a=-span+i*2*span/240;return `${i?'L':'M'}${px(a).toFixed(3)} ${py(geluFull(a)).toFixed(3)}`;}).join('');
 return `<svg class="gelu-curve" viewBox="0 0 480 258" role="img" aria-label="GELU activation curve. Input ${x.toFixed(4)} maps to ${y.toFixed(4)}. Negative values are smoothly suppressed, not hard-clipped.">
 <path class="gelu-grid" d="M${left} ${py(1)}H${right}M${left} ${py(span)}H${right}"/>
 <path class="gelu-axis" d="M${left} ${py(0)}H${right}M${px(0)} ${bottom}V${top}"/>
 <path class="gelu-identity" d="M${px(ymin)} ${py(ymin)}L${px(span)} ${py(span)}"/>
 <path class="gelu-function" d="${path}"/>
 <path data-gelu-trace class="gelu-trace" pathLength="1" d="M${px(x)} ${py(0)}V${py(y)}H${px(0)}"/>
 <circle data-gelu-dot class="gelu-dot" cx="${px(x)}" cy="${py(y)}" r="5"/>
 ${[-span,-span/2,0,span/2,span].map(v=>`<text x="${px(v)}" y="${bottom+16}" text-anchor="middle">${v}</text>`).join('')}
 <text x="${px(0)-9}" y="${py(span)+4}" text-anchor="end">${span}</text><text x="14" y="16">After GELU(x)</text><text x="380" y="254">Before x</text>
 </svg><div class="gelu-legend"><span><i></i>GELU (demo’s tanh approximation)</span><span><i></i>Unchanged: y = x</span></div>`;
}
export function geluScalarMarkup(data,x=data.x,preview=false){
 const y=geluFull(x),gate=x===0?.5:y/x;
 const insight=x<-1?'A large negative input is pushed toward zero.':x<0?'A small negative input stays negative — unlike ReLU, GELU does not hard-clip it.':x===0?'Zero stays zero. The smooth scaling factor here is 0.5.':x<1?'A small positive input is reduced smoothly.':'A large positive input passes through nearly unchanged.';
 return `<div class="gelu-scalar-heading">${preview?'Sandbox value · matrices unchanged':`Actual p${data.row} · feature ${data.feature}`}</div>
 <div class="gelu-scalar-equation"><span><small>Input x</small><b>${x.toFixed(6)}</b></span><em>×</em><span><small>Smooth factor ≈ Φ(x)</small><b>${gate.toFixed(6)}</b></span><em>=</em><span><small>GELU(x)</small><b>${y.toFixed(6)}</b></span></div>
 <p>${insight}</p><small>${preview?'An isolated example, not a replacement for the selected feature.':`Stored FP32 output: ${data.y.toFixed(7)}. Formula − stored value: ${(y-data.y).toExponential(2)} (rounding).`}</small>`;
}
export function animateGeluTrace(root,{reduced=false,hidden=false}={}){
 const animations=[];
 if(!reduced&&!hidden){
  const line=root.querySelector('[data-gelu-trace]'),dot=root.querySelector('[data-gelu-dot]');
  if(line?.animate)animations.push(line.animate([{strokeDashoffset:1},{strokeDashoffset:0}],{duration:950,easing:'ease-in-out'}));
  if(dot?.animate)animations.push(dot.animate([{opacity:.25},{opacity:1}],{duration:950,easing:'ease-out'}));
 }
 return {dispose(){animations.forEach(a=>a.cancel());}};
}
export function createGeluOverlay({document:doc=globalThis.document,getRun,getState,onSelect=()=>{},onClose=()=>{},renderer=createMatrixRenderer('gelu')}){
 const win=doc.defaultView,media=win.matchMedia('(prefers-reduced-motion: reduce)'),dialog=doc.createElement('dialog');
 dialog.id='full-gelu-dialog';dialog.className='full-math-dialog gelu-dialog';dialog.setAttribute('aria-labelledby','full-gelu-title');dialog.setAttribute('aria-describedby','full-gelu-summary');doc.body.append(dialog);
 let data,source,animation=null,preview=false,value=0,disposed=false,active=false;
 const q=s=>dialog.querySelector(s);
 function pause(){animation?.dispose();animation=null;}
 function replay(){pause();if(dialog.open)animation=animateGeluTrace(dialog,{reduced:media.matches,hidden:doc.hidden});}
 function inspect(animate=false){
  pause();q('[data-gelu-plot]').innerHTML=geluCurveMarkup(value);q('[data-gelu-scalar]').innerHTML=geluScalarMarkup(data,value,preview);
  dialog.dataset.preview=String(preview);q('[data-gelu-sandbox-value]').textContent=value.toFixed(3);q('[data-gelu-slider]').value=value;q('[data-gelu-restore]').disabled=!preview;if(animate)replay();
 }
 function select(row,feature){
  const next=projectorGeluData(getRun(),row,feature);if(!next)return;pause();onSelect(next.row,next.feature);data=projectorGeluData(getRun(),next.row,next.feature);build();replay();
 }
 function build(){
  const scroll=dialog.scrollTop;preview=false;value=data.x;renderer.reset();const width=data.input.cols.toLocaleString(),shape=`${data.input.rows} × ${width}`,span=Math.max(4,Math.ceil(Math.abs(value)));
  dialog.dataset.row=data.row;dialog.dataset.feature=data.feature;
  dialog.innerHTML=`<header class="fm-dialog-header"><div><span>PROJECTOR · ELEMENTWISE ACTIVATION</span><h2 id="full-gelu-title">GELU: a smooth filter for each feature</h2></div><button type="button" data-gelu-close aria-label="Close GELU explanation">×</button></header>
 <div class="fm-dialog-body"><p id="full-gelu-summary" class="gelu-summary"><b>Gaussian Error Linear Unit.</b> After W₁, GELU scales each value independently. It adds nonlinearity before W₂ — no matrix multiplication, no mixing between patches, and no change in shape.</p>
 <div class="gelu-matrices">${renderer.markup(data.input,{label:'Before · first linear Z',row:data.row,col:data.feature,height:100,click:select})}<div class="gelu-elementwise"><strong>GELU →</strong><small>same cell<br>same shape</small></div>${renderer.markup(data.output,{label:'After · GELU(Z)',row:data.row,col:data.feature,height:100,click:select})}</div>
 <div class="gelu-controls"><label>Patch <input type="number" data-gelu-patch min="0" max="${data.input.rows-1}" value="${data.row}"></label><label>Feature <input type="number" data-gelu-feature min="0" max="${data.input.cols-1}" value="${data.feature}"></label><span>${shape} → ${shape} · click either matrix to inspect</span></div>
 <div class="gelu-inspector"><div data-gelu-plot></div><div><div data-gelu-scalar></div><div class="gelu-definition"><code>GELU(x) = x · Φ(x)</code><p>Φ is the standard normal cumulative distribution function: a smooth factor from 0 to 1, applied independently — not attention weights or a softmax.</p><small>The demo uses the tanh approximation:</small><code>½x · [1 + tanh(√(2/π) · (x + 0.044715x³))]</code></div></div></div>
 <div class="gelu-playground"><label>Try an input <input type="range" data-gelu-slider min="${-span}" max="${span}" step="0.025" value="${value}" aria-label="GELU playground input"><output data-gelu-sandbox-value></output></label><div>${[-2,-1,0,1,2].map(x=>`<button type="button" data-gelu-example="${x}" aria-label="Try input ${x}">${x>0?'+':''}${x}</button>`).join('')}<button type="button" data-gelu-restore>↩ Selected feature</button><button type="button" data-gelu-replay>↻ Replay trace</button></div><small>Playground only. Changing this input does not change the model’s matrices.</small></div>
 <details class="gelu-detail"><summary>Compare nearby features before and after · p${data.row}</summary><div class="gelu-feature-strip">${Array.from({length:data.range.end-data.range.start},(_,i)=>{const j=data.range.start+i,a=matrixValue(data.input,data.row,j),b=matrixValue(data.output,data.row,j);return `<button type="button" data-gelu-cell="${j}" aria-pressed="${j===data.feature}"><small>feature ${j}</small><span style="background:rgb(${activationRGB(a)})">${a.toFixed(5)}</span><i>↓ GELU</i><span style="background:rgb(${activationRGB(b)})">${b.toFixed(5)}</span></button>`;}).join('')}</div><p>Actual stored values. Every one of the ${width} features is transformed; this is only a viewing window.</p></details>
 <p class="gelu-footer"><b>Why here?</b> Without GELU, two bias-free linear projections would collapse into one linear projection. GELU makes this mapping nonlinear. Its output is the input to W₂. <a href="https://arxiv.org/abs/1606.08415" target="_blank" rel="noopener noreferrer">GELU paper ↗</a></p></div>`;
  renderer.draw(dialog);inspect();dialog.scrollTop=scroll;
 }
 function open(origin){if(disposed)return;const s=getState();data=projectorGeluData(getRun(),s.selected,s.projectorFeature);if(!data)return;source=origin;active=true;pause();dialog.innerHTML='';doc.documentElement.classList.add('gelu-open');if(!dialog.open)dialog.showModal();build();dialog.scrollTop=0;q('[data-gelu-close]').focus({preventScroll:true});replay();}
 function close(){pause();if(dialog.open)dialog.close();}
 function closed(){if(dialog.open||!active)return;active=false;pause();renderer.reset();dialog.innerHTML='';doc.documentElement.classList.remove('gelu-open');(source?.isConnected?source:doc.querySelector('[data-full-gelu]'))?.focus({preventScroll:true});onClose();}
 function click(e){
  if(renderer.pick(e,dialog))return;const b=e.target.closest('button');
  if(b?.hasAttribute('data-gelu-close'))close();
  else if(b?.hasAttribute('data-gelu-example')){preview=true;value=+b.dataset.geluExample;inspect(true);}
  else if(b?.hasAttribute('data-gelu-restore')){preview=false;value=data.x;inspect(true);}
  else if(b?.hasAttribute('data-gelu-replay'))replay();
  else if(b?.hasAttribute('data-gelu-cell')){const feature=+b.dataset.geluCell;select(data.row,feature);q('.gelu-detail').open=true;q(`[data-gelu-cell="${feature}"]`)?.focus({preventScroll:true});}
  else if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)close();}
 }
 function commit(e){const input=e.target;if(!input.matches('[data-gelu-patch], [data-gelu-feature]')||input.value===''||!Number.isFinite(+input.value))return;const patch=input.hasAttribute('data-gelu-patch');select(patch?+input.value:data.row,patch?data.feature:+input.value);q(patch?'[data-gelu-patch]':'[data-gelu-feature]')?.focus({preventScroll:true});}
 function key(e){if(renderer.pick(e,dialog))return;if(e.key==='Enter'&&e.target.matches('input[type="number"]')){e.preventDefault();commit(e);}}
 function input(e){if(e.target.hasAttribute('data-gelu-slider')){value=+e.target.value;preview=true;inspect();}}
 const visibility=()=>{if(doc.hidden)pause();},reduction=()=>{if(media.matches)pause();},resize=()=>{if(dialog.open)renderer.draw(dialog);};
 dialog.addEventListener('close',closed);dialog.addEventListener('cancel',pause);dialog.addEventListener('click',click);dialog.addEventListener('keydown',key);dialog.addEventListener('change',commit);dialog.addEventListener('input',input);dialog.addEventListener('pointermove',renderer.hover);
 doc.addEventListener('visibilitychange',visibility);media.addEventListener('change',reduction);win.addEventListener('resize',resize);
 return {open,close,get isOpen(){return dialog.open;},dispose(){disposed=true;close();closed();pause();doc.removeEventListener('visibilitychange',visibility);media.removeEventListener('change',reduction);win.removeEventListener('resize',resize);dialog.remove();}};
}
