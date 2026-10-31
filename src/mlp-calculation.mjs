import {fullOperation} from './full-operations.mjs';
import {fullDotTrace,MULTIPLICATION_PLAYBACK_MS,multiplicationPlaybackCount} from './full-overlay.mjs';
import {matrixValue} from './full-config.mjs';
import {geluFull} from './full-engine.mjs';
/** A single inline inspector of the cached, full-width MLP. Never recomputes the model. */
export const MLP_CALCULATION_TIMING=Object.freeze({branches:MULTIPLICATION_PLAYBACK_MS,mix:700,down:MULTIPLICATION_PLAYBACK_MS});
export const MLP_CALCULATION_MS=MLP_CALCULATION_TIMING.branches+MLP_CALCULATION_TIMING.mix+MLP_CALCULATION_TIMING.down;
export function mlpCalculationFrame(elapsed,width,ffWidth){
 const time=Math.max(0,elapsed),boundary=MLP_CALCULATION_TIMING.branches+MLP_CALCULATION_TIMING.mix;
 return {phase:time<MLP_CALCULATION_TIMING.branches?'branches':time<boundary?'activation':time<MLP_CALCULATION_MS?'contraction':'complete',up:multiplicationPlaybackCount(width,0,time),gate:multiplicationPlaybackCount(width,0,time),down:multiplicationPlaybackCount(ffWidth,0,Math.max(0,time-boundary))};
}
export function mlpCalculationData(run,state){
 const c=run.config,l=run.layers[state.layer];if(!l)return null;
 const bounded=(value,limit)=>Math.max(0,Math.min(limit-1,Math.trunc(value)||0)),row=bounded(state.selected+(c.cls?1:0),l.input.rows),wide=bounded(state.wideFeature,c.ffWidth),feature=bounded(state.feature,c.width);
 const ops=['up',...(c.gated?['gate']:[]),'down'].map(key=>({key,operation:fullOperation(run,{...state,selected:row-(c.cls?1:0)},'mlp-'+key),feature:key==='down'?feature:wide}));
 return {c,l,row,wide,feature,ops};
}
export function mlpCalculationTraces(data){return Object.fromEntries(data.ops.map(({key,operation,feature})=>[key,fullDotTrace(operation,data.row,feature)]));}
export function mlpCalculationMarkup(renderer,run,state,onSelect=()=>{},onFeature=()=>{}){
 const d=mlpCalculationData(run,state);if(!d)return '';const {c,l,row,wide,feature}=d,n=l.input.rows,patch=row-(c.cls?1:0),fmt=x=>x.toLocaleString();
 function matrix(m,label,id,col=-1,click=null,weight=false){return `<div data-mlp-matrix="${id}">${renderer.markup(m,{label,row:weight?-1:row,col,height:112,scale:weight?Math.sqrt(3/m.rows):2,click})}</div>`;}
 function operationCard(key){
  const {operation:op,feature:col}=d.ops.find(o=>o.key===key),down=key==='down',gate=key==='gate',w=down?'W₂':gate?'W₃':'W₁',output=down?'Δ':gate?'B':'A',input=down?'C':'X';
  return `<article class="mi-operation" data-mlp-operation="${key}"><header><h4>${w} · ${down?'Contraction':gate?'Parallel expansion':c.gated?'Expansion → SiLU gate':'Expansion → GELU'}</h4><span>${input} × ${w} = ${output}</span></header><div class="mi-equation">
  ${matrix(op.input,input,`${key}-input`,-1,(r,f)=>onSelect(r,f,op.input.cols))}<b aria-hidden="true">×</b>${matrix(op.weights,w,`${key}-weights`,col,(_r,f)=>onFeature(down?'feature':'wide',f),true)}<b aria-hidden="true">=</b>${matrix(op.output,`${output} · complete output`,`${key}-output`,col,(r,f)=>onSelect(r,f,op.output.cols))}
  </div><div class="mi-operation-footer"><span>${down?'All expanded features contribute to each output.':gate?'The same X enters W₃ directly, not the output of W₁.':c.gated?'Apply SiLU to A, then multiply by B elementwise.':'Apply GELU to A before contraction.'}</span><small data-mlp-count="${key}">${fmt(op.input.cols)} / ${fmt(op.input.cols)} terms</small></div></article>`;
 }
 return `<section class="mi-calculation" id="mlp-calculation" data-mlp-calculation data-mlp-row="${row}" data-mlp-block="${state.layer}" data-mlp-wide-feature="${wide}" data-mlp-output-feature="${feature}" aria-label="${c.gated?'All three MLP multiplications':'Both MLP multiplications'} in one connected calculation">
 <div class="mi-toolbar"><div><strong>${c.gated?'One input · two parallel branches · one contraction':'One input · GELU · one contraction'}</strong><span>${patch<0?'CLS':`Patch p${patch}`} · block ${state.layer+1} / ${c.depth} · ${n} total ${c.cls?'tokens':'patches'}</span></div><button type="button" data-mlp-play>↻ Replay calculation</button><button type="button" data-mlp-calculation-toggle="close" aria-expanded="true" aria-controls="mlp-calculation" aria-label="Collapse MLP calculation">− Calculation</button></div>
 <p class="mi-context">Every matrix keeps its full dimensions. Highlights follow one dot product per projection; output matrices always show their complete cached values.</p>
 <div class="mi-branches ${c.gated?'':'mi-single-branch'}">${operationCard('up')}${c.gated?operationCard('gate'):''}</div>
 <div class="mi-join" aria-hidden="true">${c.gated?'<span>A ↓</span><span>↓ B</span>':'<span>A ↓</span>'}</div>
 <div class="mi-activation" data-mlp-activation><strong>C = ${c.gated?'SiLU(A) ⊙ B':'GELU(A)'}</strong><span>${n} × ${fmt(c.ffWidth)} → ${n} × ${fmt(c.ffWidth)} · ${c.gated?'multiply matching elements, not matrices':'elementwise activation'}</span><small>${c.gated?'Both branches finish before the product. No tokens or features are summed here.':'GELU acts on each element; it does not mix tokens.'} C stays full size and feeds W₂ below.</small></div>
 <div class="mi-join mi-down-arrow" aria-hidden="true">C ↓</div>${operationCard('down')}
 <div class="mi-residual"><span class="mi-residual-formula">Δ + H = H′<small>Add the pre-MLP residual, not normalized X.</small></span>${matrix(l.afterAttention,'H · residual input','residual',feature,(r,f)=>onSelect(r,f,c.width))}<b aria-hidden="true">→</b>${matrix(l.output,'H′ · after residual','final',feature,(r,f)=>onSelect(r,f,c.width))}</div>
 <div class="mi-feature-controls"><label>Expanded feature <input data-mlp-feature="wide" type="number" min="0" max="${c.ffWidth-1}" value="${wide}"></label><label>Output feature <input data-mlp-feature="feature" type="number" min="0" max="${c.width-1}" value="${feature}"></label></div>
 <p class="mi-status" data-mlp-status aria-live="off">Complete · every multiplication sums all of its inputs.</p>
 <p class="mi-context">Shared deterministic untrained weights; biases omitted. The patch column, feature controls and matrix selections update this whole calculation together.</p>
 </section>`;
}
export function mlpActivationTrace(data){
 const a=matrixValue(data.l.preActivation,data.row,data.wide),activated=data.c.gated?a/(1+Math.exp(-a)):geluFull(a),b=data.c.gated?matrixValue(data.l.gate,data.row,data.wide):null;
 return {a,activated,b,computed:b===null?activated:activated*b,expected:matrixValue(data.l.expanded,data.row,data.wide)};
}
export function mlpCalculationNumbersMarkup(run,state){
 const d=mlpCalculationData(run,state);if(!d)return '';const activation=mlpActivationTrace(d);
 return `<p class="mi-activation-values">Expanded feature ${d.wide}: <code>${d.c.gated?`SiLU(${activation.a.toFixed(6)}) × ${activation.b.toFixed(6)}`:`GELU(${activation.a.toFixed(6)})`} ≈ ${activation.computed.toFixed(7)}</code> · stored C = ${activation.expected.toFixed(7)}. FP32 rounding may differ from inspection arithmetic.</p><div class="mi-numeric-traces">${d.ops.map(({key,operation,feature})=>`<section class="mi-numeric-trace"><h4>${key==='up'?'W₁':key==='gate'?'W₃':'W₂'} · row ${d.row}, output feature ${feature}</h4><label>Included terms <input type="range" data-mlp-term="${key}" min="0" max="${operation.input.cols}" value="${operation.input.cols}"></label><code data-mlp-term-value="${key}"></code><strong data-mlp-sum="${key}"></strong><code data-mlp-expected="${key}"></code><small data-mlp-rounding="${key}"></small></section>`).join('')}</div>`;
}
export function mountMlpCalculation(root,{run,state,renderer,onFeature,autoplay=false}){
 const doc=root.ownerDocument,win=doc.defaultView,media=win.matchMedia('(prefers-reduced-motion: reduce)'),data=mlpCalculationData(run,state),host=root.closest('.expansion-inspection')||root;
 if(!data)return {dispose(){},play(){}};
 const traces=mlpCalculationTraces(data),ids=Object.fromEntries([...root.querySelectorAll('[data-mlp-matrix]')].map(el=>[el.dataset.mlpMatrix,el.querySelector('[data-full-matrix]')?.dataset.fullMatrix]));
 const q=selector=>host.querySelector(selector);let frame=null,disposed=false,playing=false,last=0,elapsed=MLP_CALCULATION_MS,lastPaint=-Infinity,entry=null;
 function writeTrace(key,count){const {operation:op,feature}=data.ops.find(o=>o.key===key),trace=traces[key];
  q(`[data-mlp-count="${key}"]`).textContent=`${count.toLocaleString()} / ${op.input.cols.toLocaleString()} terms`;
  const range=q(`[data-mlp-term="${key}"]`);if(range){range.value=count;const i=count-1;
   q(`[data-mlp-term-value="${key}"]`).textContent=count?`x[${i}] ${matrixValue(op.input,data.row,i).toFixed(6)} × W[${i},${feature}] ${matrixValue(op.weights,i,feature).toFixed(6)}`:'No terms included yet.';
   const sum=q(`[data-mlp-sum="${key}"]`);sum.textContent=`${count===op.input.cols?'Complete inspection sum':'Partial sum'}: ${trace.prefix[count].toFixed(7)}`;sum.dataset.value=trace.prefix[count];
   q(`[data-mlp-expected="${key}"]`).textContent=`Stored FP32 output[${data.row},${feature}] = ${trace.expected.toFixed(7)}`;
   q(`[data-mlp-rounding="${key}"]`).textContent=count===op.input.cols?`Float64 sum − FP32 output: ${(trace.total-trace.expected).toExponential(2)} (rounding / accumulation order).`:'Output matrices above remain complete; this trace is a partial inspection only.';
  }
  root.dataset[key+'Count']=count;
  const term=count?count-1:-1;for(const [part,r,c] of [['input',data.row,term],['weights',term,feature],['output',data.row,feature]]){const id=ids[`${key}-${part}`];if(id)renderer.highlight(root,id,r,c);}
 }
 function paint(){const step=mlpCalculationFrame(elapsed,data.c.width,data.c.ffWidth);root.dataset.phase=step.phase;root.dataset.playing=String(playing);
  root.querySelectorAll('[data-mlp-operation]').forEach(el=>el.classList.toggle('is-current',step.phase==='branches'?el.dataset.mlpOperation!=='down':step.phase==='contraction'&&el.dataset.mlpOperation==='down'));
  root.querySelector('[data-mlp-activation]').classList.toggle('is-current',step.phase==='activation');
  q('[data-mlp-status]').textContent=step.phase==='branches'?`W₁${data.c.gated?' and W₃':''}: reading the same normalized input${data.c.gated?' in parallel':''}.`:step.phase==='activation'?`${data.c.gated?'Apply SiLU to A, then multiply by B elementwise':'Apply GELU to A'}. This produces the full C matrix.`:step.phase==='contraction'?'W₂: sum across every feature of C to produce the MLP update Δ.':'Complete · all terms included, then add residual H to Δ.';
  for(const {key} of data.ops)writeTrace(key,step[key]);
  q('[data-mlp-play]').textContent=playing?'Ⅱ Pause':elapsed>=MLP_CALCULATION_MS?'↻ Replay calculation':'▶ Resume calculation';
 }
 function pause(){playing=false;if(frame!==null)win.cancelAnimationFrame(frame);frame=null;root.dataset.playing='false';q('[data-mlp-play]').textContent=elapsed>=MLP_CALCULATION_MS?'↻ Replay calculation':'▶ Resume calculation';}
 function tick(now){frame=null;if(disposed||!playing)return;if(!root.isConnected){dispose();return;}if(doc.hidden||doc.querySelector('dialog[open]')){pause();return;}elapsed=Math.min(MLP_CALCULATION_MS,elapsed+Math.max(0,now-last));last=now;
  if(now-lastPaint>=45||elapsed===MLP_CALCULATION_MS){lastPaint=now;paint();}if(elapsed===MLP_CALCULATION_MS){pause();return;}frame=win.requestAnimationFrame(tick);
 }
 function play(){if(disposed)return;pause();if(elapsed>=MLP_CALCULATION_MS)elapsed=0;if(media.matches){elapsed=MLP_CALCULATION_MS;paint();return;}if(doc.hidden)return;playing=true;last=win.performance.now();lastPaint=-Infinity;paint();frame=win.requestAnimationFrame(tick);}
 function click(e){if(e.target.closest('[data-mlp-play]')){playing?pause():play();}}
 function input(e){if(!e.target.hasAttribute('data-mlp-term'))return;const key=e.target.dataset.mlpTerm,op=data.ops.find(o=>o.key===key);if(!op)return;pause();elapsed=MLP_CALCULATION_MS;root.dataset.phase='inspection';root.querySelectorAll('.is-current').forEach(el=>el.classList.remove('is-current'));writeTrace(key,Math.max(0,Math.min(op.operation.input.cols,Math.trunc(+e.target.value)||0)));q('[data-mlp-status]').textContent='Manual dot-product inspection. Stored matrices are unchanged.';q('[data-mlp-play]').textContent='↻ Replay calculation';}
 function change(e){if(!e.target.hasAttribute('data-mlp-feature')||e.target.value===''||!Number.isFinite(+e.target.value))return;pause();onFeature(e.target.dataset.mlpFeature,+e.target.value);}
 const key=e=>{if(e.key==='Enter'&&e.target.hasAttribute('data-mlp-feature')){e.preventDefault();change(e);}},visibility=()=>{if(doc.hidden)pause();},preference=()=>{if(media.matches){entry?.cancel();pause();elapsed=MLP_CALCULATION_MS;paint();}},resize=()=>{if(root.dataset.phase==='inspection'){for(const {key} of data.ops)writeTrace(key,+root.dataset[key+'Count']);}else paint();};
 function dispose(){if(disposed)return;pause();disposed=true;entry?.cancel();host.removeEventListener('click',click);host.removeEventListener('input',input);host.removeEventListener('change',change);host.removeEventListener('keydown',key);doc.removeEventListener('visibilitychange',visibility);media.removeEventListener('change',preference);win.removeEventListener('resize',resize);}
 host.addEventListener('click',click);host.addEventListener('input',input);host.addEventListener('change',change);host.addEventListener('keydown',key);doc.addEventListener('visibilitychange',visibility);media.addEventListener('change',preference);win.addEventListener('resize',resize);paint();
 if(autoplay){if(!media.matches&&root.animate)entry=root.animate([{opacity:.35,transform:'translateY(-12px)'},{opacity:1,transform:'none'}],{duration:550,easing:'ease-out'});play();}
 return {dispose,play};
}
