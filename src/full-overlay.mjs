import {createMatrixRenderer,featureMarkup} from './full-matrix.mjs';
import {matrixValue} from './full-config.mjs';
import {fullOperation} from './full-operations.mjs';
export function fullDotTrace(op,row,col){let sum=0;const prefix=new Float64Array(op.input.cols+1);for(let k=0;k<op.input.cols;k++){sum+=matrixValue(op.input,row,k)*matrixValue(op.weights,k,col);prefix[k+1]=sum;}return {prefix,total:sum,expected:matrixValue(op.output,row,col)};}
export const MULTIPLICATION_PLAYBACK_MS=2400;
/** Elapsed-time playback only reveals a longer prefix; no terms are skipped from the sum. */
export function multiplicationPlaybackCount(total,initial,elapsed){
 const start=Math.max(0,Math.min(total,Math.floor(initial)));
 return Math.min(total,start+Math.floor(total*Math.max(0,elapsed)/MULTIPLICATION_PLAYBACK_MS));
}
export function createFullOverlay({getRun,getState,onPatch,onClose=()=>{}}){
 const dialog=document.createElement('dialog');dialog.className='full-math-dialog';dialog.id='full-math-dialog';dialog.setAttribute('aria-labelledby','full-math-title');document.body.append(dialog);
 const renderer=createMatrixRenderer('overlay'),reduced=matchMedia('(prefers-reduced-motion: reduce)');let op,kind,feature=0,term=0,count=0,trace,playing=false,frame=0,last=null,playedMs=0,playbackBase=0,origin,focusSelector,ids={};
 const q=s=>dialog.querySelector(s);
 function pause(){playing=false;cancelAnimationFrame(frame);frame=0;last=null;dialog.classList.remove('is-playing');if(q('[data-fm-play]'))q('[data-fm-play]').textContent='▶ Play dot product';}
 function compute(){trace=fullDotTrace(op,op.row,feature);}
 function progress(){if(!dialog.open)return;dialog.dataset.row=op.row;dialog.dataset.feature=feature;dialog.dataset.count=count;dialog.dataset.complete=count===op.input.cols;
  q('[data-fm-progress]').value=count;const i=Math.max(0,count-1),a=matrixValue(op.input,op.row,i),b=matrixValue(op.weights,i,feature);
  q('[data-fm-term]').textContent=`x[${i}] ${a.toFixed(6)} × W[${i},${feature}] ${b.toFixed(6)} = ${(a*b).toFixed(7)}`;
  q('[data-fm-sum]').textContent=`${count} / ${op.input.cols.toLocaleString()} terms → ${trace.prefix[count].toFixed(7)}`;q('[data-fm-sum]').dataset.value=trace.prefix[count];q('[data-fm-target]').textContent=`FP32 output[${op.row},${feature}] = ${trace.expected.toFixed(7)}`;q('[data-fm-target]').dataset.value=trace.expected;
  q('[data-fm-error]').textContent=count===op.input.cols?`Float64 inspection sum minus FP32 kernel output: ${(trace.total-trace.expected).toExponential(2)} (rounding / accumulation order).`:'Partial sum. The complete output matrix remains visible for context.';
  renderer.highlight(dialog,ids.input,op.row,i);renderer.highlight(dialog,ids.weights,i,feature);renderer.highlight(dialog,ids.output,op.row,feature);
 }
 function tick(now){
  if(!playing||!dialog.open)return;
  if(last!==null)playedMs+=Math.max(0,now-last);last=now;
  const next=multiplicationPlaybackCount(op.input.cols,playbackBase,playedMs);
  if(next!==count){count=next;progress();}
  if(count===op.input.cols){pause();q('[data-fm-play]').textContent='↻ Replay dot product';return;}
  frame=requestAnimationFrame(tick);
 }
 function play(){
  pause();if(count===op.input.cols)count=0;
  if(reduced.matches){count=op.input.cols;progress();return;}
  playbackBase=count;playedMs=0;last=null;playing=true;dialog.classList.add('is-playing');q('[data-fm-play]').textContent='Ⅱ Pause';progress();frame=requestAnimationFrame(tick);
 }
 function select(row,col){pause();if(row!==op.row)onPatch(row-op.patchOffset);op=fullOperation(getRun(),getState(),kind);feature=Math.max(0,Math.min(op.output.cols-1,Math.floor(col)));term=0;count=0;build();}
 function crop(){const r0=Math.min(Math.max(0,term-3),Math.max(0,op.input.cols-7)),c0=Math.min(Math.max(0,feature-3),Math.max(0,op.output.cols-7));let rows='';for(let r=r0;r<Math.min(op.input.cols,r0+7);r++){rows+=`<tr><th>${r}</th><td>${matrixValue(op.input,op.row,r).toFixed(5)}</td>${Array.from({length:Math.min(7,op.output.cols-c0)},(_,j)=>`<td class="${c0+j===feature?'is-selected':''}">${matrixValue(op.weights,r,c0+j).toFixed(5)}</td>`).join('')}</tr>`;}
 return `<details class="fm-weight-cutaway"><summary>Exact weight values · input rows ${r0}–${Math.min(op.input.cols-1,r0+6)}, output columns ${c0}–${Math.min(op.output.cols-1,c0+6)}</summary><label>Input feature <input type="number" data-fm-term-index min="0" max="${op.input.cols-1}" value="${term}"></label><div class="fm-table-scroll"><table><thead><tr><th>input i</th><th>x[i]</th>${Array.from({length:Math.min(7,op.output.cols-c0)},(_,j)=>`<th>W[i,${c0+j}]</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div><p>Only this numeric window is displayed. The dot product sums all ${op.input.cols.toLocaleString()} inputs.</p></details>`;
 }
 function build(){const cutawayOpen=q('.fm-weight-cutaway')?.open,scroll=dialog.scrollTop;compute();renderer.reset();renderer.setPatchOffset(op.patchOffset);const run=getRun();
  dialog.innerHTML=`<header class="fm-dialog-header"><div><span>${op.blockIndex===null?'':`BLOCK ${op.blockIndex+1} · `}REAL DIMENSIONS · UNTRAINED WEIGHTS</span><h2 id="full-math-title">${op.title}</h2></div><button type="button" data-fm-close aria-label="Close matrix multiplication">×</button></header><div class="fm-dialog-body"><div class="fm-batch-equation"><span>X <b>${op.input.rows} × ${op.input.cols.toLocaleString()}</b></span><i>×</i><span>W <b>${op.weights.rows.toLocaleString()} × ${op.weights.cols.toLocaleString()}</b></span><i>=</i><span>Y <b>${op.output.rows} × ${op.output.cols.toLocaleString()}</b></span></div><p class="fm-caption">${op.description}</p>
  <div class="fm-dialog-matrices">${renderer.markup(op.input,{label:'Input X',row:op.row,col:term,height:185,click:(r,c)=>{pause();onPatch(r-op.patchOffset);op=fullOperation(getRun(),getState(),kind);term=c;count=c+1;build();}})}<b>×</b>${renderer.markup(op.weights,{label:'Shared weights W',row:term,col:feature,scale:Math.sqrt(3/op.weights.rows),height:185,click:(r,c)=>{pause();term=r;feature=c;count=r+1;build();}})}<b>=</b>${renderer.markup(op.output,{label:'Complete output Y',row:op.row,col:feature,height:185,click:select})}</div>
  <div class="fm-dot-controls"><label>Patch <input type="number" data-fm-patch min="${op.patchOffset?-1:0}" max="${run.patchCount-1}" value="${op.row-op.patchOffset}"></label><span>${op.row-op.patchOffset<0?'[CLS]':`p${op.row-op.patchOffset}`} · selected row ${op.row}</span><label>Output feature <input type="number" data-fm-feature min="0" max="${op.output.cols-1}" value="${feature}"></label></div>
  <div class="fm-running-sum"><code data-fm-term></code><strong data-fm-sum></strong><code data-fm-target></code><small data-fm-error></small></div>
  <div class="fm-play-controls"><button type="button" data-fm-play>▶ Play dot product</button><button type="button" data-fm-step>Step +1</button><button type="button" data-fm-reset>Reset</button><label>Included terms <input type="range" data-fm-progress min="0" max="${op.input.cols}" value="${count}"></label><span>All ${op.input.cols.toLocaleString()} terms included · pause, scrub or step for individual terms</span></div>
  ${featureMarkup(op.output,op.row,feature,{label:'Output row',key:'overlay'})}${crop()}<p class="fm-after">${op.after}</p></div>`;
  const canvases=[...dialog.querySelectorAll('[data-full-matrix]')];ids={input:canvases[0].dataset.fullMatrix,weights:canvases[1].dataset.fullMatrix,output:canvases[2].dataset.fullMatrix};renderer.draw(dialog);progress();if(cutawayOpen)q('.fm-weight-cutaway').open=true;dialog.scrollTop=scroll;
 }
 function open(next,source){pause();kind=next;op=fullOperation(getRun(),getState(),kind);if(!op)return;feature=0;term=0;count=0;origin=source;focusSelector=source?.dataset.windowOp?`[data-window-op="${source.dataset.windowOp}"]`:source?.id?'#'+source.id:source?.dataset.fullOp?`${source.closest('.stage-expansion')?'.stage-expansion:not([hidden]) ':''}[data-full-op="${source.dataset.fullOp}"]`:'#stage-trigger-2';dialog.innerHTML='<button autofocus>Loading</button>';document.documentElement.classList.add('multiplication-open');dialog.showModal();build();q('[data-fm-close]').focus({preventScroll:true});play();}
 function close(){pause();dialog.close();}dialog.addEventListener('close',()=>{if(dialog.open)return;pause();dialog.innerHTML='';renderer.reset();document.documentElement.classList.remove('multiplication-open');(origin?.isConnected?origin:document.querySelector(focusSelector)||document.querySelector('#full-patch-wires [tabindex="0"]')||document.querySelector('#stage-trigger-2'))?.focus({preventScroll:true});onClose();});dialog.addEventListener('cancel',pause);
 dialog.addEventListener('click',e=>{if(renderer.pick(e,dialog))return;const b=e.target.closest('button');if(b?.hasAttribute('data-fm-close'))close();else if(b?.hasAttribute('data-fm-play'))playing?pause():play();else if(b?.hasAttribute('data-fm-step')){pause();count=Math.min(count+1,op.input.cols);progress();}else if(b?.hasAttribute('data-fm-reset')){pause();count=0;progress();}else if(b?.hasAttribute('data-full-page'))select(op.row,+b.dataset.fullPage.split(':')[1]);else if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)close();}});
 dialog.addEventListener('keydown',e=>renderer.pick(e,dialog));dialog.addEventListener('pointermove',renderer.hover);
 dialog.addEventListener('input',e=>{if(e.target.hasAttribute('data-fm-progress')){pause();count=+e.target.value;progress();}});
 function commitNumber(e){const v=Math.floor(Number(e.target.value));if(e.target.value===''||!Number.isFinite(v))return;if(e.target.hasAttribute('data-full-feature')||e.target.hasAttribute('data-fm-feature'))select(op.row,v);else if(e.target.hasAttribute('data-fm-patch'))select(Math.max(0,Math.min(op.input.rows-1,Math.floor(v)+op.patchOffset)),feature);else if(e.target.hasAttribute('data-fm-term-index')){pause();term=Math.max(0,Math.min(op.input.cols-1,Math.floor(v)));count=term+1;build();q('.fm-weight-cutaway').open=true;}}
 dialog.addEventListener('change',commitNumber);dialog.addEventListener('keydown',e=>{if(e.key==='Enter'&&e.target.matches('input[type="number"]')){e.preventDefault();commitNumber(e);}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});reduced.addEventListener('change',()=>{if(reduced.matches)pause();});window.addEventListener('resize',()=>{if(dialog.open){renderer.draw(dialog);progress();}});
 return {open,close,get isOpen(){return dialog.open;}};
}
