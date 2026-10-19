/** Small, exact matrix-product cutaways. All operands come from the live encoder. */
import {outputColor} from './readout.mjs';
const mmNumber=(x,d=4)=>Number(x).toFixed(d);
const mmEscape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function multiplicationOperation(s,patches,result,kind){
 const token=s.selected+(s.model==='vit'?1:0),l=result.layers[s.layer];
 const patch=s.selected<0?'[CLS]':`p${String(s.selected).padStart(2,'0')} · (${patches[s.selected].row}, ${patches[s.selected].col})`;
 const common={kind,patch,layer:s.layer,row:token,hasCls:s.model==='vit',shared:'Same weights for every patch. Different input vectors; independent products.',inputName:'x',weightName:'W',outputName:'y',after:''};
 if(kind==='embedding'){
  if(s.selected<0)throw Error('[CLS] is a learned vector, not a pixel projection.');
  return {...common,title:'Patch embedding',row:s.selected,hasCls:false,inputs:patches.map(p=>p.pixels),outputs:result.embeddings,input:patches[s.selected].pixels,weights:result.projectionWeights,output:result.embeddings[s.selected],weightName:'Wₑ',outputName:'e',inputLabel:'Flattened, normalized RGB',after:s.model==='vit'?'Next: add the learned position vector.':'Next: RMS-normalize the embedding. 2D RoPE comes inside attention.'};
 }
 if(kind.startsWith('mlp-')){
  const op=kind.slice(4);if(!['up','down','gate'].includes(op)||op==='gate'&&s.model!=='pixtral')throw Error('Unavailable MLP operation');
  return {...common,inputs:op==='down'?l.expanded:l.ffnInput,outputs:op==='down'?l.mlpDelta:op==='gate'?l.gate:l.preActivation,title:op==='up'?'MLP Expansion':op==='down'?'MLP Contraction':'MLP Gate projection',input:op==='down'?l.expanded[token]:l.ffnInput[token],weights:l.weights[op],output:op==='down'?l.mlpDelta[token]:op==='gate'?l.gate[token]:l.preActivation[token],weightName:op==='up'?'W₁':op==='down'?'W₂':'W₃',outputName:op==='down'?'Δh':op==='gate'?'g':'z',inputLabel:op==='down'?'After activation / gating':s.model==='vit'?'After pre-MLP LayerNorm':'After pre-MLP RMSNorm',after:op==='down'?'Next: add Δh to the pre-MLP residual. This product alone is not the final hidden state.':s.model==='vit'?'Next: GELU(z), then the 48 → 24 contraction. This product is before GELU.':op==='up'?'Next: SiLU(z) ⊙ g, then contraction. W₃ supplies the separate gate vector g.':'Next: multiply g elementwise with SiLU(z). This gate projection itself has no activation.'};
 }
 if(kind==='projector-first'||kind==='projector-second'){
  if(s.model!=='pixtral')throw Error('The projector belongs to the Pixtral path.');
  const second=kind==='projector-second';
  return {...common,inputs:second?result.adapterHidden:result.final,outputs:second?result.projected:result.adapterPre,title:second?'Projector · second linear':'Projector · first linear',input:second?result.adapterHidden[token]:result.final[token],weights:result.adapterWeights[second?1:0],output:second?result.projected[token]:result.adapterPre[token],inputLabel:second?'After projector GELU':'Final encoder vector',weightName:second?'W₂':'W₁',after:second?'This vector occupies one image slot in the symbolic text-decoder prefix.':'Next: GELU, then the second projector linear layer.'};
 }
 throw Error('Unknown multiplication');
}
export function multiplicationTrace(op,column){
 if(!Number.isInteger(column)||column<0||column>=op.output.length)throw Error('Invalid output feature');
 const terms=op.input.map((x,i)=>x*op.weights[i][column]),sums=[0];
 for(const term of terms)sums.push(sums.at(-1)+term);
 return {terms,sums,total:sums.at(-1),expected:op.output[column]};
}
export function multiplicationProgress(trace,count){
 const n=Math.max(0,Math.min(trace.terms.length,Math.floor(count))),i=Math.max(0,n-1);
 return {count:n,index:i,term:n?trace.terms[i]:0,sum:trace.sums[n],complete:n===trace.terms.length};
}
export function multiplicationRowStep(current,key,count,hasCls){
 const first=hasCls?-1:0,last=count-1;
 if(key==='Home')return first;if(key==='End')return last;
 const delta={ArrowDown:1,ArrowRight:1,ArrowUp:-1,ArrowLeft:-1}[key];
 return delta===undefined?null:Math.max(first,Math.min(last,current+delta));
}
function mlpSmallVector(values){return `<span class="mlp-lane-vector" style="--lane-width:${values.length}" aria-hidden="true">${values.map(v=>`<i style="background:${outputColor(v)}"></i>`).join('')}</span>`;}
/** Each lane has different activations, NOT independently trained weights. */
export function mlpLanesMarkup(s,patches,result,{compact=false}={}){
 const l=result.layers[s.layer],hasCls=s.model==='vit';
 const wire=(wide)=>`<svg class="mlp-lane-wire" viewBox="0 0 22 18" preserveAspectRatio="none" aria-hidden="true"><path d="${wide?'M0 6 C10 6 12 2 22 2 L22 16 C12 16 10 12 0 12 Z':'M0 2 C10 2 12 6 22 6 L22 12 C12 12 10 16 0 16 Z'}"/><path class="mlp-lane-signal" d="M0 9 H22"/></svg>`;
 return `<div class="mlp-lanes ${compact?'is-compact':''}"><div class="mlp-lanes-titles"><span>token · norm</span><button type="button" data-multiply="mlp-up" aria-haspopup="dialog">MLP Expansion ↗<small>24 → 48</small></button><button type="button" data-multiply="mlp-down" aria-haspopup="dialog">Contraction ↗<small>48 → 24</small></button></div>
 <div class="mlp-lanes-scroll" role="group" aria-label="Independent MLP paths for every token" data-selected="${s.selected}" data-signature="${s.model}:${s.patch}:${s.rows}:${s.cols}">${l.ffnInput.map((v,i)=>{const p=i-(hasCls?1:0),label=p<0?'[CLS]':`p${String(p).padStart(2,'0')}`;return `<div class="mlp-lane ${p===s.selected?'is-selected':''}" data-mlp-lane="${p}"><button type="button" class="mlp-lane-input" data-mlp-patch="${p}" aria-pressed="${p===s.selected}" tabindex="${p===s.selected?0:-1}" aria-label="Select ${label} MLP path"><span>${label}</span>${mlpSmallVector(v)}</button>${wire(true)}<button type="button" class="mlp-lane-expand" data-multiply="mlp-up" data-multiply-patch="${p}" aria-haspopup="dialog" aria-label="Animate ${label} MLP Expansion" tabindex="${p===s.selected?0:-1}">${mlpSmallVector(l.expanded[i])}</button>${wire(false)}<button type="button" class="mlp-lane-contract" data-multiply="mlp-down" data-multiply-patch="${p}" aria-haspopup="dialog" aria-label="Animate ${label} MLP Contraction" tabindex="${p===s.selected?0:-1}">${mlpSmallVector(l.mlpDelta[i])}</button></div>`;}).join('')}</div>
 <div class="mlp-lanes-footer"><span>${hasCls?'GELU':'SiLU × gate'} between products · + residual after</span>${!hasCls?'<button type="button" data-multiply="mlp-gate" aria-haspopup="dialog">Inspect W₃ gate ↗</button>':''}<strong>Same W₁ / W₂${hasCls?'':' / W₃'} on every row</strong></div></div>`;
}
/** Complete product, never just a sampled output row. The active cell remains the full result. */
export function multiplicationOutputMarkup(op){
 const cols=op.output.length;
 return `<div class="mm-result"><div class="mm-matrix-label"><strong>Output ${op.outputName.toUpperCase()}</strong><small>${op.outputs.length} × ${cols}</small></div>
 <div class="mm-result-matrix" role="group" aria-label="Full multiplication output: ${op.outputs.length} tokens by ${cols} features" style="--mm-result-rows:${op.outputs.length};--mm-result-cols:${cols}">${op.outputs.map((row,r)=>{
  const patch=r-(op.hasCls?1:0),name=patch<0?'[CLS]':`p${String(patch).padStart(2,'0')}`;
  return `<button type="button" class="mm-result-row" data-mm-result-row="${r}" aria-label="Select ${name} output row, ${cols} features" aria-pressed="${r===op.row}" tabindex="${r===op.row?0:-1}">${row.map((v,j)=>`<span data-mm-result-column="${j}" data-mm-result-token="${r}" data-mm-result-value="${v}" style="background:${outputColor(v)}" title="${name} · feature ${j}: ${mmNumber(v,8)}"></span>`).join('')}</button>`;
 }).join('')}</div><div class="mm-axis"><span>tokens ↓</span><span>features →</span></div><small class="mm-result-caption">All computed values · click a row / cell<br>Outlined cell = the dot product being inspected</small></div>`;
}
export function multiplicationMarkup(op,s,patches,imageURL){
 const n=op.input.length,m=op.output.length,first=op.kind==='embedding'?0:s.model==='vit'?-1:0;
 const operations=op.kind.startsWith('mlp-')?[['mlp-up','Expansion'],...(s.model==='pixtral'?[['mlp-gate','Gate']]:[]),['mlp-down','Contraction']]:op.kind.startsWith('projector-')?[['projector-first','First linear'],['projector-second','Second linear']]:[['embedding','RGB → embedding']];
 return `<header class="mm-header"><div><span class="mm-eyebrow">MATRIX MULTIPLICATION · ${op.kind.startsWith('mlp-')?`BLOCK ${s.layer+1}`:'VISION ENCODER'}</span><h2 id="multiplication-title">${op.title}</h2></div><button type="button" class="mm-close" data-mm-close aria-label="Close multiplication overlay">×</button></header>
 <div class="mm-context"><div class="mm-operation-tabs" role="group" aria-label="Linear operation">${operations.map(([kind,label])=>`<button type="button" data-mm-operation="${kind}" aria-pressed="${kind===op.kind}">${label}</button>`).join('')}</div><span>LIVE VALUES · UNTRAINED WEIGHTS · BIAS OMITTED</span></div>
 <div class="mm-layout"><aside class="mm-patches"><strong>PATCH SEQUENCE</strong><p>${patches.length} patches${first<0?' + CLS':''}</p><div class="mm-patch-list" role="group" aria-label="Choose a token for multiplication">${Array.from({length:patches.length-first},(_,i)=>i+first).map(p=>{const patch=patches[p],label=p<0?'[CLS]':`p${String(p).padStart(2,'0')}`,style=p<0?'':`background-image:url('${imageURL}');background-size:${s.cols*100}% ${s.rows*100}%;background-position:${s.cols===1?0:patch.col/(s.cols-1)*100}% ${s.rows===1?0:patch.row/(s.rows-1)*100}%`;return `<button type="button" data-mm-patch="${p}" aria-pressed="${p===s.selected}" tabindex="${p===s.selected?0:-1}" aria-label="Inspect ${label} multiplication"><i style="${mmEscape(style)}" aria-hidden="true">${p<0?'CLS':''}</i><span>${label}</span></button>`;}).join('')}</div><small>One patch at a time.<br>No mixing between rows.</small></aside>
 <div class="mm-main"><div class="mm-equation"><span>${op.patch} · row ${op.row}</span><code>X <small>${op.inputs.length} × ${n}</small> <b>×</b> ${op.weightName} <small>${n} × ${m}</small> <b>=</b> ${op.outputName.toUpperCase()} <small>${op.outputs.length} × ${m}</small></code></div>
 <p class="mm-shared">${op.shared}</p>
 <div class="mm-board"><svg class="mm-math-wires" aria-hidden="true"></svg><div class="mm-input"><div class="mm-matrix-label">${op.inputName}<small>row ${op.row}</small></div><canvas id="mm-input" aria-label="Input vector, displayed vertically to align with weight rows" role="img"></canvas><small>${op.inputLabel}<br>${n} values · selected row shown vertically</small></div><b class="mm-times">×</b>
 <div class="mm-weights"><div class="mm-matrix-label">${op.weightName} <small>all ${n} × ${m} weights · click any cell</small></div><canvas id="mm-weights" role="img" tabindex="0" aria-label="Weight matrix. Arrow keys select an input row and output column."></canvas><div class="mm-axis"><span>input features ↓</span><span>output features →</span></div></div>
 <b class="mm-equals">=</b>${multiplicationOutputMarkup(op)}</div>
 <div class="mm-calculation" data-mm-calculation><div><small data-mm-term-label>READY · INPUT i = 0</small><code><span data-mm-input-value></span> <b>×</b> <span data-mm-weight-value></span> <b>=</b> <strong data-mm-product></strong></code></div><span class="mm-moving-term" aria-hidden="true">→</span><div><small data-mm-sum-label>PARTIAL SUM · 0 / ${n}</small><code data-mm-sum>0.000000</code></div><span class="mm-target"><small>OUTPUT CELL · row ${op.row}, j=<span data-mm-column>0</span></small><code data-mm-target></code></span></div>
 <div class="mm-outputs-heading"><span>Selected row ${op.row} · ${m} output features · click to inspect a column</span><small>Signed values · coral − / blue +</small></div><div class="mm-outputs" role="group" aria-label="Output feature selection">${op.output.map((v,j)=>`<button type="button" data-mm-output="${j}" aria-pressed="${j===0}" tabindex="${j===0?0:-1}" style="--mm-value:${outputColor(v)}" title="${op.outputName}[${j}] = ${mmNumber(v,8)}"><small>${j}</small><span>${mmNumber(v,2)}</span></button>`).join('')}</div>
 <p class="mm-after">${op.after} <span>Displayed numbers are rounded; sums use full precision. Weight colors use a separate scale from activations.</span></p></div></div>
 <footer class="mm-controls"><div><button type="button" data-mm-play>Pause</button><button type="button" data-mm-step>Step +1</button><button type="button" data-mm-sweep>Sweep this row</button><button type="button" data-mm-reset>Reset</button></div><label>Terms <input type="range" min="0" max="${n}" value="0" data-mm-progress aria-label="Terms included in the dot product"></label><span data-mm-status role="status" aria-live="polite">One dot product, term by term.</span></footer>`;
}
/** Dialog leaves architecture geometry, selection and zoom level intact. */
export function createMultiplicationOverlay({document:doc,getData,onPatch,onClose}){
 const win=doc.defaultView,root=doc.createElement('dialog');root.id='multiplication-dialog';root.className='mm-dialog';root.setAttribute('aria-labelledby','multiplication-title');doc.body.append(root);
 const reduce=win.matchMedia('(prefers-reduced-motion: reduce)');
 let op=null,kind='',column=0,count=0,trace=null,raf=0,last=0,playing=false,mode='dot',opener=null,returnSelector='',inputBitmap=null,weightBitmap=null,disposed=false;
 const q=selector=>root.querySelector(selector);
 function stop(){playing=false;win.cancelAnimationFrame(raf);raf=0;last=0;root.classList.remove('is-playing');if(q('[data-mm-play]'))q('[data-mm-play]').textContent=count>=op.input.length?'Replay dot product':'Play dot product';}
 function bitmap(width,height){const c=doc.createElement('canvas');c.width=width;c.height=height;return c;}
 function prepareMatrices(){
  const n=op.input.length,m=op.output.length;inputBitmap=bitmap(64,576);weightBitmap=bitmap(960,576);
  const ig=inputBitmap.getContext('2d'),wg=weightBitmap.getContext('2d'),cw=960/m,ch=576/n;
  op.input.forEach((v,i)=>{ig.fillStyle=outputColor(v);ig.fillRect(0,i*ch,64,Math.max(.2,ch-.5));op.weights[i].forEach((w,j)=>{wg.fillStyle=outputColor(w*Math.sqrt(n));wg.fillRect(j*cw,i*ch,Math.max(.2,cw-.6),Math.max(.2,ch-.5));});});
 }
 function paint(){
  if(!root.open||!op)return;
  const n=op.input.length,m=op.output.length,f=multiplicationProgress(trace,count),i=f.index;
  root.dataset.column=column;root.dataset.count=count;root.dataset.complete=f.complete;
  const ic=q('#mm-input'),wc=q('#mm-weights');
  if(ic.width!==64||ic.height!==576){ic.width=64;ic.height=576;}if(wc.width!==960||wc.height!==576){wc.width=960;wc.height=576;}
  const ig=ic.getContext('2d'),wg=wc.getContext('2d');ig.clearRect(0,0,64,576);ig.drawImage(inputBitmap,0,0);wg.clearRect(0,0,960,576);wg.drawImage(weightBitmap,0,0);
  const cw=960/m,ch=576/n;wg.fillStyle='#7654d624';wg.fillRect(column*cw,0,cw,576);wg.strokeStyle='#8661cb';wg.lineWidth=2;wg.strokeRect(column*cw+1,1,cw-2,574);
  for(const [g,w,x] of [[ig,64,0],[wg,cw,column*cw]]){g.fillStyle='#e5aa3d55';g.fillRect(x,i*ch,w,Math.max(3,ch));g.strokeStyle='#ad711a';g.lineWidth=2;g.strokeRect(x+1,i*ch+1,w-2,Math.max(2,ch-2));}
  root.querySelectorAll('[data-mm-result-row]').forEach(b=>{const active=+b.dataset.mmResultRow===op.row;b.setAttribute('aria-pressed',active);b.tabIndex=active?0:-1;});
  root.querySelectorAll('[data-mm-result-column]').forEach(cell=>{const selected=+cell.dataset.mmResultToken===op.row&&+cell.dataset.mmResultColumn===column;cell.classList.toggle('is-active-cell',selected);cell.classList.toggle('is-finished-cell',selected&&f.complete);});
  root.dataset.row=op.row;
  q('[data-mm-column]').textContent=column;q('[data-mm-input-value]').textContent=mmNumber(op.input[i]);q('[data-mm-input-value]').dataset.value=op.input[i];q('[data-mm-weight-value]').textContent=mmNumber(op.weights[i][column]);q('[data-mm-weight-value]').dataset.value=op.weights[i][column];q('[data-mm-product]').textContent=mmNumber(trace.terms[i],6);q('[data-mm-product]').dataset.value=trace.terms[i];
  q('[data-mm-term-label]').textContent=`${count?'CURRENT TERM':'NEXT TERM'} · i = ${i}, j = ${column}`;
  q('[data-mm-sum-label]').textContent=`${f.complete?'COMPLETE DOT PRODUCT':'PARTIAL SUM'} · ${count} / ${n} terms`;
  const sum=q('[data-mm-sum]');sum.textContent=mmNumber(f.sum,6);sum.dataset.value=f.sum;q('[data-mm-target]').textContent=mmNumber(op.output[column],6);q('[data-mm-target]').dataset.value=op.output[column];
  q('[data-mm-progress]').value=count;
  q('[data-mm-play]').textContent=playing?'Pause':f.complete?'Replay dot product':'Play dot product';
  if(!playing)q('[data-mm-status]').textContent=f.complete?`Output ${column} complete. Sum matches the live encoder output.`:`Column ${column}: ${count} of ${n} terms. Play, step, or scrub to continue.`;
  root.querySelectorAll('[data-mm-output]').forEach(b=>{const active=+b.dataset.mmOutput===column;b.setAttribute('aria-pressed',active);b.tabIndex=active?0:-1;b.classList.toggle('is-computed',mode==='sweep'&&+b.dataset.mmOutput<column||active&&f.complete);});
  q('#mm-weights').setAttribute('aria-label',`Weight matrix ${n} by ${m}. Input row ${i}, output column ${column}: ${mmNumber(op.weights[i][column],6)}. Arrow keys change the selected cell.`);
  const board=q('.mm-board').getBoundingClientRect(),a=ic.getBoundingClientRect(),b=wc.getBoundingClientRect(),wires=q('.mm-math-wires');
  const x1=a.right-board.left,y1=a.top-board.top+(i+.5)*a.height/n,x2=b.left-board.left+(column+.5)*b.width/m,y2=b.top-board.top+(i+.5)*b.height/n,bottom=b.bottom-board.top;
  const dest=q('.mm-result-matrix').getBoundingClientRect(),dx=dest.left-board.left+(column+.5)*dest.width/m,dy=dest.top-board.top+(op.row+.5)*dest.height/op.outputs.length;
  wires.setAttribute('viewBox',`0 0 ${board.width} ${board.height}`);wires.innerHTML=`<path d="${patchConnectionPath(x1,y1,x2,y2)}"/><circle cx="${x2}" cy="${y2}" r="3"/><path class="mm-to-output" d="M${x2},${bottom} C${x2},${bottom+17} ${dest.left-board.left-10},${bottom+17} ${dest.left-board.left-10},${dy} H${dx}"/><circle class="mm-result-dot" cx="${dx}" cy="${dy}" r="2.5"/>`;

 }
 function setColumn(j,{terms=0}={}){stop();column=Math.max(0,Math.min(op.output.length-1,j));trace=multiplicationTrace(op,column);count=terms;mode='dot';paint();}
 function tick(now){
  if(!playing||!root.open||disposed)return;
  if(!last)last=now;
  if(now-last>=220){
   last=now;
   if(mode==='dot'){count=Math.min(op.input.length,count+Math.max(1,Math.ceil(op.input.length/48)));paint();if(count===op.input.length){stop();q('[data-mm-status]').textContent=`Output ${column} complete. Sum matches the live encoder output.`;return;}}
   else{if(count===op.input.length){column++;trace=multiplicationTrace(op,column);}count=op.input.length;paint();if(column===op.output.length-1){stop();q('[data-mm-status]').textContent=`All ${op.output.length} outputs computed from the same input vector.`;return;}}
  }
  raf=win.requestAnimationFrame(tick);
 }
 function play(sweep=false){
  stop();mode=sweep?'sweep':'dot';
  if(sweep){column=0;trace=multiplicationTrace(op,column);count=0;}
  else if(count===op.input.length)count=0;
  if(reduce.matches){if(sweep){column=op.output.length-1;trace=multiplicationTrace(op,column);}count=op.input.length;paint();q('[data-mm-status]').textContent='Reduced motion: result shown immediately. Step or scrub to inspect terms.';return;}
  playing=true;root.classList.add('is-playing');q('[data-mm-play]').textContent='Pause';
  q('[data-mm-status]').textContent=sweep?'One full dot product per output column.':op.input.length>48?`${Math.ceil(op.input.length/48)} terms per animation step; every term is included. Scrub or Step +1 for a single term.`:'Follow each input × weight into the running sum.';
  paint();raf=win.requestAnimationFrame(tick);
 }
 function build({autoplay=false,focus=null}={}){
  stop();const data=getData(),oldScroll=q('.mm-patch-list')?.scrollTop||0,oldLeft=q('.mm-patch-list')?.scrollLeft||0;op=multiplicationOperation(data.s,data.patches,data.result,kind);column=Math.min(column,op.output.length-1);count=0;trace=multiplicationTrace(op,column);mode='dot';
  root.innerHTML=multiplicationMarkup(op,data.s,data.patches,data.imageURL);prepareMatrices();paint();
  const list=q('.mm-patch-list'),selected=q('[data-mm-patch][aria-pressed="true"]');list.scrollTop=oldScroll;list.scrollLeft=oldLeft;
  if(selected){const a=list.getBoundingClientRect(),b=selected.getBoundingClientRect();if(b.top<a.top)list.scrollTop-=a.top-b.top;else if(b.bottom>a.bottom)list.scrollTop+=b.bottom-a.bottom;if(b.left<a.left)list.scrollLeft-=a.left-b.left;else if(b.right>a.right)list.scrollLeft+=b.right-a.right;}
  if(focus)q(focus)?.focus({preventScroll:true});if(autoplay)play();
 }
 function close(){if(!root.open)return;stop();root.close();}
 function onClosed(){if(root.open)return;stop();root.innerHTML='';doc.documentElement.classList.remove('multiplication-open');(opener?.isConnected?opener:doc.querySelector(returnSelector))?.focus({preventScroll:true});onClose?.();}
 function open(operation,origin){
  if(disposed)return;kind=operation;opener=origin;
  const scope=origin?.closest('.stage-expansion')?'.stage.is-expanded .stage-expansion ':'#architecture ';
  returnSelector=origin?.id?`#${origin.id}`:origin?.hasAttribute('data-multiply')?`${scope}[data-multiply="${operation}"]${origin.hasAttribute('data-multiply-patch')?`[data-multiply-patch="${origin.dataset.multiplyPatch}"]`:':not([data-multiply-patch])'}`:'#stage-trigger-4';
  if(origin?.hasAttribute('data-multiply-patch'))onPatch(+origin.dataset.multiplyPatch);
  const data=getData();if(kind==='embedding'&&data.s.selected<0)onPatch(0);
  stop();column=0;count=0;root.innerHTML='<button type="button" data-mm-close aria-label="Close multiplication overlay">×</button>';doc.documentElement.classList.add('multiplication-open');if(!root.open)root.showModal();build({autoplay:true,focus:'[data-mm-close]'});
 }
 function click(e){const b=e.target.closest('button');
  if(!b){if(e.target===root){const r=root.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)close();}return;}
  if(b.hasAttribute('data-mm-result-row')){
   const row=+b.dataset.mmResultRow,j=e.target.closest('[data-mm-result-column]')?.dataset.mmResultColumn;
   selectResult(row,j===undefined?column:+j);return;
  }
  if(b.hasAttribute('data-mm-close'))return close();
  if(b.hasAttribute('data-mm-patch')){onPatch(+b.dataset.mmPatch);build({autoplay:false,focus:`[data-mm-patch="${b.dataset.mmPatch}"]`});return;}
  if(b.hasAttribute('data-mm-operation')){kind=b.dataset.mmOperation;column=0;build({autoplay:true,focus:`[data-mm-operation="${kind}"]`});return;}
  if(b.hasAttribute('data-mm-output')){setColumn(+b.dataset.mmOutput);q(`[data-mm-output="${column}"]`).focus({preventScroll:true});q('[data-mm-status]').textContent=`Column ${column} selected. Play, step, or scrub its dot product.`;return;}
  if(b.hasAttribute('data-mm-play')){if(playing)stop();else play();return;}
  if(b.hasAttribute('data-mm-sweep'))return play(true);
  if(b.hasAttribute('data-mm-step')){stop();count=Math.min(count+1,op.input.length);paint();return;}
  if(b.hasAttribute('data-mm-reset')){stop();mode='dot';count=0;paint();return;}
 }
 function selectResult(row,j){
  const next=Math.max(0,Math.min(op.outputs.length-1,row));stop();
  if(next!==op.row){onPatch(next-(op.hasCls?1:0));column=Math.max(0,Math.min(op.output.length-1,j));build({focus:`[data-mm-result-row="${next}"]`});}
  else{setColumn(j);q(`[data-mm-result-row="${next}"]`)?.focus({preventScroll:true});}
 }
 function keyboard(e){
  const b=e.target.closest('[data-mm-patch]'),data=getData();
  if(b){const next=multiplicationRowStep(+b.dataset.mmPatch,e.key,data.patches.length,kind!=='embedding'&&data.s.model==='vit');if(next!==null){e.preventDefault();onPatch(next);build({focus:`[data-mm-patch="${next}"]`});}return;}
  const resultRow=e.target.closest('[data-mm-result-row]');
  if(resultRow){
   if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const row=e.key==='Home'?0:e.key==='End'?op.outputs.length-1:op.row+(e.key==='ArrowDown'?1:e.key==='ArrowUp'?-1:0);selectResult(row,column+(e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0));}return;
  }
  const out=e.target.closest('[data-mm-output]');
  if(out){const next=multiplicationRowStep(+out.dataset.mmOutput,e.key,op.output.length,false);if(next!==null){e.preventDefault();setColumn(next);q(`[data-mm-output="${next}"]`).focus({preventScroll:true});}return;}
  if(e.target.id==='mm-weights'&&['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End'].includes(e.key)){
   e.preventDefault();const i=Math.max(0,count-1),row=e.key==='Home'?0:e.key==='End'?op.input.length-1:Math.max(0,Math.min(op.input.length-1,i+(e.key==='ArrowDown'?1:e.key==='ArrowUp'?-1:0)));
   setColumn(column+(e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0),{terms:row+1});
  }
 }
 root.addEventListener('click',click);root.addEventListener('keydown',keyboard);root.addEventListener('close',onClosed);
 root.addEventListener('cancel',()=>stop());
 root.addEventListener('input',e=>{if(e.target.hasAttribute('data-mm-progress')){stop();mode='dot';count=+e.target.value;paint();}});
 root.addEventListener('click',e=>{if(e.target.id==='mm-weights'){const r=e.target.getBoundingClientRect(),i=Math.min(op.input.length-1,Math.max(0,Math.floor((e.clientY-r.top)/r.height*op.input.length))),j=Math.min(op.output.length-1,Math.max(0,Math.floor((e.clientX-r.left)/r.width*op.output.length)));setColumn(j,{terms:i+1});}});
 const onResize=()=>{if(root.open)paint();},onVisibility=()=>{if(doc.hidden)stop();},onReduced=()=>{if(reduce.matches)stop();};win.addEventListener('resize',onResize);doc.addEventListener('visibilitychange',onVisibility);reduce.addEventListener('change',onReduced);
 return {open,close,get isOpen(){return root.open;},dispose(){close();disposed=true;stop();win.removeEventListener('resize',onResize);doc.removeEventListener('visibilitychange',onVisibility);reduce.removeEventListener('change',onReduced);root.remove();}};
}

/** Map visible patch thumbnails to their own embedding rows, never to another patch. */
export function patchConnectionPath(x1,y1,x2,y2){const bend=Math.max(8,(x2-x1)*.48);return `M${x1},${y1} C${x1+bend},${y1} ${x2-bend},${y2} ${x2},${y2}`;}
export function mountPatchConnections(root,getData){
 const doc=root.ownerDocument,win=doc.defaultView,svg=doc.createElementNS('http://www.w3.org/2000/svg','svg');svg.id='patch-embedding-wires';svg.setAttribute('aria-hidden','true');root.append(svg);let frame=0;
 function draw(){frame=0;if(root.classList.contains('has-expansion')){svg.replaceChildren();return;}
  const sequence=root.querySelector('#patch-sequence'),embedding=root.querySelector('#embedding-matrix');if(!sequence||!embedding)return;
  const origin=root.getBoundingClientRect(),seq=sequence.getBoundingClientRect(),dest=embedding.getBoundingClientRect(),{s,result}=getData();if(!result||dest.width===0)return;
  svg.setAttribute('viewBox',`0 0 ${origin.width} ${origin.height}`);
  const paths=[];sequence.querySelectorAll('[data-patch]').forEach(b=>{const r=b.getBoundingClientRect(),y=r.top+r.height/2;if(y<seq.top||y>seq.bottom)return;
   const patch=+b.dataset.patch,index=patch+(s.model==='vit'?1:0),x1=r.right-origin.left,y1=y-origin.top,x2=dest.left-origin.left,y2=dest.top-origin.top+(index+.5)*dest.height/result.input.length;
   paths.push(`<path data-connection-patch="${patch}" class="${patch===s.selected?'is-selected':''}" d="${patchConnectionPath(x1,y1,x2,y2)}"/>`);
  });svg.innerHTML=paths.join('');
 }
 function schedule(){win.cancelAnimationFrame(frame);frame=win.requestAnimationFrame(draw);}
 const resize=new win.ResizeObserver(schedule);resize.observe(root);root.querySelector('#patch-sequence').addEventListener('scroll',schedule,{passive:true});win.addEventListener('resize',schedule);
 return {schedule,dispose(){win.cancelAnimationFrame(frame);resize.disconnect();win.removeEventListener('resize',schedule);root.querySelector('#patch-sequence')?.removeEventListener('scroll',schedule);svg.remove();}};
}
