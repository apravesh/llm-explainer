import {activationRGB} from './full-matrix.mjs';
/** A separate, deterministic teaching MLP. No encoder state or weights are changed. */
export const SWIGLU_LAB_INPUTS=Object.freeze([
 Object.freeze([1,-.5,.75,.2]),Object.freeze([-.6,1,.1,-.8]),Object.freeze([.2,.3,-1,1.2])
]);
export function swigluLabWeights(){
 const weights=(rows,cols,offset)=>Array.from({length:rows},(_,r)=>Array.from({length:cols},(_,c)=>Math.sin((r*cols+c+offset)*1.37)*.7));
 return {w1:weights(4,6,1),w3:weights(4,6,31),w2:weights(6,4,71)};
}
export function swigluLabTrace(input=SWIGLU_LAB_INPUTS[0]){
 if(input.length!==4||!input.every(Number.isFinite))throw Error('SwiGLU toy input must contain four finite features.');
 const x=[...input],weights=swigluLabWeights(),linear=(v,w)=>w[0].map((_,j)=>v.reduce((sum,a,i)=>sum+a*w[i][j],0));
 const pre=linear(x,weights.w1),gate=pre.map(a=>a/(1+Math.exp(-a))),values=linear(x,weights.w3),mixed=gate.map((a,i)=>a*values[i]),output=linear(mixed,weights.w2);
 return {x,weights,pre,gate,values,mixed,output};
}
export const SWIGLU_TRACE_MS=2800;
export function swigluTraceStage(elapsed){return Math.min(4,Math.floor(Math.max(0,elapsed)/700));}
const sgNumber=x=>Math.abs(x)<.000005?'0.00000':x.toFixed(5);
function sgVector(vector,x,y,width=112){
 const pitch=width/vector.length;
 return vector.map((value,i)=>`<rect x="${x+i*pitch}" y="${y}" width="${pitch-2}" height="18" rx="2" fill="rgb(${activationRGB(value).join(',')})"/>`).join('');
}
export function swigluDiagramMarkup(trace,compact=false){
 const text=(copy,x,y,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${copy}</text>`;
 const box=compact?{input:[220,5,140],gate:[32,145,185],values:[363,145,185],mix:[201,284,177],down:[188,420,86],output:[306,407,103]}:{input:[10,94,140],gate:[217,14,185],values:[217,169,185],mix:[513,94,177],down:[720,109,86],output:[827,94,103]};
 const card=(id,title,caption,vector,dimension)=>{const [x,y,w]=box[id],down=id==='down';return `<g data-sg-node="${id}"><rect class="sg-node-bg" x="${x}" y="${y}" width="${w}" height="${down?74:103}" rx="9"/>${text(title,x+14,y+24,'sg-title')}${caption?text(caption,x+14,y+41,'sg-caption'):''}${vector?sgVector(vector,x+14,y+55,w-28):''}${text(dimension,x+14,y+(down?60:91),'sg-dimension')}</g>`;};
 const marker=compact?'sg-arrow-compact':'sg-arrow';
 const edges=compact?[
  ['branches','M260 108V118H124V140'],['branches','M320 108V118H455V140'],
  ['mix','M124 249V262H248V279'],['mix','M455 249V262H332V279'],
  ['output','M243 387V415'],['output','M274 457H301']
 ]:[
  ['branches','M150 146H169Q181 146 181 134V76Q181 66 195 66H212'],
  ['branches','M150 146H169Q181 146 181 158V211Q181 221 195 221H212'],
  ['mix','M402 66H460Q480 66 480 86V123Q480 133 491 133H508'],
  ['mix','M402 221H460Q480 221 480 201V169Q480 159 491 159H508'],
  ['output','M690 146H715'],['output','M806 146H822']
 ];
 return `<svg class="sg-diagram ${compact?'sg-compact-diagram':'sg-wide-diagram'}" viewBox="0 0 ${compact?'580 526':'940 294'}" role="img" aria-label="SwiGLU for one toy patch. The same four-feature input enters two parallel four-by-six projections: W1 followed by SiLU, and W3 without activation. Multiply their six matching values, then use W2 to contract six features back to four. All values are already computed; animation traces their dependencies.">
 <defs><marker id="${marker}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#a6b8d3"/></marker></defs>
 <g class="sg-links" marker-end="url(#${marker})">${edges.map(([id,d])=>`<path data-sg-edge="${id}" d="${d}"/>`).join('')}</g>
 ${card('input','Input x','toy patch features',trace.x,'1 × 4')}
 ${card('gate','W₁ → SiLU','Gate branch · 4 × 6',trace.gate,'g = SiLU(xW₁) · 1 × 6')}
 ${card('values','W₃','Parallel expansion · 4 × 6',trace.values,'v = xW₃ · 1 × 6')}
 ${text(compact?'same x → both branches':'Parallel branches, not W₁ → W₃',compact?222:226,compact?134:148,'sg-parallel')}
 ${card('mix','g ⊙ v','multiply matching features',trace.mixed,'1 × 6 · no patch mixing')}
 ${card('down','W₂','contract',null,'6 × 4')}
 ${card('output','Output','',trace.output,'1 × 4')}
 </svg>`;
}
export function swigluValuesMarkup(trace){
 return `<div class="sg-input-values"><span>Input x: <code>${trace.x.map(sgNumber).join(', ')}</code></span><span>Output: <code>${trace.output.map(sgNumber).join(', ')}</code></span></div>
 <div class="sg-table-scroll"><table><caption>One row per expanded feature. Every product is included in W₂’s contraction.</caption><thead><tr><th>Feature</th><th>xW₁</th><th>g = SiLU(xW₁)</th><th>v = xW₃</th><th>g ⊙ v</th></tr></thead><tbody>${trace.pre.map((a,j)=>`<tr><th scope="row">${j}</th><td>${sgNumber(a)}</td><td>${sgNumber(trace.gate[j])}</td><td>${sgNumber(trace.values[j])}</td><td data-sg-product="${j}">${sgNumber(trace.mixed[j])}</td></tr>`).join('')}</tbody></table></div><p class="sg-exact-note">Numbers are rounded for display. Both projections use all four input features; each output sums all six products. Toy weights are fixed, synthetic and untrained.</p>`;
}
export function swigluLabMarkup(){return `
 <div class="sg-intuition"><div><h3>The MLP you know</h3><p>GPT-2 expands once, applies GELU, then contracts.</p><code>GELU(xW₁)W₂</code></div><div><h3>SwiGLU adds a parallel branch</h3><p>Expand the same input twice. Apply SiLU to one branch, multiply matching features, then contract.</p><code>[SiLU(xW₁) ⊙ (xW₃)]W₂</code></div></div>
 <div class="sg-workbench"><div class="sg-toolbar"><div><strong>Follow one patch</strong><span>Isolated toy MLP · 4 → 6 → 4</span></div><div class="sg-presets" role="group" aria-label="SwiGLU toy input">${['A','B','C'].map((name,i)=>`<button type="button" data-sg-preset="${i}" aria-pressed="${i===0}" aria-label="Use toy patch ${name}">Patch ${name}</button>`).join('')}</div><button type="button" data-sg-replay>↻ Trace the branches</button></div>
 <div class="sg-diagram-scroll" role="region" aria-label="SwiGLU branch diagram" tabindex="0"><div data-sg-diagram></div></div>
 <p class="sg-progress" data-sg-status aria-live="off">Both branches read the same input. Click “Trace the branches” to follow the calculation.</p>
 <details class="fa-values-disclosure sg-values"><summary>Show actual values</summary><div data-sg-values></div></details></div>
 <div class="sg-takeaways"><p><b>SiLU makes the gate nonlinear.</b> SiLU(z) = z · sigmoid(z), also called Swish when its scale parameter is 1. SwiGLU means Swish-Gated Linear Unit.</p><p><b>⊙ is not another matrix multiplication.</b> Feature 0 multiplies feature 0, feature 1 multiplies feature 1, and so on. These gates are not probabilities or binary switches: they can amplify, suppress, or change a feature’s sign.</p><p><b>Same weights, independent patches.</b> Every patch reuses W₁, W₃ and W₂. Attention mixes patches; this MLP transforms each patch’s features. The output above is the MLP update, before adding the residual.</p></div>
 <div class="sg-footer"><span>The toy is not a slice of the model. Pixtral’s encoder uses two <b>1,024 → 4,096</b> branches and a <b>4,096 → 1,024</b> contraction. Row-vector notation; biases omitted.</span><a href="https://arxiv.org/abs/2002.05202" target="_blank" rel="noopener noreferrer">SwiGLU paper ↗</a><a href="https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/transformer_layers.py" target="_blank" rel="noopener noreferrer">Reference implementation ↗</a></div>`;}
export function mountSwigluLab(root){
 const doc=root.ownerDocument,win=doc.defaultView,reduced=win.matchMedia('(prefers-reduced-motion: reduce)');
 let preset=0,frame=null,last=0,elapsed=0,playing=false,shownStage=-1,visible=false,started=false,disposed=false;
 root.innerHTML=swigluLabMarkup();const q=selector=>root.querySelector(selector);
 const messages=['Read x: toy features representing a normalized patch input.','Both projections run in parallel on the same x. Apply SiLU only to W₁’s result.','Multiply the two six-feature vectors element by element. No summing and no mixing between patches here.','W₂ sums across the six gated features to produce a four-feature output.','Complete: expand twice → activate one branch → multiply → contract.'];
 function phase(stage){if(disposed||stage===shownStage)return;shownStage=stage;root.dataset.stage=stage;q('[data-sg-status]').textContent=messages[stage];
  root.querySelectorAll('[data-sg-node]').forEach(node=>{const active=stage===0?['input']:stage===1?['gate','values']:stage===2?['mix']:stage===3?['down','output']:[];node.classList.toggle('is-active',active.includes(node.dataset.sgNode));});
  root.querySelectorAll('[data-sg-edge]').forEach(edge=>edge.classList.toggle('is-active',edge.dataset.sgEdge===({1:'branches',2:'mix',3:'output'}[stage])));
 }
 function pause(){playing=false;if(frame!==null)win.cancelAnimationFrame(frame);frame=null;root.dataset.playing='false';}
 function tick(now){frame=null;if(disposed||!playing)return;if(!root.isConnected){dispose();return;}if(doc.hidden||!visible){pause();return;}elapsed+=Math.max(0,now-last);last=now;phase(swigluTraceStage(elapsed));if(elapsed>=SWIGLU_TRACE_MS){pause();return;}frame=win.requestAnimationFrame(tick);}
 function resume(){if(disposed||playing||doc.hidden||!visible||!root.isConnected)return;if(reduced.matches){elapsed=SWIGLU_TRACE_MS;phase(4);return;}if(elapsed>=SWIGLU_TRACE_MS)return;playing=true;root.dataset.playing='true';last=win.performance.now();frame=win.requestAnimationFrame(tick);}
 function replay(){pause();started=true;elapsed=0;shownStage=-1;phase(0);resume();}
 function render(){const trace=swigluLabTrace(SWIGLU_LAB_INPUTS[preset]);q('[data-sg-diagram]').innerHTML=swigluDiagramMarkup(trace)+swigluDiagramMarkup(trace,true);q('[data-sg-values]').innerHTML=swigluValuesMarkup(trace);root.dataset.preset=preset;root.querySelectorAll('[data-sg-preset]').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.sgPreset===preset));shownStage=-1;phase(4);}
 function click(e){const b=e.target.closest('button');if(!b||!root.contains(b))return;if(b.hasAttribute('data-sg-preset')){pause();preset=+b.dataset.sgPreset;render();replay();}else if(b.hasAttribute('data-sg-replay'))replay();}
 const observer=typeof win.IntersectionObserver==='function'?new win.IntersectionObserver(entries=>{if(disposed)return;const e=entries.at(-1);visible=e.isIntersecting&&e.intersectionRatio>=.2;if(!visible)pause();else if(!started)replay();else resume();},{threshold:[0,.2]}):null;
 const visibility=()=>{if(doc.hidden)pause();else resume();},preference=()=>{if(reduced.matches){pause();elapsed=SWIGLU_TRACE_MS;phase(4);}};
 function dispose(){if(disposed)return;pause();disposed=true;observer?.disconnect();root.removeEventListener('click',click);doc.removeEventListener('visibilitychange',visibility);reduced.removeEventListener('change',preference);}
 root.addEventListener('click',click);doc.addEventListener('visibilitychange',visibility);reduced.addEventListener('change',preference);render();if(observer)observer.observe(q('.sg-workbench'));else{visible=true;replay();}
 return {dispose};
}
