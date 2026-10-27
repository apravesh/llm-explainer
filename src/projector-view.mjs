/** Real teaching-model tensors; production widths are explicitly separate. */
import {DIM,LLM_DIM} from './engine.mjs';
import {outputColor} from './readout.mjs';
const pvNumber=(v,d=5)=>Number(v).toFixed(d);
export const PIXTRAL_WIDTHS=Object.freeze({patch:16,encoder:1024,decoder:5120,layers:24});
export function projectorTensors(result){
 return [
  {key:'input',label:'Encoder output H',values:result.final},
  {key:'pre',label:'First linear Z',values:result.adapterPre},
  {key:'hidden',label:'GELU(Z)',values:result.adapterHidden},
  {key:'output',label:'Projected image features Y',values:result.projected}
 ];
}
function projectorMatrix(tensor,selected,{compact=false}={}){
 const {key,label,values}=tensor,rows=values.length,cols=values[0].length;
 const cells=(row,r)=>row.map((v,j)=>`<i data-projector-cell="${key}:${r}:${j}" data-projector-value="${v}" style="background:${outputColor(v)}" title="${label} · p${r}, feature ${j}: ${pvNumber(v,7)}"></i>`).join('');
 return `<div class="pv-matrix ${compact?'pv-compact-matrix':''}" ${compact?'aria-hidden="true"':`role="group" aria-label="${label}: ${rows} patches by ${cols} features"`} data-projector-matrix="${key}" style="--pv-rows:${rows};--pv-cols:${cols}">${values.map((row,r)=>compact?`<span class="pv-row ${r===selected?'is-selected':''}">${cells(row,r)}</span>`:`<button type="button" class="pv-row" data-projector-patch="${r}" data-projector-source="${key}" aria-label="Inspect patch ${r} in ${label}" aria-pressed="${r===selected}" tabindex="${r===selected?0:-1}">${cells(row,r)}</button>`).join('')}</div>`;
}
function projectorWeights(weights,operation,label){
 return `<button type="button" class="pv-weights" data-multiply="${operation}" aria-haspopup="dialog" aria-label="Animate ${label} projector multiplication"><strong>${label}</strong><span class="pv-weight-grid" style="--pv-weight-cols:${weights[0].length}" aria-hidden="true">${weights.map(row=>row.map(v=>`<i style="background:${outputColor(v*Math.sqrt(weights.length))}"></i>`).join('')).join('')}</span><small>${weights.length} × ${weights[0].length}</small><span class="pv-weight-cta">▶ Multiply</span></button>`;
}
export function compactProjectorMarkup(s,result){
 return `<div class="pv-compact"><strong>Projector</strong><div class="pv-compact-flow">${projectorMatrix({key:'input',label:'Encoder output',values:result.final},s.selected,{compact:true})}<span>→</span>${projectorMatrix({key:'output',label:'Projected features',values:result.projected},s.selected,{compact:true})}</div><div class="pv-compact-shapes"><span>${result.final.length} × ${DIM}</span><span>${result.projected.length} × ${LLM_DIM}</span></div><small>Linear → GELU → Linear</small><span>→ text decoder</span><em>Inspect projector +</em></div>`;
}
export function projectorFlowMarkup(s,patches,result){
 const tensors=projectorTensors(result),n=patches.length;
 const matrix=t=>`<div class="pv-tensor"><div class="pv-label"><strong>${t.label}</strong><span>${n} × ${t.values[0].length}</span></div>${projectorMatrix(t,s.selected)}<small>patches ↓ · features →</small></div>`;
 return `<section class="pv-flow" aria-label="Two-layer projector matrix computation">
 <div class="pv-scale"><strong>Teaching model—not Pixtral-12B’s actual width</strong><div><span>Here: <b>${DIM} → ${LLM_DIM} → ${LLM_DIM}</b></span><span>Pixtral-12B: <b>1,024 → 5,120 → 5,120</b></span></div><p>Pixtral-12B uses 1,024 features per patch and 24 vision layers. Our 24-feature untrained model is deliberately smaller. Matrices below contain actual teaching-model values; they are not slices of pretrained Pixtral weights.</p><a href="https://huggingface.co/mistralai/Pixtral-12B-2409/blob/main/params.json" target="_blank" rel="noopener">Official model configuration ↗</a></div>
 <div class="pv-intro"><strong>Same patches. A wider representation.</strong><span>Select any row · p${String(s.selected).padStart(2,'0')} is highlighted in every matrix</span></div>
 <div class="pv-equation" aria-label="H times W1 equals Z; apply GELU; times W2 equals Y"><code>H · W₁ = Z</code><span>→ GELU →</span><code>A · W₂ = Y</code><small>Bias omitted in this teaching model</small></div>
 <div class="pv-scroll"><div class="pv-workbench">
 ${matrix(tensors[0])}<div class="pv-operator"><span>×</span>${projectorWeights(result.adapterWeights[0],'projector-first','W₁')}<span>=</span></div>
 ${matrix(tensors[1])}<div class="pv-activation"><span>→</span><strong>GELU</strong><small>elementwise</small><span>→</span></div>
 ${matrix(tensors[2])}<div class="pv-operator"><span>×</span>${projectorWeights(result.adapterWeights[1],'projector-second','W₂')}<span>=</span></div>
 ${matrix(tensors[3])}
 </div></div>
 <div class="pv-footnote"><span>All ${n} patches remain present: neither linear layer nor GELU merges tokens.</span><span>Coral − / blue + · hover for exact values · each weight matrix is shared across patches.</span></div>
 <div class="pv-selected-trace"><strong>p${s.selected} · feature 0 across the projector</strong><code>H: ${pvNumber(result.final[s.selected][0])} → Z: ${pvNumber(result.adapterPre[s.selected][0])} → GELU(Z): ${pvNumber(result.adapterHidden[s.selected][0])} → Y: ${pvNumber(result.projected[s.selected][0])}</code><small>Each linear output mixes all input features; this trace compares feature 0, not scalar-only multiplication.</small></div>
 </section>`;
}
