/** The prepared block-1 input: the exact tensor shown by the compact embedding canvas. */
import {outputColor} from './readout.mjs';
const embedNumber=(v,d=3)=>Number(v).toFixed(d);
export function embeddingRepresentation(s,result){
 const hasCls=s.model==='vit',index=s.selected+(hasCls?1:0),rows=result.input;
 return {rows,count:rows.length,width:rows[0].length,hasCls,index,selected:rows[index]};
}
export function embeddingMatrixMarkup(s,patches,result){
 const r=embeddingRepresentation(s,result);
 return `<div id="expanded-embedding-matrix" class="ro-output-matrix eb-matrix" role="group" aria-label="Input embedding matrix: ${r.count} tokens by ${r.width} features" aria-describedby="embedding-matrix-help" style="--output-cols:${r.width}">${r.rows.map((values,index)=>{
  const patch=index-(r.hasCls?1:0),label=patch<0?'[CLS]':`p${String(patch).padStart(2,'0')}`;
  const name=patch<0?'CLS token':`patch ${patch}, row ${patches[patch].row}, column ${patches[patch].col}`;
  return `<button type="button" class="ro-matrix-row eb-matrix-row" data-embedding-patch="${patch}" aria-label="Inspect input embedding for ${name}, ${r.width} features" aria-pressed="${index===r.index}" tabindex="${index===r.index?0:-1}"><span class="ro-matrix-row-label">${label}</span><span class="ro-matrix-cells" aria-hidden="true">${values.map((v,d)=>`<i data-embedding-value="${v}" style="background:${outputColor(v)}" title="${label} · feature ${d}: ${embedNumber(v,7)}"></i>`).join('')}</span></button>`;
 }).join('')}</div>`;
}
export function embeddingOverviewMarkup(s,patches,result){
 const r=embeddingRepresentation(s,result),cls=s.selected<0,name=cls?'[CLS]':`p${s.selected} (${patches[s.selected].row}, ${patches[s.selected].col})`;
 return `<section class="eb-overview" aria-label="Complete input embedding matrix">
 <div class="ro-output-banner"><strong>Every token has a row—before attention begins.</strong><p>The same ${r.count} × ${r.width} matrix as the overview, now enlarged. ${r.hasCls?`${patches.length} patch vectors plus [CLS], with learned position vectors already added.`:`${patches.length} projected patch vectors, already RMS-normalized. There is no [CLS] row.`} These are the inputs to block 1—not the final encoder output.</p></div>
 <div class="ro-output-layout eb-layout">
  <div class="ro-matrix-panel"><div class="ro-matrix-heading"><span>ALL INPUT EMBEDDINGS</span><strong>${r.count} × ${r.width}</strong></div>
   <div class="ro-matrix-axis"><span>tokens ↓</span><span>hidden features →</span></div>
   <div class="ro-matrix-columns" aria-hidden="true" style="--output-cols:${r.width}"><span></span><span>${Array.from({length:r.width},(_,j)=>`<i>${j}</i>`).join('')}</span></div>
   <div class="eb-matrix-scroll" data-signature="${s.model}:${s.patch}:${s.rows}:${s.cols}" data-selected="${s.selected}">${embeddingMatrixMarkup(s,patches,result)}</div>
   <div class="ro-matrix-legend"><span>≤ −2</span><i aria-hidden="true"></i><span>0</span><i aria-hidden="true"></i><span>≥ +2</span></div>
   <p id="embedding-matrix-help" class="ro-matrix-help">Click a row to inspect it. ↑ ↓ or Home / End to move.<br>Hover a cell for its value. Colors match the compact matrix exactly.</p>
  </div>
  <span class="ro-output-arrow" aria-hidden="true">→</span>
  <div class="ro-selected-output eb-selected"><div class="ro-section-label">ONE ROW · TOKEN ${r.index+1} OF ${r.count}</div>
   <div class="ro-vector"><div class="ro-vector-label">Block 1 input · ${name}<small>${r.width} values</small></div><div class="ro-vector-cells" data-embedding-vector="${r.selected.join(',')}">${r.selected.map((v,d)=>`<span style="background:${outputColor(v)}" title="Feature ${d}: ${embedNumber(v,7)}"><small>${d}</small>${embedNumber(v)}</span>`).join('')}</div></div>
   <p class="ro-copy"><strong>The outlined row and these values are the same vector.</strong> Choose a row here or a patch in the left sequence; the matrix, vector, and calculation stay in sync.</p>
   <div class="eb-preparation"><span>HOW THIS ROW IS FORMED</span><code>${cls?'learned [CLS] + position[0]':r.hasCls?'flatten(patch) × Wₑ + position':'RMSNorm(flatten(patch) × Wₑ)'}</code>
   <p>${cls?'[CLS] is a separate learned vector, not a projected image patch. Its position vector is added before block 1.':r.hasCls?'The projection gives 24 features. Then a learned position vector is added once, before block 1.':'RMSNorm follows the projection. 2D RoPE is applied later to Q and K inside attention—not added to these embeddings.'}</p></div>
   ${cls?'':`<button type="button" class="mm-launch eb-animate" data-multiply="embedding" aria-haspopup="dialog">▶ Animate this patch’s projection</button><p class="ro-copy ro-muted">The animation shows the linear projection before ${r.hasCls?'position addition':'RMSNorm'}; the matrix above includes that final preparation.</p>`}
  </div>
 </div>
 <p class="ro-copy ro-muted">The block slider does not change this input matrix. These values are always before block 1, computed from the untrained teaching model.</p></section>`;
}
