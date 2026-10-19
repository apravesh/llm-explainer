/** Encoder readout is real. Classifiers and autoregressive decoding are schematic. */
import {DIM,LLM_DIM} from './engine.mjs';
const rofmt=(x,d=3)=>Number(x).toFixed(d);
function readoutVector(label,values,extra='',colored=false){
 return `<div class="ro-vector"><div class="ro-vector-label">${label}<small>${values.length} values</small></div><div class="ro-vector-cells" ${extra}>${values.map((v,i)=>`<span ${colored?`style="background:${outputColor(v)}"`:""} title="Dimension ${i}: ${rofmt(v,7)}"><small>${i}</small>${rofmt(v)}</span>`).join('')}</div></div>`;
}
/** One shared activation palette for the compact matrix, expanded matrix and values. */
export function outputColor(v){
 const t=Math.min(Math.abs(v)/2,1),a=v>=0?[107,143,207]:[204,143,115];
 return `rgb(${a.map(x=>Math.round(245+(x-245)*t)).join(',')})`;
}
/** Every displayed cell is the same final encoder activation in both zoom states. */
export function outputMatrixMarkup(s,patches,result,{compact=false}={}){
 const r=finalRepresentation(s,result);
 const rows=result.final.map((values,index)=>{
  const patch=index-(r.hasCls?1:0),label=patch<0?'[CLS]':`p${String(patch).padStart(2,'0')}`;
  const name=patch<0?'CLS token':`patch ${patch}, row ${patches[patch].row}, column ${patches[patch].col}`;
  const cells=values.map((v,dimension)=>`<i data-output-value="${v}" style="background:${outputColor(v)}" title="${label} · feature ${dimension}: ${rofmt(v,7)}"></i>`).join('');
  if(compact)return `<span class="ro-compact-row ${index===r.index?'is-selected':''}">${cells}</span>`;
  return `<button type="button" class="ro-matrix-row" data-output-patch="${patch}" aria-label="Inspect final output for ${name}, ${r.width} features" aria-pressed="${index===r.index}" tabindex="${index===r.index?0:-1}"><span class="ro-matrix-row-label">${label}</span><span class="ro-matrix-cells" aria-hidden="true">${cells}</span></button>`;
 }).join('');
 return compact
  ? `<div class="ro-compact-matrix" style="--output-rows:${r.count};--output-cols:${r.width}" aria-hidden="true">${rows}</div>`
  : `<div id="output-matrix" class="ro-output-matrix" style="--output-cols:${r.width}" role="group" aria-label="Final encoder matrix: ${r.count} tokens by ${r.width} features" aria-describedby="output-matrix-help">${rows}</div>`;
}
export function finalRepresentation(s,result){
 const cls=s.model==='vit',index=cls?s.selected+1:s.selected;
 return {count:result.final.length,width:result.final[0].length,patchCount:result.final.length-(cls?1:0),hasCls:cls,index,selected:result.final[index]};
}
/** Image spans use actual projected patch vectors, while text slots are symbols.
 * The last row delimiter is IMAGE_END, not an additional IMAGE_BREAK. */
export function decoderPrefix(rows,cols,generated=0){
 if(!Number.isInteger(rows)||!Number.isInteger(cols)||rows<1||cols<1)throw Error('Positive patch-grid dimensions required');
 if(!Number.isInteger(generated)||generated<0||generated>2)throw Error('Use 0, 1, or 2 appended example tokens');
 const tokens=[{kind:'text',label:'T₀',description:'Illustrative text before the image'}];
 for(let r=0;r<rows;r++){
  for(let c=0;c<cols;c++)tokens.push({kind:'image',patch:r*cols+c,row:r,col:c,label:`p${r*cols+c}`});
  tokens.push({kind:r===rows-1?'end':'break',label:r===rows-1?'[IMAGE_END]':'[IMAGE_BREAK]'});
 }
 tokens.push({kind:'text',label:'T₁',description:'Illustrative text prompt after the image'});
 for(let i=1;i<=generated;i++)tokens.push({kind:'generated',label:i===1?'t₁':'t₂'});
 return tokens;
}
export function compactNextMarkup(model){
 return model==='vit'
 ? '<div class="next-compact"><strong>[CLS]</strong><span>one summary vector</span><i>↓</i><strong>Class scores</strong><small>classifier not run</small><div class="next-compact-rule"></div><span>Need text instead?</span><b>Patch vectors → LLM</b><em>Explore both paths +</em></div>'
 : '<div class="next-compact"><strong>All patch vectors</strong><i>↓</i><strong>Project to LLM width</strong><span>+ text embeddings</span><i>↓</i><strong>Causal text decoder</strong><span>next text token</span><small>decoder schematic only</small><em>Follow the handoff +</em></div>';
}
export function outputReadoutMarkup(s,patches,result){
 const r=finalRepresentation(s,result),name=s.selected<0?'[CLS]':`p${s.selected} (${patches[s.selected].row}, ${patches[s.selected].col})`;
 return `<section class="ro-output" aria-label="Final vision encoder output"><div class="ro-output-banner"><strong>The encoder outputs vectors—not class labels or text.</strong><p>After all ${result.layers.length} teaching blocks${r.hasCls?' and final LayerNorm':''}: <b>${r.count} tokens × ${r.width} features</b>. ${r.hasCls?`That is ${r.patchCount} patch vectors plus one [CLS] vector.`:`That is one vector for each of the ${r.patchCount} patches, with no [CLS] token.`}</p></div>
 <div class="ro-output-layout">
  <div class="ro-matrix-panel"><div class="ro-matrix-heading"><span>ALL OUTPUT TOKENS</span><strong>${r.count} × ${r.width}</strong></div>
   <div class="ro-matrix-axis"><span>tokens ↓</span><span>hidden features →</span></div>
   <div class="ro-matrix-columns" aria-hidden="true" style="--output-cols:${r.width}"><span></span><span>${Array.from({length:r.width},(_,j)=>`<i>${j}</i>`).join('')}</span></div>
   <div class="ro-matrix-scroll" data-signature="${s.model}:${s.patch}:${s.rows}:${s.cols}" data-selected="${s.selected}">${outputMatrixMarkup(s,patches,result)}</div>
   <div class="ro-matrix-legend"><span>≤ −2</span><i aria-hidden="true"></i><span>0</span><i aria-hidden="true"></i><span>≥ +2</span></div>
   <p id="output-matrix-help" class="ro-matrix-help">Click a row to inspect it. ↑ ↓ or Home / End to move.<br>Hover a cell for its value. Colors show signed activations, not attention probabilities.</p>
  </div>
  <span class="ro-output-arrow" aria-hidden="true">→</span>
  <div class="ro-selected-output"><div class="ro-section-label">ONE ROW · TOKEN ${r.index+1} OF ${r.count}</div>
   ${readoutVector(`Final encoder state · ${name}`,r.selected,`data-final-vector="${r.selected.join(',')}"`,true)}
   <p class="ro-copy"><strong>One token, all ${r.width} features.</strong> The outlined row and these numbers are the same vector, with the same colors. Select a row here or a patch in the left sequence; both views stay in sync.</p>
   <p class="ro-copy">Each row belongs to one token but can contain information from the rest of the image. Selecting a patch changes the row you inspect—not how many rows the encoder returns.</p>
   <p class="ro-copy ro-muted">These activations come from untrained weights, so they do not imply learned visual understanding.</p>
  </div>
 </div>
 <div class="ro-boundary"><div><span>VISION ENCODER ENDS HERE</span><strong>${r.count} × ${DIM} representation</strong></div><b>→</b><button type="button" class="ro-next-link" data-expand-step="6">${r.hasCls?'[CLS], classification, or a text decoder?':'Project these vectors into a text decoder'} <span>What happens next →</span></button></div>
 <p class="ro-copy ro-muted">The inspected block slider does not change this final output: it is always after all ${result.layers.length} blocks. Full-size encoders use larger hidden widths; this demo uses ${DIM}.</p></section>`;
}
function clsExplanation(s,result){
 const steps=[['Learned starting vector',result.cls],['+ position → block 1 input',result.input[0]],...result.layers.map((l,i)=>[`After block ${i+1}`,l.output[0]]),['Final LayerNorm → classifier input',result.final[0]]];
 const chosen=Math.min(Math.max(0,s.clsStep||0),steps.length-1);
 return `<section class="ro-cls" aria-label="How the CLS token works"><div class="ro-section-label">CLASSIFICATION PATH · CLASSIC ViT</div><h4>What is [CLS]?</h4><p class="ro-copy"><strong>[CLS] means “classification token.”</strong> It is a learned vector prepended <em>before</em> the image patches—not an image crop, not a predicted label, and not a text caption. It participates in every bidirectional attention block, so patches can update its hidden state.</p>
 <div class="ro-cls-strip"><span class="ro-cls-token">[CLS]</span><span>p₀</span><span>p₁</span><span>…</span><span>pₙ₋₁</span><b>→ all encoder blocks →</b><span class="ro-cls-token">final [CLS]</span></div>
 <div class="ro-cls-controls"><label for="cls-step">Follow its hidden state</label><select id="cls-step" data-inspect="cls-step">${steps.map(([label],i)=>`<option value="${i}" ${i===chosen?'selected':''}>${label}</option>`).join('')}</select></div>
 ${readoutVector(steps[chosen][0],steps[chosen][1],`data-cls-vector="${steps[chosen][1].join(',')}"`)}
 <div class="ro-classifier"><span>final [CLS] · 1 × ${DIM}</span><b>→</b><span>classifier head</span><b>→</b><span>class logits → probabilities</span><small>Architecture only · classifier not run</small></div>
 <p class="ro-copy">Classification training makes this token useful as a whole-image summary. It is not an average by definition, and it is not guaranteed to be a meaningful summary in this untrained demo. The other patch states still exist; the classification head simply reads [CLS].</p>
 <div class="ro-switch-path"><strong>This is not the Pixtral text-generation path.</strong><p>A vision-language model typically keeps multiple patch features rather than relying on one [CLS] vector. Pixtral has no [CLS] token.</p><button type="button" data-next-model="pixtral">See patch vectors → text decoder →</button></div></section>`;
}
function decoderExplanation(s,patches,result){
 const generated=Math.min(Math.max(s.decoderStep||0,0),2),prefix=decoderPrefix(s.rows,s.cols,generated),query=prefix.length-1,nextLabel=['t₁','t₂','t₃'][generated];
 const selected=Math.max(0,s.selected);
 return `<section class="ro-decoder" aria-label="How image vectors enter a text decoder"><div class="ro-section-label">VISION-LANGUAGE PATH · PIXTRAL</div><h4>Keep the patches. Match the decoder’s width.</h4>
 <p class="ro-copy">Pixtral keeps <strong>all ${patches.length} patch vectors</strong>. The same two-layer projector maps each from the encoder width to the language model’s input width. It changes the number of features, not the number of image patches.</p>
 <div class="ro-project-flow"><div><small>ENCODER OUTPUT</small><strong>${patches.length} × ${DIM}</strong><span>continuous patch states</span></div><b>→</b><div><small>PROJECTOR · COMPUTED</small><strong>${DIM} → ${LLM_DIM} → ${LLM_DIM}</strong><span>Linear → GELU → Linear</span></div><b>→</b><div><small>DECODER-READY IMAGE FEATURES</small><strong>${patches.length} × ${LLM_DIM}</strong><span>same patches, new width</span></div></div>
 ${readoutVector(`Projected image feature · p${selected}`,result.projected[selected],`data-projector-vector="${result.projected[selected].join(',')}"`)}
 <button type="button" class="ro-math-button" data-multiply="projector-first" aria-haspopup="dialog">▶ Animate the projector multiplication</button>
 <button type="button" class="ro-math-button" data-projector-detail aria-expanded="${!!s.projectorDetail}" aria-controls="projector-detail">${s.projectorDetail?'− Hide':'+ Inspect'} the projector multiplication</button>
 <p class="ro-copy ro-muted">Teaching dimensions shown above. Pixtral 12B uses encoder width 1,024 and decoder width 5,120; its projector is 1,024 → 5,120 → 5,120.</p>
 <div class="ro-section-label">ONE JOINT SEQUENCE · IMAGE VECTORS + TEXT EMBEDDINGS</div>
 <p class="ro-copy">Text tokens use the decoder’s embedding lookup. Image slots instead receive the projected continuous image vectors. Both have the same width. Row-break and image-end tokens mark the image’s structure; they are separate from the patch features. <strong>The encoder does not turn the image into words first.</strong></p>
 <div class="ro-stream-caption"><span>ILLUSTRATIVE PREFIX · NOT ACTUAL TEXT TOKENIZATION</span><span>${prefix.filter(t=>t.kind==='image').length} image slots · ${s.rows-1} row breaks · 1 image end</span></div>
 <div class="ro-joint-stream" role="group" aria-label="Decoder prefix with image and text slots">${prefix.map((t,i)=>t.kind==='image'?`<button type="button" class="ro-stream-token ro-image-token ${t.patch===selected?'is-selected':''}" data-readout-patch="${t.patch}" aria-label="Inspect projected patch ${t.patch}" aria-pressed="${t.patch===selected}" title="Continuous ${LLM_DIM}-dimensional vector for patch (${t.row}, ${t.col})">${t.label}</button>`:`<span class="ro-stream-token ro-${t.kind}-token ${i===query?'is-query':''}" title="${t.description||t.label}">${t.label}${i===query?'<small>last input → predicts next</small>':''}</span>`).join('')}</div>
 <div class="ro-legend"><span><i class="ro-legend-image"></i>projected image features</span><span><i class="ro-legend-text"></i>text embeddings / special tokens</span><span><i class="ro-legend-generated"></i>appended text token</span></div>
 <div class="ro-generation"><div class="ro-decoder-box"><small>FROM HERE: SCHEMATIC · NOT RUN</small><strong>Causal text decoder</strong><span>1D positions over the joint sequence</span><span>last position attends to earlier slots + itself</span></div><b>→</b><div><strong>Vocabulary logits</strong><span>one score per text-token ID</span><span>softmax → choose a token</span></div><b>→</b><div class="ro-next-token"><small>NEXT TEXT TOKEN</small><strong>${nextLabel}</strong><span>symbolic example only</span></div></div>
 <div class="ro-generation-steps" role="group" aria-label="Illustrative text generation steps">${['1 · Build the prefix','2 · Append t₁','3 · Append t₂'].map((label,i)=>`<button type="button" data-decoder-step="${i}" aria-pressed="${i===generated}">${label}</button>`).join('')}</div>
 <p class="ro-generation-status" aria-live="polite">${generated===0?'The last prompt token T₁ predicts the first answer token t₁.':`The chosen token t${generated===1?'₁':'₂'} is embedded and appended to the prefix; that last position now predicts ${nextLabel}.`} Repeat to generate text. Tokens t₁/t₂/t₃ are placeholders, not model predictions.</p>
 <p class="ro-copy">This differs from an encoder–decoder design with a separate cross-attention module: <strong>Pixtral inserts image features into the causal decoder’s own input sequence</strong>. Image patches were already contextualized by bidirectional vision attention before the decoder sees them. The language decoder then produces text autoregressively.</p>
 <div class="ro-switch-path"><strong>No [CLS] bottleneck here.</strong><p>Each projected patch occupies its own decoder input slot. A single summary token would discard that per-patch interface.</p><button type="button" data-next-model="vit">Compare with the [CLS] classification path →</button></div></section>`;
}
export function nextStageMarkup(s,patches,result){
 const isPix=s.model==='pixtral';
 return `<div class="ro-next-stage"><div class="ro-two-paths" role="group" aria-label="Choose the encoder readout path"><button type="button" class="ro-path ${isPix?'':'is-active'}" data-next-model="vit" aria-pressed="${!isPix}" aria-controls="readout-path-content"><span>CLASSIFICATION ViT</span><strong>final [CLS] → class scores</strong><small>A summary readout, not a text decoder</small><span class="ro-path-cue">${isPix?'Explore this path →':'Currently viewing'}</span></button><button type="button" class="ro-path ${isPix?'is-active':''}" data-next-model="pixtral" aria-pressed="${isPix}" aria-controls="readout-path-content"><span>PIXTRAL</span><strong>all patches → projector → text decoder</strong><small>Many image slots, then next-token prediction</small><span class="ro-path-cue">${isPix?'Currently viewing':'Explore this path →'}</span></button></div><div id="readout-path-content">${isPix?decoderExplanation(s,patches,result):clsExplanation(s,result)}</div>
 <div class="ro-sources"><span>Architecture references</span><a href="https://arxiv.org/abs/2010.11929" target="_blank" rel="noopener">ViT paper ↗</a><a href="https://arxiv.org/abs/2410.07073" target="_blank" rel="noopener">Pixtral paper ↗</a></div></div>`;
}
