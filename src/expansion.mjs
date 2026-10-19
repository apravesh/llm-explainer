import {embeddingOverviewMarkup} from './embedding-view.mjs';
import {mlpLanesMarkup} from './multiplication.mjs';
import {outputReadoutMarkup,nextStageMarkup} from './readout.mjs';
import { multiHeadMarkup, multiHeadView } from './multi-head.mjs';
import { attentionFlowMarkup, attentionScoreDetail } from './attention-flow.mjs';
/** In-place architecture inspection. Every number comes from the live engine. */
import { dot, DIM, HEAD_DIM } from './engine.mjs';
const nf=(n,d=3)=>Number(n).toFixed(d);
const safe=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const tint=(v,scale=1)=>{const t=Math.min(Math.abs(v)/scale,1),a=v>=0?[115,151,217]:[214,155,127];return `rgb(${a.map(x=>Math.round(249+(x-249)*t)).join(',')})`;};
export const STAGE_NAMES=['Input image','Patches','Embedding','Self-attention','MLP','Output','What next?'];
const names=STAGE_NAMES;

export function projectionTrace(input,weights,dimension){
 if(!input.length||weights.length!==input.length||!Number.isInteger(dimension)||dimension<0||dimension>=weights[0].length)throw Error('Invalid projection trace dimensions');
 const terms=input.map((x,i)=>x*weights[i][dimension]);
 return {terms,sum:terms.reduce((a,b)=>a+b,0),inputCount:input.length,outputCount:weights[0].length};
}
export function scalarAddress(index,patchSize){return {row:Math.floor(index/(patchSize*3)),col:Math.floor(index/3)%patchSize,channel:index%3};}
export function visibleRows(length,selected,count=9){const start=Math.min(Math.max(0,Math.floor(selected/3)*3-3),Math.max(0,length-count));return Array.from({length:Math.min(count,length)},(_,i)=>start+i);}

/** Every patch stays selectable while a stage is expanded. Thumbnails retain identity,
 * not a claim that deeper hidden states are still raw pixels. */
export function patchRailMarkup(s,patches,imageURL){
 const hasCls=s.model==='vit'&&s.expanded>=2;
 const item=(index,label,position,thumb)=>`<button type="button" class="token-rail-item" data-rail-patch="${index}" aria-label="Inspect ${index<0?'CLS token':`patch ${index}, ${position}`}" aria-pressed="${index===s.selected}" tabindex="${index===s.selected?0:-1}">${thumb}<span class="token-rail-label"><strong>${label}</strong><small>${position}</small></span><span class="token-rail-pointer" aria-hidden="true">→</span></button>`;
 const cls=hasCls?item(-1,'[CLS]','special token','<span class="token-rail-cls" aria-hidden="true">CLS</span>'):'';
 const items=patches.map((p,i)=>{
  const x=s.cols>1?p.col/(s.cols-1)*100:0,y=s.rows>1?p.row/(s.rows-1)*100:0;
  const style=`background-image:url('${imageURL}');background-size:${s.cols*100}% ${s.rows*100}%;background-position:${x}% ${y}%;`;
  return item(i,`p${String(i).padStart(2,'0')}`,`(${p.row}, ${p.col})`,`<span class="token-rail-thumb" style="${safe(style)}" aria-hidden="true"></span>`);
 }).join('');
 return `<aside class="token-rail" aria-label="Patch sequence"><div class="token-rail-heading"><span>PATCH SEQUENCE</span><strong>${patches.length} patches${hasCls?' + CLS':''}</strong></div><div class="token-rail-list" role="group" aria-label="Choose a patch to inspect" data-selected="${s.selected}" data-signature="${s.model}:${s.patch}:${s.rows}:${s.cols}:${hasCls}">${cls}${items}</div><p class="token-rail-help">All patches · scroll to explore<br> Arrow keys to select</p></aside>`;
}
export function nextRailPatch(current,key,count,hasCls){
 const first=hasCls?-1:0,last=count-1;
 if(key==='Home')return first;if(key==='End')return last;
 const delta={ArrowDown:1,ArrowRight:1,ArrowUp:-1,ArrowLeft:-1}[key];
 return delta===undefined?null:Math.max(first,Math.min(last,current+delta));
}
/** Minimal movement to reveal the selected item, without scrolling the page. */
export function railScrollOffset(offset,extent,start,end){
 if(start<offset)return Math.max(0,start);
 if(end>offset+extent)return Math.max(0,end-extent);
 return offset;
}

function tabs(type,options,current){return `<div class="expand-tabs" role="group" aria-label="${safe(type)}">${options.map(([key,label])=>`<button data-inspect="${type}" data-value="${key}" aria-pressed="${key===current}">${label}</button>`).join('')}</div>`;}
function rgbAt(p,pixel){return p.pixels.slice(pixel*3,pixel*3+3).map(x=>Math.round((x+1)*127.5));}
function pixelInspector(s,p){const pixel=Math.floor(s.scalar/3),rgb=rgbAt(p,pixel),a=scalarAddress(s.scalar,s.patch);return `<div class="pixel-inspector"><div class="expand-label">SELECTED PATCH <strong>${s.patch} × ${s.patch} × 3</strong></div><div class="pixel-grid" role="group" aria-label="Pixels in selected patch" style="--pixels:${s.patch}">${Array.from({length:s.patch*s.patch},(_,i)=>{const rgb=rgbAt(p,i);return `<button data-pixel="${i}" style="background:rgb(${rgb.join(',')})" aria-label="Pixel row ${Math.floor(i/s.patch)}, column ${i%s.patch}. RGB ${rgb.join(', ')}" aria-pressed="${i===pixel}" tabindex="${i===pixel?0:-1}" title="(${Math.floor(i/s.patch)}, ${i%s.patch}) RGB ${rgb.join(', ')}"></button>`;}).join('')}</div><div class="pixel-coordinate"><label>row <input id="pixel-row" data-inspect="pixel-row" aria-label="Pixel row within patch" type="number" min="0" max="${s.patch-1}" value="${a.row}"></label><label>col <input id="pixel-col" data-inspect="pixel-col" aria-label="Pixel column within patch" type="number" min="0" max="${s.patch-1}" value="${a.col}"></label></div><div class="rgb-readout">${rgb.map((v,c)=>`<button data-inspect="channel" data-value="${c}" class="channel-${c}" aria-label="Inspect ${['red','green','blue'][c]} channel" aria-pressed="${c===a.channel}"><span>${'RGB'[c]}</span><strong>${v}</strong></button>`).join('')}</div><div class="pixel-hint">Choose a pixel, then its R, G, or B channel.</div></div>`;}

/** Patchification only: reorder RGB channels; do not apply learned parameters. */
function flattenView(s,p){
 const pixel=Math.floor(s.scalar/3),start=Math.min(Math.max(0,pixel-1),s.patch*s.patch-4),a=scalarAddress(s.scalar,s.patch),raw=rgbAt(p,pixel)[a.channel];
 const groups=Array.from({length:4},(_,k)=>start+k);
 return `<div class="flatten-view">
  <div class="flatten-shape"><span>${s.patch} × ${s.patch} × 3 <small>pixel grid</small></span><b>→</b><span>${p.pixels.length} values <small>one flattened patch</small></span></div>
  <div class="scalar-picker"><label for="scalar-index">Input index i <input id="scalar-index" data-inspect="scalar-index" aria-label="Input scalar index" type="number" min="0" max="${p.pixels.length-1}" value="${s.scalar}"></label><span>Row by row, pixel by pixel, R → G → B.</span></div>
  <div class="flatten-window-note">${start===0?'Start of vector':`… ${start*3} earlier values omitted`}</div>
  <div class="flatten-pixel-groups">${groups.map(px=>`<div class="flatten-pixel-group ${px===pixel?'selected-pixel-group':''}"><div class="flatten-pixel-label">Pixel (${Math.floor(px/s.patch)}, ${px%s.patch})</div><div class="flatten-rgb-triplet">${rgbAt(p,px).map((v,c)=>{const idx=px*3+c;return `<button data-scalar="${idx}" data-raw="${v}" data-normalized="${p.pixels[idx]}" aria-pressed="${idx===s.scalar}" aria-label="Flat index ${idx}, ${'RGB'[c]} channel ${v}, normalized ${nf(p.pixels[idx],5)}"><span class="flat-channel channel-${c}">${'RGB'[c]}</span><strong>${v}</strong><small>${nf(p.pixels[idx],3)}</small><span class="flat-index">i=${idx}</span></button>`;}).join('')}</div></div>`).join('')}</div>
  <div class="flatten-window-note">${groups.at(-1)===s.patch*s.patch-1?'End of vector':`… ${p.pixels.length-(groups.at(-1)+1)*3} later values omitted`}</div>
  <div class="flatten-selected" aria-live="polite"><span class="expand-label">SAME VALUE, A NEW ADDRESS</span><code>pixel[${a.row}, ${a.col}, ${'RGB'[a.channel]}] → flat[${s.scalar}] = <strong>${raw}</strong></code><p>i = 3 × (row × ${s.patch} + column) + channel, where R = 0, G = 1, B = 2.</p></div>
  <div class="flatten-normalization"><span class="expand-label">FIXED PREPROCESSING · NOT A LEARNED PROJECTION</span><code>x[${s.scalar}] = ${raw} / 127.5 − 1 = <strong data-flat-value="${p.pixels[s.scalar]}">${nf(p.pixels[s.scalar],7)}</strong></code><p>Large numbers show raw RGB; small numbers show the normalized inputs used by this demo. Normalization changes the scale, not the number of values.</p></div>
 </div>`;
}

/** A cutaway of a complete matrix, with explicit omission markers, never fabricated cells. */
function multiplicationView({input,weights,output,scalar,dimension,inputName='x',matrixName='Wₑ',outputName='e',pixel=false}){
 const j=Math.min(dimension,weights[0].length-1),i=Math.min(scalar,input.length-1),n=input.length,m=weights[0].length,trace=projectionTrace(input,weights,j),rows=visibleRows(n,i),startCol=Math.floor(j/4)*4,cols=Array.from({length:Math.min(4,m-startCol)},(_,k)=>startCol+k),raw=pixel?Math.round((input[i]+1)*127.5):null;
 const labels=r=>pixel?`${Math.floor(r/3)}${'RGB'[r%3]}`:String(r);
 return `<div class="matrix-product">
 <div class="scalar-picker"><label for="scalar-index">Input index i <input id="scalar-index" data-inspect="scalar-index" aria-label="Input scalar index" type="number" min="0" max="${n-1}" value="${i}"></label><span>Highlight a value × a weight; sum the whole column.</span></div>
 <div class="product-equation"><span>${inputName} <small>1 × ${n}</small></span><b>×</b><span>${matrixName} <small>${n} × ${m}</small></span><b>=</b><span>${outputName} <small>1 × ${m}</small></span></div>
 <div class="product-cutaway">
  <div class="input-cutaway"><div class="cutaway-label">${inputName} <small>${pixel?'pixel / channel':'vertical view'}</small></div><div class="cutaway-omission">${rows[0]>0?'⋮':'START'}</div><div class="input-cells">${rows.map(r=>`<button data-scalar="${r}" aria-pressed="${r===i}" title="${inputName}[${r}] = ${nf(input[r],7)}"><small>${labels(r)}</small><span>${nf(input[r])}</span></button>`).join('')}</div><div class="cutaway-omission">${rows.at(-1)<n-1?'⋮':'END'}</div><div class="cutaway-caption">${n} values</div></div>
  <span class="product-sign" aria-hidden="true">×</span>
  <div class="weight-cutaway"><div class="cutaway-label">${matrixName} <small>actual weights</small></div><div class="cutaway-omission">${rows[0]>0?'⋮':'START'}</div><div class="weight-grid" style="--weight-cols:${cols.length}">${rows.map(r=>cols.map(c=>`<button data-weight="${r},${c}" class="${c===j?'weight-column ':''}${r===i?'weight-row':''}" aria-label="${matrixName} row ${r}, column ${c}: ${nf(weights[r][c],6)}" aria-pressed="${r===i&&c===j}" style="--cell:${tint(weights[r][c],Math.sqrt(3/n))}">${nf(weights[r][c])}</button>`).join('')).join('')}</div><div class="cutaway-omission">${rows.at(-1)<n-1?'⋮':'END'}</div><div class="weight-columns">${cols.map(c=>`<button data-dimension="${c}" aria-pressed="${c===j}">j=${c}</button>`).join('')}</div><div class="cutaway-caption">Rows ${rows[0]}–${rows.at(-1)} / ${n}; columns ${startCol}–${cols.at(-1)} / ${m}</div></div>
  <span class="product-sign" aria-hidden="true">=</span>
  <div class="output-cutaway"><div class="cutaway-label">${outputName} <small>pick a dimension</small></div><div class="output-cell-grid">${output.map((v,c)=>`<button data-dimension="${c}" aria-label="Output dimension ${c}, value ${nf(v,6)}" aria-pressed="${c===j}" style="--cell:${tint(v,2)}"><small>${c}</small><span>${nf(v)}</span></button>`).join('')}</div><div class="cutaway-caption">All ${m} dimensions</div></div>
 </div>
 <div class="selected-product" aria-live="polite"><span class="trace-tag">ONE TERM</span><code>${pixel?`${'RGB'[i%3]} ${raw} → `:''}${inputName}[${i}] ${nf(input[i],5)} <b>×</b> ${matrixName}[${i},${j}] ${nf(weights[i][j],5)} <b>=</b> <strong>${nf(trace.terms[i],7)}</strong></code></div>
 <div class="dot-sum"><span>SUM ALL ${n} TERMS</span><code>${outputName}[${j}] = Σᵢ ${inputName}[i] · ${matrixName}[i,${j}] = <strong data-trace-sum="${trace.sum}" data-trace-output="${output[j]}">${nf(trace.sum,7)}</strong></code></div>
 </div>`;
}
function numberVector(label,values){return `<div class="expanded-vector"><span>${label}</span><div>${values.map((v,i)=>`<span style="background:${tint(v,2)}" title="Dimension ${i}: ${nf(v,6)}">${nf(v,2)}</span>`).join('')}</div>`;}

export function expansionMarkup(s,patches,result,imageURL){
 const step=s.expanded,selected=Math.max(0,s.selected),p=patches[selected],token=s.model==='vit'?s.selected+1:selected,l=result.layers[s.layer],head=l.heads[s.head],isPix=s.model==='pixtral';
 const title=step===1?'Rearrange pixels. Keep every value.':step===2?'One row per token. All embeddings in view.':step===0?'Zoom all the way into the pixels.':step===3?'One image. Multiple attention heads.':step===4?'Inside the feed-forward network.':step===5?'The final output is a matrix of vectors.':'From image representations to a task.';
 let body='',sub='',controls=`<div class="inspected-token"><span>${step===3?'Query token':'Inspecting'}</span><strong>${s.selected<0?'[CLS]':`p${String(s.selected).padStart(2,'0')} · (${p.row}, ${p.col})`}</strong><small>select from the patch sequence</small></div>`;
 if(step<=2){
  if(s.selected<0){sub='[CLS] is a separate trainable vector, not a projected image patch.';body=`<div class="cls-explanation">${numberVector('Raw [CLS]',result.cls)}${numberVector('+ position',result.positions[0])}${numberVector('Input [CLS]',result.input[0])}<p>Choose a patch from the sequence to inspect pixel projection.</p></div>`;}
  else{
   s.scalar=Math.max(0,Math.min(p.pixels.length-1,s.scalar));s.dimension=Math.min(DIM-1,s.dimension);
   sub=step===0?'An image is a grid of RGB numbers. Choose a patch, then inspect any of its pixels.':step===1?`Split the image into ${patches.length} non-overlapping patches. Each ${s.patch} × ${s.patch} RGB patch becomes a list of ${p.pixels.length} values. No learned weights are used here.`:`Same weights for every patch. ${s.patch} × ${s.patch} × 3 = ${p.pixels.length} inputs; ${DIM} embedding dimensions.`;
   if(step===0){body=`<div class="input-expansion"><div><div class="expand-label">PREPROCESSED IMAGE <strong>${s.width} × ${s.height}</strong></div><div class="expanded-image-grid" style="background-image:url('${imageURL}');grid-template-columns:repeat(${s.cols},1fr);aspect-ratio:${s.cols}/${s.rows}">${patches.map((pt,i)=>`<button data-expanded-patch="${i}" aria-label="Inspect image patch ${i}" aria-pressed="${i===selected}"></button>`).join('')}</div><p class="pixel-hint">${isPix?'Aspect-preserving resize + edge padding':'Fixed-size center crop'} · ${patches.length} patches</p></div><span class="product-sign">→</span>${pixelInspector(s,p)}<div class="pixel-explanation"><h4>Three numbers per pixel.</h4><p>The highlighted cell is one pixel, not a whole patch. Its R, G, and B values each range from 0 to 255.</p><p>For this teaching model, normalize each channel:</p><code>x = RGB / 127.5 − 1</code><p>Then flatten in row-major pixel order, keeping R, G, B together.</p><button class="expand-forward" data-expand-step="1">Follow the flattening →</button></div></div>`;}
   else if(step===1){body=`<div class="projection-expansion flatten-expansion">${pixelInspector(s,p)}<div class="flatten-connector"><span>→</span><small>reorder<br>only</small></div>${flattenView(s,p)}</div><div class="patch-embedding-handoff"><div><strong>Still pixels—not an embedding yet.</strong><p>One patch gives one ${p.pixels.length}-value input vector, not ${p.pixels.length} tokens. In the next stage, a shared weight matrix maps it to ${DIM} hidden dimensions.</p></div><button class="expand-forward" data-expand-step="2">Next: multiply by embedding weights →</button></div>`;}
   else{body=`<div class="projection-expansion">${pixelInspector(s,p)}<div class="flatten-connector"><span>→</span><small>prepared<br>input</small></div>${multiplicationView({input:p.pixels,weights:result.projectionWeights,output:result.embeddings[selected],scalar:s.scalar,dimension:s.dimension,pixel:true,patchSize:s.patch})}</div><div class="projection-footnote"><strong>Flatten order:</strong> pixel (0,0) R G B → pixel (0,1) R G B → … · Normalize: RGB / 127.5 − 1 · Bias omitted in this teaching projection.</div>`;
    if(step===2){const j=s.dimension;body+=`<div class="position-after"><span class="expand-label">AFTER THE PROJECTION</span>${isPix?`<code>RMSNorm(e)[${j}] = ${nf(result.input[token][j],6)}</code><p>2D RoPE is applied to Q and K inside each attention block—not added here.</p>`:`<code>e[${j}] ${nf(result.embeddings[selected][j],5)} + position[${j}] ${nf(result.positions[token][j],5)} = x[${j}] <strong>${nf(result.input[token][j],6)}</strong></code><p>The position vector is added once, before block 1. It is not part of the pixel projection.</p>`}</div>`;}
   }
  }
 } else if(step===3){
  s.dimension=Math.min(s.attentionInspect==='merge'?DIM-1:HEAD_DIM-1,s.dimension);s.scalar=Math.min(DIM-1,s.scalar);
  sub=`Block ${s.layer+1} · ${l.heads.length} parallel heads. Each head has narrower Q/K/V vectors and its own attention weights. They all see the same tokens. Select a head to inspect it, then follow the outputs back into the residual stream.`;
  body=multiHeadMarkup(s,patches,l);
  if(s.headDetailOpen||s.attentionInspect&&s.attentionInspect!=='merge'){
   body+=`<section id="head-detail" class="mh-head-detail" aria-label="Head ${s.head+1} detail"><div class="mh-detail-heading"><div><span class="mh-detail-eyebrow">INSIDE HEAD ${s.head+1}</span><h4>${HEAD_DIM} features per token · all ${l.input.length} tokens</h4></div><button type="button" data-head-detail-close>↑ Back to all heads</button></div>${attentionFlowMarkup(s,patches,head,{embedded:true})}</section>`;
  }else body+='<div id="head-detail" hidden></div>';
  if(s.attentionInspect){
   const section=s.attentionInspect;
   let arithmetic='';
   if(['q','k','v'].includes(section)){
    const op=s.qkv||section,headStart=s.head*HEAD_DIM,w=l.weights[op].map(row=>row.slice(headStart,headStart+HEAD_DIM)),inputToken=op==='q'?token:s.key,raw=op==='q'?head.rawQ[inputToken]:op==='k'?head.rawK[inputToken]:head.values[inputToken];
    arithmetic=tabs('qkv',[['q','Query Q'],['k','Key K'],['v','Value V']],op)+multiplicationView({input:l.normalized[inputToken],weights:w,output:raw,scalar:s.scalar,dimension:s.dimension,inputName:'h̄',matrixName:`W${op.toUpperCase()}`,outputName:op.toUpperCase()});
   }else if(section==='merge'){
    const merged=multiHeadView(l,token,!!s.compareCausal);
    arithmetic=multiplicationView({input:merged.concatenated,weights:l.weights.o,output:merged.projected,scalar:s.scalar,dimension:s.dimension,inputName:'concat',matrixName:'Wₒ',outputName:'Δh'})+`<div class="position-after"><span class="expand-label">ADD THE ORIGINAL INPUT</span><code>h′[${s.dimension}] = h[${s.dimension}] + Δh[${s.dimension}] = ${nf(l.input[token][s.dimension],6)} + ${nf(merged.projected[s.dimension],6)} = <strong>${nf(merged.residual[s.dimension],7)}</strong></code><p>This is the attention sublayer’s residual output, before the MLP.${s.compareCausal?' Comparison only; the actual encoder output is unchanged.':''}</p></div>`;
   }else arithmetic=attentionScoreDetail(s,patches,head);
   const title={q:'Query projection',k:'Key projection',v:'Value projection',scores:'Scaled dot product',softmax:'Row-wise softmax',values:'Weighted value sum',rope:'2D rotary positions',merge:'Concatenate heads → output projection → residual'}[section];
   body+=`<section class="af-calculation" id="attention-calculation" aria-label="${title}"><div class="af-calculation-heading"><h4>${title}</h4><button type="button" data-attn-close>− Close calculation</button></div>${s.compareCausal?'<p class="af-comparison-note">Decoder-mask comparison is active. All displayed heads use the comparison mask; Q/K/V and the actual encoder run remain unchanged.</p>':''}${arithmetic}</section>`;
  }else body+='<div id="attention-calculation" hidden></div>';
 } else if(step===4){
  const op=s.ffn||'up',w=l.weights[op],input=op==='down'?l.expanded[token]:l.ffnInput[token],output=op==='down'?l.mlpDelta[token]:op==='gate'?l.gate[token]:l.preActivation[token];s.scalar=Math.min(input.length-1,s.scalar);s.dimension=Math.min(output.length-1,s.dimension);
  sub=`Block ${s.layer+1}. Each token is transformed independently; no patch-to-patch mixing happens here.`;
  body=tabs('ffn',[['up','1 · Expand 24 → 48'],...(isPix?[['gate','2 · Gate 24 → 48']]:[]),['down',`${isPix?'3':'2'} · Contract 48 → 24`]],op)+multiplicationView({input,weights:w,output,scalar:s.scalar,dimension:s.dimension,inputName:op==='down'?'a':'h̄',matrixName:op==='down'?'W₂':op==='gate'?'W₃':'W₁',outputName:op==='down'?'Δh':'z'})+`<div class="ffn-expanded-math"><div><span class="expand-label">NONLINEARITY</span><p>${isPix?'SiLU(W₁h̄) ⊙ W₃h̄':'GELU(W₁h̄)'} produces the 48-dimensional vector passed to W₂.</p>${numberVector('activated',l.expanded[token])}</div><div><span class="expand-label">RESIDUAL CONNECTION</span><code>h′[0] = h[0] + Δh[0]</code><code>${nf(l.afterAttention[token][0],5)} + ${nf(l.mlpDelta[token][0],5)} = <strong>${nf(l.output[token][0],5)}</strong></code></div></div>`;
  body=mlpLanesMarkup(s,patches,result)+`<details class="mlp-static-details"><summary>Static numeric inspector · weights, activations & residual</summary>${body}</details>`;
 } else if(step===5){
  sub=`After all ${result.layers.length} encoder blocks, each token has ${DIM} features. The encoder itself has not selected a class or generated a word.`;
  body=outputReadoutMarkup(s,patches,result);
 } else {
  sub=isPix?'All patch vectors enter the language model through a projector. Text generation starts in the causal decoder, not in the vision encoder.':'A classification ViT uses the extra [CLS] token as a summary readout. Vision-language models such as Pixtral use a different path to text.';
  body=nextStageMarkup(s,patches,result);
  if(isPix&&s.projectorDetail){
   const op=s.adapter||'first',idx=op==='first'?0:1,w=result.adapterWeights[idx],input=idx?result.adapterHidden[token]:result.final[token],output=idx?result.projected[token]:result.adapterPre[token];s.scalar=Math.min(input.length-1,s.scalar);s.dimension=Math.min(31,s.dimension);
   body+=`<section class="ro-projector-detail" id="projector-detail"><h4>Actual teaching projector · p${selected}</h4>${tabs('adapter',[['first','1 · Linear 24 → 32'],['second','2 · GELU → Linear 32 → 32']],op)}${multiplicationView({input,weights:w,output,scalar:s.scalar,dimension:s.dimension,inputName:idx?'GELU(z)':'h',matrixName:idx?'W₂':'W₁',outputName:idx?'LLM':'z'})}</section>`;
  }else body+='<div id="projector-detail" hidden></div>';
 }

 if(step===2){
  sub=`${result.input.length} token vectors × ${DIM} features. Select any row to follow its input preparation; the full matrix stays visible.`;
  body=embeddingOverviewMarkup(s,patches,result)+`<details id="embedding-arithmetic" class="eb-arithmetic" ${s.embeddingDetail?'open':''}><summary>${s.selected<0?'Inspect the learned CLS vector and position addition':'Inspect pixel projection and input preparation'} <span>exact values & weights</span></summary>${body}</details>`;
 }

 return `<div class="expansion-topline"><span class="expansion-breadcrumb">ARCHITECTURE / ${names[step].toUpperCase()}</span><button data-collapse class="collapse-expansion" aria-label="Collapse expanded section">− Collapse</button></div><div class="expansion-workspace">${patchRailMarkup(s,patches,imageURL)}<div class="expansion-inspection"><div class="expansion-intro"><h3>${title}</h3><p>${sub}</p></div><div class="expansion-controls">${controls}<span class="weights-disclosure">${step<=1?'Live pixels · no learned parameters':step===6?'Vectors computed · decoder/classifier schematic':'Live values · untrained weights'}</span></div>${body}<div class="expansion-nav"><button data-expand-step="${step-1}" ${step===0?'disabled':''}>← ${names[step-1]||'Previous'}</button><span>Click another stage to open it in place</span><button data-expand-step="${step+1}" ${step===names.length-1?'disabled':''}>${names[step+1]||'Next'} →</button></div></div></div>`;
}
