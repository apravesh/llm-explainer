import {withCausalAttention} from './attention-stages.mjs';
import {attentionWindowMarkup} from './attention-window.mjs';
/** A single, connected attention calculation. Detail is opt-in; tensors remain full size. */
import {viewMatrix,rowOf} from './full-config.mjs';
import {featureMarkup} from './full-matrix.mjs';
export function simpleAttentionData(run,state){
 const c=run.config,l=run.layers[state.layer],head=l.heads[state.head],query=state.selected+(c.cls?1:0),key=state.key,offset=state.head*c.headWidth;
 const data={c,l,head,headIndex:state.head,query,key,n:l.input.rows,
  q:viewMatrix(l.q,{colStart:offset,cols:c.headWidth}),k:viewMatrix(l.k,{colStart:offset,cols:c.headWidth}),v:viewMatrix(l.values,{colStart:offset,cols:c.headWidth}),
  score:head.scores.data[query*head.scores.cols+key],weight:head.attention.data[query*head.attention.cols+key]};
 return state.attentionCausal?withCausalAttention(data):data;
}
export function attentionMixPrefix(data,count,feature){
 const limit=Math.max(0,Math.min(data.n,Math.floor(count)));let sum=0;
 for(let k=0;k<limit;k++)sum+=data.head.attention.data[data.query*data.n+k]*data.l.values.data[k*data.c.width+data.c.headWidth*data.headIndex+feature];
 return sum;
}
const saNum=(v,d=4)=>Number(v).toFixed(d);
function saToken(i,c){return c.cls&&i===0?'CLS':`p${i-(c.cls?1:0)}`;}
export function simpleAttentionMarkup(renderer,run,s,onPair){
 const a=simpleAttentionData(run,s),{c,n}=a;
 return `<div class="sa-simple sa-windowed" id="full-head-detail" data-detail-head="${s.head}">
 <div class="sa-context"><strong>${a.causalComparison?'Decoder-mask comparison · not the encoder':'Encoder attention · no causal mask'}</strong><span>${n} tokens · ${c.headWidth} features per head</span></div>
 <div class="sa-workbench aw-workbench"><div data-attention-window>${attentionWindowMarkup(a,s)}</div></div>
 <div class="sa-play-row"><span class="sa-auto-mixing" data-value-mixing-label>Value mixing · loops automatically</span><span id="full-value-mix" aria-live="off">All ${n} weighted value vectors summed.</span></div>
 <div class="sa-merge">${a.causalComparison?'<small>Comparison only · encoder and later blocks unchanged.</small>':''}<span>${c.heads} heads × ${c.headWidth}</span><b>→</b><span>concatenate</span><b>→</b><button type="button" data-full-op="attention-output">Wₒ projection ↗</button><b>+</b><span>input residual</span><b>→</b><strong>${c.width.toLocaleString()}-feature output</strong></div>
 </div>`;
}
export function simpleAttentionDetails(renderer,run,s,causalHead){
 const a=simpleAttentionData(run,s),{c,l,head,query,key}=a,opened=s.attentionPanels||{},open=name=>opened[name]?'open':'';
 let dot=0;for(let d=0;d<c.headWidth;d++)dot+=l.q.data[query*c.width+s.head*c.headWidth+d]*l.k.data[key*c.width+s.head*c.headWidth+d];
 const scores=rowOf(head.scores,query),max=Math.max(...scores),den=scores.reduce((sum,x)=>sum+Math.exp(x-max),0),numerator=Math.exp(a.score-max);
 return `<div class="sa-details"><details id="sa-detail-scores" data-attention-disclosure="scores" ${open('scores')}><summary>Inspect the selected score & softmax</summary><div class="sa-exact-math"><code>Σ over ${c.headWidth} Q/K features = ${saNum(dot,7)}</code><code>scaled score = ${saNum(dot,7)} / √${c.headWidth} ≈ ${saNum(a.score,7)}</code><code>softmax: exp(score − max) / Σ exp(row − max)</code><code>${saNum(numerator,7)} / ${saNum(den,7)} ≈ ${saNum(a.weight,7)}</code><p>FP32 engine values may differ slightly from the Float64 inspection sum. ${a.causalComparison?'Future keys are blocked in this comparison only.':'Every key is included; no causal mask.'}</p></div></details>
 <details id="sa-detail-vectors" data-attention-disclosure="vectors" ${open('vectors')}><summary>Inspect exact Q, K, V and head-output features</summary>${featureMarkup(a.q,query,s.headFeature,{label:c.rope?'Q after RoPE':'Query',key:'head'})}${featureMarkup(a.k,key,s.headFeature,{label:c.rope?'K after RoPE':'Key',key:'head'})}${featureMarkup(a.v,key,s.headFeature,{label:'Value at selected key',key:'head'})}${featureMarkup(head.out,query,s.headFeature,{label:'Head output · weighted sum of V',key:'head'})}</details>
 <details id="sa-detail-positions" data-attention-disclosure="positions" ${open('positions')}><summary>${c.rope?'How 2D RoPE fits here':'Where position information enters'}</summary><p>${c.rope?`${c.headWidth/2} coordinate pairs per head alternate patch rows and columns; frequency = 10,000^(−2 × pair / ${c.headWidth}). Each block rotates its fresh Q/K vectors before scoring. V is not rotated. The projection overlays show pre-RoPE values.`:'Learned absolute position vectors are added before block 1. Q, K, and V are computed from each block’s normalized hidden state.'}</p></details>
 <details id="sa-detail-residual" data-attention-disclosure="residual" ${open('residual')}><summary>Inspect concatenation, Wₒ and residual output</summary><p>Concatenate all ${c.heads} head outputs, preserving head order: ${c.heads} × ${c.headWidth} = ${c.width}. Apply Wₒ and add the original block input. Heads are not averaged.</p><button type="button" class="fa-action" data-full-op="attention-output">▶ Inspect output projection</button>${featureMarkup(l.afterAttention,query,s.feature,{label:'After attention + residual'})}</details>
 <details id="sa-detail-causal" data-attention-disclosure="causal" ${open('causal')}><summary>Optional: compare with a decoder’s causal mask</summary>${opened.causal?`<p>Same Q/K/V with future keys blocked. This comparison does not alter the encoder or later blocks.</p>${renderer.markup(causalHead(l,c,s.head).attention,{label:'Causal comparison',row:query,col:key,height:210,positive:true})}`:''}</details></div>`;
}
/** Coordinates are canvas centers; the V bypass deliberately avoids score/softmax nodes. */
export function attentionFlowPaths(boxes){
 const {q,k,v,scores,weights,mix,output}=boxes,mid=r=>({x:r.x+r.width,y:r.y+r.height/2});
 const curve=(start,end)=>`M${start.x},${start.y} C${start.x+(end.x-start.x)*.5},${start.y} ${start.x+(end.x-start.x)*.5},${end.y} ${end.x},${end.y}`;
 const baseline=Math.max(v.y+v.height,scores.y+scores.height,weights.y+weights.height,output.y+output.height)+32;
 return [
  {id:'q-scores',kind:'q',d:curve(mid(q),{x:scores.x-4,y:scores.y+scores.height*.33})},
  {id:'k-scores',kind:'k',d:curve(mid(k),{x:scores.x-4,y:scores.y+scores.height*.67})},
  {id:'scores-softmax',kind:'q',d:curve(mid(scores),{x:weights.x-4,y:weights.y+weights.height/2})},
  {id:'weights-mix',kind:'q',d:curve(mid(weights),{x:mix.x-4,y:mix.y+mix.height*.34})},
  {id:'v-mix',kind:'v',d:`M${v.x+v.width},${v.y+v.height/2} C${v.x+v.width+15},${v.y+v.height/2} ${v.x+v.width+8},${baseline} ${v.x+v.width+25},${baseline} L${mix.x-13},${baseline} Q${mix.x-5},${baseline} ${mix.x-5},${baseline-8} L${mix.x-5},${mix.y+mix.height*.76} L${mix.x},${mix.y+mix.height*.76}`},
  {id:'mix-output',kind:'v',d:`M${mix.x+mix.width/2},${mix.y+mix.height} L${output.x+output.width/2},${output.y-5}`}
 ];
}
export function drawSimpleAttentionWires(root){
 const bench=root.querySelector('.sa-workbench');if(!bench||bench.classList.contains('aw-workbench'))return;const base=bench.getBoundingClientRect();if(!base.width)return;
 const boxes={};for(const name of ['q','k','v','scores','weights','mix','output']){const node=bench.querySelector(`[data-sa-node="${name}"]`),target=node.querySelector('canvas')||node,r=target.getBoundingClientRect();boxes[name]={x:r.left-base.left,y:r.top-base.top,width:r.width,height:r.height};}
 const svg=bench.querySelector('.sa-wires');svg.setAttribute('viewBox',`0 0 ${base.width} ${base.height}`);
 svg.querySelector('[data-sa-wire-paths]').innerHTML=attentionFlowPaths(boxes).map(p=>`<path data-sa-wire="${p.id}" class="sa-wire sa-wire-${p.kind}" d="${p.d}" marker-end="url(#sa-arrow-${p.kind})"/>`).join('');
}
