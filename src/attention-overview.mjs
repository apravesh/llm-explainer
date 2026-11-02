/** Process-only counterpart to the collapsed MLP. No synthetic mini-matrix values. */
export function attentionProcessMarkup(run,state){
 const c=run.config,d=c.headWidth,fmt=n=>n.toLocaleString();
 const text=(x,y,value,cls='mp-title')=>`<text x="${x}" y="${y}" class="${cls}">${value}</text>`;
 const node=(key,x,y,w,title,caption)=>`<g class="mp-node ap-${key}" data-ap-node="${key}"><rect x="${x}" y="${y}" width="${w}" height="44" rx="6"/>${text(x+w/2,y+18,title)}${text(x+w/2,y+33,caption,'mp-caption')}</g>`;
 const edges=[
  ['input-q','M37 85H47V24H58'],['input-k','M47 85H58'],['input-v','M47 85V146H58'],
  ['q-weights','M116 24H132Q141 24 141 35V64H157'],['k-weights','M116 85H157'],
  ['weights-mix','M308 74H349'],['v-mix','M116 146H386V101'],['mix-output','M424 74H449']
 ];
 return `<div class="mp-overview ap-overview" data-attention-process-overview data-block="${state.layer}" data-head="${state.head}">
 <button type="button" class="mp-process ap-process" data-attention-process data-expand-step="3" aria-label="Expand Self-attention process flow" aria-controls="stage-expansion-3" aria-expanded="${state.expanded===3}">
 <svg viewBox="0 0 490 200" role="img" aria-label="Normalized input is projected into Q, K and V for each head. ${c.rope?'2D RoPE rotates Q and K. ':''}Q times K transpose, scaled by the square root of ${d}, then softmax produces attention weights over all keys without a causal mask. Weights multiply V to mix patches. Concatenate all ${c.heads} head outputs, project through W o and add the input residual.">
 <defs><marker id="ap-process-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#9aacc6"/></marker></defs>
 <g class="mp-wires" marker-end="url(#ap-process-arrow)">${edges.map(([name,path])=>`<path data-ap-edge="${name}" d="${path}"/>`).join('')}</g>
 ${text(18,81,'X')}${text(18,96,'input','mp-caption')}
 ${node('q',62,2,54,'Q','Wq')}${node('k',62,63,54,'K','Wk')}${node('v',62,124,54,'V','Wv')}
 ${node('weights',162,52,146,`QKᵀ / √${d}`,'softmax → weights')}
 ${node('mix',354,52,70,'× V','mix patches')}
 ${text(466,72,'O')}${text(466,87,'head out','mp-caption')}
 ${text(245,188,`all ${c.heads} heads → concat → Wₒ + residual H → H′`,'mp-residual')}
 </svg>
 </button>
 <p class="mp-dimensions">${c.heads} heads × ${fmt(d)} features → ${fmt(c.width)}</p>
 <p class="mp-help">${c.rope?'2D RoPE on Q/K. ':''}No causal mask · every patch can attend to every patch.<br>Click to inspect the matrices.</p>
 </div>`;
}
