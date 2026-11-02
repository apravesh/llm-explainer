/** Collapsed overview shows dependencies only. Exact matrices live in the expanded flow. */
export function mlpProcessMarkup(run,state){
 const c=run.config,gated=c.gated,name=gated?'Gated MLP':'MLP';
 const text=(x,y,value,cls='mp-title')=>`<text x="${x}" y="${y}" class="${cls}">${value}</text>`;
 const box=(x,y,w,title,caption,kind='')=>`<g class="mp-node ${kind}"><rect x="${x}" y="${y}" width="${w}" height="44" rx="6"/>${text(x+w/2,y+18,title)}${text(x+w/2,y+33,caption,'mp-caption')}</g>`;
 const path=d=>`<path d="${d}"/>`,marker=gated?'mp-arrow-swiglu':'mp-arrow-gelu';
 const branches=gated?[
  'M48 70H65Q72 70 72 63V32Q72 25 80 25H91','M72 70V109Q72 116 80 116H91',
  'M217 25H234Q244 25 244 36V60Q244 67 253 67H269','M217 116H234Q244 116 244 106V84Q244 77 253 77H269',
  'M323 72H353','M418 72H446'
 ]:['M48 70H91','M217 70H353','M418 70H446'];
 return `<div class="mp-overview" data-mlp-process-overview data-block="${state.layer}">
 <button type="button" class="mp-process" data-mlp-process data-expand-step="4" aria-label="Expand ${name} process flow" aria-controls="stage-expansion-4" aria-expanded="${state.expanded===4}">
 <svg viewBox="0 0 490 ${gated?172:125}" role="img" aria-label="${gated?'The same normalized input enters W1 then SiLU and a parallel W3 expansion; multiply matching features, then contract through W2.':'Expand through W1, apply GELU, then contract through W2.'} Add the pre-MLP residual to the update.">
 <defs><marker id="${marker}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#9aacc6"/></marker></defs>
 <g class="mp-wires" marker-end="url(#${marker})">${branches.map(path).join('')}</g>
 ${text(28,67,'X')}${text(28,82,'input','mp-caption')}
 ${box(96,gated?3:48,121,gated?'W₁ → SiLU':'W₁ → GELU',gated?'expand + activate':'expand + activate','mp-gate')}
 ${gated?box(96,94,121,'W₃','parallel expansion','mp-values'):''}
 ${gated?box(274,50,49,'⊙','multiply','mp-mix'):''}
 ${box(358,50,60,'W₂','contract')}
 ${text(466,69,'Δ')}${text(466,83,'update','mp-caption')}
 ${text(245,gated?160:114,'+ residual H → H′','mp-residual')}
 </svg>
 </button>
 <p class="mp-dimensions">${c.width.toLocaleString()} → ${c.ffWidth.toLocaleString()} → ${c.width.toLocaleString()}</p>
 <p class="mp-help">${gated?'Two branches, same input.':'One expansion, then activation.'} Shared weights for every patch.<br>Click to inspect the matrices.</p>
 </div>`;
}
