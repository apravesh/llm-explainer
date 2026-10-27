/** Browsing one attention head never changes the parallel multi-head computation. */
export function nextStackHead(current,key,count){
 if(key==='Home')return 0;if(key==='End')return count-1;
 const delta={ArrowLeft:-1,ArrowUp:-1,ArrowRight:1,ArrowDown:1}[key];
 return delta===undefined?null:(current+delta+count)%count;
}
export function headStackMarkup(renderer,config,heads,state,query,onPair,{content=null}={}){
 const h=state.head,head=heads[h],n=head.attention.rows,d=config.headWidth,total=heads.length;
 const weight=head.attention.data[query*n+state.key];
 return `<section class="fa-head-stack" aria-label="Stacked attention heads" data-head-count="${total}" data-selected-head="${h}">
 <div class="fa-stack-toolbar"><div><span>PARALLEL ATTENTION HEADS</span><strong id="head-stack-position" aria-live="polite">Head ${h+1} of ${total}</strong></div><div class="fa-stack-arrows"><button type="button" data-stack-step="-1" aria-label="Previous stacked head" aria-controls="head-stack-front">←</button><button type="button" data-stack-step="1" aria-label="Next stacked head" aria-controls="head-stack-front">→</button></div></div>
 <div class="fa-stack-deck"><div class="fa-stack-sheet sheet-back" aria-hidden="true"></div><div class="fa-stack-sheet sheet-middle" aria-hidden="true"></div><div class="fa-stack-sheet sheet-near" aria-hidden="true"></div>
 <article id="head-stack-front" class="fa-stack-front ${content===null?'':'is-connected-flow'}" aria-labelledby="head-stack-position" data-head-preview="${h}">
  ${content===null?`<div class="fa-stack-matrix">${renderer.markup(head.attention,{label:`Head ${h+1} attention`,row:query,col:state.key,height:150,positive:true,click:onPair})}</div>
  <div class="fa-stack-summary"><span class="fa-stack-range">Projected features ${h*d}–${(h+1)*d-1}</span><strong>${d} features per head. All ${n} tokens.</strong><p>Q, K and V each have shape <b>${n} × ${d}</b>. Attention weights are <b>${n} × ${n}</b>; no causal mask.</p><div class="fa-stack-weight"><span>query ${query} → key ${state.key}</span><b>${(weight*100).toFixed(3)}%</b></div><small>All ${total} heads run in parallel. This stack only chooses which head you inspect.</small></div>`:content}
 </article></div>
 <div class="fa-stack-selector" role="group" aria-label="Choose an attention head">${heads.map((_,i)=>`<button type="button" data-full-head="${i}" aria-label="Inspect head ${i+1} of ${total}" aria-controls="head-stack-front full-head-detail" aria-pressed="${i===h}" tabindex="${i===h?0:-1}" title="Head ${i+1} · features ${i*d}–${(i+1)*d-1}">${i+1}</button>`).join('')}</div>
 <p class="fa-stack-help">Choose a head, or use ← → / Home / End. ${content===null?'Its full computation is shown below.':'All heads run in parallel; only the selected head is shown.'}</p>
 </section>`;
}
