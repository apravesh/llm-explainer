const count=n=>Math.round(n).toLocaleString('en-US');
const compact=n=>n>=1e9?(n/1e9).toFixed(2)+'B':n>=1e6?(n/1e6).toFixed(2)+'M':count(n);
const memory=n=>n>=1073741824?(n/1073741824).toFixed(2)+' GiB':(n/1048576).toFixed(2)+' MiB';
const work=n=>n>=1e12?(n/1e12).toFixed(2)+' TFLOPs':(n/1e9).toFixed(2)+' GFLOPs';
/** Symbolic parameter counts use f′ for the rounded matched-budget width. */
export function parameterFormulas(c,a){
 const width=a.gated&&!a.moe&&a.f!==c.f?'f′':'f';
 const attention=c.kv===c.h?'4d²':'2d² + 2d·dKV',kv=c.kv===c.h?'d²':'d·dKV';
 const expert=`${a.gated?3:2}d${width}`,ffn=a.moe?`E × ${expert}`:expert,router=a.moe?'dE':'0';
 return {width,attention,kv,expert,ffn,router,ffnTotal:a.moe?`${ffn} + dE`:ffn,ffnActive:a.moe?`k × ${expert} + dE`:expert,
  stored:`${attention} + ${ffn}${a.moe?' + dE':''}`,active:`${attention} + ${a.moe?`k × ${expert} + dE`:expert}`,projection:a.moe?'Edf':`d${width}`};
}
function parameterLegend(c,a){return `<dl class="bv-symbols" aria-label="Parameter notation"><div><dt>d = ${count(c.d)}</dt><dd>hidden features per token</dd></div><div><dt>f = ${count(c.f)}</dt><dd>MLP intermediate width</dd></div>${a.f!==c.f?`<div><dt>f′ = ${count(a.f)}</dt><dd>matched width: 2f/3, rounded up to 64</dd></div>`:''}<div><dt>Hq = ${c.h} · Hkv = ${c.kv}</dt><dd>query heads · key/value heads</dd></div><div><dt>dh = d/Hq = ${a.dh}</dt><dd>features per head</dd></div><div><dt>dKV = Hkv × dh = ${a.dkv}</dt><dd>combined key/value width</dd></div>${a.moe?`<div><dt>E = ${c.experts} · k = ${c.k}</dt><dd>stored experts · selected per token</dd></div>`:''}<div><dt>L = ${c.layers}</dt><dd>blocks; stack count = L × block count</dd></div></dl>`;}
function meter(n,total,kind){return `<span class="bv-meter"><i class="${kind}" style="width:${100*n/total}%"></i></span>`;}
function weight(name,shape,n,total,kind,formula){return `<div class="bv-weight"><span><b>${name}</b><small>${shape}</small><code class="bv-weight-formula">${formula}</code></span><strong>${compact(n)}</strong>${meter(n,total,kind)}</div>`;}
export function parameterDiagram(b,key='ordinary'){
 const c=b.c,a=b.variants[key],expertCount=a.moe?c.experts:1;
 const formula=parameterFormulas(c,a);
 const names={ordinary:'Ordinary MLP',swiglu:'SwiGLU',matched:'Budget-matched SwiGLU',moe:'Routed SwiGLU experts'};
 return `<div class="bv-caption"><strong>One transformer block · ${names[key]}</strong><span>${count(a.stored)} stored parameters · ${count(a.active)} active per token</span></div>
 ${parameterLegend(c,a)}
 <div class="bv-block" data-visual-variant="${key}"><span class="bv-port">Token<br><b>h</b></span><span class="bv-arrow">→</span>
 <article class="bv-module"><h3>Self-attention</h3><p class="bv-total">${compact(a.attention)} <code class="bv-total-formula">= ${formula.attention}</code><small>${(100*a.attention/a.stored).toFixed(1)}% of block</small></p><p class="bv-step">Norm → Q/K/V projections</p>
 ${weight('WQ',`${c.d} × ${c.d}`,a.q,a.stored,'attn','d²')}${weight('WK',`${c.d} × ${a.dkv}`,a.key,a.stored,'attn',formula.kv)}${weight('WV',`${c.d} × ${a.dkv}`,a.value,a.stored,'attn',formula.kv)}
 <div class="bv-no-weights">QKᵀ → scale → softmax → × V<small>0 learned parameters here · still compute work</small></div>${weight('WO',`${c.d} × ${c.d}`,a.o,a.stored,'attn','d²')}<p class="bv-step">+ input residual →</p></article>
 <span class="bv-arrow">→</span><article class="bv-module bv-ffn"><h3>${names[key]}</h3><p class="bv-total">${compact(a.ffnStored+a.router)} <code class="bv-total-formula">= ${formula.ffnTotal}</code><small>${(100*(a.ffnStored+a.router)/a.stored).toFixed(1)}% of block</small></p>
 ${a.moe?`<div class="bv-router">${weight('Router',`${c.d} × ${c.experts}`,a.router,a.stored,'router','dE')}<small>Choose ${c.k} of ${c.experts} experts per token</small></div>`:''}
 <p class="bv-step">Norm → ${a.gated?'two parallel expansions':'expansion'}</p>
 ${weight(a.gated?'W₁ → SiLU':'W₁ → GELU',`${c.d} × ${a.f}${a.moe?` × ${expertCount} experts`:''}`,c.d*a.f*expertCount,a.stored,'ffn',formula.projection)}
 ${a.gated?weight('W₃',`${c.d} × ${a.f}${a.moe?` × ${expertCount} experts`:''}`,c.d*a.f*expertCount,a.stored,'ffn',formula.projection)+'<div class="bv-no-weights">SiLU(xW₁) ⊙ xW₃<small>Featurewise gate · no extra weight matrix</small></div>':''}
 ${weight('W₂ · contract',`${a.f} × ${c.d}${a.moe?` × ${expertCount} experts`:''}`,c.d*a.f*expertCount,a.stored,'ffn',formula.projection)}
 ${a.moe?`<div class="bv-experts" aria-label="${c.experts} experts; ${c.k} active for one token">${Array.from({length:Math.min(c.experts,16)},(_,i)=>`<span class="${i<c.k?'is-used':''}">E${i+1}</span>`).join('')}${c.experts>16?`<small>+ ${c.experts-16} more</small>`:''}</div><p class="small">Illustrative selection; ${compact(a.ffnActive)} = k × 3df expert weights used per token. Including the router: ${compact(a.ffnActive+a.router)} = ${formula.ffnActive}. All ${c.experts} sets are stored.</p>`:''}
 <p class="bv-step">${a.moe?'Weighted sum → ':''}+ residual →</p></article><span class="bv-arrow">→</span><span class="bv-port">Token<br><b>h′</b></span></div>
 <p class="bv-block-equation"><strong>Block totals:</strong> stored = <code>${formula.stored}</code> = ${count(a.stored)}; active/token = <code>${formula.active}</code> = ${count(a.active)}.</p>
 <p class="small">Cards show structure, not physical size. Every thin meter uses the same denominator: total stored parameters in this block. Counts omit norms, biases and embeddings; full assumptions remain below.</p>`;
}
export function computeDiagram(b){
 const total=b.projection+b.mixing+b.ffn+b.router,c=b.c;
 return `<div class="bv-caption"><strong>Same transformer · different work with more tokens</strong><span>Same-width SwiGLU MoE · B=${c.batch}, n=${count(c.n)} · full-square forward</span></div><div class="bv-compute">
 <article class="bv-module"><h3>Self-attention</h3><div class="bv-work-node"><b>Q/K/V + output projections</b><strong>${work(b.projection)}</strong>${meter(b.projection,total,'attn')}<small>2Bn × projection parameters</small></div><div class="bv-arrow">↓</div><div class="bv-work-node"><b>QKᵀ + attention_weights × V</b><strong>${work(b.mixing)}</strong>${meter(b.mixing,total,'attn')}<small>4Bn²d · grows quadratically with n</small></div></article><span class="bv-arrow">→</span>
 <article class="bv-module bv-ffn"><h3>Router + selected MLPs</h3><div class="bv-work-node"><b>Router projection</b><strong>${work(b.router)}</strong>${meter(b.router,total,'router')}<small>2Bn × dE</small></div><div class="bv-arrow">↓</div><div class="bv-work-node"><b>${c.k} experts per token</b><strong>${work(b.ffn)}</strong>${meter(b.ffn,total,'ffn')}<small>2Bn × k × 3df · linear in n</small></div></article></div>
 <div class="bv-memory"><article><span>Weights · all ${c.layers} blocks</span><strong>${memory(b.variants.moe.weightBytes)}</strong><small>Independent of context length</small></article><article><span>Decoder KV cache · all blocks</span><strong>${memory(b.cacheBytes)}</strong><small>K/V reused at the next decode step · linear in n</small></article><article><span>Naïve attention score buffer · one block</span><strong>${memory(b.naiveScoreBytes)}</strong><small>Quadratic in n · FlashAttention avoids materializing this full buffer</small></article></div><p class="small">FLOPs above are per block; memory boxes are separate allocations with their scope labeled. They are not comparable bars or a peak-memory total. Encoder inference has no autoregressive KV cache.</p>`;
}
export function gpuDiagram(b){const g=b.gemm,max=Math.max(g.computeMs,g.memoryMs);return `<div class="bv-caption"><strong>One expansion operation on a GPU</strong><span>X[${count(g.M)}, ${b.c.d}] × W[${b.c.d}, ${b.c.f}] → Y[${count(g.M)}, ${b.c.f}]</span></div>
 <div class="bv-gpu"><article><h3>Device memory</h3><div class="bv-tiles"><span>X</span><span>W</span></div><strong>${memory(g.bytes)}</strong><small>Ideal minimum reads + output write</small></article><span class="bv-arrow">⇄</span><article><h3>On-chip tile reuse</h3><div class="bv-tile-grid" aria-hidden="true">${'<i></i>'.repeat(12)}</div><strong>${g.intensity.toFixed(1)} FLOPs/byte</strong><small>More token rows reuse W</small></article><span class="bv-arrow">→</span><article><h3>Matrix operations</h3><div class="bv-tiles"><span>×</span><span>+</span></div><strong>${work(g.flops)}</strong><small>Supported Tensor Core kernels</small></article></div>
 <div class="bv-time"><h3>Which resource sets the ideal lower bound?</h3><div><span>Compute</span>${meter(g.computeMs,max,'ffn')}<strong>${g.computeMs.toPrecision(3)} ms</strong></div><div><span>Memory traffic</span>${meter(g.memoryMs,max,'attn')}<strong>${g.memoryMs.toPrecision(3)} ms</strong></div><p><b>${g.bound==='compute'?'Compute':'Memory bandwidth'}</b> is the larger lower bound. Hardware ratio: ${g.ridge.toFixed(1)} FLOPs/byte. Actual runtime will be higher; this is not a GPU benchmark.</p></div>`;}
export function activateBudgetTab(root,id,{focus=false}={}){
 const tabs=[...root.querySelectorAll('[data-budget-tab]')],chosen=tabs.find(t=>t.dataset.budgetTab===id);if(!chosen)return;
 tabs.forEach(tab=>{const on=tab===chosen;tab.setAttribute('aria-selected',String(on));tab.tabIndex=on?0:-1;const panel=root.querySelector('#'+tab.getAttribute('aria-controls'));panel.hidden=!on;});if(focus)chosen.focus();
}
export function mountBudgetTabs(root){
 const click=e=>{const tab=e.target.closest('[data-budget-tab]');if(tab)activateBudgetTab(root,tab.dataset.budgetTab);};
 const key=e=>{const tab=e.target.closest('[data-budget-tab]');if(!tab||!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...root.querySelectorAll('[data-budget-tab]')],i=tabs.indexOf(tab),next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;activateBudgetTab(root,tabs[next].dataset.budgetTab,{focus:true});};
 root.addEventListener('click',click);root.addEventListener('keydown',key);return ()=>{root.removeEventListener('click',click);root.removeEventListener('keydown',key);};
}
