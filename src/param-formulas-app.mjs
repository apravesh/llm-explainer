import {PF_DEFAULTS, PF_CATEGORIES, parameterFormulas, compactCount} from './param-formulas.mjs';

const root = document.getElementById('formula-models');
const state = {...PF_DEFAULTS};
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
let focusCat = null;

// Linked highlighting: hovering any part of one category dims every other category.
function applyFocus() {
 root.classList.toggle('pf-focusing', !!focusCat);
 root.querySelectorAll('[data-cat]').forEach(el => {
  const match = el.dataset.cat === focusCat || !!el.querySelector(`[data-cat="${focusCat}"]`);
  el.classList.toggle('pf-dim', !!focusCat && !match);
  el.classList.toggle('pf-hit', !!focusCat && el.dataset.cat === focusCat);
 });
}
function setFocus(cat) { if (cat === focusCat) return; focusCat = cat; applyFocus(); }
const pct = (a, b) => b ? a / b * 100 : 0;
const pctText = v => v >= 10 ? `${v.toFixed(0)}%` : v >= 1 ? `${v.toFixed(1)}%` : v > 0 ? '<1%' : '0%';

// A weight matrix drawn to scale: each grid cell is one d × d square (d² weights).
const tile = (w, h, label, cls, title = label) => `<span class="pf-tile ${cls}" style="--w:${w};--h:${h}" title="${title}"><em>${label}</em></span>`;

function attentionTiles() {
 return `<div class="pf-qkvo">${['Q', 'K', 'V', 'O'].map(n => tile(1, 1, `W<sub>${n}</sub>`, 'attn', `W_${n}: d × d`)).join('')}</div>`;
}
function mlpTiles(cls = 'mlp') {
 return `<div class="pf-mlp-tiles">${tile(4, 1, '', cls, 'W_in: d × 4d')}${tile(4, 1, '', cls, 'W_out: 4d × d')}</div><span class="pf-caption">W<sub>in</sub> d×4d · W<sub>out</sub> 4d×d</span>`;
}

function blockDiagram(r) {
 const {config: c} = r;
 let ffn;
 if (c.model === 'moe') {
  const experts = Array.from({length: c.E}, (_, i) => `<span class="pf-expert${i < c.k ? ' on' : ''}" title="Expert ${i + 1}: 8d²${i < c.k ? ' · active for this token' : ' · stored, idle for this token'}"><i></i><i></i></span>`).join('');
  ffn = `<div data-cat="mlp" class="pf-node mlp moe"><b>Router + ${c.E} experts</b>
   <div class="pf-router" data-cat="router">${tile(Math.max(.25, c.E / 16), 1, '', 'router', 'Router W_r: d × E')}<small>Router d × E · tiny</small></div>
   <div class="pf-experts" style="--cols:${Math.min(c.E, 4)}">${experts}</div>
   <code>E · 8d² stored</code><code class="pf-active">k · 8d² run per token</code></div>`;
 } else ffn = `<div data-cat="mlp" class="pf-node mlp"><b>MLP</b>${mlpTiles()}<code>2 · d · 4d = 8d²</code></div>`;
 return `<div class="pf-block"><span class="pf-repeat">× L = ${c.L} blocks</span>
  <div class="pf-row">
   <div data-cat="norm" class="pf-node norm"><b>Norm</b>${tile(1, .14, '', 'norm', 'gain vector: d')}<code>d</code></div>
   <div data-cat="attn" class="pf-node attn"><b>Attention</b>${attentionTiles()}<code>4 · d²</code></div>
   <div data-cat="norm" class="pf-node norm"><b>Norm</b>${tile(1, .14, '', 'norm', 'gain vector: d')}<code>d</code></div>
   ${ffn}
  </div>
  <p class="pf-residual">Residual additions and activations have no weights.</p>
 </div>`;
}

function embeddingNode(c, title, formula, note) {
 const squares = c.V / c.d, shown = Math.min(squares, 9);
 return `<div data-cat="embed" class="pf-node embed wide"><b>${title}</b><div class="pf-strip">${tile(shown, 1, '', 'embed', `${formula}: V × d = ${squares.toFixed(1)} d×d squares`)}${squares > shown ? '<span class="pf-more">…</span>' : ''}</div><code>${formula}</code><small>${note}</small></div>`;
}

function keyChip(c) {
 return `<div class="pf-key">${tile(1, 1, '', 'key', 'one d × d matrix')}<span>= one d × d matrix = d² = <b>${(c.d * c.d).toLocaleString()}</b> weights</span></div>`;
}

function diagram(r) {
 const c = r.config, down = '<div class="pf-down" aria-hidden="true">↓</div>';
 if (c.model === 'vision') {
  const patchSq = 3 * c.p * c.p / c.d;
  return `<div data-cat="io" class="pf-node io"><b>Image → ${c.p}×${c.p} RGB patches</b><small>each patch is a vector of 3p² = ${3 * c.p * c.p} numbers</small></div>${down}
  <div data-cat="vision" class="pf-node vision wide"><b>Patch embedding</b><div class="pf-strip">${tile(Math.min(patchSq, 9), 1, '', 'vision', `W_patch: 3p² × d = ${patchSq.toFixed(2)} d×d squares`)}</div><code>3p² · d</code><small>2D RoPE positions add no weights</small></div>${down}
  ${blockDiagram(r)}${down}
  <div data-cat="vision" class="pf-node vision wide"><b>Projector → LLM width D = ${c.Dmul}d</b><div class="pf-proj">${tile(c.Dmul, 1, '', 'vision', 'd × D')}${tile(c.Dmul, c.Dmul, '', 'vision', 'D × D')}</div><span class="pf-caption">W₁ d×D · W₂ D×D</span><code>d·D + D·D = ${c.Dmul}d² + ${c.Dmul * c.Dmul}d²</code></div>${down}
  <div data-cat="io" class="pf-node io"><b>Image tokens for the language model</b></div>`;
 }
 return `${embeddingNode(c, 'Token embedding', 'V · d', `one d-vector per vocabulary entry · V/d = ${(c.V / c.d).toFixed(1)} squares`)}${down}
 ${blockDiagram(r)}${down}
 <div data-cat="norm" class="pf-node norm wide"><b>Final norm</b><code>d</code></div>${down}
 ${c.tied ? `<div data-cat="embed" class="pf-node embed wide tied"><b>LM head</b><code>tied: reuses the V · d embedding</code></div>` : embeddingNode(c, 'LM head', 'd · V', 'scores every vocabulary entry')}`;
}

function bar(r, which) {
 const total = which === 'active' ? r.active : r.stored;
 const segs = Object.entries(r.byCat).filter(([, v]) => v[which] > 0).map(([cat, v]) => {
  const share = pct(v[which], total);
  return `<span class="pf-seg ${cat}" data-cat="${cat}" style="flex-grow:${v[which]}" title="${PF_CATEGORIES[cat].label}: ${compactCount(v[which])} (${pctText(share)})">${share >= 7 ? `${PF_CATEGORIES[cat].short} ${pctText(share)}` : ''}</span>`;
 }).join('');
 return `<div class="pf-bar-row"><span>${which === 'active' ? 'Used per token' : r.config.model === 'moe' ? 'Stored' : 'Whole model'}<b>${compactCount(total)}</b></span><div class="pf-bar">${segs}</div></div>`;
}

function blockSplit(r) {
 const c = r.config;
 if (c.model === 'moe') {
  const stored = 4 + 8 * c.E, active = 4 + 8 * c.k;
  return `<div class="pf-split"><div><span class="pf-eq">4d² + <i>E</i>·8d² = ${stored}d²</span><small>stored per block</small></div><div><span class="pf-eq">4d² + <i>k</i>·8d² = ${active}d²</span><small>run per token</small></div></div>
  <p class="pf-insight">With E = ${c.E} and k = ${c.k}, a block <strong>stores ${(stored / 12).toFixed(1)}×</strong> the weights of a dense block but each token <strong>runs only ${(active / 12).toFixed(1)}×</strong>. Attention falls to ${pctText(pct(4, stored))} of stored block weights.</p>`;
 }
 return `<div class="pf-split"><div><span class="pf-eq"><span class="c-attn">4d²</span> + <span class="c-mlp">8d²</span> = 12d²</span><small>one block</small></div><div class="pf-mini-bar"><span class="pf-seg attn" data-cat="attn" style="flex-grow:4">1/3 attention</span><span class="pf-seg mlp" data-cat="mlp" style="flex-grow:8">2/3 MLP</span></div></div>`;
}

function insight(r) {
 const c = r.config;
 if (c.model === 'vision') {
  const proj = r.byCat.vision.stored, blocks = r.byCat.attn.stored + r.byCat.mlp.stored;
  return `No vocabulary, so no V · d tables. The patch embedding is small (3p² · d). The projector, ${c.Dmul}d² + ${c.Dmul * c.Dmul}d² = ${c.Dmul + c.Dmul * c.Dmul}d², weighs as much as ${((c.Dmul + c.Dmul * c.Dmul) / 12).toFixed(1)} encoder blocks — ${pctText(pct(proj, r.stored))} of this encoder. Blocks hold ${pctText(pct(blocks, r.stored))}.`;
 }
 const emb = r.byCat.embed.stored;
 return `Embeddings scale with V · d, blocks with L · d². Here V · d${c.tied ? '' : ' (×2)'} is ${pctText(pct(emb, r.stored))} of the ${c.model === 'moe' ? 'stored ' : ''}model. Shrink d or L, or grow V, and the vocabulary tables take over — that is why small models often tie the embedding and LM head.`;
}

function table(r) {
 return `<table class="pf-table"><thead><tr><th>Part</th><th>Formula</th><th>Count</th><th>Share</th></tr></thead><tbody>${r.parts.map(x => `<tr data-cat="${x.cat}" tabindex="0"><th><i class="pf-dot ${x.cat}"></i>${x.label}</th><td><code>${x.formula}</code>${x.shorthand !== x.formula ? `<small>${x.shorthand}</small>` : ''}</td><td>${compactCount(x.count)}${x.active !== x.count ? `<small>${compactCount(x.active)} run</small>` : ''}</td><td>${pctText(pct(x.count, r.stored))}</td></tr>`).join('')}</tbody></table>`;
}

function controls() {
 const c = state, opt = (vals, cur, fmt = v => v) => vals.map(v => `<option value="${v}"${String(v) === String(cur) ? ' selected' : ''}>${fmt(v)}</option>`).join('');
 const llm = c.model !== 'vision';
 return `<label>Width d<select data-pf="d">${opt([64, 128, 256, 512, 1024], c.d)}</select></label>
 <label>Blocks L <output>${c.L}</output><input data-pf="L" type="range" min="1" max="32" value="${c.L}"></label>
 ${llm ? `<label>Vocabulary V<select data-pf="V">${opt([1000, 8000, 32000], c.V, v => v.toLocaleString())}</select></label><label class="pf-check"><input type="checkbox" data-pf="tied"${c.tied ? ' checked' : ''}> Tie embedding &amp; LM head</label>` : `<label>LLM width D<select data-pf="Dmul">${opt([2, 4, 5], c.Dmul, v => `${v}d`)}</select></label>`}
 ${c.model === 'moe' ? `<label>Experts E <output>${c.E}</output><input data-pf="E" type="range" min="2" max="16" value="${c.E}"></label><label>Active k <output>${c.k}</output><input data-pf="k" type="range" min="1" max="${Math.min(4, c.E)}" value="${Math.min(c.k, c.E)}"></label>` : ''}`;
}

function render(withControls = true) {
 const r = parameterFormulas(state);
 root.querySelectorAll('[data-pf-model]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.pfModel === state.model)));
 if (withControls) root.querySelector('#pf-controls').innerHTML = controls();
 const diagramEl = root.querySelector('#pf-diagram');
 diagramEl.innerHTML = keyChip(r.config) + diagram(r);
 if (withControls && !reduced.matches) { diagramEl.classList.remove('pf-enter'); void diagramEl.offsetWidth; diagramEl.classList.add('pf-enter'); }
 applyFocus();
 root.querySelector('#pf-split').innerHTML = blockSplit(r);
 root.querySelector('#pf-bars').innerHTML = bar(r, 'stored') + (state.model === 'moe' ? bar(r, 'active') : '');
 root.querySelector('#pf-insight').textContent = insight(r);
 root.querySelector('#pf-table').innerHTML = table(r);
}

if (root) {
 root.addEventListener('click', e => { const b = e.target.closest('[data-pf-model]'); if (b) { state.model = b.dataset.pfModel; render(); } });
 const catOf = e => e.target.closest?.('[data-cat]')?.dataset.cat || null;
 root.addEventListener('pointerover', e => setFocus(catOf(e)));
 root.addEventListener('pointerleave', () => setFocus(null));
 root.addEventListener('focusin', e => setFocus(catOf(e)));
 root.addEventListener('focusout', e => { if (!root.contains(e.relatedTarget)) setFocus(null); });
 root.addEventListener('input', e => {
  const key = e.target.dataset?.pf; if (!key) return;
  state[key] = e.target.type === 'checkbox' ? e.target.checked : Number(e.target.value);
  if (e.target.type === 'range') e.target.previousElementSibling.textContent = e.target.value;
  if (key === 'E') {
   const kInput = root.querySelector('[data-pf="k"]');
   state.k = Math.min(state.k, state.E, 4);
   kInput.max = Math.min(4, state.E); kInput.value = state.k; kInput.previousElementSibling.textContent = state.k;
  }
  render(false);
 });
 render();
}
