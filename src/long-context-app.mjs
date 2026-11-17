// Parameters page, 03 Optimizations: local and global attention layers.
import {canSee, pairs, compare, layerKinds, PATTERNS} from './long-context.mjs';

const $ = s => document.querySelector(s), nf = v => Math.round(v).toLocaleString('en-US');
const big = v => v >= 1e9 ? `${(v / 1e9).toFixed(2)} billion` : v >= 1e6 ? `${(v / 1e6).toFixed(1)} million` : nf(v);
const pct = v => v >= 0.995 ? '100%' : v < 0.01 ? `${(v * 100).toFixed(1)}%` : `${Math.round(v * 100)}%`;
const TOY = 16, state = {w: 4, row: null, n: 131072, real: 1024, pattern: 'mixed'};

function grid(w, title, sub) {
 const c = 15, pad = 22, size = TOY * c, cells = [];
 for (let i = 0; i < TOY; i++) for (let j = 0; j < TOY; j++) {
  const on = canSee(i, j, w), hot = state.row === i;
  cells.push(`<rect class="lc-cell${on ? ' on' : j > i ? ' future' : ''}${hot && on ? ' hot' : ''}" data-row="${i}" x="${pad + j * c}" y="${pad + i * c}" width="${c - 2}" height="${c - 2}" rx="2"/>`);
 }
 const count = state.row == null ? `${pairs(TOY, w)} pairs compared` : `token ${state.row + 1} looks at ${Math.min(state.row + 1, w ?? TOY)} token${Math.min(state.row + 1, w ?? TOY) === 1 ? '' : 's'}`;
 return `<figure class="lc-fig"><figcaption><b>${title}</b><span>${sub}</span></figcaption>
<svg viewBox="0 0 ${size + pad + 4} ${size + pad + 4}" role="img" aria-label="${title}: which earlier tokens each token can look at">
<text class="lc-axis" x="${pad + size / 2}" y="12" text-anchor="middle">token it looks at →</text>
<text class="lc-axis" transform="translate(12 ${pad + size / 2}) rotate(-90)" text-anchor="middle">token that looks →</text>${cells.join('')}</svg>
<p class="lc-count">${count}</p></figure>`;
}

function renderToy() {
 $('#lc-grids').innerHTML = grid(null, 'Global layer', 'every earlier token') + grid(state.w, 'Local layer', `only the last ${state.w} tokens`);
 $('#lc-w-out').textContent = state.w;
}

function renderReal() {
 const {n, real: w, pattern} = state, r = compare({n, w, pattern});
 const bar = (label, share, value, base, unit) => `<div class="lc-bar"><span>${label}</span><div class="lc-track"><i style="width:${Math.max(share * 100, 0.6)}%"></i></div><strong>${pct(share)}</strong><small>${big(value)} ${unit}, against ${big(base)} if every layer were global</small></div>`;
 $('#lc-stack').innerHTML = layerKinds(pattern).map((k, i) => `<i class="${k}" title="layer ${i + 1}: ${k}">${k === 'global' ? 'G' : 'L'}</i>`).join('') + '<span>first 12 layers</span>';
 $('#lc-bars').innerHTML = bar('Pairs compared, per layer', r.pairsShare, r.pairsPerLayer, r.pairsGlobal, 'pairs') + bar('Tokens kept in the KV cache, per layer', r.cacheShare, r.cachePerLayer, r.cacheGlobal, 'tokens');
 const p = PATTERNS[pattern];
 $('#lc-result').textContent = pattern === 'global' ? `Every layer compares all ${nf(n)} tokens with each other and keeps all of them in its cache. This is the baseline.`
  : pattern === 'local' ? (w >= n ? `The window of ${nf(w)} is as long as the prompt, so nothing is saved yet. Windows pay off once the prompt is longer than the window.` : `No layer looks further back than ${nf(w)} tokens, so the cost stops growing with n². A token ${nf(n)} positions back can still matter, but only by being passed along, layer after layer.`)
  : `Out of every ${p.local + p.global} layers, ${p.local} look back only ${nf(w)} tokens and 1 sees all ${nf(n)}. The model can still connect any two tokens directly in the global layers, for about ${pct(r.cacheShare)} of the cache and ${pct(r.pairsShare)} of the comparisons.`;
}

function mount() {
 const host = $('#lc-lab'); if (!host) return;
 renderToy(); renderReal();
 host.addEventListener('input', e => { if (e.target.id === 'lc-w') { state.w = Number(e.target.value); renderToy(); } });
 host.addEventListener('change', e => {
  if (e.target.id === 'lc-n') state.n = Number(e.target.value);
  else if (e.target.id === 'lc-real-w') state.real = Number(e.target.value);
  else if (e.target.name === 'lc-pattern') state.pattern = e.target.value;
  else return;
  renderReal();
 });
 const grids = $('#lc-grids');
 grids.addEventListener('pointerover', e => { const c = e.target.closest('[data-row]'); const row = c ? Number(c.dataset.row) : null; if (row !== state.row) { state.row = row; renderToy(); } });
 grids.addEventListener('pointerleave', () => { if (state.row != null) { state.row = null; renderToy(); } });
}
mount();
