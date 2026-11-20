// Parameters page, 03 Optimizations: the KV cache, and grouped-query attention (GQA).
// Both labs are drawings plus counting; no model is run.
import {cacheStats, headGroups, kvHeadChoices} from './gqa.mjs';

const $ = s => document.querySelector(s), esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const WORDS = ['The', 'little', 'robot', 'reads', 'a', 'book'];
const state = {step: 4, kv: 2};

/* ---------- KV cache: what is computed now, what is reused ---------- */
function cacheFigure() {
 const s = state.step, y = i => 46 + 34 * i, chip = (cls, x, i, t) => `<rect class="kc-chip ${cls}" x="${x}" y="${y(i) - 10}" width="46" height="20" rx="5"/><text class="kc-chip-t ${cls}" x="${x + 23}" y="${y(i) + 4}" text-anchor="middle">${t}</text>`;
 let rows = '', lines = '';
 for (let i = 0; i < s; i++) {
  const now = i === s - 1;
  rows += `<text class="kc-word${now ? ' now' : ''}" x="150" y="${y(i) + 4}" text-anchor="end">${esc(WORDS[i])}</text>`
   + chip(now ? 'q now' : 'q gone', 190, i, 'Q') + chip(now ? 'k now' : 'k kept', 320, i, 'K') + chip(now ? 'v now' : 'v kept', 372, i, 'V');
  lines += `<path class="kc-line" d="M236 ${y(s - 1)}C280 ${y(s - 1)} 276 ${y(i)} 320 ${y(i)}"/>`;
 }
 const frame = s > 1 ? `<rect class="kc-frame" x="312" y="${y(0) - 18}" width="114" height="${34 * (s - 1) + 2}" rx="9"/><text class="kc-cap" x="434" y="${y(0) + 4}">KV CACHE</text><text class="kc-cap soft" x="434" y="${y(0) + 18}">kept from earlier steps</text>` : '';
 return `<svg viewBox="0 0 580 262" role="img" aria-label="Writing token ${s}: its Query, Key and Value are computed now; the Keys and Values of the ${s - 1} earlier tokens come from the cache">
<text class="kc-head q" x="213" y="16" text-anchor="middle">Query</text><text class="kc-head k" x="343" y="16" text-anchor="middle">Key</text><text class="kc-head v" x="395" y="16" text-anchor="middle">Value</text>
${frame}${lines}${rows}<text class="kc-cap now" x="434" y="${y(s - 1) + 4}">NEW · computed now</text></svg>`;
}

function renderCache() {
 const s = state.step, word = WORDS[s - 1], without = s * (s + 1) / 2;
 $('#kc-figure').innerHTML = cacheFigure();
 $('#kc-step').textContent = `Token ${s} of ${WORDS.length}`;
 $('#kc-back').disabled = s <= 1; $('#kc-next').disabled = s >= WORDS.length;
 const bar = (label, v, cls) => `<div class="kc-bar"><span>${label}</span><div class="kc-track"><i class="${cls}" style="width:${v / 21 * 100}%"></i></div><strong>${v}</strong></div>`;
 $('#kc-counts').innerHTML = `<p class="pf-label">TOKENS SENT THROUGH THE WEIGHTS SO FAR</p>${bar('With the cache', s, 'with')}${bar('Without it', without, 'without')}`;
 $('#kc-result').innerHTML = s === 1 ? `“${esc(word)}” is the first token. Its Query, Key and Value are computed, and its Key and Value go into the cache.`
  : `“${esc(word)}” is the only token that goes through the weights. Its Query is compared with its own Key and the ${s - 1} cached Key${s === 2 ? '' : 's'}, then its Key and Value join the cache. Without a cache, all ${s} tokens would be computed again.`;
}

/* ---------- GQA: query heads sharing Key and Value heads ---------- */
const Q_HEADS = 8, TOKENS = 8192;
function gqaFigure() {
 const groups = headGroups(Q_HEADS, state.kv), qx = i => 44 + 62 * i;
 let s = '';
 for (const g of groups) {
  const cx = g.queryHeads.reduce((t, h) => t + qx(h - 1) + 20, 0) / g.queryHeads.length;
  for (const h of g.queryHeads) s += `<path class="gq-line" d="M${qx(h - 1) + 20} 62C${qx(h - 1) + 20} 96 ${cx} 92 ${cx} 124"/>`;
  s += `<rect class="gq-chip k" x="${cx - 27}" y="124" width="26" height="22" rx="5"/><text class="gq-t" x="${cx - 14}" y="139" text-anchor="middle">K${g.kvHead}</text><rect class="gq-chip v" x="${cx + 1}" y="124" width="26" height="22" rx="5"/><text class="gq-t" x="${cx + 14}" y="139" text-anchor="middle">V${g.kvHead}</text>`;
 }
 for (let i = 0; i < Q_HEADS; i++) s += `<rect class="gq-chip q" x="${qx(i)}" y="40" width="40" height="22" rx="5"/><text class="gq-t" x="${qx(i) + 20}" y="55" text-anchor="middle">Q${i + 1}</text>`;
 return `<svg viewBox="0 0 560 170" role="img" aria-label="${Q_HEADS} query heads sharing ${state.kv} key and value head${state.kv === 1 ? '' : 's'}">
<text class="gq-cap" x="44" y="24">${Q_HEADS} QUERY HEADS · always all of them</text><text class="gq-cap" x="44" y="166">${state.kv} KEY AND VALUE HEAD${state.kv === 1 ? '' : 'S'} · this is what the cache keeps</text>${s}</svg>`;
}

function renderGqa() {
 const r = cacheStats({queryHeads: Q_HEADS, kvHeads: state.kv, tokens: TOKENS}), mib = v => `${v / 2 ** 20} MiB`, pct = `${r.share * 100}%`;
 const name = r.kind === 'MHA' ? 'Multi-head attention' : r.kind === 'MQA' ? 'Multi-query attention (MQA)' : 'Grouped-query attention (GQA)';
 $('#gq-figure').innerHTML = gqaFigure();
 document.querySelectorAll('#gq-choices button').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.kv) === state.kv)));
 $('#gq-kept').innerHTML = `<p class="pf-label">KEPT IN THE CACHE FOR EACH TOKEN</p><div class="gq-slots">${Array.from({length: Q_HEADS}, (_, i) => `<i class="${i < state.kv ? 'on' : ''}"><b></b><b></b></i>`).join('')}</div>
<div class="kc-bar"><span>Cache size against ordinary multi-head attention</span><div class="kc-track"><i class="with" style="width:${r.share * 100}%"></i></div><strong>${pct}</strong></div>
<p class="gq-size">One layer, ${TOKENS.toLocaleString('en-US')} tokens: <b>${mib(r.bytes)}</b>${r.kind === 'MHA' ? '' : ` instead of ${mib(r.mhaBytes)}`}</p>`;
 $('#gq-result').textContent = r.kind === 'MHA' ? `${name}: every Query head has its own Key and Value head. Nothing is shared, so this is the full-size cache.`
  : r.kind === 'MQA' ? `${name}: all ${Q_HEADS} Query heads read the same Key and Value head. The cache is ${r.reduction} times smaller, but one shared view of the tokens usually costs some quality.`
  : `${name}: each Key and Value head is shared by ${r.groupSize} Query heads. The cache is ${r.reduction} times smaller, and every Query head still decides for itself which tokens to look at.`;
}

function mount() {
 if ($('#kc-lab')) {
  renderCache();
  $('#kc-lab').addEventListener('click', e => { const b = e.target.closest('button[data-step]'); if (!b) return; state.step = Math.min(WORDS.length, Math.max(1, state.step + Number(b.dataset.step))); renderCache(); });
 }
 if ($('#gq-lab')) {
  const labels = {[Q_HEADS]: 'its own (multi-head)', 1: 'all share one (MQA)'};
  $('#gq-choices').innerHTML = kvHeadChoices(Q_HEADS).reverse().map(k => `<button type="button" data-kv="${k}">${k}<small>${labels[k] || `groups of ${Q_HEADS / k}`}</small></button>`).join('');
  renderGqa();
  $('#gq-choices').addEventListener('click', e => { const b = e.target.closest('button[data-kv]'); if (!b) return; state.kv = Number(b.dataset.kv); renderGqa(); });
 }
}
mount();
