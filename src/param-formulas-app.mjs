import {PF_DEFAULTS, PF_CATEGORIES, PF_PRESETS, parameterFormulas, workFormulas, compactCount, compactWork, unitText} from './param-formulas.mjs';

const root = document.getElementById('formula-models');
const state = {...PF_DEFAULTS, ...PF_PRESETS.gpt2.config};
let preset = 'gpt2';   // start from the model the Transformer page follows
let current = null;
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

// The architecture, drawn in the Transformer page's visual language. Every part that holds weights is a zone
// (data-cat for linked highlighting, data-part for the formula card) with its formula and count written under it.
const svgEsc = s => String(s).replace(/[&<>"]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[ch]));
const curve = (x1, y1, x2, y2) => { const m = (x1 + x2) / 2; return `M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}`; };
const part = (r, key) => r.parts.find(x => x.key === key);
const sumParts = (r, keys) => keys.map(k => part(r, k)).filter(Boolean).reduce((n, x) => n + x.count, 0);
function badge(r, x, y, keys, formula, cat, w = 150) {
 const count = sumParts(r, keys);
 return `<g class="pf-badge" data-cat="${cat}" data-part="${keys[0]}" tabindex="0"><rect x="${x - w / 2}" y="${y}" width="${w}" height="40" rx="8"/><text class="pf-b-f" x="${x}" y="${y + 16}" text-anchor="middle">${svgEsc(formula)}</text><text class="pf-b-n" x="${x}" y="${y + 32}" text-anchor="middle">${count ? `${compactCount(count)} · ${pctText(pct(count, r.stored))}` : 'no extra weights'}</text></g>`;
}
function archFlow(r, w = null) {
 const c = r.config, vision = c.model === 'vision', moe = c.model === 'moe', rows = w ? [128, 196, 264] : [196], mi = i => rows.length === 1 ? 1 : i, W = 1180, H = w && moe ? 438 : 392, by = 338;
 const names = vision ? ['patch', 'patch', 'patch'] : ['token', 'token', 'token'];
 const single = rows.length === 1;
 const wpart = key => w.parts.find(x => x.key === key);
 const wbadge = (x, y, key, cat, width, zone) => { const p = wpart(key); return `<g class="pf-badge${cat === 'mix' ? ' mix' : ''}" data-cat="${cat}" data-part="${zone}" tabindex="0"><rect x="${x - width / 2}" y="${y}" width="${width}" height="40" rx="8"/><text class="pf-b-f" x="${x}" y="${y + 16}" text-anchor="middle">${svgEsc(p.formula)}</text><text class="pf-b-n" x="${x}" y="${y + 32}" text-anchor="middle">${p.flops ? `${compactWork(p.flops)} · ${pctText(pct(p.flops, w.total))}` : 'almost no work'}</text></g>`; };
 let s = `<svg id="${w ? 'pw-flow' : 'pf-flow'}" class="pf-flow" viewBox="0 0 ${W} ${H}" role="img" aria-label="${vision ? 'A vision encoder' : moe ? 'A mixture-of-experts language model' : 'A language model'} drawn as on the Transformer page, with the number of weights in each part">`;
 // ----- embedding
 const embKeys = vision ? ['patch'] : ['embed', 'positions'], embCat = vision ? 'vision' : 'embed';
 s += `<g class="pf-zone" data-cat="${embCat}" data-part="${embKeys[0]}"><text class="pf-col" x="24" y="40">${vision ? 'Patch embedding' : 'Embedding'}</text>`;
 rows.forEach((y, i) => { s += `${vision ? `<rect class="pf-patch" x="62" y="${y - 12}" width="24" height="24" rx="2"/>` : `<text class="pf-tok" x="92" y="${y + 4}" text-anchor="end">${names[i]}</text>`}<path class="pf-wire" d="M${vision ? 90 : 98},${y} H128"/><rect class="pf-vec" x="130" y="${y - 15}" width="12" height="30" rx="2"/>`; });
 s += `<rect class="pf-hitbox" x="14" y="60" width="150" height="250"/></g>`;
 s += w ? wbadge(88, by, vision ? 'patch' : 'embed', embCat, 150, embKeys[0]) : badge(r, 88, by, embKeys, vision ? '3p² · d' : c.pos > 0 ? 'V · d + n_ctx · d' : 'V · d', embCat, 150);
 // ----- block frame
 s += `<rect class="pf-frame" x="196" y="62" width="712" height="256" rx="10"/><rect class="pf-chip-bg" x="214" y="51" width="200" height="22"/><text class="pf-chip" x="224" y="66">TRANSFORMER BLOCK × ${c.L}</text>`;
 // ----- attention: Q K V squares, the head card with its dot matrix, and Out
 s += `<g class="pf-zone" data-cat="attn" data-part="attn">`;
 // One Q, K, V block (the three weight matrices), as on the Transformer page: rows squeeze into it and fan out again.
 const cy = (rows[0] + rows.at(-1)) / 2, sq = y => cy + (y - cy) * 0.5, qt = sq(rows[0]) - 15, qb = sq(rows.at(-1)) + 15;
 s += `<rect class="pf-wblock" x="224" y="${qt}" width="26" height="${qb - qt}" rx="6"/>`;
 [['q', 227], ['k', 234.5], ['v', 242]].forEach(([g, x]) => { s += `<rect class="pf-${g}" x="${x}" y="${qt + 3}" width="5" height="${qb - qt - 6}" rx="2.5"/><text class="pf-mini" x="${x + 2.5}" y="${qt - 5}" text-anchor="middle">${g.toUpperCase()}</text>`; });
 rows.forEach(y => { s += `<path class="pf-wire" d="${curve(142, y, 224, sq(y))}"/>${[['q', -6.5], ['k', 0], ['v', 6.5]].map(([g, dy]) => `<circle class="pf-${g}" cx="253" cy="${sq(y) + dy}" r="2"/>`).join('')}`; });
 s += `<rect class="pf-card" x="300" y="92" width="330" height="208" rx="6"/><rect class="pf-chip-bg" x="316" y="81" width="${c.kvRatio === 1 ? 226 : 286}" height="22"/><text class="pf-chip head" x="326" y="96">MULTI-HEAD SELF-ATTENTION${c.kvRatio === 1 ? '' : ' · GROUPED K/V'}</text>`;
 const mx = 444, my = 182, tp = 26;
 rows.forEach((y, i) => {
  const m = mi(i);
  s += `<path class="pf-wq" d="${curve(255, sq(y) - 6.5, 336, my + m * tp)} H${mx - 14}"/><path class="pf-wk" d="${curve(255, sq(y), 330, 122 + m * 8)} H${mx + m * tp} V${my - 14}"/><path class="pf-wv" d="${curve(255, sq(y) + 6.5, 336, 272 + m * 8)} H530 C566,${272 + m * 8} 566,${my + m * tp} 596,${my + m * tp}"/>`;
  s += `<rect class="pf-o" x="596" y="${my + m * tp - 2}" width="16" height="4" rx="1.5"/>`;
 });
 // the grid of attention scores is drawn the same in both views: it stands for "every token looks at every token"
 for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { const masked = !vision && j > i; s += `<circle class="pf-dot-cell${masked ? ' masked' : ''}" cx="${mx + j * tp}" cy="${my + i * tp}" r="8"${masked ? '' : ` fill-opacity="${[0.85, 0.35, 0.55][(i + j * 2) % 3]}"`}/>`; }
 s += `<text class="pf-lbl k" x="346" y="${single ? 122 : 114}">Key</text><text class="pf-lbl q" x="346" y="${my - 12}">Query</text><text class="pf-lbl v" x="346" y="262">Value</text><text class="pf-lbl o" x="592" y="${my - 16}">Out</text><text class="pf-sub" x="${mx - 12}" y="${my + 3 * tp - 2}">${vision ? 'ATTENTION · NO MASK' : 'ATTENTION'}</text>`;
 s += `<rect class="pf-hitbox" x="204" y="76" width="434" height="236"/></g>`;
 s += w ? wbadge(326, by, 'proj', 'attn', 196, 'attn') + wbadge(530, by, 'mix', 'mix', 190, 'attn') : badge(r, 420, by, ['attn'], c.kvRatio === 1 ? 'L · 4d²' : 'L · (2d² + 2·d·d_kv)', 'attn', 190);
 // ----- MLP, or router + experts
 const mlpCat = 'mlp';
 s += `<g class="pf-zone" data-cat="${mlpCat}" data-part="${moe ? 'experts' : 'mlp'}"><text class="pf-name" x="${moe ? 760 : 721}" y="${single ? rows[0] - 34 : 98}" text-anchor="middle">${moe ? `MoE layer · ${c.k} of ${c.E} experts run` : c.mats === 3 ? 'Gated MLP' : 'MLP'}</text>`;
 // The layer after attention is drawn once: its weights are shared, and every token's vector passes through the same ones.
 const top = rows[0] - 15, bot = rows.at(-1) + 15, t2 = top - 7, b2 = bot + 7;
 rows.forEach((y, i) => { s += `<path class="pf-wo" d="${curve(612, my + mi(i) * tp, 668, y)}"/><rect class="pf-vec in" x="668" y="${y - 15}" width="10" height="30" rx="2"/>`; });
 if (moe) {
  const shown = Math.min(c.E, 8), ex = 716, gap = 13;
  s += `<g data-cat="router" data-part="router"><rect class="pf-router" x="688" y="${top}" width="20" height="${bot - top}" rx="5"/><text class="pf-router-t" x="698" y="${(top + bot) / 2 + 3.5}" text-anchor="middle">R</text></g>`;
  for (let e = 0; e < shown; e++) s += `<rect class="pf-expert" x="${ex + e * gap}" y="${top}" width="10" height="${bot - top}" rx="3"/>`;
  rows.forEach((y, i) => {
   const on = new Set(Array.from({length: Math.min(c.k, shown)}, (_, n) => (i * 3 + n * 4 + 1) % shown));
   s += `<path class="pf-wire" d="M678,${y} H688"/>`;
   for (const e of on) s += `<rect class="pf-expert on" x="${ex + e * gap}" y="${y - 7}" width="10" height="14" rx="2"/>`;
   s += `<path class="pf-wire" d="M${ex + shown * gap - 2},${y} H824"/><circle class="pf-sum" cx="832" cy="${y}" r="8"/><text class="pf-sum-t" x="832" y="${y + 4}" text-anchor="middle">Σ</text><rect class="pf-vec out" x="848" y="${y - 15}" width="10" height="30" rx="2"/>`;
  });
  s += `<text class="pf-note" x="760" y="${b2 + 14}" text-anchor="middle">${w ? `one layer · each token runs its own ${c.k} experts` : `one router and ${c.E} experts, shared by every token`}</text>`;
 } else {
  const gated = c.mats === 3;
  const mt = sq(rows[0]) - 22, mb = sq(rows.at(-1)) + 22;
  s += `<path class="pf-band" d="M690,${mt + 12} L708,${mt} V${mb} L690,${mb - 12} Z"/><rect class="pf-wide" x="708" y="${mt}" width="26" height="${mb - mt}" rx="4"/>${gated ? `<rect class="pf-gate" x="721" y="${mt}" width="13" height="${mb - mt}" rx="4"/>` : ''}<path class="pf-band" d="M734,${mt} L752,${mt + 12} V${mb - 12} L734,${mb} Z"/>`;
  rows.forEach(y => { s += `<path class="pf-thru" d="${curve(678, y, 692, sq(y))} H750 ${curve(750, sq(y), 764, y).replace('M750,' + sq(y), '')}"/><rect class="pf-vec out" x="764" y="${y - 15}" width="10" height="30" rx="2"/>`; });
  s += `<text class="pf-note" x="721" y="${b2 + 14}" text-anchor="middle">${w ? `one MLP · runs once per ${vision ? 'patch' : 'token'}` : `one MLP · shared by every ${vision ? 'patch' : 'token'}`}</text>`;
 }
 s += `<rect class="pf-hitbox" x="652" y="84" width="${moe ? 214 : 132}" height="226"/></g>`;
 if (w) s += moe ? wbadge(694, by + 46, 'router', 'router', 130, 'router') + wbadge(762, by, 'experts', 'mlp', 196, 'experts') : wbadge(728, by, 'mlp', 'mlp', 176, 'mlp');
 else if (moe) s += badge(r, 700, by, ['router'], 'L · d·E', 'router', 96) + badge(r, 812, by, ['experts'], `L · E · ${c.mats}·d·f`, 'mlp', 118);
 else s += badge(r, 720, by, ['mlp'], `L · ${c.mats}·d·f`, 'mlp', 140);
 // ----- the blocks that follow
 const bx = moe ? 868 : 800;
 if (!moe) rows.forEach(y => { s += `<path class="pf-wire" d="M774,${y} H${bx}"/>`; for (let b = 0; b < 3; b++) s += `<rect class="pf-more" x="${bx + b * 12}" y="${y - 15}" width="7" height="30" rx="2" opacity="${0.9 - b * 0.25}"/>`; });
 if (!moe) s += `<text class="pf-sub" x="${bx}" y="${single ? rows[0] - 40 : 92}">${c.L - 1} MORE</text><text class="pf-sub" x="${bx}" y="${single ? rows[0] - 28 : 104}">BLOCKS</text>`;
 // ----- the end: LM head and probabilities, or the projector
 if (vision) {
  s += `<g class="pf-zone" data-cat="vision" data-part="proj"><text class="pf-col" x="962" y="40">Projector</text>`;
  s += `<path class="pf-band green" d="M960,${top} L990,${t2} V${b2} L960,${bot} Z"/><rect class="pf-projw" x="990" y="${t2}" width="14" height="${b2 - t2}" rx="4"/><text class="pf-note" x="988" y="${b2 + 14}" text-anchor="middle">${w ? 'runs once per patch' : 'one shared projector'}</text>`;
  rows.forEach(y => { s += `<path class="pf-wire" d="M${bx + 31},${y} H960"/><rect class="pf-llmvec" x="1010" y="${y - 20}" width="10" height="40" rx="3"/><path class="pf-wire" d="M1020,${y} H1040m-5,-4l5,4l-5,4"/>`; });
  s += `<rect class="pf-llm" x="1046" y="100" width="34" height="192" rx="8"/><text class="pf-llm-t" transform="translate(1067,196) rotate(90)" text-anchor="middle">LANGUAGE MODEL</text><rect class="pf-hitbox" x="944" y="60" width="150" height="250"/></g>`;
  s += w ? wbadge(1016, by, 'proj-out', 'vision', 190, 'proj') : badge(r, 1016, by, ['proj'], 'd·D + D·D', 'vision', 150);
 } else {
  const last = rows.at(-1), x0 = moe ? 866 : bx + 31, list = [96, 126, 156, 186, 216, 246, 276];
  s += `<g class="pf-zone${c.tied ? ' tied' : ''}" data-cat="embed" data-part="head"><text class="pf-col" x="1040" y="40">Probabilities</text>`;
  list.forEach((y, i) => { s += `<path class="pf-fan" d="M${x0},${last} C${x0 + 40},${last} ${x0 + 70},${y} 1010,${y} H1034" opacity="${i < 5 ? 1 : 0.9 - (i - 4) * 0.35}"/><rect class="pf-word" x="1040" y="${y - 5}" width="${[58, 44, 66, 38, 52, 46, 60][i]}" height="10" rx="3" opacity="${i < 5 ? 1 : 0.9 - (i - 4) * 0.35}"/><rect class="pf-pbar" x="1112" y="${y - 2}" width="${[46, 22, 12, 8, 5, 3, 2][i]}" height="4" rx="2" opacity="${i < 5 ? 1 : 0.9 - (i - 4) * 0.35}"/>`; });
  s += `<text class="pf-name" x="1060" y="312" text-anchor="middle">LM head${c.tied ? ' · shared with the embedding' : ''}</text><rect class="pf-hitbox" x="${x0 + 30}" y="60" width="${1166 - x0 - 30}" height="262"/></g>`;
  s += w ? wbadge(1060, by, 'head', 'embed', 190, 'head') : badge(r, 1060, by, ['head'], c.tied ? '0 · reuses V · d' : 'd · V', 'embed', 190);
 }
 return s + '</svg>';
}
// The formula card. Each weight matrix of the part is listed with its shape and size; the sizes are then
// added up for one block and multiplied by the number of blocks, so the total can be followed step by step.
function partCard(r, key) {
 const c = r.config, x = part(r, key); if (!x) return '';
 const n = v => Number(v).toLocaleString('en-US'), share = `${pctText(pct(x.count, r.stored))} of the model`;
 const row = (name, shape, a, b, note = '') => ({name, shape, a, b, count: a * b, note});
 const mlpRows = () => c.mats === 3 ? [row('W₁', 'f × d', c.f, c.d, 'widen'), row('W₃', 'f × d', c.f, c.d, 'the gate'), row('W₂', 'd × f', c.d, c.f, 'narrow')] : [row('W_in', 'f × d', c.f, c.d, 'widen'), row('W_out', 'd × f', c.d, c.f, 'narrow')];
 const spec = {
  embed: {rows: [row('W_E', 'V × d', c.V, c.d, 'token table')].concat(c.pos > 0 ? [row('W_P', 'n_ctx × d', c.pos, c.d, 'position table')] : []), say: c.pos > 0 ? 'One row of d numbers for every token in the vocabulary, and one for every position in the context.' : 'One row of d numbers for every token in the vocabulary.'},
  attn: {rows: [row('W_Q', 'd × d', c.d, c.d, 'query'), row('W_K', c.kvRatio === 1 ? 'd × d' : 'd_kv × d', c.dkv, c.d, 'key'), row('W_V', c.kvRatio === 1 ? 'd × d' : 'd_kv × d', c.dkv, c.d, 'value'), row('W_O', 'd × d', c.d, c.d, 'output')], per: 'block', times: [[c.L, 'blocks']], say: c.kvRatio === 1 ? 'Four matrices of the same size in every block.' : 'With grouped-query attention several query heads share one key/value head, so W_K and W_V are narrower.'},
  mlp: {rows: mlpRows(), per: 'block', times: [[c.L, 'blocks']], say: c.mats === 3 ? 'A gated (SwiGLU) MLP has three matrices: two widen d to f, one narrows it back.' : 'Two matrices: one widens d to f, one narrows it back.'},
  router: {rows: [row('W_r', 'E × d', c.E, c.d, 'scores the experts')], per: 'block', times: [[c.L, 'blocks']], say: 'One small matrix per block. It gives each token a score for every expert.'},
  experts: {rows: mlpRows(), per: 'expert', times: [[c.E, 'experts'], [c.L, 'blocks']], say: `Every expert is a complete MLP. Each token runs only ${c.k} of the ${c.E}, so ${compactCount(x.active)} of these weights are used per token.`},
  head: c.tied ? {rows: [], say: 'No extra weights. The last step reuses the token table W_E (V × d), read the other way round: each row times the final vector gives that token\'s score.'} : {rows: [row('W_out', 'V × d', c.V, c.d, 'one row per token')], say: 'A second table of the same shape as the embedding. Each row times the final vector gives that token\'s score.'},
  patch: {rows: [row('W_patch', 'd × 3p²', c.d, 3 * c.p * c.p, 'pixels → vector')], say: `A ${c.p} × ${c.p} patch has ${3 * c.p * c.p} colour values. One matrix turns them into a vector. There is no vocabulary table.`},
  proj: {rows: [row('W_a', 'D × d', c.D, c.d, 'to the LLM\'s width'), row('W_b', 'D × D', c.D, c.D, '')], say: 'Two layers that resize every patch vector to the width of the language model.'},
  norm: {rows: [], say: `L · 2d + d = ${c.L} × 2 × ${n(c.d)} + ${n(c.d)}. A handful of scaling numbers around each block; too few to see.`}
 }[key];
 const rows = spec.rows, per = rows.reduce((t, q) => t + q.count, 0);
 let h = `<h4>${svgEsc(x.label)}</h4>`;
 if (rows.length) {
  h += `<table class="pf-pop-t"><tbody>${rows.map(q => `<tr><th>${q.name}${q.note ? `<small>${svgEsc(q.note)}</small>` : ''}</th><td>${q.shape}</td><td>${n(q.a)} × ${n(q.b)}</td><td>${compactCount(q.count)}</td></tr>`).join('')}</tbody></table>`;
  const steps = [];
  if (rows.length > 1) steps.push(`<span>${rows.map(q => compactCount(q.count)).join(' + ')}</span><b>= ${compactCount(per)}</b><small>${spec.per ? `one ${spec.per}` : 'together'}</small>`);
  let run = per;
  for (const [k, what] of spec.times || []) { run *= k; steps.push(`<span>× ${n(k)} ${what}</span><b>= ${compactCount(run)}</b><small>${what === 'blocks' ? 'the whole model' : 'one block'}</small>`); }
  if (steps.length) h += `<div class="pf-pop-sum">${steps.map(t => `<div>${t}</div>`).join('')}</div>`;
 }
 return `${h}<p class="pf-pop-n"><b>${x.count ? compactCount(x.count) : '0'}</b> weights · ${share}</p><p>${svgEsc(spec.say)}</p>`;
}
// What the letters stand for, with this model's own values.
const work = {n: 1024, mode: 'read'};
function symbols(r) {
 const c = r.config, n = v => Number(v).toLocaleString('en-US'), item = (sym, val, text) => `<span><i><b>${sym}</b> = ${val}</i><small>${text}</small></span>`;
 return (c.model === 'vision' ? item('p', c.p, 'patch size in pixels: an image is cut into p × p squares') : item('V', n(c.V), 'vocabulary size: how many different tokens the model knows'))
  + item('d', n(c.d), `width: how many numbers describe one ${c.model === 'vision' ? 'patch' : 'token'} inside the model`)
  + item('L', c.L, 'blocks stacked one after another') + item('f', n(c.f), 'width inside the MLP')
  + `<span><i><b>n</b> = <em id="pf-sym-n">${n(work.n)}</em></i><small>${c.model === 'vision' ? 'patches' : 'tokens'} processed at once: no weights depend on it, only the work (set in Part 2)</small></span>`
  + (c.model === 'moe' ? item('E', c.E, 'experts per block') + item('k', c.k, 'experts each token runs') : '') + (c.model === 'vision' ? item('D', n(c.D), 'width of the language model it feeds') : '');
}

/* ---------- Part 2: the same diagram, counting work instead of weights ---------- */
let currentWork = null;
// Each operation of the part, what it multiplies, and what it costs; then the total for the workload.
function workCard(w, key) {
 const c = w.config, n = v => Number(v).toLocaleString('en-US'), write = w.mode === 'write', toks = write ? 1 : w.n;
 const op = (name, what, flops, note = '') => ({name, what, flops, note}), tw = (a, b) => 2 * a * b;
 const mlpOps = () => c.mats === 3 ? [op('W₁ · x', `2 × ${n(c.f)} × ${n(c.d)}`, tw(c.f, c.d), 'widen'), op('W₃ · x', `2 × ${n(c.f)} × ${n(c.d)}`, tw(c.f, c.d), 'the gate'), op('W₂ · h', `2 × ${n(c.d)} × ${n(c.f)}`, tw(c.d, c.f), 'narrow')] : [op('W_in · x', `2 × ${n(c.f)} × ${n(c.d)}`, tw(c.f, c.d), 'widen'), op('W_out · h', `2 × ${n(c.d)} × ${n(c.f)}`, tw(c.d, c.f), 'narrow')];
 const perTok = ops => ops.reduce((t, q) => t + q.flops, 0);
 const table = ops => `<table class="pf-pop-t"><tbody>${ops.map(q => `<tr><th>${q.name}${q.note ? `<small>${svgEsc(q.note)}</small>` : ''}</th><td>${q.what}</td><td>${compactWork(q.flops)}</td></tr>`).join('')}</tbody></table>`;
 const sum = lines => `<div class="pf-pop-sum">${lines.map(([a, b, d]) => `<div><span>${a}</span><b>= ${compactWork(b)}</b><small>${d}</small></div>`).join('')}</div>`;
 // per: operations for one token in one unit; extra: [[count, what, label after multiplying]]
 const scale = (label, per, first, extra = []) => { const lines = [[label, per, first]]; let run = per; for (const [k, what, tag] of extra) { run *= k; lines.push([`× ${n(k)} ${what}`, run, tag]); } if (toks > 1) { run *= toks; lines.push([`× ${n(toks)} tokens`, run, 'this workload']); } return sum(lines); };
 const share = f => `<p class="pf-pop-n"><b>${f ? compactWork(f) : '0'}</b> operations · ${pctText(pct(f, w.total))} of the work</p>`;
 const flopsOf = k => w.parts.find(x => x.key === k)?.flops || 0;
 const plus = ops => ops.map(q => compactWork(q.flops)).join(' + ');
 if (key === 'attn') {
  const proj = [op('W_Q · x', `2 × ${n(c.d)} × ${n(c.d)}`, tw(c.d, c.d), 'query'), op('W_K · x', `2 × ${n(c.dkv)} × ${n(c.d)}`, tw(c.dkv, c.d), 'key'), op('W_V · x', `2 × ${n(c.dkv)} × ${n(c.d)}`, tw(c.dkv, c.d), 'value'), op('W_O · out', `2 × ${n(c.d)} × ${n(c.d)}`, tw(c.d, c.d), 'output')];
  const mixOne = write ? 4 * w.n * c.d : 4 * w.n * w.n * c.d, shape = write ? `2 × ${n(w.n)} × ${n(c.d)}` : `2 × ${n(w.n)}² × ${n(c.d)}`;
  return `<h4>Attention: two kinds of work</h4><p class="pf-pop-k">1 · Projections (use the weights)</p>${table(proj)}${scale(plus(proj), perTok(proj), 'one token, one block', [[c.L, 'blocks', 'one token']])}${share(flopsOf('proj'))}
  <p class="pf-pop-k">2 · Mixing (no weights at all)</p>${table([op('Q · Kᵀ', shape, mixOne / 2, 'score every pair'), op('weights · V', shape, mixOne / 2, 'blend the values')])}${sum([[`${compactWork(mixOne / 2)} + ${compactWork(mixOne / 2)}`, mixOne, 'one block'], [`× ${n(c.L)} blocks`, mixOne * c.L, 'this workload']])}${share(flopsOf('mix'))}
  <p>${write ? 'The new token is compared with every token already in the context, so this grows with the context length.' : 'Every token is compared with every other token, so this grows with n². Double the prompt and this part costs four times as much.'}</p>`;
 }
 if (key === 'mlp') { const ops = mlpOps(); return `<h4>MLP</h4>${table(ops)}${scale(plus(ops), perTok(ops), 'one token, one block', [[c.L, 'blocks', 'one token']])}${share(flopsOf('mlp'))}<p>About two operations per weight, for every token. This grows in step with the number of tokens.</p>`; }
 if (key === 'experts') { const ops = mlpOps(); return `<h4>Experts · ${c.k} of ${c.E} run</h4>${table(ops)}${scale(plus(ops), perTok(ops), 'one token, one expert', [[c.k, 'experts that run', 'one token, one block'], [c.L, 'blocks', 'one token']])}${share(flopsOf('experts'))}<p>All ${c.E} experts are stored, but a token only pays for the ${c.k} it is sent to.</p>`; }
 if (key === 'router') { const ops = [op('W_r · x', `2 × ${c.E} × ${n(c.d)}`, tw(c.E, c.d), 'score the experts')]; return `<h4>Router</h4>${table(ops)}${scale(compactWork(ops[0].flops), ops[0].flops, 'one token, one block', [[c.L, 'blocks', 'one token']])}${share(flopsOf('router'))}<p>Choosing the experts costs almost nothing.</p>`; }
 if (key === 'head') { const f = flopsOf('head'); return `<h4>LM head</h4>${table([op('W_out · h', `2 × ${n(c.V)} × ${n(c.d)}`, f, 'score every token')])}${share(f)}<p>One row of the table per vocabulary token, but only for the last position: that is the one being predicted. It happens once, not once per token read.</p>`; }
 if (key === 'embed' || key === 'positions') return `<h4>Embedding lookup</h4>${share(0)}<p>Turning a token ID into its vector is picking a row out of the table. Nothing is multiplied. The table is ${compactCount(w.weights.parts.find(x => x.key === 'embed').count)} of the model's weights and almost none of the work.</p>`;
 if (key === 'patch') { const one = tw(3 * c.p * c.p, c.d); return `<h4>Patch embedding</h4>${table([op('W_patch · pixels', `2 × ${n(c.d)} × ${3 * c.p * c.p}`, one, 'pixels → vector')])}${sum([[`× ${n(toks)} patches`, one * toks, 'this workload']])}${share(flopsOf('patch'))}<p>One small multiplication per patch.</p>`; }
 if (key === 'proj') { const ops = [op('W_a · h', `2 × ${n(c.D)} × ${n(c.d)}`, tw(c.D, c.d)), op('W_b · a', `2 × ${n(c.D)} × ${n(c.D)}`, tw(c.D, c.D))]; return `<h4>Projector</h4>${table(ops)}${sum([[plus(ops), perTok(ops), 'one patch'], [`× ${n(toks)} patches`, perTok(ops) * toks, 'this workload']])}${share(flopsOf('proj-out'))}<p>Run once per patch, after the last block.</p>`; }
 return '';
}
const WORK_CATS = {embed: 'Embeddings & LM head', attn: 'Attention · projections', mix: 'Attention · mixing', mlp: 'MLP / experts', router: 'Router', vision: 'Patch embedding & projector', norm: 'Norms'};
const WORK_SHORT = {embed: 'Embed', attn: 'Attn', mix: 'Mixing', mlp: 'MLP', router: 'Router', vision: 'In/out', norm: 'Norm'};
function shareBar(label, total, cats, fmt, attnName = 'Projections') {
 const segs = Object.entries(cats).filter(([, v]) => v > 0).map(([cat, v]) => { const sh = pct(v, total); return `<span class="pf-seg ${cat}" style="flex-grow:${v}" title="${WORK_CATS[cat]}: ${fmt(v)} (${pctText(sh)})">${sh >= 9 ? `${cat === 'attn' ? attnName : WORK_SHORT[cat]} ${pctText(sh)}` : ''}</span>`; }).join('');
 return `<div class="pf-bar-row"><span>${label}<b>${fmt(total)}</b></span><div class="pf-bar">${segs}</div></div>`;
}
function workInsight(w) {
 const c = w.config, share = k => pct(w.byCat[k] || 0, w.total), mix = share('mix'), n = v => Number(v).toLocaleString('en-US'), ws = k => pct(w.weights.byCat[k]?.stored || 0, w.weights.stored);
 const lead = w.mode === 'write' ? `Writing one more token after ${n(w.n)} tokens of context` : c.model === 'vision' ? `Reading an image of ${n(w.n)} patches` : `Reading a prompt of ${n(w.n)} tokens`;
 const emb = c.model === 'vision' ? '' : ` The vocabulary tables are ${pctText(ws('embed'))} of the weights and ${pctText(share('embed'))} of the work.`;
 return `${lead} takes about ${compactWork(w.total)} operations. Attention’s mixing step holds no weights at all, yet it is ${pctText(mix)} of the work here${w.mode === 'write' ? ', and it grows with every token already in the context' : mix > 30 ? ', because it grows with n²' : '; make the prompt longer and it takes over, because it grows with n²'}.${emb}`;
}
function renderWork() {
 const el = document.getElementById('pw-diagram'); if (!el) return;
 const vision = state.model === 'vision';
 const w = workFormulas(state, {n: work.n, mode: vision ? 'read' : work.mode}); currentWork = w;
 el.innerHTML = archFlow(w.weights, w);
 const opts = [16, 128, 1024, 8192, 32768].map(v => `<option value="${v}"${v === work.n ? ' selected' : ''}>${v.toLocaleString('en-US')}</option>`).join('');
 document.getElementById('pw-controls').innerHTML = `<span class="pw-model">Model from Part 1: <b>${preset === 'custom' ? 'your own dimensions' : PF_PRESETS[preset].label.split(' · ')[0]}</b></span>
 <span class="pf-switch" role="group" aria-label="Workload"><button type="button" data-pw-mode="read" aria-pressed="${w.mode === 'read'}">${vision ? 'Read an image' : 'Read a prompt'}</button>${vision ? '' : `<button type="button" data-pw-mode="write" aria-pressed="${w.mode === 'write'}">Write one more token</button>`}</span>
 <label>${vision ? 'Patches' : w.mode === 'write' ? 'Tokens already in the context' : 'Tokens in the prompt'} n<select id="pw-n">${opts}</select></label>`;
 const wb = Object.fromEntries(Object.entries(w.weights.byCat).map(([k, v]) => [k, v.stored]));
 document.getElementById('pw-bars').innerHTML = shareBar('Weights', w.weights.stored, wb, compactCount, 'Attn') + shareBar('Work', w.total, {embed: w.byCat.embed || 0, attn: w.byCat.attn || 0, mix: w.byCat.mix || 0, mlp: w.byCat.mlp || 0, router: w.byCat.router || 0, vision: w.byCat.vision || 0}, compactWork);
 document.getElementById('pw-insight').textContent = workInsight(w);
}
function mountWork(pop) {
 const host = document.getElementById('part-2'); if (!host) return;
 host.addEventListener('click', e => { const b = e.target.closest('[data-pw-mode]'); if (b) { work.mode = b.dataset.pwMode; renderWork(); } });
 host.addEventListener('change', e => { if (e.target.id === 'pw-n') { work.n = Number(e.target.value); renderWork(); const sn = document.querySelector('#pf-sym-n'); if (sn) sn.textContent = work.n.toLocaleString('en-US'); } });
 const show = el => {
  const html = currentWork && workCard(currentWork, el.dataset.part); if (!html) { pop.hidden = true; return; }
  pop.innerHTML = html; pop.hidden = false;
  const t = (el.querySelector('.pf-hitbox') || el).getBoundingClientRect(), wd = pop.offsetWidth, h = pop.offsetHeight, above = t.top - h - 10 > 8;
  pop.style.left = Math.max(8, Math.min(innerWidth - wd - 8, t.left + t.width / 2 - wd / 2)) + 'px'; pop.style.top = Math.max(8, above ? t.top - h - 10 : Math.min(innerHeight - h - 8, t.bottom + 10)) + 'px';
 };
 host.addEventListener('pointerover', e => { const el = e.target.closest?.('#pw-flow [data-part]'); if (el) show(el); else pop.hidden = true; });
 host.addEventListener('focusin', e => { const el = e.target.closest?.('#pw-flow [data-part]'); if (el) show(el); });
 host.addEventListener('pointerleave', () => { pop.hidden = true; });
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
 const c = r.config, u = r.blockUnits, a = unitText(u.attn), m = unitText(r.expertUnits), dense = u.attn + r.expertUnits;
 if (c.model === 'moe') {
  const stored = u.attn + u.mlp, active = u.attn + u.mlpActive;
  return `<div class="pf-split"><div><span class="pf-eq">${a}d² + <i>E</i>·${m}d² = ${unitText(stored)}d²</span><small>stored per block</small></div><div><span class="pf-eq">${a}d² + <i>k</i>·${m}d² = ${unitText(active)}d²</span><small>run per token</small></div></div>
  <p class="pf-insight">With E = ${c.E} and k = ${c.k}, a block <strong>stores ${(stored / dense).toFixed(1)}×</strong> the weights of a dense block but each token <strong>runs only ${(active / dense).toFixed(1)}×</strong>. Attention falls to ${pctText(pct(u.attn, stored))} of stored block weights.</p>`;
 }
 return `<div class="pf-split"><div><span class="pf-eq"><span class="c-attn">${a}d²</span> + <span class="c-mlp">${m}d²</span> = ${unitText(dense)}d²</span><small>one block</small></div><div class="pf-mini-bar"><span class="pf-seg attn" data-cat="attn" style="flex-grow:${u.attn}">${pctText(pct(u.attn, dense))} attention</span><span class="pf-seg mlp" data-cat="mlp" style="flex-grow:${u.mlp}">${pctText(pct(u.mlp, dense))} MLP</span></div></div>`;
}

function insight(r) {
 const c = r.config;
 if (c.model === 'vision') {
  const proj = r.byCat.vision.stored, blocks = r.byCat.attn.stored + r.byCat.mlp.stored;
  return `No vocabulary, so no V · d tables. The patch embedding is small (3p² · d). The projector, ${c.Dmul}d² + ${c.Dmul * c.Dmul}d² = ${c.Dmul + c.Dmul * c.Dmul}d², weighs as much as ${((c.Dmul + c.Dmul * c.Dmul) / (r.blockUnits.attn + r.expertUnits)).toFixed(1)} encoder blocks — ${pctText(pct(proj, r.stored))} of this encoder. Blocks hold ${pctText(pct(blocks, r.stored))}.`;
 }
 const emb = r.byCat.embed.stored;
 return `Embeddings scale with V · d, blocks with L · d². Here V · d${c.tied ? '' : ' (×2)'} is ${pctText(pct(emb, r.stored))} of the ${c.model === 'moe' ? 'stored ' : ''}model. Shrink d or L, or grow V, and the vocabulary tables take over — that is why small models often tie the embedding and LM head.`;
}

function table(r) {
 return `<table class="pf-table"><thead><tr><th>Part</th><th>Formula</th><th>Count</th><th>Share</th></tr></thead><tbody>${r.parts.map(x => `<tr data-cat="${x.cat}" tabindex="0"><th><i class="pf-dot ${x.cat}"></i>${x.label}</th><td><code>${x.formula}</code>${x.shorthand !== x.formula ? `<small>${x.shorthand}</small>` : ''}</td><td>${compactCount(x.count)}${x.active !== x.count ? `<small>${compactCount(x.active)} run</small>` : ''}</td><td>${pctText(pct(x.count, r.stored))}</td></tr>`).join('')}</tbody></table>`;
}

function controls() {
 const c = state, opt = (vals, cur, fmt = v => v) => [...new Set([...vals, cur])].sort((x, y) => x - y).map(v => `<option value="${v}"${String(v) === String(cur) ? ' selected' : ''}>${fmt(v)}</option>`).join('');
 const llm = c.model !== 'vision', num = v => Number(v).toLocaleString();
 return `<label>Width d<select data-pf="d">${opt([64, 128, 256, 512, 768, 1024, 4096], c.d, num)}</select></label>
 <label>Blocks L <output>${c.L}</output><input data-pf="L" type="range" min="1" max="48" value="${c.L}"></label>
 ${llm ? `<label>Vocabulary V<select data-pf="V">${opt([1000, 8000, 32000, 50257, 128256], c.V, num)}</select></label><label class="pf-check"><input type="checkbox" data-pf="tied"${c.tied ? ' checked' : ''}> Tie embedding &amp; LM head</label>` : `<label>LLM width D<select data-pf="Dmul">${opt([2, 4, 5], c.Dmul, v => `${v}d`)}</select></label>`}
 <label>MLP<select data-pf="mlp"><option value="gelu"${c.mlp === 'gelu' ? ' selected' : ''}>GELU · 2 matrices</option><option value="swiglu"${c.mlp === 'swiglu' ? ' selected' : ''}>SwiGLU · 3 matrices</option></select></label>
 <label>MLP width f<select data-pf="fr">${opt([2.6667, 3.5, 4], c.fr, v => `${unitText(v)}d`)}</select></label>
 <label>Attention<select data-pf="kvRatio">${opt([.25, 1], c.kvRatio, v => v === 1 ? 'Full multi-head' : `GQA · KV width ${unitText(v)}d`)}</select></label>
 ${c.model === 'moe' ? `<label>Experts E <output>${c.E}</output><input data-pf="E" type="range" min="2" max="16" value="${c.E}"></label><label>Active k <output>${c.k}</output><input data-pf="k" type="range" min="1" max="${Math.min(4, c.E)}" value="${Math.min(c.k, c.E)}"></label>` : ''}`;
}
function presetBar() {
 return `<label>Model<select id="pf-preset">${Object.entries(PF_PRESETS).filter(([key]) => key !== 'toy').concat([['toy', PF_PRESETS.toy]]).map(([key, p]) => `<option value="${key}"${key === preset ? ' selected' : ''}>${key === 'toy' ? 'Small toy numbers · to play with' : p.label}</option>`).join('')}${preset === 'custom' ? '<option value="custom" selected>Custom</option>' : ''}</select></label><span>${preset === 'custom' ? 'Your own mix of dimensions.' : PF_PRESETS[preset].note}</span>`;
}

function render(withControls = true) {
 const r = parameterFormulas(state);
 root.querySelectorAll('[data-pf-model]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.pfModel === state.model)));
 root.querySelector('#pf-presets').innerHTML = presetBar();
 if (withControls) root.querySelector('#pf-controls').innerHTML = controls();
 const diagramEl = root.querySelector('#pf-diagram');
 current = r;diagramEl.innerHTML = archFlow(r);
 root.querySelector('#pf-symbols').innerHTML = symbols(r);
 const custom = root.querySelector('#pf-custom'); if (custom && (preset === 'toy' || preset === 'custom')) custom.open = true;
 if (withControls && !reduced.matches) { diagramEl.classList.remove('pf-enter'); void diagramEl.offsetWidth; diagramEl.classList.add('pf-enter'); }
 applyFocus();
 root.querySelector('#pf-split').innerHTML = blockSplit(r);
 root.querySelector('#pf-bars').innerHTML = bar(r, 'stored') + (state.model === 'moe' ? bar(r, 'active') : '');
 root.querySelector('#pf-insight').textContent = insight(r);
 root.querySelector('#pf-table').innerHTML = table(r);
 renderWork();
}

if (root) {
 root.addEventListener('click', e => { const b = e.target.closest('[data-pf-model]'); if (b) { state.model = b.dataset.pfModel; preset = 'custom'; render(); } });
 root.addEventListener('change', e => {
  if (e.target.id !== 'pf-preset' || !PF_PRESETS[e.target.value]) return;
  preset = e.target.value; Object.assign(state, PF_DEFAULTS, PF_PRESETS[preset].config); render();
 });
 const catOf = e => e.target.closest?.('[data-cat]')?.dataset.cat || null;
 root.addEventListener('pointerover', e => setFocus(catOf(e)));
 root.addEventListener('pointerleave', () => { setFocus(null); pop.hidden = true; });
 // formula card for the part under the pointer (or focused)
 const pop = root.querySelector('#pf-pop');
 const showPop = el => {
  const html = current && partCard(current, el.dataset.part); if (!html) { pop.hidden = true; return; }
  pop.innerHTML = html; pop.hidden = false;
  const t = (el.querySelector('.pf-hitbox') || el).getBoundingClientRect(), w = pop.offsetWidth, h = pop.offsetHeight, above = t.top - h - 10 > 8;
  pop.style.left = Math.max(8, Math.min(innerWidth - w - 8, t.left + t.width / 2 - w / 2)) + 'px'; pop.style.top = (above ? t.top - h - 10 : Math.min(innerHeight - h - 8, t.bottom + 10)) + 'px';
 };
 root.addEventListener('pointerover', e => { const el = e.target.closest?.('#pf-flow [data-part]'); if (el) showPop(el); else pop.hidden = true; });
 root.addEventListener('focusin', e => { const el = e.target.closest?.('#pf-flow [data-part]'); if (el) showPop(el); });
 mountWork(pop);
 addEventListener('scroll', () => { pop.hidden = true; }, {passive: true});
 root.addEventListener('focusin', e => setFocus(catOf(e)));
 root.addEventListener('focusout', e => { if (!root.contains(e.relatedTarget)) setFocus(null); });
 root.addEventListener('input', e => {
  const key = e.target.dataset?.pf; if (!key) return;
  state[key] = e.target.type === 'checkbox' ? e.target.checked : key === 'mlp' ? e.target.value : Number(e.target.value);
  preset = 'custom';
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
