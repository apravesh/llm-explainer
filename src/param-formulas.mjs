// Formula-first parameter ledger for three whole-model sketches.
// Every count is derived from the formula shown beside it; dimensions are deliberately small.

export const PF_DEFAULTS = Object.freeze({model: 'dense', d: 256, L: 8, V: 8000, E: 8, k: 2, p: 16, Dmul: 4, tied: false});

export const PF_CATEGORIES = {
 embed: {label: 'Embeddings & LM head', short: 'Embed'},
 attn: {label: 'Attention', short: 'Attn'},
 mlp: {label: 'MLP / experts', short: 'MLP'},
 router: {label: 'Router', short: 'Router'},
 vision: {label: 'Patch embedding & projector', short: 'In/out'},
 norm: {label: 'Norms', short: 'Norm'}
};

// Bias-free projections, RMSNorm-style gains (d per norm), ordinary GELU MLP with f = 4d.
export function parameterFormulas(input = {}) {
 const c = {...PF_DEFAULTS, ...input};
 const {model, d, L, V, p} = c, f = 4 * d, E = Math.max(1, c.E), k = Math.min(Math.max(1, c.k), E), D = c.Dmul * d;
 const parts = [];
 const add = (key, cat, label, formula, shorthand, count, active = count) => parts.push({key, cat, label, formula, shorthand, count, active});
 if (model === 'vision') {
  add('patch', 'vision', 'Patch embedding', '3p² · d', `${3 * p * p} · d`, 3 * p * p * d);
  add('pos', 'vision', 'Positions', '0 · 2D RoPE', 'no weights', 0);
 } else add('embed', 'embed', 'Token embedding', 'V · d', 'V · d', V * d);
 add('attn', 'attn', 'Attention · W_Q W_K W_V W_O', 'L · 4 · d²', 'L · 4d²', L * 4 * d * d);
 if (model === 'moe') {
  add('router', 'router', 'Router W_r', 'L · d · E', 'L · dE', L * d * E);
  add('experts', 'mlp', `${E} experts · W_in W_out each`, 'L · E · 2·d·f', `L · ${E} · 8d²`, L * E * 2 * d * f, L * k * 2 * d * f);
 } else add('mlp', 'mlp', 'MLP · W_in W_out', 'L · 2·d·f', 'L · 8d²', L * 2 * d * f);
 add('norm', 'norm', 'Norm gains', 'L · 2d + d', '≈ 0', L * 2 * d + d);
 if (model === 'vision') add('proj', 'vision', 'Projector to LLM width D', 'd·D + D·D', `${c.Dmul}d² + ${c.Dmul * c.Dmul}d²`, d * D + D * D);
 else add('head', 'embed', c.tied ? 'LM head · tied to embedding' : 'LM head', c.tied ? '0 · shared' : 'd · V', c.tied ? 'reuses V · d' : 'd · V', c.tied ? 0 : d * V);
 const stored = parts.reduce((s, x) => s + x.count, 0), active = parts.reduce((s, x) => s + x.active, 0);
 const byCat = Object.fromEntries(Object.keys(PF_CATEGORIES).map(cat => [cat, {stored: 0, active: 0}]));
 for (const x of parts) { byCat[x.cat].stored += x.count; byCat[x.cat].active += x.active; }
 const blockUnits = model === 'moe' ? {attn: 4, mlp: 8 * E, mlpActive: 8 * k} : {attn: 4, mlp: 8, mlpActive: 8};
 return {config: {...c, f, E, k, D}, parts, stored, active, byCat, blockUnits};
}

export function compactCount(n) {
 if (n === 0) return '0';
 const units = [[1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
 for (const [v, s] of units) if (Math.abs(n) >= v) { const x = n / v; return `${x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2)}${s}`; }
 return String(n);
}
