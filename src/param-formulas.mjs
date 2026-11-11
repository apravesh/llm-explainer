// Formula-first parameter ledger for three whole-model sketches.
// Every count is derived from the formula shown beside it; dimensions are deliberately small.

export const PF_DEFAULTS = Object.freeze({model: 'dense', d: 256, L: 8, V: 8000, E: 8, k: 2, p: 16, Dmul: 4, tied: false, fr: 4, mlp: 'gelu', kvRatio: 1, pos: 0});

// Published architecture dimensions of well-known open models. Counts follow the same bias-free formulas,
// so they land within a fraction of a percent of the official totals.
export const PF_PRESETS = Object.freeze({
 toy: {label: 'Toy · small numbers', note: 'Deliberately small, so the formulas are easy to read.', config: {}},
 gpt2: {label: 'GPT-2 small · 124M', note: 'GELU MLP with f = 4d, full multi-head attention, learned positions, embedding tied to the LM head.', config: {model: 'dense', d: 768, L: 12, V: 50257, tied: true, fr: 4, mlp: 'gelu', kvRatio: 1, pos: 1024}},
 mistral: {label: 'Mistral 7B · 7.2B', note: 'SwiGLU MLP with f = 3.5d, grouped-query attention (8 KV heads for 32 query heads), RoPE positions.', config: {model: 'dense', d: 4096, L: 32, V: 32000, tied: false, fr: 3.5, mlp: 'swiglu', kvRatio: .25, pos: 0}},
 llama3: {label: 'Llama 3 8B · 8.0B', note: 'Same block shape as Mistral 7B. The 128K vocabulary makes the two embedding tables four times larger.', config: {model: 'dense', d: 4096, L: 32, V: 128256, tied: false, fr: 3.5, mlp: 'swiglu', kvRatio: .25, pos: 0}},
 mixtral: {label: 'Mixtral 8x7B · 46.7B', note: 'Mistral 7B blocks with 8 SwiGLU experts per block and top-2 routing.', config: {model: 'moe', d: 4096, L: 32, V: 32000, tied: false, fr: 3.5, mlp: 'swiglu', kvRatio: .25, pos: 0, E: 8, k: 2}},
 pixtral: {label: 'Pixtral 12B vision encoder · 0.4B', note: 'Gated MLP with f = 4d, 16 × 16 patches, 2D RoPE, and a two-layer projector into the 5,120-wide language model.', config: {model: 'vision', d: 1024, L: 24, p: 16, Dmul: 5, fr: 4, mlp: 'swiglu', kvRatio: 1, pos: 0}}
});
export const unitText = n => String(Math.round(n * 100) / 100);

export const PF_CATEGORIES = {
 embed: {label: 'Embeddings & LM head', short: 'Embed'},
 attn: {label: 'Attention', short: 'Attn'},
 mlp: {label: 'MLP / experts', short: 'MLP'},
 router: {label: 'Router', short: 'Router'},
 vision: {label: 'Patch embedding & projector', short: 'In/out'},
 norm: {label: 'Norms', short: 'Norm'}
};

// Bias-free projections and RMSNorm-style gains (d per norm). An MLP has 2 (GELU) or 3 (SwiGLU) matrices of d × f.
// Grouped-query attention shrinks W_K and W_V to d × d_kv, with d_kv = kvRatio · d.
export function parameterFormulas(input = {}) {
 const c = {...PF_DEFAULTS, ...input};
 const {model, d, L, V, p} = c, f = Math.round(c.fr * d), mats = c.mlp === 'swiglu' ? 3 : 2, r = c.kvRatio, dkv = Math.round(r * d);
 const E = Math.max(1, c.E), k = Math.min(Math.max(1, c.k), E), D = c.Dmul * d;
 const attnUnits = 2 + 2 * dkv / d, mlpUnits = mats * f / d, perMlp = mats * d * f, perAttn = 2 * d * d + 2 * d * dkv;
 const parts = [];
 const add = (key, cat, label, formula, shorthand, count, active = count) => parts.push({key, cat, label, formula, shorthand, count, active});
 if (model === 'vision') {
  add('patch', 'vision', 'Patch embedding', '3p² · d', `${3 * p * p} · d`, 3 * p * p * d);
  add('pos', 'vision', 'Positions', '0 · 2D RoPE', 'no weights', 0);
 } else {
  add('embed', 'embed', 'Token embedding', 'V · d', 'V · d', V * d);
  if (c.pos > 0) add('positions', 'embed', 'Learned positions', 'n_ctx · d', `${c.pos} · d`, c.pos * d);
 }
 add('attn', 'attn', 'Attention · W_Q W_K W_V W_O', r === 1 ? 'L · 4 · d²' : 'L · (2·d² + 2·d·d_kv)', `L · ${unitText(attnUnits)}d²`, L * perAttn);
 if (model === 'moe') {
  add('router', 'router', 'Router W_r', 'L · d · E', 'L · dE', L * d * E);
  add('experts', 'mlp', `${E} experts · ${mats} matrices each`, `L · E · ${mats}·d·f`, `L · ${E} · ${unitText(mlpUnits)}d²`, L * E * perMlp, L * k * perMlp);
 } else add('mlp', 'mlp', mats === 3 ? 'MLP · W₁ W₃ W₂' : 'MLP · W_in W_out', `L · ${mats}·d·f`, `L · ${unitText(mlpUnits)}d²`, L * perMlp);
 add('norm', 'norm', 'Norm gains', 'L · 2d + d', '≈ 0', L * 2 * d + d);
 if (model === 'vision') add('proj', 'vision', 'Projector to LLM width D', 'd·D + D·D', `${c.Dmul}d² + ${c.Dmul * c.Dmul}d²`, d * D + D * D);
 else add('head', 'embed', c.tied ? 'LM head · tied to embedding' : 'LM head', c.tied ? '0 · shared' : 'd · V', c.tied ? 'reuses V · d' : 'd · V', c.tied ? 0 : d * V);
 const stored = parts.reduce((s, x) => s + x.count, 0), active = parts.reduce((s, x) => s + x.active, 0);
 const byCat = Object.fromEntries(Object.keys(PF_CATEGORIES).map(cat => [cat, {stored: 0, active: 0}]));
 for (const x of parts) { byCat[x.cat].stored += x.count; byCat[x.cat].active += x.active; }
 const blockUnits = model === 'moe' ? {attn: attnUnits, mlp: mlpUnits * E, mlpActive: mlpUnits * k} : {attn: attnUnits, mlp: mlpUnits, mlpActive: mlpUnits};
 return {config: {...c, f, mats, dkv, E, k, D}, parts, stored, active, byCat, blockUnits, expertUnits: mlpUnits};
}

export function compactCount(n) {
 if (n === 0) return '0';
 const units = [[1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
 for (const [v, s] of units) if (Math.abs(n) >= v) { const x = n / v; return `${x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2)}${s}`; }
 return String(n);
}
