// Long context: what a global and a local (sliding-window) attention layer cost.
// Plain counting, no model: pairs of tokens compared in the mixing step, and tokens whose keys and values
// a layer has to keep in the KV cache. A local layer lets a token look at itself and the w − 1 tokens before it.

export const canSee = (i, j, w = null) => j <= i && (w == null || i - j < w);

// Pairs compared when a prompt of n tokens is read once by one causal layer.
export function pairs(n, w = null) {
 if (w == null || w >= n) return n * (n + 1) / 2;
 return w * (w + 1) / 2 + (n - w) * w;
}

// Tokens whose keys and values one layer keeps while writing with n tokens in the context.
export const cached = (n, w = null) => w == null ? n : Math.min(n, w);

export const PATTERNS = Object.freeze({
 global: {label: 'All global', local: 0, global: 1},
 mixed: {label: '5 local : 1 global', local: 5, global: 1},
 local: {label: 'All local', local: 1, global: 0},
});

// Average over one repeat of the pattern, compared with every layer being global.
export function compare({n, w, pattern}) {
 const p = PATTERNS[pattern], layers = p.local + p.global;
 const pairsPerLayer = (p.local * pairs(n, w) + p.global * pairs(n)) / layers;
 const cachePerLayer = (p.local * cached(n, w) + p.global * cached(n)) / layers;
 return {pairsPerLayer, cachePerLayer, pairsGlobal: pairs(n), cacheGlobal: cached(n),
  pairsShare: pairsPerLayer / pairs(n), cacheShare: cachePerLayer / cached(n)};
}

// Layer kinds for a stack of `count` layers: the pattern repeated, global layer last in each repeat.
export function layerKinds(pattern, count = 12) {
 const p = PATTERNS[pattern], cycle = [...Array(p.local).fill('local'), ...Array(p.global).fill('global')];
 return Array.from({length: count}, (_, i) => cycle[i % cycle.length]);
}
