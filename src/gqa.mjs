// Equal-sized query groups, with the same d_h for each Q, K and V head.
// Cache counts are logical K/V payload bytes for ONE layer and ONE sequence,
// not allocator overhead, model weights, temporary buffers or measured speed.
function positiveInteger(value, name) {
 if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive safe integer`);
}

export function kvHeadChoices(queryHeads) {
 positiveInteger(queryHeads, 'queryHeads');
 return Array.from({length: queryHeads}, (_, i) => i + 1).filter(k => queryHeads % k === 0);
}

export function headGroups(queryHeads, kvHeads) {
 positiveInteger(queryHeads, 'queryHeads');
 positiveInteger(kvHeads, 'kvHeads');
 if (queryHeads % kvHeads !== 0) throw new RangeError('kvHeads must divide queryHeads');
 const groupSize = queryHeads / kvHeads;
 return Array.from({length: kvHeads}, (_, group) => ({
  kvHead: group + 1,
  queryHeads: Array.from({length: groupSize}, (_, i) => group * groupSize + i + 1),
 }));
}

export function cacheStats({queryHeads, kvHeads, tokens, headDim = 128, bytesPerElement = 2}) {
 const groups = headGroups(queryHeads, kvHeads);
 for (const [name, value] of Object.entries({tokens, headDim, bytesPerElement})) positiveInteger(value, name);
 const mhaBytes = 2 * tokens * queryHeads * headDim * bytesPerElement;
 if (!Number.isSafeInteger(mhaBytes)) throw new RangeError('cache size exceeds exact integer range');
 const bytes = 2 * tokens * kvHeads * headDim * bytesPerElement;
 return {groups, groupSize: queryHeads / kvHeads, bytes, mhaBytes,
  share: kvHeads / queryHeads, reduction: queryHeads / kvHeads,
  // One-head attention is both endpoints; prefer the MHA name in that degenerate case.
  kind: kvHeads === queryHeads ? 'MHA' : kvHeads === 1 ? 'MQA' : 'GQA'};
}
