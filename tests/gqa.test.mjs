import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {cacheStats, headGroups, kvHeadChoices} from '../src/gqa.mjs';
const read = name => readFile(new URL('../' + name, import.meta.url), 'utf8');

test('equal-sized groups cover each query exactly once, including MHA and MQA', () => {
 for (const q of [4, 8, 16]) for (const kv of kvHeadChoices(q)) {
  const groups = headGroups(q, kv);
  assert.equal(groups.length, kv);
  assert.deepEqual(groups.flatMap(g => g.queryHeads), Array.from({length: q}, (_, i) => i + 1));
  groups.forEach((g, i) => { assert.equal(g.kvHead, i + 1); assert.equal(g.queryHeads.length, q / kv); });
 }
 assert.deepEqual(headGroups(8, 2), [{kvHead: 1, queryHeads: [1, 2, 3, 4]}, {kvHead: 2, queryHeads: [5, 6, 7, 8]}]);
 assert.deepEqual(kvHeadChoices(12), [1, 2, 3, 4, 6, 12]);
});

test('the default example is exactly 8 MiB, one quarter of MHA, with four queries per group', () => {
 const r = cacheStats({queryHeads: 8, kvHeads: 2, tokens: 8192});
 assert.equal(r.bytes, 8 * 2 ** 20); assert.equal(r.mhaBytes, 32 * 2 ** 20);
 assert.equal(r.share, 1 / 4); assert.equal(r.reduction, 4); assert.equal(r.groupSize, 4); assert.equal(r.kind, 'GQA');
});

test('all control combinations satisfy the exact cache formula and endpoint comparison', () => {
 for (const queryHeads of [4, 8, 16]) for (const kvHeads of kvHeadChoices(queryHeads)) for (const tokens of [8192, 32768, 131072]) {
  const r = cacheStats({queryHeads, kvHeads, tokens});
  assert.equal(r.bytes, 2 * tokens * kvHeads * 128 * 2);
  assert.equal(r.bytes / r.mhaBytes, kvHeads / queryHeads);
  assert.equal(r.kind, kvHeads === queryHeads ? 'MHA' : kvHeads === 1 ? 'MQA' : 'GQA');
  assert.equal(r.bytes * r.reduction, r.mhaBytes);
 }
 assert.equal(cacheStats({queryHeads: 1, kvHeads: 1, tokens: 1}).share, 1);
});

test('tokens, head width and precision scale payload, not the head-sharing ratio', () => {
 const args = {queryHeads: 8, kvHeads: 2, tokens: 8192, headDim: 128, bytesPerElement: 2};
 const baseline = cacheStats(args);
 for (const key of ['tokens', 'headDim', 'bytesPerElement']) {
  const doubled = cacheStats({...args, [key]: args[key] * 2});
  assert.equal(doubled.bytes, 2 * baseline.bytes); assert.equal(doubled.share, baseline.share);
 }
});

test('invalid dimensions, unequal groups and inexact byte counts are rejected', () => {
 const args = {queryHeads: 8, kvHeads: 2, tokens: 8192, headDim: 128, bytesPerElement: 2};
 for (const key of Object.keys(args)) for (const bad of [0, -1, 1.5, NaN, Infinity, '2']) {
  assert.throws(() => cacheStats({...args, [key]: bad}), RangeError, `${key}=${bad}`);
 }
 assert.throws(() => headGroups(8, 3), /divide/);
 assert.throws(() => headGroups(4, 8), /divide/);
 assert.throws(() => cacheStats({...args, tokens: Number.MAX_SAFE_INTEGER}), /exact integer/);
});

test('KV cache subsection uses a simple accessible comparison table instead of the old diagram', async () => {
 const html = await read('analytics.html');
 assert.ok(html.indexOf('id="lc-lab"') < html.indexOf('id="gqa-title"'));
 assert.match(html, /KV Cache and Grouped Query Attention \(GQA\)/);
 assert.match(html, /Old Q is not discarded because it changes/);
 assert.match(html, /sequence-specific intermediate vectors, not model weights/);
 assert.match(html, /<caption>Same 8 query heads/);
 assert.match(html, /<th scope="row">Multi-head attention \(MHA\)<\/th><td>8<\/td><td>8<\/td><td>8<\/td>/);
 assert.match(html, /<th scope="row">Grouped-query attention \(GQA\)<\/th><td>8<\/td><td>2<\/td><td>2<\/td>/);
 assert.match(html, /<th scope="row">Multi-query attention \(MQA\)<\/th><td>8<\/td><td>1<\/td><td>1<\/td>/);
 for (const gone of ['gqa-app.mjs','id="gqa-groups"','data-gqa-kind','id="gqa-q"']) assert.ok(!html.includes(gone));
 for (const part of ['src/long-context-app.mjs','src/param-formulas-app.mjs','cache_explanation','not a universal speedup','ONE LAYER, ONE SEQUENCE']) assert.ok(html.includes(part));
 const ids=[...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]);assert.equal(new Set(ids).size,ids.length);
});
