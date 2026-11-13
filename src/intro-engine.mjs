// A tiny, honest stand-in for "the black box": it predicts the next word by looking up what followed the
// same words in a small set of hand-written example texts. It is NOT a neural network and not a real LLM.
// A real LLM learns patterns that generalize; this one can only reuse what its examples contain.

export const END = '<end>', NEWLINE = '\n';

const STORIES = [
 'once upon a time there was a small dragon who was afraid of fire . every night the dragon looked at the stars and dreamed of flying . one day the dragon met a young girl who was not afraid . she taught the dragon that fire could be warm and kind . and they lived happily ever after .',
 'once upon a time there was a small robot who wanted to learn to paint . one day the robot found a box of paint in the garden . the robot painted the sky , the trees and the river . and they lived happily ever after .',
 'once upon a time there was a young princess who loved to read . every night the princess looked at the stars and dreamed of flying . one day the princess met a small dragon who was not afraid . and they lived happily ever after .'
];
const SENTENCES = [
 'the cat sat on the mat .', 'the cat sat on the mat .', 'the cat sat on the mat and purred .', 'the cat sat on the sofa .', 'the cat sat on the windowsill .',
 'the dog sat on the rug .', 'the dog sat on the floor .', 'the cat chased the mouse .', 'the dog chased the ball .',
 'the sky is blue .', 'the sky is blue .', 'the sky is clear tonight .', 'the grass is green .', 'the sun is bright .',
 'the capital of france is paris .', 'the capital of japan is tokyo .', 'the capital of italy is rome .', 'two plus two is four .'
];
export const CHAT_EXAMPLES = [
 ['what is the capital of france ?', 'the capital of france is paris .'],
 ['what is the capital of japan ?', 'the capital of japan is tokyo .'],
 ['what is the capital of italy ?', 'the capital of italy is rome .'],
 ['what is two plus two ?', 'two plus two is four .'],
 ['what color is the sky ?', 'the sky is blue .'],
 ['tell me a story .', 'once upon a time there was a small dragon who loved to read .'],
 ['hello !', 'hello ! how can i help you today ?']
];

export const tokenize = text => text.trim().split(/ +/).filter(Boolean);
/** The chat template: the whole conversation as one stream of words, ending where the assistant should start. */
export const chatPrompt = question => ['user', ':', ...tokenize(question.toLowerCase()), NEWLINE, 'assistant', ':'];

export const CORPUS = Object.freeze([
 ...STORIES.map(s => [...tokenize(s), END]),
 ...SENTENCES.map(s => [...tokenize(s), END]),
 ...CHAT_EXAMPLES.map(([q, a]) => [...chatPrompt(q), ...tokenize(a), END])
].map(Object.freeze));

const MAX_CONTEXT = 12;
/**
 * Next-word distribution for a context. For every suffix length L that occurs in the examples, count what
 * came next; longer matches get more weight (2^L, levelling off after 4 words), so specific evidence
 * leads but shorter matches keep a real chance. That keeps random stories varied, while long contexts
 * (a whole chat question) still decide the top pick.
 */
const FLOOR_SHARE = 0.01, FLOOR_WORDS = 8;
let commonWords = null;
const COMMON_WORDS = {[Symbol.iterator]() {
 if (!commonWords) {
  const counts = new Map();
  for (const line of CORPUS) for (const t of line) if (/^[a-z]+$/i.test(t)) counts.set(t, (counts.get(t) || 0) + 1);
  const top = [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, FLOOR_WORDS), total = top.reduce((n, [, c]) => n + c, 0);
  commonWords = top.map(([t, c]) => [t, c / total]);
 }
 return commonWords[Symbol.iterator]();
}};
export function predict(context, {top = 5} = {}) {
 if (!Array.isArray(context) || !context.length) throw new RangeError('Need at least one context word');
 const scores = new Map();
 let matched = 0;
 for (let L = Math.min(MAX_CONTEXT, context.length); L >= 1; L--) {
  const suffix = context.slice(-L), counts = new Map();
  let total = 0;
  for (const line of CORPUS) for (let i = 0; i + L < line.length; i++) {
   let same = true;
   for (let j = 0; j < L; j++) if (line[i + j] !== suffix[j]) { same = false; break; }
   if (same) { counts.set(line[i + L], (counts.get(line[i + L]) || 0) + 1); total++; }
  }
  if (!total) continue;
  matched = Math.max(matched, L);
  const weight = 2 ** Math.min(L, 4);
  for (const [token, n] of counts) scores.set(token, (scores.get(token) || 0) + weight * n / total);
 }
 if (!scores.size) return {candidates: [], matched: 0};
 const sum = [...scores.values()].reduce((a, b) => a + b, 0);
 const all = [...scores].map(([token, s]) => ({token, p: s / sum})).sort((a, b) => b.p - a.p || (a.token < b.token ? -1 : 1));
 // A real model gives every word some chance. When the corpus offers fewer than `top` continuations, the list
 // shown is filled with its most common words at a token chance (well under 1%, marked rare). They can be
 // picked by hand but are never drawn at random: `all` stays the distribution that choose() samples from.
 const candidates = all.slice(0, top);
 for (const [token, share] of COMMON_WORDS) {
  if (candidates.length >= top) break;
  if (!scores.has(token)) candidates.push({token, p: FLOOR_SHARE * share, rare: true});
 }
 return {candidates, all, matched};
}

/** Deterministic uniform numbers in [0, 1) so "roll the dice" replays identically. */
export function nextRandom(seed) { const next = (Math.imul(seed >>> 0, 1664525) + 1013904223) >>> 0; return {seed: next, value: next / 4294967296}; }

/** Pick the next word: the most likely one, or a random draw in proportion to the probabilities. */
export function choose(prediction, u = null) {
 const list = prediction.all || prediction.candidates;
 if (!list.length) return null;
 if (u === null) return list[0].token;
 let sum = 0;
 for (const c of list) { sum += c.p; if (u < sum) return c.token; }
 return list.at(-1).token;
}

/** Run the predictor repeatedly: append its choice and ask again, until it emits the end marker or a limit. */
export function generate(context, {limit = 40, stopAt = [END], seed = null} = {}) {
 const out = [];
 let ctx = [...context], s = seed;
 for (let i = 0; i < limit; i++) {
  const prediction = predict(ctx);
  let u = null;
  if (s !== null) { const r = nextRandom(s); s = r.seed; u = r.value; }
  const token = choose(prediction, u);
  if (token === null) break;
  out.push(token); ctx.push(token);
  if (stopAt.includes(token)) break;
 }
 return out;
}

const PROPER = new Set(['france', 'japan', 'italy', 'paris', 'tokyo', 'rome', 'i']);
/** Turn tokens back into readable text: no space before punctuation, capitals after sentence ends. */
export function detokenize(tokens, {capitalizeFirst = true} = {}) {
 let text = '', capital = capitalizeFirst;
 for (const raw of tokens) {
  if (raw === END) continue;
  if (raw === NEWLINE) { text += '\n'; capital = true; continue; }
  let word = PROPER.has(raw) ? raw[0].toUpperCase() + raw.slice(1) : raw;
  if (raw === 'user' || raw === 'assistant') word = raw[0].toUpperCase() + raw.slice(1);
  else if (capital && /^[a-z]/.test(word)) word = word[0].toUpperCase() + word.slice(1);
  const glue = /^[.,!?:]$/.test(raw) || !text || text.endsWith('\n') ? '' : ' ';
  text += glue + word;
  capital = /^[.!?]$/.test(raw) || (capital && raw === ':' ? true : capital && !/^[a-z]/i.test(raw));
  if (raw === ':') capital = true;
 }
 return text;
}
export const displayToken = token => token === NEWLINE ? '↵' : token === END ? 'end' : PROPER.has(token) || token === 'user' || token === 'assistant' ? token[0].toUpperCase() + token.slice(1) : token;

export const NEXT_WORD_PROMPTS = ['the cat sat on the', 'the capital of france is', 'the sky is', 'once upon a time there was a'];
export const STORY_PROMPTS = ['once upon a time', 'one day the robot', 'every night the princess'];
