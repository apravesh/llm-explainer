import {END, NEWLINE, CHAT_EXAMPLES, NEXT_WORD_PROMPTS, STORY_PROMPTS, tokenize, chatPrompt, predict, choose, nextRandom, detokenize, displayToken} from './intro-engine.mjs';

const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const pct = p => p >= .995 ? '100%' : p < .005 ? '<1%' : `${Math.round(p * 100)}%`;
const chips = (el, items, current, attr) => { el.innerHTML = items.map((text, i) => `<button type="button" ${attr}="${i}" aria-pressed="${i === current}">${esc(text)}</button>`).join(''); };
const sentenceCase = text => detokenize(tokenize(text));

// The text so far as HTML: the reader's opening in dark ink, the model's words in blue, the newest word marked.
function grownText(tokens, start, markNewest) {
 const full = detokenize(tokens), given = detokenize(tokens.slice(0, start)), rest = full.slice(given.length);
 if (!rest) return `<span class="given">${esc(given)}</span>`;
 const cut = markNewest ? rest.lastIndexOf(' ') + 1 : rest.length;
 return `<span class="given">${esc(given)}</span><span class="grown">${esc(rest.slice(0, cut))}${cut < rest.length ? `<span class="new">${esc(rest.slice(cut))}</span>` : ''}</span>`;
}

/* ---------- The machine: words in → black box → a list of guesses ---------- */
// `pick` is the word the loop will take next (highlighted); bars are buttons when `choosable`.
function machine(el, context, prediction, {pick = null, choosable = false, fresh = false, newest = -1, step = false} = {}) {
 const shown = context.slice(-9), hidden = context.length - shown.length;
 let rows = prediction.candidates.slice(0, 5);
 if (pick && !rows.some(c => c.token === pick)) rows = [...rows.slice(0, 4), prediction.all.find(c => c.token === pick)];
 const max = rows[0]?.p || 1;
 el.classList.toggle('fresh', fresh && !step && !reduced.matches);
 el.classList.toggle('step', step && !reduced.matches);   // one word was added: only the changed parts animate
 el.innerHTML = `<div class="mc-in"><p class="string-label">THE TEXT SO FAR</p><div class="mc-words">${hidden > 0 ? '<span class="mc-more">…</span>' : ''}${shown.map((w, i) => `<span class="mc-word ${hidden + i === newest ? 'new' : ''}" data-i="${hidden + i}" style="--d:${i * 45}ms">${esc(displayToken(w))}</span>`).join('')}</div></div>
 <span class="mc-arrow" aria-hidden="true"><i></i></span>
 <div class="mc-box" role="img" aria-label="The language model, shown as a black box"><b>LLM</b><small>black box</small></div>
 <span class="mc-arrow out" aria-hidden="true"><i></i></span>
 <div class="mc-out"><p class="string-label">ITS GUESSES FOR THE NEXT WORD</p><div class="mc-bars">${rows.length ? rows.map((c, i) => {
  const inner = `<span class="mc-cand">${c.token === END ? '<em>stop here</em>' : esc(displayToken(c.token))}</span><span class="mc-track"><i style="width:${Math.max(2, c.p / max * 100)}%"></i></span><b>${pct(c.p)}</b>`;
  const cls = `mc-row ${c.token === pick ? 'pick' : ''}`, style = `style="--d:${i * 90}ms"`;
  return choosable ? `<button type="button" class="${cls}" ${style} data-choose="${esc(c.token)}" aria-label="Choose ${c.token === END ? 'stop' : esc(c.token)}, ${pct(c.p)}">${inner}</button>` : `<div class="${cls}" ${style}>${inner}</div>`;
 }).join('') : '<p class="small">Nothing more to add.</p>'}</div></div>`;
}

/* ---------- 01 · One guess ---------- */
let nwPrompt = 0;
function renderNextWord(fresh = true) {
 chips($('#nw-prompts'), NEXT_WORD_PROMPTS.map(p => `${sentenceCase(p)} …`), nwPrompt, 'data-nw');
 const context = tokenize(NEXT_WORD_PROMPTS[nwPrompt]), prediction = predict(context), [first, second] = prediction.candidates;
 machine($('#nw-machine'), context, prediction, {pick: first.token, fresh});
 $('#nw-result').textContent = first.p > .95
  ? `Almost certain: “${displayToken(first.token)}”. Some beginnings leave only one sensible next word.`
  : `Not one answer but a chance for every word: “${displayToken(first.token)}” ${pct(first.p)}, “${displayToken(second.token)}” ${pct(second.p)}, and so on, adding up to 100%.`;
}
$('#nw-prompts').addEventListener('click', e => { const b = e.target.closest('[data-nw]'); if (!b) return; nwPrompt = +b.dataset.nw; renderNextWord(); $(`[data-nw="${nwPrompt}"]`).focus({preventScroll: true}); });

/* ---------- 02 · The loop ---------- */
// A new seed on every start, so 'surprise me' gives a different run each time you start over.
const freshSeed = () => (Math.random() * 4294967296) >>> 0;
const loop = {prompt: 0, context: [], start: 0, dice: false, seed: 7, roll: null, asked: 1, timer: null, playing: false};
const loopDone = () => loop.context.at(-1) === END || (loop.context.length > loop.start && loop.context.at(-1) === '.') || !predict(loop.context).candidates.length || loop.context.length - loop.start >= 60;
function loopRoll() { if (!loop.dice) { loop.roll = null; return; } const r = nextRandom(loop.seed); loop.seed = r.seed; loop.roll = r.value; }
function renderLoop(fresh = false, newest = -1, step = false) {
 chips($('#loop-prompts'), NEXT_WORD_PROMPTS.map(p => `${sentenceCase(p)} …`), loop.prompt, 'data-loop');
 $$('[data-loop-pick]').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.loopPick === 'dice') === loop.dice)));
 const done = loopDone(), visible = loop.context.filter(t => t !== END), added = visible.length - loop.start;
 $('#loop-text').innerHTML = grownText(visible, loop.start, fresh && newest >= 0) + (done ? '' : '<i class="caret" aria-hidden="true"></i>');
 if (done) { $('#loop-machine').innerHTML = '<p class="mc-finished">The box added a full stop. The sentence is finished.</p>'; $('#loop-machine').classList.remove('fresh'); }
 else { const prediction = predict(loop.context); machine($('#loop-machine'), loop.context, prediction, {pick: choose(prediction, loop.roll), choosable: true, fresh, newest, step}); }
 $('#loop-lab').classList.toggle('done', done);
 $('#loop-result').textContent = done ? `${added} word${added === 1 ? '' : 's'} added, one guess at a time. Nothing was planned ahead.`
  : added === 0 ? 'The highlighted word will be added next. Or click another word to choose it yourself.'
  : `Added “${displayToken(loop.context.at(-1))}”. The longer text goes back into the box.`;
 $('#loop-next').disabled = done || loop.playing; $('#loop-play').disabled = done && !loop.playing; $('#loop-play').textContent = loop.playing ? 'Pause' : '▶ Keep going';
 $('#loop-count').textContent = `The box has been asked ${loop.asked} time${loop.asked === 1 ? '' : 's'}`;
 placeLoopBack();
}
// The return arrow must end where the chosen word really goes: right after the last word of the text.
function placeLoopBack() {
 const back = $('#loop-lab .loop-back'), words = $$('#loop-machine .mc-word'), last = words[words.length - 1];
 if (!back || !last || !back.offsetParent) return;
 back.style.marginLeft = '0px'; back.style.setProperty('--rise', '0px');
 const start = back.getBoundingClientRect(), word = last.getBoundingClientRect();
 back.style.marginLeft = `${Math.max(0, word.right + 7 - start.left)}px`;
 back.style.setProperty('--rise', `${Math.max(0, start.top - word.bottom - 7)}px`);
}
addEventListener('resize', placeLoopBack);
function loopStop() { if (loop.timer !== null) clearTimeout(loop.timer); loop.timer = null; loop.playing = false; }
function loopStart(prompt = loop.prompt) { loopStop(); loop.prompt = prompt; loop.context = tokenize(NEXT_WORD_PROMPTS[prompt]); loop.start = loop.context.length; loop.seed = freshSeed(); loop.asked = 1; loopRoll(); renderLoop(true); }
function loopAdd(token) {
 if (loopDone()) return;
 const prediction = predict(loop.context), chosen = token ?? choose(prediction, loop.roll), box = $('#loop-machine');
 // remember where things are, so only what changes has to move
 const from = [...box.querySelectorAll('.mc-row')].find(r => r.dataset.choose === chosen)?.querySelector('.mc-cand')?.getBoundingClientRect();
 const before = new Map([...box.querySelectorAll('.mc-word')].map(w => [w.dataset.i, w.getBoundingClientRect()]));
 const bars = box.querySelector('.mc-bars'), oldBars = bars?.cloneNode(true), held = {height: box.offsetHeight, top: bars?.getBoundingClientRect().top};
 loop.context.push(chosen);
 if (!loopDone()) { loop.asked++; loopRoll(); }
 renderLoop(true, loop.context.length - 1, true);
 if (!reduced.matches && !loopDone()) loopMotion(box, before, from, oldBars, held);
}
// After a word is added: earlier words slide left to make room, the chosen word travels back along the
// return arrow to the end of the text, and the old guesses fade while the box is asked again.
function loopMotion(box, before, from, oldBars, held) {
 const ease = 'cubic-bezier(.2,.7,.2,1)';
 box.style.minHeight = `${held.height}px`;   // keep the lab the same height until the old guesses have gone
 for (const w of box.querySelectorAll('.mc-word:not(.new)')) {
  const was = before.get(w.dataset.i); if (!was) continue;
  const now = w.getBoundingClientRect(), dx = was.left - now.left, dy = was.top - now.top;
  if (dx || dy) w.animate([{transform: `translate(${dx}px,${dy}px)`}, {transform: 'none'}], {duration: 450, easing: ease});
 }
 const word = box.querySelector('.mc-word.new'), back = $('#loop-lab .loop-back');
 if (word && from) {
  const now = word.getBoundingClientRect(), sx = from.left - now.left, sy = from.top + from.height / 2 - (now.top + now.height / 2);
  const rail = back && back.offsetParent ? back.getBoundingClientRect().bottom - (now.top + now.height / 2) : null;
  const frames = rail === null || rail < 20
   ? [{transform: `translate(${sx}px,${sy}px)`}, {transform: 'none'}]
   : (() => { const down = Math.abs(rail - sy), across = Math.abs(sx), up = Math.abs(rail), all = down + across + up || 1;
      return [{transform: `translate(${sx}px,${sy}px)`, offset: 0}, {transform: `translate(${sx}px,${rail}px)`, offset: down / all}, {transform: `translate(0px,${rail}px)`, offset: (down + across) / all}, {transform: 'none', offset: 1}]; })();
  word.classList.add('flying');
  word.animate(frames, {duration: 900, easing: 'ease-in-out'}).finished.catch(() => {}).then(() => word.classList.remove('flying'));
 }
 const out = box.querySelector('.mc-out');
 if (oldBars && out) { oldBars.classList.add('mc-bars-old'); oldBars.setAttribute('aria-hidden', 'true'); oldBars.querySelectorAll('button').forEach(b => { b.disabled = true; b.removeAttribute('data-choose'); }); out.append(oldBars); oldBars.style.top = `${held.top - out.getBoundingClientRect().top}px`;
  oldBars.animate([{opacity: 1}, {opacity: 0}], {duration: 700, delay: 500, fill: 'forwards'}).finished.catch(() => {}).then(() => { oldBars.remove(); box.style.minHeight = ''; placeLoopBack(); }); }
 else box.style.minHeight = '';
}
function loopPlay() { loop.timer = setTimeout(() => { loop.timer = null; if (!loop.playing || document.hidden || loopDone()) { loopStop(); renderLoop(); return; } loopAdd(); if (loopDone()) { loopStop(); renderLoop(); } else loopPlay(); }, reduced.matches ? 250 : 3100); }
$('#loop-prompts').addEventListener('click', e => { const b = e.target.closest('[data-loop]'); if (b) loopStart(+b.dataset.loop); });
$('#loop-next').addEventListener('click', () => { loopStop(); loopAdd(); });
$('#loop-play').addEventListener('click', () => { if (loop.playing) { loopStop(); renderLoop(); return; } loop.playing = true; renderLoop(); loopPlay(); });
$('#loop-reset').addEventListener('click', () => loopStart());
$('#loop-machine').addEventListener('click', e => { const b = e.target.closest('[data-choose]'); if (!b || loop.playing) return; loopAdd(b.dataset.choose); });
$$('[data-loop-pick]').forEach(b => b.addEventListener('click', () => { loop.dice = b.dataset.loopPick === 'dice'; loopStart(); }));

/* ---------- 03 · Story completion ---------- */
const story = {prompt: 0, context: [], start: 0, dice: false, seed: 3, timer: null, playing: false, untilEnd: false, last: null};
const storyDone = () => story.context.at(-1) === END || !predict(story.context).candidates.length || story.context.length > 90;
function renderStory(fresh = false) {
 chips($('#story-prompts'), STORY_PROMPTS.map(p => `${sentenceCase(p)} …`), story.prompt, 'data-story');
 $$('[data-story-pick]').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.storyPick === 'dice') === story.dice)));
 const visible = story.context.filter(t => t !== END), written = visible.slice(story.start), done = storyDone();
 $('#story-text').innerHTML = grownText(visible, story.start, fresh) + (done ? '' : '<i class="caret" aria-hidden="true"></i>');
 $('#story-thought').innerHTML = story.last ? `The box just chose <b>“${esc(story.last.token === END ? 'stop here' : displayToken(story.last.token))}”</b> · it gave that word a ${pct(story.last.p)} chance` : 'Bold words are yours. Blue words are the box’s.';
 $('#story-count').textContent = `${written.length} word${written.length === 1 ? '' : 's'} written by the box`;
 $('#story-sentence').disabled = done || story.playing; $('#story-all').disabled = done && !story.playing; $('#story-all').textContent = story.playing && story.untilEnd ? 'Pause' : '▶ Write the whole story';
 $('#story-sentence').textContent = done ? 'The story is finished' : 'Write the next sentence →';
}
function storyStop() { if (story.timer !== null) clearTimeout(story.timer); story.timer = null; story.playing = false; }
function storyStart(prompt = story.prompt) { storyStop(); story.prompt = prompt; story.context = tokenize(STORY_PROMPTS[prompt]); story.start = story.context.length; story.seed = freshSeed(); story.last = null; renderStory(); }
function storyWord() {
 const prediction = predict(story.context);
 let u = null; if (story.dice) { const r = nextRandom(story.seed); story.seed = r.seed; u = r.value; }
 const token = choose(prediction, u);
 story.last = {token, p: prediction.all.find(c => c.token === token).p}; story.context.push(token);
 return token;
}
function storyRun() {
 story.timer = setTimeout(() => {
  story.timer = null;
  if (!story.playing || document.hidden || storyDone()) { storyStop(); renderStory(); return; }
  const token = storyWord(), sentenceEnd = token === '.' && !story.untilEnd;
  if (storyDone() || sentenceEnd) { storyStop(); renderStory(true); return; }
  renderStory(true); storyRun();
 }, reduced.matches ? 0 : 170);
}
function storyWrite(untilEnd) {
 if (story.playing) { storyStop(); renderStory(); return; }
 if (storyDone()) return;
 story.playing = true; story.untilEnd = untilEnd; renderStory();
 if (reduced.matches) { for (;;) { const t = storyWord(); if (storyDone() || (t === '.' && !untilEnd)) break; } storyStop(); renderStory(); } else storyRun();
}
$('#story-prompts').addEventListener('click', e => { const b = e.target.closest('[data-story]'); if (b) storyStart(+b.dataset.story); });
$('#story-sentence').addEventListener('click', () => storyWrite(false));
$('#story-all').addEventListener('click', () => storyWrite(true));
$('#story-reset').addEventListener('click', () => storyStart());
$$('[data-story-pick]').forEach(b => b.addEventListener('click', () => { story.dice = b.dataset.storyPick === 'dice'; storyStart(); }));

/* ---------- 04 · Chat is a completion with a script ---------- */
// Stages: 0 question typed · 1 script wrapped around it · 2 the box continues, word by word · 3 it chose "stop".
const chat = {question: 0, stage: 0, reply: [], timer: null};
function renderChat() {
 chips($('#chat-questions'), CHAT_EXAMPLES.map(([q]) => sentenceCase(q)), chat.question, 'data-chat');
 const question = CHAT_EXAMPLES[chat.question][0], qText = sentenceCase(question), words = chat.reply.filter(t => t !== END), stopped = chat.reply.at(-1) === END;
 const replyText = detokenize(words);
 $('#chat-window').innerHTML = `<div class="bubble user"><small>YOU</small><p>${esc(qText)}</p></div><div class="bubble assistant ${chat.stage < 2 ? 'waiting' : ''}"><small>ASSISTANT</small><p>${chat.stage < 2 ? '<span class="typing"><i></i><i></i><i></i></span>' : esc(replyText) + (stopped ? '' : '<i class="caret" aria-hidden="true"></i>')}</p></div>`;
 $('#chat-raw').innerHTML = `${chat.stage >= 1 ? '<span class="script">User: </span>' : ''}<span class="user">${esc(qText)}</span>${chat.stage >= 1 ? '<span class="script"><span class="nl">↵</span>\nAssistant:</span>' : ''}${chat.stage >= 2 ? ` <span class="model">${esc(replyText)}</span>` : ''}${stopped ? '<span class="stop">stop</span>' : chat.stage >= 1 ? '<i class="caret" aria-hidden="true"></i>' : ''}`;
 $('#chat-demo').dataset.stage = chat.stage;
 $('#chat-result').textContent = [
  'You type a question.',
  'The app wraps it in a script and stops right after “Assistant:”. This is all the box receives.',
  'The box continues the text, one word at a time. The likeliest continuation of this script is a reply.',
  'The box chose “stop”. The app shows the reply in a bubble. Same loop as before; only the beginning changed.'][chat.stage];
}
function chatStop() { if (chat.timer !== null) clearTimeout(chat.timer); chat.timer = null; }
function chatStep() {
 if (chat.stage === 0) chat.stage = 1;
 else if (chat.stage === 1) chat.stage = 2;
 else if (chat.stage === 2) {
  const context = [...chatPrompt(CHAT_EXAMPLES[chat.question][0]), ...chat.reply], token = choose(predict(context));
  chat.reply.push(token ?? END);
  if (chat.reply.at(-1) === END || chat.reply.length > 30) chat.stage = 3;
 }
 renderChat();
}
function chatRun() {
 if (chat.stage === 3) return;
 chat.timer = setTimeout(() => { chat.timer = null; if (document.hidden) return; chatStep(); chatRun(); }, reduced.matches ? 0 : chat.stage === 0 ? 1100 : chat.stage === 1 ? 1500 : 260);
}
function chatStart(question = chat.question) { chatStop(); chat.question = question; chat.stage = 0; chat.reply = []; renderChat(); chatRun(); }
$('#chat-questions').addEventListener('click', e => { const b = e.target.closest('[data-chat]'); if (b) chatStart(+b.dataset.chat); });
$('#chat-replay').addEventListener('click', () => chatStart());

document.addEventListener('visibilitychange', () => {
 if (document.hidden) { if (loop.playing) { loopStop(); renderLoop(); } if (story.playing) { storyStop(); renderStory(); } chatStop(); }
 else if (chat.stage < 3 && chat.timer === null && chatSeen) chatRun();
});

// The chat demo plays the first time it scrolls into view, so a first-time reader sees it happen.
let chatSeen = false;
renderNextWord(); loopStart(3); storyStart(0); renderChat();
if ('IntersectionObserver' in window) new IntersectionObserver((entries, observer) => { if (entries.some(e => e.isIntersecting)) { chatSeen = true; observer.disconnect(); chatStart(); } }, {threshold: .35}).observe($('#chat-demo'));
else { chatSeen = true; chatStart(); }
