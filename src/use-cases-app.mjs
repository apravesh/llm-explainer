import {USE_CASES, SEARCH_DEMO, typingUnits, queryPrompt, searchPrompt} from './use-cases-data.mjs';

const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const CLASS = {app: 'script', user: 'user', model: 'model', tool: 'tool'};
// The raw text stream: every segment in its colour, line breaks shown as ↵ so the reader sees they are characters too.
const rawSegment = (who, text) => `<span class="${CLASS[who]}">${esc(text).replace(/\n/g, '<span class="nl">↵</span>\n')}</span>`;
const caret = '<i class="caret" aria-hidden="true"></i>';

/* ---------- 02 · Use cases: two live labs (creative writing, chat) and two scripted illustrations ---------- */
const use = {index: 0, seg: 0, units: 0, timer: null, started: false};
const useCase = () => USE_CASES[use.index];
const useDone = () => use.seg >= useCase().segments.length;
// Text revealed so far for segment i: null if not reached, partial while the model is typing it.
function revealed(i) {
 const s = useCase().segments[i];
 if (i < use.seg) return s.text;
 if (i > use.seg) return null;
 return s.who === 'model' ? typingUnits(s.text).slice(0, use.units).join('') : s.text;
}
function screen() {
 const c = useCase(), segs = c.segments, text = i => revealed(i), typing = i => i === use.seg && segs[i].who === 'model';
 if (c.key === 'code') { const more = text(1), accepted = useDone();
  const lines = (segs[0].text + (more ?? '')).split('\n'), given = segs[0].text.split('\n').length - 1;
  return `<div class="code-editor"><div class="code-tab">helpers.py</div><pre>${lines.map((line, n) => `<span class="ln">${n + 1}</span><span class="${n >= given && more !== null ? (accepted ? 'accepted' : 'ghost') : ''}">${esc(line) || ' '}</span>${n === lines.length - 1 && !accepted ? caret : ''}`).join('\n')}</pre><p class="code-hint">${more === null ? 'You stop typing…' : accepted ? 'Suggestion accepted.' : 'Grey text is the model’s suggestion.'}</p></div>`; }
 // agent: a task card and a growing list of request → result steps
 const rows = [];
 segs.forEach((s, i) => { const t = text(i); if (t === null) return;
  if (s.who === 'model' && s.kind === 'action') rows.push(`<li class="act"><small>MODEL ASKS FOR</small><code>${esc(t)}</code>${typing(i) ? caret : ''}</li>`);
  else if (s.who === 'tool') rows.push(`<li class="res"><small>APP RUNS IT · RESULT</small><span>${esc(t)}</span></li>`);
  else if (s.who === 'model') rows.push(`<li class="fin"><small>MODEL REPORTS</small><span>${esc(t)}</span>${typing(i) ? caret : ''}</li>`); });
 return `<div class="agent-card"><p class="agent-task"><small>YOUR TASK</small>${esc(segs[1].text)}</p><ol class="agent-steps">${rows.join('') || '<li class="idle">Waiting for the model’s first request…</li>'}</ol></div>`;
}
function renderUse() {
 const c = useCase();
 $('#use-tabs').innerHTML = USE_CASES.map((u, i) => `<button type="button" data-use="${i}" aria-pressed="${i === use.index}"><small>${u.kicker}</small>${u.label}</button>`).join('');
 $('#use-label').textContent = c.note;
 $('#use-facts').innerHTML = `<div><small>YOU GIVE IT</small>${esc(c.give)}</div><div><small>IT WRITES</small>${esc(c.writes)}</div><div><small>WHERE YOU MEET IT</small>${esc(c.where)}</div>`;
 // one panel at a time: a live lab (its id is c.lab) or the scripted two-view illustration
 for (const panel of $$('#uses-lab .use-panel')) panel.hidden = panel.id !== (c.lab || 'use-scripted');
 $('#use-point').textContent = c.point; $('#use-point').hidden = !c.point; $('#use-controls').hidden = !!c.lab;
 if (c.lab) return;
 $('#use-screen').innerHTML = screen();
 $('#use-raw').innerHTML = c.segments.map((s, i) => { const t = revealed(i); return t === null ? '' : rawSegment(s.who, t); }).join('') + (useDone() ? '<span class="stop">stop</span>' : caret);
 $('#use-raw').scrollTop = $('#use-raw').scrollHeight;
}
function useStop() { if (use.timer !== null) clearTimeout(use.timer); use.timer = null; }
function useTick() {
 if (useDone()) return;
 const s = useCase().segments[use.seg], total = s.who === 'model' ? typingUnits(s.text).length : 0;
 const delay = reduced.matches ? 0 : s.who === 'model' ? (use.units === 0 ? 650 : use.units >= total ? 500 : 95) : s.who === 'tool' ? 800 : s.who === 'user' ? 550 : 250;
 use.timer = setTimeout(() => {
  use.timer = null; if (document.hidden) return;
  if (s.who === 'model' && use.units < total) use.units++; else { use.seg++; use.units = 0; }
  renderUse(); useTick();
 }, delay);
}
function useStart(index = use.index) {
 useStop(); use.index = index; use.seg = 0; use.units = 0; use.started = true; renderUse(); useTick();
 document.dispatchEvent(new CustomEvent('usecase:shown', {detail: useCase().key}));   // the live labs listen for this
}
// Old links: #stories opens Creative writing and #chat opens Chat.
function useFromHash() {
 const index = USE_CASES.findIndex(u => u.lab && `#${u.lab}` === location.hash);
 if (index < 0) return;
 useStart(index); $('#uses-lab').scrollIntoView();
}
addEventListener('hashchange', useFromHash);
$('#use-tabs').addEventListener('click', e => { const b = e.target.closest('[data-use]'); if (!b) return; useStart(+b.dataset.use); $(`[data-use="${use.index}"]`).focus({preventScroll: true}); });
$('#use-replay').addEventListener('click', () => useStart());

/* ---------- 03 · Chat with web search (a simple RAG setup) ---------- */
const STEPS = {
 on: [
  {key: 'ask', node: 'You ask', text: 'You ask about something that happened yesterday. The model was trained long before that.'},
  {key: 'query', node: 'Model writes a search query', text: 'First the app asks the model for a search query. Writing a query is just another continuation.'},
  {key: 'search', node: 'App runs the search', text: 'The app, not the model, sends that query to a search engine and gets back a few results.'},
  {key: 'paste', node: 'App pastes results into the prompt', text: 'The app builds a new prompt: the search results first, then your question, then “Assistant:”.'},
  {key: 'answer', node: 'Model writes the answer', text: 'The model continues the text. Everything it needs is now right there in the prompt, so it can answer and point to its sources.'},
  {key: 'done', node: 'You get an answer with sources', text: 'The reply appears with numbered sources, so you can check where each fact came from.'}],
 off: [
  {key: 'ask', node: 'You ask', text: 'You ask about something that happened yesterday. The model was trained long before that.'},
  {key: 'paste', node: 'App adds the chat script', text: 'With search off, the prompt is only your question wrapped in the usual script.'},
  {key: 'answer', node: 'Model writes the answer', text: 'Nothing in the prompt mentions the race, and nothing in its training could. A well-behaved model says it does not know. A careless one may invent a winner.'},
  {key: 'done', node: 'You get “I do not know”', text: 'Same model, same question. The only thing missing was the text it needed to read.'}]
};
const search = {mode: 'on', step: 0, units: 0, timer: null, playing: false, started: false};
const steps = () => STEPS[search.mode], stepKey = () => steps()[search.step].key, reached = key => steps().findIndex(s => s.key === key) <= search.step && steps().some(s => s.key === key);
const typedPart = (text, key) => stepKey() === key ? typingUnits(text).slice(0, search.units).join('') : reached(key) ? text : null;
const typingTotal = () => stepKey() === 'query' ? typingUnits(SEARCH_DEMO.query).length : stepKey() === 'answer' ? typingUnits(search.mode === 'on' ? SEARCH_DEMO.answer : SEARCH_DEMO.answerWithoutSearch).length : 0;
const searchTyping = () => search.units < typingTotal();
function renderSearch(fresh = false) {
 const d = SEARCH_DEMO, on = search.mode === 'on', key = stepKey(), lab = $('#search-lab');
 lab.dataset.step = key; lab.classList.toggle('fresh', fresh && !reduced.matches);
 $$('[data-search-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.searchMode === search.mode)));
 $('#search-flow').innerHTML = steps().map((s, i) => `<span class="flow-node ${i === search.step ? 'current' : i < search.step ? 'done' : ''}"><small>${i + 1}</small>${s.node}</span>${i < steps().length - 1 ? '<span class="flow-wire" aria-hidden="true"></span>' : ''}`).join('');
 const answerText = on ? d.answer : d.answerWithoutSearch, answer = typedPart(answerText, 'answer'), finished = key === 'done';
 $('#search-chat').innerHTML = `<div class="bubble user"><small>YOU</small><p>${esc(d.question)}</p></div><div class="bubble assistant ${answer === null ? 'waiting' : ''}"><small>ASSISTANT${on && reached('query') && answer === null ? ' · SEARCHING THE WEB' : ''}</small><p>${answer === null ? '<span class="typing"><i></i><i></i><i></i></span>' : esc(answer) + (finished ? '' : caret)}</p>${finished && on ? `<p class="sources">${d.sources.map(n => `<span title="${esc(d.results[n - 1].site)}">[${n}] ${esc(d.results[n - 1].site)}</span>`).join('')}</p>` : ''}</div>`;
 const query = on ? typedPart(d.query, 'query') : null, searched = on && reached('search');
 $('#search-engine').className = `search-engine ${on ? '' : 'off'} ${searched ? 'searched' : ''}`;
 $('#search-engine').innerHTML = !on ? '<p class="engine-off">Search is off. Nothing is looked up.</p>'
  : `<div class="search-box"><span aria-hidden="true">⌕</span><span>${query === null ? '<em>waiting for a query</em>' : esc(query) + (key === 'query' ? caret : '')}</span></div>${searched ? d.results.map((r, i) => `<div class="search-hit ${reached('paste') ? 'used' : ''}" style="--d:${i * 180}ms"><small>[${i + 1}] ${esc(r.site)}</small><b>${esc(r.title)}</b><span>${esc(r.snippet)}</span></div>`).join('') : '<p class="engine-off">No results yet.</p>'}`;
 // The prompt pane shows prompt 1 (ask for a query) until the results arrive, then prompt 2 (results + question).
 const second = reached('paste'), segments = !reached(on ? 'query' : 'paste') ? [] : second ? searchPrompt(d.question, on ? d.results : []) : queryPrompt(d.question);
 $('#search-raw-label').textContent = !segments.length ? 'WHAT THE MODEL RECEIVES AND WRITES' : on ? (second ? 'PROMPT 2 OF 2 · RESULTS PASTED IN ABOVE YOUR QUESTION' : 'PROMPT 1 OF 2 · ASK THE MODEL FOR A SEARCH QUERY') : 'THE ONLY PROMPT · NO RESULTS TO READ';
 const written = second ? answer : query;
 $('#search-raw').innerHTML = !segments.length ? '<span class="idle">Nothing has been sent to the model yet.</span>'
  : segments.map((s, i) => `<span class="seg-in" style="--d:${i * 90}ms">${rawSegment(s.who, s.text)}</span>`).join('') + (written ? rawSegment('model', written) : '') + (finished || (!second && reached('search')) ? '<span class="stop">stop</span>' : caret);
 $('#search-raw').scrollTop = $('#search-raw').scrollHeight;
 $('#search-result').textContent = steps()[search.step].text;
 const last = search.step === steps().length - 1;
 $('#search-next').disabled = last || search.playing; $('#search-next').textContent = last ? 'Finished' : searchTyping() ? 'Skip typing →' : 'Next step →';
 $('#search-play').disabled = last && !search.playing; $('#search-play').textContent = search.playing ? 'Pause' : '▶ Play';
}
function searchStop() { if (search.timer !== null) clearTimeout(search.timer); search.timer = null; search.playing = false; }
// Type the model's words for the current step; when playing, move on to the next step afterwards.
function searchTick() {
 const last = search.step === steps().length - 1;
 if (!searchTyping() && (!search.playing || last)) { if (last) search.playing = false; renderSearch(); return; }
 const delay = reduced.matches ? 0 : searchTyping() ? (search.units === 0 ? 700 : 110) : stepKey() === 'search' ? 2200 : stepKey() === 'paste' ? 2600 : 1500;
 search.timer = setTimeout(() => {
  search.timer = null; if (document.hidden) { search.playing = false; renderSearch(); return; }
  if (searchTyping()) { search.units++; renderSearch(); } else { search.step++; search.units = 0; renderSearch(true); }
  searchTick();
 }, delay);
}
function searchGo(step) { if (search.timer !== null) clearTimeout(search.timer); search.timer = null; search.step = step; search.units = 0; renderSearch(true); searchTick(); }
function searchStart(mode = search.mode) { searchStop(); search.mode = mode; search.started = true; searchGo(0); }
$('#search-next').addEventListener('click', () => { if (searchTyping()) { if (search.timer !== null) clearTimeout(search.timer); search.timer = null; search.units = typingTotal(); renderSearch(); } else searchGo(search.step + 1); });
$('#search-play').addEventListener('click', () => { if (search.playing) { searchStop(); renderSearch(); return; } search.playing = true; renderSearch(); if (search.timer === null) searchTick(); });
$('#search-reset').addEventListener('click', () => searchStart());
$$('[data-search-mode]').forEach(b => b.addEventListener('click', () => searchStart(b.dataset.searchMode)));

document.addEventListener('visibilitychange', () => {
 if (document.hidden) { useStop(); searchStop(); renderSearch(); }
 else { if (use.started && !useDone() && use.timer === null) useTick(); if (search.started && search.timer === null && searchTyping()) searchTick(); }
});
renderUse(); renderSearch();
// Each lab starts the first time it scrolls into view, so a first-time reader sees it happen.
const onFirstView = (el, run) => { if (!('IntersectionObserver' in window)) { run(); return; } new IntersectionObserver((entries, o) => { if (entries.some(e => e.isIntersecting)) { o.disconnect(); run(); } }, {threshold: .3}).observe(el); };
useFromHash();
onFirstView($('#uses-lab'), () => { if (!use.started) useStart(); });
onFirstView($('#search-lab'), () => { search.started = true; search.playing = true; renderSearch(); searchTick(); });
