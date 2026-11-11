import {INITIAL_TRAINING_STATE, cloneTrainingState, backward, trainStep} from './moe-training.mjs';

const root = document.getElementById('moe-training');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const NAMES = ['E1', 'E2', 'E3', 'E4'];
const fmt = (v, d = 3) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(d);

/* ---------- 1. Dense vs MoE forward/backward diagrams ---------- */

const box = (x, y, w, h, cls, title, sub = '') => `<g class="tn ${cls}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/><text x="${x + w / 2}" y="${y + h / 2 + (sub ? -3 : 4)}" text-anchor="middle">${title}</text>${sub ? `<text class="tn-sub" x="${x + w / 2}" y="${y + h / 2 + 11}" text-anchor="middle">${sub}</text>` : ''}</g>`;
const wire = (d, cls = 'on') => `<path class="tw ${cls}" d="${d}"/>`;
const badge = (x, y, text, cls = '') => `<text class="tb ${cls}" x="${x}" y="${y}" text-anchor="middle">${text}</text>`;

const markers = `<defs>
 <marker id="tw-head-fwd" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#7f9fe0"/></marker>
 <marker id="tw-head-back" viewBox="0 0 8 8" refX="1" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M8 0 L0 4 L8 8 z" fill="#d48a6e"/></marker>
</defs>`;

export function denseDiagram() {
 return `<svg viewBox="0 0 440 230" role="img" aria-label="Dense layer: forward and backward passes use every weight">${markers}
 ${wire('M58 115 H120')}${wire('M280 115 H318')}${wire('M352 115 H384')}
 ${box(18, 95, 40, 40, 'io', 'x')}
 ${box(120, 50, 160, 130, 'ffn grad', 'Dense FFN', 'W₁ · W₂')}
 ${box(318, 95, 34, 40, 'io', 'y')}${box(384, 95, 50, 40, 'loss', 'Loss')}
 ${badge(200, 158, '∇ every weight', 'back-only')}
 ${badge(200, 158, 'every weight runs', 'fwd-only')}
 </svg>`;
}

export function moeDiagram() {
 const ys = [52, 94, 136, 178], sel = [0, 2];
 const experts = ys.map((y, i) => box(100, y, 80, 30, sel.includes(i) ? 'expert grad' : 'expert idle', NAMES[i])).join('');
 const inWires = ys.map((y, i) => wire(`M44 115 C72 115 74 ${y + 15} 100 ${y + 15}`, sel.includes(i) ? 'on' : 'off')).join('');
 return `<svg viewBox="0 0 440 230" role="img" aria-label="MoE layer: gradients reach only the selected experts and the router through the gate values">
 ${wire('M27 95 V24 H70')}
 ${inWires}
 ${wire('M180 24 H226 V57')}${wire('M180 24 H252 V141')}
 ${wire('M180 67 H216')}${wire('M180 151 H242')}
 ${wire('M236 67 C272 67 276 115 296 115')}${wire('M262 151 C282 151 284 115 296 115')}
 ${wire('M324 115 H342')}${wire('M376 115 H390')}
 ${box(10, 95, 34, 40, 'io', 'x')}
 ${box(70, 8, 110, 32, 'router grad', 'Router Wᵣ')}
 ${experts}
 <g class="tn mul"><circle cx="226" cy="67" r="10"/><text x="226" y="71" text-anchor="middle">×</text></g>
 <g class="tn mul"><circle cx="252" cy="151" r="10"/><text x="252" y="155" text-anchor="middle">×</text></g>
 <text class="tg" x="234" y="44">g₁</text><text class="tg" x="260" y="128">g₃</text>
 <g class="tn sum"><circle cx="310" cy="115" r="14"/><text x="310" y="120" text-anchor="middle">Σ</text></g>
 ${box(342, 95, 34, 40, 'io', 'y')}${box(390, 95, 46, 40, 'loss', 'Loss')}
 ${badge(140, 220, 'E2, E4 idle · no ∇', 'muted-b')}
 </svg>`;
}

function setupCompare() {
 const wrap = root.querySelector('#train-compare');
 wrap.querySelector('[data-diagram="dense"]').innerHTML = denseDiagram();
 wrap.querySelector('[data-diagram="moe"]').innerHTML = moeDiagram();
 const buttons = [...wrap.querySelectorAll('[data-pass]')];
 const caption = wrap.querySelector('#train-pass-caption');
 const captions = {
  forward: 'Forward: the dense layer runs all its weights. The MoE token runs the router, then only E1 and E3, scaled by gates g₁ and g₃.',
  backward: 'Backward: the gradient retraces exactly the forward path. Dense updates every weight. MoE updates E1, E3 and — through g₁ and g₃ — the router. E2 and E4 get nothing from this token.'
 };
 const set = pass => {
  wrap.dataset.pass = pass;
  buttons.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.pass === pass)));
  caption.textContent = captions[pass];
 };
 buttons.forEach(b => b.addEventListener('click', () => set(b.dataset.pass)));
 set('forward');
}

/* ---------- 2. Router training lab ---------- */

const S = 300, R = 1.45, px = v => S / 2 + v / R * (S / 2), py = v => S / 2 - v / R * (S / 2);
const arrow = (x0, y0, x1, y1, cls, extra = '') => `<line class="${cls}" x1="${px(x0)}" y1="${py(y0)}" x2="${px(x1)}" y2="${py(y1)}" marker-end="url(#tl-head)" ${extra}/>`;

const lab = {state: cloneTrainingState(), mode: 'top2', trainExperts: false, balance: false, step: 0, history: [], everRan: new Set(), timer: 0, visible: true, lastRoute: '', note: ''};

function planeSvg(g) {
 const {state} = lab, selected = new Set(g.selected);
 let tip = [0, 0], chain = '';
 for (const i of g.selected) {
  const next = [tip[0] + g.gates[i] * state.experts[i][0], tip[1] + g.gates[i] * state.experts[i][1]];
  chain += `<line class="tl-part ex-${i}" x1="${px(tip[0])}" y1="${py(tip[1])}" x2="${px(next[0])}" y2="${py(next[1])}"/>`;
  tip = next;
 }
 const experts = state.experts.map((e, i) => `${arrow(0, 0, e[0], e[1], `tl-expert ex-${i}${selected.has(i) ? ' sel' : ''}`)}<text class="tl-name ex-${i}${selected.has(i) ? ' sel' : ''}" x="${px(e[0] * 1.13)}" y="${py(e[1] * 1.13) + 4}" text-anchor="middle">${NAMES[i]}</text>`).join('');
 const t = state.target;
 return `<svg viewBox="0 0 ${S} ${S}" role="img" aria-label="Expert output vectors, the mixed output y and the target">
 <defs><marker id="tl-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L8 4 L0 8 z" fill="context-stroke"/></marker></defs>
 <line class="tl-axis" x1="0" y1="${S / 2}" x2="${S}" y2="${S / 2}"/><line class="tl-axis" x1="${S / 2}" y1="0" x2="${S / 2}" y2="${S}"/>
 ${experts}
 ${chain}
 <line class="tl-err" x1="${px(g.y[0])}" y1="${py(g.y[1])}" x2="${px(t[0])}" y2="${py(t[1])}"/>
 <circle class="tl-target" cx="${px(t[0])}" cy="${py(t[1])}" r="9"/><circle class="tl-target-dot" cx="${px(t[0])}" cy="${py(t[1])}" r="2.5"/>
 <text class="tl-tlabel" x="${px(t[0]) + 13}" y="${py(t[1]) - 9}">target</text>
 <circle class="tl-y" cx="${px(g.y[0])}" cy="${py(g.y[1])}" r="5"/><text class="tl-ylabel" x="${px(g.y[0]) + 9}" y="${py(g.y[1]) + 15}">y</text>
 </svg>`;
}

function sparkline() {
 const h = lab.history.slice(-120);
 if (h.length < 2) return '<svg viewBox="0 0 200 40" aria-hidden="true"></svg>';
 const max = Math.max(...h, 1e-6);
 const pts = h.map((v, i) => `${(i / (h.length - 1)) * 200},${38 - v / max * 34}`).join(' ');
 return `<svg viewBox="0 0 200 40" aria-hidden="true"><polyline points="${pts}"/></svg>`;
}

function describe(g) {
 const {mode} = lab;
 if (mode === 'top1') return 'With one expert and a renormalized gate, g = p / p = 1 no matter what the router says. The loss has no path back to the router: every router gradient is exactly 0. This is why Switch Transformer multiplies by the raw probability instead.';
 const idleNeverRan = [0, 1, 2, 3].filter(i => !lab.everRan.has(i));
 if (lab.note) return lab.note;
 const bestIdle = idleNeverRan.find(i => i === 2);
 if (bestIdle !== undefined && !lab.balance) return 'E3 points almost straight at the target, but it has never been selected. It never runs, so neither E3 nor its router score receives any gradient. Try adding the load-balancing loss.';
 if (lab.step === 0) return 'Press Step to run one forward pass, backpropagate the loss and update the router. Arrows on each bar show which way its logit moves.';
 return `Selected: ${g.selected.map(i => NAMES[i]).join(' + ')}. Only these experts and the router logits that feed their gates are updated by the task loss.`;
}

function renderLab() {
 const g = backward(lab.state, lab.mode, {balance: lab.balance ? .05 : 0});
 const lr = .6;
 root.querySelector('#tl-plane').innerHTML = planeSvg(g);
 root.querySelector('#tl-bars').innerHTML = g.probs.map((p, i) => {
  const sel = g.selected.includes(i), move = -lr * g.logits[i], dir = Math.abs(move) < 5e-4 ? 'none' : move > 0 ? 'up' : 'down';
  return `<div class="tl-bar ex-${i}${sel ? ' sel' : ''}">
  <span class="tl-bar-name">${NAMES[i]}</span>
  <span class="tl-bar-track"><i style="width:${(p * 100).toFixed(1)}%"></i></span>
  <span class="tl-bar-p">p ${fmt(p, 2)}</span>
  <span class="tl-bar-state">${sel ? `runs · g ${fmt(g.gates[i], 2)}` : 'idle'}</span>
  <span class="tl-bar-grad ${dir}" title="Change to this router logit on the next step">${dir === 'none' ? '· no signal' : `${dir === 'up' ? '▲' : '▼'} ${fmt(Math.abs(move), 3)}`}</span>
 </div>`;
 }).join('');
 root.querySelector('#tl-loss').textContent = fmt(g.loss, 3);
 root.querySelector('#tl-step').textContent = String(lab.step);
 root.querySelector('#tl-spark').innerHTML = sparkline();
 root.querySelector('#tl-message').textContent = describe(g);
 root.querySelector('#tl-mode-help').textContent = {
  top2: 'Two experts run. Their gates are renormalized to sum to 1, so the router learns how to split the work between them.',
  switch: 'One expert runs, scaled by its raw softmax probability p (Switch Transformer style). The router can now learn, and the softmax denominator touches every logit.',
  top1: 'One expert runs with gate = 1. A useful counterexample: the router cannot learn from the task loss at all.'
 }[lab.mode];
 root.querySelectorAll('[data-tl-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.tlMode === lab.mode)));
 root.querySelector('#tl-play').textContent = lab.timer ? 'Pause' : 'Train ▶';
 root.querySelector('#tl-play').setAttribute('aria-pressed', String(!!lab.timer));
}

function doStep() {
 const before = backward(lab.state, lab.mode, {balance: lab.balance ? .05 : 0});
 before.selected.forEach(i => lab.everRan.add(i));
 if (!lab.history.length) lab.history.push(before.loss);
 lab.state = trainStep(lab.state, lab.mode, {lr: .6, trainExperts: lab.trainExperts, balance: lab.balance ? .05 : 0}).state;
 lab.step++;
 const after = backward(lab.state, lab.mode, {balance: lab.balance ? .05 : 0});
 lab.history.push(after.loss);
 const route = after.selected.join(','), prev = before.selected.join(',');
 if (route !== prev) {
  const gained = after.selected.filter(i => !before.selected.includes(i)), lost = before.selected.filter(i => !after.selected.includes(i));
  lab.note = `Route changed at step ${lab.step}: ${gained.map(i => NAMES[i]).join(', ')} replaced ${lost.map(i => NAMES[i]).join(', ')}. ${gained.includes(2) ? 'Now that E3 runs, the task loss can finally reward it — watch its probability climb.' : lab.balance ? 'On one repeated token the balancing loss can only spread the work, so it keeps rotating the partner expert.' : ''}`;
  lab.noteUntil = lab.step + 25;
 } else if (lab.noteUntil && lab.step > lab.noteUntil) { lab.note = ''; lab.noteUntil = 0; }
}

function stop() { clearInterval(lab.timer); lab.timer = 0; }
function play() {
 if (lab.timer) { stop(); renderLab(); return; }
 if (reduced.matches) { for (let i = 0; i < 40; i++) doStep(); renderLab(); return; }
 lab.timer = setInterval(() => {
  if (!lab.visible || document.hidden) return;
  doStep(); renderLab();
  if (lab.step >= 400) { stop(); renderLab(); }
 }, 90);
 renderLab();
}
function reset() {
 stop();
 Object.assign(lab, {state: cloneTrainingState(INITIAL_TRAINING_STATE), step: 0, history: [], everRan: new Set(), note: '', noteUntil: 0});
 renderLab();
}

function setupLab() {
 root.querySelector('#tl-stepbtn').addEventListener('click', () => { stop(); doStep(); renderLab(); });
 root.querySelector('#tl-play').addEventListener('click', play);
 root.querySelector('#tl-reset').addEventListener('click', reset);
 root.querySelectorAll('[data-tl-mode]').forEach(b => b.addEventListener('click', () => { lab.mode = b.dataset.tlMode; reset(); }));
 root.querySelector('#tl-experts').addEventListener('change', e => { lab.trainExperts = e.target.checked; renderLab(); });
 root.querySelector('#tl-balance').addEventListener('change', e => { lab.balance = e.target.checked; lab.note = ''; renderLab(); });
 if ('IntersectionObserver' in window) new IntersectionObserver(([entry]) => { lab.visible = entry.isIntersecting; }).observe(root.querySelector('.tl-lab'));
 renderLab();
}

if (root) { setupCompare(); setupLab(); }
