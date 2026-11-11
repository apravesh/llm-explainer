// Toy MoE training model: one token, four experts whose outputs are 2-D vectors.
// Pure functions so the arithmetic can be tested independently of the page.

export const TRAINING_MODES = {
 top2: {k: 2, renormalize: true, label: 'Top-2 · renormalized gates'},
 switch: {k: 1, renormalize: false, label: 'Top-1 · raw probability'},
 top1: {k: 1, renormalize: true, label: 'Top-1 · gate fixed at 1'}
};

export const INITIAL_TRAINING_STATE = Object.freeze({
 logits: [1.2, 0.9, 0.15, -0.4],
 experts: [[1.05, 0.25], [-0.35, -0.55], [0.22, 1.12], [-1.0, 0.1]],
 target: [0.3, 1.0]
});

export function cloneTrainingState(state = INITIAL_TRAINING_STATE) {
 return {logits: [...state.logits], experts: state.experts.map(e => [...e]), target: [...state.target]};
}

export function softmax(values) {
 const max = Math.max(...values), ex = values.map(v => Math.exp(v - max)), sum = ex.reduce((a, b) => a + b, 0);
 return ex.map(v => v / sum);
}

// Top-k by logit; ties resolved by lower index, like a stable sort.
export function topK(values, k) {
 return values.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k).map(([, i]) => i).sort((a, b) => a - b);
}

export function forward(state, mode = 'top2') {
 const {k, renormalize} = TRAINING_MODES[mode];
 const probs = softmax(state.logits), selected = topK(state.logits, k);
 const selectedMass = selected.reduce((s, i) => s + probs[i], 0);
 const gates = probs.map((p, i) => selected.includes(i) ? (renormalize ? p / selectedMass : p) : 0);
 const y = [0, 0];
 for (const i of selected) { y[0] += gates[i] * state.experts[i][0]; y[1] += gates[i] * state.experts[i][1]; }
 const err = [y[0] - state.target[0], y[1] - state.target[1]];
 const loss = .5 * (err[0] ** 2 + err[1] ** 2);
 return {probs, selected, gates, y, err, loss};
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1];

// Backpropagation for the toy layer.
// Task loss L = ½‖y − t‖², with y = Σ_selected gᵢ·Eᵢ.
// Optional Switch-style balancing loss: α·N·Σ fᵢ·pᵢ, where fᵢ is the (non-differentiable)
// share of routed slots expert i received and pᵢ the full-softmax probability.
export function backward(state, mode = 'top2', {balance = 0} = {}) {
 const f = forward(state, mode), {renormalize, k} = TRAINING_MODES[mode], n = state.logits.length;
 const dGate = state.logits.map((_, i) => f.selected.includes(i) ? dot(f.err, state.experts[i]) : 0);
 const taskLogit = new Array(n).fill(0);
 if (renormalize) {
  // Gates are a softmax over the selected logits only.
  const mean = f.selected.reduce((s, i) => s + f.gates[i] * dGate[i], 0);
  for (const i of f.selected) taskLogit[i] = f.gates[i] * (dGate[i] - mean);
 } else {
  // Gates are the full-softmax probabilities, so every logit is touched through the denominator.
  const mean = f.probs.reduce((s, p, i) => s + p * dGate[i], 0);
  for (let i = 0; i < n; i++) taskLogit[i] = f.probs[i] * (dGate[i] - mean);
 }
 const share = state.logits.map((_, i) => f.selected.includes(i) ? 1 / k : 0);
 const dProbBalance = share.map(s => balance * n * s);
 const meanBalance = f.probs.reduce((s, p, i) => s + p * dProbBalance[i], 0);
 const balanceLogit = f.probs.map((p, i) => p * (dProbBalance[i] - meanBalance));
 const balanceLoss = balance * n * share.reduce((s, v, i) => s + v * f.probs[i], 0);
 const logits = taskLogit.map((g, i) => g + balanceLogit[i]);
 const experts = state.experts.map((_, i) => f.selected.includes(i) ? [f.gates[i] * f.err[0], f.gates[i] * f.err[1]] : [0, 0]);
 return {...f, dGate, taskLogit, balanceLogit, logits, experts, balanceLoss};
}

export function trainStep(state, mode = 'top2', {lr = .6, trainExperts = false, balance = 0} = {}) {
 const g = backward(state, mode, {balance}), next = cloneTrainingState(state);
 next.logits = state.logits.map((z, i) => z - lr * g.logits[i]);
 if (trainExperts) next.experts = state.experts.map((e, i) => [e[0] - lr * g.experts[i][0], e[1] - lr * g.experts[i][1]]);
 return {state: next, grads: g};
}
