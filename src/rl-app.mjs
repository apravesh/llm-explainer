import {ANSWERS,initialRLState,advanceRL,initialReinforceState,advanceReinforce,initialDpoState,advanceDpo,probabilities,inferSnapshot,clippedObjective,reinforceObjective,dpoObjective,random} from './rl-engine.mjs';
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)],fmt=(x,n=3)=>x.toFixed(n);
// Three methods share one page. `last` is the phase that completes a round.
const ALGOS={
 reinforce:{last:4,init:mode=>initialReinforceState(mode),advance:advanceReinforce,delays:[3200,3200,1700,1800,3000],
  note:'REINFORCE: one model. It writes four answers, each is compared with the group average, and one gradient step makes the better ones more likely. Then it starts over with fresh answers.',
  overview:[['INPUT','Prompt'],['INFERENCE','Model runs 4×'],['ROLLOUTS','4 responses'],['CHECK','Reward function'],['COMPARE','Group average'],['TRAIN','One update']],back:['then the updated model writes four new answers','updated: the next four answers come from the new weights'],
  labels:['Run the model 4 times →','Score the answers →','Compare with the group →','Update the model →','Run the next group →']},
 grpo:{last:7,init:mode=>initialRLState(mode),advance:advanceRL,delays:[3200,3200,1700,1800,1500,1500,1500,3200],
  note:'GRPO adds three things to REINFORCE: a frozen copy that writes the answers, a ratio that tracks how far the learner has moved from it, and clipping that pauses a push once it has moved 20%. Together they let one group be reused for several updates.',
  overview:[['INPUT','Prompt'],['INFERENCE','Frozen copy runs 4×'],['ROLLOUTS','4 responses'],['CHECK','Reward function'],['COMPARE','Group average'],['TRAIN','Update learner ×3']],back:['after 3 updates, copy the learner’s weights into the frozen copy and start the next round','weights copied: the next round samples from the improved model'],
  labels:['Run the frozen copy 4 times →','Score the answers →','Compare with the group →','Update the learner 1 / 3 →','Update the learner 2 / 3 →','Update the learner 3 / 3 →','Copy weights to the frozen copy →','Run the next group →']},
 dpo:{last:4,init:()=>initialDpoState(),advance:advanceDpo,delays:[2600,2600,2200,2000,2600],
  note:'DPO needs no sampling and no reward function. It learns from pairs where one answer was marked as preferred, and pushes the preferred answer above the rejected one, measured against a frozen reference copy.',
  overview:[['DATA','Prompt'],['DATA','Preferred vs rejected'],['COMPARE','Learner vs reference'],['LOSS','Does it prefer the right one?'],['TRAIN','One update']],back:['repeat on the same pairs; the reference copy never changes','updated: the same pairs are scored again with the new weights'],
  labels:['Load the preference pairs →','Compare with the reference →','Measure the loss →','Update the learner →','Score the pairs again →']}
};
let algo='reinforce',state=ALGOS[algo].init('correct'),servingVersion=0,inferenceSeed=981,inferenceResult=null,inferencePin=null,tally=[0,0,0],timer=null,epoch=0,playing=false,pins=[],history=[probabilities(state.learner)[1]],roundMarks=[];
const reduced=matchMedia('(prefers-reduced-motion: reduce)'),lastPhase=()=>ALGOS[algo].last;
function bars(logits,ticks=null){return probabilities(logits).map((p,i)=>`<div class="rl-probability ${i===1?'correct-answer':''}"><code>${ANSWERS[i]}</code><span><i style="width:${100*p}%"></i>${ticks?`<b class="tick" style="left:${100*ticks[i]}%" title="frozen copy: ${fmt(100*ticks[i],1)}%"></b>`:''}</span><b>${fmt(100*p,1)}%</b></div>`).join('');}
// The sampling strip: three segments sized by probability; each pin is one uniform random number.
function strip(probs,marks){return `<div class="strip-track">${probs.map((p,i)=>`<span class="seg ${i===1?'correct-answer':''}" style="flex-grow:${p}"><b>${ANSWERS[i]}</b></span>`).join('')}${marks.map((m,i)=>`<i class="pin" style="left:${100*m.u}%;--d:${i*160}ms" title="random number ${fmt(m.u,3)} lands on ${m.answer}"><em>${m.label}</em></i>`).join('')}</div>`;}
function curve(){const w=300,h=70,n=Math.max(history.length-1,1),x=i=>6+(w-12)*i/n,y=p=>h-8-(h-18)*p;
 return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Probability of answering 4 after each of ${history.length-1} updates: now ${fmt(history.at(-1)*100,1)} percent"><line class="top" x1="6" y1="${y(1)}" x2="${w-6}" y2="${y(1)}"/><line x1="6" y1="${y(0)}" x2="${w-6}" y2="${y(0)}"/>${roundMarks.map(i=>`<line class="mark" x1="${x(i)}" y1="${y(0)}" x2="${x(i)}" y2="${y(1)}"/>`).join('')}${history.length>1?`<polyline points="${history.map((p,i)=>`${x(i).toFixed(1)},${y(p).toFixed(1)}`).join(' ')}"/>`:''}<circle cx="${x(history.length-1).toFixed(1)}" cy="${y(history.at(-1)).toFixed(1)}" r="3.5"/><text x="${w-6}" y="${y(1)-3}" text-anchor="end">100%</text></svg>`;}
function cancel(){epoch++;clearTimeout(timer);timer=null;playing=false;}
function overview(phase){
 const cfg=ALGOS[algo],nodes=cfg.overview,done=phase===cfg.last;
 const active=algo==='dpo'?[[0],[0,1],[2],[3],[4]][phase]:phase===0?[0]:phase===1?[1,2]:phase===2?[3]:phase===3?[4]:phase<=6&&!(algo==='grpo'&&phase===7)?[5]:[],reached=Math.max(...active,done?nodes.length-1:0);
 return `<div class="ov-row">${nodes.map(([kicker,name],i)=>`<span class="ov-node ${active.includes(i)?'current':phase>0&&i<=reached?'done':''}" style="--i:${i}"><small>${kicker}</small>${name}${algo==='grpo'&&i===5&&phase>=4&&phase<=6?` <b>${phase-3}/3</b>`:''}</span>${i<nodes.length-1?`<span class="ov-wire ${active.includes(i+1)&&phase>0?'current':''}" style="--i:${i}" aria-hidden="true"><i></i></span>`:''}`).join('')}</div><div class="ov-back ${done?'current':''}" style="--n:${nodes.length}"><span>${cfg.back[done?1:0]}</span></div>`;}
// GRPO only: how far the learner's probability for this answer has drifted from the frozen copy's (ratio 1 = no drift).
// The shaded band is the allowed ±20%; a push pauses once the marker leaves the band in its favoured direction.
function driftMeter(term,advantage){
 const lo=.6,hi=1.4,pos=r=>100*(Math.min(hi,Math.max(lo,r))-lo)/(hi-lo),state=advantage===0?'idle':term.clipped?'paused':'moving';
 return `<span class="drift ${state}" role="img" aria-label="Now ${fmt(term.ratio,2)} times as likely as when sampled; the limit is ${advantage<0?'0.8':'1.2'} times${term.clipped?', reached: push paused':''}"><span class="drift-track"><i class="band"></i><i class="start"></i><i class="limit ${advantage<0?'low':'high'}"></i><i class="mark" style="left:${pos(term.ratio)}%"></i></span><span class="drift-scale"><em>0.8×</em><em>same</em><em>1.2×</em></span></span><small>${term.clipped?`${fmt(term.ratio,2)}× · past the 20% limit, push paused`:`now ${fmt(term.ratio,2)}× as likely as when sampled`}</small>`;}
function reuseBanner(phase){
 const pips=(done,total)=>`<span class="pips" aria-hidden="true">${Array.from({length:total},(_,i)=>`<i class="${i<done?'on':''}"></i>`).join('')}</span>`;
 if(algo==='grpo'&&phase>=3)return phase===3?`${pips(0,3)}<span><b>Next: these same four answers will be used for 3 updates.</b> No new answers are generated in between.</span>`:phase<=6?`${pips(phase-3,3)}<span><b>Same four answers, reused: update ${phase-3} of 3.</b> The drift meters show how far each answer has moved from the frozen copy.</span>`:`${pips(3,3)}<span><b>Used 3 times.</b> The frozen copy is now refreshed, and the next round writes four new answers.</span>`;
 if(algo==='reinforce'&&phase>=3)return phase===3?`${pips(0,1)}<span><b>Next: one update from these four answers.</b> Then they are thrown away.</span>`:`${pips(1,1)}<span><b>Used once, then discarded.</b> The updated model must write four new answers before it can learn again.</span>`;
 return '';}
function sampledLanes(phase,group){
 const scored=phase>=2,compared=phase>=3,groupProbs=group?probabilities(state.snapshots[group.version].logits):probabilities(state.sampler);
 const objective=compared&&algo==='grpo'?clippedObjective(state.learner,group.samples,group.stats.advantages):null,maxA=compared?Math.max(...group.stats.advantages.map(Math.abs),1e-9):1;
 $('#lane-head').className='rl-lane rl-lane-head';
 $('#lane-head').innerHTML=`<span>Run</span><span>Prompt in</span><span>Model samples</span><span>Response</span><span>Reward</span><span>Versus group average</span><span>${algo==='grpo'?'Learning signal · drift from frozen copy':'Learning signal'}</span>`;
 return Array.from({length:4},(_,i)=>{
  const sample=group?.samples[i],reward=scored?group.rewards[i]:null,a=compared?group.stats.advantages[i]:null,sign=a>0?'positive':a<0?'negative':'neutral',term=objective?.terms[i];
  const signal=!compared?'':a===0?'no signal':term?.clipped?'moved enough · paused':a>0?'▲ make more likely':'▼ make less likely';
  const ratio=term?driftMeter(term,a):'';
  return `<div class="rl-lane ${compared?sign:''} ${sample?'ran':''}" role="listitem" style="--d:${i*420}ms" aria-label="Run ${i+1}: ${sample?`answered ${sample.answer}`:'not run yet'}${scored?`, reward ${reward}`:''}${compared?`, advantage ${fmt(a,2)}`:''}">
   <span class="lane-id"><b>#${i+1}</b><small>${sample?`C${group.version}`:'idle'}</small></span>
   <span class="lane-prompt">${sample?'What is 2 + 2?':'—'}</span>
   <span class="lane-model">${sample?strip(groupProbs,[{u:pins[i],answer:sample.answer,label:'↓'}]):'<span class="strip-track idle"></span>'}</span>
   <span class="lane-answer">${sample?`<b>${sample.answer}</b>`:'<b class="empty">?</b>'}</span>
   <span class="lane-reward"><span class="reward ${scored?(state.rewardMode==='constant'?'flat':reward?'good':'bad'):'wait'}">${scored?`${state.rewardMode==='constant'?'':reward?'✓ ':'✗ '}reward ${reward}`:'not scored'}</span></span>
   <span class="lane-adv"><span class="adv-track" aria-hidden="true"><i class="${sign}" style="width:${compared?Math.abs(a)/maxA*46:0}%"></i></span><small>${compared?`${a>0?'+':''}${fmt(a,2)}`:'&nbsp;'}</small></span>
   <span class="lane-signal">${signal||'&nbsp;'}${ratio}</span></div>`;}).join('');
}
function pairLanes(phase,group){
 $('#lane-head').className='rl-lane rl-lane-head pair';
 $('#lane-head').innerHTML='<span>Pair</span><span>Prompt</span><span>Preferred</span><span>Rejected</span><span>Learner vs reference</span><span>Loss</span><span>Learning signal</span>';
 const pairs=group?.pairs||[{},{}],terms=phase>=2?(phase===4?dpoObjective(state.learner,state.sampler,group.pairs).terms:group.terms):null,span=terms?Math.max(...terms.map(t=>Math.abs(t.margin)),1.5):1.5;
 return pairs.map((pair,i)=>{const t=terms?.[i],loaded=!!group;
  return `<div class="rl-lane pair ${loaded?'ran':''} ${phase>=3?'positive':''}" role="listitem" style="--d:${i*420}ms" aria-label="Pair ${i+1}${loaded?`: prefer ${ANSWERS[pair.chosen]} over ${ANSWERS[pair.rejected]}`:''}">
   <span class="lane-id"><b>#${i+1}</b><small>${loaded?'labelled':'idle'}</small></span>
   <span class="lane-prompt">${loaded?'What is 2 + 2?':'—'}</span>
   <span class="lane-answer chosen">${loaded?`<b>${ANSWERS[pair.chosen]}</b><small>✓ preferred</small>`:'<b class="empty">?</b>'}</span>
   <span class="lane-answer rejected">${loaded?`<b>${ANSWERS[pair.rejected]}</b><small>✗ rejected</small>`:'<b class="empty">?</b>'}</span>
   <span class="lane-adv"><span class="adv-track" aria-hidden="true"><i class="${t&&t.margin<0?'negative':'positive'}" style="width:${t?Math.abs(t.margin)/span*46:0}%"></i></span><small>${t?`margin ${t.margin>=0?'+':''}${fmt(t.margin,2)} vs reference`:'&nbsp;'}</small></span>
   <span class="lane-loss">${t&&phase>=3?`<b>${fmt(t.loss,3)}</b><small>−ln σ(margin)</small>`:'<b class="wait">—</b>'}</span>
   <span class="lane-signal">${t&&phase>=3?`▲ ${ANSWERS[pair.chosen]} &nbsp; ▼ ${ANSWERS[pair.rejected]}<small>push strength ${fmt(t.push*100,0)}%</small>`:'&nbsp;'}</span></div>`;}).join('');
}
function render(fresh=false){
 const phase=state.phase,group=state.group,cfg=ALGOS[algo],scored=phase>=2,compared=phase>=3,lab=$('.rl-lab');
 lab.dataset.phase=phase;lab.dataset.algo=algo;lab.classList.toggle('fresh',fresh&&!reduced.matches);
 $$('[data-rl-algo]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.rlAlgo===algo)));
 $('#rl-algo-note').textContent=cfg.note;$('#rl-phases').innerHTML=overview(phase);
 $('#sampler-title').textContent=algo==='dpo'?'REFERENCE COPY · NEVER CHANGES · πref':'FROZEN COPY · WRITES THE ANSWERS · πold';
 $('#learner-title').textContent=algo==='reinforce'?'THE MODEL · WRITES AND LEARNS · πθ':'LEARNER COPY · GETS UPDATED · πθ';
 $('#sampler-version').textContent=`Checkpoint C${algo==='dpo'?0:state.samplerVersion}`;
 $('#learner-version').textContent=`${state.updates} update${state.updates===1?'':'s'} so far`;
 $('#sampler-bars').innerHTML=bars(state.sampler);
 $('#learner-bars').innerHTML=bars(state.learner,algo==='reinforce'?(state.previous?probabilities(state.previous):null):probabilities(state.sampler));
 $('#learner-curve').innerHTML=curve();
 $('#lanes-kicker').textContent=algo==='dpo'?'TWO PREFERENCE PAIRS':'FOUR SEPARATE GENERATIONS';
 $('#lanes-title').textContent=algo==='dpo'?'Same prompt, two answers, one marked as better':algo==='grpo'?'Same prompt, same frozen model, four independent runs':'Same prompt, same model, four independent runs';
 $('#group-label').textContent=algo==='dpo'?(group?`scored by the learner after ${group.version} update${group.version===1?'':'s'}`:'a fixed, hand-labelled dataset'):group?`written by checkpoint C${group.version}`:'ready to run checkpoint C0';
 $('#rollout-rows').innerHTML=algo==='dpo'?pairLanes(phase,group):sampledLanes(phase,group);
 const banner=reuseBanner(phase);$('#reuse-banner').hidden=!banner;$('#reuse-banner').innerHTML=banner;$('#reuse-banner').className=`reuse-banner ${algo}`;
 const average=compared&&algo!=='dpo'?`Group average reward = ${fmt(group.stats.mean,2)}. Bars to the right beat the average, bars to the left fell short. `:'';
 $('#group-summary').textContent=algo==='dpo'?[
  'Nothing loaded yet. DPO trains on stored preference pairs: nothing is sampled and nothing is scored by a checker.',
  'Two records, each saying “for this prompt, this answer is better than that one”. A person, a rule or a stronger model could have labelled them.',
  'Margin: how much more the learner favours the preferred answer over the rejected one than the reference copy does. Before any training it is exactly 0.',
  `Loss = −ln σ(margin), averaged over the pairs = ${group?.loss!=null?fmt(group.loss,3):'—'}. It falls toward 0 as the margin grows.`,
  `One update done. The margins grew, so the loss on these pairs fell from ${state.lastUpdate?fmt(state.lastUpdate.before.loss,3):'—'} to ${state.lastUpdate?fmt(state.lastUpdate.after.loss,3):'—'}.`][phase]
  :!group?'Nothing has run yet. A rollout is plain inference: the model answers, and no weights change.':!scored?'Each run drew its own random number, so the same model gave different answers. Repeats are normal. Nothing has been judged yet.':!compared?`Rewards: [${group.rewards.join(', ')}]. The checker returns a score, never the right answer.`:group.stats.std<1e-12?`Every answer got reward ${group.rewards[0]}, so none is better than the group average. Advantage is 0 everywhere and nothing is learned from this group.`:algo==='reinforce'?`${average}Advantage = reward − average.`:`${average}Advantage = (reward − average) / spread, spread = ${fmt(group.stats.std,2)}.`;
 $('#rl-next').textContent=cfg.labels[phase];$('#rl-next').disabled=playing;$('#rl-fast').disabled=playing;
 $('#rl-play').textContent=playing?'Pause':'▶ Play one round';
 $('#rl-count').textContent=`${state.updates} optimizer step${state.updates===1?'':'s'} · ${state.snapshots.length} snapshot${state.snapshots.length===1?'':'s'}`;
 $('#rl-status').textContent={
 reinforce:[
  'Checkpoint C0: equally likely to say 3, 4 or 5.',
  `Four separate runs of the model (C${group?.version}). Each run reads the prompt, draws one random number on its own probability strip, and writes one response. No training yet.`,
  'The checker scored each response on its own. It gives a number, not a correct response to imitate.',
  'Each response is compared with the group average. That difference decides which way, and how hard, to push it.',
  `One update: above-average answers became more likely, below-average ones less. Tick marks show the bars before the update. Saved as C${state.round}; the next four answers come from it.`],
 grpo:[
  'Both copies start with identical weights: checkpoint C0, equally likely to say 3, 4 or 5.',
  `Four separate runs of the frozen copy C${group?.version}. Each run reads the prompt, draws one random number on its own probability strip, and writes one response. No training yet.`,
  'The checker scored each response on its own. It gives a number, not a correct response to imitate.',
  'Each response is compared with its own group. These comparisons and the frozen probabilities now stay fixed for three updates.',
  'Update 1 / 3: the learner’s bars move toward above-average answers. The tick marks show where the frozen copy still is.',
  'Update 2 / 3: same four responses and scores, reused. Only the learner’s probabilities are recomputed.',
  'Update 3 / 3 done. If an answer has already moved more than 20% from the frozen copy, clipping pauses its push.',
  `Saved the learner as C${state.round} and copied it into the frozen copy. The next group will be written by C${state.round}. The lanes still show the group from C${group?.version}.`],
 dpo:[
  'The learner and the reference copy start identical (C0). The reference will stay at C0 for the whole run.',
  'The dataset is loaded. No model has generated anything.',
  'Both copies are asked how likely each answer is. Only the difference between them matters.',
  'Each pair gets a loss: small when the learner already leans toward the preferred answer, larger when it does not.',
  `One update: “4” went up, “3” and “5” went down. Tick marks show the reference copy. Saved as C${state.round}.`]}[algo][phase];
 const exact=algo==='grpo'?$('#rl-exact'):$(`#rl-exact-${algo}`);
 if(algo==='grpo'){const objective=compared?clippedObjective(state.learner,group.samples,group.stats.advantages):null;
  exact.innerHTML=!objective?'<p>Compare a group to reveal the current ratios and exact gradient.</p>':`<div class="rl-numeric"><strong>Current learner evaluated on the recorded group from C${group.version}</strong><p>${objective.terms.map((t,i)=>`#${i+1}: ρ = ${fmt(t.ratio)}${t.clipped?' (clipped plateau)':''}`).join(' · ')}</p><p>L = ${fmt(objective.loss,5)} · ∇L = [${objective.gradient.map(g=>fmt(g,5)).join(', ')}]</p>${state.lastUpdate?`<p>Most recent step used ∇L = [${state.lastUpdate.before.gradient.map(g=>fmt(g,5)).join(', ')}]. The gradient above is recomputed <em>after</em> that step.</p>`:''}<p>Current logits θ = [${state.learner.map(x=>fmt(x,5)).join(', ')}]</p></div>`;}
 else{const used=state.lastUpdate?.before||(algo==='reinforce'?(compared?reinforceObjective(state.learner,group.samples,group.stats.advantages):null):(phase>=2?dpoObjective(state.learner,state.sampler,group.pairs):null));
  exact.innerHTML=!used?'<p>Step through a round to reveal the exact loss and gradient.</p>':`<div class="rl-numeric"><strong>${state.lastUpdate?'The step just taken':'The step about to be taken'}</strong><p>L = ${fmt(used.loss,5)} · ∇L = [${used.gradient.map(g=>fmt(g,5)).join(', ')}]</p><p>Current logits θ = [${state.learner.map(x=>fmt(x,5)).join(', ')}]</p></div>`;}
 renderInference();
}
function renderInference(fresh=false){
 $('#inference-checkpoint').innerHTML=state.snapshots.map(s=>`<option value="${s.version}" ${s.version===servingVersion?'selected':''}>C${s.version} · ${s.updates===0?'initial':s.updates+' optimizer steps'}</option>`).join('');
 $('#checkpoint-timeline').innerHTML=state.snapshots.map(s=>{const p=probabilities(s.logits)[1];return `<button type="button" data-snapshot="${s.version}" aria-pressed="${s.version===servingVersion}" class="${s.version===servingVersion?'selected':''}"><span class="snap-bar"><i style="height:${Math.max(3,p*46)}px"></i></span><strong>C${s.version}</strong><small>says “4” ${fmt(p*100,0)}%</small><small>after ${s.updates} updates</small></button>`;}).join('');
 const snapshot=state.snapshots.find(s=>s.version===servingVersion),probs=probabilities(snapshot.logits),total=tally.reduce((a,b)=>a+b,0);
 $('#inference-bars').innerHTML=bars(snapshot.logits);
 $('#inference-strip').innerHTML=strip(probs,inferencePin===null?[]:[{u:inferencePin,answer:inferenceResult.answer,label:'↓'}]);$('#inference-strip').classList.toggle('fresh',fresh&&!reduced.matches);
 $('#inference-answer').textContent=inferenceResult?`C${servingVersion} answered: ${inferenceResult.answer}`:`C${servingVersion} loaded. No answer sampled yet.`;
 $('#inference-tally').innerHTML=total?`<p class="string-label">${total} ANSWER${total===1?'':'S'} FROM C${servingVersion} SO FAR</p>${ANSWERS.map((a,i)=>`<div class="rl-probability ${i===1?'correct-answer':''}"><code>${a}</code><span><i style="width:${100*tally[i]/total}%"></i><b class="tick" style="left:${100*probs[i]}%" title="expected ${fmt(100*probs[i],1)}%"></b></span><b>${tally[i]}×</b></div>`).join('')}`:'';
}
function inferOnce(){const snapshot=state.snapshots.find(s=>s.version===servingVersion);inferencePin=random(inferenceSeed).value;inferenceResult=inferSnapshot(snapshot,inferenceSeed);inferenceSeed=inferenceResult.seed;tally[ANSWERS.indexOf(inferenceResult.answer)]++;}
function selectSnapshot(version){servingVersion=version;inferenceResult=null;inferencePin=null;tally=[0,0,0];renderInference();}
function step(fresh=true){
 if(algo!=='dpo'&&(state.phase===0||state.phase===lastPhase())){let seed=state.seed;pins=[];for(let i=0;i<4;i++){const r=random(seed);seed=r.seed;pins.push(r.value);}}
 const updatesBefore=state.updates;state=ALGOS[algo].advance(state);
 if(state.updates>updatesBefore)history.push(probabilities(state.learner)[1]);
 if(state.phase===lastPhase())roundMarks.push(history.length-1);
 render(fresh);}
function queue(){const ticket=epoch;timer=setTimeout(()=>{timer=null;if(ticket!==epoch||!playing)return;if(document.hidden){cancel();render();return;}step();if(state.phase===lastPhase()){cancel();render();return;}queue();},reduced.matches?0:ALGOS[algo].delays[state.phase]);}
function runRounds(count=5){cancel();playing=true;const ticket=epoch;let left=count;
 const tick=()=>{if(ticket!==epoch||!playing)return;if(document.hidden||left===0){cancel();render();return;}step(false);if(state.phase===lastPhase())left--;timer=setTimeout(tick,left===0?0:150);};
 if(reduced.matches){while(left>0){step(false);if(state.phase===lastPhase())left--;}cancel();render();}else tick();}
$('#rl-next').addEventListener('click',()=>{cancel();step();});
$('#rl-play').addEventListener('click',()=>{if(playing){cancel();render();return;}playing=true;render();queue();});
$('#rl-fast').addEventListener('click',()=>runRounds());
function reset(){cancel();state=ALGOS[algo].init($('#rl-reward').value);servingVersion=0;inferenceSeed=981;inferenceResult=null;inferencePin=null;tally=[0,0,0];pins=[];history=[probabilities(state.learner)[1]];roundMarks=[];render();}
$('#rl-reset').addEventListener('click',reset);$('#rl-reward').addEventListener('change',reset);
$$('[data-rl-algo]').forEach(b=>b.addEventListener('click',()=>{if(!ALGOS[b.dataset.rlAlgo])return;algo=b.dataset.rlAlgo;reset();}));
$('#inference-checkpoint').addEventListener('change',e=>{servingVersion=+e.target.value;selectSnapshot(servingVersion);});
$('#checkpoint-timeline').addEventListener('click',e=>{const b=e.target.closest('[data-snapshot]');if(!b)return;selectSnapshot(+b.dataset.snapshot);$(`[data-snapshot="${servingVersion}"]`).focus({preventScroll:true});});
$('#inference-sample').addEventListener('click',()=>{inferOnce();renderInference(true);});
$('#inference-many').addEventListener('click',()=>{for(let i=0;i<20;i++)inferOnce();renderInference(true);});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing){cancel();render();}});reduced.addEventListener('change',()=>{if(playing){cancel();render();}});
document.addEventListener('click',e=>{if(e.target.closest('a[href^="#ref-"]'))$('#sources').open=true;});
render();
