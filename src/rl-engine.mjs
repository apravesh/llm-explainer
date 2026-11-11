// One prompt, three possible ONE-TOKEN completions. This is a categorical policy,
// not an LLM. It implements a clipped GRPO-style objective with beta (KL) = 0.
export const ANSWERS=['3','4','5'];
export const PROMPT='What is 2 + 2?';
export const GROUP_SIZE=4,UPDATES_PER_GROUP=3,LEARNING_RATE=.15,CLIP=.2;
export function probabilities(logits){
 if(logits.length!==3||!logits.every(Number.isFinite))throw new RangeError('Need three finite logits');
 const m=Math.max(...logits),e=logits.map(x=>Math.exp(x-m)),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);
}
export function draw(probs,u){
 if(!(u>=0&&u<1)||probs.length!==3||probs.some(p=>!Number.isFinite(p)||p<0)||Math.abs(probs.reduce((a,b)=>a+b,0)-1)>1e-9)throw new RangeError('Invalid sampling distribution');
 let sum=0;for(let i=0;i<probs.length;i++){sum+=probs[i];if(u<sum)return i;}return probs.length-1;
}
export function random(seed){const next=(seed+0x6D2B79F5)>>>0;let t=next;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return {seed:next,value:((t^(t>>>14))>>>0)/4294967296};}
export function sampleGroup(logits,seed){const probs=probabilities(logits),samples=[];for(let i=0;i<GROUP_SIZE;i++){const r=random(seed);seed=r.seed;const action=draw(probs,r.value);samples.push({action,answer:ANSWERS[action],oldProbability:probs[action]});}return {samples,seed};}
export function groupAdvantages(rewards){
 if(rewards.length!==GROUP_SIZE||!rewards.every(Number.isFinite))throw new RangeError('Need four finite rewards');
 const mean=rewards.reduce((a,b)=>a+b,0)/rewards.length,std=Math.sqrt(rewards.reduce((s,r)=>s+(r-mean)**2,0)/rewards.length);
 return {mean,std,advantages:rewards.map(r=>std<1e-12?0:(r-mean)/std)};
}
export function clippedObjective(logits,samples,advantages){
 if(samples.length!==GROUP_SIZE||advantages.length!==GROUP_SIZE||!advantages.every(Number.isFinite))throw new RangeError('Invalid group');
 const probs=probabilities(logits),gradient=[0,0,0];let loss=0;
 const terms=samples.map((s,i)=>{if(!Number.isInteger(s.action)||s.action<0||s.action>2||!(s.oldProbability>0&&s.oldProbability<=1))throw new RangeError('Invalid rollout');
 const a=advantages[i],ratio=probs[s.action]/s.oldProbability,clippedRatio=Math.max(1-CLIP,Math.min(1+CLIP,ratio));
 const clipped=a>0&&ratio>1+CLIP||a<0&&ratio<1-CLIP;
 const objective=Math.min(ratio*a,clippedRatio*a);loss-=objective/GROUP_SIZE;
 if(!clipped)for(let j=0;j<3;j++)gradient[j]-=a*ratio*(Number(j===s.action)-probs[j])/GROUP_SIZE;
 return {ratio,clippedRatio,clipped,advantage:a,objective};});
 return {loss,gradient,terms,probabilities:probs};
}
export function policyUpdate(logits,samples,advantages){const before=clippedObjective(logits,samples,advantages),next=logits.map((x,i)=>x-LEARNING_RATE*before.gradient[i]);return {logits:next,before,after:clippedObjective(next,samples,advantages)};}
export function initialRLState(rewardMode='correct'){if(!['correct','constant'].includes(rewardMode))throw new RangeError('Invalid reward mode');return {phase:0,round:0,updates:0,seed:1,rewardMode,learner:[0,0,0],sampler:[0,0,0],samplerVersion:0,snapshots:[{version:0,updates:0,logits:[0,0,0]}],group:null,lastUpdate:null};}
export function advanceRL(current){
 const s=structuredClone(current);
 if(s.phase===0||s.phase===7){const sampled=sampleGroup(s.sampler,s.seed);s.seed=sampled.seed;s.group={samples:sampled.samples,version:s.samplerVersion,rewards:null,stats:null};s.lastUpdate=null;s.phase=1;return s;}
 s.phase++;
 if(s.phase===2)s.group.rewards=s.group.samples.map(x=>s.rewardMode==='constant'?1:Number(x.answer==='4'));
 if(s.phase===3)s.group.stats=groupAdvantages(s.group.rewards);
 if(s.phase>=4&&s.phase<=6){s.lastUpdate=policyUpdate(s.learner,s.group.samples,s.group.stats.advantages);s.learner=s.lastUpdate.logits;s.updates++;}
 if(s.phase===7){s.round++;s.snapshots.push({version:s.round,updates:s.updates,logits:[...s.learner]});s.sampler=[...s.learner];s.samplerVersion=s.round;}
 return s;
}
export function inferSnapshot(snapshot,seed){const probs=probabilities(snapshot.logits),r=random(seed),action=draw(probs,r.value);return {answer:ANSWERS[action],probabilities:probs,seed:r.seed};}

// ---------- REINFORCE with a mean baseline: one model, one update per group ----------
// Loss L = −mean_i A_i · log π(a_i), with A_i = r_i − mean(r). No frozen copy, ratio or clipping.
export const REINFORCE_RATE=1;
export function meanBaseline(rewards){
 if(rewards.length!==GROUP_SIZE||!rewards.every(Number.isFinite))throw new RangeError('Need four finite rewards');
 const mean=rewards.reduce((a,b)=>a+b,0)/rewards.length,std=Math.sqrt(rewards.reduce((s,r)=>s+(r-mean)**2,0)/rewards.length);
 return {mean,std,advantages:rewards.map(r=>std<1e-12?0:r-mean)};
}
export function reinforceObjective(logits,samples,advantages){
 if(samples.length!==GROUP_SIZE||advantages.length!==GROUP_SIZE||!advantages.every(Number.isFinite))throw new RangeError('Invalid group');
 const probs=probabilities(logits),gradient=[0,0,0];let loss=0;
 samples.forEach((s,i)=>{if(!Number.isInteger(s.action)||s.action<0||s.action>2)throw new RangeError('Invalid rollout');
  loss-=advantages[i]*Math.log(probs[s.action])/GROUP_SIZE;
  for(let j=0;j<3;j++)gradient[j]-=advantages[i]*(Number(j===s.action)-probs[j])/GROUP_SIZE;});
 return {loss,gradient,probabilities:probs};
}
export function reinforceUpdate(logits,samples,advantages){const before=reinforceObjective(logits,samples,advantages),next=logits.map((x,i)=>x-REINFORCE_RATE*before.gradient[i]);return {logits:next,before,after:reinforceObjective(next,samples,advantages)};}
export function initialReinforceState(rewardMode='correct'){return {...initialRLState(rewardMode),algo:'reinforce',previous:null};}
/** Phases: 0 ready · 1 sampled · 2 scored · 3 compared · 4 updated (then sample again). */
export function advanceReinforce(current){
 const s=structuredClone(current);
 if(s.phase===0||s.phase===4){const sampled=sampleGroup(s.learner,s.seed);s.seed=sampled.seed;s.group={samples:sampled.samples,version:s.round,rewards:null,stats:null};s.lastUpdate=null;s.phase=1;return s;}
 s.phase++;
 if(s.phase===2)s.group.rewards=s.group.samples.map(x=>s.rewardMode==='constant'?1:Number(x.answer==='4'));
 if(s.phase===3)s.group.stats=meanBaseline(s.group.rewards);
 if(s.phase===4){s.previous=[...s.learner];s.lastUpdate=reinforceUpdate(s.learner,s.group.samples,s.group.stats.advantages);s.learner=s.lastUpdate.logits;s.updates++;s.round++;s.snapshots.push({version:s.round,updates:s.updates,logits:[...s.learner]});s.sampler=[...s.learner];s.samplerVersion=s.round;}
 return s;
}

// ---------- DPO: learn from preference pairs, no sampling and no reward function ----------
// Loss L = mean_i −ln σ(β · m_i), with margin m_i = [log πθ(chosen) − log πref(chosen)] − [log πθ(rejected) − log πref(rejected)].
// The reference policy is frozen for the whole run. With one-token answers the log-partition terms cancel, so m is a difference of logits.
export const DPO_BETA=1,DPO_RATE=1;
export const DPO_PAIRS=Object.freeze([{chosen:1,rejected:0},{chosen:1,rejected:2}].map(Object.freeze));
const sigmoid=x=>1/(1+Math.exp(-x));
export function dpoObjective(logits,reference,pairs=DPO_PAIRS){
 const p=probabilities(logits),q=probabilities(reference),gradient=[0,0,0];let loss=0;
 if(!pairs.length||pairs.some(x=>![x.chosen,x.rejected].every(i=>Number.isInteger(i)&&i>=0&&i<3)||x.chosen===x.rejected))throw new RangeError('Invalid preference pairs');
 const terms=pairs.map(({chosen,rejected})=>{const margin=(Math.log(p[chosen])-Math.log(q[chosen]))-(Math.log(p[rejected])-Math.log(q[rejected])),agree=sigmoid(DPO_BETA*margin),pairLoss=-Math.log(agree),push=1-agree;
  loss+=pairLoss/pairs.length;gradient[chosen]-=DPO_BETA*push/pairs.length;gradient[rejected]+=DPO_BETA*push/pairs.length;
  return {chosen,rejected,margin,agree,loss:pairLoss,push};});
 return {loss,gradient,terms,probabilities:p};
}
export function dpoUpdate(logits,reference,pairs=DPO_PAIRS){const before=dpoObjective(logits,reference,pairs),next=logits.map((x,i)=>x-DPO_RATE*before.gradient[i]);return {logits:next,before,after:dpoObjective(next,reference,pairs)};}
export function initialDpoState(){return {...initialRLState('correct'),algo:'dpo',previous:null};}
/** Phases: 0 ready · 1 pairs loaded · 2 compared with the reference · 3 loss · 4 updated (then reuse the pairs). The sampler stays at C0 as the reference. */
export function advanceDpo(current){
 const s=structuredClone(current);
 if(s.phase===0||s.phase===4){s.group={pairs:DPO_PAIRS.map(x=>({...x})),version:s.round,terms:null,loss:null};s.lastUpdate=null;s.phase=1;return s;}
 s.phase++;
 if(s.phase===2)s.group.terms=dpoObjective(s.learner,s.sampler,s.group.pairs).terms;
 if(s.phase===3)s.group.loss=dpoObjective(s.learner,s.sampler,s.group.pairs).loss;
 if(s.phase===4){s.previous=[...s.learner];s.lastUpdate=dpoUpdate(s.learner,s.sampler,s.group.pairs);s.learner=s.lastUpdate.logits;s.updates++;s.round++;s.snapshots.push({version:s.round,updates:s.updates,logits:[...s.learner]});}
 return s;
}
