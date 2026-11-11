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
