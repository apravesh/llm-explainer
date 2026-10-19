/** Deterministic, untrained sparse FFN. Never produces text or semantic predictions. */
export const MOE_DIM=8, MOE_HIDDEN=16, MOE_EXPERTS=4;
export function moeRng(seed){return ()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
export function moeSoftmax(logits){const max=Math.max(...logits),e=logits.map(x=>Math.exp(x-max)),sum=e.reduce((a,b)=>a+b,0);return e.map(v=>v/sum);}
export function moeLinear(x,w){if(x.length!==w.length)throw Error('Input/weight dimensions do not match');return w[0].map((_,j)=>x.reduce((sum,v,i)=>sum+v*w[i][j],0));}
export function moeNorm(x){const mean=x.reduce((a,b)=>a+b,0)/x.length,variance=x.reduce((a,v)=>a+(v-mean)**2,0)/x.length;return x.map(v=>(v-mean)/Math.sqrt(variance+1e-5));}
export const moeGelu=x=>.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x**3)));
function moeWeights(n,m,seed,scale=1){const r=moeRng(seed),s=Math.sqrt(3/n)*scale;return Array.from({length:n},()=>Array.from({length:m},()=> (2*r()-1)*s));}
export function createMoeModel(seed=108){return {router:moeWeights(MOE_DIM,MOE_EXPERTS,seed),experts:Array.from({length:MOE_EXPERTS},(_,i)=>({up:moeWeights(MOE_DIM,MOE_HIDDEN,seed+10+i*2),down:moeWeights(MOE_HIDDEN,MOE_DIM,seed+11+i*2,.65)}))};}
export function teachingTokens(text){const labels=String(text).trim().split(/\s+/u).filter(Boolean).slice(0,12);return labels.map((label,index)=>{let hash=2166136261;for(const char of `${label}:${index}`)hash=Math.imul(hash^char.codePointAt(0),16777619);const random=moeRng(hash);return {label,index,hidden:Array.from({length:MOE_DIM},()=>Math.round((random()*4-2)*100)/100)};});}
export function routeToken(normalized,router,{k=1,weighting='selected'}={}){
 if(![1,2].includes(k))throw Error('Teaching router supports top-1 or top-2');
 if(!['selected','full'].includes(weighting))throw Error('Unknown gate weighting');
 const logits=moeLinear(normalized,router),probabilities=moeSoftmax(logits);
 const selected=logits.map((_,i)=>i).sort((a,b)=>logits[b]-logits[a]||a-b).slice(0,k);
 const selectedMass=selected.reduce((sum,i)=>sum+probabilities[i],0);
 const gates=probabilities.map((p,i)=>selected.includes(i)?(weighting==='selected'?p/selectedMass:p):0);
 return {logits,probabilities,selected,gates,selectedMass,k,weighting};
}
export function expertForward(x,expert){const pre=moeLinear(x,expert.up),activated=pre.map(moeGelu),output=moeLinear(activated,expert.down);return {pre,activated,output};}
export function runMoeToken(hidden,model,{k=1,weighting='selected',mode='sparse'}={}){
 if(hidden.length!==MOE_DIM||hidden.some(x=>!Number.isFinite(x)))throw Error('Need eight finite hidden-state values');
 if(!['sparse','dense'].includes(mode))throw Error('Unknown FFN mode');
 const normalized=moeNorm(hidden);
 const router=mode==='sparse'?routeToken(normalized,model.router,{k,weighting}):null;
 const selected=router?router.selected:[0],gates=router?router.gates:[1,0,0,0];
 // Critically: no forward pass through unselected experts.
 const experts=model.experts.map((expert,i)=>selected.includes(i)?expertForward(normalized,expert):null);
 const contributions=experts.map((e,i)=>e?e.output.map(x=>x*gates[i]):null);
 const mixed=Array.from({length:MOE_DIM},(_,d)=>contributions.reduce((sum,x)=>sum+(x?x[d]:0),0));
 const output=hidden.map((x,d)=>x+mixed[d]);
 return {hidden:hidden.slice(),normalized,router,selected,gates,experts,contributions,mixed,output,mode,k,weighting,expertEvaluations:selected.length};
}
export function runMoeBatch(tokens,model,options){const traces=tokens.map(t=>runMoeToken(t.hidden,model,options)),loads=Array(MOE_EXPERTS).fill(0);traces.forEach(t=>t.selected.forEach(i=>loads[i]++));return {traces,loads,assignments:loads.reduce((a,b)=>a+b,0)};}
export function moeCost({mode='sparse',k=1}={}){const expert=2*MOE_DIM*MOE_HIDDEN,router=MOE_DIM*MOE_EXPERTS;return {expert,router:mode==='dense'?0:router,stored:mode==='dense'?expert:router+MOE_EXPERTS*expert,active:mode==='dense'?expert:router+k*expert,expertEvaluations:mode==='dense'?1:k};}
export function moeMixPrefix(trace,count){return Array.from({length:MOE_DIM},(_,d)=>trace.selected.slice(0,count).reduce((sum,e)=>sum+trace.contributions[e][d],0));}
