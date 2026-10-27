/** Exact intermediate attention values. The optional causal path never mutates encoder tensors. */
import {matrix,matrixValue} from './full-config.mjs';
const attentionStageCache=new WeakMap();
export const ATTENTION_STAGE_NAMES=Object.freeze({dot:'Dot product',scaled:'Scaling · Mask',softmax:'Softmax'});
function extrema(a){let min=Infinity,max=-Infinity;for(const v of a.data)if(Number.isFinite(v)){min=Math.min(min,v);max=Math.max(max,v);}return {min,max};}
export function attentionStages(data){
 const original=data.encoderHead||data.head;
 let cached=attentionStageCache.get(original);
 if(!cached){
  const n=original.scores.rows,d=data.c.headWidth,dot=matrix(n,n);
  // Float64 dot products are retained for inspection; scaling is the engine's stored FP32 value.
  dot.data=new Float64Array(n*n);
  for(let q=0;q<n;q++)for(let k=0;k<n;k++){let sum=0;for(let i=0;i<d;i++)sum+=matrixValue(data.q,q,i)*matrixValue(data.k,k,i);dot.data[q*n+k]=sum;}
  cached={dot,encoder:{dot,scaled:original.scores,softmax:original.attention,out:original.out,causal:false,divisor:Math.sqrt(d),ranges:{dot:extrema(dot),scaled:extrema(original.scores),softmax:extrema(original.attention)}}};
  attentionStageCache.set(original,cached);
 }
 if(!data.causalComparison)return cached.encoder;
 if(!cached.causal){
  const n=original.scores.rows,d=data.c.headWidth,scaled=matrix(n,n),weights=matrix(n,n),out=matrix(n,d);
  for(let q=0;q<n;q++){
   let max=-Infinity;for(let k=0;k<n;k++){const s=k>q?-Infinity:original.scores.data[q*n+k];scaled.data[q*n+k]=s;max=Math.max(max,s);}
   let den=0;for(let k=0;k<n;k++){weights.data[q*n+k]=Math.exp(scaled.data[q*n+k]-max);den+=weights.data[q*n+k];}
   for(let k=0;k<n;k++)weights.data[q*n+k]/=den;
   for(let i=0;i<d;i++){let sum=0;for(let k=0;k<n;k++)sum+=weights.data[q*n+k]*matrixValue(data.v,k,i);out.data[q*d+i]=sum;}
  }
  cached.causal={dot:cached.dot,scaled,softmax:weights,out,causal:true,divisor:Math.sqrt(d),ranges:{dot:extrema(cached.dot),scaled:extrema(scaled),softmax:extrema(weights)}};
 }
 return cached.causal;
}
export function withCausalAttention(data){
 const original=data.encoderHead||data.head,stages=attentionStages({...data,encoderHead:original,causalComparison:true});
 const head={...original,scores:stages.scaled,attention:stages.softmax,out:stages.out};
 return {...data,encoderHead:original,causalComparison:true,head,score:head.scores.data[data.query*data.n+data.key],weight:head.attention.data[data.query*data.n+data.key]};
}
export function attentionStageValue(data,query,key,stage='softmax'){
 const stages=attentionStages(data),m=stages[stage];
 if(!m||!Number.isInteger(query)||!Number.isInteger(key)||query<0||key<0||query>=m.rows||key>=m.cols)return null;
 return {stage,label:ATTENTION_STAGE_NAMES[stage],value:m.data[query*m.cols+key],causal:stages.causal,blocked:stages.causal&&key>query&&stage!=='dot',total:m.cols,divisor:stages.divisor};
}
export function attentionStageTrace(data,query=data.query,key=data.key){
 const stages=attentionStages(data),n=data.n,row=Array.from({length:n},(_,k)=>stages.scaled.data[query*n+k]);
 const max=Math.max(...row),terms=row.map(v=>Math.exp(v-max)),denominator=terms.reduce((sum,v)=>sum+v,0);
 return {query,key,raw:stages.dot.data[query*n+key],scaledBeforeMask:(data.encoderHead||data.head).scores.data[query*n+key],scaled:row[key],divisor:stages.divisor,blocked:stages.causal&&key>query,causal:stages.causal,max,numerator:terms[key],denominator,total:n,probability:stages.softmax.data[query*n+key]};
}
export function stageNumber(value,digits=4){return value===-Infinity?'−∞':Number(value).toFixed(digits);}
/** Expand a single attention panel into three panels, leaving the outer architecture intact. */
export function animateAttentionStages(root,{expanded=true,reduced=false}={}){
 const animations=[];if(reduced)return {dispose(){}};
 const panels=[...root.querySelectorAll('[data-attention-panel]')];
 panels.forEach((panel,i)=>{if(typeof panel.animate!=='function')return;const dx=expanded?(1-i)*140:0;
  animations.push(panel.animate([{opacity:.2,transform:`translateX(${dx}px)`},{opacity:1,transform:'translateX(0)'}],{duration:650,delay:expanded?i*70:0,easing:'cubic-bezier(.22,.75,.18,1)'}));
 });
 return {dispose(){animations.forEach(a=>a.cancel());}};
}
