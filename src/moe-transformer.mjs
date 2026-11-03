import {MOE_DIM,moeRng,moeLinear,moeNorm,moeSoftmax,createMoeModel,runMoeBatch} from './moe-engine.mjs';
export const MOE_BLOCKS=4,MOE_HEADS=2;
// Deliberately symbolic vocabulary: untrained scores must not masquerade as word predictions.
export const MOE_TOY_VOCAB=Object.freeze(Array.from({length:8},(_,i)=>`v${i}`));
function weights(rows,cols,seed,scale=1){const random=moeRng(seed),s=scale*Math.sqrt(3/rows);return Array.from({length:rows},()=>Array.from({length:cols},()=>(2*random()-1)*s));}
export function createMoeTransformer({activation='gelu',depth=MOE_BLOCKS,seed=108}={}){
 if(!Number.isInteger(depth)||depth<1||depth>24)throw Error('Use 1–24 teaching blocks');
 return {width:MOE_DIM,heads:MOE_HEADS,depth,activation,vocabulary:MOE_TOY_VOCAB,
  blocks:Array.from({length:depth},(_,i)=>{const base=seed+10000*(i+1);return {attention:{q:weights(8,8,base+1),k:weights(8,8,base+2),v:weights(8,8,base+3),o:weights(8,8,base+4,.45)},ffn:createMoeModel(base+100,{activation})};}),lmHead:weights(8,MOE_TOY_VOCAB.length,seed+900001,.6)};
}
export function moePosition(index,width=MOE_DIM){return Array.from({length:width},(_,j)=>{const angle=index/10000**(2*Math.floor(j/2)/width);return j%2?Math.cos(angle):Math.sin(angle);});}
export function moeCausalAttention(input,w,heads=MOE_HEADS){
 if(!input.length||input.some(row=>row.length!==MOE_DIM||row.some(x=>!Number.isFinite(x)))||!Number.isInteger(heads)||heads<1||MOE_DIM%heads)throw Error('Invalid attention shape');
 const normalized=input.map(moeNorm),q=normalized.map(x=>moeLinear(x,w.q)),k=normalized.map(x=>moeLinear(x,w.k)),v=normalized.map(x=>moeLinear(x,w.v)),headWidth=MOE_DIM/heads;
 const headTraces=Array.from({length:heads},(_,head)=>{
  const offset=head*headWidth,scores=input.map((_,i)=>input.map((_,j)=>j>i?-Infinity:q[i].slice(offset,offset+headWidth).reduce((s,x,d)=>s+x*k[j][offset+d],0)/Math.sqrt(headWidth)));
  const probabilities=scores.map(moeSoftmax),output=probabilities.map(p=>Array.from({length:headWidth},(_,d)=>p.reduce((sum,weight,j)=>sum+weight*v[j][offset+d],0)));
  return {scores,probabilities,output};
 });
 const combined=input.map((_,i)=>headTraces.flatMap(h=>h.output[i])),delta=combined.map(x=>moeLinear(x,w.o)),output=input.map((row,i)=>row.map((x,j)=>x+delta[i][j]));
 return {input,normalized,q,k,v,headWidth,heads:headTraces,combined,delta,output};
}
export function runMoeTransformer(tokens,model,options={}){
 if(!tokens.length||tokens.length>12||tokens.some(t=>t.hidden.length!==MOE_DIM||t.hidden.some(x=>!Number.isFinite(x))))throw Error('Need 1–12 finite eight-feature input vectors');
 const positions=tokens.map((_,i)=>moePosition(i)),input=tokens.map((t,i)=>t.hidden.map((x,j)=>x+positions[i][j]));
 let hidden=input;const blocks=model.blocks.map((block,index)=>{
  const attention=moeCausalAttention(hidden,block.attention,model.heads),ffn=runMoeBatch(tokens.map((t,i)=>({...t,hidden:attention.output[i]})),block.ffn,options),output=ffn.traces.map(t=>t.output);
  const result={index,input:hidden,attention,ffn,output};hidden=output;return result;
 });
 const final=hidden.map(moeNorm),logits=final.map(x=>moeLinear(x,model.lmHead)),probabilities=logits.map(moeSoftmax);
 return {positions,input,blocks,final,logits,probabilities};
}
export function adjacentMoeBlock(index,key,depth){
 if(key==='Home')return 0;if(key==='End')return depth-1;
 const delta={ArrowLeft:-1,ArrowUp:-1,ArrowRight:1,ArrowDown:1}[key];return delta===undefined?null:Math.max(0,Math.min(depth-1,index+delta));
}
