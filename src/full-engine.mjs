import {FULL_CONFIGS,matrix,rowOf,weightAt,weightDescriptor,blockWeights,projectionWeights,projectorWeights} from './full-config.mjs';
export function normalizeFull(a,rms=false,eps=1e-5){const out=matrix(a.rows,a.cols);for(let r=0;r<a.rows;r++){const row=rowOf(a,r);let mean=0;if(!rms){for(const x of row)mean+=x;mean/=a.cols;}let variance=0;for(const x of row)variance+=(x-mean)**2;const scale=1/Math.sqrt(variance/a.cols+eps);for(let d=0;d<a.cols;d++)out.data[r*a.cols+d]=(row[d]-mean)*scale;}return out;}
export function addFull(a,b){const out=matrix(a.rows,a.cols);for(let i=0;i<a.data.length;i++)out.data[i]=a.data[i]+b.data[i];return out;}
export function geluFull(x){return .5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x*x*x)));}
export function activateFull(a,gate=null){const out=matrix(a.rows,a.cols);for(let i=0;i<a.data.length;i++){const x=a.data[i];out.data[i]=gate?x/(1+Math.exp(-x))*gate.data[i]:geluFull(x);}return out;}
export function rotaryFull(a,c,positions){const out=matrix(a.rows,a.cols,a.data.slice());for(let r=0;r<a.rows;r++)for(let h=0;h<c.heads;h++)for(let pair=0;pair<c.headWidth/2;pair++){const i=r*a.cols+h*c.headWidth+pair*2,frequency=10000**(-2*pair/c.headWidth),angle=positions[r][pair%2]*frequency,cs=Math.cos(angle),sn=Math.sin(angle),x=a.data[i],y=a.data[i+1];out.data[i]=x*cs-y*sn;out.data[i+1]=x*sn+y*cs;}return out;}
export function attentionFull(q,k,v,c,causal=false){const n=q.rows,heads=[],combined=matrix(n,c.width);for(let h=0;h<c.heads;h++){
 const scores=matrix(n,n),weights=matrix(n,n),output=matrix(n,c.headWidth),offset=h*c.headWidth;
 for(let r=0;r<n;r++){
  let max=-Infinity;for(let j=0;j<n;j++){let score=0;for(let d=0;d<c.headWidth;d++)score+=q.data[r*c.width+offset+d]*k.data[j*c.width+offset+d];score=causal&&j>r?-Infinity:score/Math.sqrt(c.headWidth);scores.data[r*n+j]=score;max=Math.max(max,scores.data[r*n+j]);}
  let den=0;for(let j=0;j<n;j++){const x=Math.exp(scores.data[r*n+j]-max);weights.data[r*n+j]=x;den+=weights.data[r*n+j];}for(let j=0;j<n;j++)weights.data[r*n+j]/=den;
  for(let d=0;d<c.headWidth;d++){let sum=0;for(let j=0;j<n;j++)sum+=weights.data[r*n+j]*v.data[j*c.width+offset+d];output.data[r*c.headWidth+d]=sum;combined.data[r*c.width+offset+d]=sum;}
 }
 heads.push({scores,attention:weights,out:output});
}return {heads,combined};}
export async function runFullEncoder({pixels,rows,cols,config,backend,onProgress=()=>{},onEmbedding=()=>{},onLayer=()=>{},onOutput=()=>{},onProjector=()=>{},isCancelled=()=>false}){
 const c=config,n=pixels.rows;const check=()=>{if(isCancelled())throw Error('Cancelled');};let completed=0;
 const total=1+c.depth*(c.gated?7:6)+(c.decoderWidth?2:0);const multiply=async(a,w,label)=>{check();onProgress({completed,total,label});const out=await backend.multiply(a,w);check();for(const x of out.data)if(!Number.isFinite(x))throw Error(`Non-finite activation in ${label}`);completed++;return out;};
 const projection=projectionWeights(c,pixels.cols),embeddings=await multiply(pixels,projection,'Patch projection');
 let input,cls=null,positions=null;
 if(c.cls){cls=matrix(1,c.width);positions=matrix(n+1,c.width);const cw=weightDescriptor(1,c.width,1993,.03),pw=weightDescriptor(n+1,c.width,1997,.15);input=matrix(n+1,c.width);for(let d=0;d<c.width;d++)cls.data[d]=weightAt(cw,0,d);for(let r=0;r<n+1;r++)for(let d=0;d<c.width;d++){positions.data[r*c.width+d]=weightAt(pw,r,d);input.data[r*c.width+d]=(r===0?cls.data[d]:embeddings.data[(r-1)*c.width+d])+positions.data[r*c.width+d];}}
 else input=normalizeFull(embeddings,true,c.eps);
 onEmbedding({config:c,embeddings,input,cls,positions,projectionWeights:projection});let tokens=input;
 const coordinates=Array.from({length:n},(_,i)=>[Math.floor(i/cols),i%cols]);
 for(let l=0;l<c.depth;l++){
  const w=blockWeights(c,l),normalized=normalizeFull(tokens,c.rms,c.eps);
  const rawQ=await multiply(normalized,w.q,`Block ${l+1}/${c.depth} · Q`),rawK=await multiply(normalized,w.k,`Block ${l+1}/${c.depth} · K`),values=await multiply(normalized,w.v,`Block ${l+1}/${c.depth} · V`);
  const q=c.rope?rotaryFull(rawQ,c,coordinates):rawQ,k=c.rope?rotaryFull(rawK,c,coordinates):rawK,{heads,combined}=attentionFull(q,k,values,c);
  const delta=await multiply(combined,w.o,`Block ${l+1}/${c.depth} · attention output`),afterAttention=addFull(tokens,delta),ffnInput=normalizeFull(afterAttention,c.rms,c.eps);
  const preActivation=await multiply(ffnInput,w.up,`Block ${l+1}/${c.depth} · MLP expansion`),gate=c.gated?await multiply(ffnInput,w.gate,`Block ${l+1}/${c.depth} · MLP gate`):null,expanded=activateFull(preActivation,gate);
  const mlpDelta=await multiply(expanded,w.down,`Block ${l+1}/${c.depth} · MLP contraction`),output=addFull(afterAttention,mlpDelta);
  const layer={input:tokens,normalized,weights:w,rawQ,rawK,values,q,k,heads,combined,delta,afterAttention,ffnInput,preActivation,gate,expanded,mlpDelta,output};onLayer(layer,l);tokens=output;
 }
 const final=c.cls?normalizeFull(tokens,false,c.eps):tokens;onOutput(final);
 if(c.decoderWidth){const adapterWeights=projectorWeights(c),adapterPre=await multiply(final,adapterWeights[0],'Projector · first linear'),adapterHidden=activateFull(adapterPre),projected=await multiply(adapterHidden,adapterWeights[1],'Projector · second linear');onProjector({adapterPre,adapterHidden,projected,adapterWeights});}
 onProgress({completed:total,total,label:'Complete'});
}
