/** Exact scalar autodiff for a tiny causal transformer. UI animation reveals, never invents, math. */
export const PRETRAIN_VOCAB=Object.freeze(['<bos>','the','cat','dog','sat','on','mat','rug']);
export const PRETRAIN_BATCH=Object.freeze([[0,1,2,4,5,6],[0,1,3,4,5,7]].map(Object.freeze));
export const PRETRAIN_DIM=4,PRETRAIN_FFN=8;
class Value{
 constructor(data,parents=[],back=()=>{}){this.data=data;this.grad=0;this.parents=parents;this.back=back;}
 add(other){const b=value(other),a=this;let out=new Value(a.data+b.data,[a,b],()=>{a.grad+=out.grad;b.grad+=out.grad;});return out;}
 mul(other){const b=value(other),a=this;let out=new Value(a.data*b.data,[a,b],()=>{a.grad+=b.data*out.grad;b.grad+=a.data*out.grad;});return out;}
 exp(){const a=this;let out=new Value(Math.exp(a.data),[a],()=>a.grad+=out.data*out.grad);return out;}
 log(){const a=this;let out=new Value(Math.log(a.data),[a],()=>a.grad+=out.grad/a.data);return out;}
 tanh(){const a=this;let out=new Value(Math.tanh(a.data),[a],()=>a.grad+=(1-out.data**2)*out.grad);return out;}
 pow(n){const a=this;let out=new Value(a.data**n,[a],()=>a.grad+=n*a.data**(n-1)*out.grad);return out;}
}
const value=x=>x instanceof Value?x:new Value(x),sum=arr=>arr.reduce((a,b)=>a.add(b),value(0));
function reverse(loss){const ordered=[],seen=new Set(),stack=[[loss,false]];while(stack.length){const [v,done]=stack.pop();if(done){ordered.push(v);continue;}if(seen.has(v))continue;seen.add(v);stack.push([v,true]);for(const p of v.parents)stack.push([p,false]);}ordered.forEach(v=>v.grad=0);loss.grad=1;for(let i=ordered.length-1;i>=0;i--)ordered[i].back();}
const linear=(x,w)=>w[0].map((_,j)=>sum(x.map((v,i)=>v.mul(w[i][j]))));
const add=(a,b)=>a.map((x,i)=>x.add(b[i]));
function norm(x){const mean=sum(x).mul(1/x.length),centered=x.map(v=>v.add(mean.mul(-1))),inv=sum(centered.map(v=>v.mul(v))).mul(1/x.length).add(1e-5).pow(-.5);return centered.map(v=>v.mul(inv));}
function softmax(logits){const max=Math.max(...logits.map(v=>v.data)),exp=logits.map(v=>v.add(-max).exp()),den=sum(exp);return {p:exp.map(v=>v.mul(den.pow(-1))),logZ:den.log().add(max)};}
export function createPretrainModel(seed=73){let state=seed>>>0;const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};const weights=(rows,cols,scale=1)=>Array.from({length:rows},()=>Array.from({length:cols},()=>(random()*2-1)*Math.sqrt(3/rows)*scale));
 return {embedding:weights(8,4,.65),q:weights(4,4,.6),k:weights(4,4,.6),v:weights(4,4,.6),o:weights(4,4,.4),up:weights(4,8,.5),down:weights(8,4,.4),lm:weights(4,8,.6)};
}
function validate(model,batch){if(!Array.isArray(batch)||!batch.length||batch.length>2||batch.some(seq=>!Array.isArray(seq)||seq.length<2||seq.length>6||seq.some(id=>!Number.isInteger(id)||id<0||id>=8)))throw Error('Use one or two sequences, 2–6 valid token IDs each');
 const shapes={embedding:[8,4],q:[4,4],k:[4,4],v:[4,4],o:[4,4],up:[4,8],down:[8,4],lm:[4,8]};for(const [key,[r,c]] of Object.entries(shapes))if(model[key]?.length!==r||model[key].some(row=>row.length!==c||row.some(v=>!Number.isFinite(v))))throw Error('Invalid model matrix '+key);
}
export function pretrainForward(model,batch=PRETRAIN_BATCH,{gradients=true}={}){
 validate(model,batch);const w=Object.fromEntries(Object.entries(model).map(([key,m])=>[key,m.map(row=>row.map(v=>value(v)))]));const traces=[],losses=[];
 for(let b=0;b<batch.length;b++){
  const seq=batch[b],inputs=seq.slice(0,-1),targets=seq.slice(1),embed=inputs.map((id,t)=>w.embedding[id].map((v,j)=>v.add(j%2?Math.cos(t/10000**(2*Math.floor(j/2)/4)):Math.sin(t/10000**(2*Math.floor(j/2)/4))))),normalized=embed.map(norm),q=normalized.map(x=>linear(x,w.q)),k=normalized.map(x=>linear(x,w.k)),v=normalized.map(x=>linear(x,w.v));
  for(let t=0;t<inputs.length;t++){
   const scores=k.slice(0,t+1).map(key=>sum(q[t].map((a,j)=>a.mul(key[j]))).mul(.5)),attention=softmax(scores).p,context=Array.from({length:4},(_,j)=>sum(attention.map((a,i)=>a.mul(v[i][j])))),afterAttention=add(embed[t],linear(context,w.o)),ffnInput=norm(afterAttention),wide=linear(ffnInput,w.up).map(a=>a.tanh()),hidden=add(afterAttention,linear(wide,w.down)),logits=linear(norm(hidden),w.lm),{p,logZ}=softmax(logits),loss=logZ.add(logits[targets[t]].mul(-1));
   losses.push(loss);traces.push({batch:b,position:t,input:inputs[t],target:targets[t],context:inputs.slice(0,t+1),embedding:w.embedding[inputs[t]].map(v=>v.data),probabilities:p.map(v=>v.data),logits:logits.map(v=>v.data),loss:loss.data,attention:attention.map(v=>v.data),hidden:hidden.map(v=>v.data),nodes:{embedding:embed[t],attention:afterAttention,mlp:hidden,logits}});
  }
 }
 const total=sum(losses).mul(1/losses.length);if(gradients)reverse(total);
 const grads=Object.fromEntries(Object.entries(w).map(([key,m])=>[key,m.map(row=>row.map(v=>v.grad))]));
 const summary=Object.entries(w).map(([key,m])=>({key,count:m.length*m[0].length,norm:Math.sqrt(grads[key].flat().reduce((n,g)=>n+g*g,0))}));
 return {loss:total.data,count:losses.length,grads,summary,traces:traces.map(({nodes,...trace})=>({...trace,activationGradients:Object.fromEntries(Object.entries(nodes).map(([key,row])=>[key,row.map(v=>v.grad)]))}))};
}
export function applyPretrainSgd(model,grads,rate=.1){if(!Number.isFinite(rate)||rate<=0||rate>1)throw Error('Learning rate must be in (0,1]');const next=Object.fromEntries(Object.entries(model).map(([key,m])=>[key,m.map((row,i)=>row.map((x,j)=>x-rate*grads[key][i][j]))]));validate(next,PRETRAIN_BATCH);return next;}
export function pretrainStep(model,batch=PRETRAIN_BATCH,rate=.1){const before=pretrainForward(model,batch),afterModel=applyPretrainSgd(model,before.grads,rate),after=pretrainForward(afterModel,batch,{gradients:false});return {before,model:afterModel,after};}
export const PRETRAIN_PHASES=Object.freeze(['batch','forward','loss','backward','update']);
