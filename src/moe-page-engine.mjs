// MoE page maths.
// The SIZES shown on the page are those of a real MoE model, Mixtral 8x7B (Mistral AI): 32 blocks, 32 heads,
// width 4,096 (heads of 128), experts of 14,336, 8 experts per block of which the top 2 run.
// The NUMBERS drawn (attention scores, router scores, which experts are chosen, their weights) cannot come from
// Mixtral itself: no model is downloaded. They are real arithmetic of a small untrained stand-in with the same
// layout (32 blocks, 32 heads, 8 experts, top-2) but a narrower width, built here from seeded random weights.
// So the routes show how routing works, not what Mixtral actually does. The next-token lists are illustrative.
import {moeRng} from './moe-engine.mjs';

// ---- what the page displays: Mixtral 8x7B's published dimensions
export const BLOCKS=32,HEADS=32,WIDTH=4096,HEAD_DIM=WIDTH/HEADS,FF=14336,EXPERTS=8,TOP_K=2;
export const MIXTRAL=Object.freeze({name:'Mixtral 8x7B',blocks:32,heads:32,width:4096,ff:14336,experts:8,topK:2,stored:46.7e9,active:12.9e9});
// ---- what is actually computed: the stand-in's own, much smaller widths (same counts of blocks, heads and experts)
export const STAND_IN=Object.freeze({width:64,heads:HEADS,headDim:2,ff:96,blocks:BLOCKS,experts:EXPERTS,topK:TOP_K});
// Six-token sentences. `next` is an illustrative list of next-token guesses: this untrained model cannot predict words.
export const EXAMPLES=Object.freeze([
 {text:'The little robot reads a book',next:[[' about',.24],['.',.21],[' to',.13],[' and',.11],[' on',.07],[' with',.05],[' every',.03],[' called',.02]]},
 {text:'Bright stars shine above quiet oceans',next:[['.',.31],[' and',.19],[',',.14],[' at',.08],[' of',.06],[' while',.04],[' as',.03],[' where',.02]]},
 {text:'One two three four five six',next:[[' seven',.86],[',',.04],[' and',.02],[' or',.01],['.',.01],[' eight',.01],[' six',.005],[' times',.003]]}
]);


const SW=STAND_IN.width,SH=STAND_IN.headDim,SF=STAND_IN.ff;
const weights=(rows,cols,seed,scale=1)=>{const random=moeRng(seed),s=scale*Math.sqrt(3/rows);return Array.from({length:rows},()=>Array.from({length:cols},()=>(2*random()-1)*s));};
const linear=(x,w)=>w[0].map((_,j)=>{let t=0;for(let i=0;i<x.length;i++)t+=x[i]*w[i][j];return t;});
const norm=x=>{const m=x.reduce((a,b)=>a+b,0)/x.length,v=x.reduce((a,b)=>a+(b-m)**2,0)/x.length;return x.map(a=>(a-m)/Math.sqrt(v+1e-5));};
const softmax=xs=>{const m=Math.max(...xs),e=xs.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);};
const silu=x=>x/(1+Math.exp(-x));
// Query and Key weights are scaled up so that the stand-in's scores, divided by √128 like Mixtral's, still spread out.
const QK_GAIN=7;
export function createModel(seed=108){
 return {width:SW,heads:STAND_IN.heads,depth:BLOCKS,activation:'swiglu',
  blocks:Array.from({length:BLOCKS},(_,b)=>{const base=seed+10000*(b+1);return {
   attention:{q:weights(SW,SW,base+1,QK_GAIN),k:weights(SW,SW,base+2,QK_GAIN),v:weights(SW,SW,base+3),o:weights(SW,SW,base+4,.45)},
   // each expert is a gated (SwiGLU) MLP, as in Mixtral: W1 and W3 widen, W2 narrows
   ffn:{activation:'swiglu',router:weights(SW,EXPERTS,base+100,2.2),experts:Array.from({length:EXPERTS},(_,e)=>({up:weights(SW,SF,base+110+e*3),gate:weights(SW,SF,base+111+e*3),down:weights(SF,SW,base+112+e*3,.65)}))}};})};
}
const MODEL=createModel();
// A fixed vector for each word and position, so the same sentence always gives the same picture.
function embed(label,index){let h=2166136261;for(const ch of `${label}:${index}`)h=Math.imul(h^ch.codePointAt(0),16777619);const random=moeRng(h);return Array.from({length:SW},(_,j)=>(2*random()-1)+(j%2?Math.cos:Math.sin)(index/10000**(2*Math.floor(j/2)/SW)));}
function expertForward(x,e){const a=linear(x,e.up),g=linear(x,e.gate),activated=a.map((v,i)=>silu(v)*g[i]);return {activated,output:linear(activated,e.down)};}
const cache=new Map();
// Runs the whole stand-in once per sentence and keeps, per block: the queries and keys, and one route per token.
export function run(text){
 if(cache.has(text))return cache.get(text);
 const labels=String(text).trim().split(/\s+/u).filter(Boolean).slice(0,12);
 let hidden=labels.map(embed);
 const blocks=MODEL.blocks.map(block=>{
  // causal multi-head attention
  const xs=hidden.map(norm),q=xs.map(x=>linear(x,block.attention.q)),k=xs.map(x=>linear(x,block.attention.k)),v=xs.map(x=>linear(x,block.attention.v));
  const mixedAll=hidden.map(()=>Array(SW).fill(0));
  for(let head=0;head<STAND_IN.heads;head++){
   const off=head*SH;
   hidden.forEach((_,i)=>{
    const p=softmax(hidden.map((_,j)=>{if(j>i)return -Infinity;let t=0;for(let d=0;d<SH;d++)t+=q[i][off+d]*k[j][off+d];return t/Math.sqrt(HEAD_DIM);}));
    for(let d=0;d<SH;d++)mixedAll[i][off+d]=p.reduce((t,w,j)=>t+w*v[j][off+d],0);
   });
  }
  const afterAttention=hidden.map((row,i)=>{const delta=linear(mixedAll[i],block.attention.o);return row.map((x,j)=>x+delta[j]);});
  // the MoE layer: score the experts, keep the top k, softmax over those, run only them, add the weighted outputs
  const routes=afterAttention.map(input=>{
   const scores=linear(norm(input),block.ffn.router);
   const chosen=scores.map((s,e)=>[s,e]).sort((a,b)=>b[0]-a[0]).slice(0,TOP_K).map(x=>x[1]).sort((a,b)=>a-b);
   const z=chosen.reduce((t,e)=>t+Math.exp(scores[e]),0),gates=scores.map((s,e)=>chosen.includes(e)?Math.exp(s)/z:0);
   const experts=block.ffn.experts.map((e,i)=>chosen.includes(i)?expertForward(norm(input),e):null);
   const mixed=Array.from({length:SW},(_,d)=>chosen.reduce((t,e)=>t+gates[e]*experts[e].output[d],0));
   return {scores,chosen,gates,experts,mixed,input,output:input.map((x,d)=>x+mixed[d])};
  });
  const loads=Array(EXPERTS).fill(0);routes.forEach(r=>r.chosen.forEach(e=>loads[e]++));
  hidden=routes.map(r=>r.output);
  return {attention:{q,k},routes,loads};
 });
 const out={tokens:labels,blocks};cache.set(text,out);return out;
}
// Query · Key for every pair of tokens in one head, before scaling and masking.
export function attentionRaw(attention,head){
 const off=head*SH,n=attention.q.length;
 return Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>{let s=0;for(let d=0;d<SH;d++)s+=attention.q[i][off+d]*attention.k[j][off+d];return s;}));
}
// Scale by √(head size), hide later tokens, softmax each row: the same three stages as on the Transformer page.
// The divisor is Mixtral's √128, the number the page shows.
export function attentionStages(raw){
 const scaled=raw.map(row=>row.map(v=>v/Math.sqrt(HEAD_DIM)));
 return {scaled,weights:scaled.map((row,i)=>softmax(row.map((v,j)=>j>i?-Infinity:v)))};
}
// Weight counts of ONE MoE layer at Mixtral's sizes, and of one ordinary gated MLP of the same shape.
// A gated expert has three matrices (two of FF × WIDTH, one of WIDTH × FF).
export function layerCost(){
 const expert=3*WIDTH*FF,router=WIDTH*EXPERTS;
 return {expert,router,stored:router+EXPERTS*expert,active:router+TOP_K*expert,dense:expert};
}
