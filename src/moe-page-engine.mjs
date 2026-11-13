// MoE page maths. A tiny UNTRAINED transformer with a Mixture-of-Experts layer in every block, run for real:
// width 8, 2 causal heads of 4, 4 blocks, and per block a router plus 8 expert MLPs (8 → 16 → 8) of which the top 2 run.
// It reuses the teaching engine (src/moe-engine.mjs, src/moe-transformer.mjs) and only widens the pool to 8 experts.
// Every router score, choice, gate weight, expert output and attention score on the page is this model's arithmetic.
// Its weights are seeded random numbers, so the routes carry no meaning. The next-token lists are illustrative.
import {MOE_DIM,MOE_HIDDEN,moeRng,teachingTokens} from './moe-engine.mjs';
import {runMoeTransformer} from './moe-transformer.mjs';

export const BLOCKS=4,HEADS=2,WIDTH=MOE_DIM,HEAD_DIM=WIDTH/HEADS,FF=MOE_HIDDEN,EXPERTS=8,TOP_K=2;
// Published sizes of a real MoE model, for the comparison rows (Mixtral 8x7B, Mistral AI).
export const MIXTRAL=Object.freeze({name:'Mixtral 8x7B',blocks:32,heads:32,width:4096,ff:14336,experts:8,topK:2,stored:46.7e9,active:12.9e9});
// Six-token sentences. `next` is an illustrative list of next-token guesses: this untrained model cannot predict words.
export const EXAMPLES=Object.freeze([
 {text:'The little robot reads a book',next:[[' about',.24],['.',.21],[' to',.13],[' and',.11],[' on',.07],[' with',.05],[' every',.03],[' called',.02]]},
 {text:'Bright stars shine above quiet oceans',next:[['.',.31],[' and',.19],[',',.14],[' at',.08],[' of',.06],[' while',.04],[' as',.03],[' where',.02]]},
 {text:'One two three four five six',next:[[' seven',.86],[',',.04],[' and',.02],[' or',.01],['.',.01],[' eight',.01],[' six',.005],[' times',.003]]}
]);

const weights=(rows,cols,seed,scale=1)=>{const random=moeRng(seed),s=scale*Math.sqrt(3/rows);return Array.from({length:rows},()=>Array.from({length:cols},()=>(2*random()-1)*s));};
// Same shapes and seeding scheme as createMoeTransformer(), with EXPERTS experts per block instead of four.
export function createModel(seed=108){
 return {width:WIDTH,heads:HEADS,depth:BLOCKS,activation:'gelu',
  blocks:Array.from({length:BLOCKS},(_,b)=>{const base=seed+10000*(b+1);return {
   attention:{q:weights(WIDTH,WIDTH,base+1),k:weights(WIDTH,WIDTH,base+2),v:weights(WIDTH,WIDTH,base+3),o:weights(WIDTH,WIDTH,base+4,.45)},
   ffn:{activation:'gelu',router:weights(WIDTH,EXPERTS,base+100),experts:Array.from({length:EXPERTS},(_,e)=>({up:weights(WIDTH,FF,base+110+e*2),down:weights(FF,WIDTH,base+111+e*2,.65)}))}};}),
  lmHead:weights(WIDTH,8,seed+900001,.6)};
}
const MODEL=createModel();
const cache=new Map();
// Runs the whole stack once per sentence and keeps, per block: the attention trace and one route per token.
export function run(text){
 if(cache.has(text))return cache.get(text);
 const tokens=teachingTokens(text),stack=runMoeTransformer(tokens,MODEL,{k:TOP_K,weighting:'selected'});
 const blocks=stack.blocks.map(b=>{
  const routes=b.ffn.traces.map(t=>({
   scores:t.router.logits,chosen:[...t.selected].sort((x,y)=>x-y),gates:t.gates,
   // only the chosen experts were evaluated; the others are null
   experts:t.experts.map(e=>e&&{activated:e.activated,output:e.output}),mixed:t.mixed,input:t.hidden,output:t.output}));
  const loads=Array(EXPERTS).fill(0);routes.forEach(r=>r.chosen.forEach(e=>loads[e]++));
  return {attention:b.attention,routes,loads};
 });
 const out={tokens:tokens.map(t=>t.label),blocks};cache.set(text,out);return out;
}
// Query · Key for every pair of tokens in one head, before scaling and masking.
export function attentionRaw(attention,head){
 const off=head*HEAD_DIM,n=attention.q.length;
 return Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>{let s=0;for(let d=0;d<HEAD_DIM;d++)s+=attention.q[i][off+d]*attention.k[j][off+d];return s;}));
}
const softmax=xs=>{const m=Math.max(...xs),e=xs.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);};
// Scale by √(head size), hide later tokens, softmax each row: the same three stages as on the Transformer page.
export function attentionStages(raw){
 const scaled=raw.map(row=>row.map(v=>v/Math.sqrt(HEAD_DIM)));
 return {scaled,weights:scaled.map((row,i)=>softmax(row.map((v,j)=>j>i?-Infinity:v)))};
}
// Weight counts of one MoE layer of this toy model, and of one ordinary MLP of the same shape.
export function layerCost(){
 const expert=2*WIDTH*FF,router=WIDTH*EXPERTS;
 return {expert,router,stored:router+EXPERTS*expert,active:router+TOP_K*expert,dense:expert};
}
