/** A deterministic, UNTRAINED teaching transformer. No semantic predictions. */
export const DIM = 24, HEADS = 3, HEAD_DIM = 8, DEPTH = 4, FF_DIM = 48, LLM_DIM = 32;
export function rng(seed = 42) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
export const dot = (a,b) => a.reduce((s,x,i) => s+x*b[i],0);
export function softmax(x) { const m=Math.max(...x), e=x.map(v=>Math.exp(v-m)), s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s); }
export function rotatePair([a,b], angle) { const c=Math.cos(angle), s=Math.sin(angle); return [a*c-b*s,a*s+b*c]; }
/** Pixtral paper convention: alternate row / column pairs. */
export function rope(v, row, col, mode='2d', width=6, base=10000) { const out=[]; for(let i=0;i<v.length/2;i++) { const freq=base**(-2*i/v.length); const p=mode==='2d' ? (i%2===0?row:col) : row*width+col; out.push(...rotatePair(v.slice(2*i,2*i+2),p*freq)); } return out; }
export function norm(x, rms=false) { const mean=rms?0:x.reduce((a,b)=>a+b,0)/x.length; const sd=Math.sqrt(x.reduce((s,v)=>s+(v-mean)**2,0)/x.length+1e-5); return x.map(v=>(v-mean)/sd); }
export const gelu=x=>0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*x*x*x)));
const silu=x=>x/(1+Math.exp(-x));
function weights(rows,cols,seed,scale=1) { const r=rng(seed), s=scale*Math.sqrt(3/rows); return Array.from({length:rows},()=>Array.from({length:cols},()=> (r()*2-1)*s)); }
export function linear(x,w) { const out=Array(w[0].length).fill(0); for(let i=0;i<x.length;i++) for(let j=0;j<out.length;j++) out[j]+=x[i]*w[i][j]; return out; }
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const projectionCache=new Map();
const blocks=Array.from({length:DEPTH},(_,l)=>({q:weights(DIM,DIM,100+l*10),k:weights(DIM,DIM,101+l*10),v:weights(DIM,DIM,102+l*10),o:weights(DIM,DIM,103+l*10,.45),up:weights(DIM,FF_DIM,104+l*10,.7),down:weights(FF_DIM,DIM,105+l*10,.45),gate:weights(DIM,FF_DIM,106+l*10,.7)}));
const adapter1=weights(DIM,LLM_DIM,888),adapter2=weights(LLM_DIM,LLM_DIM,889);
/** Preprocessing: teaching-sized crop (ViT), aspect-preserving resize + pad (Pixtral). */
export function imageShape(width,height,patch,model) { if(model==='vit')return {width:96,height:96}; const scale=128/Math.max(width,height); return {width:Math.max(patch,Math.ceil(width*scale/patch)*patch),height:Math.max(patch,Math.ceil(height*scale/patch)*patch)}; }
/** RGBA pixels -> row-major flattened RGB patches, normalized to [-1,1]. */
export function patchify(data,width,height,patch) { const cols=width/patch,rows=height/patch; if(!Number.isInteger(cols)||!Number.isInteger(rows))throw Error('Image dimensions must be patch-aligned'); const result=[]; for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const pixels=[],mean=[0,0,0];for(let y=0;y<patch;y++)for(let x=0;x<patch;x++){const offset=((r*patch+y)*width+c*patch+x)*4;for(let z=0;z<3;z++){pixels.push(data[offset+z]/127.5-1);mean[z]+=data[offset+z]/(patch*patch);}}result.push({row:r,col:c,pixels,mean});}return result; }
export function runEncoder(patches,cols,model='vit') {
  if(!patches.length)throw Error('Need at least one patch');
  const size=patches[0].pixels.length;
  if(!projectionCache.has(size))projectionCache.set(size,weights(size,DIM,42+size));
  const projectionWeights=projectionCache.get(size);
  const embeddings=patches.map(p=>linear(p.pixels,projectionWeights));
  const posRandom=rng(221), positions=Array.from({length:patches.length+1},()=>Array.from({length:DIM},()=> (posRandom()-.5)*.6));
  const clsRandom=rng(555), cls=Array.from({length:DIM},()=> (clsRandom()-.5)*.2);
  let tokens=model==='vit' ? [cls,...embeddings].map((v,i)=>add(v,positions[i])) : embeddings.map(x=>norm(x,true));
  const input=tokens.map(x=>x.slice()),layers=[];
  for(let l=0;l<DEPTH;l++){
    const w=blocks[l],normalized=tokens.map(x=>norm(x,model==='pixtral'));
    const q0=normalized.map(x=>linear(x,w.q)),k0=normalized.map(x=>linear(x,w.k)),v=normalized.map(x=>linear(x,w.v));
    const heads=[]; const combined=tokens.map(()=>[]);
    for(let head=0;head<HEADS;head++){
      let q=q0.map(x=>x.slice(head*HEAD_DIM,(head+1)*HEAD_DIM)),k=k0.map(x=>x.slice(head*HEAD_DIM,(head+1)*HEAD_DIM));
      const values=v.map(x=>x.slice(head*HEAD_DIM,(head+1)*HEAD_DIM)),rawQ=q,rawK=k;
      if(model==='pixtral'){q=q.map((x,i)=>rope(x,patches[i].row,patches[i].col));k=k.map((x,i)=>rope(x,patches[i].row,patches[i].col));}
      const scores=q.map(query=>k.map(key=>dot(query,key)/Math.sqrt(HEAD_DIM))),attention=scores.map(softmax);
      const out=attention.map(row=>Array.from({length:HEAD_DIM},(_,d)=>row.reduce((s,a,j)=>s+a*values[j][d],0)));
      out.forEach((x,i)=>combined[i].push(...x)); heads.push({q,k,rawQ,rawK,values,scores,attention,out});
    }
    const delta=combined.map(x=>linear(x,w.o)),afterAttention=tokens.map((x,i)=>add(x,delta[i]));
    const ffnInput=afterAttention.map(x=>norm(x,model==='pixtral'));
    const preActivation=ffnInput.map(x=>linear(x,w.up));
    const gate=model==='pixtral'?ffnInput.map(x=>linear(x,w.gate)):null;
    const expanded=preActivation.map((a,i)=>model==='pixtral'?a.map((v,j)=>silu(v)*gate[i][j]):a.map(gelu));
    const mlpDelta=expanded.map(x=>linear(x,w.down)),output=afterAttention.map((x,i)=>add(x,mlpDelta[i]));
    layers.push({input:tokens,normalized,weights:w,heads,delta,afterAttention,ffnInput,preActivation,gate,expanded,mlpDelta,output});tokens=output;
  }
  const final=model==='vit'?tokens.map(x=>norm(x)):tokens;
  const adapterPre=model==='pixtral'?final.map(x=>linear(x,adapter1)):null;
  const adapterHidden=adapterPre?adapterPre.map(x=>x.map(gelu)):null;
  const projected=adapterHidden?adapterHidden.map(x=>linear(x,adapter2)):null;
  return {embeddings,projectionWeights,positions,cls,input,layers,final,adapterPre,adapterHidden,adapterWeights:[adapter1,adapter2],projected};
}
