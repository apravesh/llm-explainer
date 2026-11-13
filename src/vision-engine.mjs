// Vision page maths. The dimensions are Pixtral's vision encoder; the attention scores are illustrative:
// they are built from the patches' colours and positions so the patterns are plausible, not recorded from a model.
export const BLOCKS=24,HEADS=16,WIDTH=1024,HEAD_DIM=64,FF=4096,LLM_WIDTH=5120,PATCH=16,COLS=8,ROWS=6,SHOWN=6;
export const PATCHES=COLS*ROWS,PIXELS=PATCH*PATCH*3;
export const softmax=xs=>{const m=Math.max(...xs);const e=xs.map(x=>Math.exp(x-m));const s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);};
// the six patches shown as rows: a window of the reading-order sequence that contains the selected patch
export const windowStart=sel=>Math.max(0,Math.min(PATCHES-SHOWN,sel-2));
export const patchPlace=p=>({row:Math.floor(p/COLS),col:p%COLS});
export const patchName=p=>`p${p+1}`;
const hash=(...v)=>{let s=2166136261;for(const x of v){s^=x+0x9e3779b9;s=Math.imul(s,16777619)>>>0;s^=s>>>13;}return (s%2000)/1000-1;};
const OFFSETS=[[0,0],[0,1],[0,-1],[1,0],[-1,0],[0,2],[0,-2],[1,1]];
// Illustrative Query·Key scores between the given patches for one block and head.
// Early blocks lean on where a patch is (each head prefers a neighbour in one direction);
// later blocks lean on what it looks like (similar colours attend to each other).
export function attentionRaw(image,block,head,patches){
 const depth=block/(BLOCKS-1),[dr,dc]=OFFSETS[head%OFFSETS.length],sharp=[0.7,1.1,1.7][head%3];
 const wc=(head%2?0.35:0.15)+0.85*depth,wp=1-0.7*depth;
 return patches.map(a=>patches.map(b=>{
  const A=patchPlace(a),B=patchPlace(b),ca=image.mean[a],cb=image.mean[b];
  const like=1-Math.hypot(ca[0]-cb[0],ca[1]-cb[1],ca[2]-cb[2])/441.7;
  const near=Math.exp(-((B.row-A.row-dr)**2+(B.col-A.col-dc)**2)/(2*sharp*sharp));
  return +(30*wc*(like-0.72)+24*wp*near+4*hash(block,head,a,b)).toFixed(2);
 }));
}
// No mask: every patch may attend to every patch, so the stages are only scale and softmax.
export function attentionStages(raw){
 const scaled=raw.map(row=>row.map(v=>v/Math.sqrt(HEAD_DIM)));
 return {scaled,weights:scaled.map(softmax)};
}
export function illustrativeVector(seed,n=12){let s=(seed*2654435761+12345)>>>0;const out=[];for(let i=0;i<n;i++){s^=s<<13;s>>>=0;s^=s>>>17;s^=s<<5;s>>>=0;out.push(((s%2000)/1000-1));}return out;}
