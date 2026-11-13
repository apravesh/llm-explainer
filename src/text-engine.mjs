// Small, exact maths for the text transformer page. Inputs are values recorded
// from GPT-2 (src/text-data.mjs); nothing here runs a model.
export const HEAD_DIM=64;            // GPT-2 small: 768 / 12 heads
export const softmax=xs=>{const m=Math.max(...xs.filter(Number.isFinite));const e=xs.map(x=>Number.isFinite(x)?Math.exp(x-m):0);const s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);};
// raw[i][j] = query_i · key_j. Returns the three stages the page draws.
export function attentionStages(raw){
 const scaled=raw.map(r=>r.map(x=>x/Math.sqrt(HEAD_DIM)));
 const masked=scaled.map((r,i)=>r.map((x,j)=>j<=i?x:-Infinity));   // a token cannot look ahead
 return {scaled,masked,weights:masked.map(softmax)};
}
// top: [[token,id,logit],…] sorted by logit, highest first.
// sampling: {type:'top-k',value:5} or {type:'top-p',value:0.9}
export function probabilities(top,{temperature=1,sampling={type:'top-k',value:5}}={}){
 const t=Math.max(temperature,1e-6);
 const rows=top.map(([token,id,logit],rank)=>({token,id,logit,rank,scaled:logit/t,kept:false,probability:0}));
 if(sampling.type==='top-k'){const k=Math.max(1,Math.round(sampling.value));rows.forEach(r=>{r.kept=r.rank<k;});}
 else{const all=softmax(rows.map(r=>r.scaled));let sum=0;for(const r of rows){r.kept=true;sum+=all[r.rank];if(sum>=sampling.value)break;}}
 const kept=rows.filter(r=>r.kept),p=softmax(kept.map(r=>r.scaled));
 kept.forEach((r,i)=>{r.probability=p[i];});
 return rows;
}
// rnd in [0,1). Picks one kept token in proportion to its probability.
export function sample(rows,rnd){let acc=0,last=null;for(const r of rows){if(!r.kept)continue;last=r;acc+=r.probability;if(rnd<acc)return r;}return last;}
export const showToken=t=>t.replace(/^ /,'').replace(/\n/g,'\\n');
// Deterministic stand-in numbers for vectors the recording does not include
// (embeddings, MLP activations). Same seed, same picture; labelled as illustrative on the page.
export function illustrativeVector(seed,n=12){let s=(seed*2654435761+12345)>>>0;const out=[];for(let i=0;i<n;i++){s^=s<<13;s>>>=0;s^=s>>>17;s^=s<<5;s>>>=0;out.push(((s%2000)/1000-1));}return out.map(x=>Math.round(x*100)/100);}
