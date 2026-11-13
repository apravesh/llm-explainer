// Mixture of experts: explorer and section labs.
// Built to match the text page (src/text-app.mjs) step for step, so the one difference stands out:
// where the text transformer has one MLP, each token goes through a router, the top-2 of 8 expert MLPs
// and a weighted sum. Embedding, attention and the output step are drawn as on the text page.
// The numbers are real arithmetic of a tiny untrained model (src/moe-page-engine.mjs); the list of
// next-token guesses is illustrative.
import {BLOCKS,HEADS,WIDTH,HEAD_DIM,FF,EXPERTS,TOP_K,MIXTRAL,EXAMPLES,run,attentionRaw,attentionStages,layerCost} from './moe-page-engine.mjs';

const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=p=>p>=0.9995?'100%':p<=0?'0%':(p*100).toFixed(p<0.1?1:0)+'%';
const state={ex:0,block:0,head:0,expand:null,anim:null,last:null};
const ex=()=>EXAMPLES[state.ex],data=()=>run(ex().text),block=()=>data().blocks[state.block];
const raw=()=>attentionRaw(block().attention,state.head);
const names=es=>es.map(e=>e+1).join(' and ');

function syncSteppers(){
 $$('div[data-stepper]').forEach(el=>{
  el.innerHTML=`<button type="button" data-step="-1" aria-label="Previous block">‹</button><b>Block ${state.block+1} of ${BLOCKS}</b><button type="button" data-step="1" aria-label="Next block">›</button>`;
 });
}

/* ---------- explorer flow diagram ---------- */
// Same coordinates as the text page up to the end of attention. The MoE layer is wider than the MLP it
// replaces, so everything after it sits MS further right.
const MS=104,W=1230+MS,Y0=106,PITCH=54,TP=13;
const EX0=856,EP=14,ES=11,xe=e=>EX0+e*EP,xc=e=>xe(e)+ES/2;          // expert squares in the overview
const M0=884,MG=46,MW=EXPERTS*TP,mE=k=>M0+k*(MW+MG),cxE=(e,k)=>mE(k)+e*TP;   // the three router grids when the layer is open
const XO=1460,DM=XO-(902+MS),RS=170;                                 // opened: where the layer's output bar goes, and how far the picture moves left
const geo=n=>{const H=Y0+n*PITCH+48,top=86,bottom=H-30,ch=bottom-top,gy={k:top+ch*0.2,q:top+ch*0.5,v:top+ch*0.82};return {H,top,ch,yc:i=>Y0+i*PITCH+PITCH/2,ty:(g,i)=>gy[g]+(i-(n-1)/2)*TP};};
const blockLabel=()=>`TRANSFORMER BLOCK ${state.block+1} / ${BLOCKS}`,headLabel=()=>`MULTI-HEAD SELF-ATTENTION, HEAD ${state.head+1} / ${HEADS}`;
const chip=(kind,x,y,label,w)=>{
 const btn=(d,bx)=>`<g class="step" data-step="${d}" role="button" tabindex="0" aria-label="${d>0?'Next':'Previous'} ${kind}"><rect x="${bx}" y="${y-9}" width="18" height="18" rx="4"/><text x="${bx+9}" y="${y+4.5}" text-anchor="middle">${d>0?'›':'‹'}</text></g>`;
 return `<g class="chip" data-stepper="${kind}"><rect class="chip-bg" x="${x-7}" y="${y-11}" width="${w+64}" height="22"/>${btn(-1,x)}<text class="chip-label ${kind}-label" x="${x+25+w/2}" y="${y+3.5}" text-anchor="middle">${label}</text>${btn(1,x+32+w)}</g>`;
};
const curve=(x1,y1,x2,y2)=>{const m=(x1+x2)/2;return `M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}`;};
const wordLine=(y1,y2,x2)=>`M${978+MS},${y1} C${994+MS},${y1} ${1026+MS},${y2} ${1044+MS},${y2} H${Math.max(1046+MS,x2).toFixed(1)}`;
const fanArea=(y,top,bot)=>`M${976+MS},${y-9} C${994+MS},${y-9} ${1026+MS},${top} ${1044+MS},${top} V${bot} C${1026+MS},${bot} ${994+MS},${y+9} ${976+MS},${y+9} Z`;
// Attention as its three stages, exactly as on the text page. Returns the 0..1 shade of one dot, or null where the mask removed it.
const STAGE_NAMES=['Dot product','Scaling · Mask','Softmax'];
function stageShade(m,i,j,r=raw()){
 if(m===2)return j>i?null:attentionStages(r).weights[i][j];
 if(m===1&&j>i)return null;
 const flat=r.flat(),lo=Math.min(...flat),hi=Math.max(...flat);
 return (r[i][j]-lo)/((hi-lo)||1);
}
const stageRange=(m,r=raw())=>{const flat=r.flat(),lo=Math.min(...flat),hi=Math.max(...flat),d=m===1?Math.sqrt(HEAD_DIM):1;return m===2?['0','1']:[(lo/d).toFixed(1),(hi/d).toFixed(1)];};
const dotOpacity=v=>(0.07+0.93*v).toFixed(3);
// The router as its three stages: scores → top-k → softmax over the chosen. null where an expert was dropped.
const ROUTE_NAMES=['Router scores',`Top-${TOP_K}`,'Softmax'];
const scoreRange=(B=block())=>{const flat=B.routes.flatMap(r=>r.scores);return [Math.min(...flat),Math.max(...flat)];};
function routeShade(m,i,e,B=block()){
 const r=B.routes[i],on=r.chosen.includes(e);
 if(m===2)return on?r.gates[e]:null;
 if(m===1&&!on)return null;
 const [lo,hi]=scoreRange(B);return (r.scores[e]-lo)/((hi-lo)||1);
}
const FORMULAS=[
 `<h4>Dot product</h4><div class="fx">score<sub>i,j</sub> = Q<sub>i</sub> · K<sub>j</sub> = <span class="sum">Σ<sub>d=1</sub><sup>${HEAD_DIM}</sup></span> Q<sub>i,d</sub> × K<sub>j,d</sub></div><p>The Query of token i (the row) is multiplied with the Key of token j (the column), number by number, and the products are added up. Nothing here is specific to MoE. (A head has ${HEAD_DIM} numbers in this page's small model, 64 in GPT-2.)</p>`,
 `<h4>Scaling · Mask</h4><div class="fx">scaled<sub>i,j</sub> = score<sub>i,j</sub> ÷ √${HEAD_DIM} &nbsp;&nbsp;if j ≤ i<br>scaled<sub>i,j</sub> = −∞ &nbsp;&nbsp;if j &gt; i</div><p>Dividing by √${HEAD_DIM} = ${Math.sqrt(HEAD_DIM)} keeps the scores in a range softmax handles well. Tokens that come later (j &gt; i) are masked out, so a token cannot look ahead. The same as in the text transformer.</p>`,
 `<h4>Softmax</h4><div class="fx">weight<sub>i,j</sub> = <span class="frac"><span>exp(scaled<sub>i,j</sub>)</span><span><span class="sum">Σ<sub>k</sub></span> exp(scaled<sub>i,k</sub>)</span></span></div><p>Each row is turned into weights that add up to 1, and −∞ becomes 0.</p>`,
 `<h4>Out</h4><div class="fx">out<sub>i</sub> = <span class="sum">Σ<sub>j</sub></span> weight<sub>i,j</sub> × V<sub>j</sub></div><p>Each token's output is a blend of the Value vectors, mixed by its row of attention weights. Attention is shared by all experts: it runs once, before the router.</p>`,
 `<h4>MoE layer</h4><div class="fx">MoE(x) = <span class="sum">Σ<sub>e ∈ top-${TOP_K}</sub></span> g<sub>e</sub> · Expert<sub>e</sub>(x)</div><p>This replaces the single MLP of the text transformer. A router scores the ${EXPERTS} experts for this token, only the ${TOP_K} best run, and their outputs are added with weights g that sum to 1. Click to open it step by step.</p>`,
 `<h4>Router scores</h4><div class="fx">s = W<sub>r</sub> · x<br>W<sub>r</sub> : matrix(${EXPERTS} × ${WIDTH})</div><p>x is the token's vector after attention. The router is one small learned matrix with a row per expert, so the result is ${EXPERTS} scores: how well each expert suits this token. No expert has run yet.</p>`,
 `<h4>Top-${TOP_K}</h4><div class="fx">keep the ${TOP_K} highest scores<br>all others → −∞</div><p>Only the ${TOP_K} best-scoring experts stay. The other ${EXPERTS-TOP_K} are not run at all for this token: their weights are stored but do no work. It is the top-k filter from the last step of the text transformer, applied to experts instead of words.</p>`,
 `<h4>Softmax over the chosen</h4><div class="fx">g<sub>e</sub> = <span class="frac"><span>exp(s<sub>e</sub>)</span><span><span class="sum">Σ<sub>c ∈ chosen</sub></span> exp(s<sub>c</sub>)</span></span></div><p>The ${TOP_K} surviving scores are turned into weights that add up to 1. A dropped expert (−∞) gets exactly 0.</p>`,
 `<h4>Experts run</h4><div class="fx">y<sub>e</sub> = W<sub>2,e</sub> · GELU(W<sub>1,e</sub> · x)</div><p>Each chosen expert is an ordinary MLP, the same shape as the one in the text transformer (here ${WIDTH} → ${FF} → ${WIDTH}), with its own weights. Both receive the whole vector x, not a piece of it.</p>`,
 `<h4>Weighted sum</h4><div class="fx">y = <span class="sum">Σ<sub>e ∈ chosen</sub></span> g<sub>e</sub> · y<sub>e</sub><br>output = x + y</div><p>Each expert's output is multiplied by its weight and the results are added, number by number: ${WIDTH} numbers again, not two vectors side by side. The sum is added to the vector that came in (the residual connection).</p>`,
 `<h4>Output projection</h4><div class="fx">logits = W<sub>out</sub> · h<sub>last</sub></div><p>Unchanged from the text transformer: the last token's final vector is multiplied by the vocabulary matrix and softmax turns the scores into probabilities. This page's model is untrained, so the list on the right is illustrative.</p>`
];
const blocksLeft=()=>BLOCKS-1-state.block;
function paintMore(root=$('#tx-flow')){
 const left=blocksLeft(),bars=Math.min(3,left+1);
 root.querySelectorAll('.more-a').forEach(t=>{t.textContent=left?`${left} MORE`:'LAST';});
 root.querySelectorAll('.more-b').forEach(t=>{t.textContent=left===1?'BLOCK':left?'BLOCKS':'BLOCK';});
 root.querySelectorAll('.more-bar').forEach(r=>{r.style.display=+r.dataset.b<bars?'':'none';});
}
// The MoE layer, one row per token. Overview: router → the pool of experts with the chosen two lit → Σ.
// Opened: router scores → top-k → softmax over the chosen → the chosen experts' MLPs → weighted sum.
function moeRows(opening=false){
 const {tokens}=data(),n=tokens.length,B=block(),{yc,ty}=geo(n),EM=state.expand==='moe';
 const late=at=>opening?` late" style="--at:${at}ms`:'';
 const heads=(x,y)=>Array.from({length:EXPERTS},(_,e)=>`<text class="ex-h" data-e="${e}" x="${x(e)}" y="${y}" text-anchor="middle">${e+1}</text>`).join('');
 let s='';
 if(!EM){
  s+=heads(xc,Y0+9);
  tokens.forEach((t,i)=>{
   const y=yc(i),r=B.routes[i],[a,b]=r.chosen;
   s+=`<g data-t="${i}"><path class="fo" d="${curve(716+14,ty('q',i),806,y)}" stroke-width="1.2" opacity=".7"/><rect data-dim="vector(${WIDTH})" x="806" y="${y-15}" width="10" height="30" rx="2" fill="#d0bfe1"/>`;
   s+=`<path class="fr" d="M816,${y}H826"/><rect class="rt" data-dim="router W_r · matrix(${EXPERTS} × ${WIDTH}): one score per expert" x="826" y="${y-7}" width="14" height="14" rx="3"/><text class="rt-t" x="833" y="${y+3.2}" text-anchor="middle">R</text>`;
   s+=`<path class="fr" d="M840,${y-4}C847,${y-4} 845,${y-11} 852,${y-11}H${xc(b)}V${y-6.5}M${xc(a)},${y-11}V${y-6.5}"/>`;
   for(let e=0;e<EXPERTS;e++){
    const on=r.chosen.includes(e);
    s+=`<rect class="ex ${on?'run':'idle'}" data-e="${e}" data-dim="${on?`expert ${e+1} runs for “${esc(t)}” · weight ${pct(r.gates[e])}`:`expert ${e+1} · idle for “${esc(t)}”`}" x="${xe(e)}" y="${y-ES/2}" width="${ES}" height="${ES}" rx="2.5"${on?` fill-opacity="${(0.4+0.6*r.gates[e]).toFixed(2)}"`:''}/>`;
    if(on)s+=`<text class="ex-n" x="${xc(e)}" y="${y+2.6}" text-anchor="middle">${e+1}</text>`;
   }
   s+=`<path class="fx" d="M${xc(a)},${y+6.5}V${y+11}H974C981,${y+11} 979,${y} 986,${y}M${xc(b)},${y+6.5}V${y+11}"/><circle class="sg" data-dim="${r.chosen.map(e=>`${pct(r.gates[e])} × expert ${e+1}`).join(' + ')}" cx="993" cy="${y}" r="7"/><text class="sg-t" x="993" y="${y+3.6}" text-anchor="middle">Σ</text>`;
   s+=`<path class="fx" d="M1000,${y}H1006"/><rect data-dim="vector(${WIDTH})" x="${902+MS}" y="${y-15}" width="10" height="30" rx="2" fill="#a9c0f3"/></g>`;
  });
  const dy=yc(n-1)+38;
  s+=`<text class="mlp-size" x="811" y="${dy}" text-anchor="middle">${WIDTH}</text><text class="mlp-size" x="${(xe(0)+xe(EXPERTS-1)+ES)/2}" y="${dy}" text-anchor="middle">${TOP_K} of ${EXPERTS} experts run</text><text class="mlp-size" x="${907+MS}" y="${dy}" text-anchor="middle">${WIDTH}</text>`;
  return s;
 }
 const AT=[300,1100,1900,2700,3500],[lo,hi]=scoreRange(B),mid=k=>mE(k)+MW/2-TP/2,dy=yc(n-1)+38;
 // the three grids: one column per expert, one row per token
 for(let k=0;k<3;k++){
  s+=`<g class="${late(AT[k]).trim()}">${heads(e=>cxE(e,k),Y0+9)}`;
  tokens.forEach((t,i)=>{
   const y=yc(i);
   for(let e=0;e<EXPERTS;e++){const v=routeShade(k,i,e,B);s+=`<rect class="rcell${v===null?' dropped':''}" data-q="${i}" data-e="${e}" data-m="${k}" x="${cxE(e,k)-5}" y="${y-11}" width="10" height="22" rx="2"${v===null?'':` fill-opacity="${dotOpacity(v)}"`}/>`;}
  });
  s+=k===0?`<text class="leg" x="${mid(k)-20}" y="${dy}" text-anchor="end">${lo.toFixed(1)}</text><rect class="leg-bar rleg" x="${mid(k)-16}" y="${dy-6}" width="32" height="4" rx="2"/><text class="leg" x="${mid(k)+20}" y="${dy}">${hi.toFixed(1)}</text>`
   :k===1?`<text class="mlp-size" x="${mid(k)}" y="${dy}" text-anchor="middle">${EXPERTS-TOP_K} of ${EXPERTS} dropped per token</text>`
   :`<text class="leg" x="${mid(k)-20}" y="${dy}" text-anchor="end">0</text><rect class="leg-bar rleg" x="${mid(k)-16}" y="${dy-6}" width="32" height="4" rx="2"/><text class="leg" x="${mid(k)+20}" y="${dy}">1</text>`;
  if(k<2)s+=`<text class="stage-arrow" x="${mE(k)+MW-TP/2+MG/2}" y="${(yc(0)+yc(n-1))/2+4}" text-anchor="middle">→</text>`;
  s+=`</g>`;
 }
 tokens.forEach((t,i)=>{
  const y=yc(i),r=B.routes[i],st=i*40;
  s+=`<g data-t="${i}"><path class="fo" d="${curve(716+14,ty('q',i),806,y)}" stroke-width="1.2" opacity=".7"/><text class="mini" x="816" y="${y-20}" text-anchor="end">${esc(t)}</text><rect data-dim="vector(${WIDTH})" x="806" y="${y-15}" width="10" height="30" rx="2" fill="#d0bfe1"/></g>`;
  s+=`<g class="${late(AT[0]+st).trim()}"><path class="fr" d="M816,${y}H826"/><rect class="wbox" data-dim="router W_r · matrix(${EXPERTS} × ${WIDTH}), the same for every token" x="826" y="${y-10}" width="28" height="20" rx="4"/><text class="wbox-t" x="840" y="${y+4}" text-anchor="middle">W<tspan baseline-shift="sub" font-size="7">r</tspan></text><path class="emb-arrow" d="M857,${y}H872m-5,-4l5,4l-5,4"/></g>`;
  r.chosen.forEach((e,c)=>{
   const yy=y+(c?13:-13),g=r.gates[e];
   s+=`<g class="${late(AT[3]+st).trim()}"><text class="ex-name" x="1326" y="${yy+3.5}" text-anchor="end">E${e+1}</text><path class="band" style="fill:#d3dcee" d="M1330,${yy-6} L1346,${yy-11} V${yy+11} L1330,${yy+6} Z"/><rect data-dim="expert ${e+1} · GELU(W₁ · x) · vector(${FF})" x="1346" y="${yy-11}" width="12" height="22" rx="2.5" fill="#6681b5" opacity=".8"/><path class="band" style="fill:#d3dcee" d="M1358,${yy-11} L1372,${yy-6} V${yy+6} L1358,${yy+11} Z"/><rect data-dim="output of expert ${e+1} · vector(${WIDTH})" x="1372" y="${yy-6}" width="8" height="12" rx="2" fill="#a9c0f3"/></g>`;
   s+=`<g class="${late(AT[4]+st).trim()}"><text class="gate" x="1386" y="${yy+3.5}">× ${g.toFixed(2)}</text><path class="fx" d="M1424,${yy}C1434,${yy} 1432,${y} 1439,${y}"/></g>`;
  });
  s+=`<g class="${late(AT[4]+st).trim()}"><circle class="sg" data-dim="${r.chosen.map(e=>`${r.gates[e].toFixed(2)} × expert ${e+1}`).join(' + ')}" cx="1446" cy="${y}" r="7"/><text class="sg-t" x="1446" y="${y+3.6}" text-anchor="middle">Σ</text><path class="fx" d="M1453,${y}H${XO}"/><rect data-dim="vector(${WIDTH})" x="${XO}" y="${y-15}" width="10" height="30" rx="2" fill="#a9c0f3"/></g>`;
 });
 s+=`<text class="mlp-size" x="811" y="${dy}" text-anchor="middle">${WIDTH}</text><text class="mlp-size${late(AT[3])}" x="1352" y="${dy}" text-anchor="middle">${FF}</text><text class="mlp-size${late(AT[3])}" x="1376" y="${dy}" text-anchor="middle">${WIDTH}</text><text class="mlp-size${late(AT[4])}" x="${XO+5}" y="${dy}" text-anchor="middle">${WIDTH}</text>`;
 return s;
}
function flow(){
 const {tokens}=data(),n=tokens.length,{H,top,ch,yc,ty}=geo(n),r=raw();
 const fx=218,fy=48,fw=712+MS,fh=H-fy-8;
 const qkv={q:'Query',k:'Key',v:'Value'};
 // one part can be open at a time: 'attention' or 'moe'
 const X=state.expand,E=X==='attention',EM=X==='moe',mats=E?3:1,matW=n*TP,mgap=58,mx=m=>548+m*(matW+mgap),mx0=mx(0),cx=(j,m=0)=>mx(m)+j*TP,mtop=ty('q',0)-TP/2-3;
 const dA=(mats-1)*(matW+mgap),dM=EM?DM:0,rs=EM?RS:0,outX=716+dA,cardW=440+dA,anim=state.anim,prev=state.last||{dA:0,dM:0,rs:0},was=prev.dA;
 state.anim=null;state.last={dA,dM,rs};
 // everything that is not the open part fades into the background
 const dimSide=X?' xdim':'',dimBlock=EM?' xdim':'',dimMoe=X&&!EM?' xdim':'';
 const fromM=anim?prev.dM-dM:0,from=anim?was-dA:0,sx=w=>anim?((w-dA+was)/w).toFixed(4):1,openM=anim==='open'&&EM;
 const late=(on,at)=>on?` late" style="--at:${at}ms`:'';
 let s=`<title id="tx-flow-title">A sentence flowing through embedding, transformer block ${state.block+1} with its mixture-of-experts layer, and the output probabilities</title>`;
 // the opened MoE layer is wide: the whole picture moves left to keep it in view
 s+=`<g transform="translate(${-rs})"><g class="${anim?'slide':''}" style="--from:${anim?rs-prev.rs:0}px">`;
 s+=`<g class="stage${dimSide}" style="--d:0"><text class="col-title" x="20" y="30">Embedding</text>`;
 tokens.forEach((t,i)=>{s+=`<text data-t="${i}" class="tok" x="120" y="${yc(i)+4}" text-anchor="end">${esc(t)}</text><rect data-t="${i}" data-dim="vector(${WIDTH})" x="130" y="${yc(i)-15}" width="12" height="30" rx="2" fill="#cdd6e2"/>`;});
 s+=`</g>`;
 s+=`<g class="stage" style="--d:1"><rect class="frame${anim?' widen':''}" style="--sx:${anim?((fw+prev.dA+prev.dM)/(fw+dA+dM)).toFixed(4):1}" x="${fx}" y="${fy}" width="${fw+dA+dM}" height="${fh}" rx="10"/>${chip('block',fx+18,fy,blockLabel(),176)}</g>`;
 s+=`<g class="stage block-part${dimSide}" style="--d:1">`;
 tokens.forEach((t,i)=>{
  const y=yc(i);
  [['q',-12],['k',0],['v',12]].forEach(([g,dy])=>{
   s+=`<g data-t="${i}"><path class="fg" d="${curve(142,y,232,y+dy)}" stroke-width="1.2"/><rect class="${g}" data-dim="${qkv[g]} weights · matrix(${WIDTH} × ${WIDTH})" x="232" y="${y+dy-4.5}" width="9" height="9" rx="1.5" opacity=".75"/><text class="mini" x="246" y="${y+dy+3}">${g.toUpperCase()}</text></g>`;
  });
 });
 s+=`</g>`;
 // ----- head card: identical to the text page
 s+=`<g class="stage head-part${dimBlock}" style="--d:2" data-zone="attention">`;
 for(let d=3;d>=1;d--)s+=`<rect class="card back${anim?' widen':''}" style="--sx:${sx(cardW)}" x="${330+d*5}" y="${top+d*5}" width="${cardW}" height="${ch}" rx="6"/>`;
 s+=`<rect class="card front${anim?' widen':''}" style="--sx:${sx(cardW)}" x="330" y="${top}" width="${cardW}" height="${ch}" rx="6"/>`;
 for(const g of ['k','q','v']){
  s+=`<text class="lbl-${g}" x="452" y="${ty(g,0)-12}">${qkv[g]}</text>`;
  tokens.forEach((t,i)=>{
   const y=ty(g,i),dy=g==='q'?-12:g==='k'?0:12;
   s+=`<g data-t="${i}"><path class="f${g}" d="${curve(254,yc(i)+dy,380,y)}" stroke-width="1" opacity=".55"/><text class="mini" x="446" y="${y+3}" text-anchor="end">${esc(t)}</text><rect class="${g}" data-dim="${qkv[g]}, this head · vector(${HEAD_DIM})" x="452" y="${y-1.5}" width="12" height="3" rx="1"/><rect class="hit" data-dim="${qkv[g]}, this head · vector(${HEAD_DIM})" x="450" y="${y-5}" width="16" height="10"/></g>`;
  });
 }
 s+=`</g><g class="stage head-part${dimBlock}" style="--d:3" data-zone="attention">`;
 const opening=anim==='open'&&E,QK=1200,t0=300,T1=t0+QK/n+QK+500,T2=T1+500+1000+150,T3=T2+450+1000;
 const draw=k=>opening?` pathLength="1" class="draw" style="--at:${Math.round(t0+k*QK/n)}ms"`:'';
 const look=(m,i,j)=>{const v=stageShade(m,i,j,r);return v===null?['#fff','1','#e2e8f0']:['#7f58a3',dotOpacity(v),'transparent'];};
 tokens.forEach((t,j)=>{s+=`<g data-t="${j}" class="fk" opacity=".5"><path${draw(j)} d="M464,${ty('k',j)} H${cx(j)} V${mtop}" stroke-width="1"/></g>`;});
 tokens.forEach((t,i)=>{s+=`<g data-t="${i}" class="fq" opacity=".6"><path${draw(i)} d="M464,${ty('q',i)} H${mx0-TP/2-2}" stroke-width="1"/></g>`;});
 const stages=E?[0,1,2]:[2];
 stages.forEach((m,k)=>{
  s+=`<g class="${opening&&k?'grow':''}" style="--from:${-(matW+mgap)}px;--at:${[0,T1,T2][k]}ms">`;
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
   const v=stageShade(m,i,j,r);let f='';
   if(opening&&k===0)f=` pop" style="--at:${Math.round(t0+QK/n+(i*n+j)*QK/(n*n))}ms`;
   else if(opening){const a=look(m-1,i,j),b=look(m,i,j);f=` turn" style="--at:${[0,T1+500,T2+450][k]}ms;--f0:${a[0]};--o0:${a[1]};--s0:${a[2]};--f1:${b[0]};--o1:${b[1]};--s1:${b[2]}`;}
   s+=`<circle class="cell${v===null?' masked':''}${f}" data-t="${i}" data-q="${i}" data-k="${j}" data-m="${m}" cx="${cx(j,k)}" cy="${ty('q',i)}" r="5"${v===null?'':` fill="#7f58a3" fill-opacity="${dotOpacity(v)}"`}/>`;
  }
  if(E){
   const mid=mx(k)+matW/2-TP/2,by=ty('q',n-1)+TP/2,[lo,hi]=stageRange(m,r);
   s+=`<g class="stage-name" data-formula="${m}" tabindex="0" role="button" aria-label="${STAGE_NAMES[m]}: show the formula"><rect x="${mid-52}" y="${by+8}" width="104" height="19" rx="9"/><text x="${mid}" y="${by+21}" text-anchor="middle">${STAGE_NAMES[m]}</text></g>`;
   s+=`<text class="leg leg-lo" data-m="${m}" x="${mid-20}" y="${by+41}" text-anchor="end">${lo}</text><rect class="leg-bar" x="${mid-16}" y="${by+35}" width="32" height="4" rx="2"/><text class="leg leg-hi" data-m="${m}" x="${mid+20}" y="${by+41}">${hi}</text>`;
   if(k<2)s+=`<text class="stage-arrow" x="${mx(k)+matW-TP/2+mgap/2}" y="${ty('q',(n-1)/2)+4}" text-anchor="middle">→</text>`;
  }
  s+=`</g>`;
 });
 if(!E)s+=`<g class="attn-label" data-expand="attention" role="button" tabindex="0" aria-label="Open attention into its three stages"><text class="col-sub" x="${mx0-6}" y="${ty('q',n-1)+TP/2+16}">ATTENTION · UNCHANGED</text></g>`;
 const mr=cx(n-1,mats-1)+8;
 s+=`<g class="${anim?'slide':''}" style="--from:${from}px"><path class="band${opening?' late':''}" style="--at:${T3}ms;fill:#e6dcf0" d="M${mr},${ty('q',0)-6} L${outX},${ty('q',0)-2} V${ty('q',n-1)+2} L${mr},${ty('q',n-1)+6} Z"/>`;
 s+=`<g class="stage-name out-name" data-formula="3" tabindex="0" role="button" aria-label="Out: show the formula"><rect x="${outX-13}" y="${ty('q',0)-27}" width="40" height="19" rx="9"/><text x="${outX+7}" y="${ty('q',0)-14}" text-anchor="middle">Out</text></g>`;
 tokens.forEach((t,i)=>{
  s+=`<g data-t="${i}">${E?`<path class="fv${opening?' late':''}" style="--at:${T3}ms" d="M464,${ty('v',i)} H${outX-74} C${outX-30},${ty('v',i)} ${outX-44},${ty('q',i)} ${outX},${ty('q',i)}" stroke-width="1" opacity=".45"/>`:`<path class="fv" d="${curve(464,ty('v',i),outX,ty('q',i))}" stroke-width="1" opacity=".4"/>`}<rect class="o" x="${outX}" y="${ty('q',i)-1.5}" width="14" height="3" rx="1"/><rect class="hit" data-dim="Out, this head · vector(${HEAD_DIM})" x="${outX-2}" y="${ty('q',i)-5}" width="18" height="10"/></g>`;
 });
 s+=`</g></g>`;
 s+=`<g class="stage${dimBlock}" style="--d:2">${chip('head',346,top,headLabel(),282)}</g>`;
 s+=`<g transform="translate(${dA})" style="--oxL:${fx-dA}px;--oxR:${fx+fw+dM}px"><g class="${anim?'slide':''}" style="--from:${from}px">`;
 // ----- the MoE layer, where the text page has its MLP
 const pill=(k,x,w,at)=>`<g class="stage-name${late(openM,at)}" data-formula="${k}" tabindex="0" role="button" aria-label="${['Router scores',`Top-${TOP_K}`,'Softmax over the chosen','Experts run','Weighted sum'][k-5]}: show the formula"><rect x="${x-w/2}" y="${fy+40}" width="${w}" height="19" rx="9"/><text x="${x}" y="${fy+53}" text-anchor="middle">${[...ROUTE_NAMES,'Experts run','Weighted sum'][k-5]}</text></g>`;
 s+=`<g class="${dimMoe.trim()}" data-zone="moe"><g class="stage" style="--d:4"><g class="stage-name mlp-name moe-name" data-formula="4" tabindex="0" role="button" aria-label="MoE layer: show the formula"><rect x="${806-9}" y="${fy+21}" width="74" height="19" rx="9"/><text x="${806+28}" y="${fy+34}" text-anchor="middle">MoE layer</text></g>${EM?[0,1,2].map(k=>pill(5+k,mE(k)+MW/2-TP/2,[92,54,64][k],[300,1100,1900][k])).join('')+pill(8,1352,80,2700)+pill(9,1440,88,3500):''}</g>`;
 s+=`<g class="stage block-part moe-part" style="--d:4">${moeRows(openM)}</g></g>`;
 // the blocks that follow and the output step move aside when the MoE layer is open
 s+=`<g transform="translate(${dM})"><g class="${anim?'slide':''}${dimSide}" style="--from:${fromM}px"><g class="stage block-part" style="--d:4"><text class="col-sub more-a" x="${944+MS}" y="46"></text><text class="col-sub more-b" x="${944+MS}" y="58"></text>`;
 tokens.forEach((t,i)=>{
  const y=yc(i);
  s+=`<g data-t="${i}"><path class="fg" d="M${912+MS},${y} H${944+MS}" stroke-width="1.2"/>`;
  for(let b=0;b<3;b++)s+=`<rect class="more-bar" data-b="${b}" data-dim="vector(${WIDTH})" x="${944+MS+b*12}" y="${y-15}" width="7" height="30" rx="2" fill="#d3e0fa" opacity="${0.9-b*0.25}"/>`;
  s+=`</g>`;
 });
 s+=`</g>`;
 // ----- probabilities for the last token: the same step as on the text page, with a fixed illustrative list
 const list=ex().next,py=k=>88+k*30,last=n-1,bx=1132+MS;
 s+=`<g class="stage" style="--d:6"><text class="col-title" x="${1052+MS}" y="30">Probabilities</text><text class="col-sub" x="${1052+MS}" y="46">UNCHANGED · ILLUSTRATIVE</text></g>`;
 s+=`<g class="stage prob-part" style="--d:6">`;
 list.forEach((w,k)=>{s+=`<path class="fo pc" data-t="${last}" data-rank="${k}" data-y1="${yc(last)}" data-y2="${py(k)-4}" d="${wordLine(yc(last),py(k)-4,1040+MS)}"/>`;});
 s+=`<g class="stage-name fan-hit" data-formula="10" tabindex="0" role="button" aria-label="Output projection: show the formula"><path d="${fanArea(yc(last),py(0)-15,py(list.length-1)+7)}"/></g>`;
 list.forEach(([word,p],k)=>{
  const y=py(k),bw=Math.max(2,p*56);
  s+=`<g class="prob kept" data-rank="${k}"><text class="pt" x="${1124+MS}" y="${y}" text-anchor="end">${esc(word.trim())}</text><rect class="pb" x="${bx}" y="${y-6}" width="${bw.toFixed(1)}" height="4" rx="2"/><text class="pp" x="${(bx+bw+6).toFixed(1)}" y="${y}">${p<0.01?'<1%':Math.round(p*100)+'%'}</text></g>`;
 });
 s+=`</g></g></g></g></g></g></g>`;
 const svg=$('#tx-flow');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=s;svg.classList.toggle('expanded',!!X);svg.dataset.open=X||'';
 svg.style.setProperty('--oxL',`${fx}px`);svg.style.setProperty('--oxR',`${fx+fw+dA+dM}px`);svg.classList.remove('tracing');
 $('#tx-tip').hidden=true;$('#tx-formula').hidden=true;
 // each line stops just before its word
 svg.querySelectorAll('.pc').forEach(c=>{let x2=1040+MS;try{x2=svg.querySelector(`.prob[data-rank="${c.dataset.rank}"] .pt`).getBBox().x-7;}catch{}c.setAttribute('d',wordLine(+c.dataset.y1,+c.dataset.y2,x2));});
 paintMore(svg);defaultResult();
 $('#tx-sentence').innerHTML=`${esc(ex().text)} · <b>${TOP_K} of ${EXPERTS} experts</b> run per token`;
}
function defaultResult(){
 const B=block(),busy=B.loads.filter(x=>x).length,top=B.loads.indexOf(Math.max(...B.loads));
 $('#tx-result').textContent=`Block ${state.block+1} of ${BLOCKS}: each of the ${data().tokens.length} tokens is sent to ${TOP_K} of the ${EXPERTS} experts. ${busy} experts get work here${busy<EXPERTS?`, ${EXPERTS-busy} stay idle`:''}; expert ${top+1} is the busiest with ${B.loads[top]} tokens. Hover a token to see where it goes.`;
}

/* ---------- 02 who gets the work: tokens × experts for the block shown ---------- */
function who(){
 const {tokens}=data(),B=block();
 let h=`<div class="mx-who-grid" style="grid-template-columns:minmax(52px,auto) repeat(${EXPERTS},minmax(0,1fr))"><span></span>${Array.from({length:EXPERTS},(_,e)=>`<span class="cl">E${e+1}</span>`).join('')}`;
 tokens.forEach((t,i)=>{
  const r=B.routes[i];
  h+=`<span class="rl" data-row="${i}">${esc(t)}</span>`;
  for(let e=0;e<EXPERTS;e++){const on=r.chosen.includes(e);h+=`<span class="c${on?' on':''}" data-row="${i}" data-e="${e}"${on?` style="--a:${(0.18+0.6*r.gates[e]).toFixed(2)}"`:''}>${on?pct(r.gates[e]):'·'}</span>`;}
 });
 h+=`<span class="rl load">tokens</span>${B.loads.map(v=>`<span class="load${v?'':' zero'}">${v}</span>`).join('')}</div>`;
 $('#dispatch-table').innerHTML=h;
 $('#expert-loads').innerHTML=B.loads.map((v,e)=>`<div><span>E${e+1}</span><i><b style="width:${(v/tokens.length*100).toFixed(0)}%"></b></i><strong>${v}</strong></div>`).join('');
 whoResult();
}
function whoResult(i=null){
 const {tokens}=data(),B=block();
 $$('#dispatch-table [data-row]').forEach(el=>el.classList.toggle('row',i!==null&&+el.dataset.row===i));
 if(i!==null){const r=B.routes[i];$('#who-result').textContent=`“${tokens[i]}” in block ${state.block+1}: router scores ${r.chosen.map(e=>`${r.scores[e].toFixed(2)} for expert ${e+1}`).join(' and ')} are the two highest, so those two run and are mixed ${r.chosen.map(e=>pct(r.gates[e])).join(' / ')}. The other ${EXPERTS-TOP_K} experts are skipped.`;return;}
 const idle=B.loads.map((v,e)=>v?null:`E${e+1}`).filter(Boolean),top=B.loads.indexOf(Math.max(...B.loads));
 $('#who-result').textContent=`Block ${state.block+1}: ${tokens.length} tokens × ${TOP_K} experts = ${tokens.length*TOP_K} expert runs, instead of ${tokens.length*EXPERTS} if every expert ran. Busiest: E${top+1} with ${B.loads[top]} tokens. ${idle.length?`Idle in this block: ${idle.join(', ')}.`:'Every expert gets at least one token.'}`;
}

/* ---------- 04 stored vs used ---------- */
function cost(){
 const c=layerCost(),fmt=v=>v.toLocaleString('en-US'),bn=v=>(v/1e9).toFixed(1)+' billion';
 const pool=on=>`<span class="mx-pool"><i class="r" title="router"></i>${Array.from({length:EXPERTS},(_,e)=>`<i class="${e<on?'e':'e off'}"></i>`).join('')}</span>`;
 const share=MIXTRAL.active/MIXTRAL.stored;
 $('#cost-comparison').innerHTML=`<div class="mx-cost-col"><p class="string-label">ONE MoE LAYER OF THIS PAGE'S MODEL</p>
<div class="mx-cost-row"><span>Stored</span>${pool(EXPERTS)}<strong>${fmt(c.stored)}</strong></div>
<div class="mx-cost-row"><span>Used per token</span>${pool(TOP_K)}<strong>${fmt(c.active)}</strong></div>
<p class="vec-note">Router ${fmt(c.router)} weights (${EXPERTS} × ${WIDTH}) + ${EXPERTS} experts of ${fmt(c.expert)} each (${WIDTH} × ${FF}, twice). A token uses the router and ${TOP_K} experts: ${fmt(c.router)} + ${TOP_K} × ${fmt(c.expert)} = ${fmt(c.active)}. One ordinary MLP of the same shape would store and use ${fmt(c.dense)}.</p></div>
<div class="mx-cost-col"><p class="string-label">A REAL MODEL · ${MIXTRAL.name.toUpperCase()}, ALL ${MIXTRAL.blocks} BLOCKS</p>
<div class="mx-cost-row"><span>Stored</span><span class="mx-share"><i style="width:100%"></i></span><strong>${bn(MIXTRAL.stored)}</strong></div>
<div class="mx-cost-row"><span>Used per token</span><span class="mx-share"><i style="width:${(share*100).toFixed(1)}%"></i></span><strong>${bn(MIXTRAL.active)}</strong></div>
<p class="vec-note">Totals published by Mistral AI for the whole model: ${MIXTRAL.experts} experts in each of ${MIXTRAL.blocks} blocks, ${MIXTRAL.topK} of them run per token. About ${(share*100).toFixed(0)}% of the weights do the work for any one token; the rest wait for tokens that suit them better.</p></div>`;
}

/* Changing block or head redraws only that part, as on the text page. A new block also brings new routes. */
function swap(kind,dir){
 const svg=$('#tx-flow'),calm=matchMedia('(prefers-reduced-motion: reduce)').matches,r=raw();
 const sel=kind==='block'?'.head-part,.block-part':'.head-part',parts=[...svg.querySelectorAll(sel)].filter(g=>!g.classList.contains('ghost'));
 svg.querySelectorAll('.ghost').forEach(g=>g.remove());
 const ghosts=calm?[]:parts.map(g=>{const c=g.cloneNode(true);c.classList.remove('swap','stage');c.classList.add('ghost');c.setAttribute('aria-hidden','true');g.before(c);return c;});
 parts.forEach(g=>g.querySelectorAll('.cell:not(.masked)').forEach(c=>c.setAttribute('fill-opacity',dotOpacity(stageShade(+c.dataset.m,+c.dataset.q,+c.dataset.k,r)))));
 parts.forEach(g=>g.querySelectorAll('.leg[data-m]').forEach(t=>{t.textContent=stageRange(+t.dataset.m,r)[t.classList.contains('leg-lo')?0:1];}));
 if(kind==='block')parts.filter(g=>g.classList.contains('moe-part')).forEach(g=>{g.innerHTML=moeRows();});
 svg.querySelector('.block-label').textContent=blockLabel();
 parts.forEach(g=>paintMore(g));
 svg.querySelector('.head-label').textContent=headLabel();
 svg.querySelector('title').textContent=`A sentence flowing through embedding, transformer block ${state.block+1} with its mixture-of-experts layer, and the output probabilities`;
 svg.classList.remove('tracing');
 syncSteppers();who();defaultResult();
 const moving=[...parts,svg.querySelector(`.${kind}-label`),svg.querySelector(kind==='block'?'.frame':'.card.front')];
 svg.dataset.swap=kind;svg.dataset.dir=dir>0?'next':'prev';svg.style.setProperty('--dir',dir);
 moving.forEach(g=>g.classList.remove('swap'));void svg.getBoundingClientRect();moving.forEach(g=>g.classList.add('swap'));
 setTimeout(()=>ghosts.forEach(g=>g.remove()),900);
}
function setExpand(which){
 if(state.expand===which)return;
 state.expand=which;state.anim=matchMedia('(prefers-reduced-motion: reduce)').matches?null:(which?'open':'close');
 $('#tx-lab').classList.remove('fresh');$('#tx-flow').querySelectorAll('.ghost').forEach(g=>g.remove());
 flow();
}

/* ---------- wiring ---------- */
function render({fresh=false}={}){
 syncSteppers();flow();who();
 if(fresh){const lab=$('#tx-lab');lab.classList.remove('fresh');void lab.offsetWidth;lab.classList.add('fresh');}
}
function init(){
 $('#tx-example').innerHTML=EXAMPLES.map((e,i)=>`<option value="${i}">${esc(e.text)}</option>`).join('');
 $('#tx-example').addEventListener('change',e=>{state.ex=+e.target.value;state.expand=null;state.last=null;render({fresh:true});});
 cost();
 document.addEventListener('click',e=>{
  const step=e.target.closest('[data-step]');
  if(step){const kind=step.closest('[data-stepper]').dataset.stepper,max=kind==='block'?BLOCKS:HEADS;const dir=+step.dataset.step;state[kind]=(state[kind]+dir+max)%max;swap(kind,dir);return;}
  // a click on the attention head or the MoE layer opens that part; a click anywhere else in the diagram closes it
  if(e.target.closest('#tx-flow')){
   const zoneEl=e.target.closest('[data-zone]'),zone=zoneEl&&!e.target.closest('.ghost')?zoneEl.dataset.zone:null;
   if(!state.expand&&zone){setExpand(zone);return;}
   if(state.expand&&zone!==state.expand&&!e.target.closest('.chip')){setExpand(null);return;}
  }
 });
 document.addEventListener('keydown',e=>{
  if(e.key==='Escape'&&state.expand){setExpand(null);return;}
  const b=(e.key==='Enter'||e.key===' ')&&e.target.closest?.('#tx-flow .step,#tx-flow [data-expand],#tx-flow .moe-name');
  if(b){e.preventDefault();b.dispatchEvent(new MouseEvent('click',{bubbles:true}));}
 });
 const svg=$('#tx-flow');
 // tracing one token through the diagram: its row lights up, and so do the two experts it is sent to
 function trace(t){
  const {tokens}=data(),r=block().routes[+t];
  svg.classList.add('tracing');svg.querySelectorAll('[data-t]').forEach(x=>x.classList.toggle('on',x.dataset.t===t));
  svg.querySelectorAll('.ex-h').forEach(x=>x.classList.toggle('on',r.chosen.includes(+x.dataset.e)));
  $('#tx-result').textContent=`“${tokens[+t]}” is sent to experts ${names(r.chosen)} in block ${state.block+1}, mixed ${r.chosen.map(e=>pct(r.gates[e])).join(' / ')}. The other ${EXPERTS-TOP_K} experts do not run for this token.${+t===tokens.length-1?' As the last token, its final vector is the one used to predict what comes next.':''}`;
 }
 svg.addEventListener('mouseover',e=>{
  if(state.expand)return;
  const el=e.target.closest('[data-t]');if(!el)return;
  trace(el.dataset.t);
  const cell=e.target.closest('.cell');
  if(cell){
   const i=+cell.dataset.q,j=+cell.dataset.k,{tokens}=data(),w=attentionStages(raw()).weights[i][j];
   $('#tx-result').textContent=j>i?`“${tokens[i]}” cannot look ahead at “${tokens[j]}”: masked, as in any text transformer.`:`“${tokens[i]}” gives ${pct(w)} of its attention to “${tokens[j]}” in block ${state.block+1}, head ${state.head+1} (an untrained model: the pattern has no meaning).`;
  }
 });
 const tip=$('#tx-tip'),formula=$('#tx-formula');
 const near=(el,target,below=true)=>{
  const r=target.getBoundingClientRect();el.hidden=false;
  const w=el.offsetWidth,h=el.offsetHeight,fits=r.bottom+10+h<innerHeight-8,under=below&&(fits||r.top-10-h<8);
  el.style.left=Math.max(8,Math.min(innerWidth-w-8,r.left+r.width/2-w/2))+'px';el.style.top=(under?r.bottom+10:r.top-h-10)+'px';
 };
 addEventListener('scroll',()=>{tip.hidden=true;formula.hidden=true;},{passive:true});
 const showFormula=name=>{formula.innerHTML=FORMULAS[+name.dataset.formula];near(formula,name);};
 svg.addEventListener('mouseover',e=>{
  const cell=e.target.closest('.cell'),rcell=e.target.closest('.rcell'),name=e.target.closest('.stage-name');
  svg.querySelectorAll('.hot').forEach(c=>c.classList.remove('hot'));
  if(name)showFormula(name);else formula.hidden=true;
  if(cell&&state.expand==='attention'){
   const i=+cell.dataset.q,j=+cell.dataset.k,m=+cell.dataset.m,{tokens}=data(),r=raw(),{scaled,weights}=attentionStages(r);
   const q=esc(tokens[i]),k=esc(tokens[j]);
   tip.innerHTML=m===0?`<b>${q} · ${k}</b>Query · Key = ${r[i][j].toFixed(2)}`:m===1?`<b>${q} → ${k}</b>${j>i?'masked: a later token, −∞':`${r[i][j].toFixed(2)} ÷ √${HEAD_DIM} = ${scaled[i][j].toFixed(2)}`}`:`<b>${q} → ${k}</b>${j>i?'masked: weight 0':`attention weight ${pct(weights[i][j])}`}`;
   cell.classList.add('hot');near(tip,cell,false);
  }else if(rcell){
   const i=+rcell.dataset.q,e=+rcell.dataset.e,m=+rcell.dataset.m,r=block().routes[i],on=r.chosen.includes(e),t=esc(data().tokens[i]);
   const rank=[...r.scores].sort((a,b)=>b-a).indexOf(r.scores[e])+1;
   tip.innerHTML=m===0?`<b>${t} · expert ${e+1}</b>router score ${r.scores[e].toFixed(2)}`:m===1?`<b>${t} · expert ${e+1}</b>${on?`kept: score ${r.scores[e].toFixed(2)}, no. ${rank} of ${EXPERTS}`:`dropped: no. ${rank} of ${EXPERTS} → −∞`}`:`<b>${t} · expert ${e+1}</b>${on?`weight ${r.gates[e].toFixed(2)} (${pct(r.gates[e])})`:'weight 0: this expert does not run'}`;
   rcell.classList.add('hot');near(tip,rcell,false);
  }else{
   const sized=e.target.closest('[data-dim]');
   if(sized){tip.textContent=sized.dataset.dim;near(tip,sized,false);}else tip.hidden=true;
  }
 });
 svg.addEventListener('focusin',e=>{const name=e.target.closest('.stage-name');if(name)showFormula(name);});
 svg.addEventListener('focusout',()=>{formula.hidden=true;});
 svg.addEventListener('mouseleave',()=>{svg.classList.remove('tracing');svg.querySelectorAll('.on').forEach(x=>x.classList.remove('on'));tip.hidden=true;formula.hidden=true;svg.querySelectorAll('.hot').forEach(c=>c.classList.remove('hot'));defaultResult();});
 const table=$('#dispatch-table');
 table.addEventListener('mouseover',e=>{const c=e.target.closest('[data-row]');if(c)whoResult(+c.dataset.row);});
 table.addEventListener('mouseleave',()=>whoResult());
 render({fresh:true});
}
init();
