// Vision transformers (Pixtral's encoder): explorer and section lab.
// Built to match the text page (src/text-app.mjs) step for step, so the differences stand out:
// patches instead of tokens, no attention mask, 2D RoPE on Query and Key, a gated MLP, and a projector
// at the end instead of next-token probabilities. Dimensions are Pixtral's; the attention values are
// illustrative (src/vision-engine.mjs). Nothing here runs a model.
import {IMAGES} from './vision-data.mjs';
import {BLOCKS,HEADS,HEAD_DIM,COLS,ROWS,SHOWN,PATCHES,windowStart,patchPlace,patchName,attentionRaw,attentionStages,illustrativeVector} from './vision-engine.mjs';

const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const pct=p=>p>=0.9995?'100%':p<=0?'0%':(p*100).toFixed(p<0.1?2:1)+'%';
const state={img:0,sel:IMAGES[0].sel,block:0,head:0,expand:null,anim:null,last:null};
const image=()=>IMAGES[state.img];
const shown=()=>{const s=windowStart(state.sel);return Array.from({length:SHOWN},(_,i)=>s+i);};
const raw=()=>attentionRaw(image(),state.block,state.head,shown());
const where=p=>{const {row,col}=patchPlace(p);return `row ${row+1}, column ${col+1}`;};

function syncSteppers(){
 $$('div[data-stepper]').forEach(el=>{
  const kind=el.dataset.stepper,label=kind==='block'?`Block ${state.block+1} of ${BLOCKS}`:`Head ${state.head+1} of ${HEADS}`;
  el.innerHTML=`<button type="button" data-step="-1" aria-label="Previous ${kind}">‹</button><b>${label}</b><button type="button" data-step="1" aria-label="Next ${kind}">›</button>`;
 });
}

/* ---------- explorer flow diagram ---------- */
// The block, head, MLP and "more blocks" columns use the text page's coordinates, moved right by DX to make room for the image.
const DX=176,W=1290,Y0=106,PITCH=54,TP=13,CELL=20,IX=14;
const blockLabel=()=>`TRANSFORMER BLOCK ${state.block+1} / ${BLOCKS}`,headLabel=()=>`MULTI-HEAD SELF-ATTENTION, HEAD ${state.head+1} / ${HEADS}`;
const chip=(kind,x,y,label,w)=>{
 const btn=(d,bx)=>`<g class="step" data-step="${d}" role="button" tabindex="0" aria-label="${d>0?'Next':'Previous'} ${kind}"><rect x="${bx}" y="${y-9}" width="18" height="18" rx="4"/><text x="${bx+9}" y="${y+4.5}" text-anchor="middle">${d>0?'›':'‹'}</text></g>`;
 return `<g class="chip" data-stepper="${kind}"><rect class="chip-bg" x="${x-7}" y="${y-11}" width="${w+64}" height="22"/>${btn(-1,x)}<text class="chip-label ${kind}-label" x="${x+25+w/2}" y="${y+3.5}" text-anchor="middle">${label}</text>${btn(1,x+32+w)}</g>`;
};
const curve=(x1,y1,x2,y2)=>{const m=(x1+x2)/2;return `M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}`;};
// Attention as its three stages. There is no mask here, so no dot is ever removed.
const STAGE_NAMES=['Dot product','Scaling','Softmax'];
function stageShade(m,i,j,r=raw()){
 if(m===2)return attentionStages(r).weights[i][j];
 const flat=r.flat(),lo=Math.min(...flat),hi=Math.max(...flat);
 return (r[i][j]-lo)/((hi-lo)||1);
}
const stageRange=(m,r=raw())=>{const flat=r.flat(),lo=Math.min(...flat),hi=Math.max(...flat),d=m===1?8:1;return m===2?['0','1']:[(lo/d).toFixed(1),(hi/d).toFixed(1)];};
const dotOpacity=v=>(0.07+0.93*v).toFixed(3);
const FORMULAS=[
 `<h4>Dot product</h4><div class="fx">score<sub>i,j</sub> = Q<sub>i</sub> · K<sub>j</sub> = <span class="sum">Σ<sub>d=1</sub><sup>64</sup></span> Q<sub>i,d</sub> × K<sub>j,d</sub></div><p>The Query of patch i (the row) is multiplied with the Key of patch j (the column), number by number, and the 64 products are added up. Both were first rotated by 2D RoPE, so the score also depends on how far apart the two patches are.</p>`,
 `<h4>Scaling · no mask</h4><div class="fx">scaled<sub>i,j</sub> = score<sub>i,j</sub> ÷ √64 &nbsp;&nbsp;for every i, j</div><p>Dividing by √64 = 8 keeps the scores in a range softmax handles well. <b>Unlike the text model, nothing is masked:</b> an image has no “later”, so every patch may look at every other patch.</p>`,
 `<h4>Softmax</h4><div class="fx">weight<sub>i,j</sub> = <span class="frac"><span>exp(scaled<sub>i,j</sub>)</span><span><span class="sum">Σ<sub>k</sub></span> exp(scaled<sub>i,k</sub>)</span></span></div><p>Each row is turned into weights that add up to 1. In the real model a row runs over all 48 patches; the six shown here are normalised among themselves.</p>`,
 `<h4>Out</h4><div class="fx">out<sub>i</sub> = <span class="sum">Σ<sub>j</sub></span> weight<sub>i,j</sub> × V<sub>j</sub></div><p>Each patch's output is a blend of the Value vectors, mixed by its row of attention weights. The Values are not rotated by RoPE.</p>`,
 `<h4>Gated MLP</h4><div class="fx">MLP(x) = W<sub>2</sub> · ( SiLU(W<sub>1</sub> · x) ⊙ (W<sub>3</sub> · x) )</div><p>The text model's MLP is W<sub>2</sub> · GELU(W<sub>1</sub> · x). Here the widened vector is multiplied, number by number, with a second widened copy that acts as a gate (this is called SwiGLU). 1,024 → 4,096 → 1,024, the same weights for every patch.</p>`,
 `<h4>Projector</h4><div class="fx">token<sub>i</sub> = W<sub>b</sub> · GELU(W<sub>a</sub> · h<sub>i</sub>)<br>W<sub>a</sub> : matrix(5,120 × 1,024)</div><p>There is no vocabulary and no softmax at the end of a vision encoder. A small two-layer network resizes <b>every</b> patch's final vector from 1,024 to 5,120 numbers, the width of the language model, which then reads them like word embeddings.</p>`,
 `<h4>Patchify</h4><div class="fx">128 × 96 image → 8 × 6 = 48 patches of 16 × 16</div><p>The image is cut into a grid of equal squares, read left to right, top to bottom. Each patch plays the part a token plays in text. A bigger image simply gives more patches.</p>`,
 `<h4>Flatten</h4><div class="fx">16 × 16 pixels × 3 colours = 768 numbers</div><p>The red, green and blue values of a patch's 256 pixels are laid out as one list of 768 numbers. There is no vocabulary to look up: the patch itself is the input.</p>`,
 `<h4>Linear projection</h4><div class="fx">x<sub>i</sub> = W<sub>patch</sub> · pixels<sub>i</sub><br>W<sub>patch</sub> : matrix(1,024 × 768)</div><p>One learned matrix turns the 768 pixel values into the 1,024 numbers the encoder works with. The text model looks a row up in a table instead. No position is added here: Pixtral supplies position inside every attention layer, with 2D RoPE.</p>`,
 `<h4>2D RoPE</h4><div class="fx">Q<sub>i</sub> ← rotate(Q<sub>i</sub>, row<sub>i</sub>, col<sub>i</sub>)<br>K<sub>j</sub> ← rotate(K<sub>j</sub>, row<sub>j</sub>, col<sub>j</sub>)</div><p>Pairs of numbers in each Query and Key are turned by an angle set by the patch's row and column. Their dot product then depends on how far apart two patches are, in both directions. The text model adds a learned position to the embedding once, at the start. The lab further down shows the rotation.</p>`,
 `<h4>Linear · widen</h4><div class="fx">a = W<sub>1</sub> · x</div><p>x is the patch's vector, 1,024 numbers. W<sub>1</sub> is a learned matrix of 4,096 × 1,024, so a has 4,096 numbers.</p>`,
 `<h4>SiLU</h4><div class="fx">SiLU(a) = a · sigmoid(a)</div><p>Applied to each of the 4,096 numbers on its own. Positive values pass almost unchanged and negative ones are pushed towards zero, much like GELU in the text model.</p>`,
 `<h4>Gate · W₃</h4><div class="fx">g = W<sub>3</sub> · x<br>h = SiLU(a) ⊙ g</div><p>A second matrix W<sub>3</sub> (also 4,096 × 1,024) widens the same input. The two widened vectors are multiplied number by number (⊙), so one decides how much of the other gets through. This branch is what the text model's MLP does not have.</p>`,
 `<h4>Linear · narrow</h4><div class="fx">y = W<sub>2</sub> · h<br>output = x + y</div><p>W<sub>2</sub> is 1,024 × 4,096 and brings the vector back to 1,024 numbers. The result is added to the vector that came in (the residual connection).</p>`,
 `<h4>Linear · to the LLM's width</h4><div class="fx">a = W<sub>a</sub> · h<sub>i</sub><br>W<sub>a</sub> : matrix(5,120 × 1,024)</div><p>h<sub>i</sub> is a patch's final vector, 1,024 numbers. W<sub>a</sub> resizes it to 5,120, the width of the language model's own token vectors.</p>`,
 `<h4>GELU</h4><div class="fx">GELU(a) = a · Φ(a)</div><p>Applied to each of the 5,120 numbers on its own, between the two linear layers.</p>`,
 `<h4>Linear</h4><div class="fx">token<sub>i</sub> = W<sub>b</sub> · GELU(a)<br>W<sub>b</sub> : matrix(5,120 × 5,120)</div><p>The result is one image token per patch. The language model receives the 48 of them, in order, among the text tokens of the prompt.</p>`
];
const blocksLeft=()=>BLOCKS-1-state.block;
function paintMore(root=$('#tx-flow')){
 const left=blocksLeft(),bars=Math.min(3,left+1);
 root.querySelectorAll('.more-a').forEach(t=>{t.textContent=left?`${left} MORE`:'LAST';});
 root.querySelectorAll('.more-b').forEach(t=>{t.textContent=left===1?'BLOCK':left?'BLOCKS':'BLOCK';});
 root.querySelectorAll('.more-bar').forEach(r=>{r.style.display=+r.dataset.b<bars?'':'none';});
}
function flow(){
 const img=image(),ps=shown(),n=ps.length,H=Y0+n*PITCH+48,r=raw();
 const yc=i=>Y0+i*PITCH+PITCH/2;
 const top=86,bottom=H-30,ch=bottom-top,fx=218,fy=48,fw=712,fh=H-fy-8;
 const gy={k:top+ch*0.2,q:top+ch*0.5,v:top+ch*0.82};
 const ty=(g,i)=>gy[g]+(i-(n-1)/2)*TP;
 const names={q:'Query',k:'Key',v:'Value'};
 // one part can be open at a time: 'embedding', 'attention', 'mlp' or 'projector'
 const X=state.expand,E=X==='attention',EE=X==='embedding',EP=X==='projector',EM=X==='mlp',mats=E?3:1,matW=n*TP,mgap=58,mx=m=>548+m*(matW+mgap),mx0=mx(0),cx=(j,m=0)=>mx(m)+j*TP,mtop=ty('q',0)-TP/2-3;
 const dA=(mats-1)*(matW+mgap),dE=EE?250:0,dP=EP?330:0,dM=EM?248:0,outX=716+dA,cardW=440+dA,anim=state.anim,rs=EM?150:0,prev=state.last||{dA:0,dE:0,dP:0,dM:0,rs:0},was=prev.dA;
 state.anim=null;state.last={dA,dE,dP,dM,rs};
 const dimEmb=X&&!EE?' xdim':'',dimBlock=EE||EP||EM?' xdim':'',dimFrame=EE||EP?' xdim':'',dimSide=X?' xdim':'',dimMlp=X&&!EM?' xdim':'',dimProj=X&&!EP?' xdim':'';
 const fromE=anim?prev.dE-dE:0,fromM=anim?prev.dM-dM:0,fromP=anim?(prev.dM-prev.dP)-(dM-dP):0,openE=anim==='open'&&EE,openP=anim==='open'&&EP,openM=anim==='open'&&EM;
 const late=(on,at)=>on?` late" style="--at:${at}ms`:'';
 const from=anim?was-dA:0,sx=w=>anim?((w-dA+was)/w).toFixed(4):1;
 const iy=Math.round((yc(0)+yc(n-1))/2-ROWS*CELL/2),iw=COLS*CELL,ih=ROWS*CELL;
 const thumb=(p,x,y,size)=>{const {row,col}=patchPlace(p);return `<svg x="${x}" y="${y}" width="${size}" height="${size}" viewBox="${col*16} ${row*16} 16 16" preserveAspectRatio="none"><image href="${img.src}" width="128" height="96"/></svg>`;};
 let s=`<title id="tx-flow-title">An image flowing through patches, embedding, transformer block ${state.block+1} and the projector</title>`;
 // the opened gated MLP is wide: the whole picture moves left to keep it in view
 s+=`<g transform="translate(${-rs})"><g class="${anim?'slide':''}" style="--from:${anim?rs-prev.rs:0}px">`;
 // ----- the image and its grid of patches: click a patch to select it
 s+=`<g class="stage${dimEmb}" style="--d:0"><text class="col-title" x="${IX}" y="30">Image</text><text class="col-sub" x="${IX}" y="${iy-10}">${COLS} × ${ROWS} = ${PATCHES} PATCHES</text>`;
 s+=`<image href="${img.src}" x="${IX}" y="${iy}" width="${iw}" height="${ih}" preserveAspectRatio="none"/><rect class="img-frame" x="${IX}" y="${iy}" width="${iw}" height="${ih}"/>`;
 for(let p=0;p<PATCHES;p++){
  const {row,col}=patchPlace(p),i=ps.indexOf(p);
  s+=`<rect class="pcell${i>=0?' in':''}${p===state.sel?' sel':''}" data-p="${p}"${i>=0?` data-t="${i}"`:''} x="${IX+col*CELL}" y="${iy+row*CELL}" width="${CELL}" height="${CELL}"><title>${patchName(p)} · ${where(p)}</title></rect>`;
 }
 s+=`<text class="col-sub" x="${IX}" y="${iy+ih+16}">CLICK A PATCH TO FOLLOW IT</text></g>`;
 // connectors from each shown patch in the image to its row
 s+=`<g class="stage${dimEmb}" style="--d:0">`;
 ps.forEach((p,i)=>{const {row,col}=patchPlace(p);s+=`<g data-t="${i}" class="img-link"><path class="fg" d="${curve(IX+(col+1)*CELL,iy+(row+0.5)*CELL,DX+90,yc(i))}" stroke-width="1.1"/></g>`;});
 s+=`</g><g transform="translate(${DX})">`;
 // ----- embedding: patch → vector. Opened: patch → flattened pixels × projection matrix = vector
 s+=`<g class="stage${dimEmb}" style="--d:0" data-zone="embedding"><g data-expand="embedding" role="button" tabindex="0" aria-label="Open the embedding step"><text class="col-title" x="58" y="30">Embedding</text></g></g>`;
 s+=`<g class="stage${dimEmb}" style="--d:0" data-zone="embedding">`;
 ps.forEach((p,i)=>{s+=`<g data-t="${i}"><text class="ptok" x="86" y="${yc(i)+4}" text-anchor="end">${patchName(p)}</text>${thumb(p,92,yc(i)-13,26)}<rect class="thumb-frame" x="92" y="${yc(i)-13}" width="26" height="26"/></g>`;});
 if(EE){
  const pill=(k,x,w,label,at)=>`<g class="stage-name${late(openE,at)}" data-formula="${k}" tabindex="0" role="button" aria-label="${label}: show the formula"><rect x="${x-w/2}" y="${fy+4}" width="${w}" height="19" rx="9"/><text x="${x}" y="${fy+17}" text-anchor="middle">${label}</text></g>`;
  s+=pill(6,96,64,'Patchify',250)+pill(7,172,56,'Flatten',650)+pill(8,284,108,'Linear projection',1050);
  ps.forEach((p,i)=>{
   const y=yc(i),st=i*60,px=img.stripe[p];
   s+=`<g class="emb-step${late(openE,250+st)}"><path class="emb-arrow" d="M124,${y}H152m-5,-4l5,4l-5,4"/></g>`;
   s+=`<g class="emb-step${late(openE,650+st)}">${px.map((v,k)=>`<rect x="162" y="${(y-15+k*3).toFixed(1)}" width="12" height="2.4" fill="#5b6b80" opacity="${(0.15+0.8*(1-v)).toFixed(2)}"/>`).join('')}<rect class="hit" data-dim="this patch's pixels, flattened · vector(768)" x="160" y="${y-16}" width="16" height="32"/>${i?'':`<text class="emb-key" x="180" y="${y-6}">pixels</text>`}<text class="emb-num" x="180" y="${y+(i?4:8)}">768</text></g>`;
   s+=`<g class="emb-step${late(openE,1050+st)}"><text class="emb-op" x="232" y="${y+5}" text-anchor="middle">×</text><rect class="wbox" data-dim="patch projection W · matrix(1,024 × 768), the same for every patch" x="250" y="${y-14}" width="68" height="28" rx="4"/><text class="wbox-t" x="284" y="${y+4}" text-anchor="middle">W<tspan baseline-shift="sub" font-size="8">patch</tspan></text></g>`;
   s+=`<g class="emb-step${late(openE,1450+st)}"><text class="emb-op" x="${dE+110}" y="${y+5}" text-anchor="middle">=</text></g>`;
  });
  s+=`<text class="mlp-size${late(openE,1450)}" x="210" y="${yc(n-1)+38}" text-anchor="middle">no position is added here · see 2D RoPE in attention</text>`;
 }
 s+=`</g>`;
 s+=`<g transform="translate(${dE})"><g class="${anim?'slide':''}" style="--from:${fromE}px">`;
 s+=`<g class="stage${dimFrame}" style="--d:1"><rect class="frame${anim?' widen':''}" style="--sx:${anim?((fw+prev.dA+prev.dM)/(fw+dA+dM)).toFixed(4):1}" x="${fx}" y="${fy}" width="${fw+dA+dM}" height="${fh}" rx="10"/>${chip('block',fx+18,fy,blockLabel(),176)}</g>`;
 s+=`<g class="stage${dimEmb}" style="--d:0" data-zone="embedding">`;
 ps.forEach((p,i)=>{s+=`<g data-t="${i}">${EE?'':`<path class="fg" d="M118,${yc(i)} H130" stroke-width="1.1"/>`}<rect data-dim="vector(1024)" x="130" y="${yc(i)-15}" width="12" height="30" rx="2" fill="#cdd6e2"/></g>`;});
 s+=`</g><g class="stage block-part${dimSide}" style="--d:1">`;
 ps.forEach((p,i)=>{
  const y=yc(i);
  [['q',-12],['k',0],['v',12]].forEach(([g,dy])=>{
   s+=`<g data-t="${i}"><path class="fg" d="${curve(142,y,232,y+dy)}" stroke-width="1.2"/><rect class="${g}" data-dim="${names[g]} weights · matrix(1,024 × 1,024)" x="232" y="${y+dy-4.5}" width="9" height="9" rx="1.5" opacity=".75"/><text class="mini" x="246" y="${y+dy+3}">${g.toUpperCase()}</text></g>`;
  });
 });
 s+=`</g>`;
 // ----- head card
 s+=`<g class="stage head-part${dimBlock}" style="--d:2" data-zone="attention">`;
 for(let d=3;d>=1;d--)s+=`<rect class="card back${anim?' widen':''}" style="--sx:${sx(cardW)}" x="${330+d*5}" y="${top+d*5}" width="${cardW}" height="${ch}" rx="6"/>`;
 s+=`<rect class="card front${anim?' widen':''}" style="--sx:${sx(cardW)}" x="330" y="${top}" width="${cardW}" height="${ch}" rx="6"/>`;
 for(const g of ['k','q','v']){
  s+=`<text class="lbl-${g}" x="452" y="${ty(g,0)-12}">${names[g]}</text>`;
  // Query and Key are rotated by 2D RoPE before they meet; Value is not
  if(g!=='v')s+=`<g class="stage-name rope-name" data-formula="9" tabindex="0" role="button" aria-label="2D RoPE on the ${names[g]}: show the formula"><rect x="${g==='q'?500:488}" y="${ty(g,0)-24}" width="58" height="16" rx="8"/><text x="${(g==='q'?500:488)+29}" y="${ty(g,0)-12.5}" text-anchor="middle">↻ 2D RoPE</text></g>`;
  ps.forEach((p,i)=>{
   const y=ty(g,i),dy=g==='q'?-12:g==='k'?0:12;
   s+=`<g data-t="${i}"><path class="f${g}" d="${curve(254,yc(i)+dy,380,y)}" stroke-width="1" opacity=".55"/><text class="mini" x="446" y="${y+3}" text-anchor="end">${patchName(p)}</text><rect class="${g}" data-dim="${names[g]}, this head · vector(64)" x="452" y="${y-1.5}" width="12" height="3" rx="1"/><rect class="hit" data-dim="${names[g]}, this head · vector(64)" x="450" y="${y-5}" width="16" height="10"/></g>`;
  });
 }
 s+=`</g><g class="stage head-part${dimBlock}" style="--d:3" data-zone="attention">`;
 const opening=anim==='open',QK=1200,t0=300,T1=t0+QK/n+QK+500,T2=T1+500+1000+150,T3=T2+450+1000;
 const draw=k=>opening?` pathLength="1" class="draw" style="--at:${Math.round(t0+k*QK/n)}ms"`:'';
 const look=(m,i,j)=>['#7f58a3',dotOpacity(stageShade(m,i,j,r)),'transparent'];
 ps.forEach((p,j)=>{s+=`<g data-t="${j}" class="fk" opacity=".5"><path${draw(j)} d="M464,${ty('k',j)} H${cx(j)} V${mtop}" stroke-width="1"/></g>`;});
 ps.forEach((p,i)=>{s+=`<g data-t="${i}" class="fq" opacity=".6"><path${draw(i)} d="M464,${ty('q',i)} H${mx0-TP/2-2}" stroke-width="1"/></g>`;});
 const stages=E?[0,1,2]:[2];
 stages.forEach((m,k)=>{
  s+=`<g class="${opening&&k?'grow':''}" style="--from:${-(matW+mgap)}px;--at:${[0,T1,T2][k]}ms">`;
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
   const v=stageShade(m,i,j,r);let f='';
   if(opening&&k===0)f=` pop" style="--at:${Math.round(t0+QK/n+(i*n+j)*QK/(n*n))}ms`;
   else if(opening){const a=look(m-1,i,j),b=look(m,i,j);f=` turn" style="--at:${[0,T1+500,T2+450][k]}ms;--f0:${a[0]};--o0:${a[1]};--s0:${a[2]};--f1:${b[0]};--o1:${b[1]};--s1:${b[2]}`;}
   s+=`<circle class="cell${f}" data-t="${i}" data-q="${i}" data-k="${j}" data-m="${m}" cx="${cx(j,k)}" cy="${ty('q',i)}" r="5" fill="#7f58a3" fill-opacity="${dotOpacity(v)}"/>`;
  }
  if(E){
   const mid=mx(k)+matW/2-TP/2,by=ty('q',n-1)+TP/2,[lo,hi]=stageRange(m,r);
   s+=`<g class="stage-name" data-formula="${m}" tabindex="0" role="button" aria-label="${STAGE_NAMES[m]}: show the formula"><rect x="${mid-52}" y="${by+8}" width="104" height="19" rx="9"/><text x="${mid}" y="${by+21}" text-anchor="middle">${STAGE_NAMES[m]}</text></g>`;
   s+=`<text class="leg leg-lo" data-m="${m}" x="${mid-20}" y="${by+41}" text-anchor="end">${lo}</text><rect class="leg-bar" x="${mid-16}" y="${by+35}" width="32" height="4" rx="2"/><text class="leg leg-hi" data-m="${m}" x="${mid+20}" y="${by+41}">${hi}</text>`;
   if(m===1)s+=`<text class="no-mask" x="${mid}" y="${by+56}" text-anchor="middle">no mask: every dot stays</text>`;
   if(k<2)s+=`<text class="stage-arrow" x="${mx(k)+matW-TP/2+mgap/2}" y="${ty('q',(n-1)/2)+4}" text-anchor="middle">→</text>`;
  }
  s+=`</g>`;
 });
 if(!E)s+=`<g class="attn-label" data-expand="attention" role="button" tabindex="0" aria-label="Open attention into its three stages"><text class="col-sub" x="${mx0-6}" y="${ty('q',n-1)+TP/2+16}">ATTENTION · NO MASK</text></g>`;
 const mr=cx(n-1,mats-1)+8;
 s+=`<g class="${anim?'slide':''}" style="--from:${from}px"><path class="band${opening?' late':''}" style="--at:${T3}ms;fill:#e6dcf0" d="M${mr},${ty('q',0)-6} L${outX},${ty('q',0)-2} V${ty('q',n-1)+2} L${mr},${ty('q',n-1)+6} Z"/>`;
 s+=`<g class="stage-name out-name" data-formula="3" tabindex="0" role="button" aria-label="Out: show the formula"><rect x="${outX-13}" y="${ty('q',0)-27}" width="40" height="19" rx="9"/><text x="${outX+7}" y="${ty('q',0)-14}" text-anchor="middle">Out</text></g>`;
 ps.forEach((p,i)=>{
  s+=`<g data-t="${i}">${E?`<path class="fv${opening?' late':''}" style="--at:${T3}ms" d="M464,${ty('v',i)} H${outX-74} C${outX-30},${ty('v',i)} ${outX-44},${ty('q',i)} ${outX},${ty('q',i)}" stroke-width="1" opacity=".45"/>`:`<path class="fv" d="${curve(464,ty('v',i),outX,ty('q',i))}" stroke-width="1" opacity=".4"/>`}<rect class="o" x="${outX}" y="${ty('q',i)-1.5}" width="14" height="3" rx="1"/><rect class="hit" data-dim="Out, this head · vector(64)" x="${outX-2}" y="${ty('q',i)-5}" width="18" height="10"/></g>`;
 });
 s+=`</g></g>`;
 s+=`<g class="stage${dimBlock}" style="--d:2">${chip('head',346,top,headLabel(),282)}</g>`;
 s+=`<g transform="translate(${dA})" style="--oxL:${fx-dA}px;--oxR:${fx+fw+dM}px"><g class="${anim?'slide':''}" style="--from:${from}px">`;
 // ----- gated MLP. Opened: W1 → SiLU, in parallel W3, multiply, W2.
 const xo=902+dM,mpill=(k,x,w,label,at)=>`<g class="stage-name${late(openM,at)}" data-formula="${k}" tabindex="0" role="button" aria-label="${label}: show the formula"><rect x="${x-w/2}" y="${fy+40}" width="${w}" height="19" rx="9"/><text x="${x}" y="${fy+53}" text-anchor="middle">${label}</text></g>`;
 s+=`<g class="${dimMlp}" data-zone="mlp"><g class="stage" style="--d:4"><g class="stage-name mlp-name" data-formula="4" tabindex="0" role="button" aria-label="Gated MLP: show the formula"><rect x="${806-9}" y="${fy+21}" width="74" height="19" rx="9"/><text x="${806+28}" y="${fy+34}" text-anchor="middle">Gated MLP</text></g>${EM?mpill(10,893,58,'Linear W₁',300)+mpill(11,963,44,'SiLU',900)+mpill(12,1024,64,'Gate · W₃',1400)+mpill(13,1116,58,'Linear W₂',2000):''}</g>`;
 s+=`<g class="stage block-part" style="--d:4">`;
 ps.forEach((p,i)=>{
  const y=yc(i),lit=illustrativeVector(7000+ps[i],5),gate=illustrativeVector(9000+ps[i],5);
  s+=`<g data-t="${i}"><path class="fo" d="${curve(716+14,ty('q',i),806,y)}" stroke-width="1.2" opacity=".7"/><rect data-dim="vector(1024)" x="806" y="${y-15}" width="10" height="30" rx="2" fill="#d0bfe1"/>`;
  if(EM){
   const wide=(x,fill,dim)=>`<rect data-dim="${dim}" x="${x}" y="${y-22}" width="26" height="44" rx="3" fill="${fill}"/>`,cells=(x,vs,colour)=>vs.map((v,k)=>v>0?`<rect x="${x}" y="${(y-22+k*8.8).toFixed(1)}" width="26" height="8.8" fill="${colour}" opacity="${(0.35+0.6*v).toFixed(2)}" style="pointer-events:none"/>`:'').join('');
   s+=`<g class="${late(openM,300+i*50).trim()}"><path class="band" style="fill:#d3dcee" d="M816,${y-15} L880,${y-22} V${y+22} L816,${y+15} Z"/>${wide(880,'#b3c2e0','W₁ · x · vector(4096)')}</g>`;
   s+=`<g class="${late(openM,900+i*50).trim()}"><path class="emb-arrow" d="M910,${y}H944m-5,-4l5,4l-5,4"/>${wide(950,'#d3dcee','SiLU(W₁ · x) · vector(4096)')}${cells(950,lit,'#6681b5')}</g>`;
   s+=`<g class="${late(openM,1400+i*50).trim()}"><text class="emb-op" x="990" y="${y+5}" text-anchor="middle">⊙</text><path class="gate-wire" d="M811,${y+15} V${y+26} H1017 V${y+22}"/>${wide(1004,'#cfe3da','the gate, W₃ · x · vector(4096)')}${cells(1004,gate,'#5fae93')}<text class="emb-op" x="1044" y="${y+5}" text-anchor="middle">=</text>${wide(1058,'#d3dcee','gated vector · vector(4096)')}${cells(1058,lit.map((v,k)=>v>0&&gate[k]>0?v*gate[k]+0.25:0),'#6681b5')}</g>`;
   s+=`<g class="${late(openM,2000+i*50).trim()}"><path class="band" style="fill:#d3dcee" d="M1084,${y-22} L${xo},${y-15} V${y+15} L1084,${y+22} Z"/><rect data-dim="vector(1024)" x="${xo}" y="${y-15}" width="10" height="30" rx="2" fill="#a9c0f3"/></g>`;
  }else{
   s+=`<path class="band" style="fill:#d3dcee" d="M816,${y-15} L846,${y-22} V${y+22} L816,${y+15} Z"/><rect data-dim="vector(4096), gated" x="846" y="${y-22}" width="26" height="44" rx="3" fill="#b3c2e0"/><rect x="859" y="${y-22}" width="13" height="44" rx="3" fill="#bcd9cc" style="pointer-events:none"/><path class="band" style="fill:#d3dcee" d="M872,${y-22} L902,${y-15} V${y+15} L872,${y+22} Z"/><rect data-dim="vector(1024)" x="902" y="${y-15}" width="10" height="30" rx="2" fill="#a9c0f3"/>`;
  }
  s+=`</g>`;
 });
 const dy=yc(n-1)+38,size=(x,label,at)=>`<text class="mlp-size${late(openM,at)}" x="${x}" y="${dy}" text-anchor="middle">${label}</text>`;
 s+=EM?size(811,'1,024',0)+size(893,'4,096',300)+size(963,'4,096',900)+size(1017,'4,096',1400)+size(1071,'4,096',1400)+size(xo+5,'1,024',2000):size(811,'1,024',0)+size(859,'4,096',0)+size(907,'1,024',0);
 s+=`</g></g>`;
 s+=`<g transform="translate(${dM})"><g class="${anim?'slide':''}${dimSide}" style="--from:${fromM}px"><g class="stage block-part" style="--d:4"><text class="col-sub more-a" x="944" y="46"></text><text class="col-sub more-b" x="944" y="58"></text>`;
 ps.forEach((p,i)=>{
  const y=yc(i);
  s+=`<g data-t="${i}"><path class="fg" d="M912,${y} H944" stroke-width="1.2"/>`;
  for(let b=0;b<3;b++)s+=`<rect class="more-bar" data-b="${b}" data-dim="vector(1024)" x="${944+b*12}" y="${y-15}" width="7" height="30" rx="2" fill="#d3e0fa" opacity="${0.9-b*0.25}"/>`;
  s+=`</g>`;
 });
 s+=`</g></g></g>`;
 // ----- projector: every patch's final vector is resized for the language model (no vocabulary, no softmax)
 s+=`<g transform="translate(${dM-dP})"><g class="${anim?'slide':''}${dimProj}" style="--from:${fromP}px" data-zone="projector">`;
 s+=`<g class="stage" style="--d:6"><g data-expand="projector" role="button" tabindex="0" aria-label="Open the projector step"><text class="col-title" x="990" y="30">Projector</text></g></g>`;
 s+=`<g class="stage proj-part" style="--d:6">`;
 if(EP){
  const pill=(k,x,w,label,at)=>`<g class="stage-name${late(openP,at)}" data-formula="${k}" tabindex="0" role="button" aria-label="${label}: show the formula"><rect x="${x-w/2}" y="${fy+4}" width="${w}" height="19" rx="9"/><text x="${x}" y="${fy+17}" text-anchor="middle">${label}</text></g>`;
  s+=`<rect class="prob-bg" x="952" y="42" width="420" height="${H-52}" rx="8"/>`+pill(14,1025,56,'Linear',300)+pill(15,1118,50,'GELU',900)+pill(16,1211,56,'Linear',1500);
  ps.forEach((p,i)=>{
   const y=yc(i),lit=illustrativeVector(5000+p,6),bar=(x,fill,dim)=>`<rect data-dim="${dim}" x="${x}" y="${y-24}" width="22" height="48" rx="3" fill="${fill}"/>`;
   s+=`<g data-t="${i}"><path class="fg" d="M975,${y} H986" stroke-width="1.2"/><rect data-dim="final vector of ${patchName(p)} · vector(1024)" x="986" y="${y-15}" width="10" height="30" rx="2" fill="#a9c0f3"/>`;
   s+=`<g class="${late(openP,300+i*50).trim()}"><path class="band" style="fill:#d3e9df" d="M996,${y-15} L1060,${y-24} V${y+24} L996,${y+15} Z"/>${bar(1060,'#b9d9cb','W_a · h · vector(5120)')}</g>`;
   s+=`<g class="${late(openP,900+i*50).trim()}"><path class="emb-arrow" d="M1086,${y}H1146m-5,-4l5,4l-5,4"/>${bar(1152,'#d3e9df','after GELU · vector(5120)')}${lit.map((v,k)=>v>0?`<rect x="1152" y="${(y-24+k*8).toFixed(1)}" width="22" height="8" fill="#5fae93" opacity="${(0.35+0.6*v).toFixed(2)}" style="pointer-events:none"/>`:'').join('')}</g>`;
   s+=`<g class="${late(openP,1500+i*50).trim()}"><path class="emb-arrow" d="M1178,${y}H1238m-5,-4l5,4l-5,4"/>${bar(1244,'#8fc7b3',`image token for ${patchName(p)} · vector(5120)`)}<text class="proj-token" x="1274" y="${y+4}">image token</text></g></g>`;
  });
  const by=yc(n-1)+38;
  s+=`<text class="mlp-size" x="991" y="${by}" text-anchor="middle">1,024</text><text class="mlp-size${late(openP,300)}" x="1071" y="${by}" text-anchor="middle">5,120</text><text class="mlp-size${late(openP,900)}" x="1163" y="${by}" text-anchor="middle">5,120</text><text class="mlp-size${late(openP,1500)}" x="1255" y="${by}" text-anchor="middle">5,120</text>`;
 }else{
  s+=`<g class="stage-name proj-name" data-formula="5" tabindex="0" role="button" aria-label="Projector, 1,024 to 5,120: show the formula"><rect x="985" y="${fy+21}" width="92" height="19" rx="9"/><text x="1031" y="${fy+34}" text-anchor="middle">1,024 → 5,120</text></g>`;
  ps.forEach((p,i)=>{
   const y=yc(i);
   s+=`<g data-t="${i}"><path class="fg" d="M975,${y} H992" stroke-width="1.2"/><path class="band" style="fill:#d3e9df" d="M992,${y-15} L1024,${y-24} V${y+24} L992,${y+15} Z"/><rect data-dim="image token for ${patchName(p)} · vector(5120)" x="1024" y="${y-24}" width="12" height="48" rx="3" fill="#8fc7b3"/><path class="fg to-llm" d="M1036,${y} H1060m-5,-4l5,4l-5,4" stroke-width="1.2"/></g>`;
  });
  s+=`<rect class="llm-box" x="1064" y="${yc(0)-26}" width="34" height="${yc(n-1)-yc(0)+52}" rx="8"/><text class="llm-t" transform="translate(1085,${(yc(0)+yc(n-1))/2}) rotate(90)" text-anchor="middle">LANGUAGE MODEL</text>`;
  s+=`<text class="mlp-size" x="1030" y="${yc(n-1)+38}" text-anchor="middle">5,120</text>`;
 }
 s+=`</g></g></g></g></g></g></g></g></g></g>`;
 const svg=$('#tx-flow');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=s;svg.classList.toggle('expanded',!!X);svg.dataset.open=X||'';
 svg.style.setProperty('--oxL',`${fx}px`);svg.style.setProperty('--oxR',`${fx+fw+dA+dM}px`);svg.classList.remove('tracing');
 $('#tx-tip').hidden=true;$('#tx-formula').hidden=true;
 paintMore(svg);defaultResult();
 const ps0=ps[0],psN=ps[n-1];
 $('#tx-sentence').innerHTML=`128 × 96 image → <b>${PATCHES} patches</b> of 16 × 16 · showing ${patchName(ps0)}–${patchName(psN)}`;
}
function defaultResult(){
 $('#tx-result').textContent=`${PATCHES} patches go in and ${PATCHES} image tokens come out: every patch is kept, not just the last one. Following ${patchName(state.sel)} (${where(state.sel)}); block ${state.block+1} of ${BLOCKS}, head ${state.head+1} of ${HEADS} shown.`;
}

/* ---------- 02 attention: the three grids, with no mask step ---------- */
let attnTicket=0,attnBusy=false;
function attention({play=false}={}){
 const ps=shown(),n=ps.length,r=raw(),{scaled,weights}=attentionStages(r);
 const maxRaw=Math.max(...r.flat().map(Math.abs))||1,maxSc=Math.max(...scaled.flat().map(Math.abs))||1;
 const tint=(v,m)=>v>=0?`rgba(79,123,232,${(0.06+0.5*v/m).toFixed(2)})`:`rgba(199,118,91,${(0.06-0.5*v/m).toFixed(2)})`;
 const mat=cell=>{
  let h=`<div class="mat" style="grid-template-columns:minmax(40px,56px) repeat(${n},minmax(0,1fr))"><span></span>${ps.map((p,j)=>`<span class="cl" data-k="${j}">${patchName(p)}</span>`).join('')}`;
  for(let i=0;i<n;i++){h+=`<span class="rl" data-q="${i}">${patchName(ps[i])}</span>`;for(let j=0;j<n;j++)h+=cell(i,j);}
  return h+'</div>';
 };
 const c=(i,j,text,bg,order=i*n+j)=>`<span class="c" data-q="${i}" data-k="${j}" style="--i:${order};--bg:${bg}">${text}</span>`;
 const stages=[
  ['1 · Dot product <span class="fm">Q · K<sup>T</sup></span>','Query of the row patch · Key of the column patch, both rotated by 2D RoPE.',mat((i,j)=>c(i,j,r[i][j].toFixed(1),tint(r[i][j],maxRaw)))],
  [`2 · Scale <span class="fm">Q · K<sup>T</sup> ÷ √d<sub>k</sub>, &nbsp;no mask</span>`,`Divide by √${HEAD_DIM} = 8 (d<sub>k</sub> = ${HEAD_DIM}). Nothing is set to −∞.`,mat((i,j)=>c(i,j,scaled[i][j].toFixed(2),tint(scaled[i][j],maxSc)))],
  ['3 · Softmax <span class="fm">e<sup>x</sup> ÷ Σ e<sup>x</sup> &nbsp;per row</span>','Each row becomes weights that sum to 100%.',mat((i,j)=>c(i,j,Math.round(weights[i][j]*100)+'%',`rgba(127,88,163,${(0.05+0.7*weights[i][j]).toFixed(2)})`,i*4+j*0.6))]
 ];
 $('#attn-stages').innerHTML=stages.map(([t,p,m],k)=>`<div class="attn-stage" data-stage="${k+1}"><h3>${t}</h3><p>${p}</p>${m}</div>`).join('');
 playAttention(play);
}
const ATTN_STEPS=[
 [0,1,'Step 1 · Multiply. Each cell is one Query (its row) times one Key (its column). Every pair of patches is scored.'],
 [1500,2,'Step 2 · Scale. Every score is divided by √64 = 8. There is no mask: an image has no “later”, so all of the grid stays.'],
 [3000,3,'Step 3 · Softmax, one row at a time. Each row becomes weights that add up to 100%.'],
];
function playAttention(animate){
 const ticket=++attnTicket,box=$('#attn-stages'),stage=k=>box.querySelector(`[data-stage="${k}"]`),btn=$('#attn-play');
 const calm=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const finish=()=>{attnBusy=false;box.classList.remove('playing');for(const k of [1,2,3])stage(k).classList.add('show');if(btn)btn.textContent='↻ Replay the steps';const i=shown().indexOf(state.sel),w=attentionStages(raw()).weights[i];attnResult(i,w.indexOf(Math.max(...w)));};
 if(!animate||calm){box.classList.add('instant');finish();return;}
 attnBusy=true;box.classList.remove('instant');box.classList.add('playing');if(btn)btn.textContent='Playing…';
 $$('#attn-stages .c').forEach(el=>el.classList.remove('on','row'));
 for(const [at,k,text] of ATTN_STEPS)setTimeout(()=>{if(ticket!==attnTicket)return;stage(k).classList.add('show');$('#attn-result').textContent=text;},at+30);
 setTimeout(()=>{if(ticket===attnTicket)finish();},4600);
}
function attnResult(i,j){
 const ps=shown(),r=raw(),{scaled,weights}=attentionStages(r);
 $$('#attn-stages .c').forEach(el=>{el.classList.toggle('on',+el.dataset.q===i&&+el.dataset.k===j);el.classList.toggle('row',+el.dataset.q===i);});
 $('#attn-result').textContent=`${patchName(ps[i])} looking at ${patchName(ps[j])}${j>i?' (a patch that comes after it, which a text model would have masked)':''}: dot product ${r[i][j].toFixed(1)} → ÷8 = ${scaled[i][j].toFixed(2)} → softmax over the row = ${pct(weights[i][j])} of its attention (block ${state.block+1}, head ${state.head+1}; illustrative values).`;
}

/* Changing block or head redraws only that part, as on the text page. */
const onScreen=el=>{const r=el.getBoundingClientRect();return r.top<innerHeight*0.85&&r.bottom>innerHeight*0.15;};
function swap(kind,dir){
 const svg=$('#tx-flow'),calm=matchMedia('(prefers-reduced-motion: reduce)').matches,r=raw();
 const sel=kind==='block'?'.head-part,.block-part':'.head-part',parts=[...svg.querySelectorAll(sel)].filter(g=>!g.classList.contains('ghost'));
 svg.querySelectorAll('.ghost').forEach(g=>g.remove());
 const ghosts=calm?[]:parts.map(g=>{const c=g.cloneNode(true);c.classList.remove('swap','stage');c.classList.add('ghost');c.setAttribute('aria-hidden','true');g.before(c);return c;});
 parts.forEach(g=>g.querySelectorAll('.cell').forEach(c=>c.setAttribute('fill-opacity',dotOpacity(stageShade(+c.dataset.m,+c.dataset.q,+c.dataset.k,r)))));
 parts.forEach(g=>g.querySelectorAll('.leg').forEach(t=>{t.textContent=stageRange(+t.dataset.m,r)[t.classList.contains('leg-lo')?0:1];}));
 svg.querySelector('.block-label').textContent=blockLabel();
 parts.forEach(g=>paintMore(g));
 svg.querySelector('.head-label').textContent=headLabel();
 svg.querySelector('title').textContent=`An image flowing through patches, embedding, transformer block ${state.block+1} and the projector`;
 syncSteppers();attention({play:onScreen($('#attn-stages'))});defaultResult();
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
 syncSteppers();flow();attention();
 if(fresh){const lab=$('#tx-lab');lab.classList.remove('fresh');void lab.offsetWidth;lab.classList.add('fresh');}
}
function init(){
 $('#tx-example').innerHTML=IMAGES.map((e,i)=>`<option value="${i}">${e.name}</option>`).join('');
 $('#tx-example').addEventListener('change',e=>{state.img=+e.target.value;state.sel=image().sel;state.expand=null;state.last=null;render({fresh:true});});
 const grid=$('#packing-grid');if(grid)grid.innerHTML=Array.from({length:100},(_,i)=>{const r=Math.floor(i/10),c=i%10;return `<i class="${r<6&&c<6?'a':r>=6&&c>=6?'b':''}"></i>`;}).join('');
 document.addEventListener('click',e=>{
  const step=e.target.closest('[data-step]');
  if(step){const kind=step.closest('[data-stepper]').dataset.stepper,max=kind==='block'?BLOCKS:HEADS;const dir=+step.dataset.step;state[kind]=(state[kind]+dir+max)%max;swap(kind,dir);return;}
  if(e.target.closest('#tx-flow')){
   // a click on a patch in the image selects it; the six rows move to the part of the sequence that holds it
   const cell=e.target.closest('[data-p]');
   if(cell&&!state.expand){state.sel=+cell.dataset.p;state.last=null;flow();attention();trace(String(shown().indexOf(state.sel)));return;}
   const zoneEl=e.target.closest('[data-zone]'),zone=zoneEl&&!e.target.closest('.ghost')?zoneEl.dataset.zone:null;
   if(!state.expand&&zone){setExpand(zone);return;}
   if(state.expand&&zone!==state.expand&&!e.target.closest('.chip')){setExpand(null);return;}
  }
 });
 document.addEventListener('keydown',e=>{
  if(e.key==='Escape'&&state.expand){setExpand(null);return;}
  const b=(e.key==='Enter'||e.key===' ')&&e.target.closest?.('#tx-flow .step,#tx-flow [data-expand]');
  if(b){e.preventDefault();b.dispatchEvent(new MouseEvent('click',{bubbles:true}));}
 });
 const svg=$('#tx-flow');
 // tracing one patch through the diagram
 function trace(t){
  svg.classList.add('tracing');svg.querySelectorAll('[data-t]').forEach(x=>x.classList.toggle('on',x.dataset.t===t));
  const p=shown()[+t];
  $('#tx-result').textContent=`${patchName(p)} (${where(p)}): 768 pixel values → a vector of 1,024 → ${BLOCKS} blocks of attention and gated MLP → an image token of 5,120 for the language model.`;
 }
 svg.addEventListener('mouseover',e=>{
  if(state.expand)return;
  const el=e.target.closest('[data-t]');if(!el)return;
  trace(el.dataset.t);
  const cell=e.target.closest('.cell');
  if(cell){
   const i=+cell.dataset.q,j=+cell.dataset.k,ps=shown(),w=attentionStages(raw()).weights[i][j];
   $('#tx-result').textContent=`${patchName(ps[i])} gives ${pct(w)} of its attention to ${patchName(ps[j])} in block ${state.block+1}, head ${state.head+1} (illustrative, among the six patches shown).`;
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
  const cell=e.target.closest('.cell'),name=e.target.closest('.stage-name');
  svg.querySelectorAll('.cell.hot').forEach(c=>c.classList.remove('hot'));
  if(name)showFormula(name);else formula.hidden=true;
  if(cell&&state.expand==='attention'){
   const i=+cell.dataset.q,j=+cell.dataset.k,m=+cell.dataset.m,ps=shown(),r=raw(),{scaled,weights}=attentionStages(r);
   const q=patchName(ps[i]),k=patchName(ps[j]);
   tip.innerHTML=m===0?`<b>${q} · ${k}</b>Query · Key = ${r[i][j].toFixed(2)}`:m===1?`<b>${q} → ${k}</b>${r[i][j].toFixed(2)} ÷ √${HEAD_DIM} = ${scaled[i][j].toFixed(2)}`:`<b>${q} → ${k}</b>attention weight ${pct(weights[i][j])}`;
   cell.classList.add('hot');near(tip,cell,false);
  }else{
   const sized=e.target.closest('[data-dim]');
   if(sized){tip.textContent=sized.dataset.dim;near(tip,sized,false);}else tip.hidden=true;
  }
 });
 svg.addEventListener('focusin',e=>{const name=e.target.closest('.stage-name');if(name)showFormula(name);});
 svg.addEventListener('focusout',()=>{formula.hidden=true;});
 svg.addEventListener('mouseleave',()=>{svg.classList.remove('tracing');svg.querySelectorAll('.on').forEach(x=>x.classList.remove('on'));tip.hidden=true;formula.hidden=true;svg.querySelectorAll('.cell.hot').forEach(c=>c.classList.remove('hot'));defaultResult();});
 $('#attn-stages').addEventListener('mouseover',e=>{const c=e.target.closest('.c');if(c&&!attnBusy)attnResult(+c.dataset.q,+c.dataset.k);});
 $('#attn-play').addEventListener('click',()=>attention({play:true}));
 new IntersectionObserver((entries,io)=>{if(entries.some(e=>e.isIntersecting)){io.disconnect();attention({play:true});}},{threshold:0.35}).observe($('#attn-stages'));
 render({fresh:true});
}
init();
