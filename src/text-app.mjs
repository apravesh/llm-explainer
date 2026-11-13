// Text transformers (GPT-2): explorer and section labs.
// All numbers come from src/text-data.mjs (recorded from GPT-2) or are computed
// from them in src/text-engine.mjs. Nothing here runs a model.
import {EXAMPLES} from './text-data.mjs';
import {attentionStages,probabilities,sample,showToken,illustrativeVector,HEAD_DIM} from './text-engine.mjs';

const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=p=>p>=0.9995?'100%':p<=0?'0%':(p*100).toFixed(p<0.1?2:1)+'%';
const BLOCKS=12,HEADS=12;
const state={ex:0,block:0,head:0,temperature:0.8,type:'top-k',k:5,p:0.9,rnd:0.3,tok:0,sel:null,expand:null,anim:null,last:null};
const ex=()=>EXAMPLES[state.ex];
const sampling=()=>state.type==='top-k'?{type:'top-k',value:state.k}:{type:'top-p',value:state.p};
const rows=()=>probabilities(ex().top,{temperature:state.temperature,sampling:sampling()});

/* ---------- controls shared by the explorer and the probabilities lab ---------- */
function controlsMarkup(i){
 return `<label><span class="has-formula" data-formula="6" tabindex="0">Temperature</span><span class="row"><input type="range" min="0.1" max="2" step="0.1" data-c="temperature" aria-label="Temperature"><output data-o="temperature"></output></span></label>
<fieldset><legend>Sampling</legend><span class="row"><span class="radios"><label class="has-formula" data-formula="7"><input type="radio" name="tx-type-${i}" value="top-k" data-c="type">Top-k</label><label class="has-formula" data-formula="8"><input type="radio" name="tx-type-${i}" value="top-p" data-c="type">Top-p</label></span><input type="range" data-c="amount" aria-label="Top-k or top-p value"><output data-o="amount"></output></span></fieldset>`;
}
function syncControls(){
 $$('[data-c=temperature]').forEach(el=>{el.value=state.temperature;});
 $$('[data-o=temperature]').forEach(el=>{el.textContent=state.temperature.toFixed(1);});
 $$('[data-c=type]').forEach(el=>{el.checked=el.value===state.type;});
 $$('[data-c=amount]').forEach(el=>{
  if(state.type==='top-k'){el.min=1;el.max=10;el.step=1;el.value=state.k;}
  else{el.min=0.05;el.max=1;el.step=0.05;el.value=state.p;}
 });
 $$('[data-o=amount]').forEach(el=>{el.textContent=state.type==='top-k'?`k=${state.k}`:`p=${state.p.toFixed(2)}`;});
}
function syncSteppers(){
 $$('div[data-stepper]').forEach(el=>{
  const kind=el.dataset.stepper,label=kind==='block'?`Block ${state.block+1} of ${BLOCKS}`:`Head ${state.head+1} of ${HEADS}`;
  el.innerHTML=`<button type="button" data-step="-1" aria-label="Previous ${kind}">‹</button><b>${label}</b><button type="button" data-step="1" aria-label="Next ${kind}">›</button>`;
 });
}

/* ---------- explorer flow diagram ---------- */
const W=1230,Y0=106,PITCH=54,TP=13;
const blockLabel=()=>`TRANSFORMER BLOCK ${state.block+1} / ${BLOCKS}`,headLabel=()=>`MULTI-HEAD SELF-ATTENTION, HEAD ${state.head+1} / ${HEADS}`;
// A label with ‹ › buttons that sits on a frame's top edge (the Vision page's pattern).
const chip=(kind,x,y,label,w)=>{
 const btn=(d,bx)=>`<g class="step" data-step="${d}" role="button" tabindex="0" aria-label="${d>0?'Next':'Previous'} ${kind}"><rect x="${bx}" y="${y-9}" width="18" height="18" rx="4"/><text x="${bx+9}" y="${y+4.5}" text-anchor="middle">${d>0?'›':'‹'}</text></g>`;
 return `<g class="chip" data-stepper="${kind}"><rect class="chip-bg" x="${x-7}" y="${y-11}" width="${w+64}" height="22"/>${btn(-1,x)}<text class="chip-label ${kind}-label" x="${x+25+w/2}" y="${y+3.5}" text-anchor="middle">${label}</text>${btn(1,x+32+w)}</g>`;
};
const curve=(x1,y1,x2,y2)=>{const m=(x1+x2)/2;return `M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}`;};
// Attention shown as its three stages (dot product → scaling · mask → softmax · dropout).
// Returns the 0..1 shade of one dot, or null where the mask removed it.
const STAGE_NAMES=['Dot product','Scaling · Mask','Softmax · Dropout'];
function stageShade(m,i,j,raw=ex().attn[state.block][state.head]){
 if(m===2)return j>i?null:attentionStages(raw).weights[i][j];
 if(m===1&&j>i)return null;
 const flat=raw.flat(),lo=Math.min(...flat),hi=Math.max(...flat);
 return (raw[i][j]-lo)/((hi-lo)||1);
}
const stageRange=(m,raw=ex().attn[state.block][state.head])=>{const flat=raw.flat(),lo=Math.min(...flat),hi=Math.max(...flat),d=m===1?8:1;return m===2?['0','1']:[(lo/d).toFixed(1),(hi/d).toFixed(1)];};
const dotOpacity=v=>(0.07+0.93*v).toFixed(3);
const FORMULAS=[
 `<h4>Dot product</h4><div class="fx">score<sub>i,j</sub> = Q<sub>i</sub> · K<sub>j</sub> = <span class="sum">Σ<sub>d=1</sub><sup>64</sup></span> Q<sub>i,d</sub> × K<sub>j,d</sub></div><p>The Query of token i (the row) is multiplied with the Key of token j (the column), number by number, and the 64 products are added up. A high score means token j is relevant to token i.</p>`,
 `<h4>Scaling · Mask</h4><div class="fx">scaled<sub>i,j</sub> = score<sub>i,j</sub> ÷ √64 &nbsp;&nbsp;if j ≤ i<br>scaled<sub>i,j</sub> = −∞ &nbsp;&nbsp;if j &gt; i</div><p>Dividing by √64 = 8 keeps the scores in a range softmax handles well. Tokens that come later (j &gt; i) are masked out, so a token cannot look ahead.</p>`,
 `<h4>Softmax · Dropout</h4><div class="fx">weight<sub>i,j</sub> = <span class="frac"><span>exp(scaled<sub>i,j</sub>)</span><span><span class="sum">Σ<sub>k</sub></span> exp(scaled<sub>i,k</sub>)</span></span></div><p>Each row is turned into weights that add up to 1, and −∞ becomes 0. Dropout randomly zeroes some weights during training; it is switched off when the model is used.</p>`,
 `<h4>Out</h4><div class="fx">out<sub>i</sub> = <span class="sum">Σ<sub>j</sub></span> weight<sub>i,j</sub> × V<sub>j</sub></div><p>Each token's output is a blend of the Value vectors, mixed by its row of attention weights. A token it attended to strongly contributes most; a masked token contributes nothing.</p>`,
 `<h4>MLP</h4><div class="fx">MLP(x) = W<sub>2</sub> · GELU(W<sub>1</sub> · x + b<sub>1</sub>) + b<sub>2</sub></div><p>Each token's vector x (768 numbers) is widened to 3,072 by W<sub>1</sub>, passed through GELU, and narrowed back to 768 by W<sub>2</sub>. Every token uses the same weights, and the result is added back to x.</p>`,
 `<h4>Output projection</h4><div class="fx">logits = W<sub>out</sub> · h<sub>last</sub><br>p = softmax(logits ÷ T)</div><p>After the last block, the final token's vector h<sub>last</sub> (768 numbers) is multiplied by one more matrix to give a score for each of the 50,257 tokens. Temperature T and top-k or top-p then shape those scores into the probabilities on the right.</p>`,
 `<h4>Temperature</h4><div class="fx">scaled<sub>i</sub> = logit<sub>i</sub> ÷ T</div><p>Every score is divided by T before softmax. Below 1 the gaps between scores grow, so the top token wins more often and the text is more predictable. Above 1 the gaps shrink, so unlikely tokens get a real chance and the text is more varied.</p>`,
 `<h4>Top-k</h4><div class="fx">keep the k highest scores<br>all others → −∞</div><p>Only the k most likely tokens stay in the draw; the rest get probability 0. With k = 1 the model always picks its top guess.</p>`,
 `<h4>Top-p</h4><div class="fx">keep the smallest set with <span class="sum">Σ</span> p<sub>i</sub> ≥ p<br>all others → −∞</div><p>Tokens are taken from most to least likely until their probabilities add up to p. The set is small when the model is sure and large when it is not, so it adapts where top-k is fixed.</p>`,
 `<h4>Tokenization</h4><div class="fx">text → tokens → token IDs</div><p>The prompt is cut into tokens from a fixed vocabulary of 50,257 (byte-pair encoding), and each token is replaced by its ID. “empowers” becomes two tokens, “em” and “powers”.</p>`,
 `<h4>Token embedding</h4><div class="fx">e<sub>i</sub> = W<sub>E</sub>[ id<sub>i</sub> ]</div><p>The ID picks one row out of a learned table W<sub>E</sub> with 50,257 rows of 768 numbers. Tokens used in similar ways end up with similar rows.</p>`,
 `<h4>Positional encoding</h4><div class="fx">x<sub>i</sub> = W<sub>E</sub>[ id<sub>i</sub> ] + W<sub>P</sub>[ i ]</div><p>A second learned table W<sub>P</sub> has one row of 768 numbers for each position in the sentence. Adding it tells the model where the token sits. The sum x<sub>i</sub> is what enters the first block.</p>`,
 `<h4>Softmax</h4><div class="fx">p<sub>i</sub> = <span class="frac"><span>exp(s<sub>i</sub>)</span><span><span class="sum">Σ<sub>j</sub></span> exp(s<sub>j</sub>)</span></span></div><p>The scores that survive the filter are turned into probabilities that add up to 100%. A token that was filtered out (−∞) gets exactly 0%. The next token is then drawn from these.</p>`
];
function flow(){
 const {tokens,attn}=ex(),n=tokens.length,H=Y0+n*PITCH+48;
 const yc=i=>Y0+i*PITCH+PITCH/2;
 const {weights}=attentionStages(attn[state.block][state.head]);
 const list=rows(),picked=sample(list,state.rnd);
 const top=86,bottom=H-30,ch=bottom-top,fx=218,fy=48,fw=712,fh=H-fy-8;
 const gy={k:top+ch*0.2,q:top+ch*0.5,v:top+ch*0.82};
 const ty=(g,i)=>gy[g]+(i-(n-1)/2)*TP;
 // one part can be open at a time: 'embedding', 'attention' or 'probabilities'
 const X=state.expand,E=X==='attention',EE=X==='embedding',EP=X==='probabilities',mats=E?3:1,matW=n*TP,mgap=58,mx=m=>548+m*(matW+mgap),mx0=mx(0),cx=(j,m=0)=>mx(m)+j*TP,mtop=ty('q',0)-TP/2-3;
 const dA=(mats-1)*(matW+mgap),dE=EE?300:0,dP=EP?290:0,outX=716+dA,cardW=440+dA,anim=state.anim,prev=state.last||{dA:0,dE:0,dP:0},was=prev.dA;
 state.anim=null;state.last={dA,dE,dP};
 // everything that is not the open part fades into the background
 const dimEmb=E||EP?' xdim':'',dimBlock=EE||EP?' xdim':'',dimSide=X?' xdim':'',dimProb=E||EE?' xdim':'';
 const fromE=anim?prev.dE-dE:0,fromP=anim?dP-prev.dP:0,openE=anim==='open'&&EE,openP=anim==='open'&&EP;
 const late=(on,at)=>on?` late" style="--at:${at}ms`:'';
 // entrance values for the expand / collapse motion: where things come from, and how wide the frames were
 const from=anim?was-dA:0,sx=w=>anim?((w-dA+was)/w).toFixed(4):1;
 const names={q:'Query',k:'Key',v:'Value'};
 let s=`<title id="tx-flow-title">A sentence flowing through embedding, transformer block ${state.block+1} and the output probabilities</title>`;
 // column titles
 s+=`<g class="stage${dimEmb}" style="--d:0" data-zone="embedding"><g data-expand="embedding" role="button" tabindex="0" aria-label="Open the embedding step"><text class="col-title" x="20" y="30">Embedding</text></g></g>`;
 // embedding: token labels stay where they are; opened, each token shows id → token embedding + positional encoding = vector
 s+=`<g class="stage${dimEmb}" style="--d:0" data-zone="embedding">`;
 tokens.forEach((t,i)=>{s+=`<text data-t="${i}" class="tok" x="120" y="${yc(i)+4}" text-anchor="end">${esc(showToken(t))}</text>`;});
 if(EE){
  const pill=(k,x,w,label,at)=>`<g class="stage-name${late(openE,at)}" data-formula="${k}" tabindex="0" role="button" aria-label="${label}: show the formula"><rect x="${x-w/2}" y="${fy+4}" width="${w}" height="19" rx="9"/><text x="${x}" y="${fy+17}" text-anchor="middle">${label}</text></g>`;
  s+=pill(9,78,90,'Tokenization',250)+pill(10,194,106,'Token embedding',650)+pill(11,322,124,'Positional encoding',1050);
  const stripes=(x,y,vs,colour)=>vs.slice(0,10).map((v,k)=>`<rect x="${x}" y="${(y-15+k*3).toFixed(1)}" width="12" height="2.4" fill="${colour?(v>=0?'#7a9bec':'#d79278'):'#9aabc1'}" opacity="${(0.3+0.7*Math.abs(v)).toFixed(2)}"/>`).join('');
  tokens.forEach((t,i)=>{
   const y=yc(i),st=i*60;
   s+=`<g class="emb-step${late(openE,250+st)}"><path class="emb-arrow" d="M134,${y}H166m-5,-4l5,4l-5,4"/></g>`;
   s+=`<g class="emb-step${late(openE,650+st)}">${stripes(180,y,illustrativeVector(ex().tokenIds[i]),false)}${i?'':`<text class="emb-key" x="198" y="${y-6}">id</text>`}<text class="emb-num" x="198" y="${y+(i?4:8)}">${ex().tokenIds[i]}</text></g>`;
   s+=`<g class="emb-step${late(openE,1050+st)}"><text class="emb-op" x="262" y="${y+5}" text-anchor="middle">+</text>${stripes(292,y,illustrativeVector(90000+i),true)}${i?'':`<text class="emb-key" x="310" y="${y-6}">position</text>`}<text class="emb-num" x="310" y="${y+(i?4:8)}">${i}</text></g>`;
   s+=`<g class="emb-step${late(openE,1450+st)}"><text class="emb-op" x="${dE+110}" y="${y+5}" text-anchor="middle">=</text></g>`;
  });
 }
 s+=`</g>`;
 // from the embedding bars onward, everything moves right to make room when embedding is open
 s+=`<g transform="translate(${dE})"><g class="${anim?'slide':''}" style="--from:${fromE}px">`;
 s+=`<g class="stage${dimBlock}" style="--d:1"><rect class="frame${anim?' widen':''}" style="--sx:${sx(fw+dA)}" x="${fx}" y="${fy}" width="${fw+dA}" height="${fh}" rx="10"/>${chip('block',fx+18,fy,blockLabel(),176)}</g>`;
 s+=`<g class="stage${dimEmb}" style="--d:0" data-zone="embedding">`;
 tokens.forEach((t,i)=>{s+=`<rect data-t="${i}" x="130" y="${yc(i)-15}" width="12" height="30" rx="2" fill="#cdd6e2"/>`;});
 s+=`</g><g class="stage block-part${dimSide}" style="--d:1">`;
 tokens.forEach((t,i)=>{
  const y=yc(i);
  [['q',-12],['k',0],['v',12]].forEach(([g,dy])=>{
   s+=`<g data-t="${i}"><path class="fg" d="${curve(142,y,232,y+dy)}" stroke-width="1.2"/><rect class="${g}" x="232" y="${y+dy-4.5}" width="9" height="9" rx="1.5" opacity=".75"/><text class="mini" x="246" y="${y+dy+3}">${g.toUpperCase()}</text></g>`;
  });
 });
 s+=`</g>`;
 // head card with a stack behind it
 s+=`<g class="stage head-part${dimBlock}" style="--d:2" data-zone="attention">`;
 for(let d=3;d>=1;d--)s+=`<rect class="card back${anim?' widen':''}" style="--sx:${sx(cardW)}" x="${330+d*5}" y="${top+d*5}" width="${cardW}" height="${ch}" rx="6"/>`;
 s+=`<rect class="card front${anim?' widen':''}" style="--sx:${sx(cardW)}" x="330" y="${top}" width="${cardW}" height="${ch}" rx="6"/>`;
 for(const g of ['k','q','v']){
  s+=`<text class="lbl-${g}" x="452" y="${ty(g,0)-12}">${names[g]}</text>`;
  tokens.forEach((t,i)=>{
   const y=ty(g,i),dy=g==='q'?-12:g==='k'?0:12;
   s+=`<g data-t="${i}"><path class="f${g}" d="${curve(254,yc(i)+dy,380,y)}" stroke-width="1" opacity=".55"/><text class="mini" x="446" y="${y+3}" text-anchor="end">${esc(showToken(t))}</text><rect class="${g}" x="452" y="${y-1.5}" width="12" height="3" rx="1"/></g>`;
  });
 }
 s+=`</g><g class="stage head-part${dimBlock}" style="--d:3" data-zone="attention">`;
 // key lines down into matrix columns, query lines into rows
 // Opening timeline, after Transformer Explainer: the Key and Query lines draw in one token after another and the
 // dot-product dots pop in as they meet; then each next stage slides out showing the previous picture and slowly turns into its own.
 const opening=anim==='open',QK=1200,t0=300,T1=t0+QK/n+QK+500,T2=T1+500+1000+150,T3=T2+450+1000;
 const draw=k=>opening?` pathLength="1" class="draw" style="--at:${Math.round(t0+k*QK/n)}ms"`:'';
 const look=(m,i,j)=>{const v=stageShade(m,i,j);return v===null?['#fff','1','#e2e8f0']:['#7f58a3',dotOpacity(v),'transparent'];};
 tokens.forEach((t,j)=>{s+=`<g data-t="${j}" class="fk" opacity=".5"><path${draw(j)} d="M464,${ty('k',j)} H${cx(j)} V${mtop}" stroke-width="1"/></g>`;});
 tokens.forEach((t,i)=>{s+=`<g data-t="${i}" class="fq" opacity=".6"><path${draw(i)} d="M464,${ty('q',i)} H${mx0-TP/2-2}" stroke-width="1"/></g>`;});
 // attention: one matrix of weights, or its three stages when expanded
 const stages=E?[0,1,2]:[2];
 stages.forEach((m,k)=>{
  s+=`<g class="${opening&&k?'grow':''}" style="--from:${-(matW+mgap)}px;--at:${[0,T1,T2][k]}ms">`;
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
   const v=stageShade(m,i,j);let fx='';
   if(opening&&k===0)fx=` pop" style="--at:${Math.round(t0+QK/n+(i*n+j)*QK/(n*n))}ms`;
   else if(opening){const a=look(m-1,i,j),b=look(m,i,j);fx=` turn" style="--at:${[0,T1+500,T2+450][k]}ms;--f0:${a[0]};--o0:${a[1]};--s0:${a[2]};--f1:${b[0]};--o1:${b[1]};--s1:${b[2]}`;}
   s+=`<circle class="cell${v===null?' masked':''}${fx}" data-t="${i}" data-q="${i}" data-k="${j}" data-m="${m}" cx="${cx(j,k)}" cy="${ty('q',i)}" r="5"${v===null?'':` fill="#7f58a3" fill-opacity="${dotOpacity(v)}"`}/>`;
  }
  if(E){
   const mid=mx(k)+matW/2-TP/2,by=ty('q',n-1)+TP/2,[lo,hi]=stageRange(m);
   s+=`<g class="stage-name" data-formula="${m}" tabindex="0" role="button" aria-label="${STAGE_NAMES[m]}: show the formula"><rect x="${mid-52}" y="${by+8}" width="104" height="19" rx="9"/><text x="${mid}" y="${by+21}" text-anchor="middle">${STAGE_NAMES[m]}</text></g>`;
   s+=`<text class="leg leg-lo" data-m="${m}" x="${mid-20}" y="${by+41}" text-anchor="end">${lo}</text><rect class="leg-bar" x="${mid-16}" y="${by+35}" width="32" height="4" rx="2"/><text class="leg leg-hi" data-m="${m}" x="${mid+20}" y="${by+41}">${hi}</text>`;
   if(k<2)s+=`<text class="stage-arrow" x="${mx(k)+matW-TP/2+mgap/2}" y="${ty('q',(n-1)/2)+4}" text-anchor="middle">→</text>`;
  }
  s+=`</g>`;
 });
 if(!E)s+=`<g class="attn-label" data-expand="attention" role="button" tabindex="0" aria-label="Open attention into its three stages"><text class="col-sub" x="${mx0-6}" y="${ty('q',n-1)+TP/2+16}">ATTENTION</text></g>`;
 // out: weighted mix of values
 const mr=cx(n-1,mats-1)+8;
 s+=`<g class="${anim?'slide':''}" style="--from:${from}px"><path class="band${opening?' late':''}" style="--at:${T3}ms;fill:#e6dcf0" d="M${mr},${ty('q',0)-6} L${outX},${ty('q',0)-2} V${ty('q',n-1)+2} L${mr},${ty('q',n-1)+6} Z"/>`;
 s+=`<g class="stage-name out-name" data-formula="3" tabindex="0" role="button" aria-label="Out: show the formula"><rect x="${outX-13}" y="${ty('q',0)-27}" width="40" height="19" rx="9"/><text x="${outX+7}" y="${ty('q',0)-14}" text-anchor="middle">Out</text></g>`;
 tokens.forEach((t,i)=>{
  s+=`<g data-t="${i}">${E?`<path class="fv${opening?' late':''}" style="--at:${T3}ms" d="M464,${ty('v',i)} H${outX-74} C${outX-30},${ty('v',i)} ${outX-44},${ty('q',i)} ${outX},${ty('q',i)}" stroke-width="1" opacity=".45"/>`:`<path class="fv" d="${curve(464,ty('v',i),outX,ty('q',i))}" stroke-width="1" opacity=".4"/>`}<rect class="o" x="${outX}" y="${ty('q',i)-1.5}" width="14" height="3" rx="1"/></g>`;
 });
 s+=`</g></g>`;
 // MLP and the remaining blocks
 s+=`<g class="stage${dimBlock}" style="--d:2">${chip('head',346,top,headLabel(),282)}</g>`;
 s+=`<g transform="translate(${dA})" style="--oxL:${fx-dA}px;--oxR:${fx+fw}px"><g class="${anim?'slide':''}" style="--from:${from}px"><g class="${dimSide}">`;
 s+=`<g class="stage" style="--d:4"><g class="stage-name mlp-name" data-formula="4" tabindex="0" role="button" aria-label="MLP: show the formula"><rect x="${806-9}" y="${fy+21}" width="42" height="19" rx="9"/><text x="${806+12}" y="${fy+34}" text-anchor="middle">MLP</text></g><text class="col-sub" x="944" y="46">11 MORE</text><text class="col-sub" x="944" y="58">BLOCKS</text></g>`;
 s+=`<g class="stage block-part" style="--d:4">`;
 tokens.forEach((t,i)=>{
  const y=yc(i);
  s+=`<g data-t="${i}"><path class="fo" d="${curve(716+14,ty('q',i),806,y)}" stroke-width="1.2" opacity=".7"/>`;
  s+=`<rect x="806" y="${y-15}" width="10" height="30" rx="2" fill="#d0bfe1"/><path class="band" style="fill:#d3dcee" d="M816,${y-15} L846,${y-22} V${y+22} L816,${y+15} Z"/><rect x="846" y="${y-22}" width="26" height="44" rx="3" fill="#b3c2e0"/><path class="band" style="fill:#d3dcee" d="M872,${y-22} L902,${y-15} V${y+15} L872,${y+22} Z"/><rect x="902" y="${y-15}" width="10" height="30" rx="2" fill="#a9c0f3"/>`;
  s+=`<path class="fg" d="M912,${y} H944" stroke-width="1.2"/>`;
  for(let b=0;b<3;b++)s+=`<rect x="${944+b*12}" y="${y-15}" width="7" height="30" rx="2" fill="#d3e0fa" opacity="${0.9-b*0.25}"/>`;
  s+=`</g>`;
 });
 s+=`</g>`;
 // probabilities for the last token: drawn once, then updated in place by refreshProbs().
 // Opened, the list becomes a table: logit → ÷ temperature → top-k / top-p → softmax.
 const shown=list.slice(0,10),py=r=>(EP?102:84)+r*30,last=n-1,bx=EP?1372:1132,cut=state.type==='top-k'?'Top-k':'Top-p';
 s+=`</g><g transform="translate(${-dP})"><g class="${anim?'slide':''}${dimProb}" style="--from:${fromP}px" data-zone="probabilities">`;
 s+=`<g class="stage" style="--d:6"><g data-expand="probabilities" role="button" tabindex="0" aria-label="Open the output probabilities step"><text class="col-title" x="1052" y="30">Probabilities</text></g></g>`;
 s+=`<g class="stage prob-part" style="--d:6">`;
 if(EP)s+=`<rect class="prob-bg" x="1030" y="42" width="470" height="${py(9)-18}" rx="8"/>`;
 shown.forEach(r=>{s+=`<path class="fo pc" data-rank="${r.rank}" d="${curve(980,yc(last),1040,py(r.rank)-4)}"/>`;});
 s+=`<g class="stage-name fan-hit" data-formula="5" tabindex="0" role="button" aria-label="Output projection: show the formula"><path d="M974,${yc(last)-12} L1044,${py(0)-16} V${py(9)+8} L974,${yc(last)+12} Z"/></g>`;
 if(EP){
  const head=(k,x,w,label,at,cls='')=>`<g class="stage-name${cls}${late(openP,at)}" data-formula="${k}" tabindex="0" role="button" aria-label="${label}: show the formula"><rect x="${x-w/2}" y="${fy+4}" width="${w}" height="19" rx="9"/><text x="${x}" y="${fy+17}" text-anchor="middle">${label}</text></g>`;
  s+=`<text class="col-sub" x="1124" y="${fy+17}" text-anchor="end">TOKENS</text>`+head(5,1175,58,'Logits',350)+head(6,1250,84,'÷ temperature',800)+head(state.type==='top-k'?7:8,1325,58,cut,1250,' pk-head')+head(12,1410,68,'Softmax',1700);
 }
 shown.forEach(r=>{
  s+=`<g class="prob" data-rank="${r.rank}"><rect class="prob-hit" x="1034" y="${py(r.rank)-19}" width="${EP?462:196}" height="28"/><text class="pt" x="1124" y="${py(r.rank)}" text-anchor="end">${esc(showToken(r.token))}</text>`;
  if(EP)s+=`<g class="num${late(openP,350+r.rank*40)}"><rect class="nbg" x="1146" y="${py(r.rank)-16}" width="58" height="22" rx="3"/><text class="pl" x="1175" y="${py(r.rank)}" text-anchor="middle">${r.logit.toFixed(2)}</text></g><g class="num${late(openP,800+r.rank*40)}"><rect class="nbg" x="1221" y="${py(r.rank)-16}" width="58" height="22" rx="3"/><text class="ps" x="1250" y="${py(r.rank)}" text-anchor="middle"></text></g><g class="num${late(openP,1250+r.rank*40)}"><rect class="nbg" x="1296" y="${py(r.rank)-16}" width="58" height="22" rx="3"/><text class="pk" x="1325" y="${py(r.rank)}" text-anchor="middle"></text></g>`;
  s+=`<g class="${EP?`bar${late(openP,1700+r.rank*40)}`:'bar'}"><rect class="pb" x="${bx}" y="${py(r.rank)-6}" width="56" height="4" rx="2"/><text class="pp" x="${bx+6}" y="${py(r.rank)}"></text></g></g>`;
 });
 s+=`</g></g></g></g></g></g></g>`;
 const svg=$('#tx-flow');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=s;svg.classList.toggle('expanded',!!X);svg.dataset.open=X||'';
 // the block frame's left and right edges: a block change contracts the old block into one edge and grows the new one out of the other
 svg.style.setProperty('--oxL',`${fx}px`);svg.style.setProperty('--oxR',`${fx+fw+dA}px`);svg.classList.remove('tracing');
 $('#tx-tip').hidden=true;$('#tx-formula').hidden=true;
 paintProbs(list,picked);
 return picked;
}
// Temperature, top-k / top-p and Generate only touch the last step, so only that is repainted.
function paintProbs(list=rows(),picked=sample(list,state.rnd)){
 const svg=$('#tx-flow');
 for(const r of list.slice(0,10)){
  const bw=Math.max(2,r.probability*56),g=svg.querySelector(`.prob[data-rank="${r.rank}"]`),c=svg.querySelector(`.pc[data-rank="${r.rank}"]`);
  g.setAttribute('class',`prob${r.kept?' kept':''}${r===picked?' sampled':''}`);
  g.querySelector('.pb').style.transform=`scaleX(${(bw/56).toFixed(4)})`;
  const pp=g.querySelector('.pp');pp.textContent=pct(r.probability);pp.style.transform=`translateX(${bw.toFixed(1)}px)`;
  const ps=g.querySelector('.ps');if(ps){ps.textContent=r.scaled.toFixed(2);g.querySelector('.pk').textContent=r.kept?r.scaled.toFixed(2):'−∞';}
  c.style.strokeWidth=(0.6+r.probability*7).toFixed(2);c.style.opacity=r.kept?(r===picked?0.95:0.45):0;
 }
 const kh=svg.querySelector('.pk-head');if(kh){kh.dataset.formula=state.type==='top-k'?7:8;kh.querySelector('text').textContent=state.type==='top-k'?'Top-k':'Top-p';}
 $('#tx-sentence').innerHTML=`${esc(ex().prompt)} <b>${esc(showToken(picked.token))}</b>`;
 defaultResult(picked);
 return picked;
}
function refreshProbs(){state.sel=null;syncControls();probTable(paintProbs());}
function defaultResult(picked=sample(rows(),state.rnd)){
 const kept=rows().filter(r=>r.kept).length;
 $('#tx-result').textContent=`Next token drawn: “${showToken(picked.token)}” (${pct(picked.probability)}), from ${kept} candidate${kept===1?'':'s'} left after ${state.type==='top-k'?`top-k (k=${state.k})`:`top-p (p=${state.p.toFixed(2)})`} at temperature ${state.temperature.toFixed(1)}.`;
}

/* ---------- 01 embedding ---------- */
const vec=vs=>`<span class="vec" aria-hidden="true">${vs.map(v=>`<i style="background:${v>=0?`rgba(79,123,232,${(0.12+0.8*v).toFixed(2)})`:`rgba(199,118,91,${(0.12-0.8*v).toFixed(2)})`}"></i>`).join('')}<i class="more">… 768</i></span>`;
function embedding(){
 if(!$('#emb-tokens'))return;   // the Embedding section is now the original article's text and figure
 const {tokens,tokenIds}=ex();if(state.tok>=tokens.length)state.tok=0;
 $('#emb-tokens').innerHTML=tokens.map((t,i)=>`<button type="button" data-tok="${i}" aria-pressed="${i===state.tok}">${esc(showToken(t))}</button>`).join('');
 const i=state.tok,id=tokenIds[i],te=illustrativeVector(id),pe=illustrativeVector(90000+i),sum=te.map((v,j)=>Math.max(-1,Math.min(1,(v+pe[j])/1.4)));
 $('#emb-flow').innerHTML=`<span class="what">Token</span><span class="val">“${esc(showToken(tokens[i]))}”</span><span></span>
<span class="what">Token ID</span><span class="val"><code>${id.toLocaleString('en-US')}</code> of 50,257</span><span></span>
<span class="what">Token embedding</span><span class="val">row ${id.toLocaleString('en-US')} of a learned table</span>${vec(te)}
<span class="op">+</span>
<span class="what">Position embedding</span><span class="val">position ${i+1} in the sentence</span>${vec(pe)}
<span class="op">=</span>
<span class="what">Input to block 1</span><span class="val">one vector per token</span>${vec(sum)}`;
}

/* ---------- 02 attention ---------- */
// The three grids are computed once, then revealed as a short sequence so the reader sees
// each operation happen: multiply, divide, mask, normalise. playAttention() runs the sequence.
let attnTicket=0,attnBusy=false;
function attention({play=false}={}){
 const {tokens,attn}=ex(),n=tokens.length,raw=attn[state.block][state.head],{scaled,weights}=attentionStages(raw);
 const maxRaw=Math.max(...raw.flat().map(Math.abs))||1,maxSc=Math.max(...scaled.flat().map(Math.abs))||1;
 const tint=(v,m)=>v>=0?`rgba(79,123,232,${(0.06+0.5*v/m).toFixed(2)})`:`rgba(199,118,91,${(0.06-0.5*v/m).toFixed(2)})`;
 const mat=(cell)=>{
  let h=`<div class="mat" style="grid-template-columns:minmax(52px,74px) repeat(${n},minmax(0,1fr))"><span></span>${tokens.map((t,j)=>`<span class="cl" data-k="${j}">${esc(showToken(t))}</span>`).join('')}`;
  for(let i=0;i<n;i++){h+=`<span class="rl" data-q="${i}">${esc(showToken(tokens[i]))}</span>`;for(let j=0;j<n;j++)h+=cell(i,j);}
  return h+'</div>';
 };
 const c=(i,j,text,bg,cls='',order=i*n+j)=>`<span class="c${cls}" data-q="${i}" data-k="${j}" style="--i:${order};--bg:${bg}">${text}</span>`;
 const stages=[
  ['1 · Dot product <span class="fm">Q · K<sup>T</sup></span>','Query of the row token · Key of the column token.',mat((i,j)=>c(i,j,raw[i][j].toFixed(1),tint(raw[i][j],maxRaw)))],
  // a future token first shows its real scaled score, then the mask turns it into −∞
  [`2 · Scale and mask <span class="fm">Q · K<sup>T</sup> ÷ √d<sub>k</sub>, &nbsp;j &gt; i → −∞</span>`,`Divide by √${HEAD_DIM} = 8 (d<sub>k</sub> = ${HEAD_DIM}). Later tokens are set to −∞.`,mat((i,j)=>j>i?c(i,j,`<b class="was">${scaled[i][j].toFixed(2)}</b><b class="now">−∞</b>`,tint(scaled[i][j],maxSc),' masked'):c(i,j,scaled[i][j].toFixed(2),tint(scaled[i][j],maxSc)))],
  // softmax is revealed one row at a time, because each row is normalised on its own
  ['3 · Softmax <span class="fm">e<sup>x</sup> ÷ Σ e<sup>x</sup> &nbsp;per row</span>','Each row becomes weights that sum to 100%.',mat((i,j)=>j>i?c(i,j,'0','#fff',' masked done',i*4):c(i,j,Math.round(weights[i][j]*100)+'%',`rgba(127,88,163,${(0.05+0.7*weights[i][j]).toFixed(2)})`,'',i*4+j*0.6))]
 ];
 $('#attn-stages').innerHTML=stages.map(([t,p,m],k)=>`<div class="attn-stage" data-stage="${k+1}"><h3>${t}</h3><p>${p}</p>${m}</div>`).join('');
 playAttention(play);
}
const ATTN_STEPS=[
 [0,1,'show','Step 1 · Multiply. Each cell is one Query (its row) times one Key (its column). All pairs are scored, even with tokens that come later.'],
 [1500,2,'show','Step 2 · Scale. Every score is divided by √64 = 8, which keeps the numbers in a range softmax handles well.'],
 [2900,2,'mask','Then mask. Scores for later tokens (above the diagonal) are replaced by −∞, so a token cannot peek ahead.'],
 [4300,3,'show','Step 3 · Softmax, one row at a time. Each row becomes weights that add up to 100%; a −∞ score becomes exactly 0.'],
];
function playAttention(animate){
 const ticket=++attnTicket,box=$('#attn-stages'),stage=k=>box.querySelector(`[data-stage="${k}"]`),btn=$('#attn-play');
 const calm=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const finish=()=>{attnBusy=false;box.classList.remove('playing');for(const k of [1,2,3])stage(k).classList.add('show');stage(2).classList.add('mask');if(btn)btn.textContent='↻ Replay the steps';const n=ex().tokens.length,w=attentionStages(ex().attn[state.block][state.head]).weights[n-1];attnResult(n-1,w.indexOf(Math.max(...w)));};
 if(!animate||calm){box.classList.add('instant');finish();return;}
 attnBusy=true;box.classList.remove('instant');box.classList.add('playing');if(btn)btn.textContent='Playing…';
 $$('#attn-stages .c').forEach(el=>el.classList.remove('on','row'));
 for(const [at,k,cls,text] of ATTN_STEPS)setTimeout(()=>{if(ticket!==attnTicket)return;stage(k).classList.add(cls);$('#attn-result').textContent=text;},at+30);
 setTimeout(()=>{if(ticket===attnTicket)finish();},5900);
}
function attnResult(i,j){
 const {tokens,attn}=ex(),raw=attn[state.block][state.head],{scaled,weights}=attentionStages(raw);
 $$('#attn-stages .c').forEach(el=>{el.classList.toggle('on',+el.dataset.q===i&&+el.dataset.k===j);el.classList.toggle('row',+el.dataset.q===i);});
 const q=showToken(tokens[i]),k=showToken(tokens[j]);
 $('#attn-result').textContent=j>i
  ?`“${q}” cannot look at “${k}”: it comes later in the sentence, so its score is masked to −∞ and its weight is 0.`
  :`“${q}” looking at “${k}”: dot product ${raw[i][j].toFixed(1)} → ÷8 = ${scaled[i][j].toFixed(2)} → softmax over the row = ${pct(weights[i][j])} of its attention (block ${state.block+1}, head ${state.head+1}).`;
}

/* ---------- 04 probabilities ---------- */
function probTable(picked){
 const list=rows(),shown=list.slice(0,12),sel=state.sel??picked.rank;
 const cut=state.type==='top-k'?'Top-k':'Top-p';
 let h=`<span class="h" style="text-align:right">Token</span><span class="h">Logit</span><span class="h">÷ ${state.temperature.toFixed(1)}</span><span class="h">${cut}</span><span class="h">Softmax</span>`;
 for(const r of shown){
  const c=`${r.kept?' row-kept':''}${r.rank===sel?' row-sel':''}`;
  h+=`<button type="button" class="tk${c}" data-rank="${r.rank}">${esc(showToken(r.token))}</button><span class="n${c}">${r.logit.toFixed(2)}</span><span class="n${c}">${r.scaled.toFixed(2)}</span><span class="n${c}">${r.kept?r.scaled.toFixed(2):'−∞'}</span><span class="bar${c}"><i style="width:${Math.max(2,r.probability*120).toFixed(0)}px"></i>${pct(r.probability)}</span>`;
 }
 $('#prob-table').innerHTML=h;
 $('#prob-card').innerHTML=cardHtml(list[Math.min(sel,list.length-1)],list,picked);
}
// One token worked through the last three steps: divide by temperature, filter, softmax.
function cardHtml(r,list,picked){
 const kept=list.filter(x=>x.kept),best=kept[0].scaled,cut=state.type==='top-k'?'Top-k':'Top-p';
 const terms=kept.slice(0,4).map(x=>`exp(${(x.scaled-best).toFixed(2)})`).join(' + ')+(kept.length>4?' + …':'');
 return `<h3>Chance of “<b>${esc(showToken(r.token))}</b>” being drawn${r===picked?' · drawn this time':''}</h3><ol>
<li><span>1 · Temperature</span><code>${r.logit.toFixed(2)} ÷ ${state.temperature.toFixed(1)} = <b>${r.scaled.toFixed(2)}</b></code></li>
<li><span>2 · ${cut} filter</span><code>${r.kept?`kept (rank ${r.rank+1} of ${kept.length} kept)`:`dropped → −∞ (rank ${r.rank+1}, only ${kept.length} kept)`}</code></li>
<li><span>3 · Softmax, relative to the best score</span><code>${r.kept?`exp(${(r.scaled-best).toFixed(2)}) ÷ (${terms}) = <b>${pct(r.probability)}</b>`:`<b>0%</b>`}</code></li></ol>`;
}

/* Changing block or head redraws only that part: the dots fade to their new
   weights and the head card (plus the block's own columns, for a block change)
   replays a short entrance. Embedding and probabilities stay still. */
const onScreen=el=>{const r=el.getBoundingClientRect();return r.top<innerHeight*0.85&&r.bottom>innerHeight*0.15;};
function swap(kind,dir){
 const svg=$('#tx-flow'),calm=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const sel=kind==='block'?'.head-part,.block-part':'.head-part',parts=[...svg.querySelectorAll(sel)].filter(g=>!g.classList.contains('ghost'));
 svg.querySelectorAll('.ghost').forEach(g=>g.remove());
 // the outgoing block/head leaves as a copy, so the old picture is still visible while the new one arrives
 const ghosts=calm?[]:parts.map(g=>{const c=g.cloneNode(true);c.classList.remove('swap','stage');c.classList.add('ghost');c.setAttribute('aria-hidden','true');g.before(c);return c;});
 parts.forEach(g=>g.querySelectorAll('.cell:not(.masked)').forEach(c=>c.setAttribute('fill-opacity',dotOpacity(stageShade(+c.dataset.m,+c.dataset.q,+c.dataset.k)))));
 parts.forEach(g=>g.querySelectorAll('.leg').forEach(t=>{t.textContent=stageRange(+t.dataset.m)[t.classList.contains('leg-lo')?0:1];}));
 svg.querySelector('.block-label').textContent=blockLabel();
 svg.querySelector('.head-label').textContent=headLabel();
 svg.querySelector('title').textContent=`A sentence flowing through embedding, transformer block ${state.block+1} and the output probabilities`;
 syncSteppers();attention({play:onScreen($('#attn-stages'))});defaultResult();
 const moving=[...parts,svg.querySelector(`.${kind}-label`),svg.querySelector(kind==='block'?'.frame':'.card.front')];
 svg.dataset.swap=kind;svg.dataset.dir=dir>0?'next':'prev';svg.style.setProperty('--dir',dir);
 moving.forEach(g=>g.classList.remove('swap'));void svg.getBoundingClientRect();moving.forEach(g=>g.classList.add('swap'));
 setTimeout(()=>ghosts.forEach(g=>g.remove()),900);
}

// Open or close the three stages inside the head card. Only the card and what sits to its right move.
function setExpand(which){
 if(state.expand===which)return;
 state.expand=which;state.anim=matchMedia('(prefers-reduced-motion: reduce)').matches?null:(which?'open':'close');
 $('#tx-lab').classList.remove('fresh');$('#tx-flow').querySelectorAll('.ghost').forEach(g=>g.remove());
 flow();
}

/* ---------- wiring ---------- */
function render({fresh=false,keepSel=false}={}){
 if(!keepSel)state.sel=null;
 syncControls();syncSteppers();
 const picked=flow();embedding();attention();probTable(picked);
 if(fresh){const lab=$('#tx-lab');lab.classList.remove('fresh');void lab.offsetWidth;lab.classList.add('fresh');}
}
function init(){
 $$('[data-controls]').forEach((el,i)=>{el.innerHTML=controlsMarkup(i);});
 $('#tx-example').innerHTML=EXAMPLES.map((e,i)=>`<option value="${i}">${esc(e.prompt)} …</option>`).join('');
 $('#tx-example').addEventListener('change',e=>{state.ex=+e.target.value;state.rnd=0.3;state.tok=0;state.expand=null;state.last=null;render({fresh:true});});
 $('#tx-generate').addEventListener('click',()=>{state.rnd=Math.random();refreshProbs();});
 document.addEventListener('input',e=>{
  const c=e.target.dataset?.c;if(!c)return;
  if(c==='temperature')state.temperature=+e.target.value;
  else if(c==='type'){if(e.target.checked)state.type=e.target.value;}
  else if(c==='amount'){if(state.type==='top-k')state.k=Math.round(+e.target.value);else state.p=+e.target.value;}
  refreshProbs();
 });
 document.addEventListener('click',e=>{
  const step=e.target.closest('[data-step]');
  if(step){const kind=step.closest('[data-stepper]').dataset.stepper,max=kind==='block'?BLOCKS:HEADS;const dir=+step.dataset.step;state[kind]=(state[kind]+dir+max)%max;swap(kind,dir);return;}
  // a click on the embedding column, the attention head or the probabilities opens that part; a click anywhere else in the diagram closes it
  if(e.target.closest('#tx-flow')){
   const zoneEl=e.target.closest('[data-zone]'),zone=zoneEl&&!e.target.closest('.ghost')?zoneEl.dataset.zone:null;
   if(!state.expand&&zone){setExpand(zone);return;}
   if(state.expand&&zone!==state.expand&&!e.target.closest('.chip')){setExpand(null);return;}
  }
  const tok=e.target.closest('[data-tok]');if(tok){state.tok=+tok.dataset.tok;embedding();return;}
  const rank=e.target.closest('[data-rank]');if(rank){state.sel=+rank.dataset.rank;probTable(sample(rows(),state.rnd));}
 });
 document.addEventListener('keydown',e=>{
  if(e.key==='Escape'&&state.expand){setExpand(null);return;}
  const b=(e.key==='Enter'||e.key===' ')&&e.target.closest?.('#tx-flow .step,#tx-flow [data-expand]');
  if(b){e.preventDefault();b.dispatchEvent(new MouseEvent('click',{bubbles:true}));}
 });
 const svg=$('#tx-flow');
 svg.addEventListener('mouseover',e=>{
  // row tracing is for the overview only; with the three stages open, hovering explains one dot and nothing else moves
  if(state.expand)return;
  const el=e.target.closest('[data-t]');if(!el)return;
  const t=el.dataset.t;svg.classList.add('tracing');
  svg.querySelectorAll('[data-t]').forEach(x=>x.classList.toggle('on',x.dataset.t===t));
  const cell=e.target.closest('.cell');
  if(cell){
   const i=+cell.dataset.q,j=+cell.dataset.k,{tokens,attn}=ex(),w=attentionStages(attn[state.block][state.head]).weights[i][j];
   $('#tx-result').textContent=j>i?`“${showToken(tokens[i])}” cannot look ahead at “${showToken(tokens[j])}”: masked.`:`“${showToken(tokens[i])}” gives ${pct(w)} of its attention to “${showToken(tokens[j])}” in block ${state.block+1}, head ${state.head+1}.`;
  }
 });
 const tip=$('#tx-tip'),formula=$('#tx-formula');
 // popovers are fixed to the viewport so the scrolling diagram cannot clip them; they flip above when there is no room below
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
  const row=e.target.closest('.prob');
  if(row&&state.expand==='probabilities'&&!name){
   const list=rows();formula.innerHTML=`<div class="prob-card bare">${cardHtml(list[+row.dataset.rank],list,sample(list,state.rnd))}</div>`;formula.hidden=false;
   const r=row.getBoundingClientRect(),w=formula.offsetWidth,h=formula.offsetHeight;formula.style.left=Math.max(8,r.left-w-14)+'px';formula.style.top=Math.max(8,Math.min(innerHeight-h-8,r.top-h/2+14))+'px';
  }
  if(cell&&state.expand==='attention'){
   const i=+cell.dataset.q,j=+cell.dataset.k,m=+cell.dataset.m,{tokens,attn}=ex(),raw=attn[state.block][state.head],{scaled,weights}=attentionStages(raw);
   const q=esc(showToken(tokens[i])),k=esc(showToken(tokens[j]));
   tip.innerHTML=m===0?`<b>${q} · ${k}</b>Query · Key = ${raw[i][j].toFixed(2)}`:m===1?`<b>${q} → ${k}</b>${j>i?'masked: a later token, −∞':`${raw[i][j].toFixed(2)} ÷ √${HEAD_DIM} = ${scaled[i][j].toFixed(2)}`}`:`<b>${q} → ${k}</b>${j>i?'masked: weight 0':`attention weight ${pct(weights[i][j])}`}`;
   cell.classList.add('hot');near(tip,cell,false);
  }else tip.hidden=true;
 });
 svg.addEventListener('focusin',e=>{const name=e.target.closest('.stage-name');if(name)showFormula(name);});
 // the same formula cards for the sampling controls, in the explorer bar and in the probabilities section
 for(const type of ['mouseover','focusin'])document.addEventListener(type,e=>{const el=e.target.closest?.('.tx-controls .has-formula');if(el)showFormula(el);});
 for(const type of ['mouseout','focusout'])document.addEventListener(type,e=>{if(e.target.closest?.('.tx-controls .has-formula'))formula.hidden=true;});
 svg.addEventListener('focusout',()=>{formula.hidden=true;});
 svg.addEventListener('mouseleave',()=>{svg.classList.remove('tracing');svg.querySelectorAll('.on').forEach(x=>x.classList.remove('on'));tip.hidden=true;formula.hidden=true;svg.querySelectorAll('.cell.hot').forEach(c=>c.classList.remove('hot'));defaultResult();});
 $('#attn-stages').addEventListener('mouseover',e=>{const c=e.target.closest('.c');if(c&&!attnBusy)attnResult(+c.dataset.q,+c.dataset.k);});
 $('#attn-play').addEventListener('click',()=>attention({play:true}));
 // play the steps the first time the grids scroll into view
 new IntersectionObserver((entries,io)=>{if(entries.some(e=>e.isIntersecting)){io.disconnect();attention({play:true});}},{threshold:0.35}).observe($('#attn-stages'));
 render({fresh:true});
}
init();
