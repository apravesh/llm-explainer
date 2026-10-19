import {compactNextMarkup} from './readout.mjs';
import { mountMultiHead } from './multi-head.mjs';
import { mountAttentionFlow } from './attention-flow.mjs';
import { createArchitectureMotion } from './motion.mjs';
import { expansionMarkup, nextRailPatch, railScrollOffset, STAGE_NAMES } from './expansion.mjs';
import { DIM, HEADS, DEPTH, HEAD_DIM, patchify, imageShape, runEncoder, dot, rotatePair } from './engine.mjs';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const state={clsStep:0,decoderStep:0,projectorDetail:false,headDetailOpen:false,compareCausal:false,attentionInspect:null,expanded:null,scalar:408,dimension:0,qkv:'q',ffn:'up',adapter:'first',model:'vit',patch:16,layer:0,head:0,selected:21,key:22,step:3,sample:'bird',ropeMode:'2d',width:96,height:96,cols:6,rows:6};
let image, patches=[], result, imageURL='', uploadURL=null, loadTicket=0;
const surface=document.createElement('canvas'),ctx=surface.getContext('2d',{willReadFrequently:true});
const fmt=(x,n=3)=>Number(x).toFixed(n); const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function diverging(v){const t=Math.min(Math.abs(v)/2,1),a=v>=0?[107,143,207]:[204,143,115];return `rgb(${a.map(x=>Math.round(245+(x-245)*t)).join(',')})`;}
function heat(v,max){const t=Math.pow(Math.min(v/(max||1),1),.7);return `rgb(${[54,103,228].map((x,i)=>Math.round([245,248,254][i]+(x-[245,248,254][i])*t)).join(',')})`;}
function tokenIndex(){return state.model==='vit'?state.selected+1:Math.max(0,state.selected);}
function tokenLabel(i){if(state.model==='vit'){if(i===0)return '[CLS]';i--;}const p=patches[i];return p?`(${p.row}, ${p.col})`:'—';}
function currentPatch(){return patches[Math.max(0,state.selected)];}
function tokenCount(){return patches.length+(state.model==='vit'?1:0);}
function currentLayer(){return result.layers[state.layer];}
function currentHead(){return currentLayer().heads[state.head];}
function vectorRow(values){return `<div class="vector-row">${values.map((v,i)=>`<i style="background:${diverging(v)}" title="dimension ${i}: ${fmt(v,5)}"></i>`).join('')}</div>`;}
function stack(values,wide=false){return `<div class="vector-stack ${wide?'wide':''}">${values.map(v=>`<i style="background:${diverging(v)}"></i>`).join('')}</div>`;}
function cropStyle(p,size){const x=p.col*state.patch,y=p.row*state.patch;return `background-image:url('${imageURL}');background-size:${state.width/state.patch*size}px ${state.height/state.patch*size}px;background-position:-${x/state.patch*size}px -${y/state.patch*size}px;`;}
async function loadImage(src){const ticket=++loadTicket; const next=new Image();next.decoding='async';await new Promise((resolve,reject)=>{next.onload=resolve;next.onerror=()=>reject(Error('This image could not be decoded. Try a PNG, JPEG, or WebP file.'));next.src=src;});if(ticket!==loadTicket)return;image=next;recompute();}
function recompute(){
 if(!image)return;
 architectureMotion.finish();
 const shape=imageShape(image.naturalWidth,image.naturalHeight,state.patch,state.model);state.width=shape.width;state.height=shape.height;state.cols=shape.width/state.patch;state.rows=shape.height/state.patch;
 surface.width=shape.width;surface.height=shape.height;ctx.clearRect(0,0,shape.width,shape.height);ctx.fillStyle='#f4f2e9';ctx.fillRect(0,0,shape.width,shape.height);
 if(state.model==='vit'){const size=Math.min(image.naturalWidth,image.naturalHeight);ctx.drawImage(image,(image.naturalWidth-size)/2,(image.naturalHeight-size)/2,size,size,0,0,96,96);}else{const scale=128/Math.max(image.naturalWidth,image.naturalHeight),w=Math.max(1,Math.round(image.naturalWidth*scale)),h=Math.max(1,Math.round(image.naturalHeight*scale));ctx.drawImage(image,0,0,w,h);if(w<shape.width)ctx.drawImage(surface,w-1,0,1,h,w,0,shape.width-w,h);if(h<shape.height)ctx.drawImage(surface,0,h-1,shape.width,1,0,h,shape.width,shape.height-h);}
 const rgba=ctx.getImageData(0,0,shape.width,shape.height).data;patches=patchify(rgba,shape.width,shape.height,state.patch);imageURL=surface.toDataURL('image/png');
 state.selected=Math.min(Math.max(0,state.selected),patches.length-1);state.key=Math.min(state.key,tokenCount()-1);
 result=runEncoder(patches,state.cols,state.model);buildImageGrid();render();$('#error').hidden=true;
}
function buildImageGrid(){const grid=$('#image-grid');grid.style.backgroundImage=`url('${imageURL}')`;grid.style.gridTemplateColumns=`repeat(${state.cols},1fr)`;grid.style.gridTemplateRows=`repeat(${state.rows},1fr)`;grid.style.aspectRatio=`${state.width}/${state.height}`;grid.style.maxWidth=`${Math.min(184,184*state.width/state.height)}px`;grid.innerHTML=patches.map((p,i)=>`<button class="patch-hit" data-patch="${i}" aria-label="Select patch row ${p.row}, column ${p.col}" aria-pressed="${state.selected===i}" title="Patch ${i} · row ${p.row}, column ${p.col}"></button>`).join('');}
function render(){if(!result)return;
 $$('.model-switch button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.model===state.model));$$('[data-head]').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.head===state.head));
 $('#layer-number').textContent=state.layer+1;$('#block-caption').textContent=`TRANSFORMER BLOCK ${state.layer+1}`;$('#image-shape').textContent=`${state.width} × ${state.height} RGB`;$('#image-fit').textContent=state.model==='vit'?'Square crop':'Aspect-preserving';
 $('#selected-coordinate').textContent=state.selected<0?'[CLS]':`(${currentPatch().row}, ${currentPatch().col})`;
 $$('.patch-hit').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.patch===state.selected));
 $('#patch-count').textContent=`${patches.length} patches`;$('#embedding-shape').textContent=`${tokenCount()} × ${DIM}`;
 $('#position-button').textContent=state.model==='vit'?'+ Position embedding':'2D RoPE inside attention';$('#ffn-title').textContent=state.model==='vit'?'MLP':'Gated MLP';$('#ffn-type').textContent=state.model==='vit'?'GELU activation':'SiLU × gate';
 $('#matrix-caption').textContent=`${tokenCount()} × ${tokenCount()} · no causal mask`;
 $('#output-shape').textContent=state.model==='vit'?`${tokenCount()} × 24`:`${patches.length} × 24`;
 $('#output-destination').innerHTML=`<span>${state.model==='vit'?'patches + [CLS]':'patches only'}</span><small>after all ${DEPTH} blocks</small><small>vectors, not text</small>`;
 $('#next-path-preview').innerHTML=compactNextMarkup(state.model);
 renderSequence();drawEmbeddings();drawAttention();renderVectors();renderDetail();
 $$('.stage').forEach(s=>s.classList.toggle('active',+s.dataset.stage===state.step));$('#step-count').textContent=`${state.step+1} / ${STAGE_NAMES.length}`;$('#step-back').disabled=state.step===0;$('#step-next').disabled=state.step===STAGE_NAMES.length-1;renderExpansion();
}
function renderSequence(){let indices=new Set([0,1,2,Math.max(0,state.selected-1),Math.max(0,state.selected),Math.min(patches.length-1,state.selected+1),patches.length-2,patches.length-1]);let sorted=[...indices].filter(x=>x>=0).sort((a,b)=>a-b);let prev=-1;
 $('#patch-sequence').innerHTML=sorted.map(i=>{let ell=i>prev+1?'<div class="sequence-ellipsis" aria-hidden="true">⋮</div>':'';prev=i;return ell+`<button class="sequence-item" data-patch="${i}" aria-label="Select patch ${i}, row ${patches[i].row}, column ${patches[i].col}" aria-pressed="${state.selected===i}"><span class="sequence-thumb" style="background-image:url('${imageURL}');background-size:${state.cols*100}% ${state.rows*100}%;background-position:${state.cols===1?0:patches[i].col/(state.cols-1)*100}% ${state.rows===1?0:patches[i].row/(state.rows-1)*100}%"></span><span>p${String(i).padStart(2,'0')}</span></button>`;}).join('');}
function setupCanvas(id,w,h){const c=$(id),r=window.devicePixelRatio||1;c.width=Math.round(w*r);c.height=Math.round(h*r);const g=c.getContext('2d');g.scale(r,r);return g;}
function drawEmbeddings(){const w=100,h=192,g=setupCanvas('#embedding-matrix',w,h),rows=result.input,n=rows.length,cellH=h/n,cellW=w/DIM;
 rows.forEach((v,i)=>v.forEach((x,j)=>{g.fillStyle=diverging(x);g.fillRect(j*cellW,i*cellH,Math.max(.5,cellW-.4),Math.max(.5,cellH-.4));}));g.strokeStyle='#3667e4';g.lineWidth=1.2;g.strokeRect(.7,tokenIndex()*cellH+.5,w-1.4,Math.max(1,cellH-1));
 $('#embedding-matrix').setAttribute('aria-label',`${n} input token vectors, each with ${DIM} dimensions. Selected ${tokenLabel(tokenIndex())}. Use up and down arrows to select a token.`);
}
function drawAttention(){const size=164,g=setupCanvas('#attention-matrix',size,size),head=currentHead(),n=tokenCount(),cw=size/n,max=Math.max(...head.attention.flat());
 head.attention.forEach((row,i)=>row.forEach((v,j)=>{g.fillStyle=heat(v,max);g.fillRect(j*cw,i*cw,Math.max(.3,cw-.45),Math.max(.3,cw-.45));}));g.strokeStyle='#204ac7';g.lineWidth=1;g.strokeRect(.5,tokenIndex()*cw+.3,size-1,Math.max(1,cw-.6));g.strokeStyle='#c7765b';g.strokeRect(state.key*cw+.4,tokenIndex()*cw+.4,Math.max(1,cw-.8),Math.max(1,cw-.8));
 $('#attention-matrix').setAttribute('aria-label',`Attention block ${state.layer+1}, head ${state.head+1}. Query ${tokenLabel(tokenIndex())}, key ${tokenLabel(state.key)}, weight ${fmt(head.attention[tokenIndex()][state.key]*100,2)} percent. Arrow keys change the key and query.`);
}
function renderVectors(){const l=currentLayer(),i=tokenIndex(),h=currentHead();['q','k','v'].forEach((name)=>{const x=name==='v'?h.values[i]:h[name][i];$(`#${name}-swatch`).innerHTML=x.slice(0,4).map(v=>`<i style="background:${diverging(v)}"></i>`).join('');});
 $('#ffn-viz').innerHTML=stack(l.afterAttention[i])+stack(l.expanded[i],true)+stack(l.output[i]);
 const near=[Math.max(0,i-1),i,Math.min(result.final.length-1,i+1)];$('#output-vectors').innerHTML=near.map((j,z)=>`<div style="opacity:${z===1?1:.4}">${stack(result.final[j])}</div>`).join('');
}
const details=[
 ['THE INPUT','Start with pixels, not words.'],['PATCHIFICATION','From a grid to a sequence.'],['PATCH EMBEDDING','One hidden state per patch.'],['SELF-ATTENTION','Every patch, in conversation.'],['FEED-FORWARD NETWORK','Mix features, not patches.'],['THE READOUT','Representations, not sentences.'],['WHAT HAPPENS NEXT','A classifier or a text decoder?']
];
function renderDetail(){const p=currentPatch(),i=tokenIndex(),l=currentLayer(),h=currentHead(),isPix=state.model==='pixtral',d=details[state.step];let description='',formula='',note='',visual='';
 $('#detail-kicker').textContent=`0${state.step+1} / ${d[0]}`;$('#detail-title').textContent=d[1];
 if(state.step===0){description=`An RGB image is a height × width × 3 array of numbers. ${isPix?'This path preserves the original aspect ratio, then pads the right and bottom edges to whole patches.':'This teaching ViT path takes a center crop and resizes it to a fixed 96 × 96 square.'} Choose an illustration or upload your own image.`;formula=`Image shape: ${state.height} × ${state.width} × 3  →  ${state.rows} rows × ${state.cols} columns of patches`;note='Demo-sized inputs, not production model resolutions. Uploads never leave your device.';visual=`<div class="patch-crop" style="${cropStyle(p,102)}" aria-label="Magnified selected patch"></div><div class="attn-facts"><div class="fact-label">Selected patch<span class="fact-value">(${p.row}, ${p.col})</span></div><div class="fact-label">Mean RGB<span class="fact-value">${p.mean.map(x=>Math.round(x)).join(', ')}</span></div><div class="fact-label">Input preprocessing<span class="fact-value" style="font-size:10px">${isPix?'resize + edge pad':'center crop + resize'}</span></div></div>`;
 }else if(state.step===1){description=`Split the image into ${patches.length} non-overlapping ${state.patch} × ${state.patch} patches. Read the grid row by row, then flatten each patch’s RGB values into a vector. Flattening changes the shape, not the information.`;formula=`${state.patch} × ${state.patch} × 3 = ${state.patch**2*3} values per patch`;note='A patch is analogous to a text token—but it starts as continuous pixels, not a vocabulary ID.';visual=`<div class="mini-equation"><div><div class="patch-crop" style="${cropStyle(p,102)}"></div><div class="mini-vector-label">patch (${p.row}, ${p.col})</div></div><span>→</span><div><div class="mini-vector">${p.pixels.slice(0,4).map(x=>`<span>${fmt(x,2)}</span>`).join('')}<span>⋮</span></div><div class="mini-vector-label">${p.pixels.length} values</div></div></div>`;
 }else if(state.step===2){description=isPix?'The same linear projection maps every flattened patch into a hidden vector. Pixtral then RMS-normalizes these embeddings; 2D position information is introduced later, in every attention layer.':'The same linear projection maps each patch into a hidden vector. A learned [CLS] token is prepended, and a learned position vector is added to each token once, before the transformer blocks.';formula=isPix?'xᵢ = RMSNorm(flatten(patchᵢ) · Wₑ)':'xᵢ = flatten(patchᵢ) · Wₑ + positionᵢ';note='The demo uses seeded random weights and d = 24. These are not trained visual features.';visual=`<div class="detail-vector-group"><div class="labeled-vector"><span>${state.selected<0?'[CLS]':'projected'}</span>${vectorRow(state.selected<0?result.cls:result.embeddings[state.selected])}</div>${!isPix?`<div class="labeled-vector"><span>+ position</span>${vectorRow(result.positions[i])}</div>`:''}<div class="labeled-vector"><span>${isPix?'normalized':'input xᵢ'}</span>${vectorRow(result.input[i])}</div><div class="mini-vector-label">${state.selected<0?'[CLS] is a separate vector, not a patch projection.':'One cell per hidden dimension. Hover to inspect a value.'}</div></div>`;
 }else if(state.step===3){const row=h.attention[i],max=Math.max(...row),off=isPix?0:1;description=`Each patch creates a query, key, and value. Query–key scores become softmax weights, which mix the value vectors from all tokens. Unlike a text decoder, the encoder has no causal mask.${isPix?' Here, Q and K are rotated with 2D RoPE first.':''}`;formula='A = softmax(QKᵀ / √dₕ)    →    attention output = AV';note=`Computed from untrained weights, not a semantic explanation of the image. The heatmap is relative to this row’s maximum.${!isPix?' [CLS] also participates in attention.':''}`;visual=`<div><div class="detail-grid" style="grid-template-columns:repeat(${state.cols},1fr);grid-template-rows:repeat(${state.rows},1fr);aspect-ratio:${state.cols}/${state.rows}">${patches.map((p,j)=>`<button data-key="${j+off}" aria-label="Key patch row ${p.row}, column ${p.col}, weight ${fmt(row[j+off]*100,2)} percent" aria-pressed="${state.key===j+off}" title="Key (${p.row}, ${p.col}) · ${fmt(row[j+off]*100,3)}%" style="background:${heat(row[j+off],max)}"></button>`).join('')}</div><div class="attention-legend"><span>0</span><i></i><span>${fmt(max*100,1)}%</span></div></div><div class="attn-facts"><div class="fact-label">Query → key<span class="fact-value" style="font-size:11px">${tokenLabel(i)} → ${tokenLabel(state.key)}</span></div><div class="fact-label">Attention weight<span class="fact-value blue">${fmt(row[state.key]*100,2)}%</span></div><div class="fact-label">All ${row.length} weights sum to<span class="fact-value">${fmt(row.reduce((a,b)=>a+b,0),3)}</span></div></div>`;
 }else if(state.step===4){description=`Attention mixes information between patches. The ${isPix?'gated FFN':'MLP'} then transforms the features inside each patch independently, using the same weights for every token. A residual connection adds this update back to the current hidden state.`;formula=isPix?'h ← h + W₂[SiLU(W₁ · RMSNorm(h)) ⊙ W₃ · RMSNorm(h)]':'h ← h + W₂ · GELU(W₁ · LayerNorm(h))';note='Teaching dimensions: 24 → 48 → 24. Compare the residual stream before and after this block’s MLP.';visual=`<div class="detail-vector-group"><div class="labeled-vector"><span>before</span>${vectorRow(l.afterAttention[i])}</div><div class="labeled-vector"><span>+ update</span>${vectorRow(l.mlpDelta[i])}</div><div class="labeled-vector"><span>after</span>${vectorRow(l.output[i])}</div><div class="mini-vector-label">Block ${state.layer+1}, token ${tokenLabel(i)} · actual activations</div></div>`;
 }else if(state.step===5){
 description=`The vision encoder returns ${result.final.length} vectors, each ${DIM} numbers wide, after all ${DEPTH} teaching blocks. ${isPix?'There is one vector per image patch.':'There is one vector per image patch plus an extra [CLS] vector.'} These are contextual hidden states, not text or class probabilities.`;
 formula=`encoder output: ${result.final.length} × ${DIM}`;note='Open “What next?” to follow the classification or language-model handoff.';
 visual=`<div class="detail-vector-group"><div class="labeled-vector"><span>selected</span>${vectorRow(result.final[i])}</div><button class="ro-next-link" data-detail-next>What happens after the encoder? →</button></div>`;
 }else{
 description=isPix?'Pixtral projects all patch vectors to the text decoder’s embedding width. The image vectors and text embeddings form one causal-decoder sequence, whose last position predicts the next text token.':'[CLS] is a learned classification token prepended before the image patches. Its final hidden state feeds a classification head. It is not an image crop, a class label, or a generated word. Pixtral’s path to text instead keeps all patch vectors.';
 formula=isPix?'patch vectors → projector → image + text sequence → next token':'[CLS] → classifier; patches → projector → text decoder';
 note='The classifier and decoder are explanatory schematics, not executed models.';
 visual=`<div class="process-stack"><div class="process-line"><strong>Classification ViT</strong><span>[CLS] → class scores</span></div><div class="process-line"><strong>Pixtral</strong><span>patches → LLM → text</span></div><button class="ro-next-link" data-detail-next>Explore the handoff →</button></div>`;
 }

 $('#detail-description').textContent=description;$('#detail-formula').textContent=formula;$('#detail-note').textContent=note;$('#detail-visual').innerHTML=visual;renderTensorInspector();
}
function renderTensorInspector(){
 const inspector=$('#tensor-inspector');inspector.hidden=state.step!==3;if(state.step!==3)return;
 const h=currentHead(),i=tokenIndex(),j=state.key,q=h.q[i],k=h.k[j];
 const values=(label,x)=>`<div class="tensor-row"><span>${label}</span>${x.map((v,d)=>`<span title="dimension ${d}" style="background:${diverging(v)}">${fmt(v,3)}</span>`).join('')}</div>`;
 $('#tensor-content').innerHTML=`<div class="tensor-values">${state.model==='pixtral'?values('Q before RoPE',h.rawQ[i])+values('K before RoPE',h.rawK[j]):''}${values(state.model==='pixtral'?'Q after RoPE':'Q at query',q)}${values(state.model==='pixtral'?'K after RoPE':'K at key',k)}${values('Q ⊙ K',q.map((v,d)=>v*k[d]))}${values('V at key',h.values[j])}</div><div class="tensor-calculation"><span>Σ(Q ⊙ K) = <strong>${fmt(dot(q,k),5)}</strong></span><span>÷ √8 = <strong>${fmt(h.scores[i][j],5)}</strong></span><span>softmax across ${tokenCount()} keys → <strong>${fmt(h.attention[i][j]*100,3)}%</strong></span></div><p>Q belongs to query ${tokenLabel(i)}; K and V belong to key ${tokenLabel(j)}. This key contributes its V vector multiplied by the displayed softmax weight. Q and K are not passed to the next layer.</p>`;
}

function selectPatch(index,announce=true){state.selected=Math.max(state.model==='vit'?-1:0,Math.min(patches.length-1,index));state.key=tokenIndex();render();if(announce)$('#status').textContent=`Selected ${tokenLabel(tokenIndex())}. Attention block ${state.layer+1}, head ${state.head+1}.`;}
function setStep(step,scroll=false){state.step=Math.max(0,Math.min(STAGE_NAMES.length-1,step));if(state.expanded!==null)state.expanded=state.step;if(state.step<2&&state.selected<0)state.selected=0;render();if(scroll)(state.expanded===null?$('#detail'):$('.stage.is-expanded')).scrollIntoView({behavior:'smooth',block:'nearest'});}
function setModel(model){state.model=model;recompute();$('#status').textContent=`${model==='vit'?'Classic ViT':'Pixtral'} architecture. ${patches.length} patches, ${state.rows} rows by ${state.cols} columns.`;}
function handleError(error){$('#error').textContent=error.message;$('#error').hidden=false;}
let attentionController=null,multiHeadController=null;
let railViewport={top:0,left:0,selected:null,signature:null};
// A stage expands in the architecture itself; neighboring stages stay navigable.
function renderExpansion(){
 const previousMultiHead=multiHeadController?.snapshot;multiHeadController?.dispose();multiHeadController=null;
 const previousAttention=attentionController?.snapshot;attentionController?.dispose();attentionController=null;
 const oldRail=$('.token-rail-list');
 if(oldRail)railViewport={top:oldRail.scrollTop,left:oldRail.scrollLeft,selected:oldRail.dataset.selected,signature:oldRail.dataset.signature};
 const expanded=state.expanded!==null;
 $('#architecture').classList.toggle('has-expansion',expanded);
 $('#detail').hidden=expanded;
 if(state.model==='vit'&&state.ffn==='gate')state.ffn='up';
 if(state.model==='vit'&&state.attentionInspect==='rope')state.attentionInspect=null;
 $$('.stage').forEach(stage=>{
   const index=+stage.dataset.stage,active=index===state.expanded,trigger=stage.querySelector('.stage-title');
   let panel=stage.querySelector('.stage-expansion');
   if(!panel){panel=document.createElement('div');panel.className='stage-expansion';panel.id=`stage-expansion-${index}`;stage.append(panel);}
   trigger.setAttribute('aria-expanded',String(active));trigger.setAttribute('aria-controls',panel.id);trigger.id=`stage-trigger-${index}`;
   trigger.querySelector('.stage-plus').textContent=active?'−':'+';
   stage.classList.toggle('is-expanded',active);panel.hidden=!active;
   if(!active){panel.innerHTML='';return;}
   panel.setAttribute('role','region');panel.setAttribute('aria-labelledby',trigger.id);
   panel.innerHTML=expansionMarkup(state,patches,result,imageURL);
 });
 const multiHead=$('#multi-head');
 if(multiHead)multiHeadController=mountMultiHead(multiHead,{
  state,patches,layer:currentLayer(),previous:previousMultiHead,
  onHead(head){
    state.headDetailOpen=!(state.headDetailOpen&&state.head===head);state.head=head;state.attentionInspect=null;
    renderKeepingFocus(`.mh-head-heading[data-mh-open="${head}"]`);
    const detail=$('#head-detail');
    if(state.headDetailOpen&&!matchMedia('(prefers-reduced-motion: reduce)').matches)detail.animate([{opacity:0,transform:'translateY(-12px)'},{opacity:1,transform:'none'}],{duration:650,easing:'cubic-bezier(.22,.75,.18,1)'});
   },
  onInspect(){state.attentionInspect=state.attentionInspect==='merge'?null:'merge';renderKeepingFocus('[data-mh-project]');},
  onCompare(value){state.compareCausal=value;renderKeepingFocus('#compare-causal');}
 });
 const flow=$('#attention-flow');
 if(flow)attentionController=mountAttentionFlow(flow,{
   state,patches,head:currentHead(),previous:previousAttention,
   onPair(query,key,focus){state.selected=query-(state.model==='vit'?1:0);state.key=key;renderKeepingFocus(focus);},
   onInspect(kind){state.attentionInspect=state.attentionInspect===kind?null:kind;if(['q','k','v'].includes(kind))state.qkv=kind;renderKeepingFocus(`[data-attn-inspect="${kind}"]`);},
   onCompare(value){state.compareCausal=value;renderKeepingFocus('#compare-causal');}
 });
 const rail=$('.token-rail-list');
 if(rail){
   const sameGrid=rail.dataset.signature===railViewport.signature;
   rail.scrollTop=sameGrid?railViewport.top:0;rail.scrollLeft=sameGrid?railViewport.left:0;
   if(!sameGrid||rail.dataset.selected!==railViewport.selected){
     revealRailSelection(rail);
   }
 }
}
function revealRailSelection(rail){
 const selected=rail?.querySelector('[aria-pressed="true"]');if(!selected)return;
 const r=rail.getBoundingClientRect(),p=selected.getBoundingClientRect();
 rail.scrollTop=railScrollOffset(rail.scrollTop,rail.clientHeight,p.top-r.top+rail.scrollTop,p.bottom-r.top+rail.scrollTop);
 rail.scrollLeft=railScrollOffset(rail.scrollLeft,rail.clientWidth,p.left-r.left+rail.scrollLeft,p.right-r.left+rail.scrollLeft);
}
window.addEventListener('resize',()=>requestAnimationFrame(()=>revealRailSelection($('.token-rail-list'))));
const architectureMotion=createArchitectureMotion($('#architecture'),()=>state.expanded);
function openExpansion(step){
 if(!result||step<0||step>=STAGE_NAMES.length)return;
 const origin=document.activeElement;
 architectureMotion.run(step,()=>{
   state.expanded=step;setStep(step);
   if(origin&&(!origin.isConnected||origin.getClientRects().length===0))$(`#stage-trigger-${step}`).focus({preventScroll:true});
   $('#status').textContent=`${details[step][0]} expanded inside the architecture.`;
 });
}
function closeExpansion(){
 architectureMotion.finish();const previous=state.expanded;
 architectureMotion.run(null,()=>{state.expanded=null;render();if(previous!==null)$(`#stage-trigger-${previous}`).focus({preventScroll:true});});
}
function renderKeepingFocus(key){render();if(key){const el=$('.stage.is-expanded '+key);if(el)el.focus({preventScroll:true});}}
$('#architecture').addEventListener('click',e=>{
 const target=e.target.closest('button');if(!target)return;
 if(target.hasAttribute('data-next-model')){state.projectorDetail=false;state.decoderStep=0;setModel(target.dataset.nextModel);$('#stage-trigger-6').focus({preventScroll:true});$('#stage-expansion-6').scrollIntoView({behavior:'instant',block:'start'});return;}
 if(target.hasAttribute('data-decoder-step')){state.decoderStep=+target.dataset.decoderStep;renderKeepingFocus(`[data-decoder-step="${state.decoderStep}"]`);return;}
 if(target.hasAttribute('data-projector-detail')){state.projectorDetail=!state.projectorDetail;renderKeepingFocus('[data-projector-detail]');return;}
 if(target.hasAttribute('data-readout-patch')){selectPatch(+target.dataset.readoutPatch);renderKeepingFocus(`[data-readout-patch="${state.selected}"]`);return;}
 if(target.hasAttribute('data-head-detail-close')){state.headDetailOpen=false;state.attentionInspect=null;renderKeepingFocus(`.mh-head-heading[data-mh-open="${state.head}"]`);return;}
 if(target.hasAttribute('data-attn-close')){const kind=state.attentionInspect;state.attentionInspect=null;renderKeepingFocus(kind==='merge'?'[data-mh-project]':`[data-attn-inspect="${kind}"]`);return;}
 if(target.hasAttribute('data-collapse')){closeExpansion();return;}
 if(target.hasAttribute('data-expand-step')){openExpansion(+target.dataset.expandStep);return;}
 if(target.hasAttribute('data-rail-patch')){selectPatch(+target.dataset.railPatch);$('.token-rail-item[aria-pressed="true"]').focus({preventScroll:true});return;}
 if(target.hasAttribute('data-expanded-patch')){selectPatch(+target.dataset.expandedPatch);renderKeepingFocus(`[data-expanded-patch="${state.selected}"]`);return;}
 if(target.hasAttribute('data-pixel')){state.scalar=+target.dataset.pixel*3+state.scalar%3;renderKeepingFocus(`[data-pixel="${target.dataset.pixel}"]`);return;}
 if(target.hasAttribute('data-scalar')){state.scalar=+target.dataset.scalar;renderKeepingFocus(`[data-scalar="${state.scalar}"]`);return;}
 if(target.hasAttribute('data-dimension')){state.dimension=+target.dataset.dimension;renderKeepingFocus(`.output-cell-grid [data-dimension="${state.dimension}"]`);return;}
 if(target.hasAttribute('data-weight')){[state.scalar,state.dimension]=target.dataset.weight.split(',').map(Number);renderKeepingFocus(`[data-weight="${target.dataset.weight}"]`);return;}
 const kind=target.dataset.inspect;if(!kind)return;
 if(kind==='channel'){state.scalar=Math.floor(state.scalar/3)*3+(+target.dataset.value);}
 else if(['qkv','ffn','adapter'].includes(kind)){state[kind]=target.dataset.value;if(kind==='qkv'&&state.expanded===3)state.attentionInspect=target.dataset.value;}
 renderKeepingFocus(`[data-inspect="${kind}"][data-value="${target.dataset.value}"]`);
});
function updateInspectionControl(e){
 const el=e.target,kind=el.dataset.inspect;if(!kind||el.value==='')return;const n=Number(el.value);if(!Number.isFinite(n))return;
 if(kind==='patch'){state.selected=n;state.key=tokenIndex();}
 if(kind==='key')state.key=n;
 if(kind==='cls-step')state.clsStep=Math.max(0,Math.min(result.layers.length+2,n));
 if(kind==='head'){state.head=n;state.headDetailOpen=true;state.attentionInspect=null;}
 if(kind==='scalar-index')state.scalar=Math.max(0,Math.min(+el.max,Math.round(n)));
 if(kind==='pixel-row'||kind==='pixel-col'){
   const pixel=Math.floor(state.scalar/3),row=Math.floor(pixel/state.patch),col=pixel%state.patch,v=Math.max(0,Math.min(state.patch-1,Math.round(n)));
   state.scalar=((kind==='pixel-row'?v:row)*state.patch+(kind==='pixel-col'?v:col))*3+state.scalar%3;
 }
 renderKeepingFocus('#'+el.id);
}
$('#architecture').addEventListener('change',e=>{if(e.target.tagName==='SELECT')updateInspectionControl(e);});
$('#architecture').addEventListener('input',e=>{if(e.target.matches('input[data-inspect]'))updateInspectionControl(e);});
$('#architecture').addEventListener('keydown',e=>{
 if(e.key==='Escape'&&state.expanded!==null){e.preventDefault();closeExpansion();return;}
 const railItem=e.target.closest('[data-rail-patch]');
 if(railItem){const next=nextRailPatch(+railItem.dataset.railPatch,e.key,patches.length,state.model==='vit'&&state.expanded>=2);if(next!==null){e.preventDefault();selectPatch(next);$('.token-rail-item[aria-pressed="true"]').focus({preventScroll:true});}return;}
 const pixel=e.target.closest('[data-pixel]');if(pixel){const delta={ArrowRight:1,ArrowLeft:-1,ArrowUp:-state.patch,ArrowDown:state.patch}[e.key];if(delta!==undefined){e.preventDefault();const next=Math.min(state.patch**2-1,Math.max(0,+pixel.dataset.pixel+delta));state.scalar=next*3+state.scalar%3;renderKeepingFocus(`[data-pixel="${next}"]`);}}
});
[['#ffn-viz',4],['#output-vectors',5]].forEach(([id,step])=>{const el=$(id);el.setAttribute('role','button');el.tabIndex=0;el.setAttribute('aria-label',`Expand ${step===4?'MLP':'output'} computation`);el.addEventListener('click',()=>openExpansion(step));el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openExpansion(step);}});});
$$('[data-model]').forEach(b=>b.addEventListener('click',()=>setModel(b.dataset.model)));
$('#sample').addEventListener('change',()=>{if($('#sample').value==='custom')return;state.sample=$('#sample').value;loadImage(`assets/${state.sample}.svg`).catch(handleError);});
$('#patch-size').addEventListener('change',e=>{state.patch=+e.target.value;recompute();});
$('#layer').addEventListener('input',e=>{state.layer=+e.target.value;render();});
$$('[data-head]').forEach(b=>b.addEventListener('click',()=>{state.head=+b.dataset.head;render();}));
$('#image-grid').addEventListener('click',e=>{const b=e.target.closest('[data-patch]');if(b){selectPatch(+b.dataset.patch);if(state.expanded===null)openExpansion(1);}});
$('#image-grid').addEventListener('keydown',e=>{const b=e.target.closest('[data-patch]');if(!b)return;const delta={ArrowRight:1,ArrowLeft:-1,ArrowUp:-state.cols,ArrowDown:state.cols}[e.key];if(delta!==undefined){e.preventDefault();const i=Math.max(0,Math.min(patches.length-1,+b.dataset.patch+delta));selectPatch(i);$(`#image-grid [data-patch="${i}"]`).focus();}});
$('#patch-sequence').addEventListener('click',e=>{const b=e.target.closest('[data-patch]');if(b){selectPatch(+b.dataset.patch);if(state.expanded===null)openExpansion(1);}});
$$('[data-step]').forEach(b=>b.addEventListener('click',()=>state.expanded===+b.dataset.step?closeExpansion():openExpansion(+b.dataset.step)));
$('#detail-visual').addEventListener('click',e=>{if(e.target.closest('[data-detail-next]')){openExpansion(6);return;}const b=e.target.closest('[data-key]');if(b){state.key=+b.dataset.key;drawAttention();renderDetail();$('#detail-visual [data-key="'+state.key+'"]').focus({preventScroll:true});}});
$('#step-back').addEventListener('click',()=>setStep(state.step-1));$('#step-next').addEventListener('click',()=>setStep(state.step+1));
$('#tour').addEventListener('click',()=>{$('#explorer').scrollIntoView({behavior:'instant',block:'start'});openExpansion(0);$('#detail-note').textContent+=' Use the right arrow to advance through the seven stages.';});
$('#position-button').addEventListener('click',()=>openExpansion(state.model==='pixtral'?3:2));
$('#try-pixtral').addEventListener('click',()=>{setModel('pixtral');$('#explorer').scrollIntoView({behavior:'instant',block:'start'});openExpansion(6);});
$('#attention-matrix').addEventListener('click',e=>{const rect=e.currentTarget.getBoundingClientRect(),n=tokenCount(),q=Math.min(n-1,Math.floor((e.clientY-rect.top)/rect.height*n));state.selected=q-(state.model==='vit'?1:0);state.key=Math.min(n-1,Math.floor((e.clientX-rect.left)/rect.width*n));openExpansion(3);});
$('#attention-matrix').addEventListener('mousemove',e=>{const r=e.currentTarget.getBoundingClientRect(),n=tokenCount(),q=Math.min(n-1,Math.max(0,Math.floor((e.clientY-r.top)/r.height*n))),k=Math.min(n-1,Math.max(0,Math.floor((e.clientX-r.left)/r.width*n)));e.currentTarget.title=`Query ${tokenLabel(q)} → Key ${tokenLabel(k)}: ${fmt(currentHead().attention[q][k]*100,3)}%`;});
$('#attention-matrix').addEventListener('keydown',e=>{if(!e.key.startsWith('Arrow'))return;e.preventDefault();if(e.key==='ArrowRight')state.key=Math.min(tokenCount()-1,state.key+1);if(e.key==='ArrowLeft')state.key=Math.max(0,state.key-1);if(e.key==='ArrowUp')state.selected=Math.max(state.model==='vit'?-1:0,state.selected-1);if(e.key==='ArrowDown')state.selected=Math.min(patches.length-1,state.selected+1);state.step=3;render();});
$('#embedding-matrix').addEventListener('click',e=>{const r=e.currentTarget.getBoundingClientRect();selectPatch(Math.min(tokenCount()-1,Math.floor((e.clientY-r.top)/r.height*tokenCount()))-(state.model==='vit'?1:0));openExpansion(2);});
$('#embedding-matrix').addEventListener('keydown',e=>{if(e.key==='ArrowUp'||e.key==='ArrowDown'){e.preventDefault();selectPatch(state.selected+(e.key==='ArrowUp'?-1:1));setStep(2);}});
const dialog=$('#about-dialog');['#about-open','#method-open','#demo-note'].forEach(id=>$(id).addEventListener('click',()=>dialog.showModal()));$('#about-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
$('#upload-button').addEventListener('click',()=>$('#upload').click());
$('#upload').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;try{if(f.size>20*1024*1024)throw Error('Please choose an image smaller than 20 MB.');if(!f.type.startsWith('image/')||f.type==='image/svg+xml')throw Error('Please use a PNG, JPEG, WebP, GIF, or AVIF image.');const nextURL=URL.createObjectURL(f);try{await loadImage(nextURL);}catch(err){URL.revokeObjectURL(nextURL);throw err;}if(uploadURL)URL.revokeObjectURL(uploadURL);uploadURL=nextURL;if(!$('#sample option[value=custom]'))$('#sample').insertAdjacentHTML('beforeend','<option value="custom">Your image</option>');$('#sample').value='custom';state.sample='custom';}catch(err){handleError(err);}finally{e.target.value='';}});
// An isolated positional lab. Fixed vectors make the relative-position identity testable.
const labQ=[1,.2,.85,-.2,.3,.5,-.1,.7], labK=[.7,.6,.95,.1,-.3,.4,.5,.2];
function circleMarkup(pair){const q=labQ.slice(pair*2,pair*2+2),k=labK.slice(pair*2,pair*2+2),s=57;return `<div class="rotation-circle"><svg viewBox="0 0 180 175" role="img" aria-label="${pair===0?'Row':'Column'} pair rotation" id="circle-${pair}"><defs><marker id="qa${pair}" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0 0L5 2.5L0 5Z" fill="#3667e4"/></marker><marker id="ka${pair}" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0 0L5 2.5L0 5Z" fill="#c7765b"/></marker></defs><circle cx="90" cy="84" r="59" fill="none" stroke="#e4eaf3"/><path d="M17 84H163M90 11V157" stroke="#edf0f5" stroke-width="1"/><text x="157" y="97" font-size="8" fill="#a8b3c1" font-family="monospace">x</text><text x="97" y="17" font-size="8" fill="#a8b3c1" font-family="monospace">y</text><line x1="90" y1="84" x2="${90+q[0]*s}" y2="${84-q[1]*s}" stroke="#9fb8ee" stroke-dasharray="3 3"/><line x1="90" y1="84" x2="${90+k[0]*s}" y2="${84-k[1]*s}" stroke="#e5bba9" stroke-dasharray="3 3"/><g id="q-rotate-${pair}" style="transform-origin:90px 84px;transition:transform .35s ease"><line x1="90" y1="84" x2="${90+q[0]*s}" y2="${84-q[1]*s}" stroke="#3667e4" stroke-width="2" marker-end="url(#qa${pair})"/></g><g id="k-rotate-${pair}" style="transform-origin:90px 84px;transition:transform .35s ease"><line x1="90" y1="84" x2="${90+k[0]*s}" y2="${84-k[1]*s}" stroke="#c7765b" stroke-width="2" marker-end="url(#ka${pair})"/></g><circle cx="90" cy="84" r="2.5" fill="#96a7bf"/></svg><div class="circle-label" id="circle-label-${pair}"></div><div class="circle-freq" id="circle-freq-${pair}"></div></div>`;}
$('#rotation-circles').innerHTML=circleMarkup(0)+circleMarkup(1);
['qc','kc'].forEach(id=>$('#'+id).max='7');
function labPositions(){return ['qr','qc','kr','kc'].map(id=>{const el=$('#'+id);const v=Math.min(+el.max,Math.max(0,Math.round(+el.value||0)));if(el.value!==''&&+el.value!==v)el.value=v;return v;});}
function renderRope(){const [qr,qc,kr,kc]=labPositions(),scale=+$('#rope-frequency').value/10;let score=0;for(let pair=0;pair<4;pair++){const freq=10000**(-2*pair/8)*scale,qp=state.ropeMode==='2d'?(pair%2===0?qr:qc):qr*8+qc,kp=state.ropeMode==='2d'?(pair%2===0?kr:kc):kr*8+kc;score+=dot(rotatePair(labQ.slice(pair*2,pair*2+2),qp*freq),rotatePair(labK.slice(pair*2,pair*2+2),kp*freq));if(pair<2){$('#q-rotate-'+pair).style.transform=`rotate(${-qp*freq}rad)`;$('#k-rotate-'+pair).style.transform=`rotate(${-kp*freq}rad)`;$('#circle-label-'+pair).textContent=state.ropeMode==='2d'?(pair===0?'Row pair':'Column pair'):`Sequence pair ${pair+1}`;$('#circle-freq-'+pair).textContent=`θ = ${fmt(freq,2)} rad / position`;$('#circle-'+pair).setAttribute('aria-label',`${state.ropeMode==='2d'?(pair===0?'Row':'Column'):'Sequence'} rotation: query angle ${fmt(qp*freq,2)}, key angle ${fmt(kp*freq,2)} radians.`);}}
 $('#rope-score').textContent=fmt(score,4);$('#rope-displacement').textContent=state.ropeMode==='2d'?`Δrow = ${kr-qr} · Δcol = ${kc-qc}`:`Δindex = ${(kr-qr)*8+kc-qc}`;$('#frequency-label').textContent=`${fmt(scale,1)}×`;$('#rope-mode-note').textContent=`${state.ropeMode==='1d'?'Flattened index = row × 8 + column. ':''}First two of four coordinate pairs shown. The score sums all four; no softmax or √d scaling.`;$('#shift-both').disabled=qr>=20||kr>=20||qc>=7||kc>=7;$$('[data-rope]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.rope===state.ropeMode));
}
$$('[data-rope]').forEach(b=>b.addEventListener('click',()=>{state.ropeMode=b.dataset.rope;renderRope();}));['qr','qc','kr','kc','rope-frequency'].forEach(id=>$('#'+id).addEventListener('input',renderRope));
$('#shift-both').addEventListener('click',()=>{['qr','qc','kr','kc'].forEach(id=>$('#'+id).value=+$('#'+id).value+1);renderRope();$('#status').textContent=`Both positions shifted. The positional dot product remains ${$('#rope-score').textContent}.`;});
$('#packing-grid').innerHTML=Array.from({length:100},(_,i)=>{const r=Math.floor(i/10),c=i%10;return `<i class="${r<6&&c<6?'a':r>=6&&c>=6?'b':''}"></i>`;}).join('');
renderRope();loadImage('assets/bird.svg').catch(handleError);
