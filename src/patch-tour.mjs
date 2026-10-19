/** Guided playback: actual encoder values, no mutation or simulated predictions. */
import {outputColor} from './readout.mjs';
export const PATCH_TOUR_STEPS=Object.freeze([
 {label:'Pixels',duration:3200},{label:'Patches',duration:4000},{label:'Embedding',duration:6000},
 {label:'Attention',duration:6000},{label:'MLP',duration:7200},{label:'Output',duration:4800},{label:'What next?',duration:4800}
]);
const tourClamp=x=>Math.max(0,Math.min(1,x));
function tourSums(input,weights,column=0){const sums=[0];input.forEach((v,i)=>sums.push(sums.at(-1)+v*weights[i][column]));return sums;}
export function createPatchTourPlan(s,patches,result){
 if(s.selected<0||s.selected>=patches.length)throw Error('The tour needs an image patch, not CLS.');
 const index=s.selected+(s.model==='vit'?1:0),patch=patches[s.selected],layer=result.layers[s.layer],head=layer.heads[s.head];
 const attentionSums=[Array(head.values[0].length).fill(0)];
 head.attention[index].forEach((a,k)=>attentionSums.push(attentionSums.at(-1).map((v,d)=>v+a*head.values[k][d])));
 return {model:s.model,patchIndex:s.selected,index,patchSize:s.patch,cols:s.cols,rows:s.rows,patch,layerIndex:s.layer,headIndex:s.head,layer,head,result,
  label:`p${String(s.selected).padStart(2,'0')} · (${patch.row}, ${patch.col})`,
  sums:{embedding:tourSums(patch.pixels,result.projectionWeights),up:tourSums(layer.ffnInput[index],layer.weights.up),down:tourSums(layer.expanded[index],layer.weights.down)},attentionSums};
}
export function patchTourFrame(plan,step,progress){
 const p=tourClamp(progress),{patch,index,result,layer,head,sums}=plan;
 const count=n=>Math.min(n,Math.floor(p*n));
 if(step===0){const pixel=Math.min(plan.patchSize**2-1,count(plan.patchSize**2));return {kind:'pixels',pixel,rgb:patch.pixels.slice(pixel*3,pixel*3+3).map(v=>Math.round((v+1)*127.5)),row:Math.floor(pixel/plan.patchSize),col:pixel%plan.patchSize};}
 if(step===1){const scalar=Math.min(patch.pixels.length-1,count(patch.pixels.length));return {kind:'flatten',scalar,channel:scalar%3,values:patch.pixels,start:Math.min(Math.floor(scalar/12)*12,patch.pixels.length-12)};}
 if(step===2){const n=patch.pixels.length,terms=count(n);return {kind:'product',operation:'embedding',input:patch.pixels,weights:result.projectionWeights,output:result.embeddings[plan.patchIndex],terms,total:n,sum:sums.embedding[terms],expected:result.embeddings[plan.patchIndex][0]};}
 if(step===3){const terms=count(head.values.length);return {kind:'attention',terms,total:head.values.length,weights:head.attention[index],values:terms?head.values[terms-1]:head.values[0],sum:plan.attentionSums[terms],expected:head.out[index],key:Math.max(0,terms-1)};}
 if(step===4){const phase=Math.min(3,Math.floor(p*4)),local=tourClamp((p*4-phase)/.8);
  if(phase===0||phase===2){const up=phase===0,input=up?layer.ffnInput[index]:layer.expanded[index],output=up?layer.preActivation[index]:layer.mlpDelta[index],terms=Math.min(input.length,Math.floor(local*input.length));return {kind:'product',operation:up?'up':'down',phase,input,weights:layer.weights[up?'up':'down'],output,terms,total:input.length,sum:sums[up?'up':'down'][terms],expected:output[0]};}
  if(phase===1)return {kind:'activation',phase,input:layer.preActivation[index],gate:layer.gate?.[index],output:layer.expanded[index],revealed:Math.floor(local*layer.expanded[index].length)};
  return {kind:'residual',phase,input:layer.afterAttention[index],delta:layer.mlpDelta[index],output:layer.output[index],revealed:Math.floor(local*layer.output[index].length)};
 }
 if(step===5){const remaining=result.layers.slice(plan.layerIndex+1),phase=Math.min(remaining.length,Math.floor(p*(remaining.length+1))),final=phase===remaining.length;
  return {kind:'output',phase,final,block:final?result.layers.length:plan.layerIndex+phase+2,output:final?result.final[index]:remaining[phase].output[index]};
 }
 if(step===6)return {kind:'handoff',phase:Math.min(2,Math.floor(p*3)),output:plan.model==='pixtral'?result.projected[plan.patchIndex]:result.final[0],symbolic:true};
 throw Error('Unknown tour step');
}
/** A single cancellable clock; pause retains elapsed time and replay has no stale callbacks. */
export function createPatchTourClock({request,cancel,onStep=()=>{},onFrame=()=>{},onState=()=>{},durations=PATCH_TOUR_STEPS.map(s=>s.duration)}){
 let step=0,elapsed=0,last=null,frameId=null,running=false,active=false,completed=false,epoch=0;
 const snapshot=()=>({step,elapsed,progress:Math.min(1,elapsed/durations[step]),running,active,completed});
 function clear(){epoch++;if(frameId!==null)cancel(frameId);frameId=null;last=null;}
 function draw(){onFrame(snapshot());}function announce(){onState(snapshot());}
 function queue(){const ticket=epoch;frameId=request(now=>tick(now,ticket));}
 function tick(now,ticket){if(ticket!==epoch||!active||!running)return;frameId=null;
  const delta=last===null?0:Math.max(0,now-last);last=now;elapsed=Math.min(durations[step],elapsed+delta);draw();
  if(!active||!running||ticket!==epoch)return;
  if(elapsed>=durations[step]){
   if(step===durations.length-1){running=false;completed=true;clear();announce();return;}
   step++;elapsed=0;last=now;onStep(snapshot());draw();announce();
  }
  if(active&&running&&ticket===epoch)queue();
 }
 return {
  start({autoplay=true}={}){clear();step=0;elapsed=autoplay?0:durations[0];active=true;running=autoplay;completed=false;onStep(snapshot());draw();announce();if(running)queue();},
  pause(){if(!active||!running)return;running=false;clear();announce();},
  resume(){if(!active||completed||running)return;running=true;if(elapsed>=durations[step])elapsed=0;clear();draw();announce();queue();},
  seek(next,{play=false,showEnd=false}={}){if(!active)return;clear();step=Math.max(0,Math.min(durations.length-1,next));elapsed=showEnd?durations[step]:0;completed=false;running=play;onStep(snapshot());draw();announce();if(running)queue();},
  stop(){clear();running=false;active=false;completed=false;announce();},
  get snapshot(){return snapshot();}
 };
}
function tourBezier(a,b,t){const u=1-t,bend=Math.max(24,Math.abs(b.x-a.x)*.45);return {x:u**3*a.x+3*u*u*t*(a.x+bend)+3*u*t*t*(b.x-bend)+t**3*b.x,y:u**3*a.y+3*u*u*t*a.y+3*u*t*t*b.y+t**3*b.y};}
function tourNarrative(plan,step){const pix=plan.model==='pixtral',p=plan.patchIndex;
 return [
  ['Start with the selected patch.',`Follow ${plan.label}. The scanner reads its real ${plan.patchSize} × ${plan.patchSize} RGB pixels. This image crop identifies the token throughout the tour.`],
  ['A patch becomes a list—not a word.',`Read pixels row by row, keeping R, G, B together. Normalize each channel to [−1, 1]. The same patch now has ${plan.patch.pixels.length} numbers; nothing is learned here.`],
  ['Multiply pixels by shared weights.',`Watch one embedding feature accumulate from every input × weight. The whole product produces 24 features. ${pix?'RMSNorm follows; 2D RoPE comes inside attention.':'Then add a position vector; [CLS] is a separate learned token.'}`],
  ['Gather information from other tokens.',`Block ${plan.layerIndex+1}, head ${plan.headIndex+1}: weighted value vectors accumulate into this patch’s 8-feature head output. All tokens are allowed—no causal mask. The 3 heads are then concatenated, projected and added to the residual.`],
  ['Transform this patch independently.',`Watch expansion, ${pix?'SiLU × gate':'GELU'}, contraction, and the residual addition in order. Each patch has its own activations but uses the same MLP weights. Other patches are not mixed here.`],
  ['Keep following the same row.',`Continue through the remaining encoder blocks${pix?'':', then final LayerNorm'}. The matrix still contains every token. p${p} is one 24-feature row—not a generated label or word.`],
  pix?['Hand off image features—not words.',`The projector maps this patch from 24 to 32 features in the teaching model. All projected patches join text embeddings in a decoder prefix. The decoder / next-token part shown here is symbolic, not executed.`]:['Classification reads [CLS], not a patch.',`p${p} remains in the encoder output. Classification instead reads the separate final [CLS] vector, which interacted with the patches through attention. The classifier shown here is symbolic, not executed.`]
 ][step];
}
export function createPatchTour({root,trigger,getData,prepare,onStep,onAnnouncement=()=>{}}){
 const doc=root.ownerDocument,win=doc.defaultView,reduced=win.matchMedia('(prefers-reduced-motion: reduce)');
 const panel=doc.createElement('section');panel.id='patch-tour';panel.className='patch-tour';panel.hidden=true;panel.setAttribute('aria-label','Animated patch walkthrough');root.before(panel);
 panel.innerHTML=`<div class="pt-toolbar"><div class="pt-identity"><i data-pt-thumb aria-hidden="true"></i><div><span>FOLLOW A PATCH</span><strong data-pt-identity></strong></div><b data-pt-count></b></div><div class="pt-buttons"><button type="button" data-pt-prev aria-label="Previous walkthrough step">←</button><button type="button" data-pt-play>Pause</button><button type="button" data-pt-next aria-label="Next walkthrough step">→</button><button type="button" data-pt-replay>↻ Replay</button><button type="button" data-pt-close aria-label="End patch walkthrough">×</button></div></div>
 <div class="pt-steps" role="group" aria-label="Walkthrough stages">${PATCH_TOUR_STEPS.map((s,i)=>`<button type="button" data-pt-step="${i}" aria-label="Go to ${s.label}"><span>0${i+1}</span>${s.label}<i aria-hidden="true"></i></button>`).join('')}</div>
 <div class="pt-story"><div><h3 data-pt-title></h3><p data-pt-copy></p></div><div class="pt-demonstration"><canvas id="patch-tour-canvas" width="960" height="170" role="img" aria-label="Animated values for the selected patch"></canvas><div data-pt-math></div></div></div>
 <div class="pt-status"><span data-pt-status role="status" aria-live="polite"></span><span>Live teaching values · untrained model · click the diagram to explore on your own</span></div>`;
 const svg=doc.createElementNS('http://www.w3.org/2000/svg','svg');svg.id='patch-tour-signal';svg.setAttribute('aria-hidden','true');root.append(svg);
 const q=s=>panel.querySelector(s),canvas=q('#patch-tour-canvas'),g=canvas.getContext('2d');let plan=null,source=null,disposed=false,manual=false,previousStep=0,lastDraw=-1,mathFrame=null;
 const num=(v,d=4)=>Number(v).toFixed(d);
 function text(str,x,y,color='#7890b0',size=15){g.fillStyle=color;g.font=`${size}px ui-monospace,monospace`;g.textAlign='left';g.fillText(str,x,y);}
 function arrow(x,y){text('→',x,y,'#a4b3c8',24);}
 function vector(values,x,y,w,h,label,{revealed=values.length,highlight=-1}={}){text(label,x,y-10);const cw=w/values.length;values.forEach((v,i)=>{g.fillStyle=i<revealed?outputColor(v):'#edf1f8';g.fillRect(x+i*cw,y,Math.max(.5,cw-1),h);});if(highlight>=0){g.strokeStyle='#886ac2';g.lineWidth=2;g.strokeRect(x+highlight*cw,y,cw,h);} }
 function product(frame){const {input,weights,output,terms,total,sum,expected}=frame;
  vector(input,8,40,210,40,`${input.length} inputs`,{highlight:Math.max(0,terms-1)});arrow(230,66);
  text(`${weights.length} × ${output.length} weights`,277,28);const cw=260/output.length,ch=50/weights.length;weights.forEach((row,i)=>row.forEach((v,j)=>{g.fillStyle=outputColor(v*Math.sqrt(weights.length));g.fillRect(277+j*cw,40+i*ch,Math.max(.5,cw-.5),Math.max(.1,ch));}));g.strokeStyle='#9471c1';g.lineWidth=2;g.strokeRect(277,40,cw,50);arrow(559,66);vector(output,607,40,344,40,`${output.length} outputs · feature 0`,{highlight:0});
  text(`Σ ${terms} / ${total} terms = ${num(sum,6)}`,8,134,'#796797',19);text(`full result: ${num(expected,6)}`,548,134,'#8496af',17);
  return `${frame.operation==='embedding'?'Embedding':frame.operation==='up'?'MLP expansion':'MLP contraction'} · feature 0: ${terms} / ${total} terms summed. ${terms===total?'Exact output reached.':'Partial sum; full output shown for comparison.'}`;
 }
 function drawDemo(step,p){const frame=patchTourFrame(plan,step,p);mathFrame=frame;g.clearRect(0,0,960,170);let math='';
  if(frame.kind==='pixels'){
   const sz=plan.patchSize,cw=140/sz;plan.patch.pixels.forEach((_,i)=>{if(i%3)return;const px=i/3,rgb=plan.patch.pixels.slice(i,i+3).map(v=>Math.round((v+1)*127.5));g.fillStyle=`rgb(${rgb.join(',')})`;g.fillRect(10+(px%sz)*cw,8+Math.floor(px/sz)*cw,cw+.1,cw+.1);});g.strokeStyle='#4777d2';g.lineWidth=2;g.strokeRect(10+frame.col*cw,8+frame.row*cw,cw,cw);arrow(192,86);
   frame.rgb.forEach((v,i)=>{g.fillStyle=['#f4e2db','#e5efe6','#e3ecfa'][i];g.fillRect(280+i*177,38,149,65);text('RGB'[i],290+i*177,62,['#b4745b','#648a6b','#6185bc'][i]);text(String(v),327+i*177,85,'#49617d',24);});
   math=`Pixel (${frame.row}, ${frame.col}) → RGB (${frame.rgb.join(', ')}). The blue scanner is reading actual pixels.`;
  }else if(frame.kind==='flatten'){
   const cell=76;for(let i=0;i<12;i++){const index=frame.start+i,v=frame.values[index];g.fillStyle=outputColor(v);g.fillRect(8+i*cell,34,cell-5,58);text('RGB'[index%3],17+i*cell,54,'#647d9e',12);text(num(v,2),15+i*cell,78,'#425e84',14);text(String(index),15+i*cell,116,'#96a5b9',12);if(index===frame.scalar){g.strokeStyle='#487ddd';g.lineWidth=3;g.strokeRect(8+i*cell,34,cell-5,58);}}text(`12-value window · ${frame.values.length} total scalars`,8,154);
   math=`flat[${frame.scalar}] = RGB / 127.5 − 1 = ${num(frame.values[frame.scalar])}. Flattening reorders channels; it is not a learned projection.`;
  }else if(frame.kind==='product')math=product(frame);
  else if(frame.kind==='attention'){
   vector(frame.weights,8,40,300,44,`${frame.total} attention weights`,{highlight:frame.key});arrow(326,68);vector(frame.values,374,40,204,44,`V for key ${frame.key}`);arrow(599,68);vector(frame.sum,647,40,300,44,`${frame.sum.length}-feature running sum`);text(`${frame.terms} / ${frame.total} weighted value vectors added`,8,138,'#647fa7',20);
   math=`Head ${plan.headIndex+1}: Σ attention[key] × V[key]. ${frame.terms===frame.total?'Exact head output reached.':'Partial sum—not the full attention output yet.'}`;
  }else if(frame.kind==='activation'){
   vector(frame.input,8,40,340,46,'48 pre-activation features');arrow(371,68);text(plan.model==='pixtral'?'SiLU(z) × g':'GELU(z)',431,68,'#8d709d',20);arrow(623,68);vector(frame.output,680,40,269,46,'48 activated features',{revealed:frame.revealed});
   math=plan.model==='pixtral'?'Gate projection g uses the same normalized input. Apply SiLU(z) × g elementwise.':'Apply GELU independently to each feature; no patch mixing.';
  }else if(frame.kind==='residual'){
   vector(frame.input,8,40,272,45,'pre-MLP hidden state');text('+',304,71,'#a0a8b6',26);vector(frame.delta,350,40,247,45,'24-feature MLP update');text('=',622,71,'#a0a8b6',26);vector(frame.output,677,40,272,45,'after residual addition',{revealed:frame.revealed});
   math=`Feature 0: ${num(frame.input[0])} + ${num(frame.delta[0])} = ${num(frame.output[0])}. Same patch, updated hidden state.`;
  }else if(frame.kind==='output'){
   text(frame.final?'FINAL ENCODER OUTPUT':`AFTER BLOCK ${frame.block}`,8,29,'#7b92ad',16);vector(frame.output,8,61,940,50,`${plan.label} · all 24 features`);text(`${plan.result.final.length} token rows remain in the full output`,8,150,'#8298b4',16);
   math=frame.final?`${plan.model==='vit'?'After final LayerNorm. ':''}The encoder returns vectors—not words or class probabilities.`:`The same patch passes through block ${frame.block}; these are its actual hidden-state values.`;
  }else{
   vector(frame.output,8,42,310,42,plan.model==='pixtral'?'Projected patch · 32 features':'Final [CLS] · 24 features');arrow(337,67);
   const labels=plan.model==='pixtral'?['+ other patches & text','Causal text decoder','Next text token']:['Separate CLS readout','Classifier','Class scores'];
   labels.forEach((label,i)=>{g.fillStyle=i<=frame.phase?'#eaf0fb':'#f4f6f9';g.fillRect(389+i*188,39,175,57);text(label,397+i*188,71,i<=frame.phase?'#718db5':'#a9b4c2',12);});text('ARCHITECTURE ONLY · CLASSIFIER / DECODER NOT EXECUTED',8,140,'#a393b5',15);
   math=plan.model==='pixtral'?'All image features enter the decoder alongside text embeddings; the encoder has not generated text.':'The classifier reads [CLS], not the selected patch directly. No class prediction is computed.';
  }
  q('[data-pt-math]').textContent=math;canvas.setAttribute('aria-label',math);canvas.dataset.kind=frame.kind;
  if('sum' in frame)canvas.dataset.sum=Array.isArray(frame.sum)?frame.sum.join(','):frame.sum;else delete canvas.dataset.sum;
  if('expected' in frame)canvas.dataset.expected=Array.isArray(frame.expected)?frame.expected.join(','):frame.expected;else delete canvas.dataset.expected;
 }
 function box(step){const selected=plan.patchIndex;let el;
  if(step===0)el=root.querySelector(`#image-grid [data-patch="${selected}"]`);
  if(step===1)el=root.querySelector(`#patch-sequence [data-patch="${selected}"]`);
  if(step===2)el=root.querySelector('#embedding-matrix');if(step===3)el=root.querySelector('#attention-matrix');
  if(step===4)el=root.querySelector(`#ffn-viz [data-mlp-lane="${selected}"]`);
  if(step===5)el=root.querySelector('.ro-compact-row.is-selected');if(step===6)el=root.querySelector('#next-path-preview');
  if(!el)return null;const r=el.getBoundingClientRect(),a=root.getBoundingClientRect();if(!r.width||!r.height)return null;
  const rows=plan.result.input.length,row=step===2||step===3,top=r.top-a.top+(row?plan.index*r.height/rows:0),height=row?r.height/rows:r.height;
  return {x:r.left-a.left,y:top,width:r.width,height,center:{x:r.left-a.left+r.width/2,y:top+height/2}};
 }
 function drawSignal(step,progress){if(!plan||panel.hidden)return;const b=box(step),a=box(previousStep)||b;if(!b)return;const r=root.getBoundingClientRect(),p=reduced.matches?1:tourClamp(progress/.23),t=1-(1-p)**3,point=tourBezier(a.center,b.center,t),path=`M${a.center.x},${a.center.y} C${a.center.x+Math.max(24,Math.abs(b.center.x-a.center.x)*.45)},${a.center.y} ${b.center.x-Math.max(24,Math.abs(b.center.x-a.center.x)*.45)},${b.center.y} ${b.center.x},${b.center.y}`;
  svg.setAttribute('viewBox',`0 0 ${r.width} ${r.height}`);
  const rings=step===3&&mathFrame?.kind==='attention'?`<rect class="pt-key-cell" x="${b.x+mathFrame.key*b.width/plan.result.input.length}" y="${b.y}" width="${b.width/plan.result.input.length}" height="${b.height}"/>`:'';
  svg.innerHTML=`${step===previousStep?'':`<path class="pt-travel-path" d="${path}"/>`}<rect class="pt-target-box" x="${b.x-2}" y="${b.y-2}" width="${b.width+4}" height="${b.height+4}" rx="3"/>${rings}<g transform="translate(${point.x},${point.y})"><circle class="pt-travel-glow" r="${8+(reduced.matches?0:Math.sin(progress*Math.PI*6)*2)}"/><circle class="pt-travel-dot" r="4"/></g>`;
 }
 function repaint(snap){if(!plan||!snap.active)return;const p=manual?1:snap.progress,frameKey=`${snap.step}:${Math.floor(p*90)}`;panel.dataset.step=snap.step;panel.dataset.progress=snap.progress.toFixed(3);panel.style.setProperty('--pt-progress',String(snap.progress));
  if(frameKey!==lastDraw){lastDraw=frameKey;drawDemo(snap.step,p);}drawSignal(snap.step,p);
 }
 function status(snap){panel.dataset.state=!snap.active?'closed':snap.completed?'complete':snap.running?'playing':'paused';q('[data-pt-play]').textContent=snap.completed?'↻ Replay':snap.running?'Ⅱ Pause':manual?'▶ Replay step':'▶ Resume';q('[data-pt-prev]').disabled=snap.step===0;q('[data-pt-next]').disabled=snap.step===PATCH_TOUR_STEPS.length-1;
  q('[data-pt-status]').textContent=snap.completed?'Tour complete. Replay it or choose any stage to explore.':manual?'Reduced motion · use the stage buttons or arrows to advance.':snap.running?'Playing · advancing automatically through all 7 stages.':'Paused · resume when ready, or step through at your own pace.';
  trigger.setAttribute('aria-expanded',String(snap.active));trigger.querySelector('.play-icon').textContent=snap.active?'↻':'▶';
  if(snap.completed)onAnnouncement('Patch walkthrough complete.');
 }
 const clock=createPatchTourClock({request:cb=>win.requestAnimationFrame(cb),cancel:id=>win.cancelAnimationFrame(id),onFrame:repaint,onState:status,onStep:snap=>{
  previousStep=Number(panel.dataset.step||0);onStep?.(snap.step);lastDraw=-1;root.querySelectorAll('.stage').forEach(el=>el.classList.toggle('patch-tour-focus',+el.dataset.stage===snap.step));
  const [title,copy]=tourNarrative(plan,snap.step);q('[data-pt-title]').textContent=title;q('[data-pt-copy]').textContent=copy;q('[data-pt-count]').textContent=`${snap.step+1} / 7`;
  panel.querySelectorAll('[data-pt-step]').forEach(b=>{const i=+b.dataset.ptStep;b.setAttribute('aria-current',i===snap.step?'step':'false');b.classList.toggle('is-past',i<snap.step);});
  onAnnouncement(`Step ${snap.step+1} of 7. ${title}`);
 }});
 function start(){if(disposed||!prepare())return;const data=getData();source={result:data.result,selected:data.s.selected,model:data.s.model,layer:data.s.layer,head:data.s.head};plan=createPatchTourPlan(data.s,data.patches,data.result);manual=reduced.matches;panel.hidden=false;root.classList.add('has-patch-tour');q('[data-pt-identity]').textContent=plan.label;
  const thumb=q('[data-pt-thumb]');thumb.style.backgroundImage=`url('${data.imageURL}')`;thumb.style.backgroundSize=`${plan.cols*100}% ${plan.rows*100}%`;thumb.style.backgroundPosition=`${plan.cols===1?0:plan.patch.col/(plan.cols-1)*100}% ${plan.rows===1?0:plan.patch.row/(plan.rows-1)*100}%`;
  previousStep=0;panel.dataset.step='0';clock.start({autoplay:!manual});panel.scrollIntoView({behavior:'instant',block:'start'});q('[data-pt-play]').focus({preventScroll:true});
 }
 function stop({focus=false}={}){clock.stop();panel.hidden=true;svg.replaceChildren();root.classList.remove('has-patch-tour');root.querySelectorAll('.patch-tour-focus').forEach(el=>el.classList.remove('patch-tour-focus'));source=null;plan=null;lastDraw=-1;if(focus)trigger.focus({preventScroll:true});}
 function checkData(){if(!source)return;const d=getData();if(d.result!==source.result||d.s.selected!==source.selected||d.s.model!==source.model||d.s.layer!==source.layer||d.s.head!==source.head||d.s.expanded!==null)stop();}
 function click(e){const b=e.target.closest('button');if(!b)return;const snap=clock.snapshot;
  if(b.hasAttribute('data-pt-close'))return stop({focus:true});if(b.hasAttribute('data-pt-replay')||b.hasAttribute('data-pt-play')&&snap.completed)return start();
  if(b.hasAttribute('data-pt-play')){if(snap.running)clock.pause();else if(manual)clock.seek(snap.step,{showEnd:true});else clock.resume();return;}
  const next=b.hasAttribute('data-pt-step')?+b.dataset.ptStep:b.hasAttribute('data-pt-next')?snap.step+1:b.hasAttribute('data-pt-prev')?snap.step-1:null;
  if(next!==null)clock.seek(next,{play:snap.running&&!manual,showEnd:manual});
 }
 function interrupt(e){if(!clock.snapshot.active||panel.contains(e.target)||trigger.contains(e.target))return;stop();}
 function visibility(){if(doc.hidden)clock.pause();}function preference(){if(!clock.snapshot.active)return;manual=reduced.matches;clock.pause();clock.seek(clock.snapshot.step,{showEnd:manual});}
 function key(e){if(e.key==='Escape'&&clock.snapshot.active){e.preventDefault();stop({focus:true});}}
 function resize(){if(clock.snapshot.active)repaint(clock.snapshot);}
 panel.addEventListener('click',click);trigger.addEventListener('click',start);doc.addEventListener('keydown',key);doc.addEventListener('visibilitychange',visibility);win.addEventListener('resize',resize);reduced.addEventListener('change',preference);
 for(const type of ['click','input','change','keydown'])root.parentElement.addEventListener(type,interrupt,true);
 // About or a navigation link outside the explorer also hands control back to the user.
 doc.addEventListener('click',interrupt,true);
 trigger.setAttribute('aria-controls',panel.id);trigger.setAttribute('aria-expanded','false');
 return {start,stop,checkData,get active(){return clock.snapshot.active;},get snapshot(){return clock.snapshot;},dispose(){stop();disposed=true;panel.removeEventListener('click',click);trigger.removeEventListener('click',start);doc.removeEventListener('keydown',key);doc.removeEventListener('visibilitychange',visibility);doc.removeEventListener('click',interrupt,true);win.removeEventListener('resize',resize);reduced.removeEventListener('change',preference);for(const type of ['click','input','change','keydown'])root.parentElement.removeEventListener(type,interrupt,true);panel.remove();svg.remove();}};
}
