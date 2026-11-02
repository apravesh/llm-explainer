import {matrix,viewMatrix,matrixValue} from './full-config.mjs';
import {overviewPixels} from './full-matrix.mjs';
/** Read the selected token from the real encoder, never the separate 4→6→4 toy. */
export function mlpFlowData(run,state){
 const c=run.config,l=run.layers[state.layer];if(!l)return null;
 const row=Math.max(0,Math.min(l.input.rows-1,Math.trunc(state.selected+(c.cls?1:0))||0));
 const selected=m=>viewMatrix(m,{rowStart:row,rows:1});
 // SiLU alone is not stored by the worker. Apply it BEFORE pooling, in the same
 // precision used by activateFull before it multiplies by the W₃ branch.
 const activated=c.gated?matrix(1,c.ffWidth,Float64Array.from({length:c.ffWidth},(_,j)=>{const x=matrixValue(l.preActivation,row,j);return x/(1+Math.exp(-x));})):selected(l.expanded);
 return {c,row,patch:row-(c.cls?1:0),block:state.layer,total:l.input.rows,input:selected(l.ffnInput),activated,values:c.gated?selected(l.gate):null,mixed:selected(l.expanded),delta:selected(l.mlpDelta),residual:selected(l.afterAttention),output:selected(l.output)};
}
export function mlpFlowPreview(a){
 const count=Math.min(16,a.cols);let peak=0;
 for(let i=0;i<count;i++){const start=Math.floor(i*a.cols/count),end=Math.floor((i+1)*a.cols/count);let sum=0;for(let j=start;j<end;j++)sum+=matrixValue(a,0,j);peak=Math.max(peak,Math.abs(sum/(end-start)));}
 return {...overviewPixels(a,{maxCols:16,maxRows:1,scale:peak||1}),scale:peak||1};
}
function mfStrip(a,x,y,width){const p=mlpFlowPreview(a),pitch=width/p.width;return Array.from({length:p.width},(_,j)=>`<rect class="mf-feature" x="${(x+j*pitch).toFixed(2)}" y="${y}" width="${Math.max(.1,pitch-1).toFixed(2)}" height="18" rx="1" fill="rgb(${[...p.data.slice(j*4,j*4+3)].join(',')})"/>`).join('');}
export function mlpFlowDiagramMarkup(d,compact=false){
 const {c}=d,fmt=v=>v.toLocaleString(),shape=m=>`1 × ${fmt(m.cols)}`,text=(value,x,y,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${value}</text>`;
 const boxes=compact?{
  input:[203,8,174],up:[22,150,207],gate:[351,150,207],mix:[195,286,190],down:[134,419,111],delta:[293,406,182],residual:[22,548,185],output:[351,548,207]
 }:{input:[12,134,168],up:[242,27,218],gate:[242,226,218],mix:[540,134,194],down:[790,149,112],delta:[958,134,194],residual:[12,352,200],output:[958,352,194]};
 if(!c.gated){boxes.up=compact?[182,150,216]:[270,134,216];boxes.down=compact?[130,288,111]:[573,149,112];boxes.delta=compact?[290,275,190]:[766,134,194];boxes.residual=compact?[22,420,185]:[12,334,200];boxes.output=compact?[351,420,207]:[958,334,194];}
 const paths=c.gated?(compact?[
  ['branches','M250 112V128H125V144'],['branches','M330 112V128H454V144'],
  ['mix','M125 254V268H237V280'],['mix','M454 254V268H343V280'],
  ['contraction','M246 390V413'],['contraction','M245 456H287'],
  ['residual','M384 510V542'],['residual','M207 600H345']
 ]:[
  ['branches','M180 186H206Q220 186 220 171V94Q220 79 236 79'],['branches','M180 186H206Q220 186 220 201V263Q220 278 236 278'],
  ['mix','M460 79H493Q514 79 514 100V169Q514 178 534 178'],['mix','M460 278H493Q514 278 514 256V212Q514 204 534 204'],
  ['contraction','M734 186H784'],['contraction','M902 186H952'],
  ['residual','M1055 238V346'],['residual','M212 404H952']
 ]):(compact?[
  ['branches','M290 112V144'],['contraction','M238 254V270H185V282'],['contraction','M241 326H284'],['residual','M385 379V414'],['residual','M207 472H345']
 ]:[['branches','M180 186H264'],['contraction','M486 186H567'],['contraction','M685 186H760'],['residual','M863 238V297Q863 314 884 314H1055V328'],['residual','M212 386H952']]);
 const marker=`mf-${compact?'compact':'wide'}-${c.gated?'swiglu':'gelu'}`;
 function card(id,title,caption,tensor,equation,operation=null){const [x,y,w]=boxes[id],small=id==='down',height=small?74:104;
  const content=`<rect class="mf-card-bg" x="${x}" y="${y}" width="${w}" height="${height}" rx="9"/>${text(title,x+14,y+25,'mf-title')}${text(caption,x+14,y+43,'mf-caption')}${tensor?mfStrip(tensor,x+14,y+56,w-28):''}${text(equation,x+14,y+(small?62:93),'mf-dimension')}`;
  return `<g data-mf-node="${id}" class="mf-node mf-node-${id}"${operation?` role="button" tabindex="0" data-mf-operation="${operation}" aria-label="Inspect all MLP multiplications from ${id==='up'?'W1':id==='gate'?'W3':'W2'}" aria-controls="mlp-calculation" aria-expanded="${d.calculationOpen}"`:''}>${content}</g>`;
 }
 const diagramHeight=compact?(c.gated?674:545):c.gated?480:462;
 return `<svg class="mf-diagram ${compact?'mf-compact':'mf-wide'}" viewBox="0 0 ${compact?580:1166} ${diagramHeight}" role="group" aria-label="${c.gated?'SwiGLU: the same input splits into W1 plus SiLU and a parallel W3 branch, multiplied elementwise before W2 contraction.':'ViT MLP: W1 expansion, GELU, and W2 contraction.'} Then add the pre-MLP residual H. Actual encoder values for ${d.patch<0?'CLS':`patch ${d.patch}`} in block ${d.block+1}.">
 <defs><marker id="${marker}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#a6b8d3"/></marker></defs>
 <g class="mf-connections" marker-end="url(#${marker})">${paths.map(([phase,path])=>`<path data-mf-edge="${phase}" d="${path}"/>`).join('')}</g>
 ${card('input','Input x','Normalized patch features',d.input,shape(d.input))}
 ${card('up',c.gated?'W₁ → SiLU':'W₁ → GELU',`${c.gated?'Gate branch':'Expansion'} · ${fmt(c.width)} × ${fmt(c.ffWidth)}`,d.activated,`${c.gated?'g = SiLU(xW₁)':'a = GELU(xW₁)'} · ${shape(d.activated)}`,'mlp-up')}
 ${c.gated?card('gate','W₃',`Parallel expansion · ${fmt(c.width)} × ${fmt(c.ffWidth)}`,d.values,`v = xW₃ · ${shape(d.values)}`,'mlp-gate'):''}
 ${c.gated?text('Same x → both branches',compact?218:262,compact?139:183,'mf-parallel'):''}
 ${c.gated?card('mix','g ⊙ v','Multiply matching features',d.mixed,`c = g ⊙ v · ${shape(d.mixed)}`):''}
 ${card('down','W₂','Contraction',null,`${fmt(c.ffWidth)} × ${fmt(c.width)}`,'mlp-down')}
 ${card('delta','MLP update Δ','Before the residual addition',d.delta,shape(d.delta))}
 ${card('residual','Residual H','Before normalization',d.residual,shape(d.residual))}
 ${card('output','H + Δ → H′','Updated patch features',d.output,shape(d.output))}
 ${text('Residual bypass · not normalized x',compact?37:438,compact?(c.gated?660:534):c.gated?428:414,'mf-residual-label')}
 </svg>`;
}
export function mlpFlowMarkup(run,state){
 const d=mlpFlowData(run,state);if(!d)return '';d.calculationOpen=!!state.mlpCalculation;
 return `<section class="mf-flow" data-mlp-flow data-mf-row="${d.row}" data-mf-block="${d.block}" aria-label="${d.c.gated?'SwiGLU':'GELU MLP'} flow for the selected patch">
 <header class="mf-toolbar"><div><strong>${d.patch<0?'Follow CLS':`Follow patch p${d.patch}`} · block ${d.block+1} / ${d.c.depth}</strong><span>${d.c.gated?'Expand twice → activate one branch → multiply → contract':'Expand → GELU → contract'} · then add the residual</span></div><button type="button" data-mf-replay>↻ Trace the arrows</button></header>
 <div class="mf-visual">${mlpFlowDiagramMarkup(d)}${mlpFlowDiagramMarkup(d,true)}</div>
 <p class="mf-status" data-mf-status aria-live="off">${d.c.gated?'W₁ and W₃ reuse the same input in parallel. ⊙ multiplies matching features; it does not mix patches.':'GELU transforms each expanded feature independently, before contraction.'}</p>
 <div class="mf-footer"><span>Real encoder widths, not the toy example. Strips pool adjacent features and auto-scale colors independently for visibility; compare exact magnitudes in “Show actual values.” Every feature still participates. All ${d.total} ${d.c.cls?'tokens':'patches'} share these weights.</span><button type="button" data-mf-calculation-toggle aria-controls="mlp-calculation" aria-expanded="${d.calculationOpen}">${d.calculationOpen?'− Hide full calculation':'+ Inspect full calculation'}</button></div>
 </section>`;
}
export function mlpFlowFrame(elapsed){
 const time=Math.max(0,elapsed);return time<650?'input':time<1400?'branches':time<2100?'mix':time<2800?'contraction':time<3500?'residual':'complete';
}
export function mountMlpFlow(root,{gated,autoplay=false,onCalculation}){
 const doc=root.ownerDocument,win=doc.defaultView,media=win.matchMedia('(prefers-reduced-motion: reduce)');let frame=null,last=0,elapsed=3500,phase='',playing=false,disposed=false;
 const phases={input:['input'],branches:['up',...(gated?['gate']:[])],mix:gated?['mix']:['up'],contraction:['down','delta'],residual:['residual','output'],complete:[]};
 const messages={input:'Read the selected patch’s normalized input x.',branches:gated?'The same x feeds both projections. Apply SiLU only to W₁’s result.':'W₁ expands x. Apply GELU independently to each expanded feature.',mix:gated?'Multiply g and v element by element. The feature count stays the same; no patches are mixed.':'The activated features are ready for W₂.',contraction:'W₂ sums across every expanded feature to make the MLP update Δ.',residual:'Add Δ to H, the pre-MLP residual—not normalized x.',complete:gated?'Complete: expand twice → activate one branch → multiply → contract → add residual.':'Complete: expand → GELU → contract → add residual.'};
 function paint(){const next=mlpFlowFrame(elapsed);if(next===phase)return;phase=next;root.dataset.phase=phase;root.querySelectorAll('[data-mf-node]').forEach(el=>el.classList.toggle('is-active',phases[phase].includes(el.dataset.mfNode)));root.querySelectorAll('[data-mf-edge]').forEach(el=>el.classList.toggle('is-active',el.dataset.mfEdge===phase));root.querySelector('[data-mf-status]').textContent=messages[phase];}
 function pause(){playing=false;if(frame!==null)win.cancelAnimationFrame(frame);frame=null;root.dataset.playing='false';}
 function tick(now){frame=null;if(disposed||!playing)return;if(!root.isConnected){dispose();return;}if(doc.hidden||doc.querySelector('dialog[open]')){pause();return;}elapsed=Math.min(3500,elapsed+Math.max(0,now-last));last=now;paint();if(elapsed===3500){pause();return;}frame=win.requestAnimationFrame(tick);}
 function replay(){if(disposed)return;pause();elapsed=media.matches?3500:0;phase='';paint();if(media.matches||doc.hidden)return;playing=true;root.dataset.playing='true';last=win.performance.now();frame=win.requestAnimationFrame(tick);}
 function activate(target){const op=target.closest('[data-mf-operation]');if(op){pause();onCalculation('open');return true;}if(target.closest('[data-mf-calculation-toggle]')){pause();onCalculation('toggle');return true;}if(target.closest('[data-mf-replay]')){replay();return true;}return false;}
 const click=e=>activate(e.target),keydown=e=>{if(['Enter',' '].includes(e.key)&&e.target.closest('[data-mf-operation]')){e.preventDefault();activate(e.target);}},visibility=()=>{if(doc.hidden)pause();},preference=()=>{if(media.matches){pause();elapsed=3500;paint();}};
 function dispose(){if(disposed)return;pause();disposed=true;root.removeEventListener('click',click);root.removeEventListener('keydown',keydown);doc.removeEventListener('visibilitychange',visibility);media.removeEventListener('change',preference);}
 root.addEventListener('click',click);root.addEventListener('keydown',keydown);doc.addEventListener('visibilitychange',visibility);media.addEventListener('change',preference);if(autoplay)replay();else paint();
 return {dispose,replay};
}
