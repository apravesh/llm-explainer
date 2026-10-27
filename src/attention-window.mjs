import {mountAttentionWindowMotion} from './attention-window-motion.mjs';
import {attentionStages,attentionStageTrace,stageNumber,ATTENTION_STAGE_NAMES,animateAttentionStages} from './attention-stages.mjs';
import {mountAttentionRailAlignment} from './attention-rail-alignment.mjs';
import {mountAttentionCellTooltip} from './attention-tooltip.mjs';
/** A six-token viewport of full attention. Values are never re-normalized or recomputed. */
import {matrixValue,viewMatrix} from './full-config.mjs';
import {activationRGB,overviewPixels} from './full-matrix.mjs';
export const ATTENTION_WINDOW_SIZE=6;
export function attentionWindow(total,start=0,size=ATTENTION_WINDOW_SIZE){
 const count=Math.min(total,size),first=Math.max(0,Math.min(total-count,Math.trunc(start)||0));
 return {start:first,end:first+count,count,total,indices:Array.from({length:count},(_,i)=>first+i),before:first,after:total-first-count};
}
export function attentionWindowForToken(total,start,token,size=ATTENTION_WINDOW_SIZE){
 const win=attentionWindow(total,start??Math.max(0,token-2),size);
 return token>=win.start&&token<win.end?win:attentionWindow(total,Math.max(0,token-2),size);
}
export function windowFromScroll(scroll,rowPitch,total,size=ATTENTION_WINDOW_SIZE){return attentionWindow(total,Math.floor(Math.max(0,scroll)/Math.max(1,rowPitch)+.05),size);}
export function attentionWindowData(data,start,keyStart=start){
 const win=attentionWindow(data.n,start),keys=attentionWindow(data.n,keyStart),weights=win.indices.map(q=>keys.indices.map(k=>data.head.attention.data[q*data.n+k]));
 let visibleMass=0;for(const k of keys.indices)visibleMass+=data.head.attention.data[data.query*data.n+k];
 const totalMass=Array.from({length:data.n},(_,k)=>data.head.attention.data[data.query*data.n+k]).reduce((a,b)=>a+b,0);
 let max=0;for(const value of data.head.attention.data)max=Math.max(max,value);
 return {...win,keys,weights,visibleMass,hiddenMass:totalMass-visibleMass,totalMass,max};
}
export function attentionOutputWindow(data,rowStart,featureStart=0){
 const rows=attentionWindow(data.n,rowStart),features=attentionWindow(data.c.headWidth,featureStart,8);
 return {rows,features,values:rows.indices.map(r=>features.indices.map(c=>matrixValue(data.head.out,r,c)))};
}
export function windowTokenLabel(i,config){return config.cls&&i===0?'CLS':`p${String(i-(config.cls?1:0)).padStart(2,'0')}`;}
export function attentionWindowKey(query,key,keyName,total){
 if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(keyName))return null;
 return {query:Math.max(0,Math.min(total-1,keyName==='ArrowUp'?query-1:keyName==='ArrowDown'?query+1:query)),key:Math.max(0,Math.min(total-1,keyName==='ArrowLeft'?key-1:keyName==='ArrowRight'?key+1:keyName==='Home'?0:keyName==='End'?total-1:key))};
}
function awText(text,x,y,cls=''){return `<text x="${x}" y="${y}" class="${cls}">${text}</text>`;}
function awPreview(m,row,x,y,width=46,height=10,featureOffset=null){
 // Real values only: output columns are cropped, never pooled or re-computed.
 const cw=width/m.cols;return Array.from({length:m.cols},(_,j)=>{const value=matrixValue(m,row,j);return `<rect x="${x+j*cw}" y="${y}" width="${Math.max(.1,cw)}" height="${height}" fill="rgb(${activationRGB(value).join(',')})"${featureOffset===null?'/>' : ` data-output-cell="${row}:${featureOffset+j}" data-output-value="${value}"><title>Output[${row}, ${featureOffset+j}] = ${value.toPrecision(8)}</title></rect>`}`;}).join('');
}
export function attentionWindowMarkup(data,state){
 const w=attentionWindowData(data,state.attentionStart,state.attentionKeyStart??state.attentionStart),{c}=data,label=i=>windowTokenLabel(i,c),expanded=!!state.attentionBreakdown;
 const outWindow=attentionOutputWindow(data,w.start,state.attentionOutputFeature),outView=viewMatrix(data.head.out,{colStart:outWindow.features.start,cols:outWindow.features.count});
 const stages=expanded?attentionStages(data):null,cell=29,y0=153,x0=expanded?230:266,gridWidth=w.count*cell,gridHeight=gridWidth;
 const panelGap=216,panels=expanded?['dot','scaled','softmax']:['softmax'],lastX=x0+(panels.length-1)*panelGap,outX=expanded?960:618,mixX=expanded?786:444,mixWidth=155,svgWidth=expanded?1080:740,svgHeight=expanded?655:580,valueTop=expanded?435:370,mixTop=expanded?466:401;
 let svg='',links='',dots='',motionDefs='';const tokenRows={q:'',k:'',v:'',out:''};
 const viewport=(name,content,x,y,width,height)=>{const id=`aw-pan-${name}`;motionDefs+=`<clipPath id="${id}" clipPathUnits="userSpaceOnUse"><rect x="${x}" y="${y}" width="${width}" height="${height}"/></clipPath>`;return `<g clip-path="url(#${id})" data-motion-viewport="${name}">${content}</g>`;};
 const queryY=i=>y0+i*cell+cell/2,keyY=i=>35+i*17,valueY=i=>valueTop+i*17,outputY=i=>valueTop+i*17;
 const curve=(x1,y1,x2,y2,bend=.5)=>`M${x1} ${y1}C${x1+(x2-x1)*bend} ${y1} ${x1+(x2-x1)*bend} ${y2} ${x2} ${y2}`;
 const tokenGroup=(kind,m,token,y,xLabel,xStrip)=>`<g role="button" tabindex="${token===(kind==='q'||kind==='out'?data.query:data.key)?0:-1}" data-window-token="${token}" data-window-kind="${kind}" data-window-pitch="${kind==='q'?29:17}" aria-label="Inspect ${kind==='out'?'head output':kind.toUpperCase()} for ${label(token)}" class="aw-token aw-token-${kind} ${token===(kind==='q'||kind==='out'?data.query:data.key)?'is-selected':''}"><title>${label(token)} · ${kind==='out'?`features ${outWindow.features.start}–${outWindow.features.end-1} of ${c.headWidth}`:`${m.cols} actual features`}. Select the token; inspect exact features below.</title><rect class="aw-token-hit" x="${xLabel-4}" y="${y-11}" width="${xStrip-xLabel+53}" height="${kind==='q'?23:17}" rx="4"/>${awText(label(token),xLabel,y+4,'aw-token-name')}${awPreview(m,token,xStrip,y-5,46,10,kind==='out'?outWindow.features.start:null)}<path class="aw-port" d="M${xStrip+47} ${y-4}V${y+4}"/></g>`;
 w.indices.forEach((token,i)=>{
  const qy=queryY(i),oy=outputY(i);
  links+=`<path data-window-wire="q-${token}" class="aw-link aw-q-link ${token===data.query?'is-emphasized':''}" d="${curve(139,qy,x0-12,qy)}"/>`;
  links+=`<path data-window-wire="out-${token}" class="aw-link aw-output-link ${token===data.query?'is-emphasized':''}" d="${curve(mixX+mixWidth,mixTop+27,outX-4,oy,.68)}"/>`;
  tokenRows.q+=tokenGroup('q',data.q,token,qy,43,90);tokenRows.out+=tokenGroup('out',outView,token,oy,outX,outX+43);
 });
 w.keys.indices.forEach((token,i)=>{
  const ky=keyY(i),vy=valueY(i),cx=x0+(i+.5)*cell;
  links+=`<path data-window-wire="k-${token}" class="aw-link aw-k-link ${token===data.key?'is-emphasized':''}" d="M228 ${ky}H${cx-18}Q${cx} ${ky} ${cx} ${ky+18}V${y0-12}"/>`;
  links+=`<path data-window-wire="v-${token}" class="aw-link aw-v-link" d="M228 ${vy}C370 ${vy} ${mixX-5} ${vy} ${mixX} ${mixTop+36}"/>`;
  tokenRows.k+=tokenGroup('k',data.k,token,ky,139,179);tokenRows.v+=tokenGroup('v',data.v,token,vy,139,179);
 });
 links=`<path class="aw-value-ribbon" d="M228 ${valueTop-3}C360 ${valueTop-3} ${mixX-15} ${valueTop-3} ${mixX} ${mixTop+25}V${mixTop+45}C${mixX-15} ${valueTop+97} 374 ${valueTop+97} 228 ${valueTop+97}Z"/><path class="aw-link aw-attention-weights-link" d="M${lastX+gridWidth+8} ${y0+gridHeight/2}C${lastX+gridWidth+75} ${y0+gridHeight/2} ${mixX+mixWidth/2} ${mixTop-40} ${mixX+mixWidth/2} ${mixTop}" marker-end="url(#aw-weights-arrow)"/>`+links;
 const pane=(axis,content,x,y,width,height)=>`<g data-window-axis="${axis}" class="aw-scroll-pane" role="region" tabindex="0" aria-label="${axis==='query'?'Query patches — scroll rows independently':axis==='key'?'Key patches — scroll columns independently':'Head output preview — vertical scroll follows query rows; horizontal scroll changes features'}"><rect class="aw-pane-hit" x="${x}" y="${y}" width="${width}" height="${height}"/>${content}</g>`;
 const scrollAnchor=(axis,x,y,width,height)=>`<rect data-window-scroll-anchor="${axis}" x="${x}" y="${y}" width="${width}" height="${height}" fill="none" aria-hidden="true"/>`;
 svg+=pane('query',viewport('q',tokenRows.q,35,y0-2,110,gridHeight+4),35,y0-2,110,gridHeight+4)+scrollAnchor('query',20,y0-2,12,gridHeight+4);
 svg+=pane('key',viewport('k',tokenRows.k,130,23,103,w.keys.count*17+6),130,23,103,w.keys.count*17+6)+scrollAnchor('key',114,23,12,w.keys.count*17+6);
 svg+=viewport('v',tokenRows.v,130,valueTop-12,103,w.keys.count*17+6);
 svg+=pane('output',viewport('out',tokenRows.out,outX-8,valueTop-12,105,w.count*17+6),outX-8,valueTop-12,105,w.count*17+6)+scrollAnchor('output',outX+99,valueTop-12,12,w.count*17+6)+scrollAnchor('feature',outX-8,valueTop+w.count*17,105,12);
 const signedColor=(value,range)=>{const scale=Math.max(Math.abs(range.min),Math.abs(range.max),1e-8),rgb=value<0?'211,145,116':'130,91,201';return `rgba(${rgb},${.12+.8*Math.abs(value)/scale})`;};
 panels.forEach((stage,panelIndex)=>{
  const x=x0+panelIndex*panelGap,range=expanded?stages.ranges[stage]:{min:0,max:w.max};let part=`<rect class="aw-focus-background" x="${x-8}" y="${y0-8}" width="${gridWidth+16}" height="${gridHeight+16}" rx="10"/>`;
  if(w.before||w.keys.before){part+=`<g class="aw-faded-context">${w.before?w.keys.indices.map((_,i)=>`<circle cx="${x+(i+.5)*cell}" cy="${y0-18}" r="9"/>`).join(''):''}${w.keys.before?w.indices.map((_,i)=>`<circle cx="${x-18}" cy="${y0+(i+.5)*cell}" r="9"/>`).join(''):''}</g>`;part+=awText(`${w.before} queries ↑ · ${w.keys.before} keys ←`,x+8,y0-32,'aw-context-copy');}
  if(w.after||w.keys.after){part+=`<g class="aw-faded-context">${w.after?w.keys.indices.map((_,i)=>`<circle cx="${x+(i+.5)*cell}" cy="${y0+gridHeight+18}" r="9"/>`).join(''):''}${w.keys.after?w.indices.map((_,i)=>`<circle cx="${x+gridWidth+18}" cy="${y0+(i+.5)*cell}" r="9"/>`).join(''):''}</g>`;part+=awText(`${w.after} more queries ↓ · ${w.keys.after} keys →`,x,y0+gridHeight+43,'aw-context-copy');}
  let cells='';
  w.indices.forEach((q,i)=>w.keys.indices.forEach((k,j)=>{
   const weight=data.head.attention.data[q*data.n+k],value=expanded?stages[stage].data[q*data.n+k]:weight,selected=q===data.query&&k===data.key,tabstop=selected||(i===0&&j===0&&(data.query<w.start||data.query>=w.end||data.key<w.keys.start||data.key>=w.keys.end)),cx=x+(j+.5)*cell,cy=y0+(i+.5)*cell,blocked=!!data.causalComparison&&k>q&&stage!=='dot';
   const color=blocked?'#e3e5eb':stage==='softmax'?`rgba(130,91,201,${.08+.84*Math.sqrt(weight/(range.max||1))})`:signedColor(value,range);
   cells+=`<g role="button" tabindex="${tabstop?0:-1}" data-window-pair="${q}:${k}" data-window-weight="${weight}" data-attention-stage="${stage}" data-stage-value="${value}" data-stage-blocked="${blocked}" class="aw-attention-cell ${selected?'is-selected':''} ${blocked?'is-masked-cell':''}" aria-label="${ATTENTION_STAGE_NAMES[stage]}: query ${label(q)} to key ${label(k)}, ${stage==='softmax'?`${(weight*100).toFixed(3)} percent`:stageNumber(value,5)}${blocked?', masked future key':''}"><rect class="aw-cell-hit" x="${cx-cell/2}" y="${cy-cell/2}" width="${cell}" height="${cell}"/><circle cx="${cx}" cy="${cy}" r="11.5" fill="${color}"/>${blocked?`<path class="aw-mask-slash" d="M${cx-5},${cy+5}L${cx+5},${cy-5}"/>`:''}${selected?`<circle class="aw-selected-ring" cx="${cx}" cy="${cy}" r="14"/>`:''}</g>`;
  }));
  part+=`<rect data-attention-row-anchor x="${x}" y="${y0}" width="1" height="1" fill="none" aria-hidden="true"/>`+viewport(stage,cells,x-1,y0-1,gridWidth+2,gridHeight+2);
  if(expanded){
   const title=stage==='dot'?'Dot product':stage==='scaled'?'Scaling · Mask':'Softmax';
   part+=awText(title,x+8,y0+gridHeight+66,'aw-stage-title');
   const left=stage==='softmax'?'0%':stageNumber(range.min,2),right=stage==='softmax'?`${(range.max*100).toFixed(1)}%`:stageNumber(range.max,2);
   part+=awText(left,x,y0+gridHeight+85,'aw-stage-scale')+`<rect x="${x+66}" y="${y0+gridHeight+79}" width="42" height="5" fill="url(#aw-scale-${stage==='softmax'?'weights':'signed'})"/>`+awText(right,x+118,y0+gridHeight+85,'aw-stage-scale');
   const caption=stage==='dot'?`Q · Kᵀ · ${c.headWidth} terms`:stage==='scaled'?`÷ √${c.headWidth} = ${Math.sqrt(c.headWidth)} · ${data.causalComparison?'causal mask':'no causal mask'}`:'full-row normalization';
   part+=awText(caption,x,y0+gridHeight+103,'aw-stage-caption');
   if(panelIndex<2)svg+=awText('→',x+gridWidth+18,y0+gridHeight/2,'aw-stage-arrow');
  }
  if(!expanded)part+=awText('Attention_weights',x+gridWidth+27,y0+gridHeight/2-20,'aw-stage-title');
  dots+=`<g data-attention-panel="${stage}" class="aw-stage-panel">${part}</g>`;
 });
 const projection=(kind,title,x,y)=>`<g role="button" tabindex="0" data-window-op="${kind}" class="aw-projection-title" aria-label="Inspect ${title} projection">${awText(title+' ↗',x,y)}</g>`;
 svg+=projection('q','Query Q',43,132)+projection('k','Key K',139,15)+projection('v','Value V',139,valueTop-21);
 svg+=awText(data.causalComparison?'Comparison output':'Head output',outX-3,valueTop-30,'aw-output-title')+awText(`features ${outWindow.features.start}–${outWindow.features.end-1}`,outX-3,valueTop-18,'aw-caption');
 svg+=awText(`view ${w.count} × ${outWindow.features.count}`,outX,valueTop+w.count*17+28,'aw-output-dimensions')+awText(`full ${data.n} × ${c.headWidth}`,outX,valueTop+w.count*17+42,'aw-output-dimensions')+awText('rows follow Query · scroll ↕ ↔',outX-12,valueTop+w.count*17+57,'aw-caption');
 svg+=awText(`${data.n} × ${c.headWidth} · scroll ↕`,36,y0+gridHeight+26,'aw-caption')+awText(`keys ${label(w.keys.start)}–${label(w.keys.end-1)} · scroll ↕`,128,141,'aw-caption')+awText('V rows follow the Key window',133,valueTop+w.keys.count*17+20,'aw-caption');
 svg+=`<g role="button" tabindex="0" data-window-breakdown="${expanded?'close':'open'}" class="aw-formula-button" aria-label="${expanded?'Collapse attention calculation':'Expand dot product, scaling and softmax'}" aria-expanded="${expanded}">${awText(expanded?'− Collapse calculation':`Attention + · softmax(Q · Kᵀ / √${c.headWidth})`,expanded?470:219,svgHeight-14)}</g>`;
 svg+=`<rect class="aw-mix-node" x="${mixX}" y="${mixTop}" width="${mixWidth}" height="54" rx="6"/>${awText('Attention_weights × V',mixX+9,mixTop+17,'aw-mix-title')}${awText('weights = softmax probabilities',mixX+9,mixTop+31,'aw-caption')}${awText(`sum ALL ${data.n} keys`,mixX+9,mixTop+44,'aw-caption')}`;
 const queryVisible=data.query>=w.start&&data.query<w.end,keyVisible=data.key>=w.keys.start&&data.key<w.keys.end;
 const mass=`For query ${label(data.query)}: these ${w.count} keys carry ${(w.visibleMass*100).toFixed(1)}% of its attention; ${(w.hiddenMass*100).toFixed(1)}% goes to off-window keys.`;
 const trace=expanded?attentionStageTrace(data):null;
 const calculation=expanded?`<div class="aw-calculation-controls"><span><strong>${data.causalComparison?'Decoder-mask comparison':'Vision encoder'}</strong> · ${data.causalComparison?'future keys blocked; not the actual encoder run':'no causal mask'}</span><label><input type="checkbox" data-attention-causal ${data.causalComparison?'checked':''}>Compare decoder mask</label><button type="button" data-window-breakdown="replay">↻ Replay</button><button type="button" data-window-breakdown="close">− Collapse</button></div>
 <div class="aw-selected-calculation"><span>Selected ${label(data.query)} → ${label(data.key)}</span><code>Q · K = ${stageNumber(trace.raw,5)}</code><code>÷ ${trace.divisor} → ${stageNumber(trace.scaledBeforeMask,5)}${trace.blocked?' → mask: −∞':''}</code><code>softmax = ${trace.numerator.toPrecision(5)} / ${trace.denominator.toPrecision(5)} ≈ ${(trace.probability*100).toFixed(4)}%</code><small>Sum covers all ${data.n} keys, not only these ${w.count}. Inspection dot products use Float64; stored scaled scores and probabilities use FP32.</small></div>`:'';
 return `<div class="aw-window ${expanded?'aw-calculation-expanded':''}" data-window-start="${w.start}" data-window-end="${w.end}" data-key-start="${w.keys.start}" data-key-end="${w.keys.end}" data-output-feature="${outWindow.features.start}" data-window-total="${data.n}" data-calculation-expanded="${expanded}" data-calculation-mode="${data.causalComparison?'causal-comparison':'encoder'}">
 <div class="aw-toolbar"><div><strong>Q ${label(w.start)}–${label(w.end-1)} · K ${label(w.keys.start)}–${label(w.keys.end-1)} <span>· ${w.count} of ${data.n} ${c.cls?'tokens':'patches'}</span></strong><p>Scroll Query and Key independently · left patch list moves both · hover a cell for its ${expanded?'value':'weight'}.</p></div><div class="aw-window-controls"><button type="button" data-window-step="-${w.count}" ${w.before?'':'disabled'} aria-label="Previous patch window">←</button><button type="button" data-window-follow title="Bring the selected query back into view">Follow query</button><button type="button" data-window-step="${w.count}" ${w.after?'':'disabled'} aria-label="Next patch window">→</button></div><div class="aw-overview"><canvas width="84" height="84" data-attention-overview role="img" aria-label="Full ${data.n} by ${data.n} attention matrix. Bright square is the visible patch window. Click to move the window." tabindex="0"></canvas><small>Full ${data.n} × ${data.n}</small></div></div>
 ${data.causalComparison?'<p class="aw-mask-disclosure">Comparison only: Q/K/V are unchanged, but weights and head outputs shown here use a causal mask. Encoder and later-block results are unchanged.</p>':''}
 <div class="aw-diagram-scroll"><div class="aw-diagram-wrap"><svg class="aw-diagram" viewBox="0 0 ${svgWidth} ${svgHeight}" role="group" aria-label="Attention for ${w.count} visible tokens. ${expanded?'Dot product, scaling/mask, then softmax.':'Click a cell to expand its attention calculation.'}"><defs>${motionDefs}<marker id="aw-weights-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#ae93ce"/></marker><linearGradient id="aw-scale-signed"><stop offset="0%" stop-color="#d39174"/><stop offset="50%" stop-color="#f7f4fb"/><stop offset="100%" stop-color="#825bc9"/></linearGradient><linearGradient id="aw-scale-weights"><stop offset="0%" stop-color="#f3eff9"/><stop offset="100%" stop-color="#825bc9"/></linearGradient></defs><g aria-hidden="true" data-motion-viewport="wires">${links}</g>${svg}${dots}</svg></div></div>
 ${calculation}
 <div class="aw-crop-note"><strong>Query rows × key columns · a view, not a smaller calculation.</strong><span>${mass}</span><span>${data.causalComparison?`Softmax uses all ${data.query+1} eligible keys for the selected query; future keys have −∞ scores and zero weight. Grey slashed cells are masked; faded outer context is simply off-window.`:`All ${data.n} keys still participate in softmax and value mixing. Grey context is out of view, not masked.`} Output is a ${w.count} × ${outWindow.features.count} viewing window of the ${data.n} × ${c.headWidth} ${data.causalComparison?'comparison':'full'} head output; its rows follow Query and its feature columns scroll independently.</span></div>
 <div class="aw-window-selection"><span>Query <b>${label(data.query)}</b>${queryVisible?'':' (off-window)'} → key <b>${label(data.key)}</b>${keyVisible?'':' (off-window)'}</span><span>weight <b>${(data.weight*100).toFixed(3)}%</b></span><span>score <b>${stageNumber(data.score,4)}</b></span></div>
 </div>`;
}
/** Only the on-screen square is fully colored; the rest of the real matrix remains faint. */
export function drawAttentionOverview(root,data,start,keyStart=start){
 const canvas=root.querySelector('[data-attention-overview]');if(!canvas)return;const w=attentionWindow(data.n,start),keys=attentionWindow(data.n,keyStart),pixels=overviewPixels(data.head.attention,{maxCols:128,maxRows:128,positive:true});
 const doc=root.ownerDocument,buffer=doc.createElement('canvas');buffer.width=pixels.width;buffer.height=pixels.height;buffer.getContext('2d').putImageData(new ImageData(pixels.data,pixels.width,pixels.height),0,0);
 const g=canvas.getContext('2d'),size=canvas.width;g.clearRect(0,0,size,size);g.imageSmoothingEnabled=false;g.globalAlpha=.23;g.drawImage(buffer,0,0,size,size);g.globalAlpha=1;
 const startPx=keys.start/data.n*size,startY=w.start/data.n*size,length=w.count/data.n*size;g.save();g.beginPath();g.rect(startPx,startY,length,length);g.clip();g.drawImage(buffer,0,0,size,size);g.restore();g.strokeStyle='#936bc2';g.lineWidth=1.5;g.strokeRect(startPx,startY,length,length);
 canvas.dataset.windowStart=w.start;canvas.dataset.keyStart=keys.start;canvas.dataset.windowCount=w.count;
}
/** Scroll observation never changes the selected token or recreates the patch pane. */
export function mountAttentionWindow({root,rail,getData,getStart,onStart,getKeyStart=getStart,onKeyStart=()=>{},getOutputFeature=()=>0,onOutputFeature=()=>{},onPair,onInspect,onOperation,getPresentation=()=>({}),onPresentation=()=>{}}){
 const doc=root.ownerDocument,win=doc.defaultView;let frame=null,disposed=false,renderedStart=-1,renderedKeyStart=-1,renderedFeature=-1,positioning=false,calculationAnimation=null;
 const tooltip=mountAttentionCellTooltip(root,getData),scrollMotion=mountAttentionWindowMotion(root),railAlignment=mountAttentionRailAlignment(root,rail);
 const query=()=>getData().query,axis=()=>rail.scrollWidth>rail.clientWidth+4?'horizontal':'vertical';
 const controls=doc.createElement('div');controls.className='aw-scroll-overlay';
 const names={query:'Scroll query patches independently',key:'Scroll key patches independently',output:'Scroll output rows and query patches',feature:'Scroll head-output feature columns'};
 controls.innerHTML=Object.entries(names).map(([name,label])=>`<div class="aw-native-scroll ${name==='feature'?'aw-native-horizontal':''}" data-window-scroll="${name}" tabindex="0" role="region" aria-label="${label}" title="${label}"><div class="aw-scroll-extent" aria-hidden="true"></div></div>`).join('');
 root.querySelector('.aw-workbench').append(controls);const bars=Object.fromEntries([...controls.querySelectorAll('[data-window-scroll]')].map(el=>[el.dataset.windowScroll,el])),nativeFrames=new Map(),wheelRemainder={};
 function items(){return [...rail.querySelectorAll('[data-full-patch]')];}
 function firstVisible(){const list=items(),base=rail.getBoundingClientRect(),horizontal=axis()==='horizontal';let first=0;for(let i=0;i<list.length;i++){const r=list[i].getBoundingClientRect(),end=horizontal?r.right:r.bottom,start=horizontal?base.left:base.top;if(end>start+Math.max(5,(horizontal?r.width:r.height)/2)){first=i;break;}}return first;}
 function placeControls(){
  if(disposed||!root.isConnected)return;const bench=root.querySelector('.aw-workbench'),diagram=root.querySelector('.aw-diagram-scroll');if(!diagram)return;
  const base=bench.getBoundingClientRect(),bounds=diagram.getBoundingClientRect(),data=getData();
  Object.assign(controls.style,{left:`${bounds.left-base.left}px`,top:`${bounds.top-base.top}px`,width:`${diagram.clientWidth}px`,height:`${diagram.clientHeight}px`});
  for(const [name,bar] of Object.entries(bars)){
   const anchor=root.querySelector(`[data-window-scroll-anchor="${name}"]`);if(!anchor)continue;const r=anchor.getBoundingClientRect(),horizontal=name==='feature',total=horizontal?data.c.headWidth:data.n,count=Math.min(total,horizontal?8:ATTENTION_WINDOW_SIZE),index=name==='key'?renderedKeyStart:horizontal?renderedFeature:renderedStart;
   bar.hidden=total<=count;Object.assign(bar.style,{left:`${r.left-bounds.left}px`,top:`${r.top-bounds.top}px`,width:`${horizontal?r.width:Math.max(12,r.width)}px`,height:`${horizontal?Math.max(12,r.height):r.height}px`});
   const extent=bar.firstElementChild;extent.style.width=horizontal?`${bar.clientWidth+(total-count)*32}px`:'1px';extent.style.height=horizontal?'1px':`${bar.clientHeight+(total-count)*32}px`;
   if(horizontal)bar.scrollLeft=Math.max(0,index)*32;else bar.scrollTop=Math.max(0,index)*32;
  }
 }
 function update(start,keyStart=getKeyStart(),feature=getOutputFeature()){
  if(disposed)return;const data=getData(),w=attentionWindow(data.n,start),keys=attentionWindow(data.n,keyStart),f=attentionWindow(data.c.headWidth,feature,8).start;
  if(renderedStart===w.start&&renderedKeyStart===keys.start&&renderedFeature===f)return;
  tooltip.hide();calculationAnimation?.dispose();calculationAnimation=null;
  const previousStart={query:renderedStart,key:renderedKeyStart};onStart(w.start);onKeyStart(keys.start);onOutputFeature(f);renderedStart=w.start;renderedKeyStart=keys.start;renderedFeature=f;
  const target=root.querySelector('[data-attention-window]'),snapshot=scrollMotion.capture(target,previousStart),diagramScroll=target.querySelector('.aw-diagram-scroll')?.scrollLeft||0;
  target.innerHTML=attentionWindowMarkup(data,{...getPresentation(),attentionStart:w.start,attentionKeyStart:keys.start,attentionOutputFeature:f});drawAttentionOverview(root,data,w.start,keys.start);target.querySelector('.aw-diagram-scroll').scrollLeft=diagramScroll;
  scrollMotion.play(target,previousStart,{query:w.start,key:keys.start},snapshot);
  root.dataset.windowStart=w.start;root.dataset.keyStart=keys.start;root.dataset.windowCount=w.count;root.dataset.outputFeature=f;
  items().forEach((item,i)=>{item.classList.toggle('is-attention-window',i>=w.start&&i<w.end);item.classList.toggle('is-key-window',i>=keys.start&&i<keys.end);});railAlignment.schedule();placeControls();
 }
 function align(start,linked=true){const data=getData(),w=attentionWindow(data.n,start),list=items(),item=list[w.start];if(!item)return;positioning=true;const r=item.getBoundingClientRect(),base=rail.getBoundingClientRect();if(axis()==='horizontal')rail.scrollLeft+=r.left-base.left;else rail.scrollTop+=r.top-base.top;update(w.start,linked?w.start:getKeyStart());positioning=false;}
 function scroll(){if(disposed||positioning)return;if(frame!==null)win.cancelAnimationFrame(frame);frame=win.requestAnimationFrame(()=>{frame=null;if(disposed)return;const first=firstVisible();if(first!==renderedStart)update(first,first);});}
 function moveAxis(name,start){if(name==='key')update(renderedStart,start);else if(name==='feature')update(renderedStart,renderedKeyStart,start);else align(start,false);}
 function axisPosition(name){return name==='key'?renderedKeyStart:name==='feature'?renderedFeature:renderedStart;}
 function nativeScroll(e){
  if(e.target.matches('.aw-diagram-scroll')){placeControls();return;}
  const name=e.target.dataset.windowScroll;if(!name)return;const bar=bars[name],value=Math.round((name==='feature'?bar.scrollLeft:bar.scrollTop)/32);if(value===axisPosition(name))return;
  if(nativeFrames.has(name))win.cancelAnimationFrame(nativeFrames.get(name));nativeFrames.set(name,win.requestAnimationFrame(()=>{nativeFrames.delete(name);if(!disposed)moveAxis(name,Math.round((name==='feature'?bar.scrollLeft:bar.scrollTop)/32));}));
 }
 function wheel(e){const pane=e.target.closest('[data-window-axis]');if(!pane)return;let name=pane.dataset.windowAxis;const horizontal=name==='output'&&(e.shiftKey||Math.abs(e.deltaX)>Math.abs(e.deltaY));if(horizontal)name='feature';
  const delta=(horizontal?(e.shiftKey?e.deltaY:e.deltaX):e.deltaY)*(e.deltaMode===1?18:e.deltaMode===2?180:1);if(!delta)return;e.preventDefault();wheelRemainder[name]=(wheelRemainder[name]||0)+delta;const step=Math.trunc(wheelRemainder[name]/32);if(step){wheelRemainder[name]-=step*32;moveAxis(name,axisPosition(name)+step);}}
 function pair(q,k,stage='softmax'){const data=getData();onPair(Math.max(0,Math.min(data.n-1,q)),Math.max(0,Math.min(data.n-1,k)));(doc.querySelector(`[data-window-pair="${q}:${k}"][data-attention-stage="${stage}"]`)||doc.querySelector('[data-attention-overview]'))?.focus({preventScroll:true});}
 function present(expanded){tooltip.hide();onPresentation({attentionBreakdown:expanded,...(!expanded?{attentionCausal:false}:{})});const fresh=doc.querySelector('.sa-windowed');if(fresh){const reduced=win.matchMedia('(prefers-reduced-motion: reduce)').matches,animation=animateAttentionStages(fresh,{expanded,reduced});fresh.dispatchEvent(new win.CustomEvent('attention-calculation-animation',{detail:animation}));}}
 function activate(target,event){
  const toggle=target.closest('[data-window-breakdown]');if(toggle){present(toggle.dataset.windowBreakdown!=='close');doc.querySelector(`[data-window-breakdown="${toggle.dataset.windowBreakdown==='close'?'open':'close'}"]`)?.focus({preventScroll:true});return true;}
  const p=target.closest('[data-window-pair]');if(p){const [q,k]=p.dataset.windowPair.split(':').map(Number),wasExpanded=!!getPresentation().attentionBreakdown;pair(q,k,p.dataset.attentionStage||'softmax');if(!wasExpanded)present(true);return true;}
  const token=target.closest('[data-window-token]');if(token){const data=getData(),i=+token.dataset.windowToken;pair(token.dataset.windowKind==='q'||token.dataset.windowKind==='out'?i:data.query,token.dataset.windowKind==='q'||token.dataset.windowKind==='out'?data.key:i);return true;}
  const operation=target.closest('[data-window-op]');if(operation){onOperation(operation.dataset.windowOp,operation);return true;}
  const inspect=target.closest('[data-window-inspect]');if(inspect){onInspect(inspect.dataset.windowInspect);return true;}
  if(target.closest('[data-window-follow]')){align(attentionWindowForToken(getData().n,null,query()).start,false);return true;}
  const step=target.closest('[data-window-step]');if(step){align(getStart()+ +step.dataset.windowStep);root.querySelector(`[data-window-step="${step.dataset.windowStep}"]`)?.focus({preventScroll:true});return true;}
  if(target.matches('[data-attention-overview]')&&event?.type==='click'){const b=target.getBoundingClientRect(),q=Math.floor((event.clientY-b.top)/b.height*getData().n)-Math.floor(ATTENTION_WINDOW_SIZE/2),k=Math.floor((event.clientX-b.left)/b.width*getData().n)-Math.floor(ATTENTION_WINDOW_SIZE/2);onKeyStart(attentionWindow(getData().n,k).start);align(q,false);return true;}
  return false;
 }
 function click(e){activate(e.target,e);}
 function keydown(e){
  const cell=e.target.closest('[data-window-pair]');if(cell){const [q,k]=cell.dataset.windowPair.split(':').map(Number),next=attentionWindowKey(q,k,e.key,getData().n);if(next){e.preventDefault();pair(next.query,next.key,cell.dataset.attentionStage||'softmax');return;}}
  const pane=e.target.matches('[data-window-axis], [data-window-scroll]')?e.target:null;
  if(pane&&['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','PageUp','PageDown','Home','End'].includes(e.key)){
   e.preventDefault();let name=pane.dataset.windowAxis||pane.dataset.windowScroll;if(name==='output'&&['ArrowLeft','ArrowRight'].includes(e.key))name='feature';
   const total=name==='feature'?getData().c.headWidth:getData().n,count=name==='feature'?8:ATTENTION_WINDOW_SIZE,delta={ArrowUp:-1,ArrowLeft:-1,ArrowDown:1,ArrowRight:1,PageUp:-count,PageDown:count}[e.key];
   moveAxis(name,e.key==='Home'?0:e.key==='End'?total-count:axisPosition(name)+delta);if(pane.dataset.windowAxis)root.querySelector(`[data-window-axis="${pane.dataset.windowAxis}"]`)?.focus({preventScroll:true});return;
  }
  if(e.target.matches('[data-attention-overview]')&&['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End'].includes(e.key)){
   e.preventDefault();if(e.key==='Home'||e.key==='End')align(e.key==='Home'?0:getData().n-ATTENTION_WINDOW_SIZE);else moveAxis(['ArrowLeft','ArrowRight'].includes(e.key)?'key':'query',(['ArrowLeft','ArrowRight'].includes(e.key)?renderedKeyStart:renderedStart)+(['ArrowDown','ArrowRight'].includes(e.key)?1:-1));root.querySelector('[data-attention-overview]')?.focus({preventScroll:true});return;
  }
  if(['Enter',' '].includes(e.key)&&e.target.closest('svg [role="button"]')){e.preventDefault();activate(e.target,e);}
 }
 const change=e=>{if(e.target.hasAttribute('data-attention-causal')){onPresentation({attentionCausal:e.target.checked,attentionBreakdown:true});doc.querySelector('[data-attention-causal]')?.focus({preventScroll:true});}};
 const acceptAnimation=e=>{calculationAnimation?.dispose();calculationAnimation=e.detail;railAlignment.schedule();placeControls();};
 const resize=()=>{scrollMotion.cancel();align(getStart(),false);placeControls();};const observer=new win.ResizeObserver(placeControls);observer.observe(root.querySelector('.aw-workbench'));
 root.addEventListener('attention-calculation-animation',acceptAnimation);root.addEventListener('change',change);root.addEventListener('click',click);root.addEventListener('keydown',keydown);root.addEventListener('wheel',wheel,{passive:false});root.addEventListener('scroll',nativeScroll,true);rail.addEventListener('scroll',scroll,{passive:true});win.addEventListener('resize',resize);
 align(getStart(),false);
 return {align,dispose(){disposed=true;scrollMotion.dispose();calculationAnimation?.dispose();tooltip.dispose();railAlignment.dispose();observer.disconnect();controls.remove();if(frame!==null)win.cancelAnimationFrame(frame);for(const id of nativeFrames.values())win.cancelAnimationFrame(id);root.removeEventListener('attention-calculation-animation',acceptAnimation);root.removeEventListener('change',change);root.removeEventListener('click',click);root.removeEventListener('keydown',keydown);root.removeEventListener('wheel',wheel);root.removeEventListener('scroll',nativeScroll,true);rail.removeEventListener('scroll',scroll);win.removeEventListener('resize',resize);}};
}
