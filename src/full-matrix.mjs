import {isWeightMatrix,matrixRowOffset,matrixRowSummary,matrixRowGuides} from './matrix-row-guides.mjs';
import {matrixValue,featureWindow} from './full-config.mjs';
export function activationRGB(value,scale=2){const t=Math.min(1,Math.abs(value)/scale),rgb=value<0?[204,143,115]:[98,135,209];return rgb.map(v=>Math.round(248+(v-248)*t));}
export function overviewPixels(a,{maxCols=256,maxRows=160,scale=2,positive=false}={}){
 const width=Math.min(a.cols,maxCols),height=Math.min(a.rows,maxRows),data=new Uint8ClampedArray(width*height*4),sampled=a.kind==='weights'||a.kind==='view'&&a.source.kind==='weights';
 let peak=1;if(positive){peak=0;for(let r=0;r<a.rows;r++)for(let c=0;c<a.cols;c++)peak=Math.max(peak,matrixValue(a,r,c));}
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const r0=Math.floor(y*a.rows/height),r1=Math.floor((y+1)*a.rows/height),c0=Math.floor(x*a.cols/width),c1=Math.floor((x+1)*a.cols/width);let v=0,n=0;
  if(sampled){v=matrixValue(a,Math.min(a.rows-1,Math.floor((r0+r1)/2)),Math.min(a.cols-1,Math.floor((c0+c1)/2)));}
  else{for(let r=r0;r<r1;r++)for(let c=c0;c<c1;c++){v+=matrixValue(a,r,c);n++;}v/=n||1;}
  const rgb=positive?[54,103,228].map(k=>Math.round(248+(k-248)*Math.sqrt(Math.max(0,v)/(peak||1)))):activationRGB(v,scale);const i=(y*width+x)*4;data.set([...rgb,255],i);
 }
 return {data,width,height,mode:sampled?'sampled weight overview':width<a.cols||height<a.rows?'pooled overview (mean per pixel)':'one cell per value'};
}
export function selectedWindow(a,row,index=0,size=32){const range=featureWindow(a.cols,index,size);return {...range,row,values:Array.from({length:range.end-range.start},(_,i)=>matrixValue(a,row,range.start+i))};}
const matrixEscape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function createMatrixRenderer(prefix='fm'){
 let entries=new Map(),count=0,patchOffset=0;const cache=new WeakMap();
 function imageFor(a,options,doc){let images=cache.get(a);if(!images){images=new Map();cache.set(a,images);}const key=JSON.stringify(options);if(!images.has(key)){const p=overviewPixels(a,options),image=doc.createElement('canvas');image.width=p.width;image.height=p.height;image.getContext('2d').putImageData(new ImageData(p.data,p.width,p.height),0,0);images.set(key,{image,mode:p.mode});}return images.get(key);}
 function markup(a,{label='Matrix',row=-1,col=-1,height=190,click=null,positive=false,scale=2,compact=false,anchor='',rowOffset=patchOffset}={}){
  const id=`${prefix}-${++count}`;entries.set(id,{a,label,row,col,click,positive,scale,rowOffset,compact});
  const patchRows=!isWeightMatrix(a),offset=matrixRowOffset(a),summary=patchRows?matrixRowSummary(a.rows,rowOffset,offset):`${a.rows.toLocaleString()} input-feature rows`;
  return `<div ${anchor?`data-full-anchor="${matrixEscape(anchor)}"`:""} class="fm-matrix ${compact?'fm-compact':''}"><div class="fm-label"><strong>${matrixEscape(label)}</strong><span>${a.rows.toLocaleString()} × ${a.cols.toLocaleString()}</span></div><div class="fm-matrix-plot ${patchRows&&!compact?'has-row-ruler':''}">${patchRows&&!compact?`<div class="fm-row-ruler" data-row-ruler="${id}" aria-hidden="true"></div>`:''}<canvas data-full-matrix="${id}" tabindex="${click?0:-1}" role="img" aria-label="${matrixEscape(label)}: ${summary}, ${a.cols.toLocaleString()} columns${click?'. Click to select; arrows move one row or feature.':''}" style="height:${height}px"></canvas></div>${compact?'':'<small class="fm-matrix-mode" data-matrix-mode="'+id+'"></small>'}</div>`;
 }
 function drawCanvas(canvas,entry){const {a,row,col,scale,positive,rowOffset=0}=entry,rect=canvas.getBoundingClientRect(),ratio=Math.min(2,globalThis.devicePixelRatio||1),w=Math.max(1,Math.round(rect.width*ratio)),h=Math.max(1,Math.round(rect.height*ratio));if(canvas.width!==w)canvas.width=w;if(canvas.height!==h)canvas.height=h;const g=canvas.getContext('2d'),image=imageFor(a,{maxCols:Math.min(512,w),maxRows:Math.min(256,h),scale,positive},canvas.ownerDocument);g.imageSmoothingEnabled=false;g.clearRect(0,0,w,h);g.drawImage(image.image,0,0,w,h);
  const patchRows=!isWeightMatrix(a),offset=matrixRowOffset(a),guides=patchRows?matrixRowGuides(a.rows,rect.height,{selected:row,clsOffset:rowOffset,offset}):null;
  if(guides){
   g.save();
   for(const line of guides.lines){const y=Math.min(h-.5,Math.max(.5,Math.round(line.y*ratio)+.5));g.strokeStyle=line.major?'rgba(47,68,98,.48)':'rgba(47,68,98,.23)';g.lineWidth=line.major?Math.max(1,ratio*.75):Math.max(.5,ratio*.45);g.beginPath();g.moveTo(0,y);g.lineTo(w,y);g.stroke();}
   g.restore();
   const ruler=canvas.parentElement.querySelector('[data-row-ruler]');
   if(ruler){ruler.innerHTML=guides.lines.map(line=>`<i class="fm-row-tick ${line.major?'is-major':''}" style="top:${line.y/rect.height*100}%"></i>`).join('')+guides.labels.map(label=>`<span class="fm-row-label ${label.selected?'is-selected':''}" style="top:${label.y/rect.height*100}%">${label.label}</span>`).join('');}
   canvas.dataset.rowGuides=guides.lines.length;canvas.dataset.rowGuideStride=guides.stride;
  }else{delete canvas.dataset.rowGuides;delete canvas.dataset.rowGuideStride;}
  if(row>=0){g.lineWidth=1.8*ratio;g.strokeStyle='#274f94';g.strokeRect(.5*ratio,row*h/a.rows+.5*ratio,w-ratio,Math.max(ratio,h/a.rows-ratio));}
  if(col>=0){g.strokeStyle='#9767b9';g.lineWidth=ratio;g.strokeRect(col*w/a.cols,.5*ratio,Math.max(1,w/a.cols),h-ratio);if(row>=0){g.strokeStyle='#c39435';g.lineWidth=2*ratio;g.strokeRect(col*w/a.cols,row*h/a.rows,Math.max(2*ratio,w/a.cols),Math.max(2*ratio,h/a.rows));}}
  const caption=canvas.closest('.fm-matrix')?.querySelector('[data-matrix-mode]');if(caption)caption.textContent=patchRows?`${matrixRowSummary(a.rows,rowOffset,offset)} ↓ · ${image.mode} · ${guides.stride===1?'one line per row; darker every 4':'row guides thinned for legibility'} · ${positive?'keys':'features'} →`:`${image.mode} · input features ↓ / output features →`;
 }
 function draw(root){root.querySelectorAll('[data-full-matrix]').forEach(canvas=>{const entry=entries.get(canvas.dataset.fullMatrix);if(entry)drawCanvas(canvas,entry);});}
 function pick(event,root){const canvas=event.target.closest('[data-full-matrix]');if(!canvas||!root.contains(canvas))return false;const e=entries.get(canvas.dataset.fullMatrix);if(!e?.click)return false;let r=e.row<0?0:e.row,c=e.col<0?0:e.col;
  if(event.type==='keydown'){const delta={ArrowUp:[-1,0],ArrowDown:[1,0],ArrowLeft:[0,-1],ArrowRight:[0,1]}[event.key];if(delta){r+=delta[0];c+=delta[1];}else if(event.key==='Home')c=0;else if(event.key==='End')c=e.a.cols-1;else return false;event.preventDefault();}
  else{const b=canvas.getBoundingClientRect();r=Math.floor((event.clientY-b.top)/b.height*e.a.rows);c=Math.floor((event.clientX-b.left)/b.width*e.a.cols);}
  const id=canvas.dataset.fullMatrix;e.click(Math.max(0,Math.min(e.a.rows-1,r)),Math.max(0,Math.min(e.a.cols-1,c)));root.querySelector(`[data-full-matrix="${id}"]`)?.focus({preventScroll:true});return true;
 }
 function hover(event){const canvas=event.target.closest('[data-full-matrix]');if(!canvas)return;const e=entries.get(canvas.dataset.fullMatrix);if(!e)return;const b=canvas.getBoundingClientRect(),r=Math.min(e.a.rows-1,Math.max(0,Math.floor((event.clientY-b.top)/b.height*e.a.rows))),c=Math.min(e.a.cols-1,Math.max(0,Math.floor((event.clientX-b.left)/b.width*e.a.cols)));canvas.title=`${e.label}[${r}, ${c}] = ${matrixValue(e.a,r,c).toPrecision(8)} · actual coordinate (overview may be pooled)`;}
 return {markup,draw,pick,hover,setPatchOffset(value){patchOffset=value?1:0;},reset(){entries.clear();count=0;},highlight(root,id,row,col){const canvas=root.querySelector(`[data-full-matrix="${id}"]`),e=entries.get(id);if(e){e.row=row;e.col=col;if(canvas)drawCanvas(canvas,e);}},get count(){return entries.size;}};
}
export function featureMarkup(a,row,index=0,{label='Selected row',key='feature'}={}){
 const win=selectedWindow(a,row,index);return `<section class="fm-features"><div class="fm-feature-toolbar"><strong>${matrixEscape(label)}</strong><span>features ${win.start}–${win.end-1} of ${win.total.toLocaleString()}</span><label>Feature <input type="number" data-full-feature="${key}" min="0" max="${a.cols-1}" value="${win.selected}" title="Enter a feature index, then press Enter" aria-label="${matrixEscape(label)} feature index"></label><button type="button" data-full-page="${key}:${Math.max(0,win.start-32)}" ${win.start===0?'disabled':''} aria-label="Previous feature range">←</button><button type="button" data-full-page="${key}:${Math.min(a.cols-1,win.start+32)}" ${win.end===a.cols?'disabled':''} aria-label="Next feature range">→</button></div><div class="fm-feature-values">${win.values.map((v,i)=>`<button type="button" data-full-page="${key}:${win.start+i}" class="${win.start+i===win.selected?'is-selected':''}" style="background:rgb(${activationRGB(v).join(',')})" title="Feature ${win.start+i}: ${v.toPrecision(9)}"><small>${win.start+i}</small>${v.toFixed(4)}</button>`).join('')}</div><p class="fm-caption">Exact values in this range. Use the feature index or arrows to inspect all ${win.total.toLocaleString()} features—none are omitted from computation.</p></section>`;
}
