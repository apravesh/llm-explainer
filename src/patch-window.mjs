/** Shared attention/MLP viewport; embedding and output remain full-length. */
export const PATCH_WINDOW_SIZE=6;
export function patchWindow(total,start=0,size=PATCH_WINDOW_SIZE){
 const count=Math.min(total,size),first=Math.max(0,Math.min(total-count,Math.trunc(start)||0));
 return {start:first,end:first+count,count,total,indices:Array.from({length:count},(_,i)=>first+i),before:first,after:total-first-count};
}
export function patchWindowLabel(win,cls=false){
 const label=i=>cls&&i===0?'CLS':`p${String(i-(cls?1:0)).padStart(2,'0')}`;
 return `${label(win.start)}–${label(win.end-1)} · ${win.count} of ${win.total}`;
}
export function matrixPatchWindow(a,setting){
 // Explicit one-row calculations and feature/weight matrices keep their own scope.
 if(!setting||a.kind==='weights'||a.rows!==setting.total)return {start:0,end:a.rows,count:a.rows,total:a.rows};
 return patchWindow(a.rows,setting.start);
}
/** Keep the list DOM alive during scrolling; do not select a different patch. */
export function mountSharedPatchWindow({root,rail,getStart,total,cls=false,onStart}){
 const win=root.ownerDocument.defaultView;let frame=null,disposed=false;
 const list=()=>[...rail.querySelectorAll('[data-full-patch]')],horizontal=()=>win.getComputedStyle(rail).flexDirection==='row';
 function mark(){const range=patchWindow(total,getStart());list().forEach(item=>item.classList.toggle('is-patch-window',Number(item.dataset.fullPatch)+(cls?1:0)>=range.start&&Number(item.dataset.fullPatch)+(cls?1:0)<range.end));}
 function align(){const range=patchWindow(total,getStart()),items=list(),item=items.find(el=>Number(el.dataset.fullPatch)+(cls?1:0)===range.start)||items[0];if(!item)return;const r=item.getBoundingClientRect(),base=rail.getBoundingClientRect();if(horizontal())rail.scrollLeft+=r.left-base.left;else rail.scrollTop+=r.top-base.top;mark();}
 function scroll(){if(frame!==null)win.cancelAnimationFrame(frame);frame=win.requestAnimationFrame(()=>{frame=null;if(disposed)return;const base=rail.getBoundingClientRect(),h=horizontal(),item=list().find(el=>{const r=el.getBoundingClientRect();return (h?r.right:r.bottom)>(h?base.left:base.top)+2;});if(!item)return;const next=patchWindow(total,Number(item.dataset.fullPatch)+(cls?1:0)).start;if(next!==getStart())onStart(next);mark();});}
 function key(e){if(e.target!==rail||!['ArrowDown','ArrowUp','ArrowLeft','ArrowRight','PageDown','PageUp','Home','End'].includes(e.key))return;e.preventDefault();const delta={ArrowDown:1,ArrowRight:1,ArrowUp:-1,ArrowLeft:-1,PageDown:6,PageUp:-6}[e.key];const next=patchWindow(total,e.key==='Home'?0:e.key==='End'?total:getStart()+delta).start;if(next!==getStart())onStart(next);align();}
 rail.addEventListener('scroll',scroll,{passive:true});rail.addEventListener('keydown',key);win.addEventListener('resize',align);align();
 return {align,dispose(){disposed=true;if(frame!==null)win.cancelAnimationFrame(frame);rail.removeEventListener('scroll',scroll);rail.removeEventListener('keydown',key);win.removeEventListener('resize',align);}};
}
/** Shared pan feedback for canvases and the MLP/attention mini-diagrams. */
export function createPatchWindowMotion(root){
 const win=root.ownerDocument.defaultView,media=win.matchMedia('(prefers-reduced-motion: reduce)');let animations=[];
 function cancel(){animations.forEach(a=>a.cancel());animations=[];}
 function play(delta){cancel();if(!delta||media.matches||root.ownerDocument.hidden)return;
  const nodes=[...root.querySelectorAll('canvas[data-patch-window-start], [data-compact-attention-svg]')];
  nodes.forEach(el=>{if(!el.getBoundingClientRect().width||typeof el.animate!=='function')return;const distance=Math.sign(delta)*Math.min(2,Math.abs(delta))*Math.min(20,el.getBoundingClientRect().height/6);animations.push(el.animate([{transform:`translateY(${distance}px)`,opacity:.45},{transform:'translateY(0)',opacity:1}],{duration:320,easing:'cubic-bezier(.2,.7,.2,1)'}));});
 }
 const hidden=()=>{if(root.ownerDocument.hidden)cancel();};root.ownerDocument.addEventListener('visibilitychange',hidden);win.addEventListener('resize',cancel);media.addEventListener('change',cancel);
 return {play,cancel,dispose(){cancel();root.ownerDocument.removeEventListener('visibilitychange',hidden);win.removeEventListener('resize',cancel);media.removeEventListener('change',cancel);}};
}
