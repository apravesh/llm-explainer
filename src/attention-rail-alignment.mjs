/** Align the patch list with the first visible attention row, not the section heading. */
export function attentionRailOffset(currentOffset,listTop,matrixTop,horizontal=false){
 if(horizontal)return 0;
 if(![currentOffset,listTop,matrixTop].every(Number.isFinite))return Math.max(0,currentOffset||0);
 return Math.max(0,currentOffset+matrixTop-listTop);
}
export function mountAttentionRailAlignment(root,rail){
 const win=root.ownerDocument.defaultView,aside=rail.closest('.attention-window-rail');
 if(!aside)return {schedule(){},dispose(){}};
 const mobile=win.matchMedia('(max-width: 650px)');let offset=0,frame=null,disposed=false;
 function align(){
  frame=null;if(disposed||!root.isConnected||!rail.isConnected)return;
  const cell=root.querySelector('[data-attention-row-anchor]')||root.querySelector('.aw-attention-cell .aw-cell-hit');if(!cell)return;
  const next=attentionRailOffset(offset,rail.getBoundingClientRect().top,cell.getBoundingClientRect().top,mobile.matches);
  if(Math.abs(next-offset)<.5)return;offset=next;aside.style.setProperty('--attention-rail-offset',`${offset}px`);
 }
 function schedule(){if(disposed)return;if(frame!==null)win.cancelAnimationFrame(frame);frame=win.requestAnimationFrame(align);}
 const observer=new win.ResizeObserver(schedule);
 observer.observe(root);const inspection=root.closest('.expansion-inspection');if(inspection)observer.observe(inspection);
 const heading=aside.querySelector('.token-rail-heading');if(heading)observer.observe(heading);
 win.addEventListener('resize',schedule);mobile.addEventListener('change',schedule);schedule();
 return {schedule,dispose(){disposed=true;if(frame!==null)win.cancelAnimationFrame(frame);observer.disconnect();win.removeEventListener('resize',schedule);mobile.removeEventListener('change',schedule);aside.style.removeProperty('--attention-rail-offset');}};
}
