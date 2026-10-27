/** View-only scrolling. Global patch/cell identities survive each six-patch redraw. */
export const ATTENTION_PAN_DURATION=360;
const awMotionSelector='[data-window-token], [data-window-pair], [data-window-wire]';
const awPathNumbers=/-?\d*\.?\d+(?:e[-+]?\d+)?/gi;
function awPanStart(start){return typeof start==='number'?{query:start,key:start}:start;}
function awPanDelta(from,to){const a=awPanStart(from),b=awPanStart(to);return {query:a.query-b.query,key:a.key-b.key};}
export function attentionPanOffset(kind,delta,pitch=kind==='k'||kind==='v'?17:29){
 const d=awPanStart(delta);return {x:kind==='cell'?d.key*pitch:0,y:(kind==='k'||kind==='v'?d.key:d.query)*pitch};
}
export function interpolateAttentionPath(from,to,progress){
 const a=from.match(awPathNumbers)?.map(Number)||[],b=to.match(awPathNumbers)?.map(Number)||[];
 if(a.length!==b.length||from.replace(awPathNumbers,'#')!==to.replace(awPathNumbers,'#'))return to;
 let i=0;return to.replace(awPathNumbers,()=>{const index=i++;return String(a[index]+(b[index]-a[index])*progress);});
}
function awMotionRecord(node,start){
 const d=node.dataset,wire=d.windowWire!==undefined,cell=d.windowPair!==undefined;
 const kind=wire?d.windowWire.split('-')[0]:cell?'cell':d.windowKind;
 const key=wire?`wire:${d.windowWire}`:cell?`cell:${d.attentionStage}:${d.windowPair}`:`token:${kind}:${d.windowToken}`;
 const hit=wire?null:node.querySelector(cell?'.aw-cell-hit':'.aw-token-hit');
 return {node,key,kind,wire,start,pitch:+d.windowPitch||(kind==='k'||kind==='v'?17:29),viewport:node.parentNode.dataset.motionViewport,
  x:hit?+hit.getAttribute('x'):0,y:hit?+hit.getAttribute('y'):0,path:wire?node.getAttribute('d'):null,dx:0,dy:0,alpha:1};
}
/** Ghosts are visual only: never discoverable as a patch, a cell or a focus target. */
export function sanitizeAttentionGhost(node){
 for(const el of [node,...node.querySelectorAll('*')]){
  for(const attr of [...el.attributes])if(attr.name.startsWith('data-')||attr.name.startsWith('aria-')||['id','role','tabindex'].includes(attr.name))el.removeAttribute(attr.name);
  el.classList.remove('is-tooltip-active');
 }
 node.setAttribute('aria-hidden','true');node.setAttribute('inert','');node.classList.add('aw-motion-ghost');
 node.querySelectorAll('title').forEach(title=>title.remove());return node;
}
export function mountAttentionWindowMotion(root){
 const doc=root.ownerDocument,win=doc.defaultView,media=win.matchMedia('(prefers-reduced-motion: reduce)');
 let frame=null,records=null,disposed=false;
 const enabled=()=>!disposed&&!doc.hidden&&!media.matches;
 function paint(r){
  if(r.wire){r.node.setAttribute('d',r.currentPath??r.path);r.node.style.strokeOpacity=String(r.alpha);}
  else r.node.style.opacity=String(r.alpha);
  r.node.setAttribute('transform',`translate(${r.dx} ${r.dy})`);
 }
 function cancel(){
  if(frame!==null)win.cancelAnimationFrame(frame);frame=null;
  for(const r of records||[]){if(r.ghost){r.node.remove();continue;}
   r.node.removeAttribute('transform');r.node.style.removeProperty(r.wire?'stroke-opacity':'opacity');if(r.wire)r.node.setAttribute('d',r.path);
  }
  records=null;root.removeAttribute('data-window-panning');
 }
 function capture(target,start){
  // Capture the current interpolated pose BEFORE cancelling. Rapid scroll/reversal
  // continues from the on-screen position, not the previous animation's endpoint.
  const snapshot=enabled()?new Map((records||[...target.querySelectorAll(awMotionSelector)].map(node=>awMotionRecord(node,start))).map(r=>[r.key,{...r,currentPath:r.currentPath??r.path}])):null;
  cancel();return snapshot;
 }
 function play(target,previousStart,nextStart,snapshot){
  const previous=awPanStart(previousStart),nextPosition=awPanStart(nextStart),delta=awPanDelta(nextPosition,previous);
  if(!enabled()||previous.query<0||(!delta.query&&!delta.key)||!snapshot?.size)return;
  const bound=n=>Math.sign(n)*Math.min(Math.abs(n),6),enterDelta={query:bound(delta.query),key:bound(delta.key)};
  const next=[...target.querySelectorAll(awMotionSelector)].map(node=>awMotionRecord(node,nextStart)),keys=new Set(next.map(r=>r.key));
  const viewports=new Map([...target.querySelectorAll('[data-motion-viewport]')].map(node=>[node.dataset.motionViewport,node]));
  records=next.map(r=>{
   const old=snapshot.get(r.key),offset=attentionPanOffset(r.kind,enterDelta,r.pitch);
   return {...r,fromX:old?old.x+old.dx-r.x:r.wire?0:offset.x,fromY:old?old.y+old.dy-r.y:r.wire?0:offset.y,toX:0,toY:0,
    fromPath:old?.currentPath??r.path,fromAlpha:old?.alpha??0,toAlpha:1};
  });
  for(const old of snapshot.values())if(!keys.has(old.key)&&old.alpha>.015){
   const parent=viewports.get(old.viewport);if(!parent)continue;
   const offset=attentionPanOffset(old.kind,awPanDelta(old.start,nextStart),old.pitch);
   const node=sanitizeAttentionGhost(old.node.cloneNode(true));parent.append(node);
   // Ghost wires fade in place. Matched wires morph their actual Bezier points,
   // keeping Q/K/V endpoints attached to the moving rows without moving the mixer.
   records.push({...old,node,ghost:true,fromX:old.dx,fromY:old.dy,toX:old.wire?0:offset.x,toY:old.wire?0:offset.y,
    fromPath:old.currentPath,fromAlpha:old.alpha,toAlpha:0});
  }
  root.dataset.windowPanning=(delta.query||delta.key)>0?'forward':'backward';
  const started=win.performance.now();
  function tick(now){
   frame=null;if(!enabled()||!root.isConnected){cancel();return;}
   const t=Math.max(0,Math.min(1,(now-started)/ATTENTION_PAN_DURATION)),p=1-Math.pow(1-t,3);
   for(const r of records){r.dx=r.fromX+(r.toX-r.fromX)*p;r.dy=r.fromY+(r.toY-r.fromY)*p;r.alpha=r.fromAlpha+(r.toAlpha-r.fromAlpha)*p;
    if(r.wire)r.currentPath=interpolateAttentionPath(r.fromPath,r.path,p);paint(r);
   }
   if(t===1)cancel();else frame=win.requestAnimationFrame(tick);
  }
  tick(started);
 }
 const visibility=()=>{if(!enabled())cancel();};
 doc.addEventListener('visibilitychange',visibility);media.addEventListener('change',visibility);
 return {capture,play,cancel,dispose(){disposed=true;cancel();doc.removeEventListener('visibilitychange',visibility);media.removeEventListener('change',visibility);}};
}
