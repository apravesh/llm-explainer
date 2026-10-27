/** Spatial continuity for architecture expansion, without changing model state or math. */
export const ZOOM_TIMING = Object.freeze({in:1050,out:850,switch:750});
export function zoomDirection(from,to){return from===null?'in':to===null?'out':'switch';}
export function boxTransform(from,to){
 if(!from||!to||![from.x,from.y,from.width,from.height,to.x,to.y,to.width,to.height].every(Number.isFinite)||from.width<=0||from.height<=0||to.width<=0||to.height<=0)return null;
 return `translate(${from.x-to.x}px, ${from.y-to.y}px) scale(${from.width/to.width}, ${from.height/to.height})`;
}

export function createArchitectureMotion(root,getExpanded){
 const doc=root.ownerDocument,win=doc.defaultView,html=doc.documentElement;
 const reduced=win.matchMedia('(prefers-reduced-motion: reduce)');
 let current=null;
 const query=selector=>root.querySelector(selector);
 const visible=el=>el&&el.getBoundingClientRect().width>0&&el.getBoundingClientRect().height>0;
 function nodes(expanded,focus){
  const map=new Map([['encoder-diagram',root]]);
  root.querySelectorAll('.stage-title').forEach((el,i)=>map.set(`encoder-title-${i}`,el));
  const stage=query(`[data-stage="${expanded??focus}"]`);
  const panel=stage?.querySelector(expanded===null?'.stage-content':'.stage-expansion');
  if(visible(panel))map.set('encoder-focus',panel);
  const sequence=query(expanded===null?'#patch-sequence':'.token-rail');
  if(visible(sequence))map.set('encoder-sequence',sequence);
  // Carry the same actual image/patch across the two views, not a decorative proxy.
  const hero=expanded===null
   ? focus===0?query('#image-grid'):focus===1?query('.sequence-item[aria-pressed="true"] .sequence-thumb'):null
   : expanded===0?query('.expanded-image-grid'):expanded===1?query('.pixel-grid'):null;
  if(visible(hero))map.set('encoder-patch',hero);
  const output=expanded===null&&focus===5?(query('.ro-compact-matrix')||query('[data-full-anchor="output-compact"]')):expanded===5?(query('.ro-output-matrix')||query('[data-full-anchor="output-expanded"]')):null;
  if(visible(output))map.set('encoder-output',output);
  const embedding=expanded===null&&focus===2?query('#embedding-matrix'):expanded===2?(query('#expanded-embedding-matrix')||query('[data-full-anchor="embedding-expanded"]')):null;
  if(visible(embedding))map.set('encoder-embedding',embedding);
  return new Map([...map].filter(([,el])=>visible(el)));
 }
 function clearNames(record){for(const [el,previous] of record.names)el.style.viewTransitionName=previous;record.names.clear();}
 function nameNodes(record,map){clearNames(record);for(const [name,el] of map){record.names.set(el,el.style.viewTransitionName);el.style.viewTransitionName=name;}}
 function cleanup(record){
  clearNames(record);
  for(const animation of record.animations)animation.cancel();
  for(const ghost of record.ghosts)ghost.remove();
  if(current!==record)return;
  current=null;delete root.dataset.zoomPhase;delete root.dataset.zoomEngine;
  delete html.dataset.encoderZoom;html.style.removeProperty('--encoder-zoom-duration');
 }
 function finish(){
  const record=current;if(!record)return;
  // A second click can precede the browser's deferred snapshot callback. Commit once
  // before handling that click so a stale callback cannot reopen a dismissed stage.
  record.commit();record.transition?.skipTransition();cleanup(record);
 }
 function fallback(record,before){
  const oldNodes=nodes(record.from,record.to??record.from);
  const rects=new Map([...oldNodes].map(([name,el])=>[name,el.getBoundingClientRect()]));
  const oldFocus=oldNodes.get('encoder-focus');
  // Preserve outgoing detail for the reverse zoom; removed DOM must not disappear instantly.
  let ghost=null;
  if(record.from!==null&&oldFocus){
   const rect=oldFocus.getBoundingClientRect();ghost=oldFocus.cloneNode(true);
   ghost.removeAttribute('id');ghost.querySelectorAll('[id]').forEach(el=>el.removeAttribute('id'));
   ghost.className+=' zoom-outgoing-snapshot';ghost.setAttribute('aria-hidden','true');ghost.inert=true;
   Object.assign(ghost.style,{position:'fixed',left:`${rect.x}px`,top:`${rect.y}px`,width:`${rect.width}px`,height:`${rect.height}px`,margin:'0',pointerEvents:'none',zIndex:'30'});
  }
  record.commit();
  const afterNodes=nodes(record.to,record.from??record.to),duration=ZOOM_TIMING[record.direction];
  function play(el,frames,extra={}){const animation=el.animate(frames,{duration,easing:'cubic-bezier(.22,.75,.18,1)',...extra});record.animations.push(animation);return animation;}
  // FLIP moves headings and the focus region from their old screen coordinates.
  for(const [name,el] of afterNodes){
   if(name==='encoder-diagram')continue;
   const after=el.getBoundingClientRect(),old=rects.get(name),transform=boxTransform(old,after);
   if(!transform)continue;
   const focus=name==='encoder-focus';
   play(el,[{transformOrigin:'0 0',transform,opacity:focus ? 0.12 : 1},{transformOrigin:'0 0',transform:'none',opacity:1}]);
  }
  if(ghost){
   const dest=afterNodes.get('encoder-focus')?.getBoundingClientRect(),origin=rects.get('encoder-focus');
   const transform=boxTransform(dest,origin);doc.body.append(ghost);record.ghosts.push(ghost);
   play(ghost,[{transformOrigin:'0 0',transform:'none',opacity:1},{transformOrigin:'0 0',transform:transform||'scale(.85)',opacity:0}]);
  }
  const after=root.getBoundingClientRect();
  // The fallback also interpolates document flow, so the lower chapters do not jump.
  play(root,[{height:`${before.height}px`,minHeight:'0',overflow:'clip'},{height:`${after.height}px`,minHeight:'0',overflow:'clip'}]);
  Promise.allSettled(record.animations.map(a=>a.finished)).then(()=>cleanup(record));
 }
 function run(to,update){
  finish();const from=getExpanded();
  if(from===to||reduced.matches||doc.visibilityState==='hidden'){update();return;}
  const record={from,to,direction:zoomDirection(from,to),names:new Map(),animations:[],ghosts:[],transition:null,committed:false,commit:null};
  record.commit=()=>{if(record.committed)return;record.committed=true;update();};current=record;
  html.dataset.encoderZoom=record.direction;html.style.setProperty('--encoder-zoom-duration',`${ZOOM_TIMING[record.direction]}ms`);root.dataset.zoomPhase=record.direction;
  if(typeof doc.startViewTransition!=='function'){
   root.dataset.zoomEngine='flip';
   if(typeof root.animate!=='function'){record.commit();cleanup(record);return;}
   try{fallback(record,root.getBoundingClientRect());}catch(error){cleanup(record);throw error;}return;
  }
  root.dataset.zoomEngine='view-transition';
  nameNodes(record,nodes(from,to??from));
  try{
   record.transition=doc.startViewTransition(()=>{
    record.commit();
    if(current===record)nameNodes(record,nodes(to,from??to));
   });
   // A skipped snapshot (resize, background tab, repeated interaction) is expected.
   record.transition.ready.catch(()=>{});
   record.transition.finished.then(()=>cleanup(record),()=>cleanup(record));
  }catch(error){record.commit();cleanup(record);}
 }
 const onPreference=()=>{if(reduced.matches)finish();};
 const onVisibility=()=>{if(doc.visibilityState==='hidden')finish();};
 // Do not trap clicks during the animation. Finish the prior transition first.
 const onInteraction=()=>finish();
 for(const type of ['pointerdown','click','input','change','keydown'])root.parentElement.addEventListener(type,onInteraction,true);
 win.addEventListener('resize',finish);reduced.addEventListener('change',onPreference);doc.addEventListener('visibilitychange',onVisibility);
 return {run,finish,get active(){return current!==null;},dispose(){finish();for(const type of ['pointerdown','click','input','change','keydown'])root.parentElement.removeEventListener(type,onInteraction,true);win.removeEventListener('resize',finish);reduced.removeEventListener('change',onPreference);doc.removeEventListener('visibilitychange',onVisibility);}};
}
