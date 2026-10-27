/** One visible transformer block represents the selected depth, never all blocks at once. */
export const BLOCK_PLAYBACK_INTERVAL=1800;
export const BLOCK_TRANSITION_MS=420;
export function adjacentBlock(current,delta,count){return Math.max(0,Math.min(count-1,current+delta));}
export function blockKeyTarget(current,key,count){
 if(key==='Home')return 0;if(key==='End')return count-1;
 const delta={ArrowLeft:-1,ArrowUp:-1,ArrowRight:1,ArrowDown:1}[key];
 return delta===undefined?null:adjacentBlock(current,delta,count);
}
/** Injectable timer so pauses, restarts, and stale callbacks can be tested independently. */
export function createBlockPlayback({request,cancel,getCurrent,getCount,canPlay=()=>true,onSelect,onState=()=>{},interval=BLOCK_PLAYBACK_INTERVAL}){
 let playing=false,completed=false,frame=null,start=null,elapsed=0,epoch=0;
 const snapshot=()=>({playing,completed,elapsed,index:getCurrent(),count:getCount(),progress:Math.min(1,elapsed/interval)});
 function clear(){epoch++;if(frame!==null)cancel(frame);frame=null;start=null;}
 function publish(){onState(snapshot());}
 function queue(){const ticket=epoch;frame=request(now=>tick(now,ticket));}
 function tick(now,ticket){if(ticket!==epoch||!playing)return;frame=null;if(!canPlay()){pause();return;}
  if(start===null)start=now-elapsed;
  elapsed=Math.min(interval,Math.max(0,now-start));publish();
  if(elapsed>=interval){
   if(getCurrent()>=getCount()-1){playing=false;completed=true;clear();publish();return;}
   onSelect(getCurrent()+1,{autoplay:true});
   if(!playing||ticket!==epoch)return;
   elapsed=0;start=now;publish();
  }
  if(playing&&ticket===epoch)queue();
 }
 function pause(){playing=false;clear();publish();}
 function play(){if(playing||!canPlay())return;clear();completed=false;
  if(getCurrent()>=getCount()-1){onSelect(0,{autoplay:true});elapsed=0;}
  playing=true;publish();queue();
 }
 return {play,pause,
  seek(index){pause();completed=false;elapsed=0;onSelect(Math.max(0,Math.min(getCount()-1,index)),{autoplay:false});publish();},
  restart(){pause();completed=false;elapsed=0;if(!canPlay()){publish();return;}onSelect(0,{autoplay:true});playing=true;publish();queue();},
  reset(){pause();completed=false;elapsed=0;publish();},
  get snapshot(){return snapshot();}
 };
}
export function blockHandoff(run,selected,layerIndex){
 const layer=run.layers[layerIndex];if(!layer)return null;const row=selected+(run.config.cls?1:0),at=a=>a.data[row*a.cols];
 return {row,feature:0,input:at(layer.input),afterAttention:at(layer.afterAttention),output:at(layer.output),
  previousOutput:layerIndex>0&&run.layers[layerIndex-1]?at(run.layers[layerIndex-1].output):null,
  nextInput:run.layers[layerIndex+1]?at(run.layers[layerIndex+1].input):null};
}
/** Only the small arrows attached to the block diagram; no separate navigator UI. */
export function inlineBlockControlsMarkup(index,count){
 return `<span class="inline-block-controls expanded-block-controls" role="group" aria-label="Transformer block"><button type="button" data-inline-block-step="-1" aria-label="Inspect previous transformer block" aria-controls="architecture">‹</button><span>Block ${index+1} / ${count}</span><button type="button" data-inline-block-step="1" aria-label="Inspect next transformer block" aria-controls="architecture">›</button></span>`;
}
export function createInlineBlockNavigation({root,getRun,getState,beforeSelect,onSelect,onAnnounce=()=>{}}){
 const doc=root.ownerDocument,win=doc.defaultView,reduced=win.matchMedia('(prefers-reduced-motion: reduce)');
 let animations=[],disposed=false,lastRun=null;
 function cancelAnimations(){animations.forEach(a=>a.cancel());animations=[];root.removeAttribute('data-block-transition');}
 function sync(){
  if(disposed)return;const run=getRun();
  if(run!==lastRun){cancelAnimations();lastRun=run;}
  root.querySelectorAll('[data-inline-block-step]').forEach(b=>{
   const next=(getState().layer||0)+Number(b.dataset.inlineBlockStep);
   b.disabled=!run||next<0||next>=run.config.depth||!run.layers[next];
  });
 }
 function seek(index){
  if(disposed)return;const run=getRun();if(!run||!Number.isFinite(index))return;
  const next=Math.max(0,Math.min(run.config.depth-1,Math.trunc(index))),previous=getState().layer;
  if(next===previous||!run.layers[next]){sync();return;}
  beforeSelect?.();cancelAnimations();onSelect(next);sync();
  if(!reduced.matches&&doc.visibilityState!=='hidden'){
   const expanded=getState().expanded,targets=expanded===3||expanded===4?[root.querySelector('.stage.is-expanded .expansion-inspection')]:expanded===null?[...root.querySelectorAll('.encoder-group .stage-content')]:[];
   root.dataset.blockTransition=next>previous?'forward':'backward';
   targets.filter(Boolean).forEach(el=>{if(typeof el.animate==='function')animations.push(el.animate([{opacity:.25,transform:`translateY(${next>previous?12:-12}px)`},{opacity:1,transform:'translateY(0)'}],{duration:BLOCK_TRANSITION_MS,easing:'cubic-bezier(.2,.7,.2,1)'}));});
   const active=animations;Promise.allSettled(active.map(a=>a.finished)).then(()=>{if(animations===active){animations=[];root.removeAttribute('data-block-transition');}});
  }
  onAnnounce(`Transformer block ${next+1} of ${run.config.depth}. Self-attention and MLP updated; the selected patch and head are unchanged.`);
 }
 function navigate(event,key=false){
  const b=event.target.closest('[data-inline-block-step]');if(!b||!root.contains(b)||b.disabled)return;
  const run=getRun();if(!run)return;
  const next=key?blockKeyTarget(getState().layer,event.key,run.config.depth):getState().layer+Number(b.dataset.inlineBlockStep);
  if(next===null)return;if(key)event.preventDefault();
  const expanded=!!b.closest('.stage-expansion'),direction=b.dataset.inlineBlockStep;
  seek(next);
  const scope=expanded?'.stage.is-expanded .stage-expansion ':'.block-label ';
  const controls=[...root.querySelectorAll(scope+'[data-inline-block-step]')];
  const focus=controls.find(el=>el.dataset.inlineBlockStep===direction&&!el.disabled)||controls.find(el=>!el.disabled);
  focus?.focus({preventScroll:true});
 }
 const click=e=>navigate(e),keydown=e=>navigate(e,true),visibility=()=>{if(doc.hidden)cancelAnimations();},preference=()=>{if(reduced.matches)cancelAnimations();sync();};
 root.addEventListener('click',click);root.addEventListener('keydown',keydown);doc.addEventListener('visibilitychange',visibility);reduced.addEventListener('change',preference);win.addEventListener('resize',cancelAnimations);
 sync();
 return {sync,seek,reset:cancelAnimations,dispose(){disposed=true;cancelAnimations();root.removeEventListener('click',click);root.removeEventListener('keydown',keydown);doc.removeEventListener('visibilitychange',visibility);reduced.removeEventListener('change',preference);win.removeEventListener('resize',cancelAnimations);}};
}
