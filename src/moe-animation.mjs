/** A repeating visual trace of cached results; routing and model values never change. */
export const MOE_LOOP_TIMING=Object.freeze({step:750,hold:1500});
export function moeLoopFrame(elapsed,{dense=false,experts=2,includeAttention=false}={}){
 const stages=[...(includeAttention?['attention','projection']:[]),...(dense?['input','experts','combine','output']:['input','router','experts','combine','output'])];
 const duration=stages.length*MOE_LOOP_TIMING.step,cycleLength=duration+MOE_LOOP_TIMING.hold;
 const time=Math.max(0,elapsed),cycle=Math.floor(time/cycleLength),within=time%cycleLength;
 const phase=within>=duration?'complete':stages[Math.floor(within/MOE_LOOP_TIMING.step)];
 const count=phase==='combine'?Math.min(experts,Math.floor((within%MOE_LOOP_TIMING.step)/MOE_LOOP_TIMING.step*experts)+1):experts;
 return {phase,count,cycle};
}
export function mountMoeAnimation(root,{dense=false,experts=2,includeAttention=false,onFrame}){
 const doc=root.ownerDocument,win=doc.defaultView,media=win.matchMedia('(prefers-reduced-motion: reduce)');
 let frame=null,epoch=0,elapsed=0,last=null,signature='',disposed=false,inView=true;
 function paint(){const state=media.matches?{phase:'complete',count:experts,cycle:0}:moeLoopFrame(elapsed,{dense,experts,includeAttention});
  const next=`${state.phase}:${state.count}:${state.cycle}`;if(next===signature)return;signature=next;
  root.dataset.phase=state.phase;root.dataset.cycle=state.cycle;
  root.querySelectorAll('[data-moe-node]').forEach(node=>node.classList.toggle('running-node',!media.matches&&node.dataset.moeNode===state.phase));onFrame(state);
 }
 function cancel(){epoch++;if(frame!==null)win.cancelAnimationFrame(frame);frame=null;last=null;root.classList.remove('is-running');}
 function queue(){const ticket=epoch;frame=win.requestAnimationFrame(now=>{if(disposed||ticket!==epoch)return;frame=null;
  if(!root.isConnected){dispose();return;}if(doc.hidden||doc.querySelector('dialog[open]')||media.matches||!inView){sync();return;}
  if(last!==null)elapsed+=Math.max(0,now-last);last=now;paint();queue();
 });}
 function sync(){if(disposed)return;cancel();paint();
  if(media.matches){root.dataset.playback='reduced-motion';return;}
  if(doc.hidden||doc.querySelector('dialog[open]')||!inView){root.dataset.playback='paused';return;}
  root.dataset.playback='playing';root.classList.add('is-running');queue();
 }
 function preference(){elapsed=0;signature='';sync();}
 const observer=win.IntersectionObserver?new win.IntersectionObserver(entries=>{inView=entries.some(e=>e.isIntersecting);sync();},{threshold:0}):null;
 function dispose(){if(disposed)return;disposed=true;cancel();observer?.disconnect();root.querySelectorAll('[data-moe-node]').forEach(n=>n.classList.remove('running-node'));delete root.dataset.playback;doc.removeEventListener('visibilitychange',sync);media.removeEventListener('change',preference);}
 doc.addEventListener('visibilitychange',sync);media.addEventListener('change',preference);observer?.observe(root);sync();
 return {sync,dispose};
}
