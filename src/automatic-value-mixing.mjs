/** View-only playback. The complete attention tensors and selected pair never change. */
export const VALUE_MIXING_TIMING=Object.freeze({perKey:80,hold:700});
export function valueMixingFrame(elapsed,keys){
 const duration=keys*VALUE_MIXING_TIMING.perKey,cycleLength=duration+VALUE_MIXING_TIMING.hold;
 const time=Math.max(0,elapsed),cycle=Math.floor(time/cycleLength),phase=time%cycleLength;
 return {cycle,count:Math.min(keys,Math.floor(phase/VALUE_MIXING_TIMING.perKey)),complete:phase>=duration};
}
export function mountAutomaticValueMixing(root,{data,feature}){
 const doc=root.ownerDocument,win=doc.defaultView,reduced=win.matchMedia('(prefers-reduced-motion: reduce)');
 const bench=root.querySelector('.sa-workbench'),box=root.querySelector('#full-value-mix'),label=root.querySelector('[data-value-mixing-label]');
 const {c,query,head,n,l,headIndex}=data,expected=head.out.data[query*c.headWidth+feature];
 const prefix=new Float64Array(n+1);
 for(let key=0;key<n;key++)prefix[key+1]=prefix[key]+head.attention.data[query*n+key]*l.values.data[key*c.width+headIndex*c.headWidth+feature];
 let elapsed=0,last=null,frame=null,epoch=0,disposed=false,lastWritten='';
 function write(count,cycle=0){
  const signature=`${count}:${cycle}:${reduced.matches}`;if(lastWritten===signature)return;lastWritten=signature;
  box.textContent=reduced.matches?`Full head output · feature ${feature}: ${expected.toFixed(5)} (reduced motion)`:`${count}/${n} keys mixed · feature ${feature}: ${prefix[count].toFixed(5)}${count===n?' (complete)':' (partial sum)'}`;
  box.dataset.count=count;box.dataset.cycle=cycle;box.dataset.sum=prefix[count];box.dataset.expected=expected;
 }
 function cancelFrame(){epoch++;if(frame!==null)win.cancelAnimationFrame(frame);frame=null;last=null;bench.classList.remove('is-mixing');}
 function schedule(){const ticket=epoch;frame=win.requestAnimationFrame(now=>tick(now,ticket));}
 function tick(now,ticket){
  if(disposed||ticket!==epoch)return;frame=null;
  if(!root.isConnected){dispose();return;}
  if(doc.hidden||reduced.matches){sync();return;}
  if(last!==null)elapsed+=Math.max(0,now-last);last=now;
  const state=valueMixingFrame(elapsed,n);write(state.count,state.cycle);schedule();
 }
 function sync(){
  if(disposed)return;cancelFrame();
  if(reduced.matches){root.dataset.valueMixing='reduced-motion';label.textContent='Value mixing · full result';write(n);return;}
  if(doc.hidden){root.dataset.valueMixing='paused';label.textContent='Value mixing · paused while hidden';return;}
  root.dataset.valueMixing='playing';label.textContent=data.causalComparison?'Value mixing · causal comparison':'Value mixing · loops automatically';bench.classList.add('is-mixing');
  const state=valueMixingFrame(elapsed,n);write(state.count,state.cycle);schedule();
 }
 function preference(){elapsed=0;lastWritten='';sync();}
 function dispose(){if(disposed)return;disposed=true;cancelFrame();delete root.dataset.valueMixing;doc.removeEventListener('visibilitychange',sync);reduced.removeEventListener('change',preference);}
 doc.addEventListener('visibilitychange',sync);reduced.addEventListener('change',preference);sync();
 return {dispose};
}
