import {attentionStageValue,stageNumber} from './attention-stages.mjs';
/** One lightweight tooltip for the visible attention cells. Hover never selects a pair. */
export function attentionCellTooltipData(data,query,key,stage='softmax'){
 const weights=data.head.attention;
 if(!Number.isInteger(query)||!Number.isInteger(key)||query<0||key<0||query>=weights.rows||key>=weights.cols)return null;
 const weight=weights.data[query*weights.cols+key];
 const token=i=>data.c.cls&&i===0?'CLS':`p${String(i-(data.c.cls?1:0)).padStart(2,'0')}`;
 if(stage!=='softmax'){
  const cell=attentionStageValue(data,query,key,stage);if(!cell)return null;
  return {query,key,queryLabel:token(query),keyLabel:token(key),weight:cell.value,decimal:cell.value===-Infinity?'−∞':String(cell.value),percent:stageNumber(cell.value,4),total:weights.cols,title:cell.label,
   context:stage==='dot'?`Q · K uses all ${data.c.headWidth} head features; before scaling or masking.`:cell.blocked?'Future key is blocked: its score is −∞ and softmax weight is 0.':`Divide by √${data.c.headWidth} = ${cell.divisor}. ${cell.causal?'Causal comparison; this key is allowed.':'Encoder: no causal mask.'}`};
 }
 return {query,key,queryLabel:token(query),keyLabel:token(key),weight,decimal:String(weight),percent:`${(weight*100).toFixed(4)}%`,total:weights.cols,title:'Attention weight',context:data.causalComparison?`Causal comparison: softmax over ${query+1} eligible keys${key>query?'; this future key is blocked':''}.`:`Full-row softmax over all ${weights.cols} keys.`};
}
export function attentionTooltipPosition(anchor,size,viewport){
 const margin=8,gap=10,width=Math.min(size.width,Math.max(0,viewport.width-margin*2));
 const left=Math.max(margin,Math.min(anchor.left+anchor.width/2-width/2,viewport.width-width-margin));
 const above=anchor.top-size.height-gap;
 const top=Math.max(margin,Math.min(above>=margin?above:anchor.bottom+gap,viewport.height-size.height-margin));
 return {left,top};
}
export function mountAttentionCellTooltip(root,getData){
 const doc=root.ownerDocument,win=doc.defaultView,tooltip=doc.createElement('div');
 tooltip.id='attention-cell-tooltip';tooltip.className='aw-cell-tooltip';tooltip.setAttribute('role','tooltip');tooltip.hidden=true;
 tooltip.innerHTML='<div class="aw-tooltip-pair" data-tooltip-pair></div><div class="aw-tooltip-weight"><span data-tooltip-kind>Attention weight</span><strong data-tooltip-percent></strong></div><code data-tooltip-decimal></code><small data-tooltip-context></small>';
 doc.body.append(tooltip);let active=null,previousDescription=null,disposed=false,mode='pointer';
 function hide(){
  tooltip.hidden=true;
  if(active){active.classList.remove('is-tooltip-active');if(previousDescription===null)active.removeAttribute('aria-describedby');else active.setAttribute('aria-describedby',previousDescription);}
  active=null;previousDescription=null;
 }
 function show(cell){
  if(disposed||!cell||!root.contains(cell))return;
  const [q,k]=cell.dataset.windowPair.split(':').map(Number),value=attentionCellTooltipData(getData(),q,k,cell.dataset.attentionStage||'softmax');if(!value){hide();return;}
  if(active===cell&&!tooltip.hidden)return;
  hide();active=cell;previousDescription=cell.getAttribute('aria-describedby');
  cell.setAttribute('aria-describedby',[previousDescription,tooltip.id].filter(Boolean).join(' '));cell.classList.add('is-tooltip-active');
  tooltip.querySelector('[data-tooltip-pair]').textContent=`Query ${value.queryLabel} → Key ${value.keyLabel}`;
  tooltip.querySelector('[data-tooltip-kind]').textContent=value.title;tooltip.querySelector('[data-tooltip-percent]').textContent=value.percent;
  tooltip.querySelector('[data-tooltip-decimal]').textContent=value.decimal;
  tooltip.querySelector('[data-tooltip-context]').textContent=value.context;
  tooltip.dataset.stage=cell.dataset.attentionStage||'softmax';tooltip.dataset.query=q;tooltip.dataset.key=k;tooltip.dataset.weight=value.weight;
  tooltip.hidden=false;
  const position=attentionTooltipPosition(cell.getBoundingClientRect(),tooltip.getBoundingClientRect(),{width:win.innerWidth,height:win.innerHeight});
  tooltip.style.left=`${position.left}px`;tooltip.style.top=`${position.top}px`;
 }
 function enter(e){if(mode==='focus')return;const cell=e.target.closest?.('[data-window-pair]');if(cell)show(cell);else hide();}
 function move(e){mode='pointer';enter(e);}
 function focus(e){mode='focus';const cell=e.target.closest?.('[data-window-pair]');if(cell)show(cell);else hide();}
 function blur(e){leave(e);mode='pointer';}
 function pointerLeave(e){if(mode!=='focus')leave(e);}
 function leave(e){const cell=e.target.closest?.('[data-window-pair]');if(cell!==active)return;if(cell&&e.relatedTarget&&cell.contains(e.relatedTarget))return;hide();}
 function scroll(e){if(!e.target||e.target===doc||e.target.contains?.(active))hide();}
 function escape(e){if(e.key==='Escape'&&active&&!tooltip.hidden){e.preventDefault();e.stopPropagation();hide();}}
 root.addEventListener('pointerover',enter);root.addEventListener('pointermove',move);root.addEventListener('pointerout',pointerLeave);root.addEventListener('focusin',focus);root.addEventListener('focusout',blur);
 doc.addEventListener('scroll',scroll,true);doc.addEventListener('keydown',escape,true);doc.addEventListener('visibilitychange',hide);win.addEventListener('resize',hide);
 return {hide,dispose(){if(disposed)return;disposed=true;hide();root.removeEventListener('pointerover',enter);root.removeEventListener('pointermove',move);root.removeEventListener('pointerout',pointerLeave);root.removeEventListener('focusin',focus);root.removeEventListener('focusout',blur);doc.removeEventListener('scroll',scroll,true);doc.removeEventListener('keydown',escape,true);doc.removeEventListener('visibilitychange',hide);win.removeEventListener('resize',hide);tooltip.remove();}};
}
