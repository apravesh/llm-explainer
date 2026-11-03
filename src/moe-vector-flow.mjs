/** Small exact-width vectors: values are available on hover, not printed as a spreadsheet. */
const mvEscape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function moeVectorColor(v){const rgb=v<0?[204,146,121]:[104,143,207],a=Math.min(Math.abs(v)/3,1);return `rgb(${rgb.map(x=>Math.round(245+(x-245)*a)).join(',')})`;}
export function moeVectorMarkup(values,label){
 return `<span class="mv-vector" role="img" aria-label="${mvEscape(label)}: 1 by ${values.length} vector" data-vector-label="${mvEscape(label)}">${values.map((v,i)=>`<i style="background:${moeVectorColor(v)}" title="${mvEscape(label)}[${i}] = ${v.toPrecision(7)}"></i>`).join('')}</span>`;
}
export function expertVectorMarkup(trace,e){
 const expert=trace.experts[e];if(!expert)return '<span class="mv-skipped">Not selected · no vector sent</span>';
 return `<span class="mv-expert-row"><span data-expert-input="${e}"><small>same x</small>${moeVectorMarkup(trace.normalized,'x')}</span><b aria-hidden="true">→</b><span class="mv-function">f${e+1}</span><b aria-hidden="true">→</b><span data-expert-output="${e}"><small>y${e+1} · 1 × 8</small>${moeVectorMarkup(expert.output,`y${e+1}`)}</span></span>`;
}
export function contributionVectorMarkup(trace){return trace.selected.map(e=>`<div class="mv-contribution expert-${e}" data-contribution-node="${e}"><div class="mv-contribution-title"><strong>g${e+1} = ${trace.gates[e].toFixed(3)}</strong><span>× every feature</span></div><div class="mv-weighted-row"><span data-contribution-input="${e}"><small>y${e+1}</small>${moeVectorMarkup(trace.experts[e].output,`y${e+1}`)}</span><b aria-hidden="true">→</b><span data-contribution-output="${e}"><small>g${e+1} y${e+1}</small>${moeVectorMarkup(trace.contributions[e],`g${e+1} y${e+1}`)}</span></div></div>`).join('');}
/** Geometry is measured from the vector strips, not guessed offsets from whole stages. */
export function moeVectorEdges(boxes,selected,{dense=false}={}){
 const left=r=>({x:r.x,y:r.y+r.h/2}),right=r=>({x:r.x+r.w,y:r.y+r.h/2}),bottom=r=>({x:r.x+r.w/2,y:r.y+r.h}),top=r=>({x:r.x+r.w/2,y:r.y});
 const curve=(a,b)=>`M${a.x} ${a.y} C${a.x+16} ${a.y} ${b.x-16} ${b.y} ${b.x} ${b.y}`;
 const edges=[],x=right(boxes.x),busY=Math.min(boxes.router.y,...selected.map(e=>boxes.experts[e].y))-18;
 if(!dense)edges.push({id:'x-router',kind:'input',path:curve(x,left(boxes.router))});
 for(const e of selected){const target=left(boxes.experts[e]);edges.push({id:`x-expert-${e}`,kind:'dispatch',expert:e,path:`M${x.x} ${x.y} H${x.x+10} V${busY} H${target.x-12} V${target.y} H${target.x}`});
  edges.push({id:`expert-${e}-weight`,kind:'expert-output',expert:e,path:curve(right(boxes.expertOutputs[e]),left(boxes.contributions[e]))});
  const from=right(boxes.weighted[e]),to=right(boxes.sum),lane=Math.max(from.x,to.x)+8+selected.indexOf(e)*4;edges.push({id:`weight-${e}-sum`,kind:'combine',expert:e,path:`M${from.x} ${from.y} H${lane} V${to.y-5+selected.indexOf(e)*10} H${to.x}`});
 }
 edges.push({id:'sum-residual',kind:'combine',path:curve(right(boxes.sum),left(boxes.add))});
 const h=left(boxes.h),join=right(boxes.add),floor=Math.max(boxes.expertBottom||0,...Object.values(boxes.experts).map(r=>r.y+r.h),boxes.sum.y+boxes.sum.h,boxes.output.y+boxes.output.h)+24;
 edges.push({id:'h-residual',kind:'residual',path:`M${h.x} ${h.y} H${h.x-8} V${floor} H${join.x+10} V${join.y} H${join.x}`});
 const add=bottom(boxes.add),out=top(boxes.output);edges.push({id:'residual-output',kind:'output',path:`M${add.x} ${add.y} V${out.y-6} H${out.x} V${out.y}`});
 return {edges,busY,floor};
}
