/** The existing patch-to-embedding curves double as projection controls. */
export function patchProjectionPath(x1,y1,x2,y2){const mid=x1+(x2-x1)/2;return `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`;}
/** Non-overlapping hit bands split the space between neighboring curves.
 * A wide stroke is unsafe here: steep curves can be only a pixel apart. */
export function patchProjectionHitBand(links,index){
 const point=(link,t)=>({x:link.x1+(link.x2-link.x1)*(1.5*t-1.5*t*t+t*t*t),y:link.y1+(link.y2-link.y1)*(3*t*t-2*t*t*t)});
 const points=Array.from({length:33},(_,i)=>{const t=i/32,p=point(links[index],t),previous=links[index-1]&&point(links[index-1],t),next=links[index+1]&&point(links[index+1],t);return {...p,top:p.y-Math.max(0,Math.min(6,previous?(p.y-previous.y)/2:6)),bottom:p.y+Math.max(0,Math.min(6,next?(next.y-p.y)/2:6))};});
 const path=points.map((p,i)=>`${i?'L':'M'}${p.x},${p.top}`).join(' ')+[...points].reverse().map(p=>` L${p.x},${p.bottom}`).join('')+'Z';
 return {points,path};
}
export function patchProjectionLinksMarkup(links,{selected,inputWidth,outputWidth,height}){
 if(!links.length)return '';
 const tabPatch=links.some(link=>link.patch===selected)?selected:links[0].patch;
 const left=Math.min(...links.map(link=>Math.min(link.x1,link.x2))),right=Math.max(...links.map(link=>Math.max(link.x1,link.x2)));
 return `<defs><clipPath id="patch-projection-hit-clip"><rect x="${left}" y="0" width="${Math.max(0,right-left)}" height="${height}"/></clipPath></defs>`+links.map((link,index)=>{
  const {patch,x1,y1,x2,y2}=link,path=patchProjectionPath(x1,y1,x2,y2);
  return `<g id="patch-projection-link-${patch}" class="patch-projection-link ${patch===selected?'is-selected':''}" data-patch-projection="${patch}" role="button" tabindex="${patch===tabPatch?0:-1}" aria-label="Animate patch p${patch} projection to embedding" aria-haspopup="dialog" aria-controls="full-math-dialog"><title>p${patch}: ${inputWidth} RGB values → ${outputWidth} embedding features. Click to see the multiplication.</title><path class="patch-projection-hit" d="${patchProjectionHitBand(links,index).path}" clip-path="url(#patch-projection-hit-clip)" aria-hidden="true"/><path class="patch-projection-line" d="${path}" aria-hidden="true"/></g>`;
 }).join('');
}
export function projectionLinkKey(patches,current,key){
 const index=patches.indexOf(current);if(index<0||!patches.length)return null;
 if(key==='Home')return patches[0];if(key==='End')return patches.at(-1);
 const offset={ArrowUp:-1,ArrowLeft:-1,ArrowDown:1,ArrowRight:1}[key];return offset===undefined?null:patches[Math.max(0,Math.min(patches.length-1,index+offset))];
}
