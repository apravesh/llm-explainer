/** Display-only patch rulers. Matrix data and hit-testing coordinates do not change. */
export function isWeightMatrix(a){return a.kind==='weights'||a.kind==='view'&&isWeightMatrix(a.source);}
export function matrixRowOffset(a){return a.kind==='view'?a.rowStart+matrixRowOffset(a.source):0;}
export function patchRowLabel(index,clsOffset=0){return clsOffset&&index===0?'CLS':`p${index-clsOffset}`;}
export function matrixRowSummary(rows,clsOffset=0,offset=0){
 if(offset===0&&clsOffset)return `${rows} token rows · ${rows-1} patches + CLS`;
 return `${rows} patch ${rows===1?'row':'rows'}`;
}
/** Every patch gets a line/tick when readable. Labels are thinned, not token rows. */
export function matrixRowGuides(rows,height,{selected=-1,clsOffset=0,offset=0,minLabelGap=12,majorEvery=4}={}){
 if(!Number.isInteger(rows)||rows<1||!Number.isFinite(height)||height<=0)return {lines:[],labels:[],cellHeight:0,stride:0};
 const cellHeight=height/rows;
 // For very large token counts, choose a readable multiple rather than drawing a dark block.
 const stride=cellHeight>=2?1:Math.ceil(2/cellHeight);
 const lines=[];for(let row=0;row<=rows;row+=stride)lines.push({row,y:row===rows?height:row*cellHeight,major:row===0||row===rows||row%majorEvery===0});
 if(lines.at(-1)?.row!==rows)lines.push({row:rows,y:height,major:true});
 const labelStride=Math.max(1,Math.ceil(minLabelGap/cellHeight/majorEvery))*majorEvery;
 const accepted=[];const offer=row=>{if(row<0||row>=rows||accepted.some(r=>Math.abs(r-row)*cellHeight<minLabelGap))return;accepted.push(row);};
 // Selected row wins if labels collide. Edges precede regular intervals.
 if(Number.isInteger(selected))offer(selected);offer(0);offer(rows-1);
 for(let row=0;row<rows;row+=labelStride)offer(row);
 return {lines,labels:accepted.sort((a,b)=>a-b).map(row=>({row,y:(row+.5)*cellHeight,label:patchRowLabel(row+offset,clsOffset),selected:row===selected})),cellHeight,stride};
}
