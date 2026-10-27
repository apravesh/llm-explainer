/** Architecture dimensions from the published configs; synthetic weights, never a checkpoint. */
export const FULL_CONFIGS=Object.freeze({
 pixtral:Object.freeze({id:'pixtral',name:'Pixtral-12B vision encoder',width:1024,heads:16,headWidth:64,ffWidth:4096,depth:24,decoderWidth:5120,patch:16,eps:1e-5,rms:true,cls:false,gated:true,rope:true}),
 vit:Object.freeze({id:'vit',name:'ViT-B/16',width:768,heads:12,headWidth:64,ffWidth:3072,depth:12,decoderWidth:0,patch:16,eps:1e-5,rms:false,cls:true,gated:false,rope:false})
});
export function hash32(x){x=(x^(x>>>16))>>>0;x=Math.imul(x,0x7feb352d)>>>0;x=(x^(x>>>15))>>>0;x=Math.imul(x,0x846ca68b)>>>0;return (x^(x>>>16))>>>0;}
export function weightDescriptor(rows,cols,seed,scale=1){return {kind:'weights',rows,cols,seed,scale:Math.fround(scale*Math.sqrt(3/rows))};}
export function weightAt(w,r,c){return Math.fround(Math.fround((hash32((Math.imul(r,w.cols)+c+w.seed)>>>0)>>>8)/8388608-1)*w.scale);}
export function matrix(rows,cols,data=new Float32Array(rows*cols)){return {kind:'matrix',rows,cols,data};}
export function valueAt(a,r,c){return a.kind==='weights'?weightAt(a,r,c):a.data[r*a.cols+c];}
export function rowOf(a,r){return a.data.subarray(r*a.cols,(r+1)*a.cols);}
export function blockWeights(c,layer){const seed=1048576+layer*100000007;return {q:weightDescriptor(c.width,c.width,seed+11),k:weightDescriptor(c.width,c.width,seed+23),v:weightDescriptor(c.width,c.width,seed+37),o:weightDescriptor(c.width,c.width,seed+53,.45),up:weightDescriptor(c.width,c.ffWidth,seed+71,.7),down:weightDescriptor(c.ffWidth,c.width,seed+83,.45),...(c.gated?{gate:weightDescriptor(c.width,c.ffWidth,seed+97,.7)}:{})};}
export function projectionWeights(c,pixelWidth){return weightDescriptor(pixelWidth,c.width,1971);}
export function projectorWeights(c){return [weightDescriptor(c.width,c.decoderWidth,971),weightDescriptor(c.decoderWidth,c.decoderWidth,983)];}
export function sumColumn(a,w,row,col){let sum=0;const terms=new Float64Array(a.cols),prefix=new Float64Array(a.cols+1);for(let k=0;k<a.cols;k++){const term=valueAt(a,row,k)*valueAt(w,k,col);terms[k]=term;sum+=term;prefix[k+1]=sum;}return {terms,prefix,total:sum};}
export function featureWindow(length,index=0,size=32){const selected=Math.min(length-1,Math.max(0,Math.trunc(index)||0)),start=Math.floor(selected/size)*size;return {start,end:Math.min(length,start+size),selected,total:length};}
export function viewMatrix(a,{colStart=0,cols=a.cols-colStart,rowStart=0,rows=a.rows-rowStart}={}){return {kind:'view',rows,cols,source:a,rowStart,colStart};}
export function matrixValue(a,r,c){return a.kind==='view'?matrixValue(a.source,r+a.rowStart,c+a.colStart):valueAt(a,r,c);}
export function estimatedParameters(c){return 768*c.width+c.depth*(4*c.width*c.width+(c.gated?3:2)*c.width*c.ffWidth)+(c.decoderWidth?c.width*c.decoderWidth+c.decoderWidth*c.decoderWidth:0);}

/** Demo-sized image geometry only; hidden dimensions and patch size remain canonical. */
export function fullImageShape(width,height,c){
 if(!Number.isFinite(width)||!Number.isFinite(height)||width<=0||height<=0)throw Error('Positive image dimensions required');
 if(c.cls)return {width:96,height:96,drawWidth:96,drawHeight:96};
 const scale=128/Math.max(width,height),drawWidth=Math.max(1,Math.round(width*scale)),drawHeight=Math.max(1,Math.round(height*scale));
 return {width:Math.max(c.patch,Math.ceil(drawWidth/c.patch)*c.patch),height:Math.max(c.patch,Math.ceil(drawHeight/c.patch)*c.patch),drawWidth,drawHeight};
}
