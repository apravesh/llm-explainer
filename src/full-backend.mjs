import {matrix,weightAt,hash32} from './full-config.mjs';
import {WASM_KERNEL_BASE64} from './wasm-kernel.mjs';
export async function createCPUBackend(){
 const bytes=Uint8Array.from(atob(WASM_KERNEL_BASE64),c=>c.charCodeAt(0));
 let exports;try{exports=(await WebAssembly.instantiate(bytes)).instance.exports;}catch{return createJSBackend();}
 let lastKey='',weightOffset=0;
 return {name:'WebAssembly SIMD · worker CPU',async multiply(a,w){
  const padded=Math.ceil(w.cols/16)*16,alen=a.data.length*4,wlen=w.rows*padded*4,clen=a.rows*padded*4,total=alen+wlen+clen;
  if(total>exports.memory.buffer.byteLength){exports.memory.grow(Math.ceil((total-exports.memory.buffer.byteLength)/65536));lastKey='';}
  const memory=new Float32Array(exports.memory.buffer);memory.set(a.data,0);
  const key=`${w.rows}:${w.cols}:${w.seed}:${w.scale}:${alen}`;
  if(lastKey!==key){for(let r=0;r<w.rows;r++)for(let c=0;c<padded;c++)memory[alen/4+r*padded+c]=c<w.cols?weightAt(w,r,c):0;lastKey=key;}
  weightOffset=alen;exports.mul(0,weightOffset,alen+wlen,a.rows,a.cols,padded);
  const out=matrix(a.rows,w.cols);for(let r=0;r<a.rows;r++)out.data.set(memory.subarray((alen+wlen)/4+r*padded,(alen+wlen)/4+r*padded+w.cols),r*w.cols);return out;
 },destroy(){lastKey='';exports=null;}};
}
function createJSBackend(){return {name:'JavaScript · worker CPU (slower)',async multiply(a,w){const out=matrix(a.rows,w.cols),weights=new Float32Array(w.rows*w.cols);for(let r=0;r<w.rows;r++)for(let c=0;c<w.cols;c++)weights[r*w.cols+c]=weightAt(w,r,c);for(let i=0;i<a.rows;i++)for(let k=0;k<a.cols;k++){const x=a.data[i*a.cols+k];for(let j=0;j<w.cols;j++)out.data[i*w.cols+j]+=x*weights[k*w.cols+j];}return out;},destroy(){}};}
export async function createGPUBackend(gpu){
 if(!gpu)throw Error('WebGPU unavailable');const adapter=await gpu.requestAdapter();if(!adapter)throw Error('No GPU adapter');
 const device=await adapter.requestDevice();let lost=false;device.lost.then(()=>lost=true);
 const source=`struct P {m:u32,k:u32,n:u32,seed:u32,scale:f32,pad0:u32,pad1:u32,pad2:u32};
 @group(0) @binding(0) var<storage,read> a:array<f32>;
 @group(0) @binding(1) var<storage,read_write> b:array<f32>;
 @group(0) @binding(2) var<storage,read_write> c:array<f32>;
 @group(0) @binding(3) var<uniform> p:P;
 fn hash(x0:u32)->u32 {var x=x0;x=(x^(x>>16u))*0x7feb352du;x=(x^(x>>15u))*0x846ca68bu;return x^(x>>16u);}
 @compute @workgroup_size(256) fn generate(@builtin(global_invocation_id) gid:vec3u){let i=gid.x+gid.y*262144u;if(i<p.k*p.n){b[i]=(f32(hash(i+p.seed)>>8u)/8388608.0-1.0)*p.scale;}}
 var<workgroup> sa:array<f32,256>;var<workgroup> sb:array<f32,256>;
 @compute @workgroup_size(16,16) fn multiply(@builtin(local_invocation_id) lid:vec3u,@builtin(workgroup_id) wid:vec3u){
 let row=wid.y*16u+lid.y;let col=wid.x*16u+lid.x;let ix=lid.y*16u+lid.x;var sum=0.0;
 for(var base=0u;base<p.k;base+=16u){
  sa[ix]=0.0;sb[ix]=0.0;if(row<p.m&&base+lid.x<p.k){sa[ix]=a[row*p.k+base+lid.x];}if(base+lid.y<p.k&&col<p.n){sb[ix]=b[(base+lid.y)*p.n+col];}workgroupBarrier();
  for(var t=0u;t<16u;t++){sum+=sa[lid.y*16u+t]*sb[t*16u+lid.x];}workgroupBarrier();
 }if(row<p.m&&col<p.n){c[row*p.n+col]=sum;}}
 `;
 const shader=device.createShaderModule({code:source});const info=await shader.getCompilationInfo();if(info.messages.some(m=>m.type==='error'))throw Error(info.messages.filter(m=>m.type==='error').map(m=>m.message).join('; '));
 const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:4,buffer:{type:'read-only-storage'}},{binding:1,visibility:4,buffer:{type:'storage'}},{binding:2,visibility:4,buffer:{type:'storage'}},{binding:3,visibility:4,buffer:{type:'uniform'}}]});
 const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
 const gen=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module:shader,entryPoint:'generate'}}),mul=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module:shader,entryPoint:'multiply'}});
 return {name:'WebGPU · worker GPU',async multiply(a,w){
  if(lost)throw Error('GPU device lost');const size=w.rows*w.cols*4;if(size>device.limits.maxStorageBufferBindingSize)throw Error('GPU storage limit too small');
  const buffers=[];const buffer=(size,usage)=>{const b=device.createBuffer({size:Math.max(4,size),usage});buffers.push(b);return b;};
  device.pushErrorScope('validation');
  try{
   const ab=buffer(a.data.byteLength,128|8),bb=buffer(size,128),cb=buffer(a.rows*w.cols*4,128|4),pb=buffer(32,64|8),read=buffer(a.rows*w.cols*4,1|8);
   const params=new ArrayBuffer(32),u=new Uint32Array(params);u.set([a.rows,a.cols,w.cols,w.seed>>>0]);new Float32Array(params)[4]=w.scale;
   device.queue.writeBuffer(ab,0,a.data);device.queue.writeBuffer(pb,0,params);
   const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:ab}},{binding:1,resource:{buffer:bb}},{binding:2,resource:{buffer:cb}},{binding:3,resource:{buffer:pb}}]});
   const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setBindGroup(0,group);pass.setPipeline(gen);const groups=Math.ceil(w.rows*w.cols/256);pass.dispatchWorkgroups(Math.min(1024,groups),Math.ceil(groups/1024));pass.setPipeline(mul);pass.dispatchWorkgroups(Math.ceil(w.cols/16),Math.ceil(a.rows/16));pass.end();encoder.copyBufferToBuffer(cb,0,read,0,a.rows*w.cols*4);device.queue.submit([encoder.finish()]);
   await read.mapAsync(1);const out=matrix(a.rows,w.cols,new Float32Array(read.getMappedRange().slice(0)));read.unmap();return out;
  }finally{buffers.forEach(b=>b.destroy());const error=await device.popErrorScope();if(error)throw Error(error.message);}
 },destroy(){device.destroy();}};
}
export async function createBackend({gpu=globalThis.navigator?.gpu,preferCPU=false}={}){
 if(!preferCPU)try{const backend=await createGPUBackend(gpu);const w={kind:'weights',rows:19,cols:17,seed:11,scale:Math.fround(.2)},a=matrix(2,19,Float32Array.from({length:38},(_,i)=>Math.sin(i)));const out=await backend.multiply(a,w);for(let r=0;r<2;r++)for(let j=0;j<17;j++){let expected=0;for(let k=0;k<19;k++)expected+=a.data[r*19+k]*weightAt(w,k,j);if(Math.abs(out.data[r*17+j]-expected)>2e-5){backend.destroy();throw Error('GPU numerical self-check failed');}}return backend;}catch{/* SIMD worker fallback retains the same complete dimensions and weights. */}
 return createCPUBackend();
}
