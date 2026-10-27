import {FULL_CONFIGS} from './full-config.mjs';
import {createBackend} from './full-backend.mjs';
import {runFullEncoder} from './full-engine.mjs';
self.onmessage=async({data})=>{
 const started=performance.now();let backend;
 try{
  backend=await createBackend({preferCPU:data.preferCPU});self.postMessage({type:'backend',name:backend.name});
  await runFullEncoder({...data,config:FULL_CONFIGS[data.model],backend,
   onProgress:progress=>self.postMessage({type:'progress',...progress,ms:performance.now()-started}),
   onEmbedding:value=>self.postMessage({type:'embedding',value}),onLayer:(value,index)=>self.postMessage({type:'layer',value,index}),
   onOutput:value=>self.postMessage({type:'output',value}),onProjector:value=>self.postMessage({type:'projector',value})});
  self.postMessage({type:'complete',ms:performance.now()-started});
 }catch(error){self.postMessage({type:'error',message:error.message});}finally{backend?.destroy();}
};
