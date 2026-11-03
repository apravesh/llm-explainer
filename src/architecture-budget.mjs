/** Analytical weight and forward-pass counts. No model execution or performance benchmark. */
export const BUDGET_DEFAULTS=Object.freeze({d:1024,f:4096,h:16,kv:16,layers:24,experts:8,k:2,n:2048,batch:1,bytes:2,compute:100,bandwidth:1000});
export function validateBudget(c){
 for(const key of ['d','f','h','kv','layers','experts','k','n','batch'])if(!Number.isSafeInteger(c[key])||c[key]<1)throw Error(`${key} must be a positive whole number`);
 if(c.d<64||c.f<64||c.h>256||c.kv>256||c.d>16384||c.f>65536||c.layers>256||c.experts>256||c.n>131072||c.batch>256)throw Error('Choose values within the displayed teaching limits.');
 if(c.d%c.h||c.h%c.kv)throw Error('Hidden width must divide evenly into query heads; KV heads must divide query heads.');
 if(c.k>c.experts)throw Error('Selected experts cannot exceed the expert count.');
 if(![1,2,4].includes(c.bytes))throw Error('Weight/cache storage must be 1, 2 or 4 bytes.');
 if(!Number.isFinite(c.compute)||c.compute<=0||c.compute>10000||!Number.isFinite(c.bandwidth)||c.bandwidth<=0||c.bandwidth>100000)throw Error('Use positive compute and bandwidth within the displayed limits.');
 return c;
}
export function budgetVariant(c,{gated=false,f=c.f,experts=1,k=1,moe=false}={}){
 const dh=c.d/c.h,dkv=c.kv*dh,q=c.d*c.d,key=c.d*dkv,value=key,o=q,attention=q+key+value+o;
 const expert=(gated?3:2)*c.d*f,router=moe?c.d*experts:0,ffnStored=expert*(moe?experts:1),ffnActive=expert*(moe?k:1);
 const stored=attention+ffnStored+router,active=attention+ffnActive+router;
 return {dh,dkv,q,key,value,o,attention,expert,router,f,experts:moe?experts:1,k:moe?k:1,gated,moe,ffnStored,ffnActive,stored,active,stackStored:stored*c.layers,stackActive:active*c.layers,weightBytes:stored*c.layers*c.bytes,mlpShare:ffnStored/stored};
}
export function architectureBudget(input){
 const c=validateBudget(input),variants={
  ordinary:budgetVariant(c),swiglu:budgetVariant(c,{gated:true}),
  matched:budgetVariant(c,{gated:true,f:Math.max(64,Math.ceil((2*c.f/3)/64)*64)}),
  moe:budgetVariant(c,{gated:true,moe:true,experts:c.experts,k:c.k})
 };
 // Full square bidirectional attention; causal kernels can skip the masked triangle.
 const mixing=4*c.batch*c.n*c.n*c.d,projection=2*c.batch*c.n*variants.moe.attention,ffn=2*c.batch*c.n*variants.moe.ffnActive,router=2*c.batch*c.n*variants.moe.router;
 const dh=c.d/c.h,cacheBytes=2*c.layers*c.batch*c.n*c.kv*dh*c.bytes;
 // Idealized single GEMM X[M,d] W[d,f] -> Y[M,f], beta=0: read inputs/weights once, write output once.
 const M=c.batch*c.n,gemmFlops=2*M*c.d*c.f,gemmBytes=(M*c.d+c.d*c.f+M*c.f)*c.bytes;
 const computeMs=gemmFlops/(c.compute*1e12)*1000,memoryMs=gemmBytes/(c.bandwidth*1e9)*1000;
 return {c,variants,mixing,projection,ffn,router,prefillFlops:c.layers*(mixing+projection+ffn+router),cacheBytes,
  naiveScoreBytes:c.batch*c.h*c.n*c.n*c.bytes,
  decodeMixingFlops:4*c.layers*c.batch*c.n*c.d,decodeLinearFlops:2*c.layers*c.batch*variants.moe.active,
  gemm:{M,flops:gemmFlops,bytes:gemmBytes,intensity:gemmFlops/gemmBytes,ridge:c.compute*1e12/(c.bandwidth*1e9),computeMs,memoryMs,lowerBoundMs:Math.max(computeMs,memoryMs),bound:computeMs>=memoryMs?'compute':'memory bandwidth'},
  assignments:M*c.k,tokensPerExpert:M*c.k/c.experts};
}
