/** Hand-authored example records, not a real crawl or a production quality classifier. */
export const CRAWL_EXAMPLE=Object.freeze([
 {id:'a',source:'article',html:'<nav>Home · About</nav><main>The cat sat on the mat.</main>',allowed:true,quality:true},
 {id:'b',source:'story',html:'<main>The dog sat on the rug.</main>',allowed:true,quality:true},
 {id:'c',source:'mirror',html:'<main>  The cat sat on the mat. </main>',allowed:true,quality:true},
 {id:'d',source:'spam',html:'<main>BUY BUY BUY BUY BUY</main>',allowed:true,quality:false},
 {id:'e',source:'restricted',html:'<main>Not approved for this example corpus.</main>',allowed:false,quality:true},
 {id:'f',source:'article',html:'<main>The dog saw the cat.</main>',allowed:true,quality:true},
 {id:'g',source:'story',html:'<main>The cat saw the dog.</main>',allowed:true,quality:true}
]);
export function exampleCorpus(){
 const extracted=CRAWL_EXAMPLE.map(r=>({...r,text:r.html.match(/<main>(.*?)<\/main>/)?.[1].replace(/\s+/g,' ').trim()||''}));
 const eligible=extracted.filter(r=>r.allowed&&r.quality),seen=new Set(),unique=eligible.filter(r=>{if(seen.has(r.text))return false;seen.add(r.text);return true;});
 const train=unique.slice(0,-1),validation=unique.slice(-1);
 return {raw:extracted,eligible,unique,train,validation,shards:[train.slice(0,2),train.slice(2)]};
}
export const CORPUS_STAGES=Object.freeze([
 {name:'Collect',note:'Start from a crawl snapshot or approved sources. Save provenance with each record.'},
 {name:'Extract & filter',note:'Keep body text; apply source, quality and privacy review. This miniature uses hand-assigned flags, not real classifiers.'},
 {name:'Deduplicate',note:'Remove the repeated article. Real pipelines also look for near-duplicates, not just identical text.'},
 {name:'Split & shard',note:'Hold out a document for validation, then divide training text into files for streaming. Never train on the held-out split.'}
]);
export function corpusView(stage){const c=exampleCorpus();if(!Number.isInteger(stage)||stage<0||stage>=CORPUS_STAGES.length)throw Error('Invalid corpus stage');
 return c.raw.map(r=>({...r,display:stage===0?r.html:r.text,kept:stage===0||stage===1?stage===0||c.eligible.some(x=>x.id===r.id):c.unique.some(x=>x.id===r.id),status:stage===0?'raw HTML':!r.allowed?'excluded · source policy':!r.quality?'excluded · quality':stage>=2&&!c.unique.some(x=>x.id===r.id)?'duplicate':stage===3?c.validation.some(x=>x.id===r.id)?'validation · held out':`training shard ${c.shards.findIndex(s=>s.some(x=>x.id===r.id))+1}`:'kept'}));
}
