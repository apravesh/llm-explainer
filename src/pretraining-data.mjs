/** Hand-authored example records, not a real crawl or a production quality classifier. */
export const CRAWL_EXAMPLE=Object.freeze([
 {id:'a',source:'article',url:'catfacts.example/mat',html:'<!doctype html><html><head><title>Cat facts</title><script>track("visit")</script></head><body><nav>Home · About · Login</nav><main><p>The cat sat on the mat.</p></main><footer>© 2024 · Cookie settings</footer></body></html>',allowed:true,quality:true},
 {id:'b',source:'story',url:'stories.example/dog',html:'<html><body><header>Stories ▸ Animals</header><main><p>The dog sat on the rug.</p></main><aside>Subscribe to our newsletter!</aside></body></html>',allowed:true,quality:true},
 {id:'c',source:'mirror',url:'copycat.example/mat',html:'<html><body><div class="ad">Sponsored</div><main>  The cat sat on the mat. </main></body></html>',allowed:true,quality:true},
 {id:'d',source:'spam',url:'deals.example/buy',html:'<html><body><main><b>BUY BUY BUY BUY BUY</b></main><script>popup()</script></body></html>',allowed:true,quality:false},
 {id:'e',source:'restricted',url:'private.example/notes',html:'<html><body><main><p>Not approved for this example corpus.</p></main></body></html>',allowed:false,quality:true},
 {id:'f',source:'article',url:'catfacts.example/dog',html:'<html><head><style>p{color:#222}</style></head><body><nav>Home · About · Login</nav><main><p>The dog saw the cat.</p></main></body></html>',allowed:true,quality:true},
 {id:'g',source:'story',url:'stories.example/cat',html:'<html><body><header>Stories ▸ Animals</header><main><p>The cat saw the dog.</p></main><footer>Share this story</footer></body></html>',allowed:true,quality:true}
]);
/** Split one HTML string into markup, page chrome and main-content pieces. Joining every piece restores the input. */
export function htmlSegments(html){const out=[],re=/<[^>]*>/g;let last=0,inMain=false,m;
 const push=(text,kind)=>{if(text)out.push({text,kind});};
 while((m=re.exec(html))){push(html.slice(last,m.index),inMain?'content':'boilerplate');push(m[0],'tag');if(/^<main[\s>]/i.test(m[0]))inMain=true;else if(/^<\/main>/i.test(m[0]))inMain=false;last=re.lastIndex;}
 push(html.slice(last),inMain?'content':'boilerplate');return out;}
export const extractText=html=>htmlSegments(html).filter(s=>s.kind==='content').map(s=>s.text).join(' ').replace(/\s+/g,' ').trim();
export function exampleCorpus(){
 const extracted=CRAWL_EXAMPLE.map(r=>({...r,text:extractText(r.html)}));
 const eligible=extracted.filter(r=>r.allowed&&r.quality),seen=new Set(),unique=eligible.filter(r=>{if(seen.has(r.text))return false;seen.add(r.text);return true;});
 const train=unique.slice(0,-1),validation=unique.slice(-1);
 return {raw:extracted,eligible,unique,train,validation,shards:[train.slice(0,2),train.slice(2)]};
}
export const DOCUMENT_SEPARATOR='<|endoftext|>';
/** The training corpus as one long string: kept documents joined by a separator token. */
export function corpusTape(){const c=exampleCorpus(),docs=[...c.train.map(r=>({id:r.id,text:r.text,split:'train',shard:c.shards.findIndex(s=>s.some(x=>x.id===r.id))+1})),...c.validation.map(r=>({id:r.id,text:r.text,split:'validation',shard:0}))];
 const join=list=>list.map(d=>d.text+DOCUMENT_SEPARATOR).join('');
 return {docs,separator:DOCUMENT_SEPARATOR,train:join(docs.filter(d=>d.split==='train')),validation:join(docs.filter(d=>d.split==='validation')),htmlChars:c.raw.reduce((n,r)=>n+r.html.length,0)};}
export const CORPUS_STAGES=Object.freeze([
 {name:'Collect',note:'Start from a crawl snapshot or approved sources. Save provenance with each record.'},
 {name:'Extract & filter',note:'Keep body text; apply source, quality and privacy review. This miniature uses hand-assigned flags, not real classifiers.'},
 {name:'Deduplicate',note:'Remove the repeated article. Real pipelines also look for near-duplicates, not just identical text.'},
 {name:'Split & shard',note:'Hold out a document for validation, then divide training text into files for streaming. Never train on the held-out split.'}
]);
export function corpusView(stage){const c=exampleCorpus();if(!Number.isInteger(stage)||stage<0||stage>=CORPUS_STAGES.length)throw Error('Invalid corpus stage');
 return c.raw.map(r=>({...r,display:stage===0?r.html:r.text,kept:stage===0||stage===1?stage===0||c.eligible.some(x=>x.id===r.id):c.unique.some(x=>x.id===r.id),status:stage===0?'raw HTML':!r.allowed?'excluded · source policy':!r.quality?'excluded · quality':stage>=2&&!c.unique.some(x=>x.id===r.id)?'duplicate':stage===3?c.validation.some(x=>x.id===r.id)?'validation · held out':`training shard ${c.shards.findIndex(s=>s.some(x=>x.id===r.id))+1}`:'kept'}));
}
