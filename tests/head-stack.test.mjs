import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {FULL_CONFIGS,matrix} from '../src/full-config.mjs';
import {headStackMarkup,nextStackHead} from '../src/head-stack.mjs';
import {createFullViews} from '../src/full-views.mjs';

for(const model of ['pixtral','vit']){
 const c=FULL_CONFIGS[model],heads=Array.from({length:c.heads},(_,i)=>({attention:matrix(3,3,Float32Array.from({length:9},()=>.01*(i+1)))}));
 test(`${model}: one selected preview replaces the grid, with every head still reachable`,()=>{
  for(let head=0;head<c.heads;head++){
   const seen=[],renderer={markup(a,options){seen.push({a,options});return '<canvas></canvas>';}};
   const html=headStackMarkup(renderer,c,heads,{head,key:2},1,()=>{});
   assert.equal(seen.length,1);assert.strictEqual(seen[0].a,heads[head].attention);
   assert.equal((html.match(/data-head-preview=/g)||[]).length,1);
   assert.equal((html.match(/data-full-head=/g)||[]).length,c.heads);
   assert.equal((html.match(/aria-pressed="true"/g)||[]).length,1);assert.equal((html.match(/tabindex="0"/g)||[]).length,1);
   assert.match(html,new RegExp(`data-full-head="${head}"[^>]*aria-pressed="true" tabindex="0"`));
   assert.match(html,new RegExp(`Head ${head+1} of ${c.heads}`));
   assert.match(html,new RegExp(`Projected features ${head*64}–${(head+1)*64-1}`));
   assert.equal((html.match(/class="fa-stack-sheet /g)||[]).length,3);
   assert.doesNotMatch(html,/class="fa-heads"|NaN|undefined/);
  }
 });
 test(`${model}: selected preview uses the exact query/key and does not alter parallel tensors`,()=>{
  const before=structuredClone(heads),head=c.heads-1,click=()=>{};let observed;
  const html=headStackMarkup({markup(a,o){observed={a,o};return '<canvas></canvas>';}},c,heads,{head,key:2},1,click);
  assert.strictEqual(observed.a,heads[head].attention);assert.equal(observed.o.row,1);assert.equal(observed.o.col,2);assert.strictEqual(observed.o.click,click);
  assert.match(html,new RegExp(`${(heads[head].attention.data[5]*100).toFixed(3)}%`));
  assert.match(html,new RegExp(`All ${c.heads} heads run in parallel`));assert.match(html,/no causal mask/);assert.deepEqual(heads,before);
 });
 test(`${model}: head keyboard navigation wraps and Home/End reach both limits`,()=>{
  assert.equal(nextStackHead(0,'ArrowLeft',c.heads),c.heads-1);assert.equal(nextStackHead(c.heads-1,'ArrowRight',c.heads),0);
  assert.equal(nextStackHead(2,'ArrowUp',c.heads),1);assert.equal(nextStackHead(2,'ArrowDown',c.heads),3);
  assert.equal(nextStackHead(4,'Home',c.heads),0);assert.equal(nextStackHead(4,'End',c.heads),c.heads-1);
  assert.equal(nextStackHead(4,'Enter',c.heads),null);assert.equal(nextStackHead(4,'Tab',c.heads),null);
 });
}
test('stack and full detail stay synchronized on the selected head and actual feature range',()=>{
 const c=FULL_CONFIGS.pixtral,n=2,wide=()=>matrix(n,c.width,Float32Array.from({length:n*c.width},(_,i)=>i/1000));
 const layer={input:wide(),rawQ:wide(),rawK:wide(),q:wide(),k:wide(),values:wide(),afterAttention:wide(),heads:Array.from({length:c.heads},(_,h)=>({scores:matrix(n,n),attention:matrix(n,n,new Float32Array(n*n).fill(.5)),out:matrix(n,64,new Float32Array(n*64).fill(h))}))};
 const run={config:c,layers:[layer]},seen=[],renderer={markup(a,o){seen.push({a,o});return '<canvas></canvas>';}};
 const views=createFullViews(renderer,{select(){},selectPair(){}});
 const html=views.content(run,{expanded:3,selected:1,layer:0,head:15,key:0,headFeature:63,feature:1023});
 assert.match(html,/data-selected-head="15"/);assert.match(html,/data-detail-head="15"/);assert.match(html,/64 features per head/);
 assert.equal(seen.length,0);assert.match(html,/data-window-pair="1:0"/);
 for(const o of seen.filter(e=>/^Q|^K|^V/.test(e.o.label))){assert.equal(o.a.colStart,960);assert.equal(o.a.cols,64);}
 assert.equal((html.match(/data-head-preview=/g)||[]).length,1);
});
test('source and offline build use the stack and preserve accessible reduced-motion navigation',async()=>{
 const root=new URL('../',import.meta.url),html=await readFile(new URL('standalone.html',root),'utf8'),app=await readFile(new URL('src/full-app.mjs',root),'utf8');
 assert.match(html,/function headStackMarkup/);assert.match(html,/function nextStackHead/);assert.match(html,/data-stack-step/);assert.match(html,/fa-stack-sheet/);
 assert.match(app,/function chooseHead/);assert.match(app,/prefers-reduced-motion: reduce/);assert.match(app,/chooseHead\(next,/);
});
