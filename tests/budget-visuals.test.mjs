import {test} from 'node:test';import assert from 'node:assert/strict';import {readFile} from 'node:fs/promises';
import {architectureBudget,BUDGET_DEFAULTS} from '../src/architecture-budget.mjs';
import {parameterDiagram,computeDiagram,gpuDiagram,activateBudgetTab,mountBudgetTabs} from '../src/budget-visuals.mjs';
const b=architectureBudget({...BUDGET_DEFAULTS});
for(const key of ['ordinary','swiglu','matched','moe'])test(`${key}: diagram allocates all stored projections and labels scope`,()=>{const html=parameterDiagram(b,key);assert.match(html,new RegExp(`data-visual-variant="${key}"`));for(const name of ['WQ','WK','WV','WO','W₂'])assert.ok(html.includes(name));assert.equal(html.includes('W₃'),key!=='ordinary');assert.equal(html.includes('Choose 2 of 8'),key==='moe');assert.match(html,/0 learned parameters/);assert.match(html,/total stored parameters/);assert.doesNotMatch(html,/NaN|undefined/);});
test('GQA and changed MoE budgets are reflected in drawn nodes',()=>{const changed=architectureBudget({...BUDGET_DEFAULTS,kv:4,experts:32,k:4});const html=parameterDiagram(changed,'moe');assert.match(html,/1024 × 256/);assert.match(html,/Choose 4 of 32/);assert.match(html,/\+ 16 more/);assert.match(html,/× 32 experts/);});
test('compute and GPU diagrams label estimates and use calculated values',()=>{const compute=computeDiagram(b),gpu=gpuDiagram(b);assert.match(compute,/192.00 MiB/);assert.match(compute,/128.00 MiB/);assert.match(compute,/per block/);assert.match(gpu,/not a GPU benchmark/);assert.match(gpu,/Compute/);const small=architectureBudget({...BUDGET_DEFAULTS,n:1});assert.match(gpuDiagram(small),/Memory bandwidth/);});
function harness(){const panels={},tabs=['parameters','compute','gpu'].map(id=>{panels['#panel-'+id]={hidden:false};return {dataset:{budgetTab:id},attrs:{'aria-controls':'panel-'+id},tabIndex:0,setAttribute(k,v){this.attrs[k]=v;},getAttribute(k){return this.attrs[k];},focus(){this.focused=true;},closest(){return this;}};});const events={},root={querySelectorAll:()=>tabs,querySelector:id=>panels[id],addEventListener:(k,fn)=>events[k]=fn,removeEventListener:k=>delete events[k]};return {tabs,panels,events,root};}
test('tabs show one panel, retain contents, and support arrow/Home/End keys',()=>{const h=harness(),dispose=mountBudgetTabs(h.root);activateBudgetTab(h.root,'parameters');assert.equal(h.panels['#panel-parameters'].hidden,false);assert.equal(h.panels['#panel-compute'].hidden,true);h.events.keydown({target:h.tabs[0],key:'ArrowRight',preventDefault(){}});assert.equal(h.tabs[1].attrs['aria-selected'],'true');assert.ok(h.tabs[1].focused);assert.equal(h.tabs[0].tabIndex,-1);h.events.keydown({target:h.tabs[1],key:'End',preventDefault(){}});assert.equal(h.panels['#panel-gpu'].hidden,false);dispose();assert.deepEqual(h.events,{});});
test('page has three accessible views, shared controls and no deleted explanation sections',async()=>{const html=await readFile(new URL('../analytics.html',import.meta.url),'utf8');assert.equal((html.match(/role="tab" /g)||[]).length,3);assert.equal((html.match(/role="tabpanel"/g)||[]).length,3);for(const id of ['compute-diagram','gpu-diagram','design','work','sources','calculator'])assert.ok(html.includes(`id="${id}"`));assert.match(html,/Show FLOP and memory arithmetic/);assert.doesNotMatch(html,/ONE BLOCK, FOUR CHOICES|id="parameter-diagram"/);assert.ok(html.indexOf('name="n"')<html.indexOf('id="budget-tabs"'));});
test('symbolic formulas appear beside counts and explain the selected dimensions',async()=>{
 const {parameterFormulas}=await import('../src/budget-visuals.mjs');
 assert.equal(parameterFormulas(b.c,b.variants.ordinary).attention,'4d²');
 assert.equal(parameterFormulas(b.c,b.variants.ordinary).ffnTotal,'2df');
 assert.equal(parameterFormulas(b.c,b.variants.swiglu).ffnTotal,'3df');
 assert.equal(parameterFormulas(b.c,b.variants.matched).ffnTotal,'3df′');
 const moe=parameterFormulas(b.c,b.variants.moe);assert.equal(moe.ffnTotal,'E × 3df + dE');assert.equal(moe.ffnActive,'k × 3df + dE');assert.equal(moe.projection,'Edf');
 const html=parameterDiagram(b,'moe');for(const label of ['d = 1,024','f = 4,096','Hq = 16 · Hkv = 16','dh = d/Hq = 64','dKV = Hkv × dh = 1024','E = 8 · k = 2','L = 24','>4d²</code>','>E × 3df + dE</code>'])assert.ok(html.includes(label),label);
 assert.match(parameterDiagram(b,'matched'),/f′ = 2,752/);
 const gqa=architectureBudget({...BUDGET_DEFAULTS,kv:4}),f=parameterFormulas(gqa.c,gqa.variants.moe);assert.equal(f.attention,'2d² + 2d·dKV');assert.equal(f.kv,'d·dKV');assert.doesNotMatch(parameterDiagram(gqa,'moe'),/= 4d²/);
 const app=await readFile(new URL('../src/architecture-budget-app.mjs',import.meta.url),'utf8');assert.match(app,/budget-cell-formula/);assert.match(app,/cell\(a.ffnStored,f.ffn\)/);assert.match(app,/cell\(a.active,f.active/);
});

test('parameter totals lead with formulas, followed by smaller approximate numeric counts',async()=>{
 for(const key of ['ordinary','swiglu','matched','moe']){
  const html=parameterDiagram(b,key),totals=[...html.matchAll(/<p class="bv-total">([\s\S]*?)<\/p>/g)];
  assert.equal(totals.length,2);
  for(const [,total] of totals){assert.match(total,/^<code class="bv-total-formula">/);assert.ok(total.indexOf('bv-total-formula')<total.indexOf('bv-total-value'));assert.match(total,/≈ .* parameters<\/span>/);}
 }
 const css=await readFile(new URL('../analytics.css',import.meta.url),'utf8');
 assert.match(css,/#parameter-diagram \.bv-total \.bv-total-formula\{font:500 30px/);
 assert.match(css,/#parameter-diagram \.bv-total-value\{display:block;font:12px/);
});
