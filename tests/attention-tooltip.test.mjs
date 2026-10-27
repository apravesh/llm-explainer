import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matrix} from '../src/full-config.mjs';
import {attentionCellTooltipData,attentionTooltipPosition,mountAttentionCellTooltip} from '../src/attention-tooltip.mjs';
class Events{
 constructor(){this.listeners=new Map();}
 addEventListener(type,fn){if(!this.listeners.has(type))this.listeners.set(type,new Set());this.listeners.get(type).add(fn);}
 removeEventListener(type,fn){this.listeners.get(type)?.delete(fn);}
 emit(type,event={}){for(const fn of this.listeners.get(type)||[])fn(event);}
}
function harness(){
 const nodes=new Map(),win=Object.assign(new Events(),{innerWidth:1000,innerHeight:700}),doc=Object.assign(new Events(),{defaultView:win});
 let removed=false,tooltip;
 function node(){const attrs=new Map(),classes=new Set();return {dataset:{},style:{},hidden:false,innerHTML:'',textContent:'',classList:{add:x=>classes.add(x),remove:x=>classes.delete(x),contains:x=>classes.has(x)},setAttribute:(k,v)=>attrs.set(k,String(v)),getAttribute:k=>attrs.get(k)??null,removeAttribute:k=>attrs.delete(k),getBoundingClientRect:()=>({left:400,top:320,bottom:340,width:20,height:20})};}
 doc.createElement=()=>{tooltip=node();tooltip.querySelector=s=>{if(!nodes.has(s))nodes.set(s,node());return nodes.get(s);};tooltip.getBoundingClientRect=()=>({width:248,height:130});tooltip.remove=()=>removed=true;return tooltip;};doc.body={append(){}};
 const root=Object.assign(new Events(),{ownerDocument:doc}),cells=[];
 for(const pair of ['0:0','1:2','2:1']){const cell=node();cell.dataset.windowPair=pair;cell.closest=s=>s==='[data-window-pair]'?cell:null;cell.contains=n=>n===cell||n?.cell===cell;cells.push(cell);}
 root.contains=cell=>cells.includes(cell);
 const outside={closest:()=>null};
 let data={c:{cls:false},head:{attention:matrix(3,3,new Float32Array([.1,.2,.7,.3,.5,.2,.4,.1,.5]))}};
 const original=data.head.attention.data.slice(),controller=mountAttentionCellTooltip(root,()=>data);
 return {root,doc,win,cells,outside,tooltip,nodes,controller,original,get data(){return data;},setData(next){data=next;},get removed(){return removed;}};
}
test('tooltip uses the exact global attention weight, with query/key labels and percentage',()=>{
 const h=harness(),value=attentionCellTooltipData(h.data,1,2);
 assert.equal(value.weight,h.data.head.attention.data[5]);assert.equal(value.decimal,String(h.data.head.attention.data[5]));assert.equal(value.percent,'20.0000%');assert.equal(value.queryLabel,'p01');assert.equal(value.keyLabel,'p02');assert.equal(value.total,3);h.controller.dispose();
});
test('CLS and tiny weights are represented correctly without zeroing the stored value',()=>{
 const data={c:{cls:true},head:{attention:matrix(2,2,new Float32Array([.2,.8,1e-9,1]))}};
 assert.equal(attentionCellTooltipData(data,0,1).queryLabel,'CLS');assert.equal(attentionCellTooltipData(data,0,1).keyLabel,'p00');
 const tiny=attentionCellTooltipData(data,1,0);assert.ok(Number(tiny.decimal)>0);assert.equal(Number(tiny.decimal),data.head.attention.data[2]);
 assert.equal(attentionCellTooltipData(data,-1,0),null);assert.equal(attentionCellTooltipData(data,0,2),null);assert.equal(attentionCellTooltipData(data,1.2,0),null);
});
test('pointer hover immediately shows a single tooltip without mutating selection or tensors',()=>{
 const h=harness();assert.ok(h.tooltip.hidden);h.root.emit('pointerover',{target:h.cells[1]});
 assert.equal(h.tooltip.hidden,false);assert.equal(h.tooltip.dataset.query,1);assert.equal(h.tooltip.dataset.key,2);assert.equal(h.tooltip.dataset.weight,h.data.head.attention.data[5]);assert.equal(h.nodes.get('[data-tooltip-pair]').textContent,'Query p01 → Key p02');assert.equal(h.nodes.get('[data-tooltip-decimal]').textContent,String(h.data.head.attention.data[5]));assert.match(h.nodes.get('[data-tooltip-context]').textContent,/all 3 keys/);
 assert.equal(h.cells[1].getAttribute('aria-describedby'),'attention-cell-tooltip');assert.ok(h.cells[1].classList.contains('is-tooltip-active'));assert.deepEqual(h.data.head.attention.data,h.original);h.controller.dispose();
});
test('moving within the same SVG cell does not flicker and leaving hides the tooltip',()=>{
 const h=harness(),child={cell:h.cells[1],closest:()=>h.cells[1]};h.root.emit('pointerover',{target:child});h.root.emit('pointerout',{target:h.cells[1],relatedTarget:child});assert.equal(h.tooltip.hidden,false);
 h.root.emit('pointerout',{target:child,relatedTarget:h.outside});assert.equal(h.tooltip.hidden,true);assert.equal(h.cells[1].getAttribute('aria-describedby'),null);assert.equal(h.cells[1].classList.contains('is-tooltip-active'),false);h.controller.dispose();
});
test('moving to another cell replaces the content, including rows unrelated to the selected query',()=>{
 const h=harness();h.root.emit('pointerover',{target:h.cells[1]});h.root.emit('pointerover',{target:h.cells[2]});assert.equal(h.tooltip.dataset.query,2);assert.equal(h.tooltip.dataset.key,1);assert.equal(h.tooltip.dataset.weight,h.data.head.attention.data[7]);assert.equal(h.cells[1].getAttribute('aria-describedby'),null);h.controller.dispose();
});
test('keyboard focus exposes the same values and preserves an existing aria description',()=>{
 const h=harness();h.cells[0].setAttribute('aria-describedby','existing');h.root.emit('focusin',{target:h.cells[0]});assert.equal(h.tooltip.hidden,false);assert.equal(h.cells[0].getAttribute('aria-describedby'),'existing attention-cell-tooltip');h.root.emit('focusout',{target:h.cells[0],relatedTarget:h.outside});assert.equal(h.tooltip.hidden,true);assert.equal(h.cells[0].getAttribute('aria-describedby'),'existing');h.controller.dispose();
});
test('Escape dismisses the tooltip without triggering stage-collapse handlers',()=>{
 const h=harness();let prevented=false,stopped=false;h.root.emit('pointerover',{target:h.cells[0]});h.doc.emit('keydown',{key:'Escape',preventDefault(){prevented=true;},stopPropagation(){stopped=true;}});assert.equal(h.tooltip.hidden,true);assert.ok(prevented&&stopped);h.controller.dispose();
});
test('scrolling, resize, visibility change and window replacement clear stale values',()=>{
 const h=harness();for(const event of ['scroll','visibilitychange','resize','window']){h.root.emit('pointerover',{target:h.cells[1]});if(event==='window')h.controller.hide();else if(event==='resize')h.win.emit(event);else h.doc.emit(event);assert.equal(h.tooltip.hidden,true);}h.controller.dispose();
});
test('disposal removes tooltip, listener subscriptions and highlight, with no stale effects',()=>{
 const h=harness();h.root.emit('pointerover',{target:h.cells[2]});h.controller.dispose();h.controller.dispose();assert.equal(h.removed,true);assert.equal(h.tooltip.hidden,true);assert.equal(h.cells[2].getAttribute('aria-describedby'),null);
 for(const owner of [h.root,h.doc,h.win])assert.ok([...owner.listeners.values()].every(s=>s.size===0));h.root.emit('pointerover',{target:h.cells[0]});assert.equal(h.tooltip.hidden,true);
});
test('tooltip stays inside the viewport near every corner and flips below if needed',()=>{
 const size={width:248,height:130},viewport={width:390,height:844};for(const [left,top] of [[0,0],[375,0],[0,825],[375,825],[180,400]]){const p=attentionTooltipPosition({left,top,bottom:top+20,width:20},size,viewport);assert.ok(p.left>=8&&p.left+248<=382);assert.ok(p.top>=8&&p.top+130<=836);}
 assert.equal(attentionTooltipPosition({left:100,top:0,bottom:20,width:20},size,viewport).top,30);
});
test('active and offline attention windows include instant hover without the duplicate native title',async()=>{
 const read=p=>readFile(new URL('../'+p,import.meta.url),'utf8'),[source,bundle]=await Promise.all(['src/attention-window.mjs','standalone.html'].map(read));
 assert.match(source,/mountAttentionCellTooltip\(root,getData\)/);assert.match(source,/tooltip.hide\(\)/);assert.match(source,/tooltip.dispose\(\)/);assert.match(bundle,/function mountAttentionCellTooltip/);assert.match(bundle,/data-tooltip-decimal/);assert.doesNotMatch(source,/<title>\$\{label\(q\)\}/);
});

test('unrelated rail scroll and stale outgoing-cell events do not hide the new focus tooltip',()=>{
 const h=harness();h.root.emit('focusin',{target:h.cells[1]});h.root.emit('focusin',{target:h.cells[2]});
 h.root.emit('pointerout',{target:h.cells[1],relatedTarget:h.outside});assert.equal(h.tooltip.hidden,false);
 h.doc.emit('scroll',{target:{contains:()=>false}});assert.equal(h.tooltip.hidden,false);
 h.doc.emit('scroll',{target:{contains:node=>node===h.cells[2]}});assert.equal(h.tooltip.hidden,true);h.controller.dispose();
});

test('keyboard focus takes priority over stationary-pointer events after rerender until the mouse moves',()=>{
 const h=harness();h.root.emit('focusin',{target:h.cells[2]});h.root.emit('pointerover',{target:h.cells[1]});assert.equal(h.tooltip.dataset.query,2);assert.equal(h.tooltip.dataset.key,1);
 h.root.emit('pointermove',{target:h.cells[1]});assert.equal(h.tooltip.dataset.query,1);assert.equal(h.tooltip.dataset.key,2);h.controller.dispose();
});
