import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {ATTENTION_PAN_DURATION,attentionPanOffset,interpolateAttentionPath,mountAttentionWindowMotion,sanitizeAttentionGhost} from '../src/attention-window-motion.mjs';
class Events{
 constructor(){this.handlers=new Map();}
 addEventListener(type,fn){if(!this.handlers.has(type))this.handlers.set(type,new Set());this.handlers.get(type).add(fn);}
 removeEventListener(type,fn){this.handlers.get(type)?.delete(fn);}
 emit(type){for(const fn of this.handlers.get(type)||[])fn();}
}
const prop=name=>name.replace(/-([a-z])/g,(_,c)=>c.toUpperCase());
class Node{
 constructor(attrs={},children=[]){this.attrs={...attrs};this.children=[];this.dataset={};this.style={removeProperty:name=>delete this.style[prop(name)]};this.classList={remove:()=>{},add:()=>{}};for(const [key,val] of Object.entries(attrs))if(key.startsWith('data-'))this.dataset[prop(key.slice(5))]=val;children.forEach(n=>this.append(n));}
 get attributes(){return Object.entries(this.attrs).map(([name,value])=>({name,value}));}
 getAttribute(name){return this.attrs[name]??null;}
 setAttribute(name,value){this.attrs[name]=String(value);}
 removeAttribute(name){delete this.attrs[name];if(name.startsWith('data-'))delete this.dataset[prop(name.slice(5))];}
 append(node){node.parentNode=this;this.children.push(node);}
 remove(){this.removed=true;if(this.parentNode)this.parentNode.children=this.parentNode.children.filter(n=>n!==this);}
 cloneNode(){return new Node(this.attrs,this.children.map(n=>n.cloneNode()));}
 querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
 querySelectorAll(selector){const all=this.children.flatMap(n=>[n,...n.querySelectorAll('*')]);return all.filter(n=>selector==='*'||selector.split(',').some(s=>{s=s.trim();if(s[0]==='[')return n.getAttribute(s.slice(1,-1))!==null;if(s[0]==='.')return n.attrs.class===s.slice(1);return n.attrs.tag===s;}));}
}
function token(kind,id,index){return new Node({'data-window-token':String(id),'data-window-kind':kind,tabindex:'0',role:'button'},[new Node({class:'aw-token-hit',x:kind==='out'?568:43,y:index*(kind==='k'||kind==='v'?17:29),width:90,height:23})]);}
function cell(stage,q,k,i,j){return new Node({'data-window-pair':`${q}:${k}`,'data-attention-stage':stage,'data-window-weight':'0.025',tabindex:'0',role:'button'},[new Node({class:'aw-cell-hit',x:j*29,y:i*29,width:29,height:29})]);}
function wire(kind,id,index){return new Node({'data-window-wire':`${kind}-${id}`,d:`M139 ${index*29}C180 ${index*29} 210 ${index*29} 230 ${index*29}`});}
function target(start,expanded=false,keyStart=start){
 const groups=[];for(const kind of ['q','k','v','out'])groups.push(new Node({'data-motion-viewport':kind},Array.from({length:6},(_,i)=>token(kind,(kind==='k'||kind==='v'?keyStart:start)+i,i))));
 groups.push(new Node({'data-motion-viewport':'wires'},Array.from({length:6},(_,i)=>wire('q',start+i,i))));
 for(const stage of expanded?['dot','scaled','softmax']:['softmax'])groups.push(new Node({'data-motion-viewport':stage},Array.from({length:36},(_,i)=>cell(stage,start+Math.floor(i/6),keyStart+i%6,Math.floor(i/6),i%6))));
 return new Node({},groups);
}
function harness(){
 let clock=0,sequence=0;const jobs=new Map(),media=Object.assign(new Events(),{matches:false});
 const win={matchMedia:()=>media,performance:{now:()=>clock},requestAnimationFrame:fn=>{jobs.set(++sequence,fn);return sequence;},cancelAnimationFrame:id=>jobs.delete(id)};
 const doc=Object.assign(new Events(),{hidden:false,defaultView:win}),root=new Node();root.ownerDocument=doc;root.isConnected=true;
 const controller=mountAttentionWindowMotion(root),step=time=>{clock+=time;const pending=[...jobs.values()];jobs.clear();pending.forEach(fn=>fn(clock));};
 const start=(from,to,expanded=false)=>{const old=target(from,expanded),snapshot=controller.capture(old,from),next=target(to,expanded);controller.play(next,from,to,snapshot);return {old,next,snapshot};};
 return {root,doc,win,jobs,media,controller,step,start};
}
const getToken=(t,kind,id)=>t.querySelectorAll('[data-window-token]').find(n=>n.dataset.windowKind===kind&&+n.dataset.windowToken===id);
const transform=n=>n.getAttribute('transform');
test('scroll motion uses actual strip pitch and moves matrix rows AND columns',()=>{
 assert.deepEqual(attentionPanOffset('q',1),{x:0,y:29});assert.deepEqual(attentionPanOffset('out',-2),{x:0,y:-58});
 assert.deepEqual(attentionPanOffset('k',1),{x:0,y:17});assert.deepEqual(attentionPanOffset('v',-1),{x:0,y:-17});
 assert.deepEqual(attentionPanOffset('cell',2),{x:58,y:58});assert.equal(ATTENTION_PAN_DURATION,360);
});
test('wire points interpolate exactly without disconnecting their end points',()=>{
 assert.equal(interpolateAttentionPath('M139 29C180 29 210 29 230 29','M139 0C180 0 210 0 230 0',.5),'M139 14.5C180 14.5 210 14.5 230 14.5');
 assert.equal(interpolateAttentionPath('M0 0H20','M0 0V20',.5),'M0 0V20');
});
test('shared patches retain position, entrants fade in and outgoing ghosts cannot be focused or hovered',()=>{
 const h=harness(),{next}=h.start(19,20);
 assert.equal(transform(getToken(next,'q',20)),'translate(0 29)');assert.equal(transform(getToken(next,'k',20)),'translate(0 17)');
 assert.equal(getToken(next,'q',25).style.opacity,'0');assert.equal(getToken(next,'q',20).style.opacity,'1');
 const pair=next.querySelectorAll('[data-window-pair]').find(n=>n.dataset.windowPair==='20:20');assert.equal(transform(pair),'translate(29 29)');assert.equal(pair.dataset.windowWeight,'0.025');
 const ghosts=next.querySelectorAll('[inert]');assert.ok(ghosts.length>0);assert.ok(ghosts.every(n=>n.getAttribute('aria-hidden')==='true'&&!n.getAttribute('role')&&!n.getAttribute('tabindex')));
 assert.equal(next.querySelectorAll('[data-window-pair]').length,36);assert.equal(next.querySelectorAll('[data-window-token]').length,24);
 h.step(180);const mid=+transform(getToken(next,'q',20)).match(/translate\(0 ([^)]+)/)[1];assert.ok(mid>0&&mid<29);assert.ok(+getToken(next,'q',25).style.opacity>0);
 h.step(180);assert.equal(transform(getToken(next,'q',20)),null);assert.equal(next.querySelectorAll('[inert]').length,0);assert.equal(h.jobs.size,0);assert.equal(h.root.dataset.windowPanning,undefined);h.controller.dispose();
});
test('rapid scroll reversal continues at the current pose, including a returning ghost',()=>{
 const h=harness(),{next}=h.start(19,20);h.step(80);
 const before=transform(getToken(next,'q',21));const snapshot=h.controller.capture(next,20),again=target(19);h.controller.play(again,20,19,snapshot);
 const oldShift=+before.match(/translate\(0 ([^)]+)/)[1];const newShift=+transform(getToken(again,'q',21)).match(/translate\(0 ([^)]+)/)[1];assert.ok(Math.abs(newShift-(oldShift-29))<1e-10);
 assert.ok(+getToken(again,'q',19).style.opacity>0);assert.equal(h.jobs.size,1);assert.equal(h.root.dataset.windowPanning,'backward');h.step(360);assert.equal(again.querySelectorAll('[inert]').length,0);h.controller.dispose();
});
test('each expanded stage matches only its own cells, with unchanged real values',()=>{
 const h=harness(),{next}=h.start(19,20,true);assert.equal(next.querySelectorAll('[data-window-pair]').length,108);
 const shared=next.querySelectorAll('[data-window-pair]').filter(n=>n.dataset.windowPair==='21:22');assert.equal(shared.length,3);
 assert.deepEqual(shared.map(n=>n.dataset.attentionStage),['dot','scaled','softmax']);assert.ok(shared.every(n=>transform(n)==='translate(29 29)'&&n.style.opacity==='1'));h.controller.dispose();
});
test('initial mount, same window and large jumps do not animate a fictitious traversal',()=>{
 const h=harness();h.start(-1,19);assert.equal(h.jobs.size,0);h.start(19,19);assert.equal(h.jobs.size,0);
 const {next}=h.start(0,42);assert.equal(transform(getToken(next,'q',42)),'translate(0 174)');assert.equal(transform(getToken(next,'v',42)),'translate(0 102)');h.controller.dispose();
});
for(const reason of ['reduced','hidden','removed','dispose','resize'])test(`${reason} settles scroll animation and removes all ghosts/frames`,()=>{
 const h=harness(),{next}=h.start(19,20);h.step(40);
 if(reason==='reduced'){h.media.matches=true;h.media.emit('change');}if(reason==='hidden'){h.doc.hidden=true;h.doc.emit('visibilitychange');}
 if(reason==='removed'){h.root.isConnected=false;h.step(16);}if(reason==='dispose')h.controller.dispose();if(reason==='resize')h.controller.cancel();
 assert.equal(h.jobs.size,0);assert.equal(next.querySelectorAll('[inert]').length,0);assert.equal(transform(getToken(next,'q',21)),null);
 h.controller.dispose();assert.ok([...h.doc.handlers.values(),...h.media.handlers.values()].every(v=>v.size===0));
});
test('reduced-motion/hidden initial updates are immediate and have no animation frames',()=>{
 for(const reason of ['hidden','reduced']){const h=harness();if(reason==='hidden')h.doc.hidden=true;else h.media.matches=true;const {next}=h.start(19,20);assert.equal(h.jobs.size,0);assert.equal(transform(getToken(next,'q',21)),null);h.controller.dispose();}
});
test('sanitizing ghosts removes descendant identity and tooltip references',()=>{
 const n=new Node({role:'button','data-window-pair':'20:20','aria-describedby':'tip',tabindex:'0'},[new Node({id:'hit','data-hit':'yes',role:'button',tabindex:'0'}),new Node({tag:'title'})]);sanitizeAttentionGhost(n);
 assert.equal(n.getAttribute('role'),null);assert.equal(n.querySelectorAll('[id]').length,0);assert.equal(n.querySelectorAll('[data-hit]').length,0);assert.equal(n.querySelectorAll('title').length,0);
});
test('scroll integration preserves diagram horizontal position and uses a stable alignment anchor',async()=>{
 const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');const [window,rail,bundle]=await Promise.all(['src/attention-window.mjs','src/attention-rail-alignment.mjs','standalone.html'].map(read));
 assert.match(window,/scrollMotion.capture\(target,previousStart\)/);assert.match(window,/scrollMotion.play\(target,previousStart,\{query:w.start,key:keys.start\},snapshot\)/);assert.match(window,/scrollLeft=diagramScroll/);assert.match(window,/scrollMotion.dispose\(\)/);
 assert.match(window,/clipPathUnits="userSpaceOnUse"/);assert.match(rail,/querySelector\('\[data-attention-row-anchor\]'\)/);assert.match(bundle,/function mountAttentionWindowMotion/);
});

for(const axis of ['query','key'])test(`${axis}-only pan animates the correct matrix axis without moving the other token list`,()=>{
 const h=harness(),old=target(19,true,25),previous={query:19,key:25},position={query:axis==='query'?20:19,key:axis==='key'?26:25},snapshot=h.controller.capture(old,previous),next=target(position.query,true,position.key);
 h.controller.play(next,previous,position,snapshot);
 const pair=next.querySelectorAll('[data-window-pair]').find(n=>n.dataset.windowPair===`${position.query}:${position.key}`);
 assert.equal(transform(pair),axis==='query'?'translate(0 29)':'translate(29 0)');
 assert.equal(transform(getToken(next,'q',position.query)),axis==='query'?'translate(0 29)':'translate(0 0)');
 assert.equal(transform(getToken(next,'k',position.key)),axis==='key'?'translate(0 17)':'translate(0 0)');
 h.step(360);assert.equal(next.querySelectorAll('[inert]').length,0);assert.equal(h.jobs.size,0);h.controller.dispose();
});
