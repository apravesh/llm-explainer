// Tokens page: a live GPT-2 tokenizer and a step-by-step view of Byte Pair Encoding merges.
// The tokenizer is real (src/tokenizer-gpt2.mjs with GPT-2's own merge rules); nothing here is illustrative.
import {createTokenizer,VOCAB_SIZE} from './tokenizer-gpt2.mjs';

const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const EXAMPLES=[
 ['Plain sentence','The cat sat on the mat.'],
 ['Long, rare words','unbelievably antidisestablishmentarianism'],
 ['Spaces and capitals','hello Hello HELLO  hello'],
 ['Numbers','3.14159 and 1234567'],
 ['Other languages, emoji','naïve café 日本語 🙂'],
 ['The same word, twice','strawberry strawberry']
];
const WORDS=['tokenization','strawberry','unbelievably','lowest','Hello'];
const state={text:EXAMPLES[0][1],ids:false,word:WORDS[0],step:0,timer:null};
let tok=null;
// a space is part of a token, so it is drawn rather than left invisible
const visible=t=>esc(t).replace(/ /g,'<i class="sp">·</i>').replace(/\n/g,'<i class="sp">↵</i>').replace(/\t/g,'<i class="sp">→</i>');
const show=t=>t.text.includes('�')?`<i class="raw">${t.bytes} byte${t.bytes===1?'':'s'}</i>`:visible(t.text);
const chip=(t,i,cls='')=>`<span class="tk tk-c${i%6}${cls}" title="token ID ${t.id}"><span>${show(t)}</span>${state.ids?`<small>${t.id}</small>`:''}</span>`;
const words=s=>s.trim()?s.trim().split(/\s+/).length:0;

function renderTokens(){
 const tokens=tok.encode(state.text),chars=[...state.text].length,w=words(state.text);
 $('#tk-out').innerHTML=tokens.length?tokens.map((t,i)=>chip(t,i)).join(''):'<span class="tk-empty">Type something above.</span>';
 $('#tk-chars').textContent=chars.toLocaleString('en-US');$('#tk-words').textContent=w.toLocaleString('en-US');$('#tk-tokens').textContent=tokens.length.toLocaleString('en-US');
 $$('#tk-examples button').forEach((b,i)=>b.setAttribute('aria-pressed',String(EXAMPLES[i][1]===state.text)));
 const split=tokens.filter(t=>/^ ?\p{L}/u.test(t.text)).length>(state.text.match(/\p{L}+/gu)||[]).length;
 $('#tk-result').textContent=!tokens.length?'Nothing to tokenize yet.'
  :`${w} word${w===1?'':'s'} became ${tokens.length} token${tokens.length===1?'':'s'}${w?` (${(tokens.length/w).toFixed(2)} tokens per word)`:''}. ${split?'Some words were cut into pieces. A “·” shows a space that travels inside a token.':tokens.length>w?'Every word is one token here, with its space attached in front; punctuation makes up the rest.':'Here every word is one token, with its space attached in front.'}`;
 threeWays(tokens);
}
// the same text cut three ways: by character, by word, by token
function threeWays(tokens){
 const cs=[...state.text],ws=state.text.trim()?state.text.trim().split(/\s+/):[],cap=(list,f)=>list.slice(0,40).map(f).join('')+(list.length>40?'<span class="tk-more">…</span>':'');
 $('#tw-chars').innerHTML=cap(cs,c=>`<span class="tw">${visible(c)}</span>`);$('#tw-words').innerHTML=cap(ws,w=>`<span class="tw">${esc(w)}</span>`);$('#tw-tokens').innerHTML=cap(tokens,(t,i)=>chip(t,i));
 $('#tw-chars-n').textContent=cs.length;$('#tw-words-n').textContent=ws.length;$('#tw-tokens-n').textContent=tokens.length;
}

/* ---------- Byte Pair Encoding: one word built up by GPT-2's merge rules, in the order they apply ---------- */
function steps(){return tok.mergeSteps(state.word);}
function renderBpe(){
 const all=steps(),n=Math.min(state.step,all.length-1),done=n===all.length-1;
 $$('#bpe-words button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.word===state.word)));
 if($('#bpe-word').value!==state.word)$('#bpe-word').value=state.word;
 $('#bpe-ladder').innerHTML=all.slice(0,n+1).map((s,k)=>{
  const next=all[k+1],hit=k===n&&next?next.pair:null;let h='';
  for(let i=0;i<s.parts.length;i++){
   const made=s.pair&&s.parts[i]===s.pair[0]+s.pair[1],glue=hit&&s.parts[i]===hit[0]&&s.parts[i+1]===hit[1];
   h+=`<span class="bp${made?' made':''}${glue?' glue-a':''}${hit&&i>0&&s.parts[i-1]===hit[0]&&s.parts[i]===hit[1]&&!glue?' glue-b':''}">${visible(tok.text(s.parts[i]))}</span>`;
  }
  return `<div class="bpe-row${k===n?' now':''}"><small>${k===0?'bytes':`rule ${(s.rule+1).toLocaleString('en-US')}`}</small><div>${h}</div><b>${s.parts.length}</b></div>`;
 }).join('');
 const s=all[n],next=all[n+1];
 $('#bpe-result').innerHTML=n===0&&next?`“${esc(state.word)}” starts as ${s.parts.length} single bytes. The earliest rule that fits is no. ${(next.rule+1).toLocaleString('en-US')} of 50,000: glue “${visible(tok.text(next.pair[0]))}” and “${visible(tok.text(next.pair[1]))}”.`
  :done?`No more rules fit. “${esc(state.word)}” is ${s.parts.length} token${s.parts.length===1?'':'s'}: ${s.parts.map(p=>`“${visible(tok.text(p))}”`).join(' ')}. ${all.length-1} merge${all.length===2?'':'s'} were applied, always the earliest-learned rule first.`
  :`Rule ${(s.rule+1).toLocaleString('en-US')} glued “${visible(tok.text(s.pair[0]))}” + “${visible(tok.text(s.pair[1]))}” into one token (ID ${256+s.rule}). Next: rule ${(next.rule+1).toLocaleString('en-US')}.`;
 $('#bpe-next').disabled=done;$('#bpe-play').disabled=done&&!state.timer;$('#bpe-play').textContent=state.timer?'Pause':'▶ Play';
}
function stop(){if(state.timer){clearInterval(state.timer);state.timer=null;}}
function setWord(w){stop();state.word=w||WORDS[0];state.step=0;renderBpe();}

async function init(){
 $('#tk-examples').innerHTML=EXAMPLES.map(([label],i)=>`<button type="button" data-ex="${i}">${esc(label)}</button>`).join('');
 $('#bpe-words').innerHTML=WORDS.map(w=>`<button type="button" data-word="${esc(w)}">${esc(w)}</button>`).join('');
 const input=$('#tk-input');input.value=state.text;
 try{
  const res=await fetch(new URL('../assets/gpt2-merges.txt',import.meta.url));if(!res.ok)throw new Error(res.status);
  tok=createTokenizer(await res.text());
 }catch(err){
  for(const id of ['#tk-result','#bpe-result'])$(id).textContent='The tokenizer rules could not be loaded. Open this page through a web server (for example node server.mjs) rather than as a file.';
  return;
 }
 document.body.classList.add('tk-ready');
 input.addEventListener('input',()=>{state.text=input.value;renderTokens();});
 $('#tk-examples').addEventListener('click',e=>{const b=e.target.closest('[data-ex]');if(!b)return;state.text=EXAMPLES[+b.dataset.ex][1];input.value=state.text;renderTokens();});
 $('#tk-ids').addEventListener('change',e=>{state.ids=e.target.checked;renderTokens();});
 // "try it" links in the text load an example into the tokenizer
 document.addEventListener('click',e=>{const t=e.target.closest('[data-try]');if(!t)return;e.preventDefault();state.text=t.dataset.try;input.value=state.text;renderTokens();$('#tk-lab').scrollIntoView({behavior:'smooth',block:'center'});});
 $('#bpe-words').addEventListener('click',e=>{const b=e.target.closest('[data-word]');if(b)setWord(b.dataset.word);});
 $('#bpe-word').addEventListener('input',e=>setWord(e.target.value.trim().slice(0,24)));
 $('#bpe-next').addEventListener('click',()=>{stop();state.step++;renderBpe();});
 $('#bpe-reset').addEventListener('click',()=>{stop();state.step=0;renderBpe();});
 $('#bpe-play').addEventListener('click',()=>{
  if(state.timer){stop();renderBpe();return;}
  if(state.step>=steps().length-1)state.step=0;
  state.timer=setInterval(()=>{state.step++;if(state.step>=steps().length-1)stop();renderBpe();},900);renderBpe();
 });
 $('#tk-vocab').textContent=VOCAB_SIZE.toLocaleString('en-US');
 renderTokens();renderBpe();
}
init();
