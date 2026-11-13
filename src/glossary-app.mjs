// Glossary filter: hides terms that do not match what is typed. The page is complete without it.
const input=document.querySelector('#gl-filter'),count=document.querySelector('#gl-count');
const terms=[...document.querySelectorAll('.gl-term')].map(el=>({el,text:el.textContent.toLowerCase()}));
const groups=[...document.querySelectorAll('.gl-group')];
export const matches=(text,query)=>query.trim().toLowerCase().split(/\s+/).filter(Boolean).every(word=>text.includes(word));
function apply(){
 const q=input.value;let shown=0;
 for(const t of terms){const ok=matches(t.text,q);t.el.hidden=!ok;if(ok)shown++;}
 for(const g of groups)g.hidden=!g.querySelector('.gl-term:not([hidden])');
 count.textContent=q.trim()?(shown?`${shown} of ${terms.length} terms`:'No term matches. Try a shorter word.'):`${terms.length} terms`;
}
input.addEventListener('input',apply);
if(location.hash.startsWith('#term-'))document.querySelector(location.hash)?.scrollIntoView();
apply();
