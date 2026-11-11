/** Character-level BPE teaching example. Boundaries are explicit; not byte-level GPT tokenization. */
export const BPE_WORDS=Object.freeze([{word:'low',count:5},{word:'lower',count:2},{word:'newest',count:6},{word:'widest',count:3}]);
export function mergePair(tokens,pair){const result=[];for(let i=0;i<tokens.length;i++){if(i+1<tokens.length&&tokens[i]===pair[0]&&tokens[i+1]===pair[1]){result.push(tokens[i]+tokens[i+1]);i++;}else result.push(tokens[i]);}return result;}
export function pairCounts(words){const pairs=new Map();for(const {tokens,count} of words)for(let i=0;i<tokens.length-1;i++){const pair=[tokens[i],tokens[i+1]],key=JSON.stringify(pair);if(!pairs.has(key))pairs.set(key,{pair,count:0});pairs.get(key).count+=count;}
 return [...pairs.values()].sort((a,b)=>b.count-a.count||(JSON.stringify(a.pair)<JSON.stringify(b.pair)?-1:JSON.stringify(a.pair)>JSON.stringify(b.pair)?1:0));}
export function bpeTrace(steps=0){if(!Number.isInteger(steps)||steps<0||steps>20)throw Error('Use 0–20 merges');let words=BPE_WORDS.map(r=>({...r,tokens:[...r.word,'</w>']})),vocab=[...new Set(words.flatMap(w=>w.tokens))].sort(),merges=[];
 for(let step=0;step<steps;step++){const best=pairCounts(words)[0];if(!best)break;merges.push(best);const token=best.pair.join('');if(!vocab.includes(token))vocab.push(token);words=words.map(w=>({...w,tokens:mergePair(w.tokens,best.pair)}));}
 return {words,vocab,merges,pairs:pairCounts(words),tokenCount:words.reduce((n,w)=>n+w.tokens.length*w.count,0)};
}
export function encodeBpe(word,trace){if(!/^[a-z]+$/.test(word))throw Error('Use a lowercase sample word');const base=new Set(bpeTrace(0).vocab);if([...word].some(c=>!base.has(c)))throw Error('Character outside this toy alphabet');let tokens=[...word,'</w>'];for(const {pair} of trace.merges)tokens=mergePair(tokens,pair);return {tokens,ids:tokens.map(t=>trace.vocab.indexOf(t))};}

/** Every intermediate state while encoding a word: one entry per rule that actually changes it. */
export function encodeSteps(word,trace){encodeBpe(word,trace);let tokens=[...word,'</w>'];const steps=[{rule:null,tokens}];trace.merges.forEach(({pair},index)=>{const next=mergePair(tokens,pair);if(next.length!==tokens.length){tokens=next;steps.push({rule:index,pair,tokens});}});return steps;}
/** Start indices of the non-overlapping, left-to-right occurrences that mergePair will fuse. */
export function pairOccurrences(tokens,pair){const hits=[];for(let i=0;i+1<tokens.length;i++)if(tokens[i]===pair[0]&&tokens[i+1]===pair[1]){hits.push(i);i++;}return hits;}
