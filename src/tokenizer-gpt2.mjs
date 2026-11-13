// GPT-2's real tokenizer (byte-level Byte Pair Encoding), in plain JavaScript.
// It needs only the model's 50,000 merge rules (assets/gpt2-merges.txt, from OpenAI's GPT-2 release):
// a token's ID follows from them. IDs 0–255 are the single bytes, ID 256 + n is the token made by
// merge rule n, and 50256 is the end-of-text marker.
export const VOCAB_SIZE=50257,END_OF_TEXT=50256;
// GPT-2 shows each byte as one visible character: printable bytes keep their own, the rest are moved up past U+0100.
function byteChars(){
 const bs=[];for(let b=33;b<=126;b++)bs.push(b);for(let b=161;b<=172;b++)bs.push(b);for(let b=174;b<=255;b++)bs.push(b);
 const cs=bs.slice();let n=0;for(let b=0;b<256;b++)if(!bs.includes(b)){bs.push(b);cs.push(256+n++);}
 return {bytes:bs,chars:cs.map(c=>String.fromCharCode(c))};
}
// text is first cut into pieces: contractions, runs of letters or digits or other symbols (each with an optional leading space), and whitespace
const PIECES=/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
export function createTokenizer(mergesText){
 const {bytes,chars}=byteChars(),charOf=new Map(bytes.map((b,i)=>[b,chars[i]])),byteOf=new Map(chars.map((c,i)=>[c,bytes[i]]));
 const merges=mergesText.split('\n').filter(Boolean).map(line=>line.split(' '));
 const rank=new Map(merges.map(([a,b],i)=>[a+' '+b,i]));
 const id=new Map(chars.map((c,i)=>[c,i]));merges.forEach(([a,b],i)=>id.set(a+b,256+i));
 const enc=new TextEncoder(),dec=new TextDecoder('utf-8',{fatal:false});
 const toChars=piece=>[...enc.encode(piece)].map(b=>charOf.get(b));
 // the text a token stands for; a token that ends inside a multi-byte character has no text of its own
 const text=token=>dec.decode(Uint8Array.from([...token].map(c=>byteOf.get(c))));
 // Apply the merge rules to one piece. Returns every intermediate state, so the building-up can be shown.
 function mergeSteps(piece){
  let parts=toChars(piece);const steps=[{parts,rule:null}];
  for(;;){
   let best=Infinity,at=-1;
   for(let i=0;i<parts.length-1;i++){const r=rank.get(parts[i]+' '+parts[i+1]);if(r!==undefined&&r<best){best=r;at=i;}}
   if(at<0)break;
   const [a,b]=merges[best],next=[];
   for(let i=0;i<parts.length;i++){if(i<parts.length-1&&parts[i]===a&&parts[i+1]===b){next.push(a+b);i++;}else next.push(parts[i]);}
   parts=next;steps.push({parts,rule:best,pair:[a,b]});
  }
  return steps;
 }
 function encode(input){
  const out=[];
  for(const piece of input.match(PIECES)||[])for(const token of mergeSteps(piece).at(-1).parts)out.push({id:id.get(token),token,text:text(token),bytes:token.length});
  return out;
 }
 return {encode,mergeSteps,text,bytesOf:token=>[...token].map(c=>byteOf.get(c)),ids:s=>encode(s).map(t=>t.id),mergeCount:merges.length,merge:n=>merges[n]};
}
