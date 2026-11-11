// A deliberately fictional chat template; these are not IDs for any real model.
export const SPECIAL_TOKENS=['<bos>','<message>','<text>','<end>'];
export const INSTRUCT_EXAMPLES=[
 {prompt:'Say hello.',answer:'Hello!',source:'Human-written demonstration'},
 {prompt:'Name a color.',answer:'Blue.',source:'Model draft, checked by a reviewer'},
];
const pieces=text=>text.match(/[A-Za-z]+|[.!?]/g)||[];
export const INSTRUCT_VOCAB=[...SPECIAL_TOKENS,...new Set(['user','assistant',...INSTRUCT_EXAMPLES.flatMap(e=>[...pieces(e.prompt),...pieces(e.answer)])])];
export function chatTokens(example=0,{generation=false}={}){
 if(!Number.isInteger(example)||!INSTRUCT_EXAMPLES[example])throw new RangeError('Unknown example');
 const e=INSTRUCT_EXAMPLES[example],rows=[];
 const add=(text,kind,supervised=false)=>rows.push({text,id:INSTRUCT_VOCAB.indexOf(text),kind,supervised,special:SPECIAL_TOKENS.includes(text)});
 add('<bos>','boundary');add('<message>','boundary');add('user','role');add('<text>','boundary');pieces(e.prompt).forEach(t=>add(t,'prompt'));add('<end>','boundary');add('<message>','boundary');add('assistant','role');add('<text>','boundary');
 if(!generation){pieces(e.answer).forEach(t=>add(t,'answer',true));add('<end>','answer',true);}
 return rows;
}
// Illustrative target probabilities, NOT predictions from a trained model.
// A full normalized distribution is constructed so the CE/logit derivative is exact.
export function targetDistribution(target,p){
 if(!Number.isInteger(target)||target<0||target>=INSTRUCT_VOCAB.length||!(p>0&&p<1))throw new RangeError('Invalid target/probability');
 return INSTRUCT_VOCAB.map((_,i)=>i===target?p:(1-p)/(INSTRUCT_VOCAB.length-1));
}
export function lossComparison(mode='sft',override=null){
 if(!['sft','full'].includes(mode))throw new RangeError('Unknown mask');
 const rows=INSTRUCT_EXAMPLES.flatMap((_,b)=>chatTokens(b).slice(1).map((t,j)=>{
 const position=j+1,p=override&&override.batch===b&&override.position===position?override.p:[.18,.32,.6,.24,.45][(j+b)%5];
 const probabilities=targetDistribution(t.id,p),included=mode==='full'||t.supervised;
 return {...t,batch:b,position,p,probabilities,included,loss:-Math.log(p),context:chatTokens(b).slice(0,position)};
 }));
 const count=rows.filter(r=>r.included).length,sum=rows.reduce((n,r)=>n+(r.included?r.loss:0),0);
 return {rows,count,sum,loss:sum/count};
}
export function logitGradient(row,count){
 if(!(count>0))throw new RangeError('Need supervised targets');
 return row.probabilities.map((p,i)=>row.included?(p-Number(i===row.id))/count:0);
}
export const DATA_STAGES=[
 {title:'Collect demonstrations',note:'Start with varied tasks. People can write answers, existing datasets can be adapted, or models can draft answers for checking.'},
 {title:'Review & deduplicate',note:'Check correctness, instruction-following, source permissions and privacy. Reject weak or repeated examples; synthetic does not mean verified.'},
 {title:'Format & hold out',note:'Store roles and content, apply the model’s chat template, and build loss labels. Keep related tasks and near-duplicates out of the evaluation split.'},
];
export const DATA_RECORDS=[
 {prompt:'Say hello.',answer:'Hello!',source:'Human-written',kept:true,split:'Train'},
 {prompt:'Name a color.',answer:'Blue.',source:'Model draft + human check',kept:true,split:'Held-out evaluation'},
 {prompt:'Say hello.',answer:'Hello!',source:'Repeated record',kept:false,reason:'Duplicate'},
 {prompt:'Name a color.',answer:'Tuesday.',source:'Unchecked model draft',kept:false,reason:'Does not answer the task'},
];
