/** Isolated 4D teaching example. Never mutates encoder tensors or its RoPE configuration. */
export const ROPE_LAB_Q=Object.freeze([1,0,1,0]);
export const ROPE_LAB_K=Object.freeze([.8,.6,.6,.8]);
export const ROPE_LAB_POSITIONS=Object.freeze({qr:1,qc:1,kr:3,kc:2});
export const ROPE_LAB_DURATION=2600;
export const ROPE_LAB_HOLD_MS=2000;
/** No image-grid limit: reject only invalid/non-exact JS integer positions. */
export function ropeLabPosition(value){
 if(String(value).trim()==='')return null;const n=Number(value);
 return Number.isSafeInteger(n)&&n>=0?n:null;
}
const ropeLabClamp=(n,min,max)=>Math.max(min,Math.min(max,Number(n)||0));
const rlNumber=(x,n=4)=>Math.abs(x)<.5*10**-n?(0).toFixed(n):x.toFixed(n);
const rlDegrees=r=>r*180/Math.PI;
const rlVec=v=>`[${v.map(x=>rlNumber(x)).join(', ')}]`;
export function ropeLabRotate([a,b],angle){const c=Math.cos(angle),s=Math.sin(angle);return [c*a-s*b,s*a+c*b];}
export function ropeLabTrace({positions=ROPE_LAB_POSITIONS,mode='2d',scale=1,progress=1,fromAngles=null}={}){
 progress=ropeLabClamp(progress,0,1);scale=ropeLabClamp(scale,.25,2);
 const pairs=[0,1].map(i=>{
  const q=ROPE_LAB_Q.slice(2*i,2*i+2),k=ROPE_LAB_K.slice(2*i,2*i+2),frequency=(i?Math.PI/12:Math.PI/6)*scale;
  const qPosition=mode==='1d'?positions.qr*8+positions.qc:positions[i?'qc':'qr'],kPosition=mode==='1d'?positions.kr*8+positions.kc:positions[i?'kc':'kr'];
  const targetQ=qPosition*frequency,targetK=kPosition*frequency,startQ=fromAngles?.[i]?.q??0,startK=fromAngles?.[i]?.k??0;
  const qAngle=startQ+(targetQ-startQ)*progress,kAngle=startK+(targetK-startK)*progress,qRotated=ropeLabRotate(q,qAngle),kRotated=ropeLabRotate(k,kAngle);
  const products=qRotated.map((v,d)=>v*kRotated[d]),dot=products[0]+products[1];
  return {index:i,q,k,qPosition,kPosition,frequency,targetQ,targetK,qAngle,kAngle,qRotated,kRotated,products,dot,axis:mode==='1d'?'sequence':i?'column':'row'};
 });
 return {pairs,progress,products:pairs.flatMap(p=>p.products),total:pairs.reduce((sum,p)=>sum+p.dot,0),deltaRow:positions.kr-positions.qr,deltaCol:positions.kc-positions.qc};
}
export function ropeLabShift(positions,kind='both',mode='2d',scale=1){
 const keys=kind==='both'?['qr','qc','kr','kc']:['kr','kc'];if(keys.some(k=>ropeLabPosition(positions[k])===null||!Number.isSafeInteger(positions[k]+1)))return null;
 const next={...positions};keys.forEach(k=>next[k]++);
 const before=ropeLabTrace({positions,mode,scale}),after=ropeLabTrace({positions:next,mode,scale});
 return {kind,positions:next,previous:{...positions},before,after,fromAngles:before.pairs.map(p=>({q:p.qAngle,k:p.kAngle}))};
}
/** A cancellable scalar timeline. Pause/resume retains the exact current rotation. */
export function createRopeLabPlayback(win,onFrame,onEnd){
 let frame=null,current=0,playing=false,disposed=false;
 function pause(){playing=false;if(frame!==null)win.cancelAnimationFrame(frame);frame=null;}
 function play(from=0,{immediate=false}={}){
  pause();if(disposed)return;current=ropeLabClamp(from,0,1);
  if(immediate||current===1){current=1;onFrame(1);onEnd();return;}
  playing=true;const initial=current,started=win.performance.now(),duration=ROPE_LAB_DURATION*(1-initial);onFrame(current);if(!playing||disposed)return;
  function tick(now){frame=null;if(!playing||disposed)return;current=Math.min(1,initial+(1-initial)*(now-started)/duration);onFrame(current);if(!playing||disposed)return;if(current===1){playing=false;onEnd();}else frame=win.requestAnimationFrame(tick);}
  frame=win.requestAnimationFrame(tick);
 }
 return {play,pause,get playing(){return playing;},get progress(){return current;},dispose(){pause();disposed=true;}};
}
/** Automatic loop: finish the turn, hold its actual endpoint, then start again.
 * Pausing preserves both the rotation and the remaining hold time. No repeated
 * DOM updates while holding, and no animation work when paused/reduced-motion. */
export function createRopeLabLoop(win,onFrame){
 let frame=null,last=0,elapsed=0,current=0,phase='rotating',running=false,disposed=false;
 function paint(){onFrame(current,phase);}
 function schedule(){if(running&&!disposed)frame=win.requestAnimationFrame(tick);}
 function tick(now){
  frame=null;if(!running||disposed)return;elapsed+=Math.max(0,now-last);last=now;
  if(phase==='rotating'){
   current=Math.min(1,elapsed/ROPE_LAB_DURATION);
   if(current===1){phase='holding';elapsed=0;}
   paint();
  }else if(elapsed>=ROPE_LAB_HOLD_MS){phase='rotating';elapsed=0;current=0;paint();}
  schedule();
 }
 function pause(){running=false;if(frame!==null)win.cancelAnimationFrame(frame);frame=null;}
 function resume(){
  if(disposed||running)return;if(phase==='static'){phase='rotating';elapsed=0;current=0;}
  running=true;last=win.performance.now();paint();schedule();
 }
 function reset(){pause();elapsed=0;current=0;phase='rotating';if(!disposed)paint();}
 function finish(){pause();elapsed=0;current=1;phase='static';if(!disposed)paint();}
 return {resume,pause,reset,finish,get running(){return running;},get phase(){return phase;},get progress(){return current;},dispose(){disposed=true;pause();}};
}
function rlArc(start,angle,r,cx=176,cy=127){
 if(!Number.isFinite(angle)||Math.abs(angle)<1e-8)return '';
 // Draw only the remaining turn after full revolutions. Keep the true angle in
 // the math and vector endpoint; never allocate paths proportional to position.
 const turn=2*Math.PI,shown=angle%turn||(angle<0?-turn:turn),count=Math.max(2,Math.ceil(Math.abs(shown)/.12));
 return Array.from({length:count+1},(_,i)=>{const a=start+shown*i/count;return `${i?'L':'M'}${(cx+r*Math.cos(a)).toFixed(3)},${(cy-r*Math.sin(a)).toFixed(3)}`;}).join('');
}
export function ropePairTurnExplanation(p){
 const axis=p.axis==='sequence'?'index':p.axis,same=p.qPosition===p.kPosition,qStart=rlDegrees(Math.atan2(p.q[1],p.q[0])),kStart=rlDegrees(Math.atan2(p.k[1],p.k[0]));
 return `<div class="rl-pair-reason"><strong>Same rate: ${rlNumber(rlDegrees(p.frequency),1)}° per ${axis} for Q and K.</strong><span>${same?`Same ${axis} → same added turn.`:`Different ${axis} positions → different added turns.`}</span><small>Q starts at ${rlNumber(qStart,1)}°; K starts at ${rlNumber(kStart,1)}°. The arrow’s direction is its <b>starting direction + the added RoPE turn</b>${same?' — so equal turns need not make the arrows overlap.':'.'}</small></div>`;
}
export function ropeLabPairMarkup(p){
 const cx=176,cy=127,r=83,name=p.axis==='sequence'?`Sequence pair ${p.index+1}`:`${p.index?'Column':'Row'} pair`,coord=v=>[cx+r*v[0],cy-r*v[1]];
 const arrows=['q','k'].map(key=>{
  const original=p[key],current=p[key+'Rotated'],angle=p[key+'Angle'],[x0,y0]=coord(original),[x,y]=coord(current),id=`rl-${key}-${p.index}`;
  return `<defs><marker id="${id}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L8,4L0,8Z" class="rl-${key}-fill"/></marker></defs><line class="rl-original rl-${key}-stroke" x1="${cx}" y1="${cy}" x2="${x0}" y2="${y0}"/><path class="rl-turn rl-${key}-stroke" d="${rlArc(Math.atan2(original[1],original[0]),angle,key==='q'?58:68)}" marker-end="url(#${id})"/><line data-rope-arrow="${key}-${p.index}" data-angle="${angle}" class="rl-arrow rl-${key}-stroke" x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" marker-end="url(#${id})"/><text class="rl-${key}-text" x="${x+(key==='q'?8:-19)}" y="${y+(key==='q'?17:-9)}">${key.toUpperCase()}${Math.abs(angle)>1e-8?'′':''}</text>`;
 }).join('');
 return `<article class="rl-pair"><header><h3>${name}</h3><span>features ${p.index*2} & ${p.index*2+1} · ${rlNumber(rlDegrees(p.frequency),1)}° per ${p.axis==='sequence'?'index step':p.axis}</span></header>
 <div class="rl-start-values"><span class="rl-q-text">Q starts at ${rlVec(p.q)}</span><span class="rl-k-text">K starts at ${rlVec(p.k)}</span></div>
 <svg class="rl-circle" viewBox="0 0 352 255" role="img" aria-label="${name}: blue Q at ${rlVec(p.qRotated)} and brown K at ${rlVec(p.kRotated)}. Arrows show feature coordinates, not image coordinates."><circle cx="${cx}" cy="${cy}" r="${r}" class="rl-unit-circle"/><path d="M51 ${cy}H301M${cx} 21V226" class="rl-axes"/><text x="279" y="${cy+19}" class="rl-axis-label">feature ${p.index*2}</text><text x="${cx+8}" y="22" class="rl-axis-label">feature ${p.index*2+1}</text><text x="${cx+r}" y="${cy+17}" class="rl-tick">1</text><text x="${cx-r-12}" y="${cy+17}" class="rl-tick">−1</text><text x="${cx-18}" y="${cy-r+4}" class="rl-tick">1</text><text x="${cx-22}" y="${cy+r+4}" class="rl-tick">−1</text>${arrows}<circle cx="${cx}" cy="${cy}" r="2.5" class="rl-origin"/></svg>
 <div class="rl-angle-row"><small class="rl-target-label">Target turns from image positions</small><span class="rl-q-text">Q: ${p.qPosition} × ${rlNumber(rlDegrees(p.frequency),1)}° = <b>${rlNumber(rlDegrees(p.targetQ),1)}°</b></span><span class="rl-k-text">K: ${p.kPosition} × ${rlNumber(rlDegrees(p.frequency),1)}° = <b>${rlNumber(rlDegrees(p.targetK),1)}°</b></span></div>
 ${ropePairTurnExplanation(p)}
 <div class="rl-current"><span class="rl-q-text">Now Q′ ${rlVec(p.qRotated)} <small>turned ${rlNumber(rlDegrees(p.qAngle),1)}°</small></span><span class="rl-k-text">Now K′ ${rlVec(p.kRotated)} <small>turned ${rlNumber(rlDegrees(p.kAngle),1)}°</small></span></div>
 </article>`;
}
export function ropeLabSumMarkup(trace){
 return `<div class="rl-products">${trace.pairs.map(p=>`<div><h4>${p.axis==='sequence'?`Pair ${p.index+1}`:`${p.index?'Column':'Row'} pair`}</h4>${p.products.map((v,d)=>`<div class="rl-product"><small>feature ${p.index*2+d}</small><code><span class="rl-q-text">${rlNumber(p.qRotated[d],5)}</span> × <span class="rl-k-text">${rlNumber(p.kRotated[d],5)}</span> = <b>${rlNumber(v,6)}</b></code></div>`).join('')}<div class="rl-pair-sum">Pair contribution <strong>${rlNumber(p.dot,6)}</strong></div></div>`).join('')}</div><div class="rl-total"><span>${trace.progress===0?'Before rotation':trace.progress<1?'While rotating':'After RoPE'} · add both pairs</span><code>${rlNumber(trace.pairs[0].dot,6)} + (${rlNumber(trace.pairs[1].dot,6)}) = <strong id="rope-score">${rlNumber(trace.total,6)}</strong></code></div>`;
}
export function ropeLabMatrixMarkup(trace,index=0,vector='q'){
 const p=trace.pairs[index],angle=p[vector+'Angle'],v=p[vector],out=p[vector+'Rotated'],c=Math.cos(angle),s=Math.sin(angle);
 return `<p><b>R(θ) is a 2 × 2 rotation matrix.</b> It takes two feature values and returns two rotated values. θ (theta) is the extra angle to turn, counterclockwise when positive.</p>
 <div class="rl-matrix-formula"><span>R(θ) =</span><div class="rl-matrix"><span>cos θ</span><span>−sin θ</span><span>sin θ</span><span>cos θ</span></div><span>so</span><code>x′ = x cos θ − y sin θ<br>y′ = x sin θ + y cos θ</code></div>
 <div class="rl-matrix-formula rl-numeric-matrix"><span>R(${rlNumber(rlDegrees(angle),1)}°)</span><div class="rl-matrix"><span>${rlNumber(c)}</span><span>${rlNumber(-s)}</span><span>${rlNumber(s)}</span><span>${rlNumber(c)}</span></div><span>×</span><div class="rl-column"><span>${rlNumber(v[0])}</span><span>${rlNumber(v[1])}</span></div><span>=</span><div class="rl-column"><span>${rlNumber(out[0])}</span><span>${rlNumber(out[1])}</span></div></div>
 <p class="rl-component-math"><code>x′ = (${rlNumber(c)}) × ${v[0]} − (${rlNumber(s)}) × ${v[1]} = ${rlNumber(out[0],6)}<br>y′ = (${rlNumber(s)}) × ${v[0]} + (${rlNumber(c)}) × ${v[1]} = ${rlNumber(out[1],6)}</code></p><small>Trig uses radians internally: ${rlNumber(rlDegrees(angle),1)}° = ${rlNumber(angle)} rad. The arrow’s length stays ${rlNumber(Math.hypot(...v))}; only its direction changes.</small>`;
}
/** One Q patch, two independent feature pairs superimposed for comparison. */
export function ropeSingleCircleMarkup(trace){
 const cx=176,cy=127,r=83;
 const arrows=trace.pairs.map((p,i)=>{
  const [a,b]=p.qRotated,x=cx+r*a,y=cy-r*b,name=i?'Column pair':'Row pair',id=`rl-single-${i}`;
  return `<defs><marker id="${id}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" class="rl-axis-${i}-fill"/></marker></defs><path class="rl-turn rl-axis-${i}-stroke" d="${rlArc(0,p.qAngle,i?64:53)}" marker-end="url(#${id})"/><line class="rl-arrow rl-axis-${i}-stroke" data-rope-single-pair="${i}" data-angle="${p.qAngle}" x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" marker-end="url(#${id})"/><text class="rl-axis-${i}-text" x="${x+7}" y="${y+(i?20:-9)}">${name}</text>`;
 }).join('');
 return `<svg class="rl-single-circle" viewBox="0 0 352 255" role="img" aria-label="One query patch: blue row pair rotated ${rlNumber(rlDegrees(trace.pairs[0].qAngle),1)} degrees and purple column pair rotated ${rlNumber(rlDegrees(trace.pairs[1].qAngle),1)} degrees. Two different pairs shown on the same feature-coordinate plane, not consecutive turns of one vector."><circle cx="${cx}" cy="${cy}" r="${r}" class="rl-unit-circle"/><path d="M51 ${cy}H301M${cx} 21V226" class="rl-axes"/><text x="${cx+7}" y="20" class="rl-axis-label">second value</text><text x="264" y="${cy+39}" class="rl-axis-label">first value</text><text x="${cx+r}" y="${cy+16}" class="rl-tick">1</text><text x="${cx-r-13}" y="${cy+16}" class="rl-tick">−1</text><text x="${cx-15}" y="${cy-r+4}" class="rl-tick">1</text><line class="rl-single-original" x1="${cx}" y1="${cy}" x2="${cx+r}" y2="${cy}"/>${arrows}<circle cx="${cx}" cy="${cy}" r="2.5" class="rl-origin"/></svg>`;
}
/** Display-only decomposition, rounded like the on-screen angle; never used by RoPE math. */
export function ropeLabTurns(angle){
 const degrees=Number(rlDegrees(angle).toFixed(1)),turns=Math.floor(degrees/360),remainder=Number((degrees-turns*360).toFixed(1));
 return {degrees,turns,remainder};
}
export function ropeTurnsMarkup(angle){
 const t=ropeLabTurns(angle);
 return `<small class="rl-turn-summary" data-rl-turn-count="${t.turns}">Now ${rlNumber(t.degrees,1)}°${t.turns?` = ${t.turns} full ${t.turns===1?'turn':'turns'} + ${rlNumber(t.remainder,1)}°`:''}</small>`;
}
export function ropeWrapExplanationMarkup(){return `<details class="rl-wrap-explanation"><summary>Beyond 360°: how is 361° different from 1°?</summary>
 <p><b>361° = 360° + 1°.</b> That is one full turn, then one more degree. Starting from the same vector, 361° and 1° produce <b>the same final direction and the same two values</b>. The animation travels farther, but the rotated vector does not store how many full turns it made.</p>
 <p><b>Position indices are not degrees.</b> An index is multiplied by each pair’s angular frequency. In this toy, row 1 gives 30° and row 361 gives 10,830° = 30 full turns + 30° — they happen to match. Column 1 and 361 also match at these teaching speeds. This is <b>not a general 360-position reset</b> for RoPE.</p>
 <p>Different feature pairs use different frequencies. Matching one pair’s direction does not generally match every pair’s direction. The full positional rotation repeats only if every pair completes a whole number of turns for that position change.</p>
 </details>`;}
export function ropeSingleValuesMarkup(trace){
 return `<div class="rl-single-values">${trace.pairs.map((p,i)=>`<div class="rl-axis-${i}"><span><b>${i?'Column':'Row'} pair</b> · features ${i*2}, ${i*2+1}</span><code>[1, 0] → ${rlVec(p.qRotated)}</code><small>position ${p.qPosition} × ${rlNumber(rlDegrees(p.frequency),0)}° = ${rlNumber(rlDegrees(p.targetQ),0)}° target</small>${ropeTurnsMarkup(p.qAngle)}</div>`).join('')}</div>`;
}
function rlPlaybackMarkup(label){
 return `<div class="rl-compact-playback"><div><button type="button" data-rl-play>▶ ${label}</button><button type="button" data-rl-reset>Reset</button></div><label>Rotation <input type="range" data-rl-progress min="0" max="100" step="1" value="0" aria-label="Rotation progress"><output data-rl-percent>0%</output></label><p data-rl-narration></p><span class="sr-only" role="status" data-rl-status></span></div>`;
}
export function ropeLabMarkup(){return `
 <div class="rl-single-layout">
  <div class="rl-single-controls"><div class="rl-single-heading">ONE PATCH · QUERY VECTOR Q</div><h3>Position turns its feature pairs.</h3><p>A tiny Q vector has <b>4 features</b>: <code class="rl-axis-0">[1, 0]</code> for the row pair and <code class="rl-axis-1">[1, 0]</code> for the column pair. Both start pointing right.</p>
  <div class="rl-position-row"><label>Patch row <input id="qr" data-rl-position="qr" type="number" min="0" step="1" value="1" aria-label="Query row"></label><label>column <input id="qc" data-rl-position="qc" type="number" min="0" step="1" value="1" aria-label="Query column"></label></div>
  <p class="rl-single-rule"><b class="rl-axis-0">Blue:</b> row × 30° &nbsp; <b class="rl-axis-1">Purple:</b> column × 15°<br>These are separate turns of <b>different feature pairs</b>, not two turns applied to the same vector.</p>
  <p class="rl-loop-caption" data-rl-narration>Rotates automatically · holds for 2 seconds · repeats.</p>
  </div>
  <div class="rl-single-visual"><div data-rl-single-circle></div><div data-rl-single-values></div><p class="rl-single-caption">Dashed gray = starting direction. Solid arrows = the two pairs of this one Q. The circle shows feature values, not image locations. For turns beyond 360°, the arc shows only the remaining turn; the numbers retain the full angle.</p>${ropeWrapExplanationMarkup()}</div>
 </div>
 <div class="rl-overlay-links"><button type="button" data-rl-overlay="math" aria-haspopup="dialog" aria-controls="rope-detail-dialog"><span>↗</span><div><b>See the math</b><small>Rotation matrix → rotated values → dot product</small></div></button><button type="button" data-rl-overlay="comparison" aria-haspopup="dialog" aria-controls="rope-detail-dialog"><span>↗</span><div><b>Why move two patches together?</b><small>Introduce K and test relative position</small></div></button></div>
 <p class="rl-single-notice">Isolated teaching example: 4 features, not the real 64-feature heads above. The 30° / 15° angles are chosen for clarity, not Pixtral’s frequency schedule. Q’s starting values are fixed toy features, not RGB values or image coordinates.</p>`;}
export function ropeLabOverlayState(main,kind){
 return {positions:{...main.positions},mode:main.mode||'2d',scale:main.scale||1,progress:kind==='comparison'?0:main.progress,pair:0,vector:'q',experiment:null,fromAngles:null,kind};
}
export function ropeMathOverlayMarkup(state){
 return `<p class="rl-dialog-intro">One calculation in two steps: first rotate each pair with <b>R</b>; then, for attention, multiply Q′ and K′ feature by feature. This overlay copies your current Q position and rotation progress.</p>
 ${rlPlaybackMarkup('Animate the calculation')}
 <section class="rl-matrix-section"><h3>1 · R turns a pair into two new values</h3><div class="rl-matrix-controls" role="group" aria-label="Rotation matrix to inspect"><button type="button" data-rl-pair="0" aria-pressed="true">Row pair</button><button type="button" data-rl-pair="1" aria-pressed="false">Column pair</button><button type="button" data-rl-vector="q" aria-pressed="true">Q</button><button type="button" data-rl-vector="k" aria-pressed="false">K</button></div><div data-rl-matrix></div></section>
 <section class="rl-score-section"><h3>2 · Match features, multiply, and add</h3><p>A dot product needs two vectors, so <b>only here we introduce a key K</b>, at (row ${state.positions.kr}, column ${state.positions.kc}). Its starting pairs are [0.8, 0.6] and [0.6, 0.8]. Q starts with [1, 0] and [1, 0] at (row ${state.positions.qr}, column ${state.positions.qc}).</p><div class="rl-legend"><span><i class="rl-q-key"></i>Blue = Q values</span><span><i class="rl-k-key"></i>Brown = K values</span></div><div data-rl-sum></div><small>No hidden pairs contribute: all four matching-feature products are shown. This is the raw dot product, before ÷ √d and softmax — not a probability. Shown numbers are rounded; the sum uses full precision.</small></section>`;
}
export function ropeComparisonOverlayMarkup(state){return `
 <p class="rl-dialog-intro">Now use two patches: <b class="rl-q-text">blue Q</b> and <b class="rl-k-text">brown K</b>. Each has a row pair and a column pair. First rotate both by their image positions, then move both positions together and watch the score. The rotation rule is shared; <b>each patch’s own position sets its angle</b>.</p>
 <div class="rl-comparison-positions">${['q','k'].map(k=>`<div class="rl-position-row"><strong class="rl-${k}-text">${k.toUpperCase()} patch</strong><label>row <input data-rl-position="${k}r" type="number" min="0" step="1" value="${state.positions[k+'r']}" aria-label="${k==='q'?'Query':'Key'} row"></label><label>column <input data-rl-position="${k}c" type="number" min="0" step="1" value="${state.positions[k+'c']}" aria-label="${k==='q'?'Query':'Key'} column"></label></div>`).join('')}</div>
 <div data-rl-motion-start></div><div class="rl-comparison-score"><p data-rl-narration aria-live="off"></p></div><div class="rl-shift-buttons"><button type="button" data-rl-shift="both" disabled>Move both: +1 row, +1 column</button><button type="button" data-rl-shift="key" disabled>Compare: move only K</button></div>
 <div class="rl-legend"><span><i class="rl-q-key"></i>Blue = Q, not the row pair here</span><span><i class="rl-k-key"></i>Brown = K</span><span><i class="rl-dashed-key"></i>Dashed = original features</span></div>
 <div class="rl-pairs" data-rl-pairs></div>
 <section class="rl-shift-section"><h3>Move the positions, keep the starting features fixed</h3><div data-rl-comparison></div><p>Moving both gives Q and K the <b>same extra turn within each pair</b>. Their relative angle stays the same, so their dot product stays the same. Moving only K changes their relative angle.</p><p class="rl-shift-caveat">This fixes Q/K’s original feature values. It does not claim that moving image content through a whole model leaves its prediction unchanged.</p></section>
 <details class="rl-identity"><summary>The relative-position identity, in plain language</summary><code>(Rᵢq)ᵀ(Rⱼk) = qᵀRⱼ₋ᵢk</code><p>Rᵢ and Rⱼ rotate the pairs for Q’s and K’s positions. The transpose ᵀ makes the left side a dot product: multiply matching values and add. The right side depends on the displacement j − i — the row and column differences — rather than a shared shift of both positions.</p></details>`;}
function rlComparisonResult(state,trace){
 const exp=state.experiment;
 return exp?`<div class="rl-shift-result"><strong>${exp.kind==='both'?'Same relative position':'Changed relative position'}</strong><span>Q (${exp.previous.qr}, ${exp.previous.qc}) → (${state.positions.qr}, ${state.positions.qc}) · K (${exp.previous.kr}, ${exp.previous.kc}) → (${state.positions.kr}, ${state.positions.kc})</span><span>Δ(row, column): (${exp.before.deltaRow}, ${exp.before.deltaCol}) → (${trace.deltaRow}, ${trace.deltaCol})</span><code>Before ${rlNumber(exp.before.total,6)} → ${state.progress===1?'After':'Now'} ${rlNumber(trace.total,6)}</code><b>Score change: ${rlNumber(trace.total-exp.before.total,6)}</b></div>`:`<div class="rl-shift-result"><span>Δ(row, column): (${trace.deltaRow}, ${trace.deltaCol})</span><code>${state.progress===1?'After RoPE':'Current'} dot product: ${rlNumber(trace.total,6)}</code><small>${state.progress===1?'Ready: move both patches or compare moving only K.':'Move buttons unlock when the rotation is complete.'}</small></div>`;
}
export function mountRopeLab(root){
 const doc=root.ownerDocument,win=doc.defaultView,media=win.matchMedia('(prefers-reduced-motion: reduce)');
 const initial=()=>({positions:{...ROPE_LAB_POSITIONS},mode:'2d',scale:1,progress:0});
 let state=initial(),overlayState=null,source=null,disposed=false,active=false,inView=false;
 root.innerHTML=ropeLabMarkup();const q=s=>root.querySelector(s);
 const dialog=doc.createElement('dialog');dialog.id='rope-detail-dialog';dialog.className='full-math-dialog rl-dialog';dialog.setAttribute('aria-labelledby','rope-detail-title');doc.body.append(dialog);
 const dq=s=>dialog.querySelector(s);
 const playback=createRopeLabLoop(win,p=>{state.progress=p;render();});
 const detailPlayback=createRopeLabPlayback(win,p=>{if(!overlayState||overlayState.kind!=='math')return;overlayState.progress=p;renderDetail();},()=>{renderDetail();const status=dq('[data-rl-status]');if(status)status.textContent='Rotation complete.';});
 const comparisonPlayback=createRopeLabPlayback(win,p=>{if(!overlayState||overlayState.kind!=='comparison')return;overlayState.progress=p;renderDetail();},renderDetail);
 function controls(scope,s,player){
  const get=sel=>scope.querySelector(sel);get('[data-rl-progress]').value=s.progress*100;get('[data-rl-percent]').textContent=`${Math.round(s.progress*100)}%`;
  get('[data-rl-play]').textContent=player.playing?'Ⅱ Pause':s.progress===1?'↻ Replay rotation':s.progress>0?'▶ Resume rotation':s.kind==='math'?'▶ Animate the calculation':'▶ Rotate Q and K';
 }
 function render(){if(disposed)return;const trace=ropeLabTrace(state);root.dataset.progress=state.progress;root.dataset.playing=playback.running;root.dataset.phase=playback.phase;
  q('[data-rl-single-circle]').innerHTML=ropeSingleCircleMarkup(trace);q('[data-rl-single-values]').innerHTML=ropeSingleValuesMarkup(trace);
  q('[data-rl-narration]').textContent=media.matches?'Full rotation shown · reduced motion.':'Rotates automatically · holds for 2 seconds · repeats.';
 }
 function renderDetail(){if(disposed||!dialog.open||!overlayState)return;const s=overlayState,trace=ropeLabTrace(s);dialog.dataset.kind=s.kind;dialog.dataset.progress=s.progress;dialog.dataset.playing=s.kind==='comparison'?comparisonPlayback.playing:detailPlayback.playing;dialog.dataset.phase=s.kind==='comparison'?(s.progress===1?'complete':comparisonPlayback.playing?'rotating':'paused'):'manual';
  if(s.kind==='math'){controls(dialog,s,detailPlayback);
   dq('[data-rl-matrix]').innerHTML=ropeLabMatrixMarkup(trace,s.pair,s.vector);dq('[data-rl-sum]').innerHTML=ropeLabSumMarkup(trace);
   dialog.querySelectorAll('[data-rl-pair]').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.rlPair===s.pair));dialog.querySelectorAll('[data-rl-vector]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.rlVector===s.vector));
   dq('[data-rl-narration]').textContent=`Current rotation: ${Math.round(s.progress*100)}%. The matrix and all four dot-product terms use these same rotated values.`;
  }else{
   dq('[data-rl-pairs]').innerHTML=trace.pairs.map(ropeLabPairMarkup).join('');dq('[data-rl-comparison]').innerHTML=rlComparisonResult(s,trace);
   dialog.querySelectorAll('[data-rl-shift]').forEach(b=>{b.disabled=s.progress!==1||!ropeLabShift(s.positions,b.dataset.rlShift,s.mode,s.scale);});
   dq('[data-rl-narration]').textContent=s.experiment?`${s.experiment.kind==='both'?'Both move together':'Only K moves'}: score ${rlNumber(s.experiment.before.total,6)} → ${rlNumber(trace.total,6)} · change ${rlNumber(trace.total-s.experiment.before.total,6)}.`:`${s.progress===1?'After RoPE':'During rotation'} · Q′ · K′ = ${rlNumber(trace.total,6)}. All four features contribute.`;
  }
 }
 function syncInputs(scope,s){scope.querySelectorAll('[data-rl-position]').forEach(input=>{input.value=s.positions[input.dataset.rlPosition];input.setCustomValidity?.('');});}
 function close(){detailPlayback.pause();comparisonPlayback.pause();if(dialog.open)dialog.close();}
 function closed(){if(dialog.open||!active)return;active=false;detailPlayback.pause();comparisonPlayback.pause();overlayState=null;dialog.innerHTML='';doc.documentElement.classList.remove('rope-overlay-open');if(source?.isConnected)source.focus({preventScroll:true});maybeAutoplay();}
 function open(kind,origin){playback.pause();render();detailPlayback.pause();comparisonPlayback.pause();overlayState=ropeLabOverlayState(state,kind);source=origin;active=true;
  dialog.innerHTML=`<header class="fm-dialog-header"><div><span>2D ROPE · ${kind==='math'?'THE NUMBERS':'RELATIVE POSITION'}</span><h2 id="rope-detail-title">${kind==='math'?'From rotation matrix to dot product':'Two patches, one relative displacement'}</h2></div><button type="button" data-rl-close aria-label="Close RoPE explanation">×</button></header><div class="fm-dialog-body">${kind==='math'?ropeMathOverlayMarkup(overlayState):ropeComparisonOverlayMarkup(overlayState)}<p class="rl-single-notice">Isolated 4-feature example with fixed toy Q/K values and teaching angular speeds. No encoder values are changed.</p></div>`;
  doc.documentElement.classList.add('rope-overlay-open');dialog.showModal();dialog.scrollTop=0;if(kind==='comparison'){renderDetail();syncComparisonPlayback();}else renderDetail();dq('[data-rl-close]').focus({preventScroll:true});
 }
 function showDetailMotion(){if(overlayState?.kind!=='comparison')return;const anchor=dq('[data-rl-motion-start]'),header=dq('.fm-dialog-header');if(!anchor||!header)return;const offset=anchor.getBoundingClientRect().top-dialog.getBoundingClientRect().top-header.getBoundingClientRect().height-12;if(Number.isFinite(offset))dialog.scrollTop+=offset;}
 function startDetail(){showDetailMotion();if(overlayState.progress===1){overlayState.progress=0;overlayState.experiment=null;overlayState.fromAngles=null;}detailPlayback.play(overlayState.progress,{immediate:media.matches||doc.hidden});}
 function mainClick(e){const b=e.target.closest('button');if(!b||!root.contains(b)||b.disabled)return;
  if(b.hasAttribute('data-rl-overlay'))open(b.dataset.rlOverlay,b);
 }
 function detailClick(e){const b=e.target.closest('button');if(b?.hasAttribute('data-rl-close')){close();return;}
  if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)close();return;}
  if(!b||b.disabled||!overlayState)return;const s=overlayState;
  if(s.kind==='comparison'&&b.hasAttribute('data-rl-shift')){
   if(s.progress!==1)return;const experiment=ropeLabShift(s.positions,b.dataset.rlShift,s.mode,s.scale);if(!experiment)return;comparisonPlayback.pause();Object.assign(s,{experiment,fromAngles:experiment.fromAngles,positions:experiment.positions,progress:0});syncInputs(dialog,s);showDetailMotion();renderDetail();syncComparisonPlayback();return;
  }
  if(s.kind!=='math')return;
  if(b.hasAttribute('data-rl-play')){if(detailPlayback.playing){detailPlayback.pause();renderDetail();}else startDetail();}
  else if(b.hasAttribute('data-rl-reset')){detailPlayback.pause();overlayState=ropeLabOverlayState(state,s.kind);overlayState.progress=0;syncInputs(dialog,overlayState);renderDetail();}
  else if(b.hasAttribute('data-rl-pair')){s.pair=+b.dataset.rlPair;renderDetail();}
  else if(b.hasAttribute('data-rl-vector')){s.vector=b.dataset.rlVector;renderDetail();}
 }
 function slider(e){if(!e.target.hasAttribute('data-rl-progress')||!overlayState||overlayState.kind!=='math')return;detailPlayback.pause();overlayState.progress=ropeLabClamp(e.target.value/100,0,1);renderDetail();}
 function commit(e,detail=false){
  const name=e.target.dataset.rlPosition;if(!name)return;const value=ropeLabPosition(e.target.value);
  e.target.setCustomValidity?.(value===null?'Enter a non-negative whole number that JavaScript can represent exactly.':'');
  if(value===null)return;const s=detail?overlayState:state;if(!s||s.positions[name]===value)return;
  const player=detail?(s.kind==='comparison'?comparisonPlayback:detailPlayback):playback;player.pause();s.positions[name]=value;s.progress=0;s.experiment=null;s.fromAngles=null;
  // Input events replay immediately (including spinner arrows); the following
  // change/Enter event is ignored when the position is already applied.
  if(detail){if(s.kind==='comparison'){renderDetail();syncComparisonPlayback();}else{renderDetail();detailPlayback.play(0,{immediate:media.matches||doc.hidden});}}
  else{playback.reset();maybeAutoplay();}
 }
 const mainInput=e=>{if(e.target.hasAttribute('data-rl-position'))commit(e);},detailInput=e=>{if(e.target.hasAttribute('data-rl-position'))commit(e,true);else slider(e);},mainChange=e=>commit(e),detailChange=e=>commit(e,true);
 const numberKey=(e,detail)=>{if(e.key==='Enter'&&e.target.hasAttribute('data-rl-position')){e.preventDefault();commit(e,detail);}},mainKey=e=>numberKey(e,false),detailKey=e=>numberKey(e,true);
 function syncComparisonPlayback(){
  if(disposed)return;
  if(!active||!dialog.open||overlayState?.kind!=='comparison'||doc.hidden||!root.isConnected){comparisonPlayback.pause();return;}
  // Completion is terminal for this opening/edit/move. Visibility or preference
  // changes must not restart it; reopening creates a fresh state at progress 0.
  if(overlayState.progress===1){comparisonPlayback.pause();renderDetail();return;}
  if(media.matches)comparisonPlayback.play(overlayState.progress,{immediate:true});
  else if(!comparisonPlayback.playing)comparisonPlayback.play(overlayState.progress);
 }
 function maybeAutoplay(){
  if(disposed)return;
  if(!inView||doc.hidden||active||!root.isConnected){playback.pause();render();return;}
  if(media.matches){if(playback.phase!=='static')playback.finish();else render();}
  else playback.resume();
 }
 const observer=typeof win.IntersectionObserver==='function'?new win.IntersectionObserver(entries=>{
  if(disposed)return;const entry=entries.at(-1);if(!entry)return;inView=entry.isIntersecting&&entry.intersectionRatio>=.35;maybeAutoplay();
 },{threshold:[0,.35]}):null;
 const visibility=()=>{if(doc.hidden){detailPlayback.pause();comparisonPlayback.pause();renderDetail();}else syncComparisonPlayback();maybeAutoplay();},reduced=()=>{
  if(media.matches&&detailPlayback.playing&&overlayState){detailPlayback.pause();overlayState.progress=1;renderDetail();}
  syncComparisonPlayback();maybeAutoplay();
 };
 root.addEventListener('click',mainClick);root.addEventListener('input',mainInput);root.addEventListener('change',mainChange);root.addEventListener('keydown',mainKey);
 dialog.addEventListener('click',detailClick);dialog.addEventListener('input',detailInput);dialog.addEventListener('change',detailChange);dialog.addEventListener('keydown',detailKey);dialog.addEventListener('close',closed);dialog.addEventListener('cancel',close);
 doc.addEventListener('visibilitychange',visibility);media.addEventListener('change',reduced);render();
 if(observer)observer.observe(q('.rl-single-visual'));else{inView=true;maybeAutoplay();}
 return {dispose(){disposed=true;observer?.disconnect();playback.dispose();detailPlayback.dispose();comparisonPlayback.dispose();close();closed();dialog.remove();root.removeEventListener('click',mainClick);root.removeEventListener('input',mainInput);root.removeEventListener('change',mainChange);root.removeEventListener('keydown',mainKey);doc.removeEventListener('visibilitychange',visibility);media.removeEventListener('change',reduced);}};
}
