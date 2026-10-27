import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matrix,viewMatrix,weightDescriptor} from '../src/full-config.mjs';
import {createMatrixRenderer,overviewPixels} from '../src/full-matrix.mjs';
import {isWeightMatrix,matrixRowOffset,matrixRowSummary,matrixRowGuides,patchRowLabel} from '../src/matrix-row-guides.mjs';
import {MULTIPLICATION_PLAYBACK_MS,multiplicationPlaybackCount,fullDotTrace} from '../src/full-overlay.mjs';

test('48 patch rows have 49 exact boundaries, stronger guides every four, no hidden rows',()=>{
 const guides=matrixRowGuides(48,192,{selected:21});
 assert.equal(guides.stride,1);assert.equal(guides.lines.length,49);assert.equal(guides.cellHeight,4);
 assert.deepEqual(guides.lines.map(l=>l.row),Array.from({length:49},(_,i)=>i));
 for(const line of guides.lines){assert.equal(line.y,line.row*4);assert.equal(line.major,line.row%4===0);}
 assert.ok(guides.labels.some(l=>l.label==='p0'));assert.ok(guides.labels.some(l=>l.label==='p47'));assert.ok(guides.labels.some(l=>l.label==='p21'&&l.selected));
});
test('dense label spacing remains readable while the selected row is always identified',()=>{
 for(const rows of [1,2,36,37,48,64,192])for(const height of [78,125,190,315])for(const selected of [0,Math.floor(rows/2),rows-1]){
  const g=matrixRowGuides(rows,height,{selected});
  assert.equal(g.labels.filter(l=>l.selected).length,1);assert.equal(g.labels.find(l=>l.selected).row,selected);
  for(let i=1;i<g.labels.length;i++)assert.ok(g.labels[i].y-g.labels[i-1].y>=12-1e-9);
  assert.ok(g.labels.every(l=>l.row>=0&&l.row<rows&&l.y>=0&&l.y<=height));
  assert.equal(g.lines[0].y,0);assert.equal(g.lines.at(-1).y,height);
 }
});
test('large grids thin only visual guides and disclose the stride',()=>{
 const g=matrixRowGuides(192,125,{selected:191});assert.ok(g.stride>1);assert.ok(g.lines.length<193);assert.equal(g.lines.at(-1).row,192);assert.equal(g.labels.find(l=>l.selected).label,'p191');
});
test('ViT rulers distinguish CLS from patch zero and report the actual token count',()=>{
 assert.equal(patchRowLabel(0,1),'CLS');assert.equal(patchRowLabel(1,1),'p0');assert.equal(patchRowLabel(36,1),'p35');
 const g=matrixRowGuides(37,190,{clsOffset:1,selected:0});assert.equal(g.labels[0].label,'CLS');assert.ok(g.labels.some(l=>l.label==='p35'));
 assert.equal(matrixRowSummary(37,1),'37 token rows · 36 patches + CLS');assert.equal(matrixRowSummary(48),'48 patch rows');
});
test('single-row and nested views preserve patch identities instead of restarting at p0',()=>{
 const a=matrix(48,1024),view=viewMatrix(a,{rowStart:20,rows:4}),nested=viewMatrix(view,{rowStart:1,rows:1});
 assert.equal(matrixRowOffset(nested),21);assert.equal(matrixRowGuides(1,78,{offset:matrixRowOffset(nested),selected:0}).labels[0].label,'p21');
 assert.equal(matrixRowSummary(1,0,21),'1 patch row');assert.equal(matrixRowGuides(1,78,{offset:21,clsOffset:1,selected:0}).labels[0].label,'p20');
});
test('weight matrix rows are features, including nested slices, never mislabeled as patches',()=>{
 const w=weightDescriptor(1024,1024,21),slice=viewMatrix(w,{colStart:960,cols:64}),nested=viewMatrix(slice,{rowStart:10,rows:2});
 assert.ok(isWeightMatrix(w));assert.ok(isWeightMatrix(slice));assert.ok(isWeightMatrix(nested));assert.ok(!isWeightMatrix(matrix(48,1024)));
 const renderer=createMatrixRenderer('test');
 assert.doesNotMatch(renderer.markup(w),/data-row-ruler|patch rows/);assert.match(renderer.markup(w),/1,024 input-feature rows/);
 assert.doesNotMatch(renderer.markup(nested),/data-row-ruler|patch rows/);
});
test('expanded activations receive rulers, compact matrices retain the data coordinate system',()=>{
 const renderer=createMatrixRenderer('test'),a=matrix(48,1024);renderer.setPatchOffset(0);
 assert.match(renderer.markup(a),/data-row-ruler/);assert.match(renderer.markup(a),/48 patch rows, 1,024 columns/);assert.doesNotMatch(renderer.markup(a,{compact:true}),/data-row-ruler/);
 renderer.setPatchOffset(1);assert.match(renderer.markup(matrix(37,768)),/37 token rows · 36 patches \+ CLS/);
 assert.match(renderer.markup(matrix(36,768),{rowOffset:0}),/36 patch rows/);
});
test('display-only guide generation never changes activation values or pooling',()=>{
 const a=matrix(48,1024,Float32Array.from({length:48*1024},(_,i)=>Math.sin(i))),before=a.data.slice(),pixels=overviewPixels(a);
 matrixRowGuides(a.rows,190,{selected:21});assert.deepEqual(a.data,before);assert.deepEqual(overviewPixels(a),pixels);
});
test('fast dot-product playback is 2.4 seconds regardless of feature width',()=>{
 assert.equal(MULTIPLICATION_PLAYBACK_MS,2400);
 for(const total of [64,768,1024,4096,5120]){
  assert.equal(multiplicationPlaybackCount(total,0,0),0);assert.equal(multiplicationPlaybackCount(total,0,1200),Math.floor(total/2));assert.equal(multiplicationPlaybackCount(total,0,2399)<total,true);assert.equal(multiplicationPlaybackCount(total,0,2400),total);assert.equal(multiplicationPlaybackCount(total,0,10000),total);
 }
});
test('elapsed-time playback is monotonic and resumes without dropping the accumulated prefix',()=>{
 const total=5120;let last=0;for(const elapsed of [0,8,32,109,451,775,1221,1900,2400]){const next=multiplicationPlaybackCount(total,0,elapsed);assert.ok(next>=last);last=next;}assert.equal(last,total);
 assert.equal(multiplicationPlaybackCount(1024,256,0),256);assert.equal(multiplicationPlaybackCount(1024,256,1200),768);assert.equal(multiplicationPlaybackCount(1024,256,1800),1024);assert.equal(multiplicationPlaybackCount(1024,256,-100),256);
});
test('fixed fast playback includes the complete dot product without dropping terms',()=>{
 const op={input:matrix(1,1024,Float32Array.from({length:1024},(_,i)=>Math.sin(i))),weights:weightDescriptor(1024,64,71),output:matrix(1,64)},trace=fullDotTrace(op,0,63);
 const halfway=trace.prefix[multiplicationPlaybackCount(1024,0,1200)],complete=trace.prefix[multiplicationPlaybackCount(1024,0,2400)];assert.equal(halfway,trace.prefix[512]);assert.equal(complete,trace.total);assert.equal(trace.prefix.length,1025);
});
test('served and bundled app keep full dimensions and fixed fast playback and row rulers',async()=>{
 const at=name=>new URL('../'+name,import.meta.url),html=await readFile(at('vision.html'),'utf8'),bundle=await readFile(at('standalone.html'),'utf8');
 assert.match(html,/src="src\/full-app.mjs"/);assert.match(bundle,/function matrixRowGuides/);assert.match(bundle,/MULTIPLICATION_PLAYBACK_MS=2400/);assert.doesNotMatch(bundle,/data-fm-speed|Multiplication animation speed|Fast · 2.4 s|Slow · 7.2 s/);assert.match(bundle,/width:1024,heads:16,headWidth:64,ffWidth:4096,depth:24,decoderWidth:5120/);assert.doesNotMatch(bundle,/function teachingScaleMarkup|24 features, so you can follow the math/);
});

test('the shared multiplication overlay has no speed selector or mutable playback duration',async()=>{
 const source=await readFile(new URL('../src/full-overlay.mjs',import.meta.url),'utf8');
 assert.doesNotMatch(source,/data-fm-speed|<select|duration=|playedMs,duration/);
 for(const control of ['data-fm-play','data-fm-step','data-fm-reset','data-fm-progress'])assert.ok(source.includes(control));
 assert.match(source,/multiplicationPlaybackCount\(op.input.cols,playbackBase,playedMs\)/);
});
