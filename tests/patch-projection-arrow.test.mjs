import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matrix,FULL_CONFIGS,projectionWeights} from '../src/full-config.mjs';
import {fullOperation} from '../src/full-operations.mjs';
import {patchProjectionPath,patchProjectionHitBand,patchProjectionLinksMarkup,projectionLinkKey} from '../src/patch-projection-links.mjs';
const read=name=>readFile(new URL('../'+name,import.meta.url),'utf8');
function fixture(model='pixtral'){
 const c=FULL_CONFIGS[model];return {config:c,layers:[],pixels:matrix(48,768),projectionWeights:projectionWeights(c,768),embeddings:matrix(48,c.width)};
}
test('the extra projection button is removed in both served and offline pages',async()=>{
 for(const file of ['vision.html','standalone.html']){
  const html=await read(file);assert.doesNotMatch(html,/patch-projection-arrow|--projection-gap/);
 }
 const bundle=await read('standalone.html');assert.match(bundle,/function patchProjectionLinksMarkup/);assert.match(bundle,/data-patch-projection/);
});
for(const model of ['pixtral','vit'])test(`${model}: edge opens the selected raw patch projection with unchanged tensors`,()=>{
 const run=fixture(model),s={selected:21,layer:0,head:0,expanded:null},before={...s},op=fullOperation(run,s,'embedding');
 assert.equal(op.row,21);assert.equal(op.patchOffset,0);assert.strictEqual(op.input,run.pixels);assert.strictEqual(op.weights,run.projectionWeights);assert.strictEqual(op.output,run.embeddings);assert.equal(op.input.cols,768);assert.equal(op.output.cols,run.config.width);assert.deepEqual(s,before);
 const expanded=fullOperation(run,{...s,expanded:2},'embedding');assert.deepEqual(op,expanded);
});
test('projection is unavailable before data is ready, but does not wait for encoder blocks',()=>{
 const run=fixture(),s={selected:2,layer:0};
 for(const key of ['pixels','projectionWeights','embeddings'])assert.equal(fullOperation({...run,[key]:null},s,'embedding'),null);
 assert.ok(fullOperation(run,s,'embedding'));assert.equal(run.layers.length,0);
});
test('CLS is never silently represented as patch zero by the projection arrow',()=>{
 assert.equal(fullOperation(fixture('vit'),{selected:-1,layer:0},'embedding'),null);assert.equal(fullOperation(fixture('vit'),{selected:0,layer:0},'embedding').row,0);
});
test('existing curves open their own patch projection and restore focus after redraw',async()=>{
 const app=await read('src/full-app.mjs'),overlay=await read('src/full-overlay.mjs'),css=await read('styles.css');
 assert.match(app,/projection=fullOperation\(run,state,'embedding'\)/);assert.match(app,/button.disabled=!projection/);
 assert.match(app,/if\(patch!==state.selected\)selectPatch\(patch\);overlay.open\('embedding',link\)/);
 assert.match(app,/openPatchProjection\(link\)/);assert.match(app,/projectionLinkKey\(links.map/);
 assert.match(overlay,/source\?\.id\?'#'\+source.id/);assert.match(overlay,/#full-patch-wires \[tabindex="0"\]/);
 assert.match(css,/#full-patch-wires:is\(:hover,:focus-within\) \.patch-projection-link \.patch-projection-line/);assert.match(css,/\.patch-projection-link:focus-visible \.patch-projection-line/);
 assert.match(css,/\.patch-projection-hit\{stroke:none;opacity:1;pointer-events:fill;fill:transparent/);
 assert.match(app,/wires.setAttribute\('role','group'\)/);assert.doesNotMatch(app,/wires.setAttribute\('aria-hidden','true'\)/);
});
const links=[{patch:19,x1:100,y1:20,x2:140,y2:80},{patch:20,x1:100,y1:50,x2:140,y2:85},{patch:21,x1:100,y1:80,x2:140,y2:90}];
const options={selected:20,inputWidth:768,outputWidth:1024,height:320};
test('the visible connections retain their original curves with invisible, clipped hit areas',()=>{
 assert.equal(patchProjectionPath(100,20,140,80),'M100,20 C120,20 120,80 140,80');
 const html=patchProjectionLinksMarkup(links,options);assert.equal((html.match(/class="patch-projection-line"/g)||[]).length,3);assert.equal((html.match(/class="patch-projection-hit"/g)||[]).length,3);
 for(const link of links){const path=patchProjectionPath(link.x1,link.y1,link.x2,link.y2);assert.equal(html.split(`class="patch-projection-line" d="${path}"`).length-1,1);}
 assert.doesNotMatch(html,/<button|marker-end|×W/);assert.match(html,/<rect x="100" y="0" width="40" height="320"/);
});
test('each connection has patch-specific semantics and a stable identity for focus restoration',()=>{
 const html=patchProjectionLinksMarkup(links,options);for(const {patch} of links){assert.ok(html.includes(`id="patch-projection-link-${patch}"`));assert.ok(html.includes(`data-patch-projection="${patch}"`));assert.ok(html.includes(`Animate patch p${patch} projection to embedding`));}
 assert.equal((html.match(/role="button"/g)||[]).length,3);assert.equal((html.match(/aria-haspopup="dialog"/g)||[]).length,3);assert.equal((html.match(/aria-controls="full-math-dialog"/g)||[]).length,3);assert.equal((html.match(/tabindex="0"/g)||[]).length,1);
 assert.match(html,/data-patch-projection="20" role="button" tabindex="0"/);
});
test('scrolling to off-selected patches still provides one keyboard entry point and no invented CLS link',()=>{
 for(const selected of [-1,99]){const html=patchProjectionLinksMarkup(links,{...options,selected});assert.match(html,/data-patch-projection="19" role="button" tabindex="0"/);assert.equal((html.match(/tabindex="0"/g)||[]).length,1);assert.doesNotMatch(html,/data-patch-projection="-1"/);}
 assert.equal(patchProjectionLinksMarkup([],options),'');
});
test('keyboard browsing uses the visible patch sequence and does not wrap beyond its boundaries',()=>{
 const patches=[19,20,21];assert.equal(projectionLinkKey(patches,19,'ArrowDown'),20);assert.equal(projectionLinkKey(patches,21,'ArrowDown'),21);assert.equal(projectionLinkKey(patches,19,'ArrowUp'),19);assert.equal(projectionLinkKey(patches,21,'Home'),19);assert.equal(projectionLinkKey(patches,19,'End'),21);assert.equal(projectionLinkKey(patches,19,'Enter'),null);assert.equal(projectionLinkKey([],19,'Home'),null);
});

test('hit bands cannot overlap even where steep curves become tightly packed',()=>{
 const crowded=Array.from({length:9},(_,i)=>({patch:13+i,x1:260.6875,y1:79.5+i*23,x2:295.6875,y2:126.5+i*4}));
 const bands=crowded.map((_,i)=>patchProjectionHitBand(crowded,i));
 for(let i=0;i<bands.length;i++)for(let j=0;j<33;j++){
  const p=bands[i].points[j];assert.ok(p.top<p.y&&p.bottom>p.y);assert.ok(p.y-p.top<=6.000001&&p.bottom-p.y<=6.000001);
  if(i)assert.ok(bands[i-1].points[j].bottom<=p.top+1e-10);
 }
 const p20=bands[7].points[16];assert.equal(p20.x,278.1875);assert.equal(p20.y,197.5);assert.ok(p20.y<bands[8].points[16].top);
});

test('hovering or focusing any connection highlights the whole group rather than one patch',async()=>{
 for(const file of ['styles.css','standalone.html']){
  const source=await read(file);
  assert.match(source,/#full-patch-wires:is\(:hover,:focus-within\) \.patch-projection-link \.patch-projection-line\{stroke:#7195cf;stroke-width:1.5;opacity:.85\}/);
  assert.doesNotMatch(source,/\.patch-projection-link(?:\.is-selected)?:hover/);
  assert.match(source,/\.patch-projection-line\{pointer-events:none;stroke:#bccce3;stroke-width:.7;opacity:.25/);
  assert.match(source,/\.patch-projection-link:focus-visible \.patch-projection-line/);
 }
});
