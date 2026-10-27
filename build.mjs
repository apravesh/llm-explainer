/** Bundle the full-dimension site and its identical worker for double-click offline use. */
import {readFile,writeFile} from 'node:fs/promises';
const at=p=>new URL(p,import.meta.url);
const svgData=async p=>'data:image/svg+xml;base64,'+Buffer.from(await readFile(at(p))).toString('base64');
const sources=Object.fromEntries(await Promise.all(['bird','coast','poster'].map(async name=>[name,await svgData(`assets/${name}.svg`)])));
const moduleSource=async name=>(await readFile(at('src/'+name+'.mjs'),'utf8')).replace(/^import .*?;\n/gm,'').replace(/^export /gm,'');
const mainFiles=['full-config','matrix-row-guides','full-matrix','full-engine','full-operations','full-overlay','gelu-overlay','head-stack','block-navigation','attention-stages','attention-tooltip','attention-rail-alignment','attention-window-motion','attention-window','simple-attention','automatic-value-mixing','full-views','patch-projection-links','motion','rope-lab','full-app'];
const workerFiles=['full-config','wasm-kernel','full-backend','full-engine','full-worker'];
const main=(await Promise.all(mainFiles.map(moduleSource))).join('\n'),worker=(await Promise.all(workerFiles.map(moduleSource))).join('\n');
let html=await readFile(at('vision.html'),'utf8'),css=await readFile(at('styles.css'),'utf8');
const safeJSON=value=>JSON.stringify(value).replace(/</g,'\\u003c');
html=html.replace('<link rel="stylesheet" href="styles.css">',()=>`<style>\n${css}\n</style>`).replace('href="assets/icon.svg"',`href="${await svgData('assets/icon.svg')}"`);
html=html.replace('<script type="module" src="src/full-app.mjs"></script>',()=>`<script type="application/json" id="full-sample-sources">${safeJSON(sources)}</script>\n<script type="application/json" id="full-worker-source">${safeJSON(worker)}</script>\n<script type="module">\n${main}\n</script>`);
await writeFile(at('standalone.html'),html);console.log(`Built standalone.html: ${Buffer.byteLength(html).toLocaleString()} bytes. Full dimensions, embedded worker + SIMD kernel, zero external dependencies.`);
