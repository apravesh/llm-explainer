/** Bundle a fully offline, double-clickable HTML page without third-party tools. */
import {readFile,writeFile} from 'node:fs/promises';
const at=p=>new URL(p,import.meta.url);
const svgData=async p=>'data:image/svg+xml;base64,'+Buffer.from(await readFile(at(p))).toString('base64');
const samples=Object.fromEntries(await Promise.all(['bird','coast','poster'].map(async name=>[name,await svgData(`assets/${name}.svg`)])));
let html=await readFile(at('index.html'),'utf8');
const css=await readFile(at('styles.css'),'utf8');
const motion=(await readFile(at('src/motion.mjs'),'utf8')).replace(/^export /gm,'');
const attentionFlow=(await readFile(at('src/attention-flow.mjs'),'utf8')).replace(/^import .*?;\n/gm,'').replace(/^export /gm,'');
const multiHead=(await readFile(at('src/multi-head.mjs'),'utf8')).replace(/^import .*?;\n/gm,'').replace(/^export /gm,'');
const readout=(await readFile(at('src/readout.mjs'),'utf8')).replace(/^import .*?;\n/gm,'').replace(/^export /gm,'');
const expansion=(await readFile(at('src/expansion.mjs'),'utf8')).replace(/^import .*?;\n/gm,'').replace(/^export /gm,'');
const engine=(await readFile(at('src/engine.mjs'),'utf8')).replace(/^export /gm,'');
let app=(await readFile(at('src/app.mjs'),'utf8')).replace(/^import .*?;\n/gm,'');
app=app.replace('loadImage(`assets/${state.sample}.svg`)','loadImage(sampleSources[state.sample])').replace("loadImage('assets/bird.svg')","loadImage(sampleSources.bird)");
html=html.replace('<link rel="stylesheet" href="styles.css">',()=>`<style>\n${css}\n</style>`).replace('href="assets/icon.svg"',`href="${await svgData('assets/icon.svg')}"`);
html=html.replace('<script type="module" src="src/app.mjs"></script>',()=>`<script type="module">\n${engine}\n${attentionFlow}\n${multiHead}\n${readout}\n${expansion}\n${motion}\nconst sampleSources=${JSON.stringify(samples)};\n${app}\n</script>`);
await writeFile(at('standalone.html'),html);
console.log('Built standalone.html: '+Buffer.byteLength(html).toLocaleString()+' bytes, no external dependencies.');
