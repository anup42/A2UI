import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const out='C:/Users/anupk/Documents/git/A2UI/GenUICraft/presentations/e2b_qat_lora64_20260929/output';
const p=await PresentationFile.importPptx(await FileBlob.load(out+'/Gemma4_E2B_QAT_LoRA64_MTP.pptx'));
const slide=p.slides.items[0];
if(p.slides.items.length!==1) throw new Error('Expected exactly one slide');
const blob=await p.export({slide,format:'png',scale:1.5});
await fs.writeFile(out+'/Gemma4_E2B_QAT_LoRA64_MTP.png',new Uint8Array(await blob.arrayBuffer()));
console.log('Rendered exact finalized PPTX: 1 slide');
