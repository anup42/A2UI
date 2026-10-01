import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const build = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const presentation = Presentation.create({slideSize:{width:1280,height:720}});
const slide = presentation.slides.add();
slide.background.fill='#FFFFFF';
const ink='#182333', blue='#235BDC', muted='#566476';
function text(name,value,x,y,w,h,size,color=ink,bold=false) {
  const shape=slide.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;
  shape.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none',wrap:'none',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0}};
  return shape;
}

text('Title','GenUICraft\ninside Bixby',54,55,510,128,48,ink,true);
text('Route','Bengaluru → Varanasi',56,226,510,43,29,blue,true);
text('Query','Morning flights for tomorrow',56,275,510,38,25,muted);

text('Generation heading','On-device UI generation',56,361,510,40,28,ink,true);
text('Pipeline','Bixby answer → A2UI Express\n→ recovery → native flight cards',56,410,520,82,25,muted);

text('Benefit','Flight times and fares are easier\nto scan in the same conversation.',56,535,520,76,25,ink);
text('Playback','Select the video to play',56,656,510,30,20,blue);

slide.images.add({blob:new Uint8Array(await fs.readFile(path.join(build,'video-poster.png'))),contentType:'image/png',alt:'DemoVideoPoster',fit:'contain',position:{left:612,top:35,width:650,height:650}});
slide.speakerNotes.textFrame.setText([
  'Demo query: show morning flights from bengaluru to varanasi for tomorrow.',
  'Recorded in the installed Bixby app on Samsung Flip8 SM-F776U on 30 September 2026.',
  'The embedded 48.8-second video is an edited capture of one live request. Streaming text is accelerated; the circular progress indicator retains its recorded normal speed. The timer digits refer to source recording time, not inference-only latency. No speed labels or SOURCE TIME label appear in the final video.',
  'The demo shows the original Bixby answer, A2UI Express streaming, SDK recovery and native flight cards in the same conversation. Runtime evidence records on-device Gemma4 E2B, GPU FP32, MTP enabled, and one successful conversion using GENERATED_DSL_REPAIR.',
  'Displayed schedules and fares are the Bixby response and are not independently verified booking availability.',
  'Source video: C:/Users/anupk/Documents/git/A2UI/GenUICraft/demos/20260930_bixby_flight_live/Bixby_GenUICraft_Varanasi_Demo_Final.mp4',
  'Runtime evidence: C:/Users/anupk/Documents/git/A2UI/GenUICraft/demos/20260930_bixby_flight_live/runtime_evidence.txt'
].join('\n\n'));

await (await PresentationFile.exportPptx(presentation)).save(path.join(build,'draft.pptx'));
await fs.writeFile(path.join(build,'draft.png'),new Uint8Array(await (await presentation.export({slide,format:'png',scale:1.5})).arrayBuffer()));
await fs.writeFile(path.join(build,'slide-1.layout.json'),await (await slide.export({format:'layout'})).text());
console.log(JSON.stringify({draft:path.join(build,'draft.pptx'),preview:path.join(build,'draft.png')}));
