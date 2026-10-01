import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT = 'C:/Users/anupk/Documents/git/A2UI/GenUICraft/presentations/e2b_qat_lora64_20260929';
const SKILL = 'C:/Users/anupk/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const PYTHON = 'C:/Users/anupk/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES = 'C:/Users/anupk/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { finalizePresentation, resolvePresentationFont } = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')));
const font = resolvePresentationFont({fontFamily:'Arial'});
const p = Presentation.create({slideSize:{width:1280,height:720}});
const s = p.slides.add();
s.background.fill = '#FFFFFF';
const C = {ink:'#142D4E', gray:'#506178', blue:'#175CD3', pale:'#F1F6FE', edge:'#C9D7EA', orange:'#A74609', warm:'#FFF4E6', green:'#117568', mint:'#EAF7F4', rule:'#DDE4EC'};
function text(name,str,x,y,w,h,size=20,color=C.ink,bold=false,align='left') {
  const q=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  q.text=str;
  q.text.style={typeface:font,fontSize:size,bold,color,alignment:align,verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0}};
  return q;
}
function rect(name,x,y,w,h,fill=C.pale,stroke=C.edge){return s.shapes.add({geometry:'rect',name,position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:1.2}});}
function line(name,x,y,w,h,color=C.rule){return s.shapes.add({geometry:'line',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width:1}});}
function arrow(a,b,color=C.blue,from='bottom',to='top',kind='straight'){return s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{fill:color,width:2},tail:{type:'triangle',width:'sm',length:'sm'}});}

text('title','Gemma 4 E2B: QAT LoRA-64 to on-device A2UI',48,34,1184,58,41,C.ink,true);
text('subtitle','Train the target model, preserve the mobile package, and optionally accelerate decoding with MTP.',48,98,1184,32,21,C.gray);
line('title-rule',48,139,1184,0);
line('column-divider-1',407,164,0,296);
line('column-divider-2',839,164,0,296);

text('architecture-heading','01  E2B ARCHITECTURE',48,160,332,30,21,C.blue,true);
text('architecture-effective','2.3B effective parameters • text path',48,194,332,27,18,C.gray);
const input=rect('response-input',72,232,270,40,'#FFFFFF');
text('response-label','Response / Markdown tokens',80,241,254,25,18,C.ink,false,'center');
const target=rect('decoder-target',58,297,298,112,C.pale);
text('decoder-layer-count','35 decoder layers',75,310,264,31,25,C.ink,true,'center');
text('decoder-width-ple','Width 1,536 • PLE 256 per layer',70,348,274,25,17.5,C.ink,false,'center');
text('decoder-attention','4 local (512) + 1 global, repeated',70,377,274,22,16.5,C.gray,false,'center');
arrow(input,target);
const logits=text('express-output','A2UI Express tokens',88,431,238,27,20,C.ink,true,'center');
arrow(target,logits);

text('training-heading','02  QAT LoRA-64 TRAINING',438,160,374,30,21,C.orange,true);
text('training-pairs','Response → Express supervision',438,194,374,27,19,C.ink);
const qat=rect('qat-training',438,232,370,99,C.warm,'#EAC9A8');
text('lora-formula','W′ = W + (α/r) BA',454,244,338,31,26,C.orange,true,'center');
text('lora-hparams','Rank 64 • alpha 64 • 96.6M trainable',452,286,342,27,18,C.orange,false,'center');
text('qat-forward','W2/W4 weight + A8 activation fake quant',438,349,374,29,18.5,C.ink,true);
text('qat-freeze','Fixed scales; frozen base; update A/B only',438,380,374,28,18,C.gray);
text('qat-targets','205 attention / MLP projections',438,419,374,25,20,C.ink,true);
text('qat-selection','Best checkpoint: step 9,000',438,447,374,24,17.5,C.gray);

text('export-heading','03  RETAINED-SCALE EXPORT',870,160,362,30,21,C.blue,true);
text('export-stage-1','1  Rebuild QAT effective weights',870,200,362,30,19,C.ink,true);
text('export-stage-1-detail','CPU FP32 LoRA matmul; BF16 delta before add',894,232,338,47,17.5,C.gray);
text('export-stage-2','2  Patch 205 W2/W4 weight buffers',870,286,362,30,19,C.ink,true);
text('export-stage-2-detail','Retain scales, graph, tokenizer and W8 paths',894,318,338,46,17.5,C.gray);
const pkg=rect('export-package',870,380,362,45,C.pale);
text('export-file','Target + original MTP drafter → .litertlm',882,392,338,27,17.5,C.blue,true,'center');
text('runtime-precision','SDK: GPU FP32 activations; weights stay quantized',870,437,362,41,17.5,C.gray);

line('mtp-divider',48,493,1184,0);
text('mtp-heading','MTP AT INFERENCE',48,517,295,31,23,C.green,true);
text('mtp-preserved','Official drafter preserved,\nnot fine-tuned with LoRA.',48,560,287,57,20,C.ink);
text('mtp-toggle','Runtime toggle: on / off',48,628,287,27,18,C.gray);
text('mtp-context','Shares target token embeddings, last hidden state and KV cache',357,516,875,26,18,C.gray);
const drafter=rect('mtp-drafter',357,558,235,69,C.mint,'#B7D8D1');
text('mtp-drafter-title','DRAFT',373,568,203,25,18,C.green,true);
text('mtp-drafter-details','~76M • 4 layers • width 256',373,596,203,24,16.5,C.ink);
const verifier=rect('mtp-verifier',644,558,245,69,C.pale);
text('mtp-verifier-title','VERIFY',660,568,213,25,18,C.blue,true);
text('mtp-verifier-details','One parallel target pass',660,596,213,24,16,C.ink);
const accepted=rect('mtp-accepted',941,558,291,69,C.mint,'#B7D8D1');
text('mtp-accepted-title','ACCEPT / CORRECT',957,568,259,25,18,C.green,true);
text('mtp-accepted-details','Keep valid prefix; correct rejection',957,596,259,24,16,C.ink);
arrow(drafter,verifier,C.green,'right','left');
arrow(verifier,accepted,C.green,'right','left');
text('mtp-speed-note','Sequential draft tokens • one target verification pass • speedup depends on acceptance',357,641,875,28,17.5,C.gray);
line('footer-rule',48,681,1184,0);
text('verification-boundary','Export code parity: 205/205 matrices verified. Native checkpoint-to-device parity is not yet certified.',48,689,1184,22,14.5,C.gray);

s.speakerNotes.textFrame.setText(`Purpose: one-slide explanation of the actual A2UI project training/export flow as of 29 September 2026. Editable native diagram, not an illustration of a new training recipe.

ARCHITECTURE SOURCES (checked 2026-09-29)
Google Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
Official E2B config: https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json
Google Gemma 4 technical report (Table 1, Section 2 and 2.6): https://arxiv.org/html/2607.02770v2
Official mobile QAT model: https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers
E2B has 2.3B effective parameters; complete model total is 5.1B including per-layer embeddings and vision/audio encoders. This slide depicts only our text training path, not a claim that all multimodal weights are trained or resident in memory. Text backbone: 35 decoder layers, width 1536, 256-dimensional per-layer token embeddings (PLE). Attention repeats four 512-token local layers and one global layer seven times. Do not claim K=V tying for E2B: its config has attention_k_eq_v=false. Target KV is shared across 20 of 35 layers.

ACTUAL TRAINING RUN
Local run receipt: C:/Users/anupk/Downloads/r64_lora_litert_metadata/best_golden_checkpoint.training_metadata.json
runD qat_lora_sft: rank=64, alpha=64, dropout=0; 96,632,832 trainable of 5,127,855,360 parameters including adapters, 410 A/B tensors. 205 LoRA target projections: q/o and gate/up/down in all 35 layers, k/v only in layers 0-14 due to shared KV. Base model reconstructed in BF16 from verified official mobile QAT seed. Training pairs are agent response and A2UI Express completion. Clipped straight-through estimator (STE) updates LoRA only. Best Golden32 checkpoint step 9000, epoch 3.161. Training used eight GPUs, effective batch 32.
Code: training/src/ir_training/train/sft.py (228-283, 607-625); training/src/ir_training/qat/fake_quant.py (883-930, 1299-1311).
QAT forward: fixed-scale A8 input fake quant; W_eff = BF16(W_base + BF16(FP32 LoRA delta)); retained-scale W2/W4 fake quant of this effective matrix; linear; A8 output fake quant. Base weights, embeddings and quantization scales remain frozen. Frozen W8 FC activation edges are also simulated, while W8 weights remain unchanged.

ACTUAL CORRECTED EXPORT
Receipt: GenUICraft/validation/20260928_r64_qat_reexport/export_audit/device_export_report.json
Audit: GenUICraft/validation/20260928_r64_qat_reexport/export_audit/REPORT.md
Exporter: training/scripts/build_gemma4_retained_scale_litertlm.py (1538-1620, 1720-1740).
Export mode retained_scale_qat_compatible_weights_v1, arithmetic qat_bf16_delta_before_add_v1. Reconstruct effective adapted weights with CPU FP32 LoRA matmul and BF16-rounded delta before addition; encode using original retained scales; patch exactly the 205 mapped W2/W4 buffers in the original mobile target section. This is not naive BF16 merge followed by generic re-quantization. Preserve 72 frozen buffers (including W8 paths), all seed weight/A8 activation qparams, graph layout, tokenizer, metadata and non-target sections. All 205 code/dequantized-forward parity checks pass against CPU QAT calculation; native_inference_parity_verified=false. These gates do not certify CUDA checkpoint versus native mobile behavior.
Runtime profile: GenUICraft/validation/20260928_gpu_fp32_app/REPORT.md and GenUiModelProfiles.trainedE2b in GenUICraft/genuicraft/src/main/java/com/samsung/genuicraft/sdk/GenUiSession.kt. The SDK selects GPU FP32 activation/arithmetic preference; exported weight codes remain mixed W2/W4/W8, not FP32 weights. The architecture maximum context is not the configured SDK context. The SDK uses 8192 context and 2048 output tokens.

MTP DETAILS
Google MTP overview: https://ai.google.dev/gemma/docs/mtp/overview
Google inference guide: https://ai.google.dev/gemma/docs/mtp/mtp
Assistant config: https://huggingface.co/google/gemma-4-E2B-it-assistant/blob/main/config.json
HF assistant documentation: https://huggingface.co/docs/transformers/model_doc/gemma4_assistant
The original separate E2B assistant is approximately 76M parameters, four layers and width 256 (three local, one global). It reuses target token embeddings, final hidden-state features and target KV cache, avoiding independent prefill. It proposes several tokens autoregressively; the target verifies them together, accepts a prefix, and continues at first rejection with a target-produced token. Correct speculative verification is designed to preserve the target distribution; no claim of measured native parity is made here. Actual speedup depends on draft acceptance, runtime and thermal state.
The project's training metadata does not jointly train the assistant. Export preserves the MTP graph byte-for-byte; mtp_assistant_trained_or_modified=false. Toggle is available in the app/SDK through speculative decoding configuration. No full Bixby50 MTP performance claim is made in this slide.
`);

const candidate=path.join(ROOT,'.build','candidate.pptx');
const final=path.join(ROOT,'output','Gemma4_E2B_QAT_LoRA64_MTP.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(path.join(ROOT,'.build','draft.png'),new Uint8Array(await (await p.export({slide:s,format:'png',scale:1.5})).arrayBuffer()));
const result=await finalizePresentation({
  explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  workspaceDir:ROOT,candidatePath:candidate,finalPath:final,pythonExecutable:PYTHON,
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
  receiptPath:path.join(ROOT,'.build','validation.json')
});
console.log(JSON.stringify(result));
