# GenUI training review — 5 September 2026

The authorized code fixes and CPU validation are recorded in [IMPLEMENTED_FIXES.md](IMPLEMENTED_FIXES.md). Use [the GPU-PC handoff](../../GPU_PC_TRAINING_HANDOFF.md) for the next training run. The findings and checkpoint numbers below describe the original supplied evidence.

The E2B model is learning A2UI Express, but broken evaluation context and termination obscure its progress. Fix measurement, stopping, and graph-preserving data preparation before another long training run. Train 270M as a separate capacity experiment on the same validated contract; this package contains no verified 270M Golden 32 result.

Five archives were extracted separately. Six existing training files were merged with their genuine incoming changes, and six new source files were imported. Newer checkout fixes and pre-existing Android/dataset work were preserved. The full training suite passed: **339 tests**. This review did not launch training, contact the training host, regenerate model predictions, export models, or run Android inference. The recommendations below remain implementation work unless specifically identified as merged.

Evidence location: [extracted packages and analysis](C:/Users/anupk/Downloads/review/genUI-review/extracted_20260905_172922). The canonical package is `genui_review_COMPLETE_zip`; the earlier partial archive supplies three additional 270M files absent from COMPLETE. Weights and the full training corpus were excluded by the sender.

**Golden 32: what the outputs actually show**

All 128 original model predictions, all 128 supplied corrected predictions, and all 32 reference targets were inspected. The archived evaluator reads the first user message from a few-shot conversation: the travel-checklist demonstration. It therefore scores unrelated generated UIs against that demonstration rather than the actual Stage-2 response. All four checkpoints have this error. The current checkout already corrects source extraction; the merge preserved that fix.

The following scores were recomputed offline against each row's actual response and evaluation context. No weights or new model generation were needed. “Stop only” takes the archived completion through its first closing `</a2ui>` tag. “ID repair” additionally applies the supplied heuristic and is a diagnostic, not model accuracy or a production result.

| Checkpoint | Published raw score, wrong source | Re-scored raw | Re-scored stop only | Strict parses, raw → stop | Stop + ID repair, diagnostic |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 16.25 | 17.53 | 20.05 | 8 → 9 / 32 | 60.25; 32/32 parse |
| 10,000 | 22.83 | 26.29 | 29.94 | 12 → 14 / 32 | 61.44; 27/32 parse |
| 15,000 | 27.18 | 31.58 | 40.02 | 14 → 18 / 32 | 66.43; 30/32 parse |
| 20,000 | 7.62 | 8.81 | **59.80** | 4 → **27 / 32** | 65.42; 30/32 parse |

Scores use the supplied v5.4 implementation supplemented with the checkout's required schema, fixtures and renderer dependencies, which the archive omitted. Its fingerprint differs from the original host fingerprint. As a control, the review runtime reproduced **all 256 archived wrong-source per-row scores exactly, with maximum difference 0.0**. This establishes source/context binding as the cause of the observed re-score differences on these artifacts; it does not prove every scorer dependency is globally identical. The runtime dependency hashes and metric fingerprints are recorded in the evidence.

The source-corrected stop-only score increases by 39.75 points from 5k to 20k. A blanket conclusion that SFT stopped improving is unsupported. Conversely, step 20k is not ready for release: five outputs still fail after closing-tag stopping—three duplicate-ID cases, one unresolved root child, and one unclosed repetition loop. Parse success also does not establish that every intended fact or interaction is visible and correct.

Checkpoint ranking depends on the chosen decoder. Step 20k is strongest under stop-only evaluation; step 15k is slightly higher under heuristic repair. Do not pick a release using the latter ranking or compare runs with different stopping/repair settings as if they were the same metric. Re-evaluate accessible retained checkpoints, including 22k, with actual weights and one fixed deployment decoder before choosing a seed. Early 5k/10k/15k prediction files exist, but their ordinary checkpoint weight directories were not included.

Three other supplied conclusions need correction:

- **The score cap of 40 is declared-root reachability below 90%, not a general component-count cap.** For repaired 5k predictions, 15/32 hit this cap. Text can survive in detached definitions without being displayed. Do not relax this cap to reward disconnected content.
- **The reported approximately 13% content coverage is measured against the wrong response.** Correct-source coverage among repaired parseable rows is approximately 90–94%. However, this legacy metric counts words across the serialized graph, including detached content. It is not proof of visible fact coverage. Report both validity and all-row visible-fact recall, with failures contributing zero.
- **The reference targets need QA.** They parse 32/32 but score 88.56 on average; one has a production-invalid `Table.highlightColumns` string where the wire schema requires an array. Zero generated exact/semantic matches is a useful diagnostic, but exact equality to one imperfect layout should not be the primary task objective.

Golden 32 has one case per intent and comes from validation. It has already influenced checkpoint selection, so it is a development/regression set. No overlap was found in the shipped samples, but the full 169,898-row corpus is absent; full leakage exclusion cannot be certified. Re-check normalized response content, source groups, and template families across train/dev/final test. Add an untouched 500–1,000-case acceptance set and a separate stress suite for complex references, long responses, exact numbers, state and actions.

**How the captured E2B training works**

The principal run uses Stage-2 response → A2UI Express completion, with saved chat messages and completion-only labels. It starts from the QAT-derived unquantized E2B checkpoint, uses BF16, q/v-only LoRA rank 16/alpha 16, two DDP workers, microbatch 1 and accumulation 4: effective batch **8**. It trains 2,678,784 parameters, approximately 0.0525% of the loaded model. Learning rate is 5e-5 with 3% warmup; training sequence and generation limits are 4,096 tokens. Data comprises 169,898 train rows and the reduced 1,024-row validation set.

| Step | Train loss | Validation loss |
|---:|---:|---:|
| 500 | 0.19922 | 0.17710 |
| 5,000 | 0.05678 | 0.05332 |
| 10,000 | 0.04590 | 0.04648 |
| 15,000 | 0.04249 | 0.04291 |
| 20,000 | 0.04053 | 0.04107 |
| 22,000 | 0.03869 | 0.04017 |

Train and validation loss both improve. The trajectory does not establish classic overfitting, although loss alone cannot measure autonomous structured generation or rule out template leakage. Forty train rows exceed the sequence limit; this small tail needs handling, but it cannot explain the widespread generation failures.

The package's directory names overstate run status. The main SFT log ends with SIGTERM at 22,084/42,476 planned updates; checkpoint 22,000 is at epoch 1.03588 of two. GRPO's full-run log records two zero-gradient updates and then a DDP/NCCL failure. The bottom-up run includes preparation/preflight evidence but no optimizer trajectory, checkpoint or Golden score. These are captured historical states, not live host status.

**Fixes required before another long run**

| Priority | Verified problem | Concrete correction and acceptance check |
|---|---|---|
| P1 | `_align_tokenizer_and_model` replaces the generation EOS list with scalar tokenizer EOS. A reproduction changes `[1,106,50]` to `1`. | Preserve valid model end-of-turn IDs when aligning PAD/BOS/vocabulary; resolve through the actual tokenizer. Log effective post-Trainer IDs and prove native end-of-turn stopping. Mirror envelope stopping across HF, GRPO, export and Android. |
| P1 | Archived evaluator scores the few-shot user. | Preserve the current extractor; bind predictions to immutable source rows with IDs, assets and expected contracts. Assert evaluated response equals the final task input. The current source fix is already retained; historical metrics need replacement. |
| P1 | Bottom-up and duplicate-ID helpers use incomplete/heuristic reference rewriting. | Replace regex manipulation with canonical/catalog-aware graph traversal and ID rewriting. Require strict validation plus unchanged canonical semantics. Preserve Tabs, Modal, repeat/template references, shared children, state and literal text. |
| P1 | GRPO full-run reward -1/std 0 gives zero gradients, followed by different collective orders across ranks. | Fix stopping/seed validity first. Run a short single-device learning smoke, then two-device smoke with varied lengths. Verify nonzero reward variance, finite adapter changes and identical collective order before scaling. Raising timeouts does not fix ALLGATHER versus BROADCAST divergence. |
| P1 | Generic W8A32 QAT, merge metadata, and declared Q4_0/mobile WNA8O8 export do not describe one verified numeric contract. | Preserve training provenance. First prove adapter+base versus merged output parity, then choose the exact export target and train/convert/evaluate against its quantization rules. |
| P2 | `training.eval_strategy` is ignored; the current code always selects step evaluation when validation exists. | Honor and validate the strategy together with Golden callback scheduling. Avoid accidentally disabling Golden checks when correcting the key. |
| P2 | New resume validation compares regex strings as sets of characters; PEFT default resolution can also differ from saved targets. | Compare regex strings exactly, lists as normalized sets, and defaults by resolved matched modules. Validate dataset, optimizer, scheduler, parameter scope and batch identity on resume. |
| P2 | 270M resume launcher selects three GPUs but YAML selects four, potentially reintroducing excluded GPU 2. | Parameterize host paths; use one authoritative GPU list and world size. Actual three-rank effective batch is 12, not the comment's 16. Do not launch the imported recipe unchanged. |
| P2 | A `full_finetune_qat` 270M YAML exists, but QAT validation rejects it and the SFT trainer attaches PEFT. | Implement and verify a real full-SFT route before claiming full-parameter 270M training. Existing supported LoRA paths remain a baseline. |

The first EOS issue is more specific than the supplied claim that greedy decoding alone is the problem. A base `generation_config.json` does not prove the in-memory model still has those EOS values. Sampling may be an experiment, but is not a substitute for fixing termination. Transformers supports both multiple EOS IDs and stop strings; record the actual effective policy. [Transformers generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation).

The bottom-up idea itself remains worthwhile. Android resolves the reserved root after parsing all assignments, so root-last ordering is compatible. The supplied converter is not safe for the full grammar: controlled examples change a repeat template from Text B to Text C, or alter a quoted `children=[c]` literal, while remaining valid programs. The build also drops 5,673 training rows, including supported complex syntax such as Tabs. In the shipped sample, all 300 original targets validate, but 18 are rejected by the converter. We did not observe semantic corruption in the supplied 300 already-converted outputs; the counterexamples prove the implementation is incomplete, not that every produced row is damaged.

Do not call duplicate-ID repair lossless based on its content fingerprint. Its nearest-definition attachment heuristic cannot recover intent uniquely. Keep repaired results separate and use them to diagnose trainable structure, not to certify the raw model.

**Training plan for both models**

The values below are proposed pilot settings, not validated optimums. Change one factor at a time, retain the same source groups and token budget, and select on generation quality rather than loss alone.

| Phase | E2B | 270M |
|---|---|---|
| Establish trustworthy baseline | Fix EOS/context/decoder parity; regenerate the retained checkpoints under one policy. Start from the best re-evaluated seed. | Obtain the referenced 270M run's metadata and raw predictions, or establish a fresh baseline. No score is available in this package. |
| Shared clean data | Strict-validate and render-check targets; source/template-family disjoint splits; preserve exact facts, tables, URLs, actions and state. | Use the same contract and a compact, accurate prompt. Distill validated teacher sequences instead of inheriting unverified E2B predictions. |
| Small capability pilot | Start with 10k–20k diverse source groups. Hold q/v rank16 as control; compare language-only attention+MLP rank16, then rank32 if useful. Test LR 2e-5 versus 5e-5 separately. | Begin with 2k–5k selected short examples, then 10k–20k diverse ones. Compare supported all-projection LoRA with a properly implemented BF16 full-SFT route. Pilot full-SFT LR 1e-5/2e-5/5e-5; tune LoRA independently. |
| Representation A/B | Root-first versus corrected root-last on identical source rows and matching few-shot scaffolds. Keep filtering identical. | Same A/B; include tables, Tabs, Modal, repeats and shared references so the small model does not learn a reduced grammar accidentally. |
| Scale | Use the winning scope/serializer on the cleaned corpus; inspect paired dev gains at intermediate checkpoints before adding epochs. | Curriculum from short cards to composed tables/actions/state. Retain simple-example replay while adding complexity; measure per-capability scores. |
| Quantization | Choose actual Q4_0, public W8, or exact mobile numeric recipe. Compare floating, fake-quant and exported outputs. | Establish BF16 capability, compare W8 PTQ with a short lower-LR W8 QAT continuation. Keep W4 an explicit sensitivity experiment. |
| Optional optimization | GRPO only after seed validity, grounded reward and single-/multi-device learning smokes pass. MTP only after target quality/export parity. | Prefer SFT/distillation first. GRPO is optional; it cannot compensate for insufficient learned grammar or corrupted supervision. |

For E2B, inspect actual matched module names before widening LoRA. The multimodal model can contain branches unused by this text task, so a language-scoped attention/MLP selection is preferable to blindly targeting every linear layer. PEFT distinguishes regex strings from name lists and can resolve architecture defaults; save the resulting module inventory. [PEFT LoRA documentation](https://huggingface.co/docs/peft/package_reference/lora).

For 270M, full tuning is a serious experiment because this is a narrow structured generation task, but the repo's current full-QAT YAML is not an executable implementation. Google's 270M example establishes that task-specific fine-tuning and on-device conversion are feasible; it does not establish this model's A2UI performance. [Official 270M fine-tuning example](https://developers.googleblog.com/own-your-ai-fine-tune-gemma-3-270m-for-on-device/).

Prompt parity must be exact enough to test: system content, demonstrations, response delimiter, URL handling, tokenizer chat template, BOS/EOS IDs and output limits. Prepared messages override later prompt-file edits. The bottom-up builder changes assistant demonstrations while Android remains root-first; the 270M compact system prompt also differs from Android's current one. Version the prompt with each model and compare token IDs, rather than only comparing a few leading tokens.

Suggested promotion targets: 32/32 strict-valid Golden outputs under the fixed deployment decoder, no required-fact/action regressions, then at least 99% strict validity on the larger untouched set. These are proposed engineering gates, not achieved results. Report exact numerical values/units, visible fact recall, table row/column completeness, working actions/state, root reachability, token cost, latency and failure modes. Use macro/per-capability summaries and paired comparisons; one Golden case is 3.125 percentage points.

Validate the complete chain independently for **both** models: selected HF checkpoint → merged checkpoint → floating export → quantized LiteRT-LM → Android CPU/GPU. Bind model/tokenizer/prompt/scorer hashes; retain raw outputs, stop reasons, token counts, real backend/fallback and memory/latency evidence. Model context capability is not the same as the configured training sequence or exported KV-cache size. Official Gemma 4 Q4_0 and mobile mixed-precision variants are distinct; an official MTP assistant's availability does not verify a custom target/assistant pair. [Gemma quantization variants](https://ai.google.dev/gemma/docs/core), [MTP guide](https://ai.google.dev/gemma/docs/mtp/mtp).

**Deliverables and verification**

- [Merge report](C:/Users/anupk/Documents/git/A2UI/training/docs/reviews/20260905_genui_training_review/MERGE_REPORT.md): exact included/excluded source and preservation checks.
- [Golden 32 audit](C:/Users/anupk/Downloads/review/genUI-review/extracted_20260905_172922/review_analysis/golden32): per-row re-scores, controls, fingerprints and reproducible scripts.
- [Training audit](C:/Users/anupk/Downloads/review/genUI-review/extracted_20260905_172922/review_analysis/training_audit.md): training trace, log/source locations, EOS reproduction and E2B experiments.
- [Representation and 270M audit](C:/Users/anupk/Downloads/review/genUI-review/extracted_20260905_172922/review_analysis/representation_270m.md): parser/renderer evidence, counterexamples, prompt and 270M limitations.
- Local validation: 339 training tests passed; imported Python sources parse; scoped whitespace check passed; the pre-existing tracked diff outside training is byte-identical to its pre-merge snapshot. Android code was untouched by this integration. CUDA/DDP, model-weight, export and device validation remain necessary for the proposed fixes and future models.

The recommendation is to continue SFT after repairing the measurement and data/runtime contracts. The existing E2B evidence supports progress, not abandonment of SFT. Start 270M with its own clean capability baseline and measured capacity comparison; do not reuse E2B's scores, hyperparameters or export assumptions as proof for the smaller model.
