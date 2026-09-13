# Training pipeline inventory

This is the source-grounded inventory of training paths in this checkout as of
2026-09-12. `training/` is the current A2UI response-to-IR system. It consumes
completed dataset Stage-3 records and targets strict A2UI Express v1; it does
not generate Stage 1/2/3 data. `training_scripts/` and `dpo/` are separate
historical/experimental trees described below.

## Recommended end-to-end pipelines

For clone-and-run **HF capability training and checkpoint testing on both
Golden32 and Golden35**, use `training/scripts/run_golden_training.py` and the
[shared-prompt quickstart](GOLDEN_E2E_QUICKSTART.md). It supports dense E2B
LoRA, 270M full SFT, and optional 270M full W8 QAT from an SFT checkpoint.
It prepares the tracked Stage 3 source with one production scaffold, filters
all reserved Golden sources, chooses visible GPUs, and tests selected-best
and actual-final checkpoints on both cohorts. Golden32 selects checkpoints;
Golden35 is final-only. This path does not export LiteRT packages or train MTP.

Added 2026-09-14: `training/scripts/run_golden_experiments.py` runs bounded,
sequential LR/weight-decay/warmup and optional rare-component-resampling
comparisons using that capability workflow. Baseline is included; all trials
start from the same model, seed and step budget, with TensorBoard HParams and
comparison records. Golden32 selects the winner; only the locked winner is
tested on Golden35. See [Augmentation and tuning](AUGMENTATION_AND_TUNING.md).
This is opt-in screening, not a new official-QAT/export/MTP pipeline.

The separate deployment/quantization workflows below use the original
September 3 Golden32, not the archive-repeat development cohort:

| Model/purpose | Entry point | Training and selection | Deployment/evaluation |
|---|---|---|---|
| Gemma 4 E2B A2UI Express | `training/scripts/run_gemma4_e2b_a2ui_express_multiformat.py` | Reconstructed mobile BF16 seed, retained-scale effective-weight LoRA QAT, periodic exact Golden-32, best adapter | One official-format retained W2/W4/W8+A8 code-only package plus public W32, experimental W16, W8, and mixed W4/W8 packages; actual checkpoint and every package scored on the same Golden-32 with TensorBoard and a hash-bound scorecard |
| Gemma 3 270M A2UI Express | `training/scripts/run_gemma270m_a2ui_express_multiformat.py` | W8/AFP32 effective-weight LoRA QAT, periodic exact Golden-32, best adapter | Public W32, experimental W16, QAT-aligned W8, and experimental block-32 W4; optional official-Q8 topology for W8 only; actual checkpoint and all packages receive the same evidence-bound evaluation |

Both are plan-only without execution flags, isolate mutable outputs under a
unique run ID, use `A2UI_TENSORBOARD_ROOT=/tensorboard` on MLP, and refuse to
invent or accept manually entered scores. See their dedicated runbooks:

- `training/docs/gemma4_e2b_a2ui_express_multiformat_runbook.md`
- `training/docs/gemma3_270m_a2ui_express_multiformat_runbook.md`
- `training/docs/golden32_evaluation.md`

## Shared current trainers

### Adapter SFT, QLoRA, and LoRA QAT

`training/scripts/train_sft.py` is the shared current trainer. The selected
YAML controls ordinary LoRA/QLoRA, QAT-derived LoRA, or true fake-QAT LoRA. It
uses model adapters under `training/src/ir_training/models/`; Gemma, Qwen, and
Llama adapters exist, although no Llama training YAML is checked in.

Representative ordinary SFT configs:

- `training/configs/models/gemma_e2b_ir_lora.yaml`
- `training/configs/models/gemma4_e2b_ir_lora.yaml`
- `training/configs/models/gemma4_ir_lora.yaml`
- `training/configs/models/qwen_ir_lora.example.yaml`
- `training/configs/sft_a2ui_express_v1.yaml`

QAT-derived but not continued-QAT Gemma 4 profiles are the
`gemma4_e2b_ir_qat_lora*.yaml` and
`gemma4_e2b_a2ui_express_lora_remote_*.yaml` families plus
`training/configs/space_h100_train_20260807.yaml`. They fine-tune a public
unquantized QAT-derived base with LoRA; they must not be described as true QAT.

True LoRA-QAT profiles include:

- `training/configs/models/gemma4_e2b_a2ui_express_official_qat.yaml` — the
  Golden-32 multiformat E2B profile.
- `training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml` — retained
  mobile-scale E2B profile used by the lower-level mobile pipeline.
- `training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml`
  — serialization/wiring smoke only, never a quality result.
- `training/configs/models/gemma4_e2b_ir_qat_sft.yaml` — dense Q4-derived
  experiment; not approved for retained-mobile transplant.
- `training/configs/models/gemma3_270m_a2ui_express_qat.yaml` — the Golden-32
  multiformat 270M profile.
- `training/configs/models/gemma3_270m_ir_qat_sft.yaml` — earlier deployable
  270M W8/AFP32 profile.
- `training/configs/models/functiongemma_270m_ir_qat_sft.yaml` — separate
  FunctionGemma W8/A8 experiment.
- The `gemma3_270m_a2ui_express_*_space.yaml` and `*_groupvolume.yaml` configs —
  host/path-specific remote profiles, not portable defaults.

### Full-parameter Gemma 3 270M QAT

`training/scripts/train_full_finetune_qat.py` is a separate CUDA multi-GPU
trainer driven by
`training/configs/models/gemma3_270m_a2ui_express_full_qat_space.yaml`. It
trains all parameters rather than a PEFT adapter. It is not wired into the
adapter merge, official-Q8 transplant, or multiformat promotion flow.

### Optional Gemma 4 MTP drafter

`training/scripts/train_gemma4_mtp_drafter.py` and
`training/configs/models/gemma4_e2b_mtp_drafter_qat.yaml` train only the 23
deployable assistant matrices against a frozen target using teacher-forced
completion cross-entropy and W4/W8-A8 fake QAT. The E2B multiformat pipeline
preserves and enables the released MTP section by default; only training a new
drafter is disabled by default. When enabled, the trainable stage is run-scoped
and logged under the shared TensorBoard root. It is a public reconstruction, not Google's
private data mixture, distillation loss, optimizer, or observer schedule.

### GRPO

`training/scripts/train_grpo.py` is an optional TRL GRPO+LoRA stage after a
strict A2UI Express SFT checkpoint. It uses the deterministic A2UI quality
reward. `training/configs/grpo_a2ui_express_v1.yaml` is reference
configuration rather than a generic `--config` contract.

There is no implemented A2UI knowledge-distillation pipeline.

## Lower-level deployment orchestrators

- `training/scripts/run_gemma4_e2b_mobile_mtp.py` implements E2B best-adapter
  merge, retained-scale 205-projection code-only export, optional official or
  trained MTP handling, and device gates. It is the official-format engine used
  by the new E2B multiformat wrapper.
- `training/scripts/run_gemma270m_qat_litertlm.py` is the older W8-only Gemma 3
  270M path: W8/AFP32 LoRA QAT, best adapter, merge, either public INT8 or exact
  official-Q8 topology export, and Android GPU parity.
- `training/scripts/run_gemma4_mobile_qat.py` is the required portable,
  fresh-run E2B target launcher with seed, qparams, architecture, numeric,
  generation, leakage, and provenance gates.

## Evaluation and export components

- `training/scripts/prepare_dataset.py` builds strict training/evaluation pairs.
- `training/scripts/evaluate.py` generates/scores ordinary checkpoint output.
- `training/scripts/evaluate_checkpoint_on_golden.py` evaluates an adapter or
  merged checkpoint and writes TensorBoard evidence.
- `training/scripts/evaluate_litertlm_on_golden.py` uses the versioned external
  LiteRT-LM JSONL runner, then scores and logs the tested package.
- `training/scripts/benchmark_qat_mtp.py` compares target-only and MTP
  Transformers behavior; it is plan-only unless executed.
- `training/scripts/export_edge_gallery_model.py` invokes the public LiteRT
  Torch exporter. It is valid for the public comparison lanes, not as a
  substitute for retained-scale official-format E2B export.
- `training/scripts/build_gemma4_retained_scale_litertlm.py` performs the
  dedicated E2B retained-scale code transplant.
- `training/scripts/build_checkpoint_official_topology.py` performs the
  generic exact-topology checkpoint transplant used for Gemma 3 270M W8.
- `training/scripts/build_gemma4_mtp_drafter_official_topology.py` injects a
  trained assistant into the official MTP graph under strict package gates.
- `training/scripts/merge_qat_lora.py` merges a PEFT adapter into a floating HF
  checkpoint; merge itself is not quantization.
- `training/scripts/convert_qat_q4_0.py` is a separate llama.cpp GGUF Q4_0 path,
  not a LiteRT-LM export.
- `training/scripts/export_model.py` is a legacy metadata/copy handoff and does
  not itself create `.litertlm` bytes.

## Separate and legacy trees

`dpo/train_dpo.py` and `dpo/src/train.py` are generic HH-RLHF/Qwen preference
experiments. They do not consume the A2UI Express pair schema, use A2UI reward,
or participate in the recommended pipelines.

`training_scripts/` contains older BF16/LoRA and QAT implementations, including
PyTorch LoRA-QAT, TorchAO 8da4w, and Unsloth variants. It targets older data and
artifact contracts; several launchers hard-code another machine's paths, all
QAT trainers import a missing `dataloader.py`, and some scripts reference
missing converters/evaluators. Treat it as historical source, not a runnable
or supported pipeline.

Several shell/HPC wrappers under `training/scripts/` are also site-specific:
`launch_full_chain.sh`, `prepare_space_mobile_chain.sh`,
`launch_gemma3_270m_a2ui_express_compact_groupvolume.sh`, and older bootstrap
scripts. Inspect their absolute/group-volume assumptions before reuse; they are
not substitutes for the portable Python orchestrators above.

## Evidence boundary

Checked-in tests and plan/static validators establish configuration,
provenance, numeric-contract, and orchestration behavior. They do not establish
that a real GPU job, converter, LiteRT-LM runtime, or Android backend has run.
A completed remote job is authoritative only when its actual checkpoint and
every enabled deployment package have 32-row evaluation outputs, TensorBoard
records, artifact hashes, package audits, and a complete final scorecard.
