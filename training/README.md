# Response-to-IR Training

For **E2B / 270M training and automatic final testing on both Golden32 and
Golden35**, start with the [end-to-end quickstart](docs/GOLDEN_E2E_QUICKSTART.md).
For the external 151,202-row messages-only archive, use the
[final archive runbook](docs/messages_archive_final_review.md) and review its
[policy-v9 report](reports/offline_recovery_20260913_v9/REPORT.md)
before passing the recovered `train.jsonl`/`val.jsonl` through `--input-dir`.
The launcher now provides live console/file logs, CPU-parallel preparation,
progress/ETA, and content-verified preparation reuse; see the quickstart's
[startup controls](docs/GOLDEN_E2E_QUICKSTART.md#startup-progress-cpu-preparation-and-reuse).
The final v9 copy includes the source-proven separator correction and retains
112,842 training and 2,300 validation rows; run the exact-model preflight next.
The current `run_golden_training.py` command uses checked-in Stage 3 data and
Golden artifacts, aligns one production prompt, filters held-out sources,
prepares with the local model tokenizer, selects all visible GPUs, and runs
preflight, training and selected-best/final checkpoint tests. Model weights
must be supplied locally; omitting `--execute` only prints a plan.

```bash
python training/scripts/run_golden_training.py \
  --profile e2b --model-dir /models/e2b \
  --output-dir /runs/e2b-new --execute
```

Defaults are validation/save every 500 optimizer updates, Golden32 every 1,000
plus final weights, final Golden35 testing, 2,048 generated tokens and
`/tensorboard/<run-id>/`. Golden32 has 32 occurrences / 31 unique sources;
Golden35 has 35 unique references and is not used for checkpoint selection.
Their excluded originals remain reserved from training. This dense capability
workflow is distinct from official retained-scale QAT/LiteRT export and MTP.
See [GPU profiles](docs/gpu_training_profiles.md), the
[data audit](docs/data_filtering_audit_20260912.md) and
[H100 implementation notes](docs/H100_GOLDEN_TRAINING.md). CPU validation does
not establish H100 throughput, model quality or deployed-runtime parity.

This folder trains local Stage 3 models that convert Stage 2 response text into the Android A2UI Express v1 IR:

```text
<a2ui>
root=Column([content])
content=Text("...","body")
</a2ui>
```

`dataset/` remains responsible for generating queries, responses, assets, and cloud IR. `training/` consumes completed dataset runs and provides a model-agnostic path for data preparation, SFT training, evaluation, export, and Android packaging. FlatSpec is accepted only as a read-only legacy source; Compact IR v2 is migration-only and is never a training target.

## Training pipeline inventory

The detailed, file-by-file status and legacy boundaries are maintained in
`training/docs/training_pipeline_inventory.md`.

| Pipeline | Entry point | Current status |
|---|---|---|
| Dense E2B LoRA / 270M full-model training with shared-prompt Golden32 and Golden35 tests | `training/scripts/run_golden_training.py` | Current clone-and-run capability workflow; prepares tracked source data, filters held-out sources, runs GPU preflight/training, tests selected-best and final checkpoints on both sets, logs TensorBoard and scorecard; no LiteRT export |
| Gemma 4 E2B retained-scale QAT, official-format package, and W32/W16/W8/mixed-W4-W8 comparisons | `training/scripts/run_gemma4_e2b_a2ui_express_multiformat.py` | Recommended end-to-end Golden-32 handoff; W16 is experimental; dry-run first, unique run ID, periodic/final TensorBoard scores, hash-bound scorecard |
| Gemma 3 270M W8-QAT and W32/W16/W8/W4 comparisons | `training/scripts/run_gemma270m_a2ui_express_multiformat.py` | Recommended end-to-end Golden-32 handoff; W8 is QAT-aligned, W16/W4 are explicit experimental conversions |
| Checked full-model SFT/QAT and LoRA/QLoRA SFT | `training/scripts/train_sft.py` | Shared HF trainer; explicit method, complete-example loss masking, actual Linear target resolution and strict resume contract |
| Gemma 4 E2B retained-scale target/MTP merge and package gates | `training/scripts/run_gemma4_e2b_mobile_mtp.py` | Current lower-level official-topology implementation used by the E2B multiformat orchestrator |
| Gemma 3 270M W8-only QAT/LiteRT-LM path | `training/scripts/run_gemma270m_qat_litertlm.py` | Older, narrower INT8 deployment path; retained for focused Q8/Android work |
| Gemma 3 270M full-parameter QAT | `training/scripts/train_full_finetune_qat.py` | Separate multi-GPU experiment; not wired into the LoRA merge/multiformat pipeline |
| Optional Gemma 4 MTP drafter QAT | `training/scripts/train_gemma4_mtp_drafter.py` | Opt-in teacher-forced public reconstruction; not a response-to-IR model and not Google's private recipe |
| A2UI Express GRPO after SFT | `training/scripts/train_grpo.py` | Current optional policy-optimization experiment; requires a strict Express SFT checkpoint |
| Generic DPO experiment | `dpo/train_dpo.py`, `dpo/src/train.py` | Separate preference-training tree; not integrated with A2UI Express data or evaluation |
| Historical trainers and shell launchers | `training_scripts/`, selected `training/scripts/*.sh` | Legacy/site-specific; several hard-code external paths or reference missing files and are not the recommended handoff |

There is no implemented A2UI distillation pipeline. “Official-format” in the
E2B path means the released topology, precision assignment, and retained scale
bytes are preserved while trained integer codes are transplanted. The optimizer
and STE-QAT loop are repository implementations, not Google's private training
or calibration recipe.

The two recommended multiformat pipelines use the pinned 2026-09-03 Golden-32
only for evaluation. Older deployable profiles described below still use their
separate Golden-100 contract; do not combine those sets or copy Golden rows into
training data.

## Main Flow

1. Prepare response-to-IR pairs from a dataset run.

```powershell
python training/scripts/prepare_dataset.py --config training/configs/datasets/dataset_v1_stage3.yaml
```

For Gemma 4 training from a folder that contains one or more Stage 3 `genui.jsonl`
files, prepare a deterministic 90/10 train/validation split with:

```powershell
python training/scripts/prepare_dataset.py --config training/configs/datasets/stage3_folder_90_10.yaml --source-genui-dir dataset/data/runs/dataset_v3 --output-dir training/outputs/datasets/stage3_folder_90_10
```

The folder reader is recursive by default (`**/genui.jsonl`). It writes
`train.jsonl`, `val.jsonl`, `test.jsonl`, and `all.jsonl`; `all.jsonl` is used for
fixed-set evaluation jobs.

When `url_preprocessing.enabled` is true, preparation replaces every URL/URI and
local asset reference in both the response prompt and IR target with typed
placeholders. The exact placeholder map is stored in row metadata and evaluation
restores the original references before comparison or exported prediction review.

Prepare the Golden35 evaluation set once before training:

```powershell
python training/scripts/prepare_dataset.py --config training/configs/datasets/golden35_stage3_eval.yaml
```

2. Train an adapter model.

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma_e2b_ir_lora.yaml
```

The source artifact is `training/data/eval/golden35_v1/golden35.jsonl`; prepared
evaluation rows are written to `training/outputs/datasets/golden35_stage3_eval/all.jsonl`.
Its manifest retains the excluded 15 source identities as training reservations.
Use `--max-rows 35 --required-rows 35` for explicit standalone evaluation.
The archive-based restart above still uses its separate Golden32 development set.

Start Gemma 4 E2B LoRA training with Golden35 evaluation at each configured
Trainer evaluation event:

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_lora.yaml
```

The Gemma 4 E2B profile uses the fast-tokenizer loader and PEFT's
`all-linear` discovery used by the supplied working trainer. Golden evaluation
runs every `training.eval_steps`, distributes generation across DDP ranks, and
saves the highest-`overall_score` adapter under
`training/runs/gemma4_e2b_ir_lora/best_golden_checkpoint`. Per-evaluation
predictions and metrics remain under
`training/outputs/eval/gemma4_e2b_ir_lora/golden35/step_*`.

The callback is controlled by the `golden_eval` YAML block. Use
`trigger: epoch` for epoch-end evaluation, increase `interval` to evaluate less
often, or set `save_best_checkpoint: false` to retain metrics without saving a
second adapter checkpoint. `max_input_tokens + max_new_tokens` is bounded by
the configured model context.

### Gemma 4 QAT-derived LoRA and MTP workflow

This older alternative low-bit experiment is intentionally separate from the
standard Gemma 4 baseline and the recommended retained-scale pipeline below.
It fine-tunes only Google's unquantized Q4_0 QAT-derived
target with BF16 LoRA, keeps the matching assistant frozen, and requires
post-merge Q4_0 conversion plus final runtime/device validation. It is **not**
continued QAT or joint target/assistant training.

Read the durable support boundaries, recommendation, command inventory, and
promotion gates before using it:

- `training/docs/gemma4_e2b_qat_mtp_knowledge.md`
- `docs/gemma4_e2b_qat_mtp_implementation_prompt.md`

The deployable Gemma 4 E2B and Gemma 3/FunctionGemma 270M QAT profiles run an
exact, immutable Golden-100 generation test at every configured Trainer
evaluation event. The repository intentionally does not turn the existing
Golden35 into a synthetic Golden-100. Prepare a separate held-out source with
exactly 100 unique accepted rows:

```powershell
python training/scripts/prepare_dataset.py `
  --config training/configs/datasets/golden100_stage3_eval.yaml `
  --source-run-dir C:\path\to\immutable-golden100-run
```

Preparation and training both fail closed unless the split contains exactly
100 unique rows. At every `training.eval_steps`, the callback generates all 100
predictions, writes the step artifacts under
`training/outputs/eval/<run-id>/golden100/step_*`, computes the official GenUI
Representation Quality v5.4 generation score, and selects the best adapter by
`generation_reward_v5_4_avg`. It logs all finite Golden-100 aggregates through
`Trainer.log`; the headline TensorBoard tag is
`eval/golden100/v5_4_score`. Start TensorBoard with:

```powershell
tensorboard --logdir training/runs
```

This full autoregressive Golden-100 pass pauses training and can be expensive;
`eval_steps` controls frequency. It applies to the response-to-IR target SFT.
The MTP drafter is not a standalone response-to-IR model, so its training loop
continues to use drafter validation loss and separate target-plus-assistant
acceptance/latency benchmarks rather than mislabeling target quality as a
drafter v5.4 score.

This Golden-100 section describes the older deployable profiles. The newer
multiformat orchestrators listed above instead use the exact pinned Golden-32,
write training and evaluation events under the repository-root `tensorboard/`
(or `/tensorboard` via `A2UI_TENSORBOARD_ROOT` on MLP), and produce an
artifact-bound final scorecard.

Install the current Gemma 4 dependency overlay and run the no-model static
preflight:

```powershell
python -m pip install -r training/requirements-gemma4-qat.txt
python training/scripts/validate_qat_mtp_workflow.py
```

The expensive helpers are plan-only unless `--execute` is supplied:

```powershell
python training/scripts/merge_qat_lora.py
python training/scripts/benchmark_qat_mtp.py
python training/scripts/convert_qat_q4_0.py --llama-cpp-dir C:\path\to\llama.cpp
```

The later authorized training command is:

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_qat_lora.yaml
```

Do not run that command for code-only setup or validation.

This requires a GPU machine with the packages in
`training/requirements-training.txt`. On CPU-only machines, use compile/tests only;
do not run the training command.

### True QAT SFT for Gemma 4 E2B and Gemma 270M

The repository also has an opt-in fake-quantization-aware LoRA path. The Gemma
4 profile consumes the released mobile W2/W4/W8 target inventory,
fake-quantizes every matching embedding and each effective projection weight
(`base_weight + LoRA_delta`), and uses the public AI Edge min/max range convention
(`ste_ai_edge`: full signed W2/W4, narrow symmetric W8). Gemma 270M uses W8
weight fake quantization with FP32 activation edges to match its released Q8
graph. This is different from the QAT-derived `qat_mtp` profile above, does not
train an MTP assistant, and does not claim Google's private mobile
observer/calibration recipe.

The `ste_ai_edge` numerical path is regression-checked against the public
`ai-edge-quantizer` 0.8.0 implementation: min/max and scale calculations are
FLOAT32 even for BF16 training tensors, the scale floor is `1e-9`, rounding is
ties-to-even, and 256-column grouped scales follow the public
BF16-to-FP16-to-FLOAT32 storage conversion. It returns simulated values to the
original training dtype and fails instead of silently changing a non-divisible
group layout. Verify this without loading or training a model:

```powershell
python training/scripts/validate_ai_edge_qat_numeric_contract.py
```

The recommended E2B profile uses a text-only BF16 reconstruction of Google's
exact packed `gemma-4-E2B-it-qat-mobile-transformers` checkpoint. The packed
checkpoint and released `.litertlm` remain the observable numerical/layout and
graph authorities; Google does not publish its dense pre-quantization master
checkpoint. The pipeline rejects a missing/tampered reconstruction manifest,
an incompatible base, or merge provenance bound to a different seed. Its
optional trainable drafter remains a separate public reconstruction.

Important production boundary: a bounded byte audit shows that the dense
Q4-QAT seed and packed mobile checkpoint share all 262 retained language-model
tensor schemas, but only 50 tensors are byte-identical and 212 learned
norm/scalar tensors differ. A separate hash-bound graph audit maps all 262
packed-mobile BF16 tensors to the released target's FLOAT32 buffers, and all
262 are exact after FLOAT32-to-BF16 round-to-nearest-even conversion. This
proves the packed-mobile checkpoint is the public numerical authority for those
constants at BF16 precision; it does not recover the compiled buffers' lower
FLOAT32 mantissa bits. Consequently, the dense-Q4/mobile cross-checkpoint
combination remains rejected. The checked-in mobile reconstruction instead
copies those retained values and dequantizes the packed matrices, with every
output bound by a materialized manifest. Graph/GPU
topology parity is proven by the random-weight harness, but it does not make a
hybrid of Q4-seed projections and mobile-package constants an accuracy-valid
trained model. Training/export still fail closed until the local 541-tensor
mobile-seed manifest and every shard hash verify.

The QAT profiles require `lora.dropout: 0.0`: an input-dependent adapter
dropout mask has no exact equivalent in the final merged inference matrix.
Training metadata hashes the selected adapter and records the number of PEFT
wrappers using effective-weight QAT; merge manifest v4 also binds the mobile
seed manifest/local source and rejects old base-only-QAT or unbound adapters
before model loading. The mobile profile additionally requires Transformers
loading diagnostics with zero missing, unexpected, mismatched, or errored
checkpoint keys; that requirement is preserved in merge/compiler provenance.
The corrected mobile profile retains the packed checkpoint's immutable F32
weight and static A8 scales and applies QAT only to the exact 205 effective
base+LoRA projections. Frozen embeddings already contain the published
dequantized cell centers and are not dynamically requantized.

Validate all profiles without loading models or running training:

```powershell
python training/scripts/validate_qat_training.py
```

For E2B, first plan and then explicitly execute
`training/scripts/reconstruct_gemma4_mobile_training_seed.py` as documented in
`training/docs/gemma4_e2b_qat_mtp_knowledge.md`; reconstruction writes model
files but does not train. The E2B training command below refuses to load a model
until that manifest verifies. Before allocating model weights, independently
compare all 541 checkpoint headers with the actual Transformers architecture:

```powershell
python training/scripts/validate_gemma4_mobile_seed_architecture.py
```

This instantiates `Gemma4ForCausalLM` on the meta device only. It fails on an
old Transformers version, wrong class/config, non-BF16 checkpoint tensor,
missing/unexpected key, or any shape mismatch. Direct E2B SFT runs repeat this
preflight and record it in training metadata before loading real weights.

For Gemma 4 E2B, use the fail-closed portable launcher and read its remote-PC
runbook. It is plan-only by default, runs strict streamed scale/code and real
model loss/top-token plus deterministic greedy preflights with `--preflight`,
and trains only with `--execute`:

```powershell
python training/scripts/run_gemma4_mobile_qat.py --config training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml --run-id e2b_mobile_qat_smoke_plan --num-gpus 4
python training/scripts/run_gemma4_mobile_qat.py --config training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml --run-id e2b_mobile_qat_smoke_preflight --num-gpus 4 --preflight
python training/scripts/run_gemma4_mobile_qat.py --config training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml --run-id e2b_mobile_qat_smoke_train --num-gpus 4 --execute
```

That explicit profile is a bounded three-optimizer-step serialization/device
smoke: it evaluates and saves once at step 3, is not quality-promotion eligible,
and must not be mistaken for the full training schedule. Add
`--source-safetensors <local-packed-model.safetensors>` when the copied seed
manifest still names the old PC.

Do not use the direct E2B SFT command because it omits the launcher's
cross-artifact hashes and streamed full-projection gate. Other target profiles
retain their direct commands:

```powershell
python training/scripts/train_sft.py --config training/configs/models/gemma3_270m_ir_qat_sft.yaml
python training/scripts/train_sft.py --config training/configs/models/functiongemma_270m_ir_qat_sft.yaml
```

Read `training/docs/gemma4_e2b_qat_mtp_knowledge.md` before changing the
quantizer or export settings, and read
`training/docs/gemma4_mobile_qat_remote_pc_runbook.md` before launching E2B.
The retained published scales and `ste_ai_edge` integer ranges reproduce public
observable deployment semantics; they are not Google's hidden QAT trainer.
New retained-scale checkpoints must use the dedicated code-only 205-projection
export stage described below. The legacy exact-topology/public abs-max exporter
and composer remain blocked for Gemma 4 `retained_mobile` checkpoints.

The exact Google Transformers config copy remains
`configs/quantization/gemma4_e2b_mobile_public_schema.yaml` for strict source
audits. Training points to the separate
`gemma4_e2b_mobile_litertlm_schema.yaml`, whose documented W8
`per_layer_model_projection` override matches the released target graph.
Precision matching canonicalizes known PEFT/multimodal wrapper prefixes before
applying the ordered rules, so anchored entries such as `^lm_head$` remain W2
after LoRA wrapping instead of silently falling through to the default W4.

### Recommended Gemma 4 E2B Golden-32 multiformat pipeline

Use the dry-run-first orchestrator for target training, selected-checkpoint
evaluation, merge, official retained-scale export, public
W32/W16/W8/mixed-W4-W8 comparison exports, identical Golden-32 evaluation, and
the final scorecard:

```powershell
python training/scripts/run_gemma4_e2b_a2ui_express_multiformat.py `
  --run-id e2b_a2ui_YYYYMMDD_001
```

On a clean checkout the plan intentionally reports missing large
seed/package/runtime inputs as blockers and performs no writes. The complete
fresh-machine prerequisites, MTP on/off semantics, separate training/export
environments, recovery commands, TensorBoard layout, and evidence gates are in
`training/docs/gemma4_e2b_a2ui_express_multiformat_runbook.md`.

### Gemma 4 E2B mobile QAT -> best checkpoint -> MTP package

The end-to-end mobile hand-off is defined in
`training/configs/pipelines/gemma4_e2b_mobile_mtp.yaml`. It selects the
golden-set best adapter and merges it into the BF16 base. Its MTP stage has two
explicit weight sources and a target-only switch:

- `mtp.weight_source: official` (default) preserves the released drafter
  section byte-for-byte;
- `mtp.weight_source: trained` optionally trains the public four-layer
  target-conditioned assistant, then replaces only its 23 mapped W4/W8
  matrices inside the released drafter graph;
- `mtp.enabled: false` skips drafter training/transplant and makes target-only
  Android GPU parity the sole runtime gate. The package deliberately keeps the
  official drafter section byte-for-byte so graph/package identity is retained
  and the same artifact can later run with MTP enabled.

The pipeline is plan-only by default. For a completed portable-launcher run,
bind the exact resolved config, callback-created best-Golden checkpoint, exact
official package variant, and fresh destinations explicitly:

```powershell
python training/scripts/run_gemma4_e2b_mobile_mtp.py `
  --training-config <run_root>/launch/resolved_training_config.yaml `
  --best-checkpoint <run_root>/best_golden_checkpoint `
  --base-litertlm <official.litertlm> `
  --merged-model-dir <fresh_export_root>/merged_best_hf `
  --exact-output-dir <fresh_export_root>/retained_scale_export `
  --export-report <fresh_export_root>/retained_scale_export/report.json `
  --output-litertlm <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm
```

Training completion alone is not export authorization. Review the plan and
best-Golden provenance, then add both `--execute-merge` and
`--execute-retained-scale-export`. The exporter fails closed unless the
pre-step numeric/greedy gates, exact 205 retained-qparams bindings, selected
adapter and merge hashes, immutable scale/frozen-constant contract, and final
exporter report all pass. The merged directory, exact-output directory, report,
and `.litertlm` output must not already exist. It rejects the dense
Q4-seed/mobile-template hybrid and never silently falls back to abs-max scales.

```powershell
python training/scripts/run_gemma4_e2b_mobile_mtp.py `
  --training-config <run_root>/launch/resolved_training_config.yaml `
  --best-checkpoint <run_root>/best_golden_checkpoint `
  --base-litertlm <official.litertlm> `
  --merged-model-dir <fresh_export_root>/merged_best_hf `
  --exact-output-dir <fresh_export_root>/retained_scale_export `
  --export-report <fresh_export_root>/retained_scale_export/report.json `
  --output-litertlm <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm `
  --execute-merge --execute-retained-scale-export

python training/scripts/validate_litertlm_mtp_gpu.py `
  <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm `
  --inspect-graphs `
  --official-artifact <official.litertlm>
```

The initial candidate preserves the released MTP section byte-for-byte and must
be tested target-only first. Keep MTP disabled until desktop inspection,
target-only semantic validation, and Android GPU target-only gates pass.

Reproduce the bounded checkpoint audit without downloading the full dense
checkpoint (the local file should be the public packed mobile Safetensors):

```powershell
python training/scripts/audit_hf_retained_constant_parity.py `
  --remote-repo google/gemma-4-E2B-it-qat-q4_0-unquantized `
  --local-safetensors PATH_TO_PACKED_MOBILE_MODEL_SAFETENSORS `
  --local-model-id google/gemma-4-E2B-it-qat-mobile-transformers `
  --output RETAINED_CONSTANT_AUDIT_JSON
```

The script accepts only bounded HTTP `206 Partial Content` responses and exits
nonzero when any selected schema/value differs. It does not run training.

Map the packed-mobile retained constants into an exact released target package:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/audit_gemma4_mobile_retained_compiled_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --source-safetensors C:\path\to\gemma4-mobile\model.safetensors `
  --official-artifact-sha256 181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c `
  --output C:\temp\gemma4-retained-compiled-parity.json
```

The audit is read-only, requires the exact package hash, uses semantic decode
graph consumers rather than buffer order, and verifies 262/262 tensors at the
public BF16 precision boundary. It does not run training or claim recovery of
Google's FLOAT32 master checkpoint or private recipe.

The retained-scale export candidate keeps `tf_lite_mtp_drafter` byte-for-byte
official and MTP disabled. Only after target-only semantic and Android GPU gates
pass may a separately reviewed trained-drafter experiment set
`weight_source: trained` and `train_assistant: true`. Do not use the legacy
`--execute-exact-topology-export` to attach that drafter to a
`retained_mobile` target.

The drafter trainer is a public reconstruction, not Google's private recipe.
It freezes the fine-tuned target and every assistant parameter outside the 23
deployable matrices, uses target final hidden state + target token embedding +
shared target KV with a constant position ID, predicts four draft tokens with
teacher-forced completion cross-entropy, and applies W4/W8-A8 STE fake
quantization matching the released inventory. The exporter requires hashed
training provenance and then
preserves the target and every byte outside `tf_lite_mtp_drafter` while
injecting the newly quantized matrices into the official graph/layout.

The legacy `--compose` path is blocked for Gemma 4 `retained_mobile`. A package
passing desktop gates is only structurally ready for later device testing; it
is not proof of Android GPU execution, semantic quality, or useful draft
acceptance.

After installing the `androidTest` APK, compare the untouched official package
and the composed candidate without replacing any catalog model:

```powershell
python training/scripts/benchmark_android_litertlm_gpu_parity.py `
  --official C:\path\to\official.litertlm `
  --candidate C:\path\to\candidate.litertlm `
  --output-dir C:\temp\android-gpu-parity `
  --mtp --max-num-tokens 8192 --output-tokens 64 `
  --prompt "Write exactly one hundred numbered words."
```

The runner checks complete GPU delegation, signature parity, bounded decode
throughput, and MTP acceptance separately. Schema-v5 performance reports
alternate which package runs first, require three valid warm samples for each
artifact, and compare the warm medians; one failed delegation or short decode
invalidates the speed gate. Target-only runs must decode exactly
`--output-tokens`. MTP runs may finish up to
`--mtp-max-decode-overshoot` tokens above that request because LiteRT-LM checks
the cap after one four-position verifier batch; decoding fewer than requested
or exceeding that bound remains a fail-closed, non-comparable speed sample. For
random-weight graph fixtures, deterministic
`--top-k`, `--top-p`, `--temperature`, and `--seed` controls can be used to find
an identical full-length sampling run. See
`training/docs/gemma4_e2b_qat_mtp_knowledge.md` for the verified SM-F966B
results and why a preserved official drafter does not guarantee unchanged MTP
speed after target fine-tuning.

Use `run_gemma4_mobile_qat.py` for portable multi-GPU target training. After its
best-Golden checkpoint and resolved launcher config are independently reviewed,
the dedicated pipeline command above performs merge plus retained-scale export
as one explicitly authorized hand-off. For Gemma 4 `retained_mobile`, the
pipeline rejects `--execute-public-export`, `--execute-exact-topology-export`,
and `--compose`; none may be used as a substitute for the code-only exporter.

### Gemma 3 270M QAT -> LiteRT-LM INT8

The equivalent 270M pipeline is separate because Gemma 3 270M has no Gemma 4
MTP drafter. It trains the checked-in W8/activation-FP32 QAT profile (including
the embedding table), selects the golden-set
best adapter, merges it, and invokes the public `dynamic_wi8_afp32` LiteRT
exporter:

```powershell
python training/scripts/run_gemma270m_qat_litertlm.py
python training/scripts/run_gemma270m_qat_litertlm.py --execute-training
python training/scripts/run_gemma270m_qat_litertlm.py --execute-merge
python training/scripts/run_gemma270m_qat_litertlm.py --execute-export
```

The generated artifact is an INT8 `.litertlm`, not an INT4/Q4_0 model and not
an MTP package. Validate its package envelope before device testing:

```powershell
python training/scripts/validate_gemma270m_litertlm.py `
  training/outputs/pipelines/gemma3_270m_qat_litertlm/gemma3_270m_qat_int8.litertlm `
  --inspect-graphs `
  --official-artifact C:\path\to\official-gemma3-270m-it-q8.litertlm
```

The embedded-TFLite check requires a completely decoded execution contract;
with the official reference it also requires exact contract parity. Provide an
Android GPU device report for the runtime gate. Do not enable MTP for this model;
the validator treats an MTP section as unexpected rather than attaching the
Gemma 4 assistant.

For the newer end-to-end A2UI Express comparison that binds the supplied
2026-09-03 Golden-32, logs periodic and final scores below the repository-root
`tensorboard/`, and evaluates W32/W16/W8/W4 LiteRT-LM variants from one merged
W8-QAT checkpoint, use the dry-run-first entry point:

```powershell
python training/scripts/run_gemma270m_a2ui_express_multiformat.py
```

W16 and block-32 W4 require `--allow-experimental-formats`; only W8 is aligned
with the training fake-quantization objective. The complete remote-machine
procedure, external LiteRT-LM runner protocol, official-Q8 option, output
layout, and scorecard gates are in
`training/docs/gemma3_270m_a2ui_express_multiformat_runbook.md`.

To audit the publicly observable Google mobile schema without downloading
weights:

```powershell
python training/scripts/audit_gemma4_mobile_schema.py
```

To inspect a local `.litertlm` package and its embedded TFLite graph without
loading model weights into a training framework, use:

```powershell
python training/scripts/audit_litertlm_package.py C:\path\to\model.litertlm
```

To audit the observable mobile quantization contract (including INT2/INT4/INT8
weights, symmetric per-axis scales, INT8 activation edges, and a separate MTP
graph), use the optional `tflite` schema bindings:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/audit_litertlm_recipe.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --strict-mobile
```

The public AI Edge `gemma4_mixed48` recipe is captured separately in
`training/configs/quantization/gemma4_mixed48_public_recipe.json`; it is a
W4/W8 channelwise recipe and must not be described as the private mobile W2/W4/W8-A8
artifact recipe.
The recipe-audit JSON also emits `fully_connected_assignments`, grouped by
layer and operation family, so later work can compare bit assignments without
mistaking them for Google's hidden QAT/calibration schedule.
The checked-in snapshot of those observable rules is
`training/configs/quantization/gemma4_e2b_mobile_observable_contract.yaml`;
it is evidence metadata, not a drop-in private exporter recipe.

For the explicit random-weight graph parity experiment (still no training),
review the plan first and add `--execute` only in the separate LiteRT export
environment:

```powershell
python training/scripts/build_random_litertlm_parity.py `
  --model-source google/gemma-3-270m `
  --official-artifact C:\path\to\official.litertlm `
  --output-dir C:\temp\random-litertlm-parity `
  --quantization-recipe dynamic_wi8_afp32
```

Read `training/docs/gemma4_e2b_qat_mtp_knowledge.md` for the evidence levels:
random weights can test graph and quantization layout, but cannot prove equal
learned weights, private calibration, or exact Google recipe provenance.

For a small executable public-recipe check that builds a deterministic random
TFLite graph and quantizes it without Transformers or training:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_random_tflite_recipe_parity.py `
  --output-dir C:\temp\random-gemma4-public-recipe `
  --recipe gemma4_mixed48 `
  --execute
```

To exercise the observable mobile W2/W4/W8-A8 contract with a tiny random
graph, use the separate static experiment below. It applies deterministic
random calibration inputs and records the resulting TFLite layout; it does not
train, download weights, or claim Google's private recipe. The custom recipe
sets `skip_checks=true` because the public AI Edge policy does not advertise
arbitrary static W2 `FULLY_CONNECTED` operations:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_random_mobile_contract_parity.py `
  --output-dir C:\temp\random-gemma4-mobile-contract `
  --official-artifact C:\path\to\gemma-4-E2B-it.litertlm `
  --execute
```

The synthetic report should show INT2/INT4/INT8 fully-connected weights,
channelwise axis-0 symmetric scales, and symmetric INT8 input/output edges.
`synthetic_contract_match=true` is only converter/layout evidence;
`exact_official_mobile_reproduction` remains false by construction.

For stronger artifact-level evidence, randomize the weights in the actual
official TFLite section while preserving its graph and quantization layout:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_random_official_topology_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --model-type tf_lite_prefill_decode `
  --output C:\temp\gemma4-prefill-random.tflite `
  --execute `
  --verify-weight-encoding

python training/scripts/build_random_official_topology_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --model-type tf_lite_mtp_drafter `
  --output C:\temp\gemma4-mtp-random.tflite `
  --execute `
  --verify-weight-encoding
```

The harness writes standalone `.tflite` files and never rewrites the source
`.litertlm`. A matching structural and quantization-layout hash proves graph
and packing/layout parity only; random weight/scales hashes intentionally do
not match and the private QAT recipe remains unrecovered. It also supports
external TFLite constant buffers (`Buffer.offset`/`Buffer.size`), which is the
layout used by the local Gemma 3 270M INT8 candidate:

```powershell
python training/scripts/build_random_official_topology_parity.py `
  C:\path\to\gemma-3-270m-ir-int8.litertlm `
  --model-type tf_lite_prefill_decode `
  --output C:\temp\gemma270m-prefill-random.tflite `
  --execute `
  --runtime-allocate
```

That candidate yielded 127 randomized INT8 weight buffers with identical
graph/layout hashes. `--runtime-allocate` additionally checks that the
original extracted section and randomized output allocate with identical
signatures. It is a custom local export. The byte-verified official
`gemma3-270m-it-q8.litertlm` target is 304,005,120 bytes with SHA-256
`757e9119fa5bd667a2774fb470ac4afcd3190a21c677f8e69a5d6bc908abdd63`; its
v1.3 graph has six subgraphs, 9,872 operators, 11,702 tensors, 732 INT8
fully-connected weights, and six INT8 embedding tables. The local v1.5 graph
has 215 subgraphs and 4,611 tensors, so it cannot become byte-equivalent by
changing only its scale formula. The public exporter does not
expose the official artifact's audio and `tf_lite_mtp_drafter` packaging path,
so this harness is an evidence/regression tool, not a private recipe
substitute.

If the merged BF16 source safetensors for that local candidate are available,
prove its public INT8 rule directly:

```powershell
python training/scripts/audit_litertlm_weight_recipe.py `
  C:\path\to\gemma-3-270m-ir-int8.litertlm `
  --source-model C:\path\to\merged_hf `
  --model-type tf_lite_prefill_decode `
  --strict
```

The audit matched 127/127 candidate weights exactly: axis-0 scales were
`max(abs(source_row))/127` and the packed INT8 rows matched round-and-clip.
That identifies the local export as public `dynamic_wi8_afp32`/channelwise
`dynamic_wi8c_afp32`; it does not identify the gated official package or a
private QAT training recipe.

To build a genuinely fresh FlatBuffer from the official section's observable
weight inventory, use `build_fresh_random_quantized_graph.py`. It does not copy
the official graph: it creates independent random branches, applies the
observable W2/W4/W8 symmetric axis-0 rule, and compares the new quantized weight
inventory with the official one. The default is in-memory; add `--output` only
when a standalone `.tflite` fixture is wanted:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_fresh_random_quantized_graph.py `
  C:\path\to\official-gemma3-270m-it-q8.litertlm `
  --model-type TF_LITE_PREFILL_DECODE `
  --include-embeddings
```

On the verified 270M target this produced 127 fresh random branches (126
linear matrices plus the embedding table) with an identical inventory digest
and `fresh_quantization_layout_match=true`. `graph_structure_match` remains
false by design; the separate topology harness is required to prove official
cache/signature/operator parity. For Gemma 4 MTP, the same command over
`tf_lite_mtp_drafter` reproduced all 23 observable weight layouts (13 W4 and
10 W8) in a fresh 43.9 MB graph, while deliberately leaving graph structure
and learned values different.

To run the same inventory through the public AI Edge Quantizer rather than the
hand-built quantization loop, use
`build_converter_random_inventory_parity.py`. It builds an independent random
FLOAT32 FlatBuffer, applies a per-branch public W2/W4/W8 recipe, calibrates A8
edges when required, and compares only the resulting quantized FC layout:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_converter_random_inventory_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --model-type tf_lite_mtp_drafter `
  --output-dir C:\temp\random-inventory-mtp `
  --calibration-samples 2 `
  --threads 1
```

The full MTP inventory matched 23/23 layouts (13 W4, 10 W8); the full 270M
section matched 126/126 FC layouts (the embedding is skipped in this ordinary
FC graph and is covered by the topology harness); and a bounded Gemma 4
prefill run matched 8/8. These runs prove public converter/layout behavior,
not graph topology, learned weights, private QAT, or exact LiteRT-LM package
provenance. Reports intentionally keep
`converter_quantization_layout_match=true` separate from
`exact_official_model_match=false`.

### Converter-driven official-topology parity

`build_converter_topology_parity.py` is the stronger graph experiment. It
unpacks a selected official TFLite section with the public object API, replaces
its low-bit constants with deterministic random FLOAT32 constants, bypasses the
now-invalid Q/DQ edges, and sends that topology through the public AI Edge
Quantizer. It never modifies the `.litertlm` source. The report includes both
normal fingerprints (serializer-sensitive buffer indices included) and
buffer-agnostic fingerprints for execution topology and quantization layout:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/build_converter_topology_parity.py `
  C:\path\to\gemma3-270m-it-q8.litertlm `
  --model-type TF_LITE_PREFILL_DECODE `
  --output-dir C:\temp\converter-topology-270m `
  --calibration-samples 2 `
  --threads 1 `
  --in-memory
```

The verified Gemma 3 270M run preserved all six subgraphs, 9,872 operators,
11,702 tensors, 732 `FULLY_CONNECTED` operators, six `EMBEDDING_LOOKUP`
operators, and all 737 INT8 quantized tensors. The normal hash differs because
the standalone FlatBuffer is reserialized, but both
`converter_execution_topology_match_ignoring_buffer_indices` and
`converter_quantization_layout_match_ignoring_buffer_indices` are `true`.
`exact_official_model_match` remains `false`: constants are random and package
serialization/metadata are not reconstructed.

The same experiment on the official Gemma 4 `tf_lite_mtp_drafter` section is
an explicit public-boundary result. The converter retained 30 subgraphs and
all 23 FC records, but produced 357 rather than 365 operators, 615 rather than
575 tensors, 22 rather than 30 `DEQUANTIZE` operators, and 63 rather than 79
quantized tensors. Both buffer-agnostic matches are `false`. This mixed W4/W8
MTP graph therefore depends on Google's private/custom lowering path; an
independent public converter cannot claim exact MTP graph parity.

### Converter-driven random constants injected into official topology

`build_converter_random_topology_injection_parity.py` quantizes one independent
random FLOAT32 branch per official FC/embedding weight with the public AI Edge
Quantizer, then copies only the packed bytes and per-axis scales into an
in-memory copy of the released section. It preserves the official operators,
tensors, signatures, metadata, and cache wiring. Its low-level builder emits a
real `EMBEDDING_LOOKUP`, so this check covers the complete 270M inventory rather
than silently replacing the token embedding with an FC branch.

The latest verified seed-42 runs used two calibration samples and produced:

| section | inventory | public converter layout | injected structure/layout | allocation |
| --- | ---: | --- | --- | --- |
| Gemma 4 E2B `tf_lite_embedder` | 1 embedding (W2) | 1/1 | true / true | true (no default delegates) |
| Gemma 4 E2B `tf_lite_per_layer_embedder` | 35 embeddings (W4) | 35/35 | true / true | true (six batches; no default delegates) |
| Gemma 4 E2B `tf_lite_audio_encoder_hw` | 120 FC (W2/W4) | 120/120 | true / true | true (no default delegates) |
| Gemma 4 E2B `tf_lite_vision_encoder` | 112 unique FC buffers (W8) | 112/112 | true / true | true (no default delegates) |
| Gemma 4 E2B `tf_lite_prefill_decode` | 277 FC (W2/W4/W8) | 277/277 | true / true | true (35 batches; no default delegates) |
| Gemma 4 `tf_lite_mtp_drafter` | 23 (13 W4, 10 W8) | 23/23 | true / true | true (no default delegates) |
| Gemma 3 270M `TF_LITE_PREFILL_DECODE` | 127 (126 FC + 1 embedding) | 127/127 | true / true | true (no default delegates) |

The full E2B prefill independent float32 inventory is about 8.5 GiB. The
harness automatically quantizes it in deterministic eight-weight batches to
stay below FlatBuffers' 2 GiB offset limit, then combines the global ordinals
before injecting constants. The injected quantization values are random by construction, so the reports
correctly keep `quantization_values_match=false` and
`exact_official_model_match=false`. This is proof that public converter output
fits the official graph/layout and allocates in LiteRT; it is not recovery of
Google's learned QAT weights, observers, calibration data, private exporter,
or byte-identical package.

The JSON report makes the requested network check explicit:
`random_quantized_network.same=true` means complete FC/embedding inventory,
operator structure, quantization layout, and buffer-index-independent execution
topology all match. `converter_to_injected_quantization_values_match=true`
additionally proves that the converter's random packed bytes and scales survive
injection into the official section exactly. It does not compare those random
values with Google's learned values; that remains separately reported as
`quantization_values_match` and `exact_official_model_match`. With
`--runtime-allocate`, the nested
`runtime_allocation_match` is an additional allocation-only check.

Use `--rebuild-flatbuffer` for the stronger independent serialization check.
It deserializes the official section through the public
`ai_edge_litert.schema_py_generated.ModelT`, materializes external buffers,
places the AI Edge-converter random constants into that object graph, and packs
a new `TFL3` FlatBuffer. The report adds
`rebuilt_quantized_network.same=true` only when the complete execution contract,
structural/layout hashes, buffer-index-independent execution topology/layout,
converter-constant digest, and optional runtime allocation all match. The
contract includes decoded builtin/custom option values, signatures, metadata,
tensor semantics, wiring, and logical buffer sizes while masking buffer
payloads and quantization-scale values. Unsupported sparse unions or
external large custom options fail the completeness gate. A companion
`non_mapped_state_match` digest covers every buffer payload and quantization
record outside the explicit mapped FC/embedding inventory, so an unrelated
constant change also fails. Verified rebuilt runs pass for the
E2B token embedder (1/1), per-layer embedder (35/35), audio encoder (120/120),
vision encoder (112/112), main decoder (277/277), MTP (23/23), and Gemma 3
270M (127/127). The audio and vision counts are unique weight buffers; the
audit may show more operator occurrences when a buffer is reused. This still keeps
`quantization_values_match=false` and `exact_official_model_match=false`, as
required for a random model.

Fresh option-aware rebuilds also pass for the three shipping-critical graphs:
Gemma 3 270M (127 matrices), Gemma 4 E2B target (277 matrices in 35 converter
batches), and Gemma 4 MTP (23 matrices). Each has exact official/rebuilt
execution-contract hashes and host LiteRT allocation parity. A regression test
changes only `FullyConnectedOptions.keepNumDims`; the legacy structural hash
misses that change, while the execution contract correctly rejects it.

A shared-buffer audit is also mandatory. The decoder packages reuse each
packed matrix through distinct tensor/quantization tables: 270M has 737 tensor
aliases for 127 buffers (`5x1`, `122x6`), E2B has 812 aliases for 277 buffers
(`148x2`, `129x4`), and MTP has 23 one-to-one aliases. The injector and ModelT
rebuild now update and re-read every alias, reject inconsistent scale or
zero-point records, and reject any mapped buffer used outside an FC/embedding
weight slot. Earlier random packages updated only the first alias and are valid
only as historical topology/delegation evidence, not as numerically coherent
quantized fixtures. Alias-complete replacement packages supersede them.

The current public LiteRT-Torch exporter still lists quantized safetensor
support as a TODO and rejects a Hugging Face `quantization_config` with
`NotImplementedError("Quantized checkpoint is not supported yet.")`. The
maintainer's [public issue response](https://github.com/google-ai-edge/litert-torch/issues/998)
also identifies the pre-built Gemma 4 files as exports of QAT-mobile variants
and notes that quantized-safetensor conversion and MTP/audio export were not
public at that point. Therefore this harness can establish public network
compatibility, but cannot manufacture Google's private QAT/export recipe.

The report also proves the package boundary: the Gemma 4 MTP section is
44,325,712 bytes at offset 2,543,812,608 in the 2,588,147,712-byte package,
and the 270M section is 299,261,424 bytes at offset 4,734,976 in the
304,005,120-byte package. Prefix/suffix hashes show that the random fixture
changes only the selected TFLite section; tokenizer, metadata, and all other
sections remain byte-for-byte unchanged.

Example (the temporary 270M model fixtures are removed automatically):

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/build_converter_random_topology_injection_parity.py `
  C:\path\to\gemma3-270m-it-q8.litertlm `
  --model-type TF_LITE_PREFILL_DECODE `
  --output-dir C:\temp\converter-injection-270m `
  --calibration-samples 2 `
  --threads 1 `
  --in-memory `
  --rebuild-flatbuffer `
  --runtime-allocate `
  --runtime-without-default-delegates
```

Add `--package-output C:\temp\random-injected.litertlm` when a full package
fixture is needed; it streams the package and replaces only the selected
section, so the original artifact is never overwritten.

When the host does not have room for another multi-gigabyte package, retain
only the injected section and let the Android benchmark compose it in its
temporary device directory:

```powershell
python training/scripts/build_converter_random_topology_injection_parity.py `
  C:\path\to\random-target.litertlm `
  --model-type tf_lite_mtp_drafter `
  --output-dir C:\temp\random-mtp-section `
  --in-memory `
  --section-output C:\temp\random-mtp-section\random_mtp_drafter.tflite `
  --runtime-allocate --runtime-without-default-delegates

python training/scripts/benchmark_android_litertlm_gpu_parity.py `
  --official C:\path\to\official.litertlm `
  --candidate C:\path\to\random-target.litertlm `
  --candidate-section-patch C:\temp\random-mtp-section\random_mtp_drafter.tflite `
  --candidate-section-model-type tf_lite_mtp_drafter `
  --output-dir C:\temp\full-random-mtp-device-report `
  --mtp --output-tokens 8 --warm-runs 3
```

The benchmark stream-hashes the virtual full package on the host, stages the
base and section separately, patches exactly the selected byte range with
Android `toybox dd`, and requires the staged and post-run SHA-256 values to
equal the virtual host SHA. It never writes or overwrites a complete composite
package on the host. The alias-complete SM-F966B full-random run matched the official
target delegation counts (2,068 decode, 1,107 per prefill, 2,243 verify) and
the 198-node MTP drafter, all fully delegated in one partition across every
warm run. Schema-v5 median throughput was 17.5252 tok/s for the official package
and 13.1734 tok/s for the full-random candidate; median MTP acceptance was
0.266667 and 0.133333. Those weight-dependent speed and acceptance gates
correctly failed while repeated structural GPU parity passed. The target-only
schema-v5 alias-complete fixtures passed: 270M official/candidate medians were
53.2653/51.6543 tok/s (3.0245 percent regression), and E2B medians were
30.2335/29.9675 tok/s (0.8797 percent regression). No
training was performed for these topology checks.

### Merged checkpoint -> exact official topology

`build_checkpoint_official_topology.py` is the deployment path for the best
QAT+LoRA checkpoint once its numerical base passes the retained-constant gate.
It reuses the already verified converter-injection
mechanism, but its float branches come lazily from merged Hugging Face
safetensors instead of a random generator. The public AI Edge Quantizer emits
the W2/W4/W8 packed projection constants and per-axis scales; those constants
are then placed into an unchanged copy of the released target graph.

The command fails closed unless all of these are true:

- the input package SHA-256 matches the configured released artifact;
- merge metadata v4 binds the checkpoint to the canonical mobile model ID,
  exact reconstructed local source/manifest, training-config hash, effective
  merged-weight QAT run, hashed selected adapter, exact-key loading gate, and
  the SHA-256 of every merged safetensor shard and shard index;
- all 277 E2B or all 127 270M unique FC/embedding buffers map to floating-point
  checkpoint tensors with exact shapes;
- every one of those 277 or 127 official buffers has the same W2/W4/W8 QAT bit
  assignment in the selected training config; tied E2B `lm_head` is audited
  through its token-embedding source;
- for E2B, the retained-constant contract proves 262/262 exact public mobile
  values at the BF16 boundary and their compiled-buffer mapping, while the
  separate reconstruction manifest proves the complete 541-tensor mobile
  training seed; the public dense Q4 seed remains rejected at 50/262 exact
  retained tensors;
- the dedicated retained-scale exporter changes only the packed codes for the
  exact 205 trained projections using their bound qparams, preserves the
  official weight/A8 scale bytes and 72 frozen target constants, and emits a
  passing hash-bound exporter report;
- graph structure, operators, signatures, cache wiring, quantization layout,
  section size, and all bytes outside the target section remain official;
- for E2B, the default `tf_lite_mtp_drafter` SHA-256 remains byte-exact.

The final package is first written as an unpromoted `.partial` file. It is
renamed to the requested `.litertlm` only after every graph and package gate
passes. When all gates pass, this reproduces the official
graph/operators/layout with different trained weights; it does not claim
recovery of Google's private QAT trainer or calibration corpus. At present
this statement is device-proven for the 270M exact-base/random route and E2B
random topology experiments. A production E2B claim remains pending until an
actual trained, provenance-bound candidate passes quality plus both target-only
and MTP-on device gates; the Q4-seed/mobile-template hybrid is never eligible.

The preferred plan-only entry points are:

```powershell
python training/scripts/run_gemma4_e2b_mobile_mtp.py `
  --config training/configs/pipelines/gemma4_e2b_mobile_mtp.yaml `
  --training-config <run_root>/launch/resolved_training_config.yaml `
  --best-checkpoint <run_root>/best_golden_checkpoint `
  --base-litertlm <official.litertlm> `
  --merged-model-dir <fresh_export_root>/merged_best_hf `
  --exact-output-dir <fresh_export_root>/retained_scale_export `
  --export-report <fresh_export_root>/retained_scale_export/report.json `
  --output-litertlm <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm

python training/scripts/run_gemma270m_qat_litertlm.py `
  --config training/configs/pipelines/gemma3_270m_qat_litertlm.yaml `
  --official-litertlm C:\path\to\released-gemma-3-270m-it-q8.litertlm
```

After training has independently produced a Golden-best adapter, review the
plan, resolved-config identity, local checkpoint provenance, exact-205 qparams
binding and preflight reports. Only then add the two explicit execution flags
to the same E2B command. This command does not train. All output directories and
files must be fresh, and absence/failure of the exporter report blocks package
promotion. The 270M exact-base flow remains separate:

```powershell
python training/scripts/run_gemma4_e2b_mobile_mtp.py `
  --training-config <run_root>/launch/resolved_training_config.yaml `
  --best-checkpoint <run_root>/best_golden_checkpoint `
  --base-litertlm <official.litertlm> `
  --merged-model-dir <fresh_export_root>/merged_best_hf `
  --exact-output-dir <fresh_export_root>/retained_scale_export `
  --export-report <fresh_export_root>/retained_scale_export/report.json `
  --output-litertlm <fresh_export_root>/retained_scale_export/gemma4_e2b_retained_scale.litertlm `
  --execute-merge --execute-retained-scale-export

python training/scripts/run_gemma270m_qat_litertlm.py `
  --official-litertlm C:\path\to\released-gemma-3-270m-it-q8.litertlm `
  --execute-merge
python training/scripts/run_gemma270m_qat_litertlm.py `
  --official-litertlm C:\path\to\released-gemma-3-270m-it-q8.litertlm `
  --execute-exact-topology-export
```

For Gemma 4 `retained_mobile`, legacy
`--execute-exact-topology-export`, public abs-max export, and `--compose` are
hard-blocked migration paths. The first retained-scale package must preserve
the released MTP section byte-for-byte and remain MTP-off until target-only
semantic and Android GPU validation pass.

Use `--validate-android-gpu` only after the candidate exists. It invokes the
bounded parity runner on the connected device. E2B now requires two independent
schema-v5 reports: target-only must prove full GPU delegation and fixed-length
warm throughput within budget, then MTP-on must additionally prove draft
acceptance and speculative throughput. This prevents MTP acceptance from
masking target-graph speed. 270M requires the graph and fixed-length throughput
gates without MTP. Each report records the host SHA-256, verifies the staged
device SHA-256 before and after inference, binds the device-reported model path
and size to that artifact, and the pipeline validator re-hashes both the final
candidate and configured official reference before accepting the report.

The 270M QAT profile is intentionally `WI8/AFP32`: the released Q8 graph has
per-row INT8 weights (including the embedding table) but FLOAT32 activation
edges. Fake-quantizing 270M activations to INT8 would train for a graph that is
not the official runtime graph. Its base is pinned to
`google/gemma-3-270m-it`; changing to a base or FunctionGemma package requires
changing both the training model ID and the official package/hash together.
For the full E2B prefill, the automatic eight-weight batching can be overridden
with `--converter-batch-size N` when memory and FlatBuffer limits permit.
Run the same command against the Gemma 4 artifact with
`--model-type tf_lite_prefill_decode` for the main graph or
`--model-type tf_lite_mtp_drafter` for the draft graph. The same command covers
`tf_lite_embedder`, `tf_lite_per_layer_embedder`, `tf_lite_audio_encoder_hw`,
and `tf_lite_vision_encoder`; the adapter/end-marker sections contain no
quantized FC/embedding weight inventory.

The full E2B main-graph invocation used for the verified 35-batch run is:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/build_converter_random_topology_injection_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --model-type tf_lite_prefill_decode `
  --output-dir C:\temp\converter-injection-e2b `
  --calibration-samples 2 `
  --threads 1 `
  --in-memory `
  --runtime-allocate `
  --runtime-without-default-delegates
```

On the supplied E2B artifact this reports
`random_quantized_network.same=true`,
`converter_to_injected_quantization_values_match=true`,
`runtime_allocation_match=true`, `quantization_values_match=false`, and
`exact_official_model_match=false`.
The first field is the requested graph/layout network check; the latter two
must remain false because the fixture intentionally uses random constants.

If the goal is only to exercise a random quantized network with the *official*
MTP topology, `build_random_official_topology_parity.py` can patch the 23
serialized W4/W8 buffers in place while preserving every operator, tensor,
signature, and quantization-layout field. The verified MTP run randomized all
23 buffers (13 W4, 10 W8), round-tripped every low-bit buffer, and returned
`graph_structure_match=true` and `quantization_layout_match=true`. That is an
artifact-preserving regression fixture, not an independent reconstruction of
the private exporter or QAT recipe.

### Public mobile checkpoint constant parity

When the released public checkpoint is available, use
`audit_gemma4_mobile_checkpoint_parity.py` to compare its serialized constants
and scales with the supplied Gemma 4 LiteRT-LM sections:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/audit_gemma4_mobile_checkpoint_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --source-safetensors C:\path\to\gemma4-mobile\model.safetensors `
  --include-embedders `
  --strict `
  --output C:\temp\gemma4-mobile-checkpoint-parity.json
```

The audit is read-only and memory-maps the safetensors file. The public mobile
checkpoint stores W2/W4 values as unsigned offset codes; the LiteRT buffers
use the corresponding signed low-bit codes. The verified conversion is
`code - 2 (mod 4)` for W2, `code - 8 (mod 16)` for W4, and byte-for-byte copy
for W8. With that serialization mapping, the local public checkpoint matched
1/1 token-embedding buffers, 35/35 per-layer-embedding tables, and 276/276
comparable prefill/decode buffers (all scales exact). One W8
`per_layer_model_projection` is BF16 in the public checkpoint and has no source
scale, so it is reported as unresolved rather than guessed. Exact comparable
constants are strong evidence that the released checkpoint supplies the
artifact's observable mobile weights; they do not expose Google's private QAT
loss, calibration data, optimizer schedule, or exporter/package source.

The retained FLOAT32 graph constants can be audited separately with
`audit_gemma4_mobile_retained_compiled_parity.py`. Against the exact released
package SHA-256, semantic consumer mapping resolved 262/262 tensors across 242
unique graph buffers, and every compiled FLOAT32 value rounded exactly to the
public checkpoint's BF16 bytes using round-to-nearest-even. This closes the
compiled-buffer mapping gap at public precision. It does not identify the lower
FLOAT32 mantissa bits that BF16 serialization discarded and does not make the
dense Q4-QAT checkpoint numerically compatible with the mobile release.

For the MTP drafter, `audit_gemma4_mtp_assistant_parity.py` maps the supplied
23-record traversal to the public four-layer assistant safetensors and applies
the public symmetric per-output-channel min/max candidate. Both the generic
assistant and the QAT assistant have all 23 expected shapes, but each produces
0/23 exact packed buffers and 0/23 exact scale vectors for the supplied MTP
artifact. This negative result isolates the remaining private/other-source
boundary; it does not invalidate the public assistant for Transformers
speculative decoding.

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/audit_gemma4_mtp_assistant_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  C:\path\to\assistant\model.safetensors `
  --output C:\temp\gemma4-mtp-assistant-parity.json
```

The mobile-transformers checkpoint leaves the E2B per-layer model projection
as BF16 and omits its W8 scale. Test the public source against the official W8
projection (without downloading the full remote file) with:

```powershell
$env:PYTHONPATH = "training/src;training/scripts;<tflite-install-location>"
python training/scripts/audit_gemma4_mobile_projection_parity.py `
  C:\path\to\gemma-4-E2B-it.litertlm `
  --source https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/main/model.safetensors `
  --output C:\temp\gemma4-mobile-projection-parity.json
```

The current public BF16 projection fails every tested symmetric W8 scale/code
candidate; keep that result separate from the 276/276 exact packed constants.
The same audit checks BF16 rounding-cell feasibility: the dated official
section had 8,960/8,960 scales and 13,762,560/13,762,560 codes compatible with
some hidden pre-BF16 values. This supports (but does not prove) max-abs/127
before BF16 serialization and cannot recover the missing source precision. The
JSON field is `bfloat16_interval_report`.
The compressed-tensors mobile checkpoint is a schema/packing cross-check, not
an exact LiteRT-LM source because its scales are BF16 and the projection is
ignored.

The public exporter boundary is also explicit: the LiteRT-Torch maintainer
states in [issue #1044](https://github.com/google-ai-edge/litert-torch/issues/1044)
that QAT checkpoint conversion is unsupported, and merged
[PR #1050](https://github.com/google-ai-edge/litert-torch/pull/1050) makes that
failure explicit. Do not treat `export_hf` as Google's packed mobile-QAT
importer or exact MTP/audio package exporter.

For the official 270M package, compare selected source tensors without writing
the gated 536 MB checkpoint locally:

```powershell
$env:PYTHONPATH = "<tflite-install-location>"
python training/scripts/audit_litertlm_remote_source_recipe.py `
  C:\path\to\official-gemma3-270m-it-q8.litertlm `
  --source-url https://huggingface.co/<verified-public-mirror>/resolve/main/model.safetensors `
  --model-type TF_LITE_PREFILL_DECODE `
  --max-weights 10
```

When the source checkpoint is already available, replace `--source-url` with
`--source-path C:\path\to\model.safetensors`; the audit memory-maps the file
and does not copy it.

The range audit uses the canonical seven-linear Gemma 3 layer order, checks
axis-0 INT8 bytes and scales, and reports whether source-precision/rounding
differences remain. The full official 270M section comparison matched all 127
unique weights' scales to `max(abs(BF16 row))/127`; 31,142 of 268,042,240
serialized INT8 values differed (0.0116183%), with a maximum error of one and
all differences at BF16 half-step boundaries. This proves the public scale
construction for the compared mirror bytes, not Google source provenance or
its private QAT schedule.

For Slurm machines, submit the repo-owned sbatch entrypoint instead of writing an
ad-hoc script:

```bash
sbatch training/scripts/slurm_train_gemma4_e2b.sbatch
```

The sbatch file runs the training command through `srun`, activates
`/home/k_anup/gemma4_env` by default, and requests four GPUs. The training
entrypoint uses all queryable GPUs visible to the process by default. The
training runner also downgrades `bfloat16` to `float16` automatically when the
visible GPU/PyTorch setup does not support BF16. By default, SFT training fails
fast when PyTorch cannot see CUDA, because otherwise the job silently runs on CPU
and appears stuck even when `nvidia-smi` shows idle GPUs. Override paths without
editing the file:

```bash
sbatch --export=ALL,A2UI_REPO_DIR=/home/k_anup/code/GenUI,A2UI_VENV=/home/k_anup/gemma4_env training/scripts/slurm_train_gemma4_e2b.sbatch
```

If a Slurm job fails, inspect both Slurm state and the training log:

```bash
sacct -j <jobid> -o JobID,JobName%30,State,ExitCode,DerivedExitCode,Elapsed,Timelimit,MaxRSS,ReqMem,NodeList,Reason -P
scontrol show job -dd <jobid>
tail -200 training/logs/slurm-<jobid>.err
tail -200 training/logs/slurm-<jobid>.out
```

If `nvidia-smi` shows GPUs but only Xorg/display processes and `0%` utilization,
verify PyTorch from the same environment/container:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

If that reports `cuda available: False`, fix the environment rather than waiting
for training: install a CUDA-enabled PyTorch build and launch Singularity with
GPU passthrough, for example `singularity exec --nv <image> ...`.

The training entrypoint normalizes `CUDA_VISIBLE_DEVICES` before importing
PyTorch. If the shell inherits a multi-GPU value such as `0,1,2,3`, it keeps all
GPUs that `nvidia-smi --query-gpu=index` can query. To use a specific healthy
set, pass it explicitly:

```bash
A2UI_CUDA_VISIBLE_DEVICES=0,1,3 python3 training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_lora.yaml
```

To exclude a known bad device without hardcoding the full set:

```bash
A2UI_EXCLUDE_CUDA_DEVICES=2 python3 training/scripts/train_sft.py --config training/configs/models/gemma4_e2b_ir_lora.yaml
```

For the HF Trainer backend, labels are completion-only: prompt tokens are masked
with `-100`, completion/IR tokens are left trainable, and the preflight fails if
a batch would have zero trainable labels. This prevents apparent zero-gradient
runs caused by truncating away the IR completion.

Shell and sbatch files are forced to LF line endings through `.gitattributes`.
This avoids Linux shebang failures such as `cannot execute: required file not
found` caused by CRLF files copied from Windows.

3. Evaluate generated IR against the flat-spec contract and existing UI metrics.

```powershell
python training/scripts/evaluate.py --predictions training/outputs/eval/predictions.jsonl
```

4. Evaluate with dataset-compatible overall score.

```powershell
python training/scripts/evaluate.py --predictions training/outputs/eval/predictions.jsonl --weights-config dataset/configs/run.yaml --baseline-aggregate dataset/data/runs/<baseline>/aggregates.json
```

The evaluator writes `aggregate_metrics.json` with `overall_score`, `baseline_overall_score`, and `overall_score_delta_vs_baseline` when a baseline is provided.

5. Export a baseline/non-retained Gemma 4 E2B model for Google AI Edge Gallery.

Google AI Edge Gallery imports local LLMs as `.litertlm` files. After training,
merge the LoRA adapter into a Hugging Face model directory and run Google's
LiteRT Torch Hugging Face exporter:

```powershell
python -m pip install -r training/requirements-edge-export-tested.txt
hf auth login
python training/scripts/export_edge_gallery_model.py --config training/configs/export/edge_gallery_gemma4_e2b.yaml --merge-lora
adb push training/outputs/export/gemma4_e2b_ir_edge_gallery/litertlm/<model>.litertlm /sdcard/Download/
```

This generic exporter is for baseline or non-retained-scale checkpoints. Do
not use it for a `retained_mobile` E2B checkpoint or as a substitute for the
official-format 205-projection code-only exporter used by the multiformat/mobile
pipeline.

`requirements-edge-export-tested.txt` pins the conversion environment used by
the recorded graph rebuilds and Android GPU parity runs. The looser
`requirements-edge-export.txt` is suitable for experimentation, but a version
change is unverified until the numerical, topology, package, and device gates
are rerun.

For CI or CPU-only machines, validate the generated command/manifest without
running conversion:

```powershell
python training/scripts/export_edge_gallery_model.py --config training/configs/export/edge_gallery_gemma4_e2b.yaml --dry-run
```

The export wrapper writes `edge_gallery_export_plan.json`,
`run_litert_export.ps1`, `EDGE_GALLERY_IMPORT.md`, and `model_manifest.json`
under `training/outputs/export/gemma4_e2b_ir_edge_gallery`.

6. Export a legacy Android metadata package.

```powershell
python training/scripts/export_model.py --config training/configs/export/litertlm_gemma.yaml
python training/scripts/package_android_model.py --export-dir training/outputs/export/gemma_e2b_ir_litertlm
```

## Architecture

The training code is intentionally model-agnostic:

- Dataset preparation writes neutral JSONL records with `messages`, `prompt`, and `completion`.
- Model-specific formatting is isolated in `ir_training.models.ModelAdapter` implementations.
- Export targets are registry-driven so future formats can be added without changing data prep or metrics.

Initial target: Gemma E2B-family LoRA/QLoRA SFT. Future adapters can be added for Qwen, Llama, Phi, or any Hugging Face causal LM.

## Output Policy

Generated training artifacts are written under `training/outputs/`, `training/runs/`, or `training/checkpoints/`. These folders should stay uncommitted unless a small manifest/report is intentionally needed for review.
