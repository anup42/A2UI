# Official mobile retained-scale QAT pipeline

This runbook defines a separate pipeline for training the official Gemma 4 E2B
mobile seed, selecting one checkpoint on Golden32, evaluating that locked
checkpoint on Golden32, Golden35, and Bixby50, and exporting it into the exact
released mixed-precision LiteRT-LM topology. MTP inference stays disabled. The
released MTP section remains in the package byte-for-byte but is not trained,
replaced, or used by the runtime gates in this workflow.

The entry point is `training/scripts/run_official_mobile_pipeline.py`. It is
plan-only unless `--execute` is passed. Every execution requires a fresh
`--output-dir`; do not overwrite or silently resume an earlier run. Output must
be outside the model and input directories, and must not contain them.

## What this lane is, and is not

The training source is not a normal dense Gemma checkpoint and it is not the
Q4_0 unquantized checkpoint. `--model-dir` must be the manifest-verified BF16
text reconstruction of
`google/gemma-4-E2B-it-qat-mobile-transformers`, created from the exact packed
mobile `model.safetensors`. The reconstruction preserves the public mobile
cell centers and binds the retained quantization parameters needed by the
exporter.

The final target is the released mobile `wNa8o8` layout:

- 277 target constants in the official graph;
- 205 trainable projections: 60 W2 and 145 W4;
- 72 frozen constants: one W2 and 71 W8;
- the official static A8 input/output quantization parameters, including all
  1,220 mutable fully-connected alias-edge records;
- unchanged section order, graph structure, quantization layout, execution
  contract, package metadata, frozen buffers, and MTP section.

The merged BF16 Hugging Face checkpoint is only an export intermediate. It is
not the mobile artifact. The generic LiteRT Torch/absolute-max conversion path
must not be used for this lane because it recomputes scales and cannot establish
retained-mobile parity.

This workflow does not reproduce Google's private QAT data, calibration,
observer, optimizer, or exporter recipe. It also does not promise a speedup.
Training quality, exact package structure, Android GPU delegation, and measured
throughput are separate gates.

## Required inputs

Before planning a run, have all of the following locally:

1. A repository revision containing the wrapper and its supporting retained-scale
   exporter changes.
2. A Python environment installed from
   `training/requirements-gemma4-qat.txt`. That overlay includes the normal
   training requirements and raises the Transformers and PEFT minimum versions
   required by this workflow.
3. CUDA-enabled PyTorch and BF16-capable GPUs. `--devices auto` selects all
   scheduler/CUDA-visible devices. During `configure`, the wrapper detects the
   visible inventory, resolves a GPU profile, and records the selected ranks,
   batch settings, and data-worker count. Preflight verifies BF16 support. The
   training stage launches the existing DDP/torchrun command.
4. The packed mobile checkpoint files from the pinned
   `google/gemma-4-E2B-it-qat-mobile-transformers` revision
   `dd693ff40353f057ca5f07e945ad867f4afbf2ec`:
   - `model.safetensors` SHA-256
     `efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4`
   - `config.json` SHA-256
     `cf6d7dc22738b5e6beb364bac833d78b869f5a6ffd57dfc96c6be3f2abc80424`
5. The exact audited official Gemma 4 E2B mobile `.litertlm` package, SHA-256
   `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`.
   This repository does not download or infer that package.
6. A manifest-verified reconstructed mobile seed directory containing
   `config.json`, the complete Safetensors shard set,
   `mobile_training_seed_manifest.json`, `mobile_qparams.json`, and its retained
   scale sidecar. It also needs local tokenizer assets because execution forces
   Hugging Face and Transformers offline: at minimum `tokenizer.json` and
   `tokenizer_config.json`; keep `chat_template.jinja` and
   `generation_config.json` with them when supplied by the pinned checkpoint.
   The reconstruction script copies these files only when they are beside the
   source `config.json`.
7. A source-bound `--input-dir` containing `train.jsonl` and `val.jsonl`.
   Golden32, Golden35, and Bixby50 are repository-pinned evaluation artifacts;
   they must remain excluded from train and validation data.
8. Enough fresh disk space for the reconstructed seed, adapter checkpoints,
   merged BF16 model, exported package, evaluation outputs, and immutable logs.

The official public source anchors are the
[mobile checkpoint card](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers),
its [released config](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/blob/main/config.json),
and [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM). These sources
describe public artifacts and runtime support; they do not publish the private
training/export recipe used to create the release.

On the Linux/H100 host, create the environment before running the pipeline:

```bash
python3 -m venv .venv-gemma4-qat
source .venv-gemma4-qat/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/requirements-gemma4-qat.txt
```

Keep the CPU LiteRT/schema dependencies isolated from that CUDA environment.
The repository's pinned setup is:

```bash
python3.12 -m venv /absolute/path/a2ui-export-094
EXPORT_PY=/absolute/path/a2ui-export-094/bin/python
"$EXPORT_PY" -m pip install --upgrade pip
"$EXPORT_PY" -m pip install torch==2.11.0 torchao==0.17.0 \
  --index-url https://download.pytorch.org/whl/cpu
"$EXPORT_PY" -m pip install -r training/requirements-deployment-export.txt
"$EXPORT_PY" -m pip check
```

Pass the absolute virtual-environment executable as
`--exporter-python "$EXPORT_PY"`; do not resolve its symlink to the system
Python. Although this lane does not invoke the generic converter, the retained
exporter imports the pinned LiteRT/FlatBuffers schema stack from that
environment. Before training, the `assets` stage checks both exporter entry-point
imports, a tiny FlatBuffer serialization/graph-inspection roundtrip, and complete
static decoding of the actual official target graph in that interpreter.
It records graph hashes, executable and dependency versions in
`<fresh-run-root>/logs/export_environment.json` and `.log`. It does not instantiate
or convert the model or run its kernels, so runtime compatibility is still a
separate gate.

## Build the retained-compiled report and reconstructed seed

The retained-compiled report is generated locally; it is not a checked-in model
file or an artifact to invent by hand. Generate it from the same exact official
`.litertlm` package and packed mobile Safetensors that will be bound to the run:

```powershell
python training/scripts/audit_gemma4_mobile_retained_compiled_parity.py `
  <official-mobile.litertlm> `
  --source-safetensors <packed-mobile>/model.safetensors `
  --official-artifact-sha256 181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c `
  --output <evidence>/gemma4-mobile-retained-compiled-parity.json
```

Inspect the report and require its verification checks to pass. Reconstruction
uses the pinned canonical, path-independent report digest
`6744af4144688b93ef7009a67218700fcaa2b78b4186872496a8e17cd72e4008`.
This is not the raw file SHA-256: canonicalization removes only the machine-local
`artifact.path` and `source.path` fields and binds every semantic field. Omit
`--retained-compiled-report-sha256` to use that strict pinned default:

```powershell
python training/scripts/reconstruct_gemma4_mobile_training_seed.py `
  --source-safetensors <packed-mobile>/model.safetensors `
  --source-config <packed-mobile>/config.json `
  --retained-compiled-report <evidence>/gemma4-mobile-retained-compiled-parity.json `
  --output-dir <mobile-seed> `
  --plan-output <evidence>/mobile-seed-plan.json
```

Linux/H100 equivalents are:

```bash
python training/scripts/audit_gemma4_mobile_retained_compiled_parity.py \
  /absolute/path/official-mobile.litertlm \
  --source-safetensors /absolute/path/packed-mobile/model.safetensors \
  --official-artifact-sha256 181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c \
  --output /absolute/path/evidence/gemma4-mobile-retained-compiled-parity.json

python training/scripts/reconstruct_gemma4_mobile_training_seed.py \
  --source-safetensors /absolute/path/packed-mobile/model.safetensors \
  --source-config /absolute/path/packed-mobile/config.json \
  --retained-compiled-report /absolute/path/evidence/gemma4-mobile-retained-compiled-parity.json \
  --output-dir /absolute/path/mobile-seed \
  --plan-output /absolute/path/evidence/mobile-seed-plan.json
```

Each reconstruction command shown above is plan-only. Review its output, then
repeat the exact command with `--execute`. Never reconstruct over a non-empty
destination. The resulting manifest must verify the complete 541-tensor transformation: 263
direct BF16 tensors and 278 tensors dequantized from the public packed mobile
checkpoint.

## Plan the complete run

The default schedule is two epochs, learning rate `1e-5`, evaluation
every 500 optimizer updates, Golden32 generation every 1,000 updates, and a
2,048-token generation cap. Golden32 also runs at training end. If a smoke-run
`--steps` cap is shorter than `--eval-steps`, validation runs at that cap and
Golden's periodic interval rounds up to the next validation boundary, never
earlier than requested. The requested and resolved cadences are both recorded
in the training YAML; the end-of-training Golden evaluation still selects a
checkpoint. Plan it first:

```powershell
python training/scripts/run_official_mobile_pipeline.py `
  --model-dir <mobile-seed> `
  --source-safetensors <packed-mobile>/model.safetensors `
  --official-litertlm <official-mobile.litertlm> `
  --input-dir <source-bound-train-val> `
  --output-dir <fresh-run-root> `
  --devices auto `
  --epochs 2 `
  --learning-rate 1e-5 `
  --eval-steps 500 `
  --golden-every-steps 1000 `
  --max-new-tokens 2048
```

Use `--exporter-python <python-or-python.exe>` when the retained-scale exporter
runs in a separate environment. The resolved executable path is stored in
`official_mobile_plan.json`. When the option is omitted, the wrapper uses its
current Python interpreter. The `assets` stage verifies the required exporter
imports and schema APIs before any training begins.

For H100 80 GB hosts, the default GPU profiles retain a global batch of 32:

| Visible GPUs | Microbatch per GPU | Gradient accumulation | Effective batch |
| --- | --- | --- | --- |
| 2 | 1 | 16 | 32 |
| 4 | 1 | 8 | 32 |
| 8 | 1 | 4 | 32 |

One rank runs on each selected GPU. BF16, non-reentrant gradient checkpointing,
bounded CPU/data workers, and the existing safe SDPA configuration are retained.
Standalone checkpoint tests distribute cases across the selected GPUs; they run
after training so they do not contend with training ranks. CPU preparation/token
caches are reused only when their complete input contracts match. These are safe
starting profiles, not a claim of measured optimal throughput on every host.

Review the printed JSON plan, including resolved paths, stage order, holdout
roles, requested hyperparameters, export layout, and fresh output destinations.
Plan-only mode performs local path/config checks but does not probe CUDA, hash
the large official inputs, load model weights, train, merge, export, or touch a
device. Artifact identities and GPU capability are checked during execution.

## Execute

After the plan passes, repeat the same command with `--execute`. Do not change
input paths or hyperparameters between plan review and execution:

```powershell
python training/scripts/run_official_mobile_pipeline.py `
  --model-dir <mobile-seed> `
  --source-safetensors <packed-mobile>/model.safetensors `
  --official-litertlm <official-mobile.litertlm> `
  --input-dir <source-bound-train-val> `
  --output-dir <fresh-run-root> `
  --devices auto `
  --epochs 2 `
  --learning-rate 1e-5 `
  --eval-steps 500 `
  --golden-every-steps 1000 `
  --max-new-tokens 2048 `
  --execute
```

Linux/H100 command:

```bash
python training/scripts/run_official_mobile_pipeline.py \
  --model-dir /absolute/path/mobile-seed \
  --source-safetensors /absolute/path/packed-mobile/model.safetensors \
  --official-litertlm /absolute/path/official-mobile.litertlm \
  --input-dir /absolute/path/source-bound-train-val \
  --output-dir /absolute/path/fresh-run-root \
  --exporter-python "$EXPORT_PY" \
  --devices auto \
  --epochs 2 \
  --learning-rate 1e-5 \
  --eval-steps 500 \
  --golden-every-steps 1000 \
  --max-new-tokens 2048 \
  --execute
```

The wrapper owns a bounded, fail-closed stage graph:

1. Validate the packed source hash, reconstructed-seed manifest, retained scale
   contract, model architecture, input data, holdout exclusion, prompt/tokenizer
   bindings, CUDA/BF16 environment, and zero-adapter numeric/greedy parity.
2. Train retained-scale effective-weight LoRA QAT from the reconstructed mobile
   seed using DDP and immutable published weight/A8 scales.
3. Evaluate Golden32 at the configured cadence and save the callback-created
   `best_golden_checkpoint` only when the metric improves.
4. Lock checkpoint selection using
   `unique_source_generation_reward_v5_4_avg`. Golden32 contains 32 occurrences
   but 31 unique sources, so raw occurrence-weighted reward is diagnostic and
   must not select the checkpoint.
5. Evaluate the locked best checkpoint on the complete Golden32, Golden35, and
   Bixby50 cohorts. Golden35 and Bixby50 are final-only holdouts and cannot
   change checkpoint or hyperparameter selection.
6. Merge that exact adapter into its bound reconstructed BF16 seed.
7. Re-encode exactly the 205 trained W2/W4 projection code buffers using the
   retained official scales, then patch a copy of the official package.
8. Revalidate provenance, code/scale parity, frozen buffers, package bytes,
   graph/layout/execution-contract identity, and the preserved MTP section
   before atomically publishing the final `.litertlm`.

Every stage runs in a bounded worker subprocess. Before returning, the
worker writes `stage_receipts/<stage>.json`, binding the plan SHA-256 and hashes
of every declared stage artifact; the parent verifies it before starting the next
stage. A failed or interrupted stage remains evidence. The workflow has no
resume mode: use a new output directory rather than substituting an intermediate
trainer checkpoint or copying one into `best_golden_checkpoint`.

## Output layout

For run directory `<run>` whose final path component is `<run-id>`, the wrapper
writes:

- `<run>/official_mobile_plan.json` — immutable resolved orchestration plan;
- `<run>/official_mobile_manifest.json` — live/final stage state and bound
  receipts;
- `<run>/stage_receipts/<stage>.json` — per-stage plan binding and artifact
  hashes;
- `<run>/logs/<stage>.log` and stage-specific worker logs;
- `<run>/mobile_assets_verified.json` and `<run>/data_audit.json`;
- `<run>/prepared/{train,val,golden32,golden35,bixby50}.jsonl` plus preparation
  manifests;
- `<run>/configs/mobile_training.yaml` and
  `<run>/configs/mobile_deployment.yaml`;
- `<run>/training/<run-id>/launch/{resolved_training_config.yaml,launch_plan.json,preflight_report.json,training.log}`;
- `<run>/training/<run-id>/best_golden_checkpoint/`;
- `<run>/evaluations/best_{golden32,golden35,bixby50}/` with predictions,
  scored predictions, aggregate metrics, and `evaluation_result.json`;
- `<run>/merged_best_hf/`;
- `<run>/retained_scale_export/gemma4_e2b_a2ui_mobile.litertlm` and
  `gemma4_retained_scale_code_only_report.json`;
- `<run>/benchmark_command.json`, which is written even when device benchmarking
  is not requested;
- `<run>/android_gpu/target_only/android_litertlm_gpu_parity_report.json` when
  `--benchmark-android` runs successfully;
- `<run>/results.md`, written on success or failure with only verified completed
  results.

TensorBoard data is written under `<tensorboard-root>/<run-id>/`. Training uses
the `training/` subdirectory, while standalone best-checkpoint evaluations add
`evaluation_records/best_<cohort>/` JSON records and evaluation scalar tags to
the same run ID.

## Evaluation contract

- Golden32: exactly 32 rows and 31 unique sources. It is the only checkpoint
  selection cohort. Require strict A2UI Express scoring and the unique-source
  v5.4 reward.
- Golden35: exactly 35 unique rows. It is evaluated only after the Golden32
  winner is locked and is never used for selection.
- Bixby50: exactly 50 unique source-only rows. It has no reference IR. Score
  source/response faithfulness and structural validity; never report a missing
  reference match as zero and never use Bixby50 for selection.

All three best-checkpoint evaluations must complete at their exact row counts.
Retain predictions, scored predictions, aggregate metrics, raw/stopped decode
diagnostics, prompt/tokenizer hashes, TensorBoard events, and a final scorecard.
Do not repair generated IDs or malformed model output during capability scoring.

## Export and MTP-off contract

The retained-scale export is valid only when all of these are proven:

- the adapter is the callback-created Golden32 best checkpoint and its bytes,
  resolved training config, seed manifest, qparams contract, and numeric
  preflights are self-bound;
- the dense merge uses that exact adapter and reconstructed seed;
- all 205 adapter A/B pairs and merged projections map bijectively to the
  official graph, with 60 W2 and 145 W4 mutable buffers;
- zero-adapter serialization reproduces the official target section
  byte-for-byte;
- the 72 frozen target constants, all weight scales, all A8 qparams, package
  prefix/suffix, metadata, and the complete MTP section remain byte-exact;
- decoded structural, quantization-layout, and execution-contract hashes match
  the official target and both execution contracts are completely decoded;
- at least one trained code changes, while restoring the 205 selected payloads
  reconstructs the official target section byte-for-byte.

The generated deployment config represents MTP as runtime-disabled,
official-preserved, and unused.
The package intentionally still contains `tf_lite_mtp_drafter`; absence of the
section would change the official topology. Do not run drafter training, inject
assistant weights, pass an MTP runtime flag, or make an MTP acceptance/speed
claim in this workflow.

Graph equivalence is a potential promotion blocker. The target weight bytes are
expected to differ, so full target-section hash equality is neither possible nor
the right test. Promotion instead requires complete graph decoding plus equality
of the structural, quantization-layout, and execution-contract hashes. If the
inspector cannot decode an execution contract completely, or any graph/layout
hash changes, stop; package parseability and matching file size are insufficient.

## Optional Android target-only benchmark

Android work is opt-in and occurs only after the host export and structural
gates pass. `--benchmark-android` always requires an explicit
`--serial <adb-serial>`, even when only one device is connected:

```powershell
python training/scripts/run_official_mobile_pipeline.py `
  --model-dir <mobile-seed> `
  --source-safetensors <packed-mobile>/model.safetensors `
  --official-litertlm <official-mobile.litertlm> `
  --input-dir <source-bound-train-val> `
  --output-dir <fresh-run-root> `
  --devices auto `
  --epochs 2 `
  --learning-rate 1e-5 `
  --eval-steps 500 `
  --golden-every-steps 1000 `
  --max-new-tokens 2048 `
  --execute `
  --benchmark-android `
  --serial <adb-serial>
```

The wrapper writes the exact standalone Android command to
`<run>/benchmark_command.json` so the benchmark can be rerun independently. The
selected app and instrumentation test must already be installed on the target
device; this wrapper does not build or install them. The target-only comparison uses identical
prompt, sampler, token limits, and alternating official/candidate order, with
one cold run and at least three valid warm runs per artifact. It requires host
and device SHA-256 identity, successful instrumentation, full `LITERT_CL`
delegation, matching graph/signature shapes, reaching the requested decode cap,
and no more than the configured throughput regression. MTP stays false.

Full GPU delegation proves that the graph ran through the Android GPU delegate;
it does not prove A2UI semantic quality. Conversely, the Golden scores do not
prove Android delegation or speed. A target-only trained-Express semantic probe
must separately establish strict decode, canonical validation, response-fact
coverage, and native rendering before a deployment claim.

## Promotion checklist

A run is eligible for review only when:

- every input and output identity is hash-bound;
- every preflight and bounded stage passed;
- the Golden32-selected best checkpoint completed all three exact cohorts;
- Golden35 and Bixby50 were not used for selection;
- the retained-scale exporter report passed every gate and the final package
  was atomically published to a fresh path;
- the official MTP section is byte-exact and runtime MTP is disabled;
- host graph-equivalence validation passed;
- any claimed Android GPU or throughput result has its own passing device
  report.

Until those conditions are met, report the result as training, evaluation, or
serialization evidence only—not a production-ready mobile model.

## Local verification (2026-09-19)

- Regression run across the new runner, portable launcher, retained exporter,
  merge/QAT provenance, existing mobile/multiformat workflows, Golden/Bixby
  preparation and scoring, GPU profiles, and parallel generation: **295 passed,
  2 skipped**.
- After the final early-exporter graph probe was added, the affected runner,
  retained exporter and merge/QAT subset passed again: **66 passed, 1 skipped**.
- The skips were the Windows executable-symlink test (host permission) and the
  optional PEFT adapter worker smoke test (dependency unavailable). A separate
  platform-independent test still verifies that the exporter executable is
  never realpath-resolved out of its virtual environment.
- CPU integration used real preparation of all three pinned cohorts and real
  launch/evaluation bindings with tiny training fixtures. GPU inventory and
  the tiny fixture packed-file hash were mocked; no quality result is implied.
- Python compilation, CLI help and whitespace checks passed. The local schema
  roundtrip ran without loading model weights.

No full H100 training, retained-scale model export, or Android speed benchmark
was executed for this change. Run-time scores and throughput remain unmeasured;
the pipeline's real artifact, numeric and device gates must still pass on the
training/deployment hosts.
