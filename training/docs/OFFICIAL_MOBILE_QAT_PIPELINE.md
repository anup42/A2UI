# Official mobile retained-scale QAT pipeline

This runbook defines a separate pipeline for training the official Gemma 4 E2B
mobile seed, selecting one checkpoint on Golden32, evaluating that locked
checkpoint on Golden32, Golden35, and Bixby50, and exporting it into the exact
released mixed-precision LiteRT-LM topology. MTP inference stays disabled. The
released MTP section remains in the package byte-for-byte but is not trained,
replaced, or used by the runtime gates in this workflow.

The entry point is `training/scripts/run_official_mobile_pipeline.py`. It is
plan-only unless `--execute` is passed. Every execution requires a fresh
`--output-dir`; do not overwrite an earlier run. Fresh runs refuse resume by
default. A continuation is available only through the explicit
`--resume-from-checkpoint` option described below, and it also writes to a new
output directory. Output must be outside the model and input directories, and
must not contain them.

For Muse-augmented data, prefer the [standalone preprocessing runbook](SEMANTIC_AUGMENTATION.md):
generation is an independent command that publishes an E2B tokenizer-bound
folder. A training invocation keeps its original `--input-dir` and adds
`--augmentation --augmentation-dir /absolute/path/to/augmentation-run/augmented`.
The folder already includes original training rows; the consumer verifies
the matching base data, hashes, tokenizer, shared prompt and token limits, then
uses the combined dataset once without appending originals again. Training
never contacts Muse or starts/stops its server. Semantic augmentation without
the folder fails. The official seed/export/preflight requirements still apply.

Explicit whole-bundle `--prepared-input-dir` reuse with `--augmentation none`
remains available as an alternative, mutually exclusive with raw input and
`--augmentation-dir`; it is not the preferred original-data-plus-augmentation
interface.

The current workflow identifier is
`e2b_retained_mobile_golden_bixby_no_mtp_v2`. Existing plan and execute commands
remain unchanged, but v2 must start in a fresh output directory because it adds
new bound stages and evidence.

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

### Training numeric contract

The official-mobile YAML explicitly opts into
`activation_quantizer: gemma_mobile_srq`. Its forward pass follows the
[public Hugging Face `apply_srq` implementation](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/integrations/gemma_quant.py)
shape: convert the retained scalar scale to the activation dtype, compute
`round(x / scale)`, clamp to the signed A8 range `[-128, 127]`, multiply by the
same scale, and bypass quantization when the scale is zero. The backward pass is
this repository's configured straight-through estimator (STE), not evidence of
Google's private training recipe or observer behavior.

This opt-in also applies retained input/output A8 simulation to the inventoried
70 frozen per-layer W8 linear modules without fake-quantizing their frozen
weights. The 205 mutable W2/W4 projections use effective merged-weight LoRA QAT.
The run fails closed unless the exact 205-projection scope is covered and every
expected LoRA A/B tensor is actually trainable. Legacy configs that omit
`activation_quantizer: gemma_mobile_srq`, frozen activation simulation, and the
strict trainable-scope requirement retain their previous behavior.

### Numeric preflight: safety gates versus BF16 diagnostics

The v2 wrapper writes `preflight.numeric_policy: retained_mobile_safety_v1`
into the generated, hash-bound training config. This policy is accepted **only**
with `run.purpose: e2b_retained_mobile_golden_bixby_no_mtp_v2` and the explicit
retained-scale/SRQ/205-mutable/70-frozen contract. Missing or unknown policies,
using the policy in another workflow, or disabling mandatory gates fail closed.
Historical standalone configurations keep their existing BF16 parity behavior.

The BF16 reconstruction is not the same forward computation as QAT-on: the
latter applies retained W2/W4 effective-weight fake quantization, A8 SRQ, and
frozen W8-path activation simulation. Near-identical BF16/QAT outputs are not a
valid universal prerequisite for this workflow.

| Check | Official v2 behavior |
| --- | --- |
| BF16/QAT top-1 match, top-token hash equality | Diagnostic only |
| BF16/QAT completion-loss increase and ratio | Diagnostic only |
| BF16/QAT greedy common prefix and positional match | Diagnostic only |
| Finite logits and completion losses on both paths | Mandatory |
| Absolute QAT completion loss | Mandatory; default maximum 15.0, finite positive limit no greater than 15 |
| Nonempty comparable numeric probes and valid greedy probes | Mandatory |
| Identical repeated QAT-on greedy generation | Mandatory; checked from actual token sequences |
| Exactly zero fresh LoRA delta, exact 205 trainable projections and 70 frozen activation modules | Mandatory; existing scope checks retained |
| Published qparams, artifact hashes, no-op/package identity and export provenance | Mandatory; unchanged |

The previous `0.90` top-1, `0.35` loss-increase, `1.10` loss-ratio, and eight-token
cross-mode-prefix references are not lowered. They remain visible in diagnostics
but do not gate this explicitly selected policy. For example, the reported H100
losses `1.32057861` (BF16) and `0.62379508` (QAT), with top-1 match `0.742188`, no
longer fail *just because* their top-1 match is below 90%. All mandatory checks
must still succeed; this does not certify model quality or device throughput.

The console prints the policy and comparison summary. The numeric JSON report in
`<run>/training/<run-id>/launch/preflight/model_numeric_preflight.log` and
checkpoint `training_metadata.json` retain all numeric/greedy comparison metrics,
raw probe evidence, named mandatory checks, and policy identity. The launcher's
aggregate `preflight_report.json` hash-binds that log. Merge and export
independently validate that evidence against the hash-bound config, recompute the
safety gates, and reject missing/failed evidence rather than accepting only a
`passed: true` summary. Existing 205/70 scope, scale-byte, and package checks are
still enforced separately. QAT-on determinism is not BF16-vs-QAT determinism.

After pulling this change, rerun the same official-mobile command with a **new
output directory** so its immutable generated config contains the policy. Do not
edit old run configs/metadata, hashes, qparams or seed artifacts to bypass a
failure. No training or full conversion was run locally for this change; use the
real H100 preflight to validate the live runtime.

### Retained-mobile LoRA target resolution

The reconstructed seed is `Gemma4ForCausalLM` / `gemma4_text`. PEFT 0.20's
`gemma4` query/value default is not the mobile contract, and its missing
`gemma4_text` mapping previously stopped preflight with `Cannot resolve PEFT
default targets`. The three retained-mobile YAMLs keep `target_modules:
peft-default` for config compatibility, but **only** the official retained-mobile
SFT route resolves it from verified seed-bound qparams before constructing PEFT.
Ordinary model/default/explicit-selector behavior is unchanged.

`MobileQParams.trainable_projection_weight_keys()` is the common authority for
target resolution and the subsequent live QAT exact-scope check. It derives
names from the retained inventory, not a second hard-coded layer rule:

- Layers 0-14: q/k/v/o and gate/up/down (105 projections).
- Shared-KV layers 15-34: q/o and gate/up/down (100 projections).
- Head, per-layer W8 projections, embeddings, and unrelated linears stay outside
  the adapter scope. Shared-KV k/v duplicates are absent from the mapped seed.

The resolver requires all 205 exact Linear paths (including a wrapper's actual
`.linear` child, when applicable). Missing, unexpected, duplicate/shared-weight,
ambiguous or non-Linear projections fail before PEFT construction. Explicit
selectors/exclusions must still produce that exact set; there is no all-linear
fallback or `gemma4_text -> gemma4` alias. The selected qparams contract, scale
storage and inventory hashes must also match the already verified seed. Existing
resume, trainable-scope, provenance and export gates remain enabled.

PEFT 0.20 can condense a long target list into suffixes during attachment. The
retained-mobile route verifies that the actual attached A/B pairs are exactly
the same 205 paths, then restores those exact paths to the in-memory PEFT config
for serialization/resume comparison. This normalization does not rewrite an
existing checkpoint or relax rank/alpha/dropout, adapter hash or resume checks.

### Published activation-scale provenance

The v2 `assets` gate also reads the retained input/output activation scalars
directly from the SHA-pinned published packed checkpoint and compares their
FP32 bytes with the seed contract. This covers the 205 mutable projections,
70 frozen per-layer modules, and the zero-scale head placeholders. A rewritten
but internally self-consistent local A8 sidecar is not sufficient provenance.
Only scalar tensors are loaded by this check; no model weight is materialized.

The verified seed transformation mapping defines the validation boundary, not
every activation-scale name in the packed text checkpoint. Reconstruction
already omits duplicated `k_proj`/`v_proj` weights in shared-KV layers 15–34;
their 80 packed input/output scale tensors are consequently outside the
276-weight retained A8 contract. The packed text inventory can therefore have
632 scalars while the mapped contract has 552. This is not a missing-scale
failure. The validator derives membership from the verified mapping rather
than hard-coding a layer range to ignore.

`mobile_assets_verified.json` records the total text-scale count and the exact
sorted unmapped names under `published_activation_scales`, alongside the
552 validated mapped scalars. Unmapped scales remain covered by the pinned
whole-source SHA but are not treated as training bindings. All mapped roles
still require exact presence, scalar F32 type, bytes and valid values; the
205 mutable / 70 frozen / one zero-head scope counts, canonical mapping and
provenance checks are unchanged. Do not edit the seed, qparams or hashes to
resolve this inventory mismatch. After updating the code, rerun the same
pipeline command with a fresh output directory (for example, a `_v6` suffix).

No runtime KV-cache simulation occurs during training or the pretraining gate.
The pretraining report only inventories statically named TFLite cache boundary
tensors, shapes, dtypes, serialized qparams, and identifiable prefill/decode
stages. That inventory can report `not_established_unidentified`; even an
established inventory does not establish cross-signature cache-buffer mapping
or prove cache mutation, reuse, or runtime
prefill/decode correctness.

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
   Optionally add `--augmentation --augmentation-dir <sealed-augmentation-folder>`
   whose original-data binding matches that input. Alternatively, use a
   compatible frozen bundle selected with `--prepared-input-dir` and
   `--augmentation none` (never both raw and prepared input flags).
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

The default schedule is two epochs, learning rate `1e-5`, a 4,096-token
supervised training/validation sequence limit, evaluation every 500 optimizer
updates, Golden32 generation every 1,000 updates, an independent 5,120-token
evaluation prompt limit, and a 2,048-token generation cap. The independent
prompt limit applies to Golden32, Golden35, and Bixby50, so long holdout prompts
are not truncated merely because training uses a shorter context. Golden32 also
runs at training end. If a smoke-run
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
  --max-seq-length 4096 `
  --max-input-tokens 5120 `
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

## Controlled QAT-LoRA experiments: Run A, Run B, and Run C

The same official-mobile entry point supports these **fresh** experiments:

| Run | `--lora-rank` | `--lora-alpha` | `--learning-rate` | Purpose |
| --- | --- | --- | --- | --- |
| A | 16 | 16 | `1e-5` | Revised-data baseline |
| B | 32 | 32 | `1e-5` | Additional adapter capacity |
| C | 16 | 16 | `5e-6` | Gentler updates |

Fresh defaults remain rank 16, alpha 16, learning rate `1e-5`, and seed 42.
`--seed` controls initialization before model/adapter construction and the
explicit Trainer data-order seed. Rank and alpha must be positive integers;
seed must be an integer in `[0, 2**32)`. Alpha is **not** automatically increased
when rank changes: pass both flags, as below.

Only adapter rank/alpha and learning rate differ between A/B/C. All three keep
the same 205 target projections, zero LoRA dropout, retained W2/W4 and A8
scales, 70 frozen activation modules, mandatory preflights, Golden checkpoint
selection, and official-layout export. The exporter validates the adapter's
actual rank/alpha and A/B tensor shapes against its bound training config; it
does not assume rank 16. The official drafter is preserved, not trained, and
this launcher's runtime gates still use MTP off. Larger rank is not a promise
of better quality or unchanged training memory usage.

### Linux / H100 commands

From the repository root, activate the existing QAT training environment and
set the paths below. `INPUT` is the **same immutable improved dataset** containing
source-bound `train.jsonl` and `val.jsonl`; these default commands do not
regenerate or repair labels. `MODEL` must be the original verified reconstructed mobile seed,
not another run's merged checkpoint. Keep the same visible GPU selection for
all three runs. The example uses four H100s; use `0,1,2,3,4,5,6,7` for eight.

```bash
set -euo pipefail

MODEL=/absolute/path/to/reconstructed-mobile-seed
PACKED=/absolute/path/to/packed-mobile/model.safetensors
OFFICIAL=/absolute/path/to/official-mobile.litertlm
INPUT=/absolute/path/to/improved-source-bound-train-val
EXPORT_PY=/absolute/path/to/export-venv/bin/python
RUN_ROOT=/absolute/path/to/qat_lora_abc_v1
DEVICES=0,1,2,3

COMMON=(
  --model-dir "$MODEL"
  --source-safetensors "$PACKED"
  --official-litertlm "$OFFICIAL"
  --input-dir "$INPUT"
  --exporter-python "$EXPORT_PY"
  --devices "$DEVICES"
  --seed 42 --microbatch 1 --effective-batch 32
  --epochs 2 --max-seq-length 4096 --max-input-tokens 5120 --max-new-tokens 2048
  --eval-steps 500 --golden-every-steps 1000
)

MODE=()  # Plan only first. After review, set MODE=(--execute) and rerun.

python -u training/scripts/run_official_mobile_pipeline.py \
  "${COMMON[@]}" "${MODE[@]}" --output-dir "$RUN_ROOT/run_a" \
  --lora-rank 16 --lora-alpha 16 --learning-rate 1e-5

python -u training/scripts/run_official_mobile_pipeline.py \
  "${COMMON[@]}" "${MODE[@]}" --output-dir "$RUN_ROOT/run_b" \
  --lora-rank 32 --lora-alpha 32 --learning-rate 1e-5

python -u training/scripts/run_official_mobile_pipeline.py \
  "${COMMON[@]}" "${MODE[@]}" --output-dir "$RUN_ROOT/run_c" \
  --lora-rank 16 --lora-alpha 16 --learning-rate 5e-6
```

The executions are sequential; do not launch three jobs on the same GPUs at
once. Every execution needs a nonexistent per-run output directory. If a run
fails, inspect it and choose a new directory for a fresh retry rather than
rerunning all three into existing directories.

For an already generated Muse folder, keep the same `COMMON` settings,
including `--input-dir "$INPUT"`, and add
`--augmentation --augmentation-dir /absolute/path/to/augmentation-run/augmented`.
All A/B/C runs must use that **same frozen folder and original input**; do not
regenerate augmentation between trials. These student commands never call or
manage Muse. See the [independent generation and training commands](SEMANTIC_AUGMENTATION.md).

### Keep the comparison budget fixed

- Keep the same input file contents, seed/tokenizer/prompt, GPU count,
  microbatch, effective batch, sequence limit, and epoch/step horizon. Check
  that preparation reports agree on accepted rows and prepared train/validation
  SHA-256 hashes; a changed dataset is not a rank/LR-only comparison.
- The commands use two complete epochs with no step cap. Identical prepared
  tokenized data and sampling settings give the same scheduled training-token
  exposure; `4096` is a per-example limit, **not** a fixed number of tokens per
  optimizer step. Variable lengths/padding mean `steps * batch * 4096` is not
  the actual non-padding or supervised-token count.
- For a shorter pilot, add `--steps 2000` to `COMMON` for **all three** runs.
  `--steps` overrides the epoch horizon. Do not compare runs stopped at different
  budgets or silently extend just one of them.
- Golden32 remains the checkpoint selector. Golden35 and Bixby50 remain
  final-only holdouts. Inspect quality and saturation reports, then measure
  exported native quality/speed separately; CPU tests do not certify H100
  training or device throughput.

These runs start fresh from the same seed; do not pass `--resume-from-checkpoint`
to initialize B or C from A. A true interruption resume keeps its original
rank, alpha, learning rate, data, RNG, and recipe. Omitted rank/alpha/LR/seed
flags inherit the saved values on resume; conflicting explicit values fail
during planning. Historical continuations without an explicit data-order seed
keep that original configuration rather than gaining a new field.

## Explicit continuation from a Trainer checkpoint

Fresh runs remain strict fresh runs: their resolved config keeps
`training.refuse_resume: true`, and execution refuses an existing output
directory. To continue an interrupted or deliberately extended official-mobile
run, pass `--resume-from-checkpoint` with a complete numbered Hugging Face
Trainer checkpoint such as `checkpoint-3500` and choose a different, nonexistent
`--output-dir`. A callback-created best adapter, final adapter, or weight-only
folder is not resumable because it does not contain the complete Trainer state.

The checkpoint must contain hash-bound model/adapter and tokenizer files,
`trainer_state.json`, `optimizer.pt`, `scheduler.pt`, every required per-rank
`rng_state*.pth`, `golden_callback_state.json`, its resolved training config,
and matching training provenance. Continuation additionally records hashes of
the optimizer, scheduler, Trainer, RNG, and Golden state files and rechecks
them before training and export. Continuation restores optimizer,
scheduler, Trainer step/epoch, RNG, and Golden selector/cadence state exactly as
of the selected numbered checkpoint. If an end-of-run Golden evaluation occurred
after that checkpoint, its later selector state is not implicitly reconstructed.

Continuation reuses the original prepared train, validation, Golden32,
Golden35, and Bixby50 files. Model identity, prepared-data hashes, LoRA/QAT
settings, effective batch, world size, microbatch, gradient accumulation,
learning rate, optimizer, training/evaluation context and generation settings,
evaluation/save/Golden
cadences, and every other recipe field must remain unchanged. The only recipe
change permitted is the active total training horizon:

- With epoch-based training, `--epochs` is the new **total** epoch horizon. It
  may equal the saved horizon only when the selected checkpoint stopped before
  that target, or it may increase. It must not decrease.
- With step-bounded training, `--steps` is the new **total** optimizer-step
  horizon. It follows the same unfinished-or-increase rule.
- Do not switch between epoch and `max_steps` modes. When `max_steps` is active,
  changing the inactive epoch value is also rejected.
- If the horizon option is omitted, the wrapper inherits the saved total. This
  is useful only when the selected checkpoint has not yet reached that target;
  a completed horizon must be increased explicitly.

For example, this plans an epoch-based continuation to a total of four epochs.
All source/model/export flags are the same as the original run, while the source
checkpoint and fresh destination are explicit:

```powershell
python training/scripts/run_official_mobile_pipeline.py `
  --model-dir <same-mobile-seed> `
  --source-safetensors <same-packed-mobile>/model.safetensors `
  --official-litertlm <same-official-mobile.litertlm> `
  --input-dir <same-source-bound-train-val> `
  --resume-from-checkpoint <source-run>/training/<source-run-id>/trainer/checkpoint-3500 `
  --output-dir <new-continuation-run-root> `
  --devices auto `
  --epochs 4 `
  --learning-rate 1e-5 `
  --eval-steps 500 `
  --golden-every-steps 1000 `
  --max-seq-length 4096 `
  --max-input-tokens 5120 `
  --max-new-tokens 2048
```

For a continuation created before the independent evaluation limit was added,
pass its original value explicitly (normally `--max-input-tokens 4096`). Resume
validation remains strict and will not silently change an existing run's prompt
budget.

Review the plan before repeating the identical command with `--execute`. The
plan and checkpoint metadata report the original, previous, and requested
horizons plus the completed optimizer step. They also preserve the original
scheduler warmup-step count. The optimizer and scheduler state resume at the
saved step without restarting warmup; future learning-rate updates use the new
terminal horizon. This is not equivalent to training from scratch with the
longer horizon selected at the beginning.

The prior Golden best is manifest-verified and copied into the new run before
training continues. A worse later score therefore leaves a local selected best;
an improved score replaces only that new-run copy. The source run is never
rewritten. Keep the original resolved config and preparation report, every
numbered checkpoint/state file needed by the chain, and each prior selected-best
directory until merge and retained-scale export finish. Export verifies every
physical hop back to the original fresh run; missing ancestors, altered hashes,
fabricated retention evidence, or a non-physical multi-hop chain fail closed.

These continuation contracts and regressions were validated with CPU fixtures.
No H100/GPU continuation, full model training, or retained-scale export was run
as evidence for this documentation update.

The wrapper owns a bounded, fail-closed stage graph. In v2 the resolved order is
`assets`, `prepare`, `configure`, `no_op_export`, `preflight`, `training`, the
three best-checkpoint evaluations, `merge`, `export`, and the optional Android
speed benchmark:

1. Validate assets, prepare the bound datasets, and resolve the immutable
   training/deployment configs and GPU profile.
2. Before numeric preflight or training, run the real retained-scale exporter
   as a zero-adapter no-op. It re-encodes all 205 materialized seed projections,
   requires every target code buffer to match the official bytes, and proves
   byte-exact target-section and whole virtual-package identity. It does not
   publish a package copy or prove runtime inference.
3. Validate the reconstructed-seed manifest, retained-scale contract, model
   architecture, input data, holdout exclusion, prompt/tokenizer bindings,
   CUDA/BF16 environment, and zero-adapter numeric safety / repeated QAT-on
   greedy determinism. BF16-vs-QAT comparisons are recorded as diagnostics.
4. Train retained-scale effective-weight LoRA QAT from the reconstructed mobile
   seed using DDP and immutable published weight/A8 scales.
5. Evaluate Golden32 at the configured cadence and save the callback-created
   `best_golden_checkpoint` only when the metric improves.
6. Lock checkpoint selection using
   `unique_source_generation_reward_v5_4_avg`. Golden32 contains 32 occurrences
   but 31 unique sources, so raw occurrence-weighted reward is diagnostic and
   must not select the checkpoint.
7. Evaluate the locked best checkpoint on the complete Golden32, Golden35, and
   Bixby50 cohorts. Golden35 and Bixby50 are final-only holdouts and cannot
   change checkpoint or hyperparameter selection.
8. Merge that exact adapter into its bound reconstructed BF16 seed.
9. Re-encode exactly the 205 trained W2/W4 projection code buffers using the
   retained official scales, then patch a copy of the official package.
10. Revalidate provenance, code/scale parity, frozen buffers, package bytes,
   graph/layout/execution-contract identity, and the preserved MTP section
   before atomically publishing the final `.litertlm`.

Every stage runs in a bounded worker subprocess. Before returning, the
worker writes `stage_receipts/<stage>.json`, binding the plan SHA-256 and hashes
of every declared stage artifact; the parent verifies it before starting the next
stage. A failed or interrupted stage remains evidence. Never manually substitute
an intermediate checkpoint or copy one into `best_golden_checkpoint`; use only
the explicit continuation option and its verified state/lineage gates.

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
- `<run>/pretraining_noop_export.json` — the pre-training 205-buffer,
  target-section, virtual-package, qparam, graph, and static cache-inventory
  gate;
- for a fresh run, `<run>/prepared/{train,val,golden32,golden35,bixby50}.jsonl`
  plus preparation manifests; an explicit continuation instead reuses and
  hash-verifies the original prepared directory and records that path in its
  plan, config, data audit, and stage receipt;
- `<run>/configs/mobile_training.yaml` and
  `<run>/configs/mobile_deployment.yaml`;
- `<run>/training/<run-id>/launch/{resolved_training_config.yaml,launch_plan.json,preflight_report.json,training.log}`;
- `<run>/training/<run-id>/saturation_preflight.json` and
  `<run>/training/<run-id>/saturation_telemetry.json`;
- `<run>/training/<run-id>/best_golden_checkpoint/`;
- `<run>/evaluations/best_{golden32,golden35,bixby50}/` with predictions,
  scored predictions, aggregate metrics, and `evaluation_result.json`;
- `<run>/merged_best_hf/`;
- `<run>/retained_scale_export/gemma4_e2b_a2ui_mobile.litertlm` and
  `gemma4_retained_scale_code_only_report.json`;
- `<run>/benchmark_command.json`, which is written even when device benchmarking
  is not requested;
- `<run>/native_quality_command.json` — the separate plan-first Android native
  quality handoff;
- `<run>/android_gpu/target_only/android_litertlm_gpu_parity_report.json` when
  `--benchmark-android` runs successfully;
- `<run>/results.md`, written on success or failure with only verified completed
  results.

TensorBoard data is written under `<tensorboard-root>/<run-id>/`. Training uses
the `training/` subdirectory, while standalone best-checkpoint evaluations add
`evaluation_records/best_<cohort>/` JSON records and evaluation scalar tags to
the same run ID. Retained-scale saturation telemetry samples at most 2,048
values per module/role on rank zero once per active window and, in the full official profile,
opens a window every 20 optimizer steps. The three-step smoke profile samples
every optimizer step. It emits only these six aggregate TensorBoard scalars
under `train/qat_saturation/`: overall, weight, input, and output saturation
fractions, sampled values, and sampled windows. These measurements are
diagnostic; the pipeline does not invent a saturation quality threshold.

Configure these limits through `qat.saturation.sample_every_optimizer_steps`
and `qat.saturation.max_values_per_module`, or disable monitoring with
`qat.saturation.enabled: false`. Grouped weight scales are sampled without
expanding the complete scale matrix. Telemetry detaches its samples and never
retains a training autograd graph.

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

- the pre-training `no_op_export` report passed all 205 code-buffer,
  target-section, virtual-package, qparam, graph, frozen-buffer, and provenance
  checks before numeric preflight or training began;
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

The early no-op gate does not replace this final trained-export gate. The first
proves that the untouched reconstructed seed can make the exact official bytes;
the second proves that the selected trained adapter is the artifact exported and
that only its allowed 205 payloads changed.

Both gates obtain target and MTP sections from the canonical full-package
`inspect_litertlm()` report using the unique-section lookup. The weight helper
`_extract_inventory()` returns only `(selected_section, weight_records)`, not
the package report; passing its first result to package section lookup causes
a misleading `Available: []` failure after a successful 277-weight inventory.
The gates now inspect the full header separately and require the inventory's
target section to match it exactly. Missing/duplicate target or MTP sections,
invalid header ranges and differing inventory section identities still fail
closed. This is a reader-contract correction, not a change to model artifacts,
retained scales, graph checks or the byte-exact no-op requirement.

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

## Separate Android native quality evaluation

Host Hugging Face evaluation and native LiteRT quality are deliberately separate.
After the complete pipeline succeeds, plan the native run into a fresh sibling
directory and review the printed plan before allowing any device action:

```powershell
python training/scripts/evaluate_official_mobile_native.py `
  --run-dir <run> `
  --output-dir <run>_native_quality
```

Plan-only mode writes nothing. To execute the reviewed plan, repeat the command
with an explicit device and `--execute`:

```powershell
python training/scripts/evaluate_official_mobile_native.py `
  --run-dir <run> `
  --output-dir <run>_native_quality `
  --serial <adb-serial> `
  --execute
```

The Android app and instrumentation runner must already be installed. The
command does not build, install, download, train, or use desktop Vulkan; it
stages the exported package and bound requests with MTP disabled, requires full
Android GPU delegation with no CPU fallback, uses a fresh conversation per
case, and cleans only its own temporary device files. Before generation, the
runtime-rendered prompt must exactly match the bound Hugging Face prompt,
including template options. The pinned conversation API enforces the output
token limit; asynchronous decoding stops at the quote-aware A2UI closing
envelope. Output token counts come from native benchmark information, not the
number of decode calls. A failed or timed-out run preserves available raw
diagnostics but never publishes partial scores as a pass.

The native output directory contains `requests.jsonl`,
`device_requests.jsonl`, raw `device_outputs.jsonl`,
`native_runtime_rows.jsonl` with both raw generated and serving-stopped text,
instrumentation/logcat evidence, and separate
`cohorts/{golden32,golden35,bixby50}/` prediction, scored-prediction, and
aggregate files. The saved, hash-bound HF checkpoint predictions are rescored
with the same scorer and weights used for the native outputs; their results
are stored under each cohort's `checkpoint_rescored/` directory.
`comparison.json` and `results.md` compare those checkpoint metrics with fresh
native metrics, and the CLI prints the final table. The two runtimes' scores
remain distinct. Native input/output token IDs are unavailable through the pinned public
Android LiteRT API, so host prompt token IDs and hashes are retained but native
token parity is not claimed. This evaluates generated text and strict A2UI
scores, not native UI rendering; rendering remains a separate unvalidated gate.

Minimal native Golden32, Golden35, and Bixby50 metrics are logged in a separate
TensorBoard run named after the native output directory. The default root comes
from the completed training plan (`/tensorboard` by default); pass
`--tensorboard-root <path>` to change it when evaluating on another host.

## Optional Android target-only speed benchmark

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
coverage, and native rendering before a deployment claim. The native quality
command above establishes the text/scoring portion only, not rendering.

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
- any claimed native LiteRT quality result has a separate passing fresh-sibling
  native-quality report with exact coverage for all three cohorts;
- any claimed Android GPU or throughput result has its own passing device
  report.

Until those conditions are met, report the result as training, evaluation, or
serialization evidence only—not a production-ready mobile model.

## Historical local verification (2026-09-19)

- Before the v2 no-op, mobile-SRQ, saturation, and native-quality additions, a
  regression run across the runner, portable launcher, retained exporter,
  merge/QAT provenance, existing mobile/multiformat workflows, Golden/Bixby
  preparation and scoring, GPU profiles, and parallel generation: **295 passed,
  2 skipped**.
- In that same historical implementation, after the early-exporter graph probe
  was added, the affected runner,
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

## v2 verification (2026-09-19)

The integrated regression run completed with **382 passed, 1 skipped**. The
skip was the executable-symlink test because this Windows host cannot create
that symlink. This count includes the new mobile-SRQ numerical oracles,
frozen A8 hooks, strict trainable scope, source-bound scalar provenance,
no-op export gates, saturation telemetry, native-quality orchestration, and
existing mobile/legacy training/export regression coverage. These are local
CPU tests with small fixtures and mocked expensive/device boundaries, not
measured model-quality or device-runtime results.

The package-inspection regression subset (inspector, pretraining gate,
retained-scale exporter and mobile pipeline) completed with **91 passed,
1 skipped**. It covers the real inventory helper's section-only return
contract, both consuming paths, missing/duplicate sections, canonical reader
errors and mismatched section identities. The two corrected script entrypoints
also passed `--help` and Python compilation checks; no real model was exported.

The published-qparams subset has **22 passing tests**, including a synthetic
632-scale source with exactly 552 mapped roles plus all 80 shared-KV extras.
Missing, modified, wrong-type or contract-omitted mapped scales still fail;
so do wrong source hashes and removal of a required mapped weight. These tests
do not run a real packed model export or retrain the model.

Reproduce from the repository root:

```bash
python -m pytest \
  training/tests/test_mobile_srq_contract.py \
  training/tests/test_qat_training.py \
  training/tests/test_official_mobile_pipeline.py \
  training/tests/test_training_scaffold.py \
  training/tests/test_portable_gemma4_mobile_qat_launcher.py \
  training/tests/test_qat_mtp_workflow.py \
  training/tests/test_mobile_seed_architecture.py \
  training/tests/test_sft_startup_performance.py \
  training/tests/test_mobile_srq_provenance.py \
  training/tests/test_gemma4_retained_scale_pretraining_gate.py \
  training/tests/test_gemma4_retained_scale_exporter.py \
  training/tests/test_qat_saturation.py \
  training/tests/test_gemma4_mobile_training_seed.py \
  training/tests/test_gemma4_mobile_mtp_pipeline.py \
  training/tests/test_android_native_quality.py \
  training/tests/test_published_qparams.py -q -rs -p no:cacheprovider
```

The changed/new core Python modules and focused tests passed scoped Ruff
checks. Python compilation, all three relevant CLI `--help` checks, and
`git diff --check` also passed. The Kotlin probe was reviewed against the
pinned LiteRT-LM v0.16.1 API but **was not compiled or run on Android**.

No full H100 training, real retained-scale model conversion, Android build or
installation, device quality test, or speed benchmark was executed for this
change. Runtime scores and throughput remain unmeasured; the pipeline's real
artifact, numeric and device gates must still pass on the training/deployment
hosts. This is not evidence that Google's private recipe or native KV-cache
numerics have been reproduced.

## LoRA resolution fix verification (2026-09-20)

- Requested five-file regression: **180 passed, 1 skipped**.
- Extended mobile/QAT/export regression (the 16-file command above plus
  `test_lora_target_resolution.py` and `test_e2b_lora_hf_integration.py`):
  **436 passed, 1 skipped**. The skip remains the Windows executable-symlink
  restriction; two dependency deprecation warnings were also reported.
- Resolver-specific tests: **48 passed**. A tiny, randomly initialized, real
  35-layer `Gemma4ForCausalLM` with 20 shared-KV layers attaches exactly 205 PEFT
  adapters, passes live QAT scope binding, saves/reloads those adapters, and
  passes the existing strict resume validator. Wrong alpha still fails. The
  qparams values in this test are synthetic; it is not a real seed export.
- These tests ran with isolated **Transformers 5.14.1 / PEFT 0.20.0**, matching
  the reported failure, plus tokenizers 0.22.2, Accelerate 1.15.0 and CPU PyTorch
  2.13.0. Installed host packages were not replaced.
- All six changed Python files compiled; whitespace checks passed. Scoped Ruff
  ran on all six. The resolver/config/new tests are clean; the unchanged lint
  baseline remains in `sft.py` (62), `mobile_qparams.py` (2), and
  `test_qat_training.py` (2). No new findings were introduced.

Reproduce the requested subset in the training environment:

```bash
python -m pytest \
  training/tests/test_lora_target_resolution.py \
  training/tests/test_qat_training.py \
  training/tests/test_official_mobile_pipeline.py \
  training/tests/test_mobile_seed_architecture.py \
  training/tests/test_gemma4_retained_scale_exporter.py \
  -q -rs -p no:cacheprovider
```

Changed implementation files: `train/lora_targets.py` (exact resolution and
post-PEFT binding), `train/lora_config.py` (seed-bound qparams loading), and
`train/sft.py` (integration), all under `training/src/ir_training`. The
`qat/mobile_qparams.py` change is documentation only; its canonical key
selection logic is unchanged. Regression changes are in
`training/tests/test_lora_target_resolution.py` and `test_qat_training.py`.
Only comments changed in the three retained-mobile configs
(`gemma4_e2b_mobile_seed_ir_qat_sft.yaml`,
`gemma4_e2b_a2ui_express_official_qat.yaml`, and
`gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml`). This runbook and
`gemma4_e2b_qat_mtp_knowledge.md` clarify the PEFT/mobile scope distinction.

No training job, full E2B checkpoint inference/export, H100 run or generated
run-directory modification was performed. Model artifacts, seed manifests,
qparams, hashes and export contracts were not changed. Pull the fix and retry
the pipeline on the training host; all its real-data/runtime gates must still
pass. This result proves the LoRA setup/contract fix, not end-to-end GPU success.

## Numeric preflight policy fix verification (2026-09-20)

The prior hard BF16 top-1 and greedy-prefix comparisons were inappropriate for
the explicitly enabled retained-mobile SRQ simulation. The policy fix separates
those diagnostics from the mandatory gates described above; it does not lower
the 90% threshold or bypass any scope, qparam, artifact or package check.

Changed implementation files (relative to `training/`):

- `src/ir_training/qat/numeric_preflight.py`: named policy, mandatory-gate
  recomputation, and shared fail-closed export/provenance validation.
- `src/ir_training/train/sft.py`: policy binding, finite-logit validation,
  numeric/greedy diagnostic reports and concise console summary.
- `src/ir_training/qat/workflow.py` and `scripts/run_gemma4_mobile_qat.py`:
  static/launcher policy validation; legacy parity gates preserved.
- `src/ir_training/pipeline/official_mobile.py`: explicit policy in generated
  v2 configs.
- `src/ir_training/export/merge_lora.py` and
  `scripts/build_gemma4_retained_scale_litertlm.py`: require policy-matched,
  successful mandatory probe evidence at both merge and final export.
- `configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml`: comments only;
  existing threshold values unchanged.
- `tests/test_retained_numeric_preflight.py`,
  `tests/test_official_mobile_pipeline.py`, and
  `tests/test_portable_gemma4_mobile_qat_launcher.py`: numeric, provenance,
  policy-propagation and backward-compatibility regressions.
- This runbook: gate semantics, retry guidance and verification evidence.

Validation results:

- Focused eight-file command below: **325 passed, 1 skipped** (Windows symlink
  restriction).
- Extended 20-file suite (the previous 18-file LoRA/mobile suite plus
  `test_retained_numeric_preflight.py` and `test_sft_token_cache_integration.py`):
  **546 passed, 6 skipped**, with two dependency deprecation warnings. Skips:
  one Windows symlink test, four tests requiring unavailable `datasets`, and
  one integration test requiring Transformers 5.16.1 instead of the isolated
  5.14.1 stack used to match the reported H100 environment.
- All ten changed/new Python files compiled. Seven changed/new Python files
  pass Ruff; the existing findings in `sft.py` (62), `merge_lora.py` (2), and
  `run_gemma4_mobile_qat.py` (12) are unchanged, with **zero new findings**.
  `git diff --check` passed.
- Tests cover the reported 0.742188 top-1 result, large but finite relative loss
  changes, non-finite logits/losses, excessive absolute loss, repeated greedy
  nondeterminism even with a forged summary flag, missing/unknown/mismatched
  policies, nonzero adapter evidence, and wrong 205/70/qparams bindings at
  **both** merge and export boundaries. Legacy failure tests still pass.

```bash
python -m pytest \
  training/tests/test_retained_numeric_preflight.py \
  training/tests/test_qat_training.py \
  training/tests/test_official_mobile_pipeline.py \
  training/tests/test_training_scaffold.py \
  training/tests/test_portable_gemma4_mobile_qat_launcher.py \
  training/tests/test_mobile_srq_provenance.py \
  training/tests/test_gemma4_retained_scale_pretraining_gate.py \
  training/tests/test_gemma4_retained_scale_exporter.py \
  -q -rs -p no:cacheprovider
```

No training, full-model conversion, H100 job, or existing-run modification was
performed. Model artifacts, qparams, hashes, retained scales, MTP and the exact
205/70 module scopes were not changed. The real hardware preflight remains
required before training; passing local regressions is not a GPU-success claim.
