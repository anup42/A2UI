# Gemma 4 E2B A2UI Express: retained-scale QAT to multi-format LiteRT-LM

This runbook is the hand-off for training and evaluating the Gemma 4 E2B
A2UI Express model on a GPU host. The checked-in entry point is deliberately a
plan/dry-run unless an execution flag is supplied:

```bash
python training/scripts/run_gemma4_e2b_a2ui_express_multiformat.py \
  --run-id e2b_a2ui_YYYYMMDD_001
```

The command prints every resolved path, converter command, evaluation command,
validation error, and external blocker. It does not create output directories,
load a model, train, convert, evaluate, or write TensorBoard data.

## Pinned inputs and contracts

- Pipeline config:
  `training/configs/pipelines/gemma4_e2b_a2ui_express_multiformat.yaml`
- Training config:
  `training/configs/models/gemma4_e2b_a2ui_express_official_qat.yaml`
- Golden preparation config:
  `training/configs/datasets/golden32_20260903_eval.yaml`
- Golden source: `dataset/data/runs/genui_demo_golden40_gemini38_20260903`
- `genui.jsonl` SHA-256:
  `596f56a15a63d904423970c9768797ba795ff0528a04da37dddcba280a4f4e17`
- `responses.jsonl` SHA-256:
  `27406fa78d486ded4e9fcb2a78e987aa54d9f2ee8fe5db926874a1b2ff827089`
- Prepared evaluation split:
  `training/outputs/datasets/golden32_20260903_eval/all.jsonl`
- Required evaluation cardinality: exactly 32 unique cases.
- Primary promotion/reporting metric: `generation_reward_v5_4_avg`, using the
  shared dual metric calculation and `dataset/configs/run.yaml` weights.

Preparation fails closed if either source sidecar has different bytes, if a
source/IR pair is missing, or if the result is not exactly 32 unique rows. The
portable trainer also binds the source files, prepared split, resolved config,
mobile seed, and quantization parameters into launch provenance. Cross-run
leakage is checked with normalized response-content hashes because `q_000001`
style source IDs are local to a run and legitimately repeat between runs.

## Fresh-machine prerequisites

The repository contains the orchestration and small pinned Golden source, not
the multi-gigabyte seed, released package, or prepared training corpus. Before
starting a real run on another PC:

1. Prepare the normal Stage-3 training split from completed dataset runs (never
   from Golden-32):

   ```bash
   python training/scripts/prepare_dataset.py \
     --config training/configs/datasets/stage3_folder_90_10.yaml \
     --source-genui-dir /absolute/path/to/completed-stage3-root \
     --output-dir training/outputs/datasets/stage3_folder_90_10
   ```

2. Obtain the exact packed mobile checkpoint and reconstruct the trainable BF16
   text seed plus retained quantization-parameter sidecars. Plan the command,
   inspect `ready: true`, then repeat it with `--execute`:

   ```bash
   python training/scripts/reconstruct_gemma4_mobile_training_seed.py \
     --source-safetensors /absolute/path/to/packed/model.safetensors \
     --source-config /absolute/path/to/packed/config.json \
     --retained-compiled-report /absolute/path/to/gemma4-mobile-retained-compiled-parity.json \
     --output-dir training/outputs/seeds/gemma4_e2b_mobile_dequantized_text_hf
   ```

   The required output contains the model shards,
   `mobile_training_seed_manifest.json`, `mobile_qparams.json`, and
   `mobile_qparams.safetensors`. `--source-safetensors` on the multiformat
   launcher verifies/binds the packed source; it does not perform this
   reconstruction. See `training/docs/gemma4_mobile_qat_remote_pc_runbook.md`
   for the complete seed and CUDA preflight procedure. Its older Golden-100
   instructions do not replace this pipeline's pinned Golden-32 contract.

3. Obtain the exact released `.litertlm` whose SHA-256 is pinned by
   `official_qat.official_artifact_sha256`, and configure a real external
   runtime runner as described below.

The top-level plan reports missing seed directory, seed manifest, retained
qparams, training `train.jsonl`/`val.jsonl`, released package, and runtime
runner as explicit blockers. The lower-level portable launcher still performs
the deeper streamed numerical, architecture, provenance, and leakage gates.

## What is official and what is a comparison format

| Lane | Export meaning | Status |
|---|---|---|
| `official_qat` | Released Gemma 4 mobile topology and fixed W2/W4/W8 + A8 scales; only integer codes are regenerated from the trained effective weights | Canonical official-format/topology lane |
| `w32` | Public FP32 export, recipe `none` | Comparison lane, not official mobile QAT |
| `w16` | Public FP16 export, recipe `none` plus `--experimental_use_fp16=True` | Comparison lane, not official mobile QAT |
| `w8` | Public dynamic W8/AFP32 recipe `dynamic_wi8_afp32` | Comparison lane, not official mobile QAT |
| `w4` | Public `gemma4_mixed48_b32`, mixed W4/W8 block-32 | Not pure INT4 and not official mobile QAT |

There is no supported public pure-INT4 Gemma 4 E2B export represented here.
The pipeline rejects relabelling the mixed W4/W8 artifact as pure INT4.

“Official-format” describes the retained released graph, layout, precision
assignment, and scale bytes. Target optimization is this repository's
retained-scale STE LoRA-QAT implementation. It is not Google's private trainer,
data mixture, calibration schedule, observers, or an officially trained Google
derivative.

The official retained-scale lane needs the exact released LiteRT-LM package
whose SHA-256 is pinned in the pipeline config. The portable trainer needs the
exact packed official `model.safetensors` (directly or through the reconstructed
seed manifest) and refuses a hash mismatch. These large artifacts are not
committed to Git.

## TensorBoard layout

The YAML value `tensorboard` resolves to `<repo>/tensorboard` on an ordinary
checkout. On MLP, set the required filesystem root without editing YAML:

```bash
export A2UI_TENSORBOARD_ROOT=/tensorboard
```

The resulting layout is:

```text
/tensorboard/<run-id>/training/                         # Trainer event files
/tensorboard/<run-id>/mtp_training/                    # only with trained MTP opt-in
/tensorboard/<run-id>/evaluation_records/checkpoint/   # final HF checkpoint score
/tensorboard/<run-id>/evaluation_records/litertlm_w32/
/tensorboard/<run-id>/evaluation_records/litertlm_w16/
/tensorboard/<run-id>/evaluation_records/litertlm_w8/
/tensorboard/<run-id>/evaluation_records/litertlm_w4/
/tensorboard/<run-id>/evaluation_records/litertlm_official_qat/
```

The trainer evaluates Golden-32 at every configured Trainer evaluation event
(`eval_steps: 500`) and logs those metrics into the same run's training events.
The selected best checkpoint is independently evaluated and recorded after
training. Start TensorBoard with `tensorboard --logdir /tensorboard` on MLP.

## External runtime runner

LiteRT-LM scoring must execute the generated package through a real LiteRT-LM
runtime. `training/configs/eval/litertlm_external_runner.example.yaml` is only a
protocol example and is intentionally detected as an execution blocker. Copy it
outside the tracked config if desired, replace its placeholder command with the
host runtime command, and pass it with `--runner-config`. The runtime must
implement `a2ui_external_generation_v1`; see
`training/docs/golden32_evaluation.md` for the request/output JSONL contract.
Planning validates the protocol, non-empty string command, required
`{requests_path}` and `{outputs_path}` routing, and supported placeholders. The
checked-in `/absolute/path/to/...` example remains a blocker.

## Separate training and conversion environments

Use the CUDA training environment documented in the remote-PC runbook for seed
checks, QAT, checkpoint evaluation, and merge. Use a separate conversion
environment for the tested LiteRT Torch pins:

```bash
python3 -m venv .venv-gemma4-train
. .venv-gemma4-train/bin/activate
# Install the host-compatible CUDA PyTorch build first.
python -m pip install -r training/requirements-gemma4-qat.txt

deactivate
python3 -m venv .venv-litert-export
. .venv-litert-export/bin/activate
python -m pip install -r training/requirements-edge-export-tested.txt
```

Stage-by-stage execution is the recommended way to switch environments. A
single `--execute-all` job is appropriate only when the orchestrator can invoke
the tested converter and external runner through absolute executables without
mixing incompatible Python dependency sets.

Before each LiteRT-LM Golden evaluation, the pipeline runs
`audit_litertlm_package.py --include-hashes`. A malformed package or graph stops
the stage; each public comparison package must also match the inspected
W32/W16/W8 or mixed-W4/W8 precision lane before inference begins. Each
evaluation directory therefore includes
`package_inspection.json` in addition to predictions, scored predictions,
aggregate metrics, external-runner request/output JSONL, logs, and manifests.

## One supervised full execution on a prepared host

Choose a new run ID. Do not reuse one after a partial or complete attempt.

```bash
export A2UI_TENSORBOARD_ROOT=/tensorboard

python training/scripts/run_gemma4_e2b_a2ui_express_multiformat.py \
  --run-id e2b_a2ui_YYYYMMDD_001 \
  --source-safetensors /absolute/path/to/model.safetensors \
  --base-litertlm /absolute/path/to/released-gemma4-e2b.litertlm \
  --runner-config /absolute/path/to/litertlm_runner.yaml \
  --num-gpus 8 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --mtp \
  --execute-all
```

The default MTP settings use the MTP section already present in the released
official package. They do not train a new assistant. Consequently
`--execute-all` skips the optional `mtp_training` stage but preserves MTP during
official export and tells the official-package evaluator to enable it. Use
`--no-mtp` to skip drafter training/transplant and omit `--mtp-enabled` during
official-package evaluation. It does not remove the section: the released MTP
bytes stay in the official-format package so topology remains unchanged.
Public W32/W16/W8/W4 packages are target-only in either case.

The optional repository drafter trainer is only selected when all three values
are set in a local pipeline config: `mtp.enabled: true`,
`mtp.weight_source: trained`, and `mtp.train_assistant: true`. It is a
teacher-forced approximation, not Google's private MTP training recipe; the
plan prints that limitation and does not silently claim official MTP training.
Its resolved config, checkpoint, reports, and TensorBoard directory are scoped
to the same unique top-level run instead of a shared static drafter directory.

## Stage-by-stage recovery and inspection

For controlled execution, repeat `--execute-stage` in dependency order or run
one stage per invocation with the same run ID:

1. `prepare_golden`
2. `training`
3. `checkpoint_evaluation`
4. `merge`
5. `mtp_training` only for the explicit trained-assistant opt-in above
6. `official_export`
7. `public_exports`
8. `litertlm_evaluations` (package audit, then Golden-32 generation/scoring)
9. `scorecard`

Example:

```bash
python training/scripts/run_gemma4_e2b_a2ui_express_multiformat.py \
  --run-id e2b_a2ui_YYYYMMDD_001 \
  --base-litertlm /absolute/path/to/released-gemma4-e2b.litertlm \
  --runner-config /absolute/path/to/litertlm_runner.yaml \
  --execute-stage checkpoint_evaluation \
  --execute-stage merge
```

The final JSON scorecard is under
`training/outputs/pipelines/gemma4_e2b_a2ui_express_multiformat/<run-id>/`.
It requires complete metrics for the actual best checkpoint and every enabled
LiteRT-LM lane. It records `generation_reward_v5_4_avg`, each delta versus the
checkpoint, all aggregate metrics, model/package hashes and sizes, and each
package-inspection report hash. Each entry must bind the selected checkpoint
step, independently pinned Golden config and prepared-split digests, metric
version, TensorBoard sidecar, current artifact bytes, whole-package audit
digest, and public-lane precision identity. The official lane also
revalidates and hashes the retained-scale export report proving the exact
205-projection code-only transplant. Missing or stale evidence fails the
scorecard stage.

This repository addition was validated only through dry-run planning, static
contract checks, and unit tests on the development PC. It does not claim that
GPU training, conversion, LiteRT-LM execution, Android delegation, or final
Golden scores have run. Those are the explicit responsibilities of the remote
training/test host.
