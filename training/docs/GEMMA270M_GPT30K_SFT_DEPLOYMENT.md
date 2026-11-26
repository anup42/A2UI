# Gemma 3 270M GPT30K SFT deployment

The merged dataset and the launch recipe were verified on November 23, 2026.
The GPU run was pending at handoff. Read the run's deployment manifest for
current status and use the scorecard for completed results.

## Dataset and recipe

The five [GPT30K ZIP parts](../data/train/gemma270m_gpt30k/) reconstruct
`a2ui_gemma270m_sft_30k.zip` byte for byte. The archive contains 31,860
training rows and 666 validation rows from the verified 10k
and 20k GPT-6.2 Sol runs and the earlier GPT + v11 release. The merge screened
both splits against Golden and Bixby sources and preserved the source identities
and Express labels. The archived 270M tokenizer measured 31,838 training rows
and all 666 validation rows within 8,192 tokens. The launcher records the
actual token-fit counts from the selected model's tokenizer.

The recipe runs one epoch of full-model SFT. Golden32 selects the checkpoint.
Golden35 and Bixby50 are final holdouts. The existing exporter converts the
selected dense checkpoint to W8 LiteRT-LM with `dynamic_wi8_afp32`. Its
10,240-token cache covers 8,192 input tokens and 2,048 generated tokens.
Native LiteRT-LM GPU inference tests the W8 package on all three cohorts.
The scorecard contains nine HF evaluations and three W8 native evaluations.

## Pull and restore the dataset on the GPU host

Run this in Bash from the existing GPU machine checkout. The parts are regular
Git files. The commands verify each part and the assembled ZIP before extraction.
Set the repository and data paths to match your host if they differ.

```bash
set -euo pipefail
cd /group-volume/k.anup/code/A2UI
git fetch origin new_ir_changes_20260331
git switch new_ir_changes_20260331
git merge --ff-only origin/new_ir_changes_20260331

GPT30K_PARTS="$PWD/training/data/train/gemma270m_gpt30k"
GPT30K_ROOT=/group-volume/k.anup/working_dir/datasets/gpt30k-20261123
GPT30K_DATA="$GPT30K_ROOT/a2ui_gemma270m_sft_30k"
GPT30K_ARCHIVE="$GPT30K_ROOT/a2ui_gemma270m_sft_30k.zip"

mkdir -p "$GPT30K_ROOT"
(cd "$GPT30K_PARTS" && sha256sum --check SHA256SUMS)
(
  GPT30K_TMP=$(mktemp "${GPT30K_ARCHIVE}.XXXXXXXX")
  trap 'rm -f -- "$GPT30K_TMP"' EXIT
  cat "$GPT30K_PARTS"/a2ui_gemma270m_sft_30k.zip.part-{00..04} > "$GPT30K_TMP"
  printf '%s  %s\n' c460c12da52775b684b3516a6d49e219db47d65fa51bad11fe0904898dd1d9e0 "$GPT30K_TMP" |
    sha256sum --check -
  mv -- "$GPT30K_TMP" "$GPT30K_ARCHIVE"
)
unzip -n "$GPT30K_ARCHIVE" -d "$GPT30K_ROOT"
```

## Train, export W8, and run Golden32, Golden35, and Bixby50

Continue in the same shell with your existing CUDA training environment active.
Set `GPT30K_MODEL` to your local dense Gemma 3 270M model or prior dense SFT
checkpoint with its tokenizer. Choose a fresh output directory. The separate
export and LiteRT-LM runtime environments must already be installed, and native
GPU evaluation needs a working NVIDIA Vulkan device in the training container.
TensorBoard logs go beside the run unless `A2UI_TENSORBOARD_ROOT` is already set.
[Pinned environment setup](DEPLOYMENT_EXPORT_ENVIRONMENT.md).

```bash
GPT30K_MODEL=/models/gemma-3-270m-it
GPT30K_RUN=/group-volume/k.anup/working_dir/gemma270m-gpt30k-w8-001
EXPORT_PY=/group-volume/k.anup/envs/a2ui-export-094/bin/python
RUNTIME_PY=/group-volume/k.anup/envs/a2ui-litert-017/bin/python
export A2UI_TENSORBOARD_ROOT="${A2UI_TENSORBOARD_ROOT:-${GPT30K_RUN}_tensorboard}"

python -u "$PWD/training/scripts/run_gemma270m_gpt30k_deployment.py" \
  --model-dir "$GPT30K_MODEL" --input-dir "$GPT30K_DATA" --output-dir "$GPT30K_RUN" \
  --exporter-python "$EXPORT_PY" --runtime-python "$RUNTIME_PY" --execute
```

The wrapper verifies the two extracted split hashes before invoking the shared
deployment launcher. Omit `--execute` to inspect the read-only plan. For
post-training recovery, rerun the exact command with `--resume-run` added.
Recovery verifies and reuses completed stages.

## Compact-prompt full-parameter QAT and native W8 tests

Use `--compact-qat` for a new run on the same verified GPT30K splits. This
recipe starts from local dense Gemma 3 270M weights and trains every parameter
with W8 fake quantization for one epoch at a learning rate of `5e-6` and
microbatch 1. Preparation replaces the instruction scaffold in train,
validation and all three evaluation sets while preserving their source text
and targets. The compact scaffold includes the format and fidelity rules and
one source-grounded travel checklist example. With the archived Gemma 270M
tokenizer, the Bixby50 mean input falls from 4,407 to 677 tokens. Counting
the compact prompt and the stored complete target against a 5,120-token
limit, 31,850 of 31,860 training rows and all 666 validation rows fit.
The run's preparation report gives the final counts after all checks.

In Bash from the repository root, activate your CUDA training environment and
set `GPT30K_MODEL`, `GPT30K_DATA`, `EXPORT_PY` and `RUNTIME_PY` to the same
local inputs and interpreters as above. Use the original dense starting model
for a comparison with the previous run. The converter and native runtime each
need their own pinned environment. The native GPU tests require NVIDIA Vulkan
in the same container. The launcher checks these prerequisites before training.

```bash
git switch new_ir_changes_20260331
git pull --ff-only origin new_ir_changes_20260331
GPT30K_RUN="runs/gemma270m-gpt30k-compact-qat-w8-001"
python -u training/scripts/run_gemma270m_gpt30k_deployment.py \
  --compact-qat --model-dir "$GPT30K_MODEL" --input-dir "$GPT30K_DATA" \
  --output-dir "$GPT30K_RUN" --exporter-python "$EXPORT_PY" \
  --runtime-python "$RUNTIME_PY" --execute
```

Golden32 runs every 100 optimizer steps to select the checkpoint. The pipeline evaluates the best and final
checkpoints, exports the Golden32 winner to W8 LiteRT-LM, and runs the actual
W8 package on Golden32, Golden35 and Bixby50. The context ceilings are 5,120
tokens for a supervised sequence or evaluation input, plus up to 2,048
generated tokens at inference. Individual prompts use only their actual
tokens. Read the native scores in `deployment_scorecard.json`. For subsequent
app inference, take `prepared/shared_prompt.json` from the training directory
named by `deployment_manifest.json` under `plan.training.options.output_dir`.
Build messages with its system, worked user/assistant turns and task prefix,
then apply the exported tokenizer's chat template once. The LiteRT-LM file
contains the tokenizer template; the caller supplies those messages.

## Status and evidence

| Location under the run directory | What to read |
| --- | --- |
| `deployment_manifest.json` | `status`, `active_stage`, `completed`, `error`, planned stages, training settings, and bound artifacts |
| `deployment_results.md` | Human-readable comparison and the results available after a failure |
| `deployment_scorecard.json` | Completed evaluation scores and evidence links |
| `deployment/variants/w8/model.litertlm` | W8 artifact in a fresh successful run |
| `evaluations/w8_golden32/`, `evaluations/w8_golden35/`, `evaluations/w8_bixby50/` | Native predictions, aggregate metrics, and GPU runtime evidence |

After recovery, use the manifest's `artifact_directories` and stage evidence
paths for the current package and evaluations. Completion requires native GPU
evidence for Golden32 (32 rows), Golden35 (35), and Bixby50 (50).
[Shared deployment workflow](GOLDEN_GPU_DEPLOYMENT.md).

For a separate Qualcomm target compiled from the selected checkpoint, see
[SM8850 NPU export and Bixby50 testing](SM8850_NPU_EXPORT.md).
