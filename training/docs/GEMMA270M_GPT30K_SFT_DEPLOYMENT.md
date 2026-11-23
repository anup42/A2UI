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
[Pinned environment setup](DEPLOYMENT_EXPORT_ENVIRONMENT.md).

```bash
GPT30K_MODEL=/models/gemma-3-270m-it
GPT30K_RUN=/group-volume/k.anup/working_dir/gemma270m-gpt30k-w8-001
EXPORT_PY=/group-volume/k.anup/envs/a2ui-export-094/bin/python
RUNTIME_PY=/group-volume/k.anup/envs/a2ui-litert-017/bin/python

python -u "$PWD/training/scripts/run_gemma270m_gpt30k_deployment.py" \
  --model-dir "$GPT30K_MODEL" --input-dir "$GPT30K_DATA" --output-dir "$GPT30K_RUN" \
  --exporter-python "$EXPORT_PY" --runtime-python "$RUNTIME_PY" --execute
```

The wrapper verifies the two extracted split hashes before invoking the shared
deployment launcher. Omit `--execute` to inspect the read-only plan. For
post-training recovery, rerun the exact command with `--resume-run` added.
Recovery verifies and reuses completed stages.

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
