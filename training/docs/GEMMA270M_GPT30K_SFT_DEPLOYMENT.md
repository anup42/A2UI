# Gemma 3 270M GPT30K SFT deployment

The merged dataset and the launch recipe were verified on November 23, 2026.
The GPU run was pending at handoff. Read the run's deployment manifest for
current status and use the scorecard for completed results.

## Dataset and recipe

Use `a2ui_gemma270m_sft_30k.zip` from the merged dataset handoff. Copy the
archive to the GPU host or use its approved shared-storage copy. The archive
contains 31,860 training rows and 666 validation rows from the verified 10k
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

## Launch on the GPU host

Use the existing CUDA training environment. Supply a local dense Gemma 3 270M
checkpoint and the separate exporter and LiteRT-LM runtime Python paths. The
native runtime needs a working NVIDIA Vulkan device in the training container.
[Pinned environment setup](DEPLOYMENT_EXPORT_ENVIRONMENT.md).

From the repository root, verify and unpack the archive. Set the paths for the
model, runtime environments, and fresh run directory.

```bash
GPT30K_ARCHIVE=/data/staging/a2ui_gemma270m_sft_30k.zip
GPT30K_ROOT=/data/a2ui
GPT30K_DATA="$GPT30K_ROOT/a2ui_gemma270m_sft_30k"
GPT30K_MODEL=/models/gemma-3-270m-it
GPT30K_RUN=/runs/gemma270m-gpt30k-w8-001
EXPORT_PY=/group-volume/k.anup/envs/a2ui-export-094/bin/python
RUNTIME_PY=/group-volume/k.anup/envs/a2ui-litert-017/bin/python

mkdir -p "$GPT30K_ROOT"
printf '%s  %s\n' c460c12da52775b684b3516a6d49e219db47d65fa51bad11fe0904898dd1d9e0 "$GPT30K_ARCHIVE" |
  sha256sum --check - && unzip -n "$GPT30K_ARCHIVE" -d "$GPT30K_ROOT"

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
