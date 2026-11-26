# Export trained Gemma checkpoints for the Fold8 and Flip8 NPU

This is a separate deployment command for Qualcomm SM8850. It starts from the selected **270M full-QAT checkpoint** or the **merged E2B retained-scale QAT LoRA checkpoint**. Each export produces one target-only `model.litertlm` for both SM8850 phones. The native Bixby50 run uses greedy decoding and explicitly disables MTP.

The compiler starts from the trained dense weights, calibrates the main graph using 32 distinct training prompts and up to 32 decode steps per prompt, and compiles the prefill and decode graphs for SM8850. The Qualcomm recipe uses static INT8 weights and INT16 activations. Token lookup tables use weight-only INT8, including the E2B per-layer embeddings. The existing E2B mixed-precision GPU package and the 270M W8 GPU package have their own export commands. Accuracy, peak memory, and latency for these NPU exports require measurement on the phones. [Google’s NPU conversion guide](https://developers.google.com/edge/litert/conversion/pytorch/genai) · [LiteRT-LM on Android NPU](https://developers.google.com/edge/litert/next/litert_lm_npu).

## 1. Set up the export environment

Run from the repository root on Linux x86-64 with Python 3.12. Keep the original trained checkpoint, bound training YAML and prepared data accessible. Allow disk and RAM for the dense checkpoint, floating-point intermediate graphs, calibration, and compiler outputs. Use a fresh output directory for each export or test.

```bash
git pull --ff-only origin new_ir_changes_20260331
python3.12 -m venv .venvs/sm8850-npu
NPU_PY=.venvs/sm8850-npu/bin/python
"$NPU_PY" -m pip install --upgrade pip
"$NPU_PY" -m pip install torch==2.11.0 torchao==0.17.0 --index-url https://download.pytorch.org/whl/cpu
"$NPU_PY" -m pip install -r training/requirements-sm8850-npu.txt
"$NPU_PY" -m pip check
```

## 2. Export the two checkpoints

Replace the two run directory values with your actual runs. For the 270M GPT30K launcher, the deployment manifest records the underlying training directory. For E2B, `merged_best_hf` is produced by the official LoRA merge stage.

```bash
RUN270=runs/gemma270m-deployment
RUN_E2B=runs/e2b-official-mobile
TRAIN270=$("$NPU_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan"]["training"]["options"]["output_dir"])' "$RUN270/deployment_manifest.json")

"$NPU_PY" training/scripts/gemma_sm8850_npu.py export --profile 270m \
  --checkpoint "$TRAIN270/fit/training/best_golden_checkpoint" \
  --output-dir outputs/npu/270m

"$NPU_PY" training/scripts/gemma_sm8850_npu.py export --profile e2b \
  --checkpoint "$RUN_E2B/merged_best_hf" \
  --output-dir outputs/npu/e2b
```

If the 270M checkpoint is already located, pass that directory directly to `--checkpoint`. For a relocated training YAML, pass `--training-config path/to/training_config.yaml` to either command. Its bytes must match the training receipt and its prepared-data path must resolve. The default static cache holds 8,192 tokens across prompt and output. You can set `--cache-length` to a multiple of 128. Every actual Bixby50 prompt must fit together with the saved training config’s full generation allowance.

Each export directory contains the compiled package, its hashes and dispatch evidence in `export.json`, the 50 rendered requests and expected token IDs, and the bound prepared scoring split. Golden32, Golden35, and Bixby50 sources are reserved during calibration. Keep the directory together if the phones connect to another host.

## 3. Build the Android runner once

Use Linux x86-64 with Git LFS, Bazelisk, the Android SDK, Android platform tools (`adb`), and NDK r28b or later. Set `ANDROID_HOME` to your SDK installation. The build uses Google's LiteRT-LM v0.17.0 source, the matching Qualcomm dispatch library, and QAIRT 2.47.0.260601 including its Android arm64 libraries and Hexagon V81 skeleton. If the SDK and phones are on a separate host, copy the complete runtime and export directories there and install the evaluation environment from step 1.

```bash
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/28.2.13676358"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
export QAIRT_ROOT=$("$NPU_PY" -c 'from ai_edge_litert_sdk_qualcomm import get_sdk_path; print(get_sdk_path())')
bash training/scripts/build_sm8850_npu_probe.sh tools/sm8850-runtime
adb devices -l
```

## 4. Test Bixby50 on each phone

The Fold8 SM-F971B and Flip8 SM-F776U listed below both reported `ro.soc.model=SM8850`. Replace the serials when using other units. The runner checks the reported SoC and arm64 ABI before transferring the package.

```bash
for serial in R37L20059SY R3GL203AKSF; do
  for profile in 270m e2b; do
    "$NPU_PY" training/scripts/gemma_sm8850_npu.py bixby50 \
      --export-dir "outputs/npu/$profile" --runtime-dir tools/sm8850-runtime \
      --serial "$serial" --output-dir "outputs/npu-results/$serial/$profile" || exit 1
  done
done
```

Read `aggregate_metrics.json` for the v5.4 Bixby50 score and latency, `scored_predictions.jsonl` for per-source scores, `predictions.jsonl` for raw completions and timings, and `native.log` for device errors. Time is measured from native prefill through decode for each source. The evaluator requires the 50 frozen sources, exact device/host prompt token agreement, compiled Qualcomm dispatch in both prefill and decode, and the loaded Qualcomm dispatch and HTP runtime in the Android process. It retains failure logs without publishing a partial score. The phone caches each model by SHA-256.
