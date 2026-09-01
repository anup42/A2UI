#!/usr/bin/env bash
set -euo pipefail
: "${TASK_ROOT:?}"; : "${DATASET_DIR:?}"; : "${PYTHON_BIN:?}"
REPO="$TASK_ROOT/repo"; SRC="$TASK_ROOT/source_mobile"; OFFICIAL_DIR="$TASK_ROOT/official_mobile"; SEED="$TASK_ROOT/repo/outputs/seeds/gemma4_e2b_mobile_dequantized_text_hf"; LOGS="$TASK_ROOT/logs"; mkdir -p "$SRC" "$OFFICIAL_DIR" "$SEED" "$LOGS" /tensorboard
export HF_HOME="$TASK_ROOT/cache/huggingface" XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" PYTHONPATH="$REPO/src:$REPO/scripts" HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}" HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-600}"; mkdir -p "$HF_HOME" "$TMPDIR"
HF_BIN="$(command -v hf || true)"; [ -n "$HF_BIN" ] || HF_BIN="$(dirname "$PYTHON_BIN")/hf"
echo "[$(date -Is)] downloading mobile source" | tee "$LOGS/mobile_prepare.log"
"$HF_BIN" download google/gemma-4-E2B-it-qat-mobile-transformers model.safetensors config.json --local-dir "$SRC" --repo-type model 2>&1 | tee -a "$LOGS/mobile_prepare.log"
"$HF_BIN" download litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --local-dir "$OFFICIAL_DIR" --repo-type model 2>&1 | tee -a "$LOGS/mobile_prepare.log"
OFFICIAL="$OFFICIAL_DIR/gemma-4-E2B-it.litertlm"
"$PYTHON_BIN" "$REPO/scripts/audit_gemma4_mobile_retained_compiled_parity.py" "$OFFICIAL" --source-safetensors "$SRC/model.safetensors" --official-artifact-sha256 181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c --output "$SRC/gemma4-mobile-retained-compiled-parity.json" 2>&1 | tee -a "$LOGS/mobile_prepare.log"
REPORT_SHA="$(sha256sum "$SRC/gemma4-mobile-retained-compiled-parity.json" | awk '{print $1}')"
"$PYTHON_BIN" "$REPO/scripts/reconstruct_gemma4_mobile_training_seed.py" --source-safetensors "$SRC/model.safetensors" --source-config "$SRC/config.json" --retained-compiled-report "$SRC/gemma4-mobile-retained-compiled-parity.json" --retained-compiled-report-sha256 "$REPORT_SHA" --output-dir "$SEED" --plan-output "$LOGS/mobile_seed_plan.json" --execute 2>&1 | tee -a "$LOGS/mobile_prepare.log"
TASK_ROOT="$TASK_ROOT" DATASET_DIR="$DATASET_DIR" PYTHON_BIN="$PYTHON_BIN" TARGET_CONFIG_SOURCE=configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml TARGET_MODEL_SOURCE="$SEED" MTP_TOKENIZER_SOURCE=google/gemma-4-E2B-it-qat-mobile-transformers MTP_ASSISTANT_SOURCE=google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant CUDA_VISIBLE_DEVICES_VALUE=0,1 NPROC_PER_NODE=2 TENSORBOARD_ROOT=/tensorboard nohup bash "$REPO/scripts/launch_full_chain.sh" > "$LOGS/supervisor-mobile.log" 2>&1 < /dev/null
