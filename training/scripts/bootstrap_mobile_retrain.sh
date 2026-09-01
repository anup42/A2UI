#!/usr/bin/env bash
set -euo pipefail

TASK="/home/k_anup/working_dir/genui-lm-qat-mtp-20260802/retrain_20260808"
REPO="$TASK/repo"
SOURCE="$TASK/source_mobile"
SEED="$REPO/outputs/seeds/gemma4_e2b_mobile_dequantized_text_hf"
DATA="/home/k_anup/working_dir/genui-lm-qat-mtp-20260802/data/prepared_smoke"
OFFICIAL="/home/k_anup/working_dir/genui-lm-qat-mtp-20260802/official_gemma4_e2b_it_20260808.litertlm"
PY="$TASK/../venv/runtime/bin/python"
export PYTHONPATH="$REPO/src:$REPO/scripts"

mkdir -p "$TASK/logs" "$REPO/outputs/seeds"
EVIDENCE="$REPO/outputs/evidence/gemma4-mobile-retained-compiled-parity.json"
while [[ ! -f "$SOURCE/model.safetensors" || ! -f "$SOURCE/config.json" ]]; do
  sleep 30
done

echo "[$(date -Is)] verifying mobile source"
echo "efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4  $SOURCE/model.safetensors" | sha256sum -c -
echo "cf6d7dc22738b5e6beb364bac833d78b869f5a6ffd57dfc96c6be3f2abc80424  $SOURCE/config.json" | sha256sum -c -
echo "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c  $OFFICIAL" | sha256sum -c -

echo "[$(date -Is)] building retained-constant evidence"
"$PY" "$REPO/scripts/audit_gemma4_mobile_retained_compiled_parity.py" "$OFFICIAL" \
  --source-safetensors "$SOURCE/model.safetensors" \
  --official-artifact-sha256 181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c \
  --output "$SOURCE/gemma4-mobile-retained-compiled-parity.json"
mkdir -p "$(dirname "$EVIDENCE")"
cp "$SOURCE/gemma4-mobile-retained-compiled-parity.json" "$EVIDENCE"

echo "[$(date -Is)] reconstructing verified BF16 mobile training seed"
REPORT_SHA="$(sha256sum "$EVIDENCE" | awk '{print $1}')"
if [[ ! -f "$SEED/mobile_training_seed_manifest.json" ]]; then
  "$PY" "$REPO/scripts/reconstruct_gemma4_mobile_training_seed.py" \
    --source-safetensors "$SOURCE/model.safetensors" \
    --source-config "$SOURCE/config.json" \
    --retained-compiled-report "$EVIDENCE" \
    --retained-compiled-report-sha256 "$REPORT_SHA" \
    --output-dir "$SEED" \
    --plan-output "$TASK/logs/mobile_seed_plan.json" \
    --execute
else
  echo "[$(date -Is)] reusing verified materialized seed"
fi

CFG="$TASK/mobile_seed_retrain.yaml"
"$PY" - "$REPO/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml" "$CFG" "$DATA" "$SEED" <<'PY'
import sys
from pathlib import Path
import yaml

source, target, data, seed = map(Path, sys.argv[1:])
cfg = yaml.safe_load(source.read_text())
cfg["run"]["id"] = "gemma4_e2b_mobile_seed_retrain_20260808"
cfg["run"]["dataset_dir"] = str(data)
cfg["run"]["output_dir"] = str(Path(target).parent / "output" / "mobile_seed_target")
cfg["model"]["model_source"] = str(seed)
cfg["model"]["mobile_training_seed_manifest"] = str(seed / "mobile_training_seed_manifest.json")
cfg["lora"]["target_modules"] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
cfg["training"]["max_steps"] = 5
cfg["training"]["save_steps"] = 5
cfg["training"]["eval_steps"] = 5
cfg["golden_eval"]["enabled"] = False
target.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "[$(date -Is)] static QAT preflight"
"$PY" "$REPO/scripts/validate_qat_training.py" --config "$CFG" --warnings-as-errors
echo "[$(date -Is)] architecture preflight"
"$PY" "$REPO/scripts/validate_gemma4_mobile_seed_architecture.py" --training-config "$CFG" --output "$TASK/logs/mobile_seed_architecture.json"

echo "[$(date -Is)] launching 4-GPU DDP training"
cd "$REPO"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$REPO/scripts/train_sft.py" --config "$CFG"
