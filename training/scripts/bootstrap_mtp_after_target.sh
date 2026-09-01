#!/usr/bin/env bash
set -euo pipefail

TASK="/home/k_anup/working_dir/genui-lm-qat-mtp-20260802/retrain_20260808"
REPO="$TASK/repo"
PY="$TASK/../venv/runtime/bin/python"
DATA="/home/k_anup/working_dir/genui-lm-qat-mtp-20260802/data/prepared_v10_express_clean"
MTPDATA="$TASK/data/mtp_compatible"
TARGET="$TASK/output/mobile_seed_target/final_adapter"
MERGED="$REPO/outputs/mobile_mtp/merged_target"
DRAFTER="$REPO/outputs/mobile_mtp/drafter"
PIPE="$TASK/mobile_mtp_pipeline.yaml"
DCFG="$TASK/mtp_drafter_retrain.yaml"
export PYTHONPATH="$REPO/src:$REPO/scripts"

while [[ ! -f "$TARGET/adapter_model.safetensors" && ! -f "$TARGET/adapter_model.bin" ]]; do
  sleep 20
done

mkdir -p "$MTPDATA"
"$PY" - "$DATA" "$MTPDATA" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
source, output = map(Path, sys.argv[1:])
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it-qat-mobile-transformers")
for split in ("train", "val"):
    src = source / f"{split}.jsonl"
    dst = output / f"{split}.jsonl"
    count = 0
    with src.open(encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            row = json.loads(line)
            messages = row.get("messages") if isinstance(row, dict) else None
            if not isinstance(messages, list):
                continue
            last = max((i for i, msg in enumerate(messages) if isinstance(msg, dict) and msg.get("role") == "assistant"), default=-1)
            if last < 1:
                continue
            text = str(messages[last].get("content") or "")
            ids = tokenizer(text, add_special_tokens=True).get("input_ids") or []
            if len(ids) < 2:
                continue
            out.write(json.dumps({"input_ids": ids, "labels": ids}, ensure_ascii=False) + "\n")
            count += 1
    print(split, count)
PY

"$PY" - "$REPO/configs/models/gemma4_e2b_mtp_drafter_qat.yaml" "$DCFG" "$MTPDATA" "$DRAFTER" "$MERGED" <<'PY'
import sys
from pathlib import Path
import yaml
source, target, data, drafter, merged = map(Path, sys.argv[1:])
cfg = yaml.safe_load(source.read_text())
cfg["run"]["dataset_dir"] = str(data)
cfg["run"]["output_dir"] = str(drafter)
cfg["target"]["model_id_or_path"] = str(merged)
cfg["assistant"]["best_checkpoint_dir"] = str(drafter / "best_checkpoint")
cfg["training"]["max_steps"] = 5
target.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

"$PY" - "$REPO/configs/pipelines/gemma4_e2b_mobile_mtp.yaml" "$PIPE" "$TASK/mobile_seed_retrain.yaml" "$MERGED" "$DRAFTER" "$TASK/official_gemma4_e2b_it_20260808.litertlm" <<'PY'
import sys
from pathlib import Path
import yaml
source, target, train_cfg, merged, drafter, official = map(Path, sys.argv[1:])
cfg = yaml.safe_load(source.read_text())
p = cfg["pipeline"]
p["training_config"] = str(train_cfg)
p["output_dir"] = str(merged.parent)
p["source"]["base_litertlm"] = str(official)
p["source"]["merged_model_dir"] = str(merged)
p["mtp"]["enabled"] = True
p["mtp"]["weight_source"] = "trained"
p["mtp"]["train_assistant"] = True
p["mtp"]["training_config"] = str(Path(target).with_name("mtp_drafter_retrain.yaml"))
p["mtp"]["trained_checkpoint"] = str(drafter / "best_checkpoint")
p["mtp"]["target_intermediate_litertlm"] = str(merged.parent / "target_with_official_mtp.litertlm")
p["mtp"]["exact_topology_output_dir"] = str(merged.parent / "exact_drafter")
p["exact_topology"]["output_dir"] = str(merged.parent / "exact_target")
target.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

cd "$REPO"
if [[ ! -f "$MERGED/config.json" ]]; then
  "$PY" "$REPO/scripts/run_gemma4_e2b_mobile_mtp.py" --config "$PIPE" --execute-merge --best-checkpoint "$TARGET"
fi
"$PY" "$REPO/scripts/run_gemma4_e2b_mobile_mtp.py" --config "$PIPE" --execute-drafter-training
"$PY" "$REPO/scripts/run_gemma4_e2b_mobile_mtp.py" --config "$PIPE" --execute-exact-topology-export
