from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATASET_SRC = ROOT.parent / "dataset" / "src"
if DATASET_SRC.exists():
    sys.path.insert(0, str(DATASET_SRC))

from ir_training.common.config import load_yaml
from ir_training.eval.mtp_benchmark import build_mtp_benchmark_plan, run_mtp_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or execute a target-only versus Gemma 4 MTP Transformers reference benchmark."
    )
    parser.add_argument(
        "--config",
        default="training/configs/eval/gemma4_e2b_qat_mtp.yaml",
        help="Path to the QAT/MTP benchmark YAML.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Load models and run inference. Without this flag only a no-download plan is printed.",
    )
    parser.add_argument("--output-dir", help="Override run.output_dir.")
    parser.add_argument("--target-model", help="Override source.target_model_id (local merged path or Hugging Face id).")
    parser.add_argument("--target-base-model", help="Override source.target_base_model_id used for precision validation.")
    parser.add_argument("--assistant-model", help="Override source.assistant_model_id.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config).resolve())
    if args.output_dir:
        config.setdefault("run", {})["output_dir"] = args.output_dir
    if args.target_model:
        config.setdefault("source", {})["target_model_id"] = args.target_model
    if args.target_base_model:
        config.setdefault("source", {})["target_base_model_id"] = args.target_base_model
    if args.assistant_model:
        config.setdefault("source", {})["assistant_model_id"] = args.assistant_model
    plan = build_mtp_benchmark_plan(config)
    if not plan["validation"]["ok"]:
        raise SystemExit(json.dumps(plan, indent=2, ensure_ascii=False))
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print("Plan only: no model was downloaded or loaded and no inference was run.")
        return
    result = run_mtp_benchmark(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
