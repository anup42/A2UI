from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.qat_mtp.workflow import (
    summarize_issues,
    validate_benchmark_config,
    validate_training_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statically validate the Gemma 4 QAT-derived LoRA and MTP configs.")
    parser.add_argument(
        "--training-config",
        default="training/configs/models/gemma4_e2b_ir_qat_lora.yaml",
    )
    parser.add_argument(
        "--benchmark-config",
        default="training/configs/eval/gemma4_e2b_qat_mtp.yaml",
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument("--warnings-as-errors", action="store_true")
    args = parser.parse_args()

    training_config = load_yaml(Path(args.training_config).resolve())
    benchmark_config = load_yaml(Path(args.benchmark_config).resolve())
    training_result = summarize_issues(validate_training_config(training_config))
    benchmark_result = summarize_issues(validate_benchmark_config(benchmark_config))
    report = {
        "training_config": str(Path(args.training_config).resolve()),
        "benchmark_config": str(Path(args.benchmark_config).resolve()),
        "training": training_result,
        "benchmark": benchmark_result,
        "ok": training_result["ok"] and benchmark_result["ok"],
        "executes_training": False,
        "loads_models": False,
    }
    if args.warnings_as_errors and (
        training_result["warning_count"] or benchmark_result["warning_count"]
    ):
        report["ok"] = False
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
