#!/usr/bin/env python3
"""Generate a saved Muse augmentation folder; never start student training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline.golden_training import GoldenTrainingOptions
from ir_training.pipeline.semantic_preparation import prepare_semantic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("e2b", "270m"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Local student tokenizer/config directory; student weights are not required or loaded")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path, help="Source-bound train.jsonl and val.jsonl")
    source.add_argument("--source-run-dir", type=Path, help="Completed dataset run with responses.jsonl and genui.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh augmentation job directory; publishes augmented/ for training's --augmentation-dir")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, help="Evaluation prompt limit; defaults to 5120 for E2B, 4096 for 270M")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-workers", type=int, default=0)
    parser.add_argument("--progress-seconds", type=float, default=10)
    parser.add_argument("--preparation-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preparation-cache-dir", type=Path)
    parser.add_argument("--augmentation-teacher-model", default="muse_glimmer_30b_sglang_reasoning_dflash")
    parser.add_argument("--augmentation-python", type=Path, help="Python with dataset dependencies; defaults to this interpreter")
    parser.add_argument("--augmentation-max-samples", type=int, default=500, help="Attempt ceiling, not guaranteed added rows")
    parser.add_argument("--augmentation-timeout-seconds", type=float, default=7200)
    parser.add_argument("--augmentation-max-extra-fraction", type=float, default=0.10, help="Maximum extra row AND token fraction")
    parser.add_argument("--augmentation-max-family-repeats", type=int, default=2)
    parser.add_argument("--execute", action="store_true", help="Contact the running Muse server and publish data; otherwise inspect plan only")
    return parser


def main(argv: list[str] | None = None) -> int:
    values = vars(build_parser().parse_args(argv))
    execute = values.pop("execute")
    if values["max_input_tokens"] is None:
        values["max_input_tokens"] = 5120 if values["profile"] == "e2b" else 4096
    try:
        result = prepare_semantic_dataset(GoldenTrainingOptions(**values, augmentation="semantic"), execute=execute)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Standalone Muse preparation failed: {exc}", file=sys.stderr)
        return 2
    if result["status"] == "plan_only":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({key: result[key] for key in (
            "status", "augmentation_dir", "training_executed", "student_weights_loaded", "teacher_server_managed",
        )}, indent=2))
        print("Saved augmentation is ready. This command does not launch training or manage the Muse server.")
        print("To include it in an independent training run, keep --input-dir and add --augmentation --augmentation-dir <path above>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
