#!/usr/bin/env python3
"""Plan or execute sequential, equal-budget SFT/QAT hyperparameter screening."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline.experiments import ExperimentOptions, run_experiments
from ir_training.pipeline.golden_training import GoldenTrainingOptions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("e2b", "270m"), default="e2b")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh experiment directory, never auto-resumed")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-dir", type=Path)
    source.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--trial-steps", type=int, required=True, help="Equal optimizer steps per trial. Screening budget, not a full-run quality guarantee.")
    parser.add_argument("--trials-file", type=Path, help="Optional JSON list of named LR/weight-decay/warmup/augmentation trials. Baseline always added; max 12 total.")
    parser.add_argument("--include-augmentation", action="store_true", help="Append a target-preserving rare-component resampling comparison; validation/Goldens stay unchanged")
    parser.add_argument("--learning-rate", type=float, help="Override baseline LR and center of automatic half/double trials")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--augmentation-max-extra-fraction", type=float, default=0.1)
    parser.add_argument("--augmentation-max-family-repeats", type=int, default=2)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--epochs", type=float, default=1.0, help="Recorded schedule parameter; explicit trial-steps controls duration")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--golden-every-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--effective-batch", type=int)
    parser.add_argument("--dataloader-workers", type=int)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-workers", type=int, default=0)
    parser.add_argument("--progress-seconds", type=float, default=10)
    parser.add_argument("--preparation-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preparation-cache-dir", type=Path)
    parser.add_argument("--tensorboard-root", default=os.environ.get("A2UI_TENSORBOARD_ROOT") or "/tensorboard")
    parser.add_argument("--qat", action="store_true", help="270M only. E2B official retained-scale QAT remains in its separate launcher.")
    parser.add_argument("--execute", action="store_true", help="Explicitly run all trials one by one, lock a Golden32 winner, then evaluate that checkpoint once on Golden35")
    args = vars(parser.parse_args())
    execute = args.pop("execute")
    experiment = {name: args.pop(name) for name in ("trial_steps", "trials_file", "include_augmentation")}
    try:
        result = run_experiments(ExperimentOptions(base=GoldenTrainingOptions(**args), **experiment), execute=execute)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Sequential experiments failed: {exc}", file=sys.stderr, flush=True)
        return 2
    if result.get("status") == "plan_only":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"status": result["status"], "output_dir": result["plan"]["output_dir"],
                          "selected_trial": result.get("selected_trial"), "completed_trials": len(result["trials"]),
                          "tensorboard_dir": result["plan"]["tensorboard_dir"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
