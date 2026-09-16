#!/usr/bin/env python3
"""Train (optionally tune), evaluate and export/test W32/W16/W8/W4 on GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline.golden_deployment import (
    GoldenDeploymentOptions,
    run_deployment,
)
from ir_training.pipeline.golden_training import GoldenTrainingOptions
from run_golden_training import build_parser


def main() -> int:
    parser = build_parser(for_deployment=True)
    parser.description = __doc__
    parser.add_argument("--exporter-python", type=Path, required=True, help="Absolute Python in isolated compatible LiteRT Torch export environment")
    parser.add_argument("--runtime-python", type=Path, required=True, help="Absolute Python with pinned litert-lm-api GPU runtime (see runbook)")
    parser.add_argument("--tune", action="store_true", help="Sequential equal-step Golden32 trials, lock settings, then fresh full training; no screening Golden35/Bixby50")
    parser.add_argument("--trial-steps", type=int, default=1000)
    parser.add_argument("--trials-file", type=Path)
    parser.add_argument("--include-augmentation", action="store_true")
    parser.add_argument("--allow-experimental-formats", action="store_true", help="Acknowledge W16/W4 exporter/runtime limitations; never skips unsupported variants")
    parser.add_argument("--cache-length", type=int, default=8192)
    parser.add_argument("--stage-timeout-seconds", type=float, default=172800, help="Hard limit per subprocess (default 48 hours), not an idle-output timeout")
    parser.add_argument("--generation-timeout-seconds", type=float, default=7200, help="HF/LiteRT generation worker deadline per cohort")
    parser.add_argument("--case-timeout-seconds", type=float, default=600, help="LiteRT per-case progress deadline")
    parser.add_argument("--load-timeout-seconds", type=float, default=1800, help="LiteRT model-load deadline")
    parser.add_argument("--resume-run", action="store_true", help="Explicit post-training recovery in the same output directory; verify and reuse completed stages, never restart training/tuning")
    values = vars(parser.parse_args())
    execute = values.pop("execute")
    keys = ("exporter_python", "runtime_python", "tune", "trial_steps", "trials_file", "include_augmentation",
            "allow_experimental_formats", "cache_length", "stage_timeout_seconds", "generation_timeout_seconds",
            "case_timeout_seconds", "load_timeout_seconds", "resume_run")
    deployment = {key: values.pop(key) for key in keys}
    try:
        result = run_deployment(GoldenDeploymentOptions(base=GoldenTrainingOptions(**values), **deployment), execute=execute)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Full Golden deployment failed: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result if not execute else {"status": result["status"], "output_dir": result["plan"]["output_dir"],
                      "scorecard": str(Path(result["plan"]["output_dir"]) / "deployment_scorecard.json"),
                      "tensorboard_dir": result["plan"]["tensorboard_dir"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
