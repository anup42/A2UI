#!/usr/bin/env python3
"""Prepare, train and test E2B/270M on Golden32, Golden35 and Bixby50."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline.golden_training import GoldenTrainingOptions, run_pipeline


def build_parser(*, for_deployment: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("e2b", "270m"), default="e2b")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local dense HF model and tokenizer; never downloaded automatically")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh full deployment folder; no automatic restart" if for_deployment else "New run folder; existing folders require --continue-run")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-run-dir", type=Path, help="Completed Stage3 run; default is checked-in dataset/data/runs/dataset_v1")
    source.add_argument("--input-dir", type=Path, help="Existing source-bound train.jsonl and val.jsonl; both are filtered and prompt-normalized")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, help="Optional profile learning-rate override")
    parser.add_argument("--weight-decay", type=float, help="Optional profile weight-decay override")
    parser.add_argument("--warmup-ratio", type=float, help="Optional warmup fraction in [0, 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10, help="Console/TensorBoard training metric cadence in optimizer updates")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Disable only after a successful memory preflight on the target GPU")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--augmentation", choices=("none", "rare_components"), default="none", help="Optional capped train-only rare-component resampling; no invented source text or IR")
    parser.add_argument("--augmentation-max-extra-fraction", type=float, default=0.10, help="Maximum added occurrences / original training rows; at most 0.5")
    parser.add_argument("--augmentation-max-family-repeats", type=int, default=2, help="Maximum TOTAL occurrences per source family including originals; existing larger families are never removed or repeated")
    if not for_deployment:
        parser.add_argument("--evaluate-golden35", action=argparse.BooleanOptionalAction, default=True, help="Defer holdout during tuning with --no-evaluate-golden35; it remains reserved from train/val")
        parser.add_argument("--evaluate-bixby50", action=argparse.BooleanOptionalAction, default=True, help="Bixby50 source-only final holdout; --no-evaluate-bixby50 defers inference but keeps it excluded from train/val")
    parser.add_argument("--steps", type=int, help="Optional optimizer-step limit; use 20 for a smoke test")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--golden-every-steps", type=int, default=1000)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--effective-batch", type=int)
    parser.add_argument("--dataloader-workers", type=int)
    parser.add_argument("--prepare-workers", type=int, default=0, help="CPU validation workers: 0 auto (CPU affinity/quota and RAM aware, cap 16); 1 serial. Independent of GPU/dataloader workers.")
    parser.add_argument("--progress-seconds", type=float, default=10, help="Preparation progress/ETA and blocking-stage heartbeat interval")
    parser.add_argument("--preparation-cache", action=argparse.BooleanOptionalAction, default=True, help="Reuse complete preparations after input, tokenizer, preprocessing/schema and artifact hash verification")
    parser.add_argument("--preparation-cache-dir", type=Path, help="Persistent prepared-data store; default: <output parent>/.golden-preparation-cache. Owns its data independently of earlier run folders.")
    parser.add_argument("--token-cache", action=argparse.BooleanOptionalAction, default=True, help="Persist exact token IDs, masks and labels for verified reuse by preflight, training and GPU workers")
    parser.add_argument("--token-cache-dir", type=Path, help="Shared token store; default: <preparation-cache-dir>/tokens. Keep on fast persistent storage outside model, source and run directories.")
    parser.add_argument("--tensorboard-root", default=os.environ.get("A2UI_TENSORBOARD_ROOT") or "/tensorboard")
    parser.add_argument("--tensorboard-detail", choices=("minimal", "full"), default=os.environ.get("A2UI_TENSORBOARD_DETAIL") or "minimal", help="Minimal (default): headline training/Golden/runtime metrics and HParams; full: all diagnostic charts and JSON text. Disk artifacts remain complete.")
    if not for_deployment:
        parser.add_argument("--qat", action="store_true", help="270M only: initialize from a local full SFT checkpoint and train W8 QAT. E2B official QAT remains separate.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Execute the complete training, export and GPU evaluation plan" if for_deployment else "Explicitly prepare, preflight, train and evaluate on this host")
    if not for_deployment:
        mode.add_argument("--prepare-only", action="store_true", help="Load only the local tokenizer and prepare/validate all data; no weights/CUDA training")
        parser.add_argument("--continue-run", action="store_true", help="Continue verified completed stages; never silently restart failed training")
    return parser


def main() -> int:
    parser = build_parser()
    args = vars(parser.parse_args())
    mode_values = {name: args.pop(name) for name in ("execute", "prepare_only", "continue_run")}
    try:
        result = run_pipeline(GoldenTrainingOptions(**args), **mode_values)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Golden training workflow failed: {exc}", file=sys.stderr)
        return 2
    if result.get("status") == "plan_only":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"status": result["status"], "output_dir": result["plan"]["options"]["output_dir"],
                          "completed_stages": list(result["completed"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
