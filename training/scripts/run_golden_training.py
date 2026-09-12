#!/usr/bin/env python3
"""Prepare, train and test E2B/270M on shared-prompt Golden32 + Golden35."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline.golden_training import GoldenTrainingOptions, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("e2b", "270m"), default="e2b")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local dense HF model and tokenizer; never downloaded automatically")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run folder; existing folders require --continue-run")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-run-dir", type=Path, help="Completed Stage3 run; default is checked-in dataset/data/runs/dataset_v1")
    source.add_argument("--input-dir", type=Path, help="Existing source-bound train.jsonl and val.jsonl; both are filtered and prompt-normalized")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--steps", type=int, help="Optional optimizer-step limit; use 20 for a smoke test")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--golden-every-steps", type=int, default=1000)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--effective-batch", type=int)
    parser.add_argument("--dataloader-workers", type=int)
    parser.add_argument("--tensorboard-root", default=os.environ.get("A2UI_TENSORBOARD_ROOT") or "/tensorboard")
    parser.add_argument("--qat", action="store_true", help="270M only: initialize from a local full SFT checkpoint and train W8 QAT. E2B official QAT remains separate.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Explicitly prepare, preflight, train and evaluate on this host")
    mode.add_argument("--prepare-only", action="store_true", help="Load only the local tokenizer and prepare/validate all data; no weights/CUDA training")
    parser.add_argument("--continue-run", action="store_true", help="Continue verified completed stages; never silently restart failed training")
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
