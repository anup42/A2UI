#!/usr/bin/env python3
"""Plan or run experimental Gemma 4 E2B all-parameter QAT and W248 export.

Planning is the default and performs no CUDA probe, training, or export.  A real
run requires both --execute and --allow-experimental-export.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ir_training.pipeline.full_parameter_qat import (
    FullParameterQATOptions,
    run_pipeline,
    worker,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-stage", help=argparse.SUPPRESS)
    parser.add_argument("--plan-file", type=Path, help=argparse.SUPPRESS)
    for name in ("model-dir", "input-dir", "output-dir", "exporter-python"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument(
        "--devices",
        default="auto",
        help="Exactly 2, 4, or 8 scheduler-visible H100 logical indices/UUIDs, or auto for all visible devices.",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--steps", type=int, help="Optional optimizer-step cap for a bounded smoke run")
    parser.add_argument("--resume-from-checkpoint", type=Path,
                        help="Resume from a numbered full-QAT Trainer checkpoint into a fresh output directory; preserves optimizer, scheduler, RNG and Golden32 selection state.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--distributed-backend",
        choices=("ddp", "sharded"),
        default="ddp",
        help="DDP with Adafactor (default), or DeepSpeed ZeRO-2/3 with AdamW; see --zero-stage.",
    )
    parser.add_argument(
        "--zero-stage",
        type=int,
        choices=(2, 3),
        default=2,
        help="DeepSpeed ZeRO stage for the sharded backend (default: 2).",
    )
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--golden-every-steps", type=int, default=1000)
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Training/validation prompt + response limit (default: 4096).")
    parser.add_argument("--max-input-tokens", type=int, default=5120,
                        help="Independent Golden32/Golden35/Bixby50 evaluation prompt limit (default: 5120).")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--microbatch",
        type=int,
        help="Optional explicit value; this workflow accepts only 1.",
    )
    parser.add_argument("--effective-batch", type=int)
    parser.add_argument("--dataloader-workers", type=int)
    parser.add_argument("--prepare-workers", type=int, default=0)
    parser.add_argument(
        "--preparation-cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--preparation-cache-dir", type=Path)
    parser.add_argument("--tensorboard-root", default="/tensorboard")
    parser.add_argument("--progress-seconds", type=float, default=10.0)
    parser.add_argument("--stage-timeout-seconds", type=float, default=172800.0)
    parser.add_argument("--generation-timeout-seconds", type=float, default=7200.0)
    parser.add_argument(
        "--allow-experimental-export",
        action="store_true",
        help="Acknowledge fresh-graph dynamic W248 export is experimental and is not the official static-A8 mobile topology.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run in a fresh output directory. Omit for a read-only offline plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.worker_stage:
            if args.plan_file is None or args.execute:
                parser.error("Worker mode requires --plan-file, without --execute")
            worker(args.plan_file.resolve(strict=True), args.worker_stage)
            return 0
        if args.plan_file is not None:
            parser.error("--plan-file requires --worker-stage")
        for name in ("model_dir", "input_dir", "output_dir", "exporter_python"):
            if getattr(args, name) is None:
                parser.error("Missing --" + name.replace("_", "-"))
        if args.execute and not args.allow_experimental_export:
            parser.error("--execute requires --allow-experimental-export")
        values = vars(args).copy()
        execute = values.pop("execute")
        values.pop("worker_stage")
        values.pop("plan_file")
        result = run_pipeline(FullParameterQATOptions(**values), execute=execute)
        if not execute:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print(
                "Plan only. Nothing trained/exported; CUDA, seed identity, memory fit, and runtime quality are checked only during --execute.",
                flush=True,
            )
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"All-parameter QAT pipeline failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
