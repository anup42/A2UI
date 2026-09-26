#!/usr/bin/env python3
"""E2B mobile QAT -> best Golden/Bixby checkpoint -> official-layout LiteRT-LM.

Plan-only by default. See training/docs/OFFICIAL_MOBILE_QAT_PIPELINE.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ir_training.pipeline.official_mobile import (
    OfficialMobileOptions,
    run_pipeline,
    worker,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-stage", help=argparse.SUPPRESS)
    parser.add_argument("--plan-file", type=Path, help=argparse.SUPPRESS)
    for name in ("model-dir", "source-safetensors", "official-litertlm", "output-dir"):
        parser.add_argument("--" + name, type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-dir", type=Path, help="Source train/val rows to reprepare")
    source.add_argument("--prepared-input-dir", type=Path, help="Verified frozen bundle to adopt without generation or re-preparation")
    parser.add_argument("--exporter-python", type=Path, help="Retained-scale export environment; defaults to this Python. No generic converter is used.")
    parser.add_argument("--devices", default="auto", help="All scheduler-visible GPUs, or visible logical indices/UUIDs.")
    parser.add_argument("--epochs", type=float, help="Total epochs (fresh default: 2; resume default: saved horizon)")
    parser.add_argument("--steps", type=int, help="Optional total optimizer-step horizon, not extra steps on resume")
    parser.add_argument("--resume-from-checkpoint", type=Path, help="Explicit numbered Trainer checkpoint; requires a fresh output directory. Only the total training horizon may increase.")
    parser.add_argument("--learning-rate", type=float, help="Fresh default: 1e-5; resume default: saved learning rate")
    parser.add_argument("--lora-rank", type=int, help="LoRA rank on the same 205 projections (fresh default: 16; resume: saved rank)")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha (fresh default: 16; resume: saved alpha). Set explicitly when changing rank.")
    parser.add_argument("--seed", type=int, help="Initialization and data-order seed (fresh default: 42; resume: saved seed)")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--golden-every-steps", type=int, default=1000)
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Supervised training/validation sequence limit (default: 4096)")
    parser.add_argument("--max-input-tokens", type=int, default=5120,
                        help="Independent Golden32/Golden35/Bixby50 evaluation prompt limit (default: 5120)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--effective-batch", type=int)
    parser.add_argument("--dataloader-workers", type=int)
    parser.add_argument("--prepare-workers", type=int, default=0)
    parser.add_argument("--preparation-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preparation-cache-dir", type=Path)
    parser.add_argument("--tensorboard-root", default="/tensorboard")
    parser.add_argument("--progress-seconds", type=float, default=10)
    parser.add_argument("--stage-timeout-seconds", type=float, default=172800)
    parser.add_argument("--generation-timeout-seconds", type=float, default=7200)
    parser.add_argument("--augmentation", nargs="?", const="semantic", choices=("none", "rare_components", "semantic"), default="none", help="Bare flag enables train-only Muse semantic augmentation on fresh runs; rare_components repeats exact examples")
    parser.add_argument("--augmentation-teacher-model", default="muse_glimmer_30b_sglang_reasoning_dflash")
    parser.add_argument("--augmentation-python", type=Path, help="Dataset generation Python environment; defaults to this interpreter")
    parser.add_argument("--augmentation-max-samples", type=int, default=500)
    parser.add_argument("--augmentation-timeout-seconds", type=float, default=7200)
    parser.add_argument("--augmentation-max-extra-fraction", type=float, default=0.10)
    parser.add_argument("--augmentation-max-family-repeats", type=int, default=2)
    parser.add_argument("--benchmark-android", action="store_true", help="After export, run matched target-only GPU speed gate on an explicitly selected Android device.")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial")
    parser.add_argument("--execute", action="store_true", help="Run all stages in a fresh output directory; otherwise only print a plan.")
    args = parser.parse_args(argv)
    try:
        if args.worker_stage:
            if args.plan_file is None or args.execute:
                parser.error("Worker mode requires --plan-file, without --execute")
            worker(args.plan_file.resolve(strict=True), args.worker_stage)
            return 0
        if args.plan_file is not None:
            parser.error("--plan-file requires --worker-stage")
        for name in ("model_dir", "source_safetensors", "official_litertlm", "output_dir"):
            if getattr(args, name) is None:
                parser.error("Missing --" + name.replace("_", "-"))
        if args.input_dir is None and args.prepared_input_dir is None and args.resume_from_checkpoint is None:
            parser.error("Missing --input-dir or --prepared-input-dir")
        values = vars(args).copy()
        execute = values.pop("execute")
        values.pop("worker_stage")
        values.pop("plan_file")
        result = run_pipeline(OfficialMobileOptions(**values), execute=execute)
        if not execute:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("Plan only. Nothing trained/exported; artifact identities and GPU capability are checked on --execute.")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Official mobile pipeline failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
