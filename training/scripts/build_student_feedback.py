#!/usr/bin/env python3
"""Probe training-only QAT student failures and construct replay/preference artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
sys.path.insert(0, str(ROOT / "dataset/src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe", help="Plan by default; --execute runs the existing QAT evaluator")
    p.add_argument("--prepared-dir", required=True, type=Path)
    p.add_argument("--sft-config", required=True, type=Path)
    p.add_argument("--sft-checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-rows", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("build", help="CPU scoring; publish a new immutable feedback bundle")
    p.add_argument("--probe-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-extra-token-fraction", type=float, default=.10)
    p.add_argument("--max-replays-per-family", type=int, default=1)
    p.add_argument("--min-quality-gap", type=float, default=.05)
    args = parser.parse_args()
    from ir_training.pipeline.student_feedback import probe, build
    from ir_training.data.student_feedback import FeedbackOptions
    def absolute(path):
        return path.expanduser().absolute()
    if args.command == "probe":
        result = probe(absolute(args.prepared_dir), absolute(args.sft_config), absolute(args.sft_checkpoint),
                       absolute(args.output_dir), max_rows=args.max_rows, seed=args.seed,
                       max_new_tokens=args.max_new_tokens, execute=args.execute)
    else:
        result = build(absolute(args.probe_dir), absolute(args.output_dir),
                       FeedbackOptions(args.seed, args.max_extra_token_fraction,
                                       args.max_replays_per_family, args.min_quality_gap))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
