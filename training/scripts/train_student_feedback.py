#!/usr/bin/env python3
"""Plan/run isolated SFT, student-error replay, or DPO+SFT QAT-LoRA experiments."""
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
    parser.add_argument("--feedback-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm", choices=("sft", "replay", "dpo_sft"), required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--sft-weight", type=float, default=1.)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--max-hard-exposures", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args())
    args["feedback_dir"] = args["feedback_dir"].expanduser().absolute()
    args["output"] = args.pop("output_dir").expanduser().absolute()
    from ir_training.train.student_feedback import train
    result = train(**args)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
