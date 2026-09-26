#!/usr/bin/env python3
"""Generate synthetic train-only sources and admit newly generated Stage 3 labels.

Requires an already-running registered Muse endpoint. --max-new-samples bounds
source attempts, not accepted examples. All nine recipes run round-robin; a
budget below nine cannot cover all nine. No server launch or teacher fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline.training_augmentation import (
    DEFAULT_TEACHER,
    generate_training_augmentations,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--max-new-samples", type=int, default=90)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    if args.max_new_samples < 1:
        parser.error("--max-new-samples must be positive")
    try:
        manifest = generate_training_augmentations(args.donors, args.output_dir,
            teacher_model=args.teacher_model, max_new_samples=args.max_new_samples, seed=args.seed)
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
