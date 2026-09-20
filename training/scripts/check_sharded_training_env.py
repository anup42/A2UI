#!/usr/bin/env python3
"""Quick sharded dependency/NVTX check before the full model preflight.

Run inside the dedicated Linux/CUDA sharded venv. No model, optimizer, or
distributed engine is created. This does not replace the mandatory H100 gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.train.sharded_environment import check_sharded_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--effective-batch", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        result = check_sharded_environment(
            world_size=args.world_size, effective_batch=args.effective_batch,
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Sharded environment check failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, indent=2), flush=True)
    print("Environment check passed; full model/memory preflight is still required.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
