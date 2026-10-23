#!/usr/bin/env python3
"""Audit or explicitly regenerate v11 labels using an already-running Muse server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline.training_augmentation import DEFAULT_TEACHER
from pipeline.v11_regeneration import regenerate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly call Muse; default is a read-only plan")
    parser.add_argument("--resume", action="store_true", help="Continue only an exact hash-compatible execution")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum selected labels; 0 audits all (source holds always checked)")
    parser.add_argument("--max-attempts", type=int, default=3, help="Bounded Stage 3 rounds per selected label (1..5)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--prompt-max-tokens", type=int, default=16000)
    parser.add_argument("--context-tokens", type=int, default=32768, help="Must match the Muse server context capacity")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    try:
        result = regenerate(**vars(args))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"v11 regeneration failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
