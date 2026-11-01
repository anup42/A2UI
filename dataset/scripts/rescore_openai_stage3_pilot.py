#!/usr/bin/env python3
"""Offline rescore of a sealed pilot; writes assessment reports to a fresh sibling."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline.openai_stage3_rescore import rescore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = rescore(args.pilot_dir, args.output_dir)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(1, f"Offline rescore failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
