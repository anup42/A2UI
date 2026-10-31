#!/usr/bin/env python3
"""Prepare, explicitly submit, and validate a separate 1,000-source OpenAI Batch pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline import openai_stage3_batch as pilot
from pipeline import openai_stage3_batch_api as batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Offline: select training sources, capture canonical prompts, estimate cost")
    prepare.add_argument("--input-dir", type=Path, required=True, help="Existing train/val JSONL directory or checked v11 gzip bundle")
    prepare.add_argument("--samples", type=int, default=1000)
    prepare.add_argument("--seed", type=int, default=123)
    prepare.add_argument("--model", choices=tuple(pilot.PRICES), default="gpt-6.1-sol")
    prepare.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    prepare.add_argument("--max-output-tokens", type=int, default=8192, help="Per request, including reasoning tokens")
    submit = commands.add_parser("submit", help="Paid: upload and submit ONE 24-hour Batch; never auto-retry creation")
    submit.add_argument("--max-estimated-cost-usd", type=float, required=True, help="Local estimate allowance, not a provider billing cap")
    status = commands.add_parser("status", help="Read remote status without generation")
    status.add_argument("--batch-id", help="Recover a lost create response; must match the prepared input and metadata")
    collect = commands.add_parser("collect", help="Download once; replay canonical Stage 3 and strict fidelity gates offline")
    for command in (prepare, submit, status, collect):
        command.add_argument("--output-dir", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        result = (pilot.prepare if command == "prepare" else getattr(batch, command))(**args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"OpenAI Stage 3 Batch pilot failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
