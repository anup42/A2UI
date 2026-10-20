#!/usr/bin/env python3
"""Build fidelity-admitted DPO preferences from E2B/270M GRPO training audits only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
sys.path.insert(0, str(ROOT / "dataset" / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo-run-dir", type=Path, required=True,
                        help="Stopped source run with its original plan, prepared data, and periodic audits")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--min-reward-gap", type=float, default=0.01,
                        help="Minimum current mapped reward gap (reward range -1..1), default 0.01")
    parser.add_argument("--min-pairs", type=int, default=None,
                        help="Minimum pairs; defaults to the run's effective batch")
    args = parser.parse_args()
    from ir_training.train.dpo_preference_run import load_preference_run
    from ir_training.train.dpo_preferences import build_preferences
    if args.output_dir.exists():
        parser.error("--output-dir must be new")
    print("Verifying source run, train/held-out separation and local tokenizer (no model loaded)...", flush=True)
    context = load_preference_run(args.grpo_run_dir)
    print(f"Rescoring {len(context['audit_paths'])} {context['family']} training audit files...", flush=True)
    report = build_preferences(context["audit_paths"], context["train_rows"], context["tokenizer"],
        args.output_dir, limits=context["limits"], reward_config=context["reward_config"],
        forbidden_rows=context["forbidden_rows"], evidence=context["evidence"],
        min_reward_gap=args.min_reward_gap,
        min_pairs=context["effective_batch"] if args.min_pairs is None else args.min_pairs)
    print(json.dumps({key: report[key] for key in ("status", "pairs", "preferences_path", "counts")}, indent=2))


if __name__ == "__main__":
    main()
