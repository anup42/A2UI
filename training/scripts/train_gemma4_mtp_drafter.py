#!/usr/bin/env python3
"""Plan or explicitly train the Gemma 4 E2B target-conditioned MTP drafter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.train.mtp_drafter import (
    MTPDrafterTrainingError,
    build_drafter_training_plan,
    train_mtp_drafter,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="training/configs/models/gemma4_e2b_mtp_drafter_qat.yaml",
    )
    parser.add_argument(
        "--target-model",
        help="Override the frozen merged target checkpoint used for conditioning.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually load models and train. Omit for a static plan.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = load_yaml(config_path)
        if args.execute:
            result = train_mtp_drafter(
                config,
                config_path=config_path,
                target_model_override=args.target_model,
            )
        else:
            result = build_drafter_training_plan(
                config,
                config_path=config_path,
                target_model_override=args.target_model,
            )
    except (OSError, ValueError, MTPDrafterTrainingError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not args.execute:
        print("Plan only: no model download, model load, or training was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
