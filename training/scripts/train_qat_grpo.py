#!/usr/bin/env python3
"""Train E2B/270M QAT LoRA GRPO from a verified SFT adapter."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training" / "src"))
sys.path.insert(0, str(REPO_ROOT / "dataset" / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dependency-preflight-only", action="store_true")
    args = parser.parse_args()
    from ir_training.common.config import load_yaml
    from ir_training.train.cuda_runtime import training_attention_policy
    from ir_training.train.qat_grpo import train_qat_grpo
    path = Path(args.config).resolve(strict=True)
    training_attention_policy(train_qat_grpo)(load_yaml(path), path,
        dependency_preflight_only=args.dependency_preflight_only)


if __name__ == "__main__":
    main()
