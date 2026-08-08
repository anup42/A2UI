#!/usr/bin/env python3
"""Validate the reconstructed Gemma 4 seed against Transformers on meta."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.qat.mobile_seed_architecture import (
    MobileSeedArchitectureError,
    validate_mobile_seed_architecture,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare every reconstructed Gemma 4 checkpoint key/shape with "
            "Gemma4ForCausalLM instantiated on the meta device."
        )
    )
    parser.add_argument(
        "--training-config",
        default="training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml",
    )
    parser.add_argument(
        "--transformers-path",
        help="Optional isolated site-packages directory to prepend for this check.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.transformers_path:
        transformers_path = Path(args.transformers_path).expanduser().resolve()
        if not transformers_path.is_dir():
            parser.error(f"Transformers path is not a directory: {transformers_path}")
        sys.path.insert(0, str(transformers_path))
    config_path = Path(args.training_config).expanduser().resolve()
    try:
        config = load_yaml(config_path)
        model = config.get("model") if isinstance(config.get("model"), dict) else {}
        report = validate_mobile_seed_architecture(model, base=ROOT)
    except (OSError, ValueError, MobileSeedArchitectureError) as exc:
        print(f"Gemma 4 mobile architecture preflight failed: {exc}", file=sys.stderr)
        return 2
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["verified"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
