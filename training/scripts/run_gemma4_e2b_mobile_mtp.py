"""Plan or run the Gemma 4 E2B mobile QAT + default-MTP pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.pipeline.gemma4_mobile_mtp import (
    Gemma4MobileMTPPipelineError,
    run_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "QAT train -> select best golden checkpoint -> merge -> optional public "
            "export -> compose official default MTP into LiteRT-LM."
        )
    )
    parser.add_argument(
        "--config",
        default="training/configs/pipelines/gemma4_e2b_mobile_mtp.yaml",
        help="Pipeline YAML configuration.",
    )
    parser.add_argument("--execute-training", action="store_true", help="Run QAT SFT now.")
    parser.add_argument("--execute-merge", action="store_true", help="Merge the best adapter now.")
    parser.add_argument(
        "--execute-public-export",
        action="store_true",
        help="Run the explicitly enabled public LiteRT Torch standalone export.",
    )
    parser.add_argument(
        "--execute-exact-topology-export",
        action="store_true",
        help=(
            "Quantize the merged checkpoint into the official target graph and "
            "write the final .litertlm while preserving the default MTP section."
        ),
    )
    parser.add_argument(
        "--compose",
        action="store_true",
        help="Compose a compatible target section with the official MTP package.",
    )
    parser.add_argument(
        "--validate-android-gpu",
        action="store_true",
        help="Compare official and candidate packages on the connected Android GPU.",
    )
    parser.add_argument("--adb")
    parser.add_argument("--serial")
    parser.add_argument("--best-checkpoint")
    parser.add_argument("--base-litertlm")
    parser.add_argument("--target-litertlm")
    parser.add_argument("--target-section")
    parser.add_argument("--output-litertlm")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    try:
        config = load_yaml(config_path)
        plan = run_pipeline(
            config,
            config_path=config_path,
            execute_training=args.execute_training,
            execute_merge=args.execute_merge,
            execute_public_export=args.execute_public_export,
            execute_exact_topology_export=args.execute_exact_topology_export,
            compose_package=args.compose,
            validate_android_gpu=args.validate_android_gpu,
            adb_override=args.adb,
            serial_override=args.serial,
            best_checkpoint_override=args.best_checkpoint,
            base_litertlm_override=args.base_litertlm,
            target_litertlm_override=args.target_litertlm,
            target_section_override=args.target_section,
            output_litertlm_override=args.output_litertlm,
            force=args.force,
        )
    except (OSError, ValueError, Gemma4MobileMTPPipelineError) as exc:
        print(f"Gemma 4 mobile MTP pipeline failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not any(
        [
            args.execute_training,
            args.execute_merge,
            args.execute_public_export,
            args.execute_exact_topology_export,
            args.compose,
            args.validate_android_gpu,
        ]
    ):
        print("Plan only: no training, model loading, conversion, or package write was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
