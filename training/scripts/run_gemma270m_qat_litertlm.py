"""Plan or run the Gemma 3 270M QAT -> LiteRT-LM INT8 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.pipeline.gemma270m_litertlm import (
    Gemma270MPipelineError,
    run_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QAT train -> best checkpoint -> merge -> public Gemma 3 270M LiteRT-LM INT8 export."
    )
    parser.add_argument(
        "--config",
        default="training/configs/pipelines/gemma3_270m_qat_litertlm.yaml",
    )
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--execute-merge", action="store_true")
    parser.add_argument("--execute-export", action="store_true")
    parser.add_argument(
        "--execute-exact-topology-export",
        action="store_true",
        help=(
            "Quantize the merged checkpoint into the released Q8 graph and "
            "write the final LiteRT-LM package."
        ),
    )
    parser.add_argument(
        "--validate-android-gpu",
        action="store_true",
        help="Compare official and candidate packages on the connected Android GPU.",
    )
    parser.add_argument("--adb")
    parser.add_argument("--serial")
    parser.add_argument("--best-checkpoint")
    parser.add_argument("--output-litertlm")
    parser.add_argument("--artifact")
    parser.add_argument("--official-litertlm")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    try:
        config = load_yaml(config_path)
        plan = run_pipeline(
            config,
            config_path=config_path,
            execute_training=args.execute_training,
            execute_merge=args.execute_merge,
            execute_export=args.execute_export,
            execute_exact_topology_export=args.execute_exact_topology_export,
            validate_android_gpu=args.validate_android_gpu,
            adb_override=args.adb,
            serial_override=args.serial,
            best_checkpoint_override=args.best_checkpoint,
            output_litertlm_override=args.output_litertlm,
            artifact_override=args.artifact,
            official_litertlm_override=args.official_litertlm,
        )
    except (OSError, ValueError, Gemma270MPipelineError) as exc:
        print(f"Gemma 270M LiteRT-LM pipeline failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not any(
        (
            args.execute_training,
            args.execute_merge,
            args.execute_export,
            args.execute_exact_topology_export,
            args.validate_android_gpu,
        )
    ):
        print("Plan only: no training, model loading, conversion, or artifact write was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
