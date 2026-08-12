"""Plan or run Gemma 4 E2B mobile QAT with official or trained MTP weights."""

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
            "Plan or run the provenance-gated Gemma 4 retained-scale merge/export, "
            "preserving the released LiteRT-LM topology and MTP bytes."
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
        "--execute-drafter-training",
        action="store_true",
        help=(
            "Run the opt-in target-conditioned drafter QAT stage. Requires "
            "pipeline.mtp.weight_source=trained."
        ),
    )
    parser.add_argument(
        "--execute-public-export",
        action="store_true",
        help="Rejected legacy flag: public abs-max export is unsafe for retained_mobile.",
    )
    parser.add_argument(
        "--execute-exact-topology-export",
        action="store_true",
        help=(
            "Rejected legacy flag. Use --execute-retained-scale-export."
        ),
    )
    parser.add_argument(
        "--execute-retained-scale-export",
        action="store_true",
        help=(
            "Run the dedicated exact-205 retained-scale code-only exporter and "
            "recheck its fail-closed report."
        ),
    )
    parser.add_argument(
        "--compose",
        action="store_true",
        help="Rejected legacy flag for retained_mobile QAT.",
    )
    parser.add_argument(
        "--validate-android-gpu",
        action="store_true",
        help="Compare official and candidate packages on the connected Android GPU.",
    )
    parser.add_argument("--adb")
    parser.add_argument("--serial")
    parser.add_argument(
        "--training-config",
        help="Exact resolved launcher config copied/hash-bound into the checkpoint.",
    )
    parser.add_argument("--best-checkpoint")
    parser.add_argument("--base-litertlm")
    parser.add_argument(
        "--merged-model-dir", help="Fresh output directory for merged HF Safetensors."
    )
    parser.add_argument(
        "--exact-output-dir",
        help="Fresh retained-scale exporter working/output directory.",
    )
    parser.add_argument(
        "--export-report", help="Fresh retained-scale exporter JSON report path."
    )
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
            execute_drafter_training=args.execute_drafter_training,
            execute_public_export=args.execute_public_export,
            execute_exact_topology_export=args.execute_exact_topology_export,
            execute_retained_scale_export=args.execute_retained_scale_export,
            compose_package=args.compose,
            validate_android_gpu=args.validate_android_gpu,
            adb_override=args.adb,
            serial_override=args.serial,
            training_config_override=args.training_config,
            best_checkpoint_override=args.best_checkpoint,
            base_litertlm_override=args.base_litertlm,
            merged_model_dir_override=args.merged_model_dir,
            exact_output_dir_override=args.exact_output_dir,
            export_report_override=args.export_report,
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
            args.execute_drafter_training,
            args.execute_public_export,
            args.execute_exact_topology_export,
            args.execute_retained_scale_export,
            args.compose,
            args.validate_android_gpu,
        ]
    ):
        print("Plan only: no training, model loading, conversion, or package write was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
