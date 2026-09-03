"""Plan or explicitly execute the Gemma 3 270M multi-format pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.pipeline.gemma270m_multiformat import (
    FORMAT_ORDER,
    Gemma270MMultiformatError,
    run_pipeline,
)


def _parse_formats(value: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    return values or FORMAT_ORDER


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A2UI Express Gemma 3 270M: W8-QAT train, Golden-32 selection, "
            "merge, W32/W16/W8/W4 LiteRT-LM export, final Golden scorecard."
        )
    )
    parser.add_argument(
        "--config",
        default="training/configs/pipelines/gemma3_270m_a2ui_express_multiformat.yaml",
    )
    parser.add_argument(
        "--formats",
        default="all",
        help="Comma-separated w32,w16,w8,w4 or all (default).",
    )
    parser.add_argument(
        "--run-id",
        help="Unique run directory/TensorBoard ID; overrides pipeline.run_id.",
    )
    parser.add_argument("--prepare-golden", action="store_true")
    parser.add_argument(
        "--preflight-training",
        action="store_true",
        help="Load the real model/data, validate QAT numerics, and stop before the optimizer.",
    )
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--evaluate-checkpoint", action="store_true")
    parser.add_argument("--execute-merge", action="store_true")
    parser.add_argument("--evaluate-merged", action="store_true")
    parser.add_argument("--execute-exports", action="store_true")
    parser.add_argument("--evaluate-litertlm", action="store_true")
    parser.add_argument(
        "--execute-official-q8",
        action="store_true",
        help="Replace the W8 evaluation artifact with the optional exact official-Q8 topology export.",
    )
    parser.add_argument("--write-scorecard", action="store_true")
    parser.add_argument(
        "--allow-experimental-formats",
        action="store_true",
        help="Required to export/evaluate W16 and block-32 W4.",
    )
    parser.add_argument("--best-checkpoint")
    parser.add_argument("--merged-model")
    parser.add_argument(
        "--runner-config",
        help="Host-specific external LiteRT-LM JSONL runner YAML; overrides the checked-in example.",
    )
    parser.add_argument(
        "--official-q8-source",
        help="Released Gemma 3 270M Q8 .litertlm used only by --execute-official-q8.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    try:
        plan = run_pipeline(
            load_yaml(config_path),
            config_path=config_path,
            run_id_override=args.run_id,
            formats=_parse_formats(args.formats),
            prepare_golden=args.prepare_golden,
            preflight_training=args.preflight_training,
            execute_training=args.execute_training,
            evaluate_checkpoint=args.evaluate_checkpoint,
            execute_merge=args.execute_merge,
            evaluate_merged=args.evaluate_merged,
            execute_exports=args.execute_exports,
            evaluate_litertlm=args.evaluate_litertlm,
            execute_official_q8=args.execute_official_q8,
            write_scorecard=args.write_scorecard,
            allow_experimental_formats=args.allow_experimental_formats,
            best_checkpoint_override=args.best_checkpoint,
            merged_model_override=args.merged_model,
            runner_config_override=args.runner_config,
            official_q8_source_override=args.official_q8_source,
        )
    except (OSError, ValueError, Gemma270MMultiformatError) as exc:
        print(f"Gemma 270M multi-format pipeline failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not any(
        (
            args.prepare_golden,
            args.execute_training,
            args.preflight_training,
            args.evaluate_checkpoint,
            args.execute_merge,
            args.evaluate_merged,
            args.execute_exports,
            args.evaluate_litertlm,
            args.execute_official_q8,
            args.write_scorecard,
        )
    ):
        print(
            "Plan only: no dataset preparation, training, model loading, conversion, evaluation, or artifact write was run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
