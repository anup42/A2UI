#!/usr/bin/env python3
"""Host utilities for GenUI Anchored Criterion Judge (GACJ) v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_judge.criterion_analysis_v3 import (  # noqa: E402
    analyze_fresh_criterion_repeats,
    backtest_v2_repeat_bundle,
)
from pipeline.genui_judge.criterion_judgments_v3 import (  # noqa: E402
    append_criterion_judgment_pass,
    rebuild_criterion_packet_judgments,
)
from pipeline.genui_judge.criterion_packets_v3 import (  # noqa: E402
    build_criterion_review_packets,
)
from pipeline.genui_judge.criterion_protocol_v3 import (  # noqa: E402
    protocol_fingerprint,
    protocol_mapping,
    validate_criterion_judgment_pass,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _print(value: Any) -> None:
    print(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GenUI Anchored Criterion Judge v3 host utility"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    protocol_parser = subparsers.add_parser("protocol")
    protocol_parser.add_argument("--fingerprint-only", action="store_true")

    packets_parser = subparsers.add_parser("build-repeat96-workspace")
    packets_parser.add_argument("review_bundle_dir", type=Path)
    packets_parser.add_argument("output_dir", type=Path)
    packets_parser.add_argument("--seed", type=int, default=20260801)
    packets_parser.add_argument("--task-block-size", type=int, default=16)

    validate_parser = subparsers.add_parser("validate-pass")
    validate_parser.add_argument("judgment_json", type=Path)

    append_parser = subparsers.add_parser("append-pass")
    append_parser.add_argument("benchmark_dir", type=Path)
    append_parser.add_argument("judgment_json", type=Path)

    rebuild_parser = subparsers.add_parser("rebuild")
    rebuild_parser.add_argument("benchmark_dir", type=Path)

    backtest_parser = subparsers.add_parser("backtest-repeat96")
    backtest_parser.add_argument("review_bundle_dir", type=Path)
    backtest_parser.add_argument("output_dir", type=Path)
    backtest_parser.add_argument("--skip-packet-build", action="store_true")

    analyze_parser = subparsers.add_parser("analyze-repeat96")
    analyze_parser.add_argument("workspace_dir", type=Path)
    analyze_parser.add_argument("output_json", type=Path)

    args = parser.parse_args()
    if args.command == "protocol":
        _print(
            {"protocol_fingerprint": protocol_fingerprint()}
            if args.fingerprint_only
            else protocol_mapping()
        )
    elif args.command == "build-repeat96-workspace":
        _print(
            build_criterion_review_packets(
                args.review_bundle_dir,
                args.output_dir,
                seed=args.seed,
                task_block_size=args.task_block_size,
            )
        )
    elif args.command == "validate-pass":
        _print(validate_criterion_judgment_pass(_read_object(args.judgment_json)))
    elif args.command == "append-pass":
        _print(
            append_criterion_judgment_pass(
                args.benchmark_dir,
                _read_object(args.judgment_json),
            )
        )
    elif args.command == "rebuild":
        _print(
            {
                "completed_packets": len(
                    rebuild_criterion_packet_judgments(args.benchmark_dir)
                )
            }
        )
    elif args.command == "backtest-repeat96":
        _print(
            backtest_v2_repeat_bundle(
                args.review_bundle_dir,
                args.output_dir,
                build_packets=not args.skip_packet_build,
            )
        )
    elif args.command == "analyze-repeat96":
        _print(
            analyze_fresh_criterion_repeats(
                args.workspace_dir,
                output_path=args.output_json,
            )
        )


if __name__ == "__main__":
    main()
