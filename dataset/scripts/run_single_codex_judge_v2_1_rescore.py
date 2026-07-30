"""Host CLI for the immutable Single-Codex judge v2.1 reliability rescore."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_judge.reliability_rescore import (  # noqa: E402
    analyze_rescore_repeats,
    finalize_rescore_overlay,
    import_rescore_batch,
    initialize_reliability_rescore,
    prepare_rescore_batch,
    rescore_status,
)


def _emit(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {
                "event": event,
                **values,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _wait_for_file(path: Path, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(2.0)
    raise TimeoutError(f"timed out waiting for {path}")


def _watch_batch(args: argparse.Namespace) -> dict[str, Any]:
    screen_manifest = json.loads(
        Path(args.screen_manifest).read_text(encoding="utf-8")
    )
    screen_result = Path(str(screen_manifest["result_path"]))
    screen_ready = Path(str(screen_manifest["ready_path"]))
    _emit(
        "watch_started",
        start=args.start,
        count=args.count,
        expected_task_id=args.expected_task_id,
        adjudication=bool(args.adjudication),
    )
    _wait_for_file(screen_result, timeout_seconds=args.timeout_seconds)
    _wait_for_file(screen_ready, timeout_seconds=args.timeout_seconds)
    screen_receipt = import_rescore_batch(
        args.rescore_dir,
        start=args.start,
        count=args.count,
        pass_type="screenshot_only",
        expected_task_id=args.expected_task_id,
        adjudication=args.adjudication,
    )
    _emit("screenshot_imported", receipt=screen_receipt)
    source = prepare_rescore_batch(
        args.rescore_dir,
        start=args.start,
        count=args.count,
        pass_type="source_conditioned",
        adjudication=args.adjudication,
    )
    _emit(
        "source_released",
        batch_manifest=source["batch_manifest"],
        result_path=source["result_path"],
    )
    source_manifest = json.loads(
        Path(str(source["batch_manifest"])).read_text(encoding="utf-8")
    )
    _wait_for_file(
        Path(str(source["result_path"])),
        timeout_seconds=args.timeout_seconds,
    )
    _wait_for_file(
        Path(str(source_manifest["ready_path"])),
        timeout_seconds=args.timeout_seconds,
    )
    source_receipt = import_rescore_batch(
        args.rescore_dir,
        start=args.start,
        count=args.count,
        pass_type="source_conditioned",
        expected_task_id=args.expected_task_id,
        adjudication=args.adjudication,
    )
    _emit("source_imported", receipt=source_receipt)
    return source_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init")
    initialize.add_argument("--benchmark-dir", required=True)
    initialize.add_argument("--workspace-dir", required=True)
    initialize.add_argument(
        "--target-milestone",
        type=int,
        action="append",
        dest="target_milestones",
        required=True,
    )

    prepare = commands.add_parser("prepare-batch")
    prepare.add_argument("--rescore-dir", required=True)
    prepare.add_argument("--start", type=int, required=True)
    prepare.add_argument("--count", type=int, required=True)
    prepare.add_argument(
        "--pass-type",
        choices=("screenshot_only", "source_conditioned"),
        required=True,
    )
    prepare.add_argument("--adjudication", action="store_true")

    import_batch = commands.add_parser("import-batch")
    import_batch.add_argument("--rescore-dir", required=True)
    import_batch.add_argument("--start", type=int, required=True)
    import_batch.add_argument("--count", type=int, required=True)
    import_batch.add_argument(
        "--pass-type",
        choices=("screenshot_only", "source_conditioned"),
        required=True,
    )
    import_batch.add_argument("--expected-task-id")
    import_batch.add_argument("--adjudication", action="store_true")

    status = commands.add_parser("status")
    status.add_argument("--rescore-dir", required=True)

    repeat_analysis = commands.add_parser("repeat-analysis")
    repeat_analysis.add_argument("--rescore-dir", required=True)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--rescore-dir", required=True)
    finalize.add_argument(
        "--allow-incomplete-adjudication",
        action="store_true",
    )

    watch = commands.add_parser("watch-batch")
    watch.add_argument("--rescore-dir", required=True)
    watch.add_argument("--start", type=int, required=True)
    watch.add_argument("--count", type=int, required=True)
    watch.add_argument("--expected-task-id", required=True)
    watch.add_argument("--screen-manifest", required=True)
    watch.add_argument("--adjudication", action="store_true")
    watch.add_argument("--timeout-seconds", type=float, default=86400.0)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "init":
        result = initialize_reliability_rescore(
            args.benchmark_dir,
            args.workspace_dir,
            target_milestones=args.target_milestones,
        )
    elif args.command == "prepare-batch":
        result = prepare_rescore_batch(
            args.rescore_dir,
            start=args.start,
            count=args.count,
            pass_type=args.pass_type,
            adjudication=args.adjudication,
        )
    elif args.command == "import-batch":
        result = import_rescore_batch(
            args.rescore_dir,
            start=args.start,
            count=args.count,
            pass_type=args.pass_type,
            expected_task_id=args.expected_task_id,
            adjudication=args.adjudication,
        )
    elif args.command == "status":
        result = rescore_status(args.rescore_dir)
    elif args.command == "repeat-analysis":
        result = analyze_rescore_repeats(args.rescore_dir)
    elif args.command == "finalize":
        result = finalize_rescore_overlay(
            args.rescore_dir,
            allow_incomplete_adjudication=(
                args.allow_incomplete_adjudication
            ),
        )
    elif args.command == "watch-batch":
        result = _watch_batch(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
