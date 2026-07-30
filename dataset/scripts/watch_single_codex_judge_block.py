#!/usr/bin/env python3
"""Host-only bridge that releases pass two after pass one is durably imported."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_judge import (  # noqa: E402
    import_blind_adjudication_batch,
    import_blind_batch,
    prepare_blind_adjudication_batch,
    prepare_blind_batch,
)


def _emit(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "at": datetime.now(timezone.utc).isoformat(),
                **values,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _wait_for_import(
    result_path: Path,
    ready_path: Path,
    importer: Callable[..., dict[str, Any]],
    importer_kwargs: dict[str, Any],
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    last_error_at = 0.0
    while time.monotonic() < deadline:
        if (
            ready_path.is_file()
            and result_path.is_file()
            and result_path.stat().st_size > 0
        ):
            try:
                return importer(**importer_kwargs)
            except (json.JSONDecodeError, OSError, ValueError) as error:
                message = str(error)
                now = time.monotonic()
                if message != last_error or now - last_error_at >= 30.0:
                    _emit(
                        "result_not_ready",
                        result_path=str(result_path),
                        error=message,
                    )
                    last_error = message
                    last_error_at = now
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"timed out waiting for valid result: {result_path}; "
        f"last_error={last_error}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--expected-task-id", required=True)
    parser.add_argument("--screen-manifest", required=True)
    parser.add_argument("--adjudication", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.benchmark_dir).resolve()
    screen_manifest = json.loads(
        Path(args.screen_manifest).read_text(encoding="utf-8")
    )
    screen_result = Path(screen_manifest["result_path"]).absolute()
    screen_ready = Path(screen_manifest["ready_path"]).absolute()
    importer = (
        import_blind_adjudication_batch
        if args.adjudication
        else import_blind_batch
    )
    preparer = (
        prepare_blind_adjudication_batch
        if args.adjudication
        else prepare_blind_batch
    )
    common = {
        "benchmark_dir": root,
        "start": args.start,
        "count": args.count,
    }
    _emit(
        "watch_started",
        start=args.start,
        count=args.count,
        expected_task_id=args.expected_task_id,
        adjudication=bool(args.adjudication),
    )
    screen_receipt = _wait_for_import(
        screen_result,
        screen_ready,
        importer,
        {
            **common,
            "pass_type": "screenshot_only",
            "expected_task_id": args.expected_task_id,
        },
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    _emit("screenshot_imported", receipt=screen_receipt)
    source = preparer(
        **common,
        pass_type="source_conditioned",
    )
    _emit(
        "source_released",
        batch_manifest=source["batch_manifest"],
        result_path=source["result_path"],
    )
    source_manifest = json.loads(
        Path(source["batch_manifest"]).read_text(encoding="utf-8")
    )
    source_receipt = _wait_for_import(
        Path(source["result_path"]),
        Path(source_manifest["ready_path"]),
        importer,
        {
            **common,
            "pass_type": "source_conditioned",
            "expected_task_id": args.expected_task_id,
        },
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    _emit("source_imported", receipt=source_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
