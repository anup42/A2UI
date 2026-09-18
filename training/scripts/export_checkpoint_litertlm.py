#!/usr/bin/env python3
"""Export W32/W16/W8/W4 from an existing SFT checkpoint, without training or tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.pipeline.checkpoint_export import (
    CheckpointExportOptions,
    run_checkpoint_export,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("e2b", "270m"), required=True)
    parser.add_argument(
        "--fit-dir",
        type=Path,
        required=True,
        help="Original fit/ containing training_config.yaml, preparation_report.json and training/",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override checkpoint; default: <fit-dir>/training/best_golden_checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="NEW export directory; never overwritten",
    )
    parser.add_argument(
        "--exporter-python",
        type=Path,
        required=True,
        help="Absolute Python in the isolated compatible LiteRT Torch CPU export environment",
    )
    parser.add_argument(
        "--training-python",
        type=Path,
        default=Path(sys.executable),
        help="Absolute training-environment Python used for merge; default: this interpreter",
    )
    parser.add_argument(
        "--cache-length",
        type=int,
        default=8192,
        help="Export context capacity; must cover the saved prompt + generation budgets",
    )
    parser.add_argument(
        "--allow-experimental-formats",
        action="store_true",
        help="Required to execute W16/W4 exports",
    )
    parser.add_argument(
        "--stage-timeout-seconds",
        type=float,
        default=172800,
        help="Hard deadline per stage (default: 48 hours)",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=10,
        help="Console/file heartbeat interval",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run exports; otherwise print a read-only plan",
    )
    values = vars(parser.parse_args())
    execute = values.pop("execute")
    try:
        result = run_checkpoint_export(
            CheckpointExportOptions(**values), execute=execute
        )
    except (ValueError, OSError, RuntimeError, TypeError) as exc:
        print(f"Checkpoint export failed: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        return 130
    if execute:
        print(
            f"Manifest: {Path(result['plan']['output_dir']) / 'checkpoint_export_manifest.json'}",
            flush=True,
        )
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
