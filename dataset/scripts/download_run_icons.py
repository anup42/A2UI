from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKFILL_SCRIPT = ROOT / "scripts" / "backfill_run_assets.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download or locally backfill Bootstrap/icon media for a dataset run."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scope", choices=["genui", "all"], default="all")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument(
        "--no-local-icon-catalog",
        action="store_true",
        help="Do not use the bundled Bootstrap icon catalog as a CDN fallback.",
    )
    parser.add_argument(
        "--drop-unresolved",
        action="store_true",
        help="Drop unresolved icon metadata instead of keeping it for a later retry.",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(BACKFILL_SCRIPT),
        "--run-id",
        args.run_id,
        "--scope",
        args.scope,
        "--kind",
        "icon",
        "--workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
        "--max-bytes",
        str(args.max_bytes),
    ]
    if args.no_local_icon_catalog:
        cmd.append("--no-local-icon-catalog")
    if args.drop_unresolved:
        cmd.append("--drop-unresolved")
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
