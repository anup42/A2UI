#!/usr/bin/env python3
"""Dry-run or apply the pinned LiteRT v2.2.0 QDQ precision source patch.

This only edits an explicitly supplied LiteRT source checkout with --apply.
The default is a read-only `git apply --check` with a skipped-patch guard.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PATCH = (Path(__file__).resolve().parents[2] /
         "validation/20260929_fp16_rootcause/evidence/upstream/litert_v2.2.0_qdq_fp32.patch")
REQUIRED = (
    "tflite/delegates/gpu/common/tasks/BUILD",
    "tflite/delegates/gpu/common/tasks/quantize_and_dequantize.cc",
    "tflite/delegates/gpu/common/tasks/quantize_and_dequantize.h",
    "tflite/delegates/gpu/common/tasks/quantize_and_dequantize_test_util.cc",
    "tflite/delegates/gpu/common/selectors/simple_selectors.cc",
    "tflite/delegates/gpu/common/selectors/simple_selectors.h",
    "tflite/delegates/gpu/common/selectors/operation_selector.cc",
)


def git_context(checkout: Path) -> tuple[Path, list[str]]:
    probe = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--show-toplevel"],
        text=True, capture_output=True, check=False,
    )
    if probe.returncode:
        return checkout, []
    root = Path(probe.stdout.strip()).resolve()
    prefix = checkout.relative_to(root)
    return root, [f"--directory={prefix.as_posix()}"] if prefix.parts else []


def run_git_apply(cwd: Path, options: list[str], *, check: bool) -> None:
    command = ["git", "apply", "--verbose", *( ["--check"] if check else []),
               *options, str(PATCH)]
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                            check=False)
    output = result.stdout + result.stderr
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if result.returncode:
        raise RuntimeError(f"git apply {'--check ' if check else ''}failed ({result.returncode})")
    if "Skipped patch" in output:
        raise RuntimeError("git apply skipped at least one source file")
    if check and output.count("Checking patch ") != len(REQUIRED):
        raise RuntimeError(f"git apply did not check all {len(REQUIRED)} source files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="root of pinned LiteRT source tree")
    parser.add_argument("--apply", action="store_true", help="edit the supplied checkout after a clean dry-run")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    if not PATCH.is_file():
        parser.error(f"patch missing: {PATCH}")
    missing = [name for name in REQUIRED if not (checkout / name).is_file()]
    if missing:
        parser.error(f"not a complete source root; missing: {', '.join(missing)}")
    cwd, options = git_context(checkout)
    run_git_apply(cwd, options, check=True)
    if args.apply:
        run_git_apply(cwd, options, check=False)
        print("Applied QDQ source patch; native build and device validation remain required.")
    else:
        print("Dry-run passed; source checkout was not changed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
