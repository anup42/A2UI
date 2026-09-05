#!/usr/bin/env python3
"""Validate completeness and image integrity for an Android golden run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_image(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Missing/empty image: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        extrema = image.convert("L").getextrema()
        if extrema[0] == extrema[1]:
            raise AssertionError(f"Visually blank image: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_dir = REPO_ROOT / "dataset" / "data" / "runs" / args.run_id
    scenarios = read_jsonl(run_dir / "scenarios.jsonl")
    executions = read_jsonl(run_dir / "execution_results.jsonl")
    genui = read_jsonl(run_dir / "genui.jsonl")
    captures = read_jsonl(run_dir / "android_device_rendered" / "capture_manifest.jsonl")
    reviews = read_jsonl(run_dir / "visual_review.jsonl")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    case_ids = [row["case_id"] for row in scenarios]
    if len(case_ids) != len(set(case_ids)):
        raise AssertionError("Duplicate scenario case IDs")
    execution_by_case = {row["case_id"]: row for row in executions}
    if set(execution_by_case) != set(case_ids):
        raise AssertionError("Execution/scenario case IDs differ")

    success_ids = {case_id for case_id, row in execution_by_case.items() if row.get("success") is True}
    failure_ids = set(case_ids) - success_ids
    if {row["gen"]["case_id"] for row in genui} != success_ids:
        raise AssertionError("GenUI rows do not exactly match successful cases")
    if {row["case_id"] for row in reviews} != success_ids:
        raise AssertionError("Visual review rows do not exactly match successful cases")
    if len(captures) != len(success_ids) or not all(
        row.get("ok") is True and row.get("native_render_ok") is True for row in captures
    ):
        raise AssertionError("Capture count/status differs from successful cases")

    for scenario in scenarios:
        case_id = scenario["case_id"]
        case_dir = (
            run_dir
            / "cases"
            / f"{scenario['pair_id']}_{scenario['scenario_slug']}"
            / scenario["variant"]
        )
        for name in ("prompt.txt", "metadata.json", "manual_ui_screenshot.png", "device_logcat.txt"):
            if not (case_dir / name).exists():
                raise AssertionError(f"{case_id} missing {name}")
        check_image(case_dir / "manual_ui_screenshot.png")
        if case_id in success_ids:
            for name in ("response.md", "ir.a2ui", "app_screenshot.png", "viewport_screenshot.png"):
                if not (case_dir / name).exists() or (case_dir / name).stat().st_size == 0:
                    raise AssertionError(f"{case_id} missing/empty {name}")
            check_image(case_dir / "app_screenshot.png")
            check_image(case_dir / "viewport_screenshot.png")

    accepted = [row for row in reviews if row.get("status") == "pass"]
    expected_manifest = {
        "planned_cases": len(case_ids),
        "completed_cases": len(executions),
        "pipeline_successes": len(success_ids),
        "accepted_cases": len(accepted),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise AssertionError(f"Manifest {key}: expected {expected}, found {manifest.get(key)}")

    print(
        json.dumps(
            {
                "planned": len(case_ids),
                "successes": len(success_ids),
                "failures": len(failure_ids),
                "captures": len(captures),
                "reviewed": len(reviews),
                "accepted": len(accepted),
                "status": "ok",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
