#!/usr/bin/env python3
"""Compare the supplied rank-32/rank-64 scored prediction JSONLs.

Usage: python analyze_predictions.py PREDICTION_DIRECTORY
The script writes only derived data beside itself; it does not rescore or edit IR.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

BENCHMARKS = ("bixby50", "golden32", "golden35")
ROOT = Path(__file__).resolve().parent


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_present(rows: list[dict], name: str) -> float | None:
    values = [row["metrics"].get(name) for row in rows]
    numbers = [value for value in values if type(value) in (int, float)]
    return statistics.mean(numbers) if numbers else None


def aggregate(rows: list[dict], path: Path) -> dict:
    valid = [row for row in rows if row["metrics"]["schema_valid_strict"]]
    return {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "unique_ids": len({row["id"] for row in rows}),
        "unique_sources": len({row["source_id"] for row in rows}),
        "strict_valid": len(valid),
        "fully_root_reachable": sum(bool(row["metrics"]["fully_root_reachable_v5_4"]) for row in rows),
        "valid_but_unreachable": sum(not row["metrics"]["fully_root_reachable_v5_4"] for row in valid),
        "stop_reasons": dict(Counter(row["runtime"]["stop_reason"] for row in rows)),
        "mean_output_tokens": statistics.mean(row["runtime"]["output_tokens"] for row in rows),
        "mean_reward_v5_4": mean_present(rows, "generation_reward_v5_4"),
        "mean_reachability": mean_present(rows, "root_reachable_fraction_v5_4"),
        "mean_content_coverage_valid_only": mean_present(valid, "content_coverage"),
        "mean_dup_rate_valid_only": mean_present(valid, "dup_rate"),
        "inference_devices": dict(Counter(row["runtime"]["inference_device"] for row in rows)),
        "errors": dict(Counter(
            str(row["metrics"].get("scoring_errors_v5_4", [None])[0]).split(":", 2)[-1]
            for row in rows if not row["metrics"]["schema_valid_strict"]
        )),
    }


def case_row(benchmark: str, old: dict, new: dict) -> dict:
    def fidelity(row: dict, name: str):
        return (row["metrics"].get("fidelity_atomics_v5_4") or {}).get(name)

    return {
        "benchmark": benchmark,
        "id": old["id"],
        "source_id": old["source_id"],
        "r32_valid": old["metrics"]["schema_valid_strict"],
        "r64_valid": new["metrics"]["schema_valid_strict"],
        "r32_reachable": old["metrics"]["root_reachable_fraction_v5_4"],
        "r64_reachable": new["metrics"]["root_reachable_fraction_v5_4"],
        "r32_reward": old["metrics"]["generation_reward_v5_4"],
        "r64_reward": new["metrics"]["generation_reward_v5_4"],
        "r32_stop": old["runtime"]["stop_reason"],
        "r64_stop": new["runtime"]["stop_reason"],
        "r32_tokens": old["runtime"]["output_tokens"],
        "r64_tokens": new["runtime"]["output_tokens"],
        "r32_content_fidelity": fidelity(old, "content_unit_fidelity"),
        "r64_content_fidelity": fidelity(new, "content_unit_fidelity"),
        "r32_number_fidelity": fidelity(old, "exact_numbers_dates_units_fbeta"),
        "r64_number_fidelity": fidelity(new, "exact_numbers_dates_units_fbeta"),
        "r32_error": (old["metrics"].get("scoring_errors_v5_4") or [""])[0],
        "r64_error": (new["metrics"].get("scoring_errors_v5_4") or [""])[0],
    }


def main(directory: Path) -> None:
    summary: dict[str, dict] = {}
    paired: list[dict] = []
    for benchmark in BENCHMARKS:
        paths = [directory / f"r{rank}_{benchmark}_scored.jsonl" for rank in (32, 64)]
        old, new = (read_rows(path) for path in paths)
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert all(a[key] == b[key] for key in (
                "id", "response_text_sha256", "source_context_sha256", "expected_sha256"
            )), f"Unmatched source: {a['id']}"
            assert a["runtime"]["prompt_sha256"] == b["runtime"]["prompt_sha256"]
            assert a["runtime"]["generation_policy_sha256"] == b["runtime"]["generation_policy_sha256"]
            paired.append(case_row(benchmark, a, b))
        summary[benchmark] = {
            "r32": aggregate(old, paths[0]),
            "r64": aggregate(new, paths[1]),
            "matched_source_prompt_and_policy": True,
        }
    (ROOT / "prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "paired_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python analyze_predictions.py PREDICTION_DIRECTORY")
    main(Path(sys.argv[1]))
