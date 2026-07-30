#!/usr/bin/env python3
"""Benchmark warm v5.4 source preparation and scalar/group reward latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_SRC = REPO_ROOT / "dataset" / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.genui_quality import (  # noqa: E402
    prepare_source_context_v5_4,
    score_completion_group_v5_4,
)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values, default=0.0),
    }


def _timed(call: Callable[[], Any], iterations: int) -> tuple[list[float], Any]:
    elapsed: list[float] = []
    result: Any = None
    for _ in range(iterations):
        started = time.perf_counter()
        result = call()
        elapsed.append((time.perf_counter() - started) * 1000.0)
    return elapsed, result


def _fixture(size: int) -> tuple[str, dict[str, Any]]:
    values = [f"Unit {index} exact value {index * 104729}" for index in range(size)]
    children = [f"text_{index}" for index in range(size)]
    return (
        "\n".join(values),
        {
            "root": "root",
            "state": {},
            "elements": {
                "root": {
                    "type": "Column",
                    "props": {},
                    "children": children,
                },
                **{
                    element_id: {
                        "type": "Text",
                        "props": {"text": value},
                        "children": [],
                    }
                    for element_id, value in zip(children, values)
                },
            },
        },
    )


def _benchmark(size: int, iterations: int) -> dict[str, Any]:
    source, candidate = _fixture(size)
    prepare_source_context_v5_4(source)
    preparation, prepared = _timed(
        lambda: prepare_source_context_v5_4(source),
        iterations,
    )
    score_completion_group_v5_4([candidate], prepared)
    score_completion_group_v5_4(
        [candidate] * 8, prepared, generation_mode=True
    )
    scalar, scalar_result = _timed(
        lambda: score_completion_group_v5_4([candidate], prepared)[0],
        iterations,
    )
    group, group_result = _timed(
        lambda: score_completion_group_v5_4(
            [candidate] * 8,
            prepared,
            generation_mode=True,
        ),
        max(1, iterations // 4),
    )
    tracemalloc.start()
    score_completion_group_v5_4([candidate] * 8, prepared)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "size": size,
        "source_preparation": _distribution(preparation),
        "scalar_reward": {
            "latency": _distribution(scalar),
            "quality_0_100": scalar_result.quality_0_100,
        },
        "group_of_8": {
            "latency": _distribution(group),
            "per_completion_p50_ms": _percentile(group, 0.50) / 8.0,
            "quality_0_100": group_result[0].quality_0_100,
        },
        "matching_time_by_domain_ms": scalar_result.performance.get(
            "matching_time_by_domain_ms", {}
        ),
        "dynamic_evidence_ms": scalar_result.performance.get(
            "dynamic_evidence_ms"
        ),
        "ownership_ms": scalar_result.performance.get("ownership_ms"),
        "peak_memory_bytes": peak,
        "budget_exhaustion": scalar_result.performance.get(
            "budget_exhaustion"
        ),
        "source_contract_cache_hit_rate": scalar_result.performance.get(
            "source_contract_cache_hit_rate"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sizes",
        default="8,32,100,300,1000",
        help="Comma-separated visible-unit counts",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    report = {
        "metric_version": "5.4.0",
        "iterations": args.iterations,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "owner_approved_slo": None,
        "observed_review_reference_group_of_8_ms": {
            "100": 281.0,
            "300": 1020.0,
            "1000": 5880.0,
        },
        "results": [_benchmark(size, args.iterations) for size in sizes],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
