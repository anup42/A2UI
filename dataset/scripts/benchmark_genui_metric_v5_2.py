from __future__ import annotations

import argparse
import json
import os
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
    prepare_source_context_v5_2,
    score_completion_group_v5_2,
)
from pipeline.genui_quality.metrics_v5_2 import (  # noqa: E402
    content_fidelity_v5_2,
)


SIZES = (8, 32, 100, 257, 300, 1000)
MATCHING_ARGS = {
    "exact_dense_limit": 64,
    "max_edges": 65536,
    "top_k": 16,
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values, default=0.0),
    }


def timed(call: Callable[[], Any], iterations: int) -> tuple[list[float], Any]:
    values: list[float] = []
    result: Any = None
    for _ in range(iterations):
        started = time.perf_counter()
        result = call()
        values.append((time.perf_counter() - started) * 1000.0)
    return values, result


def spec_for(values: list[str]) -> dict[str, Any]:
    children = [f"text_{index}" for index in range(len(values))]
    return {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {"direction": "vertical"},
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
    }


def benchmark_size(size: int, iterations: int) -> dict[str, Any]:
    values = [
        f"Unit {index} exact value {index * 104729}"
        for index in range(size)
    ]
    # Duplicates deliberately create identical cheap-score ties while exact
    # occurrence-preserving preallocation remains the correct solution.
    tied = [f"Repeated bucket {index % 7}" for index in range(size)]
    candidate = spec_for(values)
    source = "\n".join(values)
    prepared = prepare_source_context_v5_2(source)

    exact_times, exact = timed(
        lambda: content_fidelity_v5_2(
            values, list(reversed(values)), **MATCHING_ARGS
        ),
        iterations,
    )
    tie_times, ties = timed(
        lambda: content_fidelity_v5_2(
            tied, list(reversed(tied)), **MATCHING_ARGS
        ),
        iterations,
    )
    scalar_times, scalar = timed(
        lambda: score_completion_group_v5_2([candidate], prepared)[0],
        iterations,
    )
    group_iterations = max(1, iterations // 4)
    group_times, group = timed(
        lambda: score_completion_group_v5_2(
            [candidate] * 8, prepared
        ),
        group_iterations,
    )

    tracemalloc.start()
    score_completion_group_v5_2([candidate] * 8, prepared)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    exact_diag = exact[1]["matching"]
    tie_diag = ties[1]["matching"]
    return {
        "size": size,
        "matching_exact": {
            "latency": distribution(exact_times),
            "fidelity": exact[0]["content_unit_fidelity"],
            **exact_diag,
        },
        "matching_tie_heavy_exact": {
            "latency": distribution(tie_times),
            "fidelity": ties[0]["content_unit_fidelity"],
            **tie_diag,
        },
        "scalar_reward": {
            "latency": distribution(scalar_times),
            "quality_0_100": scalar.quality_0_100,
            "matching_certification": scalar.matching_certification,
        },
        "group_of_8": {
            "batch_latency": distribution(group_times),
            "per_completion_p50_ms": (
                percentile(group_times, 0.50) / 8.0
            ),
            "per_completion_p95_ms": (
                percentile(group_times, 0.95) / 8.0
            ),
            "quality_0_100": group[0].quality_0_100,
            "matching_certification": group[0].matching_certification,
        },
        "peak_memory_bytes_group_of_8": peak,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    payload = {
        "metric_version": "5.2.0",
        "iterations": args.iterations,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "sizes": [
            benchmark_size(size, args.iterations) for size in SIZES
        ],
        "notes": [
            "Times are hardware-specific wall-clock measurements.",
            "Correctness and certification are not relaxed for latency.",
            "Tie-heavy exact cases preserve duplicate multiplicity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
