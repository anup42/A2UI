#!/usr/bin/env python3
"""Benchmark scalar and prepared-group GenUI metric v5.1 reward latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc
from typing import Any, Sequence


DATASET_ROOT = Path(__file__).resolve().parents[1]
DATASET_SRC = DATASET_ROOT / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.genui_quality import (  # noqa: E402
    load_v5_1_reward_config,
    prepare_source_context_v5_1,
    score_completion_group_v5_1,
    score_genui_completion_v5_1,
)


def sample(size: int) -> tuple[dict[str, Any], str]:
    children: list[str] = []
    elements: dict[str, Any] = {
        "root": {
            "type": "Stack",
            "props": {"direction": "vertical"},
            "children": children,
        }
    }
    lines: list[str] = []
    for index in range(size):
        element_id = f"text_{index}"
        value = f"Section {index + 1} preserves benchmark value {index * 7}."
        elements[element_id] = {
            "type": "Text",
            "props": {"text": value, "variant": "body"},
            "children": [],
        }
        children.append(element_id)
        lines.append(value)
    return {"root": "root", "state": {}, "elements": elements}, "\n".join(lines)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    share = position - lower
    return ordered[lower] * (1.0 - share) + ordered[upper] * share


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "mean_ms": statistics.fmean(values),
    }


def matching_observation(result: Any) -> tuple[int, bool]:
    edge_count = 0
    complete = bool(result.evidence.get("matching_complete", False))
    for key in (
        "content_assignment",
        "table_matching",
        "action_matching",
        "media_matching",
    ):
        value = result.evidence.get(key)
        if not isinstance(value, dict):
            continue
        matching = value.get("matching")
        if isinstance(matching, dict):
            edge_count += int(matching.get("evaluated_edge_count") or 0)
            complete = complete and bool(matching.get("complete", True))
    return edge_count, complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    config = load_v5_1_reward_config()
    cases: dict[str, Any] = {}
    all_scalar: list[float] = []
    all_edges: list[int] = []
    all_complete: list[bool] = []

    tracemalloc.start()
    for name, size in (("small", 8), ("medium", 32), ("large", 100)):
        spec, source = sample(size)
        score_genui_completion_v5_1(spec, source, config=config)
        timings: list[float] = []
        latest = None
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            latest = score_genui_completion_v5_1(
                spec, source, config=config
            )
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        assert latest is not None
        edge_count, complete = matching_observation(latest)
        all_scalar.extend(timings)
        all_edges.append(edge_count)
        all_complete.append(complete)
        cases[name] = {
            "source_units": size,
            "reachable_elements": size + 1,
            **latency_summary(timings),
            "evaluated_edge_count": edge_count,
            "matching_complete": complete,
        }

    group_spec, group_source = sample(32)
    prepared = prepare_source_context_v5_1(
        group_source, config=config
    )
    group = [group_spec] * 8
    score_completion_group_v5_1(group, prepared)
    group_timings: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        score_completion_group_v5_1(group, prepared)
        group_timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    payload = {
        "metric_version": "5.1.0",
        "calibration_status": "uncalibrated_engineering_score",
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "iterations": args.iterations,
        "p50_scalar_ms": statistics.median(all_scalar),
        "p95_scalar_ms": percentile(all_scalar, 0.95),
        "p50_group_of_8_ms": statistics.median(group_timings),
        "p95_group_of_8_ms": percentile(group_timings, 0.95),
        "peak_memory_bytes": peak_bytes,
        "evaluated_edge_counts": all_edges,
        "matching_complete": all(all_complete),
        "network_calls": 0,
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
