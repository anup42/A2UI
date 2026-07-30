#!/usr/bin/env python3
"""Report deterministic GenUI metric v5 p50/p95 reward latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parents[1]
DATASET_SRC = DATASET_ROOT / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.genui_quality import score_genui_completion  # noqa: E402


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
        value = f"Section {index + 1} preserves value {index * 7}."
        elements[element_id] = {
            "type": "Text",
            "props": {
                "text": value,
                "variant": "h2" if index % 8 == 0 else "body",
            },
            "children": [],
        }
        children.append(element_id)
        lines.append(value)
    return {"root": "root", "state": {}, "elements": elements}, "\n".join(lines)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    share = position - lower
    return ordered[lower] * (1.0 - share) + ordered[upper] * share


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    results: dict[str, Any] = {}
    for name, count in (("small", 4), ("medium", 32), ("large", 128)):
        spec, source = sample(count)
        # Warm schema, fingerprint, and regular-expression caches.
        score_genui_completion(spec, source)
        timings: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            score_genui_completion(spec, source)
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        results[name] = {
            "elements": count + 1,
            "iterations": args.iterations,
            "p50_ms": statistics.median(timings),
            "p95_ms": percentile(timings, 0.95),
            "mean_ms": statistics.fmean(timings),
        }
    payload = {
        "metric_version": "5.0.0",
        "network_calls": 0,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
