from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

_stopwords = {
    "the","a","an","and","or","to","of","in","on","for","with","by","is","are","was","were","be","as","it","this","that"
}


def count_tokens(text: str) -> int:
    return len(text.split())


def content_coverage(response_text: str, a2ui_json: Any) -> float:
    text = response_text.lower()
    words = [w for w in re.findall(r"[a-z0-9]+", text) if w not in _stopwords]
    if not words:
        return 0.0
    a2ui_text = json.dumps(a2ui_json, ensure_ascii=False).lower()
    hits = sum(1 for w in set(words) if w in a2ui_text)
    return hits / max(1, len(set(words)))


def dup_rate(a2ui_json: Any) -> float:
    serialized = json.dumps(a2ui_json, sort_keys=True)
    lines = serialized.split(",")
    if not lines:
        return 0.0
    counts = Counter(lines)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return dup / max(1, len(lines))


def lint_score(a2ui_json: Any) -> float:
    score = 1.0
    penalties = 0.0

    def walk(obj: Any):
        nonlocal penalties
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in {"children", "child", "components"} and not v:
                    penalties += 0.1
                if k in {"text", "label"} and v in (None, ""):
                    penalties += 0.05
                walk(v)
        elif isinstance(obj, list):
            if len(obj) == 0:
                penalties += 0.1
            for item in obj:
                walk(item)

    walk(a2ui_json)
    score = max(0.0, score - penalties)
    return score


def compression_ratio(tokens_json: int, tokens_toon: int) -> float:
    if tokens_json <= 0:
        return 0.0
    return tokens_toon / tokens_json


def compute_overall_score(aggregate: dict[str, Any], weights: dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        value = aggregate.get(f"{key}_rate")
        if value is None:
            value = aggregate.get(f"{key}_avg")
        if value is None:
            value = aggregate.get(key)
        if value is None:
            continue
        score += float(value) * float(weight)
    return score


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = int(math.ceil((p / 100.0) * len(values))) - 1
        k = max(0, min(k, len(values) - 1))
        return values[k]

    metrics = {
        "schema_valid_strict": [1.0 if r.get("validation", {}).get("schema_valid_strict") else 0.0 for r in rows],
        "content_coverage": [r.get("metrics", {}).get("content_coverage", 0.0) for r in rows],
        "lint_score": [r.get("metrics", {}).get("lint_score", 0.0) for r in rows],
        "dup_rate": [r.get("metrics", {}).get("dup_rate", 0.0) for r in rows],
        "latency_ms": [r.get("gen", {}).get("latency_ms", 0.0) for r in rows if r.get("gen")],
        "cost_usd": [r.get("gen", {}).get("cost_usd", 0.0) or 0.0 for r in rows if r.get("gen")],
    }

    return {
        "counts": len(rows),
        "schema_valid_strict_rate": mean(metrics["schema_valid_strict"]),
        "content_coverage_avg": mean(metrics["content_coverage"]),
        "lint_score_avg": mean(metrics["lint_score"]),
        "dup_rate_avg": mean(metrics["dup_rate"]),
        "latency_ms_avg": mean(metrics["latency_ms"]),
        "latency_ms_p95": pct(metrics["latency_ms"], 95),
        "cost_usd_avg": mean(metrics["cost_usd"]),
    }
