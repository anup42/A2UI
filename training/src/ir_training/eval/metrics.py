from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from ir_training.data.filters import FlatSpecValidator

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
OVERALL_SCORE_RAW_MIN = -2.0
OVERALL_SCORE_RAW_MAX = 41.1


def extract_json_text(text: str) -> str:
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    if cleaned.startswith("{"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def score_prediction(response_text: str, expected: Any | None, generated_text: str) -> dict[str, Any]:
    validator = FlatSpecValidator(require_strict=True)
    json_text = extract_json_text(generated_text)
    parsed = None
    parse_error = None
    try:
        parsed = json.loads(json_text)
    except Exception as exc:
        parse_error = str(exc)
    validation = validator.validate(parsed) if parsed is not None else None
    metrics: dict[str, Any] = {
        "json_parse_ok": parsed is not None,
        "json_parse_error": parse_error,
        "schema_valid_strict": bool(validation and validation.valid),
        "schema_error": None if validation is None else validation.reason,
        "markdown_fence_leakage": "```" in generated_text,
        "top_level_keys_ok": isinstance(parsed, dict) and {"root", "elements"}.issubset(parsed.keys()),
        "output_chars": len(generated_text),
    }
    if parsed is not None:
        try:
            from pipeline.metrics import compute_ui_metrics, content_coverage, dup_rate, lint_score  # type: ignore
            metrics.update(compute_ui_metrics(response_text, parsed))
            metrics["content_coverage"] = content_coverage(response_text, parsed)
            metrics["dup_rate"] = dup_rate(parsed)
            metrics["lint_score"] = lint_score(parsed)
        except Exception:
            pass
    if expected is not None and parsed is not None:
        metrics["exact_match"] = parsed == expected
    return metrics


def aggregate_scores(
    rows: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    baseline_aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        out: dict[str, Any] = {"count": 0}
        if weights:
            out["overall_score"] = compute_training_overall_score(out, weights)
        return out
    numeric: dict[str, list[float]] = {}
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                numeric.setdefault(key, []).append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
    out = {"count": len(rows)}
    for key, values in sorted(numeric.items()):
        if not values:
            continue
        suffix = _aggregate_suffix(key)
        out[f"{key}{suffix}"] = sum(values) / len(values)
    if weights:
        out["overall_score"] = compute_training_overall_score(out, weights)
        out["overall_score_weights"] = weights
    if baseline_aggregate:
        baseline_score = _as_float(baseline_aggregate.get("overall_score"))
        if baseline_score is not None:
            out["baseline_overall_score"] = baseline_score
            if "overall_score" in out:
                out["overall_score_delta_vs_baseline"] = float(out["overall_score"]) - baseline_score
    return out


def load_dataset_weights(weights_config_path: str | Path | None = None) -> dict[str, float]:
    root = Path(__file__).resolve().parents[4]
    default_path = root / "dataset" / "configs" / "run.yaml"
    path = Path(weights_config_path).resolve() if weights_config_path else default_path
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    evaluation = data.get("evaluation") if isinstance(data, dict) else None
    weights = evaluation.get("weights") if isinstance(evaluation, dict) else None
    if not isinstance(weights, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in weights.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def load_baseline_aggregate(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def compute_training_overall_score(aggregate: dict[str, Any], weights: dict[str, float]) -> float:
    dataset_score = _dataset_compute_overall_score(aggregate, weights)
    if dataset_score is not None:
        return dataset_score
    raw_score = 0.0
    for key, weight in weights.items():
        value = aggregate.get(f"{key}_rate")
        if value is None:
            value = aggregate.get(f"{key}_avg")
        if value is None:
            value = aggregate.get(key)
        if value is None:
            continue
        raw_score += float(value) * float(weight)
    normalized = ((raw_score - OVERALL_SCORE_RAW_MIN) / (OVERALL_SCORE_RAW_MAX - OVERALL_SCORE_RAW_MIN)) * 100.0
    return max(0.0, min(100.0, normalized))


def _dataset_compute_overall_score(aggregate: dict[str, Any], weights: dict[str, float]) -> float | None:
    dataset_src = Path(__file__).resolve().parents[4] / "dataset" / "src"
    if dataset_src.exists() and str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))
    try:
        from pipeline.metrics import compute_overall_score  # type: ignore
        return float(compute_overall_score(aggregate, weights))
    except Exception:
        return None


def _aggregate_suffix(key: str) -> str:
    rate_like = (
        key.endswith("ok")
        or key.endswith("valid_strict")
        or key.endswith("leakage")
        or key.endswith("presence")
        or key.endswith("detected")
        or key.endswith("pass")
    )
    return "_rate" if rate_like else "_avg"


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
