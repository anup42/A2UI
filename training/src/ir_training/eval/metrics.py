from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

OVERALL_SCORE_RAW_MIN = -2.0
OVERALL_SCORE_RAW_MAX = 41.1


def extract_json_text(text: str) -> str:
    """Compatibility helper for offline legacy reports only.

    Active scoring never calls this function: JSON/FlatSpec is not an
    accepted model completion.
    """
    return str(text or "").strip()


def score_prediction(
    response_text: str,
    expected: Any | None,
    generated_text: str,
    repaired_generated_text: str | None = None,
) -> dict[str, Any]:
    dataset_src = Path(__file__).resolve().parents[4] / "dataset" / "src"
    if str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))
    from pipeline.ir_formats import (  # type: ignore
        compile_express_to_wire,
        encode_express_completion,
        validate_express_completion,
    )

    validation = validate_express_completion(generated_text, repaired_generated_text)
    raw_validation = validate_express_completion(generated_text)
    repaired_validation = (
        validate_express_completion(repaired_generated_text)
        if repaired_generated_text is not None
        else None
    )

    def _wire_status(graph: dict[str, Any] | None) -> tuple[bool, str | None]:
        if graph is None:
            return False, None
        try:
            compile_express_to_wire(graph)
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}:{exc}"

    raw_graph = raw_validation.canonical_graph if raw_validation.raw_valid else None
    repaired_graph = (
        repaired_validation.canonical_graph
        if repaired_validation is not None and repaired_validation.raw_valid
        else None
    )
    raw_wire_valid, raw_wire_error = _wire_status(raw_graph)
    repaired_wire_valid, repaired_wire_error = _wire_status(repaired_graph)
    # The primary graph is raw when raw parsing succeeds; otherwise it is the
    # explicitly repaired candidate for downstream quality metrics.  The
    # distinct fields below prevent that fallback from being reported as raw
    # native validity.
    parsed = raw_graph if raw_graph is not None else repaired_graph
    wire_valid = raw_wire_valid if raw_graph is not None else repaired_wire_valid
    wire_error = raw_wire_error if raw_graph is not None else repaired_wire_error
    expected_graph = None
    expected_hash = None
    if expected is not None:
        try:
            expected_text = expected if isinstance(expected, str) else encode_express_completion(expected)
            expected_validation = validate_express_completion(expected_text)
            if expected_validation.raw_valid:
                expected_graph = expected_validation.canonical_graph
                expected_hash = expected_validation.semantic_hash
        except Exception:
            expected_graph = None
    metrics: dict[str, Any] = {
        "express_parse_ok": validation.raw_valid,
        "native_syntax_valid": raw_validation.raw_valid,
        "native_catalog_valid": raw_validation.raw_valid,
        "raw_standard_a2ui_valid": raw_wire_valid,
        "raw_standard_a2ui_error": raw_wire_error,
        "raw_canonical_semantic_valid": raw_graph is not None,
        "repaired_syntax_valid": bool(repaired_validation and repaired_validation.raw_valid),
        "repaired_catalog_valid": bool(repaired_validation and repaired_validation.raw_valid),
        "repaired_standard_a2ui_valid": repaired_wire_valid,
        "repaired_standard_a2ui_error": repaired_wire_error,
        "repaired_canonical_semantic_valid": repaired_graph is not None,
        "repair_applied": validation.repair_applied,
        "schema_valid_strict": raw_validation.raw_valid,
        "schema_error": None if raw_validation.raw_valid else (raw_validation.errors[0] if raw_validation.errors else "express_invalid"),
        "standard_a2ui_valid": wire_valid,
        "standard_a2ui_error": wire_error,
        "canonical_semantic_valid": parsed is not None,
        "semantic_hash": validation.semantic_hash,
        "markdown_fence_leakage": "```" in generated_text,
        "top_level_keys_ok": False,
        # Retain legacy metric names as explicit false aliases so old reports
        # cannot accidentally be interpreted as JSON-native validity.
        "json_parse_ok": False,
        "json_parse_error": "active_format_is_a2ui_express",
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
    if expected_graph is not None and parsed is not None:
        metrics["exact_match"] = validation.semantic_hash == expected_hash
        metrics["semantic_match"] = validation.semantic_hash == expected_hash
    elif expected is not None:
        metrics["exact_match"] = False
        metrics["semantic_match"] = False
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
