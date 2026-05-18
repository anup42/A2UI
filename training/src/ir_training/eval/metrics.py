from __future__ import annotations

import json
import re
from typing import Any

from ir_training.data.filters import FlatSpecValidator

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


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


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    numeric: dict[str, list[float]] = {}
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                numeric.setdefault(key, []).append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
    out: dict[str, Any] = {"count": len(rows)}
    for key, values in sorted(numeric.items()):
        if values:
            suffix = "_rate" if key.endswith("ok") or key.endswith("valid_strict") or key.endswith("leakage") else "_avg"
            out[f"{key}{suffix}"] = sum(values) / len(values)
    return out
