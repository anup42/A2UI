from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.data.url_preprocess import restore_url_placeholders
from ir_training.generation_policy import sha256_text, stop_express_completion
from ir_training.eval.generate import aggregate_generation_performance, prediction_source_context_hash
from ir_training.eval.metrics import (
    aggregate_scores,
    load_baseline_aggregate,
    load_dataset_weights,
    score_prediction,
)


def evaluate_predictions(
    predictions_path: str | Path,
    output_dir: str | Path | None = None,
    weights_config_path: str | Path | None = None,
    baseline_aggregate_path: str | Path | None = None,
    metric_version: str | None = None,
) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    for row in read_jsonl(predictions_path):
        if row.get("source_context_sha256") and row["source_context_sha256"] != prediction_source_context_hash(row):
            raise ValueError(f"Scoring context hash mismatch for row {row.get('id')}")
        url_map = row.get("url_map") if isinstance(row.get("url_map"), dict) else {}
        response_text = str(restore_url_placeholders(row.get("response_text") or row.get("input") or "", url_map))
        if row.get("response_text_sha256") and row["response_text_sha256"] != sha256_text(response_text):
            raise ValueError(f"Scoring source hash mismatch for row {row.get('id')}")
        generated_text = str(row.get("generated_text") or row.get("prediction") or "")
        expected = restore_url_placeholders(
            row.get("expected") or row.get("completion") or row.get("expected_json"),
            url_map,
        )
        if row.get("expected_sha256") and row["expected_sha256"] != sha256_text(str(expected or "")):
            raise ValueError(f"Expected completion hash mismatch for row {row.get('id')}")
        restored_generated_text = str(restore_url_placeholders(generated_text, url_map))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        intent = _first_text(
            row.get("intent_bucket"),
            row.get("intent"),
            metadata.get("intent_bucket"),
            metadata.get("intent"),
        )
        assets_value = row.get("assets")
        if assets_value is None:
            assets_value = metadata.get("assets")
        assets = restore_url_placeholders(assets_value, url_map)
        expected_ui_contract = restore_url_placeholders(
            _first_mapping(
                row.get("expected_ui_contract_v5_4"),
                row.get("expected_ui_contract_v5_3"),
                row.get("expected_ui_contract"),
                metadata.get("expected_ui_contract_v5_4"),
                metadata.get("expected_ui_contract_v5_3"),
                metadata.get("expected_ui_contract"),
            ),
            url_map,
        )
        expected_ui_contract_source = _first_text(
            row.get("expected_ui_contract_v5_4_source"),
            row.get("expected_ui_contract_source"),
            metadata.get("expected_ui_contract_v5_4_source"),
            metadata.get("expected_ui_contract_source"),
        )
        metrics = score_prediction(
            response_text,
            expected,
            restored_generated_text,
            metric_version=metric_version,
            intent=intent,
            assets=assets,
            expected_ui_contract=expected_ui_contract,
            expected_ui_contract_source=expected_ui_contract_source,
        )
        out = dict(row)
        if restored_generated_text != generated_text:
            out["generated_text_restored"] = restored_generated_text
        out["metrics"] = metrics
        raw_text = str(restore_url_placeholders(row.get("raw_generated_text", generated_text), url_map))
        stopped_text = stop_express_completion(raw_text)
        def score_variant(text: str) -> dict[str, Any]:
            if text == restored_generated_text:
                return dict(metrics)
            return score_prediction(response_text, expected, text,
                metric_version=metric_version, intent=intent, assets=assets,
                expected_ui_contract=expected_ui_contract,
                expected_ui_contract_source=expected_ui_contract_source)
        out["raw_metrics"] = score_variant(raw_text)
        out["serving_stopped_metrics"] = score_variant(stopped_text)
        out["serving_stopped_text"] = stopped_text
        out["diagnostic_policy"] = "quote-aware closing sentinel only; no ID or graph repair"
        rows_out.append(out)
    weights = load_dataset_weights(weights_config_path)
    baseline = load_baseline_aggregate(baseline_aggregate_path)
    aggregate = aggregate_scores(rows_out, weights=weights, baseline_aggregate=baseline)
    aggregate.update(repeated_benchmark_scores(rows_out, weights))
    aggregate.update(aggregate_generation_performance(rows_out))
    for label in ("raw", "serving_stopped"):
        aggregate[f"{label}_diagnostics"] = aggregate_scores(
            [{"metrics": row[f"{label}_metrics"]} for row in rows_out], weights=weights)
    aggregate["raw_output_scope"] = "observed runtime output only; an early-stopped run has no raw continuation to reconstruct"
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "scored_predictions.jsonl", rows_out)
        (out_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate


def repeated_benchmark_scores(rows: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
    """Report source-macro scores alongside the requested 32-occurrence score.

    Repeated generations are averaged within their source first. The donor
    consequently has the same weight as each of the other 30 unique sources.
    This is still a development benchmark, not 32 independent test cases.
    """
    declarations = [row.get("benchmark") for row in rows]
    if not any(declarations):
        return {}
    if all(isinstance(item, dict) and item.get("kind") == "fixed_strict_subset" for item in declarations):
        return fixed_subset_score_metadata(rows)
    if not all(isinstance(item, dict) and item.get("kind") == "explicit_repeated_case" for item in declarations):
        raise ValueError("Cannot mix repeated-benchmark and ordinary prediction rows")
    benchmark_ids = {item.get("benchmark_id") for item in declarations}
    if len(benchmark_ids) != 1 or len(rows) != 32 or any(
        item.get("row_count") != 32 or item.get("unique_source_count") != 31 for item in declarations
    ):
        raise ValueError("Repeated benchmark requires one identity, 32 occurrences and 31 unique sources")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source = " ".join(str(row.get("response_text") or "").split())
        if not source:
            raise ValueError("Repeated benchmark prediction is missing its source")
        groups.setdefault(sha256_text(source), []).append(row)
    if len(groups) != 31 or sorted(map(len, groups.values())) != [1] * 30 + [2]:
        raise ValueError("Repeated benchmark source multiplicity differs from 31+1 contract")
    if sum(bool(item.get("is_repeated_occurrence")) for item in declarations) != 1:
        raise ValueError("Repeated benchmark must declare exactly one repeated occurrence")
    unique_rows = []
    for group in groups.values():
        if len({row.get("expected_sha256") or sha256_text(str(row.get("expected"))) for row in group}) != 1:
            raise ValueError("Repeated source has conflicting reference completions")
        metrics = dict(group[0]["metrics"])
        keys = set().union(*(row["metrics"].keys() for row in group))
        for key in keys:
            values = [row["metrics"].get(key, 0) for row in group]
            if all(isinstance(value, (int, float, bool)) for value in values):
                metrics[key] = sum(float(value) for value in values) / len(values)
        unique_rows.append({"metrics": metrics})
    unique = aggregate_scores(unique_rows, weights=weights)
    return {
        "benchmark": {
            "id": next(iter(benchmark_ids)), "kind": "explicit_repeated_case",
            "row_count": 32, "unique_source_count": 31, "duplicate_occurrence_count": 1,
            "independent_test_set": False, "selection_policy": "unique_source_macro",
            "note": "32 rows / 31 unique cases; not comparable to either original Golden32 revision",
        },
        "unique_source_metrics": unique,
        **{f"unique_source_{key}": value for key, value in unique.items() if isinstance(value, (int, float, bool))},
    }


def fixed_subset_score_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain Golden35 identity on scores and reject partial or duplicated runs."""
    from ir_training.data.golden35_subset import APPROVED_SOURCE_IDS, BENCHMARK_ID

    if len(rows) != 35 or any(
        not isinstance(row.get("benchmark"), dict)
        or row["benchmark"].get("benchmark_id") != BENCHMARK_ID
        or row["benchmark"].get("kind") != "fixed_strict_subset"
        or row["benchmark"].get("row_count") != 35
        or row["benchmark"].get("unique_source_count") != 35
        or row["benchmark"].get("is_repeated_occurrence") is not False
        or row["benchmark"].get("occurrence_id") != row.get("id")
        for row in rows
    ):
        raise ValueError("Golden35 predictions require the complete fixed 35-case revision")
    sources = [" ".join(str(row.get("response_text") or "").split()) for row in rows]
    if {row.get("source_id") for row in rows} != set(APPROVED_SOURCE_IDS) or len(set(sources)) != 35 or not all(sources):
        raise ValueError("Golden35 prediction source membership or uniqueness differs")
    return {"benchmark": {
        "id": BENCHMARK_ID, "kind": "fixed_strict_subset", "row_count": 35,
        "unique_source_count": 35, "duplicate_occurrence_count": 0,
        "excluded_source_count": 15, "scoring_policy": "all_35_unique_cases_macro",
        "note": "Fixed strict-valid subset; aggregate is not comparable to the original 50-case cohort",
    }}


def _first_mapping(*values: Any) -> Mapping[str, Any] | None:
    return next((value for value in values if isinstance(value, Mapping)), None)


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return None
