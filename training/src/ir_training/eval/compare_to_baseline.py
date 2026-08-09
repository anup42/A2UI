from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.data.url_preprocess import restore_url_placeholders
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
        url_map = row.get("url_map") if isinstance(row.get("url_map"), dict) else {}
        response_text = str(restore_url_placeholders(row.get("response_text") or row.get("input") or "", url_map))
        generated_text = str(row.get("generated_text") or row.get("prediction") or "")
        expected = restore_url_placeholders(
            row.get("expected") or row.get("completion") or row.get("expected_json"),
            url_map,
        )
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
        rows_out.append(out)
    weights = load_dataset_weights(weights_config_path)
    baseline = load_baseline_aggregate(baseline_aggregate_path)
    aggregate = aggregate_scores(rows_out, weights=weights, baseline_aggregate=baseline)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "scored_predictions.jsonl", rows_out)
        (out_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate


def _first_mapping(*values: Any) -> Mapping[str, Any] | None:
    return next((value for value in values if isinstance(value, Mapping)), None)


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return None
