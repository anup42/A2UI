from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.eval.metrics import aggregate_scores, score_prediction


def evaluate_predictions(predictions_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    for row in read_jsonl(predictions_path):
        response_text = str(row.get("response_text") or row.get("input") or "")
        generated_text = str(row.get("generated_text") or row.get("prediction") or "")
        expected = row.get("expected") or row.get("expected_json")
        metrics = score_prediction(response_text, expected, generated_text)
        out = dict(row)
        out["metrics"] = metrics
        rows_out.append(out)
    aggregate = aggregate_scores(rows_out)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "scored_predictions.jsonl", rows_out)
        (out_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate
