#!/usr/bin/env python3
"""Write an immutable v5.4 sidecar without modifying a dataset run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


DATASET_ROOT = Path(__file__).resolve().parents[1]
DATASET_SRC = DATASET_ROOT / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.genui_quality import (  # noqa: E402
    breakdown_to_mapping,
    generation_reward_v5_4,
    load_v5_4_reward_config,
    render_artifact_quality_v5_4,
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "median": 0.0, "sd": 0.0}

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "sd": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "q05": quantile(0.05),
        "q95": quantile(0.95),
    }


def _resolve_run(value: str) -> Path:
    supplied = Path(value)
    return (
        supplied.resolve()
        if supplied.is_absolute() or supplied.exists()
        else (DATASET_ROOT / "data" / "runs" / supplied).resolve()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    run_dir = _resolve_run(args.run)
    input_path = run_dir / "genui.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Immutable v5.4 sidecar already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    source_sha256 = _sha256(input_path)
    config = load_v5_4_reward_config()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for source_row in _iter_jsonl(input_path):
        if args.limit > 0 and len(rows) >= args.limit:
            break
        response_text = str(source_row.get("response_text") or "")
        intent = str(
            source_row.get("intent_bucket")
            or source_row.get("intent")
            or ""
        ) or None
        assets = source_row.get("assets")
        persisted = next(
            (
                value
                for key in (
                    "expected_ui_contract_v5_4",
                    "expected_ui_contract_v5_3",
                    "expected_ui_contract",
                )
                for value in [source_row.get(key)]
                if isinstance(value, Mapping)
            ),
            None,
        )
        raw = source_row.get("genui_raw_completion")
        if raw is None:
            raw = source_row.get("genui_json")
        generation = generation_reward_v5_4(
            raw,
            response_text,
            intent=intent,
            assets=assets,
            expected_ui_contract=persisted,
            config=config,
        )
        artifact = render_artifact_quality_v5_4(
            source_row.get("genui_json"),
            response_text,
            intent=intent,
            assets=assets,
            expected_ui_contract=persisted,
            config=config,
        )
        rows.append(
            {
                "ui_id": source_row.get("ui_id"),
                "intent": intent,
                "generation_reward_v5_4": breakdown_to_mapping(generation),
                "render_artifact_quality_v5_4": breakdown_to_mapping(artifact),
                "identity": {
                    "metric_fingerprint": artifact.metric_fingerprint,
                    "reward_pipeline_fingerprint": (
                        artifact.reward_pipeline_fingerprint
                    ),
                    **artifact.identity,
                },
                "active_caps": artifact.active_caps,
                "dimensions": artifact.dimensions,
                "performance": artifact.performance,
            }
        )
    if _sha256(input_path) != source_sha256:
        raise RuntimeError("Source genui.jsonl changed during v5.4 rescoring")

    scores_path = output_dir / "scores.jsonl"
    scores_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    generation_values = [
        float(row["generation_reward_v5_4"]["quality_0_100"])
        for row in rows
    ]
    artifact_values = [
        float(row["render_artifact_quality_v5_4"]["quality_0_100"])
        for row in rows
    ]
    aggregate = {
        "metric_name": "GenUI Representation Quality",
        "metric_version": "5.4.0",
        "metric_fingerprint": (
            rows[0]["identity"]["metric_fingerprint"] if rows else None
        ),
        "reward_pipeline_fingerprint": (
            rows[0]["identity"]["reward_pipeline_fingerprint"]
            if rows
            else None
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "shadow_only": True,
        "calibration_status": "uncalibrated_engineering_score",
        "weight_learning_performed": False,
        "source_run": str(run_dir),
        "source_genui_sha256": source_sha256,
        "source_files_modified": False,
        "count": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "generation": _distribution(generation_values),
        "render_artifact": _distribution(artifact_values),
        "native_render_evidence_available": False,
        "human_calibration_performed": False,
    }
    (output_dir / "aggregates.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                **aggregate,
                "scores_sha256": _sha256(scores_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
