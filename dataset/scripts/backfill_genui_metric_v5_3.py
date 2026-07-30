#!/usr/bin/env python3
"""Write immutable v5.3 shadow scores without mutating the source run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
    load_v5_3_reward_config,
    score_genui_completion_v5_3,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "median": 0.0, "sd": 0.0}

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = position - lower
        return (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "sd": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "q05": quantile(0.05),
        "q25": quantile(0.25),
        "q75": quantile(0.75),
        "q95": quantile(0.95),
    }


def resolve_run(value: str) -> Path:
    supplied = Path(value)
    return (
        supplied.resolve()
        if supplied.is_absolute() or supplied.exists()
        else (DATASET_ROOT / "data" / "runs" / supplied).resolve()
    )


def load_previous(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    scores = path / "scores.jsonl" if path.is_dir() else path
    return {
        str(row.get("ui_id")): row
        for row in iter_jsonl(scores)
        if row.get("ui_id") is not None
    }


def previous_quality(row: Mapping[str, Any] | None, phase: str) -> float | None:
    if not isinstance(row, Mapping):
        return None
    value = row.get("v5_2")
    if isinstance(value, Mapping):
        value = value.get(phase)
    if isinstance(value, Mapping):
        raw = value.get("quality_0_100")
        return float(raw) if isinstance(raw, (int, float)) else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-sidecar", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    input_path = run_dir / "genui.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Immutable v5.3 sidecar already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    source_hash_before = sha256(input_path)
    previous = load_previous(
        args.previous_sidecar.resolve()
        if args.previous_sidecar is not None
        else None
    )
    config = load_v5_3_reward_config()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for source_row in iter_jsonl(input_path):
        if args.limit > 0 and len(rows) >= args.limit:
            break
        source = str(source_row.get("response_text") or "")
        intent = str(
            source_row.get("intent_bucket") or source_row.get("intent") or ""
        ) or None
        assets = source_row.get("assets")
        expected = (
            source_row.get("expected_ui_contract_v5_3")
            if isinstance(
                source_row.get("expected_ui_contract_v5_3"), Mapping
            )
            else source_row.get("expected_ui_contract")
            if isinstance(source_row.get("expected_ui_contract"), Mapping)
            else None
        )
        raw = source_row.get("genui_raw_completion")
        if raw is None:
            raw = source_row.get("genui_json")
        final = source_row.get("genui_json")
        generation = score_genui_completion_v5_3(
            raw,
            source,
            intent=intent,
            assets=assets,
            expected_ui_contract=expected,
            config=config,
        )
        artifact = score_genui_completion_v5_3(
            final,
            source,
            intent=intent,
            assets=assets,
            expected_ui_contract=expected,
            config=config,
        )
        prior = previous.get(str(source_row.get("ui_id")))
        prior_generation = previous_quality(prior, "generation")
        prior_final = previous_quality(prior, "final")
        rows.append(
            {
                "ui_id": source_row.get("ui_id"),
                "intent": intent,
                "generation_reward_v5_3": breakdown_to_mapping(generation),
                "render_artifact_quality_v5_3": breakdown_to_mapping(
                    artifact
                ),
                "comparison": {
                    "previous_metric_version": "5.2.0"
                    if prior is not None
                    else None,
                    "previous_generation_0_100": prior_generation,
                    "previous_final_0_100": prior_final,
                    "v5_3_minus_v5_2_generation_0_100": (
                        generation.quality_0_100 - prior_generation
                        if prior_generation is not None
                        else None
                    ),
                    "v5_3_minus_v5_2_final_0_100": (
                        artifact.quality_0_100 - prior_final
                        if prior_final is not None
                        else None
                    ),
                },
                "identity": {
                    "metric_fingerprint": artifact.metric_fingerprint,
                    "reward_pipeline_fingerprint": (
                        artifact.reward_pipeline_fingerprint
                    ),
                    **artifact.identity,
                },
                "active_caps": artifact.active_caps,
                "dimensions": artifact.dimensions,
                "atomic_applicability": artifact.atomic_applicability,
                "dynamic_evidence_certification": (
                    artifact.dynamic_evidence_certification
                ),
            }
        )
    if sha256(input_path) != source_hash_before:
        raise RuntimeError("Source genui.jsonl changed during v5.3 rescoring")

    scores_path = output_dir / "scores.jsonl"
    scores_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    generation_values = [
        float(row["generation_reward_v5_3"]["quality_0_100"])
        for row in rows
    ]
    final_values = [
        float(row["render_artifact_quality_v5_3"]["quality_0_100"])
        for row in rows
    ]
    deltas = [
        float(value)
        for row in rows
        if isinstance(
            value := row["comparison"][
                "v5_3_minus_v5_2_final_0_100"
            ],
            (int, float),
        )
        and math.isfinite(float(value))
    ]
    elapsed = time.perf_counter() - started
    aggregate = {
        "metric_name": "GenUI Representation Quality",
        "metric_version": "5.3.0",
        "metric_fingerprint": (
            rows[0]["identity"]["metric_fingerprint"] if rows else None
        ),
        "reward_pipeline_fingerprint": (
            rows[0]["identity"]["reward_pipeline_fingerprint"]
            if rows
            else None
        ),
        "immutable": True,
        "shadow_only": True,
        "calibration_status": "uncalibrated_engineering_score",
        "weight_learning_performed": False,
        "source_run": str(run_dir),
        "source_genui_sha256": source_hash_before,
        "source_files_modified": False,
        "count": len(rows),
        "elapsed_seconds": elapsed,
        "generation": distribution(generation_values),
        "final": distribution(final_values),
        "v5_3_minus_v5_2_final": distribution(deltas),
        "active_cap_rate": (
            sum(bool(row["active_caps"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "native_render_evidence_available": False,
        "human_calibration_performed": False,
    }
    aggregates_path = output_dir / "aggregates.json"
    aggregates_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "metric_version": "5.3.0",
        "metric_fingerprint": aggregate["metric_fingerprint"],
        "reward_pipeline_fingerprint": aggregate[
            "reward_pipeline_fingerprint"
        ],
        "source_run": str(run_dir),
        "source_genui_sha256": source_hash_before,
        "scores_sha256": sha256(scores_path),
        "aggregates_sha256": sha256(aggregates_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
