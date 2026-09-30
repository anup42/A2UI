"""Durable two-pass recording for the GenUI Anchored Criterion Judge v3."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .criterion_protocol_v3 import (
    DIMENSIONS,
    JUDGE_AUTHORITY,
    JUDGE_PROTOCOL_VERSION,
    JUDGE_SCHEMA_VERSION,
    compute_criterion_judged_scores,
    compute_dimension_score,
    protocol_fingerprint,
    validate_criterion_judgment_pass,
)


RAW_JUDGMENTS_NAME = "criterion_v3_raw_judgments.jsonl"
PACKET_JUDGMENTS_NAME = "criterion_v3_judgments_by_packet.jsonl"
RUN_STATE_NAME = "criterion_v3_run_state.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            output.append(value)
    return output


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_run_state(root: Path) -> dict[str, Any] | None:
    path = root / RUN_STATE_NAME
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def _ensure_run_state(
    root: Path,
    *,
    judge_model_identifier: str,
) -> dict[str, Any]:
    model = judge_model_identifier.strip()
    if not model:
        raise ValueError("judge_model_identifier is required")
    state = _load_run_state(root)
    if state is not None:
        if state.get("protocol_fingerprint") != protocol_fingerprint():
            raise ValueError("criterion protocol changed after run start")
        if state.get("judge_model_identifier") != model:
            raise ValueError(
                "judge model identifier changed; create a new protocol version"
            )
        return state
    state = {
        "schema_version": "genui_anchored_criterion_run.v3",
        "protocol_version": JUDGE_PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint(),
        "judge_model_identifier": model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
    }
    _atomic_write_json(root / RUN_STATE_NAME, state)
    return state


def combine_criterion_passes(
    screenshot_only: Mapping[str, Any],
    source_conditioned: Mapping[str, Any],
) -> dict[str, Any]:
    left = validate_criterion_judgment_pass(
        screenshot_only,
        expected_pass_type="screenshot_only",
    )
    right = validate_criterion_judgment_pass(
        source_conditioned,
        expected_pass_type="source_conditioned",
    )
    if left["packet_id"] != right["packet_id"]:
        raise ValueError("cannot combine passes for different packets")
    if left["judge_model_identifier"] != right["judge_model_identifier"]:
        raise ValueError("judge model identifier changed within packet")
    left_time = datetime.fromisoformat(
        str(left["judged_at"]).replace("Z", "+00:00")
    )
    right_time = datetime.fromisoformat(
        str(right["judged_at"]).replace("Z", "+00:00")
    )
    if right_time <= left_time:
        raise ValueError(
            "source_conditioned judgment must follow screenshot_only judgment"
        )
    dimensions = {**left["dimensions"], **right["dimensions"]}
    dimension_scores: dict[str, float] = {}
    dimension_breakdown: dict[str, Any] = {}
    incomplete: list[str] = []
    for dimension in DIMENSIONS:
        result = compute_dimension_score(
            dimension,
            dimensions[dimension]["criteria"],
            worst_defect_severity=dimensions[dimension][
                "worst_defect_severity"
            ],
        )
        dimension_breakdown[dimension] = {
            "base_0_100": result.base_0_100,
            "severity_ceiling_0_100": result.severity_ceiling_0_100,
            "score_0_100": result.score_0_100,
            "evidence_complete": result.evidence_complete,
            "criteria": dimensions[dimension]["criteria"],
            "defects": dimensions[dimension]["defects"],
            "worst_defect_severity": dimensions[dimension][
                "worst_defect_severity"
            ],
            "confidence_0_1": dimensions[dimension]["confidence_0_1"],
            "rationale": dimensions[dimension]["rationale"],
        }
        if not result.evidence_complete or not math.isfinite(result.score_0_100):
            incomplete.append(dimension)
        else:
            dimension_scores[dimension] = result.score_0_100
    fatal_findings = sorted(
        set(left["fatal_findings"]) | set(right["fatal_findings"])
    )
    output: dict[str, Any] = {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "authority": JUDGE_AUTHORITY,
        "protocol_version": JUDGE_PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint(),
        "packet_id": left["packet_id"],
        "judge_model_identifier": left["judge_model_identifier"],
        "dimension_breakdown": dimension_breakdown,
        "dimension_scores_0_100": dimension_scores,
        "evidence_complete": not incomplete,
        "incomplete_dimensions": incomplete,
        "fatal_findings": fatal_findings,
        "passes": {
            "screenshot_only": left,
            "source_conditioned": right,
        },
    }
    if "verified_candidate_render_failure" in fatal_findings and any(
        score > 0.0 for score in dimension_scores.values()
    ):
        raise ValueError(
            "verified candidate render failure requires zero on every complete dimension"
        )
    if incomplete:
        output.update(
            {
                "source_representation_0_100": None,
                "rendered_ux_0_100": None,
                "raw_composite_0_100": None,
                "policy_capped_composite_0_100": None,
                "policy_cap_0_100": None,
                "publication_status": "incomplete_evidence",
            }
        )
        return output
    scores = compute_criterion_judged_scores(
        dimension_scores,
        fatal_findings=fatal_findings,
    )
    output.update(
        {
            "source_representation_0_100": scores.source_representation_0_100,
            "rendered_ux_0_100": scores.rendered_ux_0_100,
            "raw_composite_0_100": scores.raw_composite_0_100,
            "policy_capped_composite_0_100": (
                scores.policy_capped_composite_0_100
            ),
            "policy_cap_0_100": scores.policy_cap_0_100,
            "publication_status": "provisional_single_model",
        }
    )
    return output


def rebuild_criterion_packet_judgments(
    benchmark_dir: str | Path,
) -> list[dict[str, Any]]:
    root = Path(benchmark_dir).resolve()
    rows = _read_jsonl(root / RAW_JUDGMENTS_NAME)
    by_packet: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        normalized = validate_criterion_judgment_pass(row)
        packet_id = normalized["packet_id"]
        pass_type = normalized["pass_type"]
        if pass_type in by_packet[packet_id]:
            raise ValueError(f"duplicate {pass_type} judgment for {packet_id}")
        by_packet[packet_id][pass_type] = normalized
    combined: list[dict[str, Any]] = []
    for packet_id in sorted(by_packet):
        passes = by_packet[packet_id]
        if set(passes) == {"screenshot_only", "source_conditioned"}:
            combined.append(
                combine_criterion_passes(
                    passes["screenshot_only"],
                    passes["source_conditioned"],
                )
            )
    _atomic_write_jsonl(root / PACKET_JUDGMENTS_NAME, combined)
    return combined


def append_criterion_judgment_pass(
    benchmark_dir: str | Path,
    judgment: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(benchmark_dir).resolve()
    normalized = validate_criterion_judgment_pass(judgment)
    _ensure_run_state(
        root,
        judge_model_identifier=normalized["judge_model_identifier"],
    )
    raw_path = root / RAW_JUDGMENTS_NAME
    existing = _read_jsonl(raw_path)
    packet_rows = [
        row
        for row in existing
        if str(row.get("packet_id") or "") == normalized["packet_id"]
    ]
    if any(
        str(row.get("pass_type") or "") == normalized["pass_type"]
        for row in packet_rows
    ):
        raise ValueError(
            f"{normalized['pass_type']} already exists for {normalized['packet_id']}"
        )
    if normalized["pass_type"] == "source_conditioned" and not any(
        str(row.get("pass_type") or "") == "screenshot_only"
        for row in packet_rows
    ):
        raise ValueError(
            "screenshot_only pass must be durably saved before source_conditioned"
        )
    if packet_rows and any(
        str(row.get("judge_model_identifier") or "")
        != normalized["judge_model_identifier"]
        for row in packet_rows
    ):
        raise ValueError("judge model identifier changed within packet")
    existing.append(normalized)
    _atomic_write_jsonl(raw_path, existing)
    combined = rebuild_criterion_packet_judgments(root)
    result = next(
        (row for row in combined if row["packet_id"] == normalized["packet_id"]),
        None,
    )
    return {
        "saved_pass": normalized,
        "packet_complete": result is not None,
        "packet_judgment": result,
        "raw_pass_count": len(existing),
        "completed_packet_count": len(combined),
    }


__all__ = [
    "PACKET_JUDGMENTS_NAME",
    "RAW_JUDGMENTS_NAME",
    "RUN_STATE_NAME",
    "append_criterion_judgment_pass",
    "combine_criterion_passes",
    "rebuild_criterion_packet_judgments",
]
