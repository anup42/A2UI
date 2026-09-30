"""Durable judgment recording, repeat adjudication, and finalization."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .protocol import (
    DIMENSIONS,
    JUDGE_AUTHORITY,
    JUDGE_SCHEMA_VERSION,
    compute_judged_scores,
    protocol_fingerprint,
    validate_judgment_pass,
)


RAW_JUDGMENTS_NAME = "raw_judgments.jsonl"
PACKET_JUDGMENTS_NAME = "judgments_by_packet.jsonl"
FINALIZED_NAME = "finalized_groundtruth.jsonl"
REPEAT_ANALYSIS_NAME = "repeat_analysis.jsonl"
RUN_STATE_NAME = "judge_run_state.json"
ADJUDICATIONS_NAME = "pending_adjudications.jsonl"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


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


def _load_packet_map(benchmark_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(
        benchmark_dir / "sealed" / "packet_identity_map.jsonl"
    ) + _read_jsonl(
        benchmark_dir / "sealed" / "adjudication_identity_map.jsonl"
    )
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        packet_id = str(row.get("packet_id") or "")
        if not packet_id or packet_id in output:
            raise ValueError("invalid or duplicate packet identity")
        output[packet_id] = row
    return output


def _load_run_state(benchmark_dir: Path) -> dict[str, Any] | None:
    path = benchmark_dir / RUN_STATE_NAME
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def _ensure_run_state(
    benchmark_dir: Path,
    *,
    model_identifier: str,
) -> dict[str, Any]:
    cleaned_model = model_identifier.strip()
    if not cleaned_model:
        raise ValueError("judge_model_identifier is required")
    state = _load_run_state(benchmark_dir)
    if state is not None:
        if state.get("protocol_fingerprint") != protocol_fingerprint():
            raise ValueError("judge protocol changed after run start")
        if state.get("judge_model_identifier") != cleaned_model:
            raise ValueError(
                "Codex model identifier changed; start a new protocol version"
            )
        return state
    state = {
        "schema_version": "genui_single_codex_judge_run.v2",
        "authority": JUDGE_AUTHORITY,
        "protocol_fingerprint": protocol_fingerprint(),
        "judge_model_identifier": cleaned_model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
    }
    _atomic_write_json(benchmark_dir / RUN_STATE_NAME, state)
    return state


def _combine_passes(
    screenshot_only: Mapping[str, Any],
    source_conditioned: Mapping[str, Any],
) -> dict[str, Any]:
    if screenshot_only["packet_id"] != source_conditioned["packet_id"]:
        raise ValueError("cannot combine passes for different packets")
    dimensions = {
        **dict(screenshot_only["dimensions"]),
        **dict(source_conditioned["dimensions"]),
    }
    scores = compute_judged_scores(dimensions)
    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "authority": JUDGE_AUTHORITY,
        "packet_id": screenshot_only["packet_id"],
        "protocol_fingerprint": protocol_fingerprint(),
        "dimensions": dimensions,
        "source_representation_0_100": scores.source_representation_0_100,
        "rendered_ux_0_100": scores.rendered_ux_0_100,
        "composite_0_100": scores.composite_0_100,
        "confidence_0_1": min(
            float(screenshot_only["confidence_0_1"]),
            float(source_conditioned["confidence_0_1"]),
        ),
        "passes": {
            "screenshot_only": dict(screenshot_only),
            "source_conditioned": dict(source_conditioned),
        },
    }


def rebuild_packet_judgments(
    benchmark_dir: str | Path,
) -> list[dict[str, Any]]:
    root = Path(benchmark_dir).resolve()
    raw = _read_jsonl(root / RAW_JUDGMENTS_NAME)
    by_packet: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in raw:
        normalized = validate_judgment_pass(row)
        packet_id = normalized["packet_id"]
        pass_type = normalized["pass_type"]
        if pass_type in by_packet[packet_id]:
            raise ValueError(
                f"duplicate {pass_type} judgment for packet {packet_id}"
            )
        by_packet[packet_id][pass_type] = normalized
    combined: list[dict[str, Any]] = []
    for packet_id in sorted(by_packet):
        passes = by_packet[packet_id]
        if set(passes) == {"screenshot_only", "source_conditioned"}:
            combined.append(
                _combine_passes(
                    passes["screenshot_only"],
                    passes["source_conditioned"],
                )
            )
    _atomic_write_jsonl(root / PACKET_JUDGMENTS_NAME, combined)
    return combined


def append_judgment_pass(
    benchmark_dir: str | Path,
    judgment: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and durably save one pass without allowing pass reordering."""

    root = Path(benchmark_dir).resolve()
    packet_map = _load_packet_map(root)
    normalized = validate_judgment_pass(judgment)
    packet_id = normalized["packet_id"]
    if packet_id not in packet_map:
        raise ValueError(f"unknown packet_id: {packet_id}")
    capture_manifest = _read_jsonl(
        root / "native_capture_manifest.jsonl"
    )
    ui_id = str(packet_map[packet_id].get("ui_id") or "")
    candidate_render_failure = any(
        str(row.get("ui_id") or "") == ui_id
        and bool(row.get("required", True))
        and row.get("failure_class") == "candidate"
        for row in capture_manifest
    )
    if candidate_render_failure and any(
        abs(float(score)) > 1e-9
        for score in normalized["dimensions"].values()
    ):
        raise ValueError(
            "verified candidate render failures must receive zero on every "
            "dimension"
        )
    model_identifier = normalized["judge_model_identifier"]
    _ensure_run_state(root, model_identifier=model_identifier)

    raw_path = root / RAW_JUDGMENTS_NAME
    existing = _read_jsonl(raw_path)
    packet_rows = [
        row for row in existing if str(row.get("packet_id")) == packet_id
    ]
    if any(
        str(row.get("pass_type")) == normalized["pass_type"]
        for row in packet_rows
    ):
        raise ValueError(
            f"{normalized['pass_type']} already exists for {packet_id}"
        )
    if (
        normalized["pass_type"] == "source_conditioned"
        and not any(
            str(row.get("pass_type")) == "screenshot_only"
            for row in packet_rows
        )
    ):
        raise ValueError(
            "screenshot_only pass must be saved before source_conditioned"
        )
    existing.append(normalized)
    _atomic_write_jsonl(raw_path, existing)
    combined = rebuild_packet_judgments(root)
    result = next(
        (row for row in combined if row["packet_id"] == packet_id),
        None,
    )
    return {
        "saved_pass": normalized,
        "packet_complete": result is not None,
        "packet_judgment": result,
        "raw_pass_count": len(existing),
        "completed_packet_count": len(combined),
    }


def _anchor_band(value: float) -> int:
    if value >= 100.0:
        return 4
    return int(value // 25.0)


def _repeat_trigger(
    original: Mapping[str, Any],
    repeat: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    composite_delta = abs(
        float(original["composite_0_100"])
        - float(repeat["composite_0_100"])
    )
    if composite_delta >= 10.0 - 1e-9:
        reasons.append("composite_delta_at_least_10")
    maximum_dimension_delta = max(
        abs(
            float(original["dimensions"][dimension])
            - float(repeat["dimensions"][dimension])
        )
        for dimension in DIMENSIONS
    )
    if maximum_dimension_delta >= 20.0 - 1e-9:
        reasons.append("dimension_delta_at_least_20")
    if min(
        float(original["confidence_0_1"]),
        float(repeat["confidence_0_1"]),
    ) < 0.75:
        reasons.append("confidence_below_0_75")
    if _anchor_band(float(original["composite_0_100"])) != _anchor_band(
        float(repeat["composite_0_100"])
    ):
        reasons.append("major_anchor_band_crossing")
    return bool(reasons), reasons


def _adjudication_packet_id(original_packet_id: str) -> str:
    return "p_" + _hash_text(
        f"{protocol_fingerprint()}|{original_packet_id}|adjudication"
    )[:24]


def build_repeat_analysis(
    benchmark_dir: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(benchmark_dir).resolve()
    packet_map = _load_packet_map(root)
    base_schedule_count = sum(
        int(row.get("occurrence", 0)) <= 1
        for row in packet_map.values()
    )
    judgments = {
        row["packet_id"]: row for row in rebuild_packet_judgments(root)
    }
    repeats: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    adjudication_identities: list[dict[str, Any]] = []
    for packet_id, identity in sorted(
        packet_map.items(),
        key=lambda item: int(item[1]["schedule_position"]),
    ):
        original_id = identity.get("repeat_of_packet_id")
        if not original_id or int(identity.get("occurrence", 0)) != 1:
            continue
        original = judgments.get(str(original_id))
        repeat = judgments.get(packet_id)
        if original is None or repeat is None:
            continue
        triggered, reasons = _repeat_trigger(original, repeat)
        dimension_deltas = {
            dimension: float(repeat["dimensions"][dimension])
            - float(original["dimensions"][dimension])
            for dimension in DIMENSIONS
        }
        row = {
            "schema_version": "genui_single_codex_repeat.v2",
            "ui_id": identity["ui_id"],
            "original_packet_id": original_id,
            "repeat_packet_id": packet_id,
            "original_schedule_position": packet_map[str(original_id)][
                "schedule_position"
            ],
            "repeat_schedule_position": identity["schedule_position"],
            "spacing": int(identity["schedule_position"])
            - int(packet_map[str(original_id)]["schedule_position"]),
            "original_composite_0_100": original["composite_0_100"],
            "repeat_composite_0_100": repeat["composite_0_100"],
            "signed_composite_delta": float(repeat["composite_0_100"])
            - float(original["composite_0_100"]),
            "absolute_composite_delta": abs(
                float(repeat["composite_0_100"])
                - float(original["composite_0_100"])
            ),
            "dimension_deltas": dimension_deltas,
            "adjudication_required": triggered,
            "adjudication_reasons": reasons,
            "adjudication_packet_id": (
                _adjudication_packet_id(str(original_id))
                if triggered
                else None
            ),
        }
        repeats.append(row)
        if triggered:
            pending_row = {
                "packet_id": row["adjudication_packet_id"],
                "original_packet_id": original_id,
                "repeat_packet_id": packet_id,
                "ui_id": identity["ui_id"],
                "intent_bucket": identity["intent_bucket"],
                "adjudication_reasons": reasons,
                "requires_fresh_task": True,
            }
            pending.append(pending_row)
            adjudication_identities.append(
                {
                    "schedule_position": base_schedule_count
                    + len(adjudication_identities),
                    "packet_id": row["adjudication_packet_id"],
                    "ui_id": identity["ui_id"],
                    "query_id": identity["query_id"],
                    "response_id": identity["response_id"],
                    "intent_bucket": identity["intent_bucket"],
                    "occurrence": 2,
                    "repeat_of_packet_id": original_id,
                    "adjudicates_repeat_packet_id": packet_id,
                }
            )
    _atomic_write_jsonl(root / REPEAT_ANALYSIS_NAME, repeats)
    _atomic_write_jsonl(root / ADJUDICATIONS_NAME, pending)
    _atomic_write_jsonl(
        root / "sealed" / "adjudication_identity_map.jsonl",
        adjudication_identities,
    )
    return repeats, pending


def _median_dimensions(
    judgments: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if len(judgments) != 3:
        raise ValueError("adjudication requires exactly three judgments")
    return {
        dimension: float(
            median(
                float(judgment["dimensions"][dimension])
                for judgment in judgments
            )
        )
        for dimension in DIMENSIONS
    }


def finalize_groundtruth(
    benchmark_dir: str | Path,
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Create 960 unique final rows while retaining raw repeat judgments."""

    root = Path(benchmark_dir).resolve()
    packet_map = _load_packet_map(root)
    selection_by_ui = {
        str(row["ui_id"]): row
        for row in _read_jsonl(root / "selection_manifest.jsonl")
    }
    contracts_by_ui = {
        str(row["ui_id"]): row
        for row in _read_jsonl(root / "expected_contracts.jsonl")
    }
    judgments = {
        row["packet_id"]: row for row in rebuild_packet_judgments(root)
    }
    repeat_rows, pending = build_repeat_analysis(root)
    unresolved = [
        row
        for row in pending
        if str(row["packet_id"]) not in judgments
    ]
    if require_complete and unresolved:
        raise ValueError(
            f"{len(unresolved)} repeat adjudications are incomplete"
        )

    repeat_by_original = {
        str(row["original_packet_id"]): row for row in repeat_rows
    }
    original_identities = [
        identity
        for identity in packet_map.values()
        if int(identity.get("occurrence", 0)) == 0
    ]
    missing_originals = [
        identity["packet_id"]
        for identity in original_identities
        if identity["packet_id"] not in judgments
    ]
    if require_complete and missing_originals:
        raise ValueError(
            f"{len(missing_originals)} original judgments are incomplete"
        )
    finalized: list[dict[str, Any]] = []
    for identity in sorted(
        original_identities,
        key=lambda row: int(row["schedule_position"]),
    ):
        packet_id = str(identity["packet_id"])
        original = judgments.get(packet_id)
        if original is None:
            continue
        dimensions = dict(original["dimensions"])
        finalization = "original"
        repeat_info = repeat_by_original.get(packet_id)
        adjudication_packet_id: str | None = None
        if repeat_info and repeat_info["adjudication_required"]:
            adjudication_packet_id = str(
                repeat_info["adjudication_packet_id"]
            )
            repeat = judgments.get(str(repeat_info["repeat_packet_id"]))
            third = judgments.get(adjudication_packet_id)
            if repeat is not None and third is not None:
                dimensions = _median_dimensions([original, repeat, third])
                finalization = "median_of_three"
            elif require_complete:
                raise ValueError(
                    f"adjudication incomplete for packet {packet_id}"
                )
        scores = compute_judged_scores(dimensions)
        ui_id = str(identity["ui_id"])
        selection = selection_by_ui.get(ui_id)
        contract = contracts_by_ui.get(ui_id)
        if selection is None or contract is None:
            raise ValueError(
                f"missing selection or expected-contract identity for {ui_id}"
            )
        finalized.append(
            {
                "schema_version": JUDGE_SCHEMA_VERSION,
                "authority": JUDGE_AUTHORITY,
                "ui_id": ui_id,
                "query_id": identity["query_id"],
                "response_id": identity["response_id"],
                "intent_bucket": identity["intent_bucket"],
                "selection_stratum": selection["selection_stratum"],
                "split": selection["split"],
                "migration_anchor": bool(
                    selection.get("migration_anchor", False)
                ),
                "source_text_sha256": contract["response_text_sha256"],
                "expected_ui_contract_hash": contract[
                    "expected_ui_contract_hash"
                ],
                "original_packet_id": packet_id,
                "repeat_packet_id": (
                    repeat_info["repeat_packet_id"] if repeat_info else None
                ),
                "adjudication_packet_id": adjudication_packet_id,
                "finalization": finalization,
                "dimensions": dimensions,
                "source_representation_0_100": (
                    scores.source_representation_0_100
                ),
                "rendered_ux_0_100": scores.rendered_ux_0_100,
                "composite_0_100": scores.composite_0_100,
                "protocol_fingerprint": protocol_fingerprint(),
            }
        )
    if require_complete and len(finalized) != 960:
        raise ValueError(
            f"expected 960 finalized rows, found {len(finalized)}"
        )
    _atomic_write_jsonl(root / FINALIZED_NAME, finalized)
    return finalized


__all__ = [
    "ADJUDICATIONS_NAME",
    "FINALIZED_NAME",
    "PACKET_JUDGMENTS_NAME",
    "RAW_JUDGMENTS_NAME",
    "REPEAT_ANALYSIS_NAME",
    "RUN_STATE_NAME",
    "append_judgment_pass",
    "build_repeat_analysis",
    "finalize_groundtruth",
    "rebuild_packet_judgments",
]
