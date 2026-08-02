"""Deterministic quality-first selection for the Stage 3 dual-format pilot."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


FORMAT_SUFFIXES = {
    "compact_ir_v2": "_cir2",
    "a2ui_express_v1": "_exp1",
}


def base_ui_id(ui_id: str) -> str:
    for suffix in FORMAT_SUFFIXES.values():
        if ui_id.endswith(suffix):
            return ui_id[: -len(suffix)]
    return ui_id


def quality_score(row: Mapping[str, Any]) -> float | None:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
    for key in (
        "genui_quality_v5_4",
        "genui_quality_v5_3",
        "genui_quality_v5_2",
        "genui_quality_v5",
        "genui_quality_v4",
        "legacy_structural_richness_score",
        "overall_score",
    ):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def token_estimate(row: Mapping[str, Any]) -> int:
    format_metrics = row.get("format_metrics")
    if isinstance(format_metrics, Mapping):
        value = format_metrics.get("estimated_tokens")
        if isinstance(value, (int, float)):
            return max(0, int(value))
    completion = row.get("model_native_output_normalized")
    if completion is None:
        completion = row.get("model_completion_raw", "")
    text = completion if isinstance(completion, str) else json.dumps(
        completion, ensure_ascii=False, separators=(",", ":")
    )
    # Match pipeline.metrics.count_tokens without importing its heavier metric
    # dependency graph: this is a deterministic whitespace-token estimate.
    return len(text.split())


def select_pair(rows: Iterable[Mapping[str, Any]], *, quality_tolerance: float) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("record_status") == "accepted"
        and bool((row.get("validation") or {}).get("schema_valid_strict"))
    ]
    scored = [(row, quality_score(row), token_estimate(row)) for row in candidates]
    scored = [entry for entry in scored if entry[1] is not None]
    if not scored:
        return {
            "selected_ui_id": None,
            "selected_format": None,
            "reason": "no_strictly_valid_scored_output",
            "eligible_ui_ids": [],
        }
    best_quality = max(float(entry[1]) for entry in scored)
    eligible = [
        entry for entry in scored if float(entry[1]) >= best_quality - max(0.0, quality_tolerance)
    ]
    selected = min(
        eligible,
        key=lambda entry: (
            entry[2],
            -float(entry[1]),
            str(entry[0].get("source_format") or ""),
            str(entry[0].get("ui_id") or ""),
        ),
    )
    return {
        "selected_ui_id": selected[0].get("ui_id"),
        "selected_format": selected[0].get("source_format"),
        "reason": "quality_first_tokens_within_tolerance",
        "quality_tolerance": max(0.0, quality_tolerance),
        "best_quality": best_quality,
        "selected_quality": float(selected[1]),
        "selected_estimated_tokens": selected[2],
        "eligible_ui_ids": [str(entry[0].get("ui_id") or "") for entry in eligible],
    }


def finalize_dual_format_pilot(
    genui_path: Path,
    report_path: Path,
    *,
    quality_tolerance: float = 1.0,
) -> dict[str, Any]:
    if not genui_path.exists():
        report = _empty_report(quality_tolerance)
        _write_json_atomic(report_path, report)
        return report

    rows = [json.loads(line) for line in genui_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ui_id = str(row.get("ui_id") or "")
        if row.get("source_format") not in FORMAT_SUFFIXES or not ui_id:
            continue
        groups.setdefault(base_ui_id(ui_id), []).append(row)

    pairs: list[dict[str, Any]] = []
    row_selection: dict[str, dict[str, Any]] = {}
    for group_id in sorted(groups):
        group_rows = groups[group_id]
        formats = {str(row.get("source_format")) for row in group_rows}
        selection = select_pair(group_rows, quality_tolerance=quality_tolerance)
        complete = set(FORMAT_SUFFIXES).issubset(formats)
        pair = {
            "base_ui_id": group_id,
            "complete": complete,
            "formats_present": sorted(formats),
            **selection,
            "outputs": [
                {
                    "ui_id": row.get("ui_id"),
                    "source_format": row.get("source_format"),
                    "record_status": row.get("record_status"),
                    "quality": quality_score(row),
                    "estimated_tokens": token_estimate(row),
                    "codec_identity": row.get("codec_identity"),
                }
                for row in sorted(group_rows, key=lambda item: str(item.get("source_format") or ""))
            ],
        }
        pairs.append(pair)
        for row in group_rows:
            ui_id = str(row.get("ui_id") or "")
            row_selection[ui_id] = {
                "base_ui_id": group_id,
                "pair_complete": complete,
                "selected": ui_id == selection.get("selected_ui_id"),
                "selected_ui_id": selection.get("selected_ui_id"),
                "selected_format": selection.get("selected_format"),
                "reason": selection.get("reason"),
                "quality_tolerance": max(0.0, quality_tolerance),
            }

    if row_selection:
        for row in rows:
            selection = row_selection.get(str(row.get("ui_id") or ""))
            if selection is not None:
                row["dual_format_selection"] = selection
        _write_jsonl_atomic(genui_path, rows)

    report = {
        "report_version": "dual_format_pilot_v1",
        "selection_policy": "strict_validity_then_quality_then_tokens_within_tolerance",
        "quality_tolerance": max(0.0, quality_tolerance),
        "format_pair_count": len(pairs),
        "complete_pair_count": sum(1 for pair in pairs if pair["complete"]),
        "comparable_pair_count": sum(1 for pair in pairs if pair.get("selected_ui_id")),
        "pairs": pairs,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _write_json_atomic(report_path, report)
    return report


def _empty_report(quality_tolerance: float) -> dict[str, Any]:
    return {
        "report_version": "dual_format_pilot_v1",
        "selection_policy": "strict_validity_then_quality_then_tokens_within_tolerance",
        "quality_tolerance": max(0.0, quality_tolerance),
        "format_pair_count": 0,
        "complete_pair_count": 0,
        "comparable_pair_count": 0,
        "pairs": [],
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


__all__ = [
    "FORMAT_SUFFIXES",
    "base_ui_id",
    "finalize_dual_format_pilot",
    "quality_score",
    "select_pair",
    "token_estimate",
]
