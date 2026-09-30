"""Immutable targeted rubric-refresh rescoring for judge reliability failures.

The v2 benchmark remains immutable.  This module creates a v2.1 sidecar that
re-scores selected 80-position milestones and every repeat counterpart touching
those milestones under a refreshed rubric and new protocol fingerprint.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .analysis import _repeat_statistics
from .protocol import (
    COMPOSITE_WEIGHTS,
    DIMENSIONS,
    JUDGE_AUTHORITY,
    PASS_DIMENSIONS,
    RUBRIC_DEFINITIONS,
    SCALE_ANCHORS,
    compute_judged_scores,
    protocol_fingerprint as base_protocol_fingerprint,
)


RESCORE_SCHEMA_VERSION = "genui_single_codex_reliability_rescore.v2.1"
RESCORE_PROTOCOL_VERSION = "genui_single_codex_visual_rubric.v2.1.0"
RESCORE_SELECTION_POLICY_VERSION = "milestone-plus-repeat-counterparts.v1"
RESCORE_PACKET_SCHEMA_VERSION = "genui_single_codex_blinded_packet.v2.1"
RESCORE_JUDGMENT_SCHEMA_VERSION = "genui_single_codex_judgment.v2.1"
EXPECTED_MODEL_IDENTIFIER = "gpt-5.6-sol"
MILESTONE_SIZE = 80
RESCORE_DIR_NAME = "reliability_rescore_v2_1"
RUBRIC_REFRESH_FILENAME = "genui_single_codex_judge_refresh_v2_1.md"
RUBRIC_REFRESH_VERSION = "v2.1"
REPEAT_RELIABILITY_NAME = "repeat_reliability_v2_1.json"
FINALIZED_SCHEMA_VERSION = "genui_single_codex_groundtruth.v2.1"
LABEL_ORIGIN = "rubric_refresh_v2_1"

RAW_NAME = "raw_judgments_v2_1.jsonl"
COMBINED_NAME = "judgments_by_packet_v2_1.jsonl"
SCHEDULE_NAME = "rescore_schedule.jsonl"
IDENTITY_NAME = "rescore_identity_map.jsonl"
ADJUDICATION_IDENTITY_NAME = "adjudication_identity_map.jsonl"
PENDING_NAME = "pending_adjudications_v2_1.jsonl"
REPEAT_NAME = "repeat_analysis_v2_1.jsonl"
OVERALL_REPEAT_NAME = "overall_repeat_analysis_v2_1.jsonl"
FINALIZED_NAME = "finalized_groundtruth_v2_1.jsonl"
MANIFEST_NAME = "rescore_manifest.json"


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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


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


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _hash_file(destination) != _hash_file(source):
            raise FileExistsError(f"rescore asset drift: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def rubric_refresh_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "prompts"
        / RUBRIC_REFRESH_FILENAME
    )


def task_instructions_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "prompts"
        / "genui_single_codex_judge_task_v2.md"
    )


def rescore_protocol_mapping(
    target_milestones: Sequence[int] = (0, 3),
) -> dict[str, Any]:
    milestones = sorted({int(value) for value in target_milestones})
    return {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "protocol_version": RESCORE_PROTOCOL_VERSION,
        "authority": JUDGE_AUTHORITY,
        "base_protocol_fingerprint": base_protocol_fingerprint(),
        "rubric_refresh_sha256": _hash_file(rubric_refresh_path()),
        "selection_policy_version": RESCORE_SELECTION_POLICY_VERSION,
        "target_milestones": milestones,
        "milestone_size": MILESTONE_SIZE,
        "dimensions": list(DIMENSIONS),
        "pass_dimensions": {
            key: list(value) for key, value in PASS_DIMENSIONS.items()
        },
        "weights": dict(COMPOSITE_WEIGHTS),
        "scale_anchors": {
            str(key): value for key, value in SCALE_ANCHORS.items()
        },
        "definitions": dict(RUBRIC_DEFINITIONS),
        "score_increment": 5,
        "ordered_passes": ["screenshot_only", "source_conditioned"],
        "model_identifier": EXPECTED_MODEL_IDENTIFIER,
        "repeat_adjudication": {
            "composite_absolute_delta_trigger": 10.0,
            "dimension_absolute_delta_trigger": 20.0,
            "confidence_below_trigger": 0.75,
            "anchor_band_width": 25.0,
            "adjudicated_value": "per_dimension_median_of_three",
            "otherwise": "retain_refreshed_original",
        },
        "blinding": {
            "metric_scores": False,
            "generator_identity": False,
            "flat_spec_json": False,
            "previous_judgments": False,
            "repeat_identity": False,
        },
    }


def rescore_protocol_fingerprint(
    target_milestones: Sequence[int] = (0, 3),
) -> str:
    return _hash_text(_canonical_json(rescore_protocol_mapping(target_milestones)))


def select_rescore_positions(
    schedule_positions: Sequence[int],
    repeat_rows: Sequence[Mapping[str, Any]],
    target_milestones: Sequence[int],
) -> tuple[list[list[int]], list[int]]:
    """Return milestone groups and all counterpart anchors outside them."""

    available = {int(value) for value in schedule_positions}
    groups: list[list[int]] = []
    base_positions: set[int] = set()
    for milestone in sorted({int(value) for value in target_milestones}):
        start = milestone * MILESTONE_SIZE
        group = list(range(start, start + MILESTONE_SIZE))
        missing = sorted(set(group) - available)
        if missing:
            raise ValueError(
                f"milestone {milestone} is incomplete; missing={missing[:5]}"
            )
        groups.append(group)
        base_positions.update(group)
    selected = set(base_positions)
    for row in repeat_rows:
        original = int(row["original_schedule_position"])
        repeated = int(row["repeat_schedule_position"])
        if original in base_positions or repeated in base_positions:
            selected.add(original)
            selected.add(repeated)
    anchors = sorted(selected - base_positions)
    return groups, anchors


def _stable_shuffle(values: Sequence[int], *, seed_text: str) -> list[int]:
    output = list(values)
    seed = int(_hash_text(seed_text)[:16], 16)
    random.Random(seed).shuffle(output)
    return output


def _base_packet_map(base_root: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(
        base_root / "sealed" / "packet_identity_map.jsonl"
    )
    output = {}
    for row in rows:
        packet_id = str(row.get("packet_id") or "")
        if not packet_id or packet_id in output:
            raise ValueError("invalid base packet identity map")
        output[packet_id] = row
    return output


def initialize_reliability_rescore(
    base_benchmark_dir: str | Path,
    workspace_dir: str | Path,
    *,
    target_milestones: Sequence[int] = (0, 3),
) -> dict[str, Any]:
    base_root = Path(base_benchmark_dir).resolve()
    target = tuple(sorted({int(value) for value in target_milestones}))
    fingerprint = rescore_protocol_fingerprint(target)
    rescore_root = base_root / RESCORE_DIR_NAME
    workspace = Path(workspace_dir).absolute()
    required = [
        base_root / "judge_schedule.jsonl",
        base_root / "repeat_analysis.jsonl",
        base_root / "finalized_groundtruth.jsonl",
        base_root / "sealed" / "packet_identity_map.jsonl",
        base_root / "packets" / "packet_manifest.jsonl",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    schedule = sorted(
        _read_jsonl(base_root / "judge_schedule.jsonl"),
        key=lambda row: int(row["schedule_position"]),
    )
    schedule_by_position = {
        int(row["schedule_position"]): row for row in schedule
    }
    repeat_rows = _read_jsonl(base_root / "repeat_analysis.jsonl")
    milestone_groups, anchors = select_rescore_positions(
        list(schedule_by_position),
        repeat_rows,
        target,
    )
    phase_groups = [
        _stable_shuffle(
            group,
            seed_text=f"{fingerprint}|milestone|{target[index]}",
        )
        for index, group in enumerate(milestone_groups)
    ]
    if anchors:
        phase_groups.append(
            _stable_shuffle(
                anchors,
                seed_text=f"{fingerprint}|anchors",
            )
        )
    packet_map = _base_packet_map(base_root)
    identity_rows: list[dict[str, Any]] = []
    public_schedule: list[dict[str, Any]] = []
    rescore_position = 0
    for phase_index, positions in enumerate(phase_groups):
        for old_position in positions:
            scheduled = schedule_by_position[old_position]
            old_packet_id = str(scheduled["packet_id"])
            identity = packet_map[old_packet_id]
            new_packet_id = (
                "r_"
                + _hash_text(
                    f"{fingerprint}|{old_packet_id}|{rescore_position}"
                )[:24]
            )
            row = {
                "schema_version": RESCORE_SCHEMA_VERSION,
                "rescore_position": rescore_position,
                "phase_index": phase_index,
                "task_milestone_index": rescore_position // MILESTONE_SIZE,
                "old_schedule_position": old_position,
                "packet_id": new_packet_id,
                "base_packet_id": old_packet_id,
                "ui_id": str(identity["ui_id"]),
                "query_id": str(identity["query_id"]),
                "response_id": str(identity["response_id"]),
                "intent_bucket": str(identity["intent_bucket"]),
                "occurrence": int(identity["occurrence"]),
                "repeat_of_base_packet_id": identity.get(
                    "repeat_of_packet_id"
                ),
            }
            identity_rows.append(row)
            public_schedule.append(
                {
                    "rescore_position": rescore_position,
                    "packet_id": new_packet_id,
                    "block_index": rescore_position // 20,
                    "task_milestone_index": (
                        rescore_position // MILESTONE_SIZE
                    ),
                }
            )
            rescore_position += 1
    selection_fingerprint = _hash_text(_canonical_json(identity_rows))
    manifest = {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "protocol_version": RESCORE_PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "base_benchmark_dir": str(base_root),
        "base_protocol_fingerprint": base_protocol_fingerprint(),
        "base_source_sha256": _hash_file(
            base_root / "selected_genui.jsonl"
        ),
        "base_finalized_sha256": _hash_file(
            base_root / "finalized_groundtruth.jsonl"
        ),
        "base_repeat_analysis_sha256": _hash_file(
            base_root / "repeat_analysis.jsonl"
        ),
        "target_milestones": list(target),
        "selection_policy_version": RESCORE_SELECTION_POLICY_VERSION,
        "selection_fingerprint": selection_fingerprint,
        "packet_count": len(identity_rows),
        "phase_counts": [len(group) for group in phase_groups],
        "workspace_dir": str(workspace),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if rescore_root.exists() or workspace.exists():
        existing = _read_json(rescore_root / MANIFEST_NAME)
        if {
            key: existing.get(key)
            for key in (
                "protocol_fingerprint",
                "selection_fingerprint",
                "workspace_dir",
            )
        } != {
            key: manifest.get(key)
            for key in (
                "protocol_fingerprint",
                "selection_fingerprint",
                "workspace_dir",
            )
        }:
            raise ValueError("immutable reliability-rescore initialization drift")
        return existing
    rescore_root.mkdir(parents=True)
    (rescore_root / "sealed").mkdir()
    workspace.mkdir(parents=True)
    (workspace / "batches").mkdir()
    (workspace / "rubric").mkdir()
    (workspace / "handoffs").mkdir()
    _atomic_write_json(rescore_root / MANIFEST_NAME, manifest)
    _atomic_write_jsonl(rescore_root / IDENTITY_NAME, identity_rows)
    _atomic_write_jsonl(rescore_root / SCHEDULE_NAME, public_schedule)
    protocol = rescore_protocol_mapping(target)
    protocol["protocol_fingerprint"] = fingerprint
    _atomic_write_json(
        rescore_root / "frozen_rescore_protocol.json",
        protocol,
    )
    _atomic_write_json(
        workspace / "rubric" / "frozen_rescore_protocol.json",
        protocol,
    )
    shutil.copy2(
        rubric_refresh_path(),
        rescore_root / rubric_refresh_path().name,
    )
    shutil.copy2(
        rubric_refresh_path(),
        workspace / "rubric" / rubric_refresh_path().name,
    )
    shutil.copy2(
        base_root / "judge_instructions.md",
        workspace / "rubric" / "judge_instructions.md",
    )
    shutil.copy2(
        task_instructions_path(),
        workspace / "rubric" / task_instructions_path().name,
    )
    _atomic_write_jsonl(workspace / "public_schedule.jsonl", public_schedule)
    _atomic_write_json(
        workspace / "workspace_manifest.json",
        {
            "schema_version": RESCORE_SCHEMA_VERSION,
            "workspace_id": "b_" + fingerprint[:20],
            "authority": JUDGE_AUTHORITY,
            "protocol_fingerprint": fingerprint,
            "selection_fingerprint": selection_fingerprint,
            "packet_count": len(identity_rows),
            "source_conditioned_packets_released_globally": False,
            "generator_identity_present": False,
            "created_at": manifest["created_at"],
        },
    )
    return manifest


def _load_rescore(
    rescore_dir: str | Path,
) -> tuple[Path, dict[str, Any], Path]:
    root = Path(rescore_dir).resolve()
    manifest = _read_json(root / MANIFEST_NAME)
    workspace = Path(str(manifest["workspace_dir"])).absolute()
    if not workspace.exists():
        raise FileNotFoundError(workspace)
    target = tuple(int(value) for value in manifest["target_milestones"])
    if manifest["protocol_fingerprint"] != rescore_protocol_fingerprint(target):
        raise ValueError("rescore protocol fingerprint drift")
    base_root = Path(str(manifest["base_benchmark_dir"])).resolve()
    immutable = {
        "base_source_sha256": base_root / "selected_genui.jsonl",
        "base_finalized_sha256": base_root / "finalized_groundtruth.jsonl",
        "base_repeat_analysis_sha256": base_root / "repeat_analysis.jsonl",
    }
    for key, path in immutable.items():
        if manifest[key] != _hash_file(path):
            raise ValueError(f"base artifact drift: {path}")
    return root, manifest, workspace


def _identity_rows(root: Path) -> list[dict[str, Any]]:
    return sorted(
        _read_jsonl(root / IDENTITY_NAME),
        key=lambda row: int(row["rescore_position"]),
    )


def _selected_rows(
    root: Path,
    *,
    start: int,
    count: int,
    adjudication: bool = False,
) -> list[dict[str, Any]]:
    if start < 0 or count <= 0 or count > 20:
        raise ValueError("batch start/count must select 1 to 20 packets")
    source = (
        _read_jsonl(root / ADJUDICATION_IDENTITY_NAME)
        if adjudication
        else _identity_rows(root)
    )
    selected = source[start : start + count]
    if len(selected) != count:
        raise ValueError("batch extends beyond the rescore schedule")
    return selected


def _batch_name(selected: Sequence[Mapping[str, Any]]) -> str:
    first = int(selected[0]["rescore_position"])
    last = int(selected[-1]["rescore_position"])
    return f"batch_{first:04d}_{last:04d}"


def _rewrite_packet(
    source_packet: Path,
    destination_packet: Path,
    destination_assets: Path,
    *,
    new_packet_id: str,
    fingerprint: str,
) -> dict[str, Any]:
    value = _read_json(source_packet)
    rewritten = deepcopy(value)
    rewritten["schema_version"] = RESCORE_PACKET_SCHEMA_VERSION
    rewritten["packet_id"] = new_packet_id
    rewritten["protocol_version"] = RESCORE_PROTOCOL_VERSION
    rewritten["protocol_fingerprint"] = fingerprint
    rewritten["rubric_refresh_version"] = RUBRIC_REFRESH_VERSION
    rubric = dict(rewritten.get("rubric") or {})
    rubric["refresh_version"] = RUBRIC_REFRESH_VERSION
    rubric["cross_viewport_policy"] = (
        "Judge the evidence set; separate source coverage from responsive loss."
    )
    rewritten["rubric"] = rubric
    for index, capture in enumerate(rewritten.get("screenshots", [])):
        source_asset = (
            source_packet.parent / str(capture["asset"])
        ).resolve()
        suffix = source_asset.suffix.lower() or ".png"
        asset_name = f"{new_packet_id}_{index:02d}{suffix}"
        _link_or_copy(source_asset, destination_assets / asset_name)
        capture["asset"] = f"../assets/{asset_name}"
    source_overview = (
        source_packet.parent / str(rewritten["overview_asset"])
    ).resolve()
    overview_name = f"{new_packet_id}_overview{source_overview.suffix.lower()}"
    _link_or_copy(source_overview, destination_assets / overview_name)
    rewritten["overview_asset"] = f"../assets/{overview_name}"
    if destination_packet.exists():
        if _read_json(destination_packet) != rewritten:
            raise ValueError(f"immutable rescore packet drift: {destination_packet}")
    else:
        _atomic_write_json(destination_packet, rewritten)
    return rewritten


def _raw_by_key(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["packet_id"]), str(row["pass_type"])): row
        for row in _read_jsonl(root / RAW_NAME)
    }


def prepare_rescore_batch(
    rescore_dir: str | Path,
    *,
    start: int,
    count: int,
    pass_type: str,
    adjudication: bool = False,
) -> dict[str, Any]:
    root, manifest, workspace = _load_rescore(rescore_dir)
    if pass_type not in PASS_DIMENSIONS:
        raise ValueError(f"unsupported pass_type: {pass_type}")
    selected = _selected_rows(
        root,
        start=start,
        count=count,
        adjudication=adjudication,
    )
    existing = _raw_by_key(root)
    if pass_type == "source_conditioned":
        missing = [
            str(row["packet_id"])
            for row in selected
            if (str(row["packet_id"]), "screenshot_only") not in existing
        ]
        if missing:
            raise ValueError(
                "source packets remain sealed until screenshot import; "
                f"missing={len(missing)}"
            )
    batch_dir = workspace / "batches" / _batch_name(selected)
    packet_dir = batch_dir / pass_type
    asset_dir = batch_dir / "assets"
    result_dir = batch_dir / "results"
    rubric_dir = batch_dir / "rubric"
    for directory in (packet_dir, asset_dir, result_dir, rubric_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "judge_instructions.md",
        task_instructions_path().name,
        rubric_refresh_path().name,
        "frozen_rescore_protocol.json",
    ):
        shutil.copy2(workspace / "rubric" / name, rubric_dir / name)
    base_root = Path(str(manifest["base_benchmark_dir"])).resolve()
    packet_rows: list[dict[str, Any]] = []
    for row in selected:
        packet_id = str(row["packet_id"])
        base_packet_id = str(row["base_packet_id"])
        source_packet = (
            base_root
            / "packets"
            / pass_type
            / f"{base_packet_id}.json"
        )
        destination_packet = packet_dir / f"{packet_id}.json"
        value = _rewrite_packet(
            source_packet,
            destination_packet,
            asset_dir,
            new_packet_id=packet_id,
            fingerprint=str(manifest["protocol_fingerprint"]),
        )
        packet_rows.append(
            {
                "schedule_position": int(row["rescore_position"]),
                "packet_id": packet_id,
                "packet_path": str(destination_packet.absolute()),
                "overview_asset": str(
                    (
                        destination_packet.parent
                        / str(value["overview_asset"])
                    ).absolute()
                ),
            }
        )
    result_path = result_dir / f"{pass_type}.jsonl"
    ready_path = result_dir / f"{pass_type}_READY.json"
    template_path = result_dir / f"{pass_type}_template.jsonl"
    template = [
        {
            "packet_id": row["packet_id"],
            "pass_type": pass_type,
            "protocol_fingerprint": manifest["protocol_fingerprint"],
            "dimensions": {
                name: None for name in PASS_DIMENSIONS[pass_type]
            },
            "confidence_0_1": None,
            "evidence_observations": [],
            "visible_defects": [],
            "cannot_assess": [],
            "rationale": "",
            "judge_task_id": "",
            "judge_model_identifier": "",
            "judged_at": "",
        }
        for row in packet_rows
    ]
    if template_path.exists():
        if _read_jsonl(template_path) != template:
            raise ValueError("immutable rescore template drift")
    else:
        _atomic_write_jsonl(template_path, template)
    batch_manifest = {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "workspace_id": "b_" + str(manifest["protocol_fingerprint"])[:20],
        "pass_type": pass_type,
        "start_position": int(selected[0]["rescore_position"]),
        "end_position_inclusive": int(selected[-1]["rescore_position"]),
        "packet_count": len(packet_rows),
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "judge_instructions_path": str(
            (rubric_dir / "judge_instructions.md").absolute()
        ),
        "task_instructions_path": str(
            (rubric_dir / task_instructions_path().name).absolute()
        ),
        "rubric_refresh_path": str(
            (rubric_dir / rubric_refresh_path().name).absolute()
        ),
        "rubric_path": str(
            (rubric_dir / "frozen_rescore_protocol.json").absolute()
        ),
        "result_path": str(result_path.absolute()),
        "ready_path": str(ready_path.absolute()),
        "result_template_path": str(template_path.absolute()),
        "ordered_pass_enforcement": (
            "Source-conditioned packets are not released until the "
            "screenshot-only results are host-recorded."
        ),
        "packets": packet_rows,
    }
    manifest_path = batch_dir / f"{pass_type}_manifest.json"
    if manifest_path.exists():
        if _read_json(manifest_path) != batch_manifest:
            raise ValueError("immutable rescore batch drift")
    else:
        _atomic_write_json(manifest_path, batch_manifest)
    return {
        "batch_manifest": str(manifest_path),
        "result_path": str(result_path),
        "packet_count": len(packet_rows),
        "pass_type": pass_type,
    }


def _score(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise ValueError(f"{name} must be finite and bounded")
    if abs(score / 5.0 - round(score / 5.0)) > 1e-9:
        raise ValueError(f"{name} must use 5-point increments")
    return score


def validate_rescore_pass(
    value: Mapping[str, Any],
    *,
    fingerprint: str,
    packet_id: str,
    pass_type: str,
    expected_task_id: str | None = None,
) -> dict[str, Any]:
    if str(value.get("packet_id") or "") != packet_id:
        raise ValueError("packet_id mismatch")
    if str(value.get("pass_type") or "") != pass_type:
        raise ValueError("pass_type mismatch")
    if str(value.get("protocol_fingerprint") or "") != fingerprint:
        raise ValueError("protocol_fingerprint mismatch")
    raw_dimensions = value.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        raise ValueError("dimensions must be an object")
    expected_dimensions = PASS_DIMENSIONS[pass_type]
    if set(raw_dimensions) != set(expected_dimensions):
        raise ValueError("pass dimensions mismatch")
    dimensions = {
        name: _score(raw_dimensions[name], name=name)
        for name in expected_dimensions
    }
    confidence = value.get("confidence_0_1")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("confidence_0_1 must be finite and bounded")

    def strings(name: str) -> list[str]:
        raw = value.get(name, [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError(f"{name} must be an array")
        return [str(item).strip() for item in raw if str(item).strip()][:32]

    rationale = str(value.get("rationale") or "").strip()
    if not rationale:
        raise ValueError("rationale is required")
    task_id = str(value.get("judge_task_id") or "").strip()
    if not task_id:
        raise ValueError("judge_task_id is required")
    if expected_task_id is not None and task_id != expected_task_id:
        raise ValueError("judge_task_id mismatch")
    model = str(value.get("judge_model_identifier") or "").strip()
    if model != EXPECTED_MODEL_IDENTIFIER:
        raise ValueError("judge model identifier drift")
    judged_at = str(value.get("judged_at") or "").strip()
    try:
        parsed = datetime.fromisoformat(judged_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("judged_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("judged_at must include timezone")
    prohibited = {
        "legacy_score",
        "v5_4_score",
        "generator_identity",
        "flat_spec",
        "previous_judgment",
        "repeat_of",
    }
    serialized_keys = {
        str(key).casefold()
        for key in value
    }
    if serialized_keys & prohibited:
        raise ValueError("prohibited blinded field present")
    return {
        "schema_version": RESCORE_JUDGMENT_SCHEMA_VERSION,
        "packet_id": packet_id,
        "pass_type": pass_type,
        "protocol_version": RESCORE_PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "dimensions": dimensions,
        "confidence_0_1": float(confidence),
        "evidence_observations": strings("evidence_observations"),
        "visible_defects": strings("visible_defects"),
        "cannot_assess": strings("cannot_assess"),
        "rationale": rationale,
        "judge_task_id": task_id,
        "judge_model_identifier": model,
        "judged_at": judged_at,
    }


def _combined(root: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _read_jsonl(root / RAW_NAME):
        grouped.setdefault(str(row["packet_id"]), {})[
            str(row["pass_type"])
        ] = row
    output: dict[str, dict[str, Any]] = {}
    combined_rows: list[dict[str, Any]] = []
    for packet_id, passes in grouped.items():
        if set(passes) != {"screenshot_only", "source_conditioned"}:
            continue
        dimensions = {
            **dict(passes["screenshot_only"]["dimensions"]),
            **dict(passes["source_conditioned"]["dimensions"]),
        }
        scores = compute_judged_scores(dimensions)
        row = {
            "schema_version": RESCORE_SCHEMA_VERSION,
            "packet_id": packet_id,
            "protocol_fingerprint": passes["screenshot_only"][
                "protocol_fingerprint"
            ],
            "dimensions": dimensions,
            "source_representation_0_100": (
                scores.source_representation_0_100
            ),
            "rendered_ux_0_100": scores.rendered_ux_0_100,
            "composite_0_100": scores.composite_0_100,
            "confidence_0_1": min(
                float(passes["screenshot_only"]["confidence_0_1"]),
                float(passes["source_conditioned"]["confidence_0_1"]),
            ),
            "passes": passes,
        }
        output[packet_id] = row
        combined_rows.append(row)
    _atomic_write_jsonl(
        root / COMBINED_NAME,
        sorted(combined_rows, key=lambda row: str(row["packet_id"])),
    )
    return output


def import_rescore_batch(
    rescore_dir: str | Path,
    *,
    start: int,
    count: int,
    pass_type: str,
    expected_task_id: str | None = None,
    adjudication: bool = False,
) -> dict[str, Any]:
    root, manifest, workspace = _load_rescore(rescore_dir)
    selected = _selected_rows(
        root,
        start=start,
        count=count,
        adjudication=adjudication,
    )
    batch_dir = workspace / "batches" / _batch_name(selected)
    result_path = batch_dir / "results" / f"{pass_type}.jsonl"
    ready_path = batch_dir / "results" / f"{pass_type}_READY.json"
    receipt_path = (
        batch_dir / "results" / f"{pass_type}_host_import.json"
    )
    if not result_path.exists() and receipt_path.exists():
        return _read_json(receipt_path)
    if not result_path.exists() or not ready_path.exists():
        raise FileNotFoundError("result and READY marker are both required")
    rows = _read_jsonl(result_path)
    expected_ids = [str(row["packet_id"]) for row in selected]
    if [str(row.get("packet_id") or "") for row in rows] != expected_ids:
        raise ValueError("result rows must exactly match manifest order")
    normalized = [
        validate_rescore_pass(
            row,
            fingerprint=str(manifest["protocol_fingerprint"]),
            packet_id=expected_ids[index],
            pass_type=pass_type,
            expected_task_id=expected_task_id,
        )
        for index, row in enumerate(rows)
    ]
    existing = _raw_by_key(root)
    if pass_type == "source_conditioned":
        for row in normalized:
            screen = existing.get((row["packet_id"], "screenshot_only"))
            if screen is None:
                raise ValueError("source pass imported before screenshot pass")
            if screen["judge_task_id"] != row["judge_task_id"]:
                raise ValueError("packet passes must use the same task")
            if screen["judge_model_identifier"] != row[
                "judge_model_identifier"
            ]:
                raise ValueError("packet passes must use the same model")
            screen_time = datetime.fromisoformat(
                str(screen["judged_at"]).replace("Z", "+00:00")
            )
            source_time = datetime.fromisoformat(
                str(row["judged_at"]).replace("Z", "+00:00")
            )
            if source_time < screen_time:
                raise ValueError("source pass timestamp precedes screenshot pass")
    raw = _read_jsonl(root / RAW_NAME)
    for row in normalized:
        key = (row["packet_id"], row["pass_type"])
        if key in existing:
            if _canonical_json(existing[key]) != _canonical_json(row):
                raise ValueError("immutable rescore judgment drift")
            continue
        raw.append(row)
        existing[key] = row
    _atomic_write_jsonl(root / RAW_NAME, raw)
    _combined(root)
    archive = (
        root
        / "sealed"
        / "judge_block_results"
        / _batch_name(selected)
        / f"{pass_type}.jsonl"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if _hash_file(archive) != _hash_file(result_path):
            raise ValueError("immutable rescore archive drift")
    else:
        shutil.copy2(result_path, archive)
    receipt = {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "pass_type": pass_type,
        "start_position": int(selected[0]["rescore_position"]),
        "end_position_inclusive": int(selected[-1]["rescore_position"]),
        "result_sha256": _hash_file(result_path),
        "saved_pass_count": len(normalized),
        "judge_task_ids": sorted(
            {str(row["judge_task_id"]) for row in normalized}
        ),
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(receipt_path, receipt)
    result_path.unlink()
    ready_path.unlink()
    if pass_type == "source_conditioned":
        for name in ("assets", "screenshot_only", "source_conditioned"):
            path = batch_dir / name
            if path.is_dir():
                shutil.rmtree(path)
        for template in (batch_dir / "results").glob("*_template.jsonl"):
            template.unlink()
    return receipt


def rescore_status(rescore_dir: str | Path) -> dict[str, Any]:
    root, manifest, _ = _load_rescore(rescore_dir)
    combined = _combined(root)
    primary = _identity_rows(root)
    adjudications = _read_jsonl(root / ADJUDICATION_IDENTITY_NAME)
    return {
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "primary_packet_count": len(primary),
        "primary_completed_count": sum(
            str(row["packet_id"]) in combined for row in primary
        ),
        "adjudication_packet_count": len(adjudications),
        "adjudication_completed_count": sum(
            str(row["packet_id"]) in combined for row in adjudications
        ),
        "raw_pass_count": len(_read_jsonl(root / RAW_NAME)),
    }


def _anchor_band(score: float) -> int:
    if score >= 100.0:
        return 3
    return int(max(0.0, score) // 25.0)


def analyze_rescore_repeats(
    rescore_dir: str | Path,
) -> dict[str, Any]:
    root, manifest, _ = _load_rescore(rescore_dir)
    combined = _combined(root)
    identities = _identity_rows(root)
    by_old_position = {
        int(row["old_schedule_position"]): row for row in identities
    }
    base_root = Path(str(manifest["base_benchmark_dir"])).resolve()
    base_repeats = _read_jsonl(base_root / "repeat_analysis.jsonl")
    selected_pair_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    adjudication_rows: list[dict[str, Any]] = []
    adjudication_start = len(identities)
    for base_row in base_repeats:
        original_position = int(base_row["original_schedule_position"])
        repeat_position = int(base_row["repeat_schedule_position"])
        if (
            original_position not in by_old_position
            or repeat_position not in by_old_position
        ):
            overall_rows.append(dict(base_row))
            continue
        original_identity = by_old_position[original_position]
        repeat_identity = by_old_position[repeat_position]
        original = combined.get(str(original_identity["packet_id"]))
        repeated = combined.get(str(repeat_identity["packet_id"]))
        if original is None or repeated is None:
            raise ValueError("rescore repeat analysis requires complete passes")
        dimension_deltas = {
            name: float(repeated["dimensions"][name])
            - float(original["dimensions"][name])
            for name in DIMENSIONS
        }
        signed = float(repeated["composite_0_100"]) - float(
            original["composite_0_100"]
        )
        reasons: list[str] = []
        if abs(signed) >= 10.0:
            reasons.append("composite_delta_at_least_10")
        if any(abs(value) >= 20.0 for value in dimension_deltas.values()):
            reasons.append("dimension_delta_at_least_20")
        if min(
            float(original["confidence_0_1"]),
            float(repeated["confidence_0_1"]),
        ) < 0.75:
            reasons.append("confidence_below_0_75")
        if _anchor_band(float(original["composite_0_100"])) != _anchor_band(
            float(repeated["composite_0_100"])
        ):
            reasons.append("major_anchor_band_crossing")
        row = {
            "schema_version": RESCORE_SCHEMA_VERSION,
            "ui_id": str(original_identity["ui_id"]),
            "original_rescore_packet_id": str(
                original_identity["packet_id"]
            ),
            "repeat_rescore_packet_id": str(repeat_identity["packet_id"]),
            "original_schedule_position": original_position,
            "repeat_schedule_position": repeat_position,
            "original_composite_0_100": float(
                original["composite_0_100"]
            ),
            "repeat_composite_0_100": float(
                repeated["composite_0_100"]
            ),
            "signed_composite_delta": signed,
            "absolute_composite_delta": abs(signed),
            "dimension_deltas": dimension_deltas,
            "adjudication_required": bool(reasons),
            "adjudication_reasons": reasons,
        }
        selected_pair_rows.append(row)
        replacement = dict(base_row)
        replacement.update(
            {
                "original_composite_0_100": row[
                    "original_composite_0_100"
                ],
                "repeat_composite_0_100": row[
                    "repeat_composite_0_100"
                ],
                "signed_composite_delta": signed,
                "absolute_composite_delta": abs(signed),
                "dimension_deltas": dimension_deltas,
                "adjudication_required": bool(reasons),
                "adjudication_reasons": reasons,
                "rescore_protocol_fingerprint": manifest[
                    "protocol_fingerprint"
                ],
            }
        )
        overall_rows.append(replacement)
        if reasons:
            index = len(pending)
            adjudication_packet_id = (
                "r_"
                + _hash_text(
                    f"{manifest['protocol_fingerprint']}|"
                    f"{original_identity['base_packet_id']}|adjudication"
                )[:24]
            )
            pending_row = {
                "schema_version": RESCORE_SCHEMA_VERSION,
                "ui_id": str(original_identity["ui_id"]),
                "packet_id": adjudication_packet_id,
                "base_packet_id": str(
                    original_identity["base_packet_id"]
                ),
                "original_rescore_packet_id": str(
                    original_identity["packet_id"]
                ),
                "repeat_rescore_packet_id": str(
                    repeat_identity["packet_id"]
                ),
                "adjudication_reasons": reasons,
                "requires_fresh_task": True,
            }
            pending.append(pending_row)
            adjudication_rows.append(
                {
                    **pending_row,
                    "rescore_position": adjudication_start + index,
                    "phase_index": len(
                        manifest.get("phase_counts", [])
                    ),
                    "task_milestone_index": (
                        (adjudication_start + index) // MILESTONE_SIZE
                    ),
                    "old_schedule_position": original_position,
                    "query_id": str(original_identity["query_id"]),
                    "response_id": str(original_identity["response_id"]),
                    "intent_bucket": str(
                        original_identity["intent_bucket"]
                    ),
                    "occurrence": 2,
                    "repeat_of_base_packet_id": str(
                        original_identity["base_packet_id"]
                    ),
                }
            )
    _atomic_write_jsonl(root / REPEAT_NAME, selected_pair_rows)
    _atomic_write_jsonl(root / OVERALL_REPEAT_NAME, overall_rows)
    _atomic_write_jsonl(root / PENDING_NAME, pending)
    _atomic_write_jsonl(
        root / ADJUDICATION_IDENTITY_NAME,
        adjudication_rows,
    )
    reliability = _repeat_statistics(overall_rows)
    result = {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "selected_repeat_pair_count": len(selected_pair_rows),
        "overall_repeat_pair_count": len(overall_rows),
        "pending_adjudication_count": len(pending),
        "repeat_reliability": reliability,
        "acceptance": {
            "icc_at_least_0_85": (
                float(reliability["icc_a_1_absolute_agreement"]) >= 0.85
            ),
            "mae_at_most_5": float(reliability["mae"]) <= 5.0,
            "absolute_bias_below_2": abs(
                float(reliability["signed_bias_prediction_minus_target"])
            )
            < 2.0,
        },
    }
    result["acceptance"]["reliability_pass"] = all(
        result["acceptance"].values()
    )
    _atomic_write_json(root / REPEAT_RELIABILITY_NAME, result)
    return result


def _score_from_dimensions(dimensions: Mapping[str, Any]) -> dict[str, float]:
    scores = compute_judged_scores(dimensions)
    return {
        "source_representation_0_100": (
            scores.source_representation_0_100
        ),
        "rendered_ux_0_100": scores.rendered_ux_0_100,
        "composite_0_100": scores.composite_0_100,
    }


def finalize_rescore_overlay(
    rescore_dir: str | Path,
    *,
    allow_incomplete_adjudication: bool = False,
) -> dict[str, Any]:
    root, manifest, _ = _load_rescore(rescore_dir)
    combined = _combined(root)
    identities = _identity_rows(root)
    by_base_packet = {
        str(row["base_packet_id"]): row
        for row in identities
        if int(row["occurrence"]) == 0
    }
    pending = _read_jsonl(root / PENDING_NAME)
    adjudications = {
        str(row["ui_id"]): combined.get(str(row["packet_id"]))
        for row in _read_jsonl(root / ADJUDICATION_IDENTITY_NAME)
    }
    missing = [
        ui_id for ui_id, value in adjudications.items() if value is None
    ]
    if missing and not allow_incomplete_adjudication:
        raise ValueError(
            f"adjudication judgments remain incomplete: {len(missing)}"
        )
    pending_by_ui = {str(row["ui_id"]): row for row in pending}
    base_root = Path(str(manifest["base_benchmark_dir"])).resolve()
    base_rows = _read_jsonl(base_root / "finalized_groundtruth.jsonl")
    output: list[dict[str, Any]] = []
    replaced = 0
    for base_row in base_rows:
        original_packet_id = str(base_row["original_packet_id"])
        identity = by_base_packet.get(original_packet_id)
        if identity is None:
            row = dict(base_row)
            row["overlay_protocol_fingerprint"] = manifest[
                "protocol_fingerprint"
            ]
            row["label_origin"] = "base_v2"
            output.append(row)
            continue
        original = combined.get(str(identity["packet_id"]))
        if original is None:
            raise ValueError("primary refreshed judgment is incomplete")
        dimensions = dict(original["dimensions"])
        finalization = "refreshed_original"
        pending_row = pending_by_ui.get(str(base_row["ui_id"]))
        third = adjudications.get(str(base_row["ui_id"]))
        if pending_row is not None and third is not None:
            repeat = combined[str(pending_row["repeat_rescore_packet_id"])]
            dimensions = {
                name: float(
                    median(
                        [
                            float(original["dimensions"][name]),
                            float(repeat["dimensions"][name]),
                            float(third["dimensions"][name]),
                        ]
                    )
                )
                for name in DIMENSIONS
            }
            finalization = "refreshed_median_of_three"
        scores = _score_from_dimensions(dimensions)
        row = dict(base_row)
        row.update(
            {
                "schema_version": FINALIZED_SCHEMA_VERSION,
                "protocol_fingerprint": manifest["protocol_fingerprint"],
                "base_protocol_fingerprint": base_row[
                    "protocol_fingerprint"
                ],
                "dimensions": dimensions,
                **scores,
                "finalization": finalization,
                "label_origin": LABEL_ORIGIN,
                "rescore_packet_id": str(identity["packet_id"]),
            }
        )
        output.append(row)
        replaced += 1
    _atomic_write_jsonl(root / FINALIZED_NAME, output)
    result = {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "finalized_count": len(output),
        "replaced_unique_count": replaced,
        "base_retained_count": len(output) - replaced,
        "incomplete_adjudication_count": len(missing),
        "protocol_fingerprint": manifest["protocol_fingerprint"],
    }
    _atomic_write_json(root / "finalization_summary.json", result)
    return result


__all__ = [
    "EXPECTED_MODEL_IDENTIFIER",
    "RESCORE_PROTOCOL_VERSION",
    "analyze_rescore_repeats",
    "finalize_rescore_overlay",
    "import_rescore_batch",
    "initialize_reliability_rescore",
    "prepare_rescore_batch",
    "rescore_protocol_fingerprint",
    "rescore_status",
    "select_rescore_positions",
    "validate_rescore_pass",
]
