"""Leak-resistant workspaces for the blinded Codex judging tasks."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from .judgments import append_judgment_pass
from .protocol import (
    JUDGE_AUTHORITY,
    PASS_DIMENSIONS,
    protocol_fingerprint,
    validate_judgment_pass,
)
from .provenance import verify_implementation_provenance


POINTER_NAME = "blind_judge_workspace.json"
WORKSPACE_SCHEMA_VERSION = "genui_single_codex_blind_workspace.v2"
_GENERATOR_PATH_MARKERS = (
    "azure",
    "claude",
    "gemini",
    "gemma",
    "gpt",
    "llama",
    "qwen",
    "vertex",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _write_jsonl(
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
            raise FileExistsError(
                f"blind-workspace asset drift: {destination}"
            )
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _assert_neutral_workspace_path(path: Path) -> None:
    lowered = str(path).casefold()
    leaked = [
        marker for marker in _GENERATOR_PATH_MARKERS if marker in lowered
    ]
    if leaked:
        raise ValueError(
            "blind workspace path contains generator/model markers: "
            + ", ".join(leaked)
        )


def _load_pointer(benchmark_dir: Path) -> tuple[Path, dict[str, Any]]:
    pointer_path = benchmark_dir / POINTER_NAME
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"initialize the blind judge workspace first: {pointer_path}"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("protocol_fingerprint") != protocol_fingerprint():
        raise ValueError("blind-workspace protocol fingerprint drift")
    workspace = Path(str(pointer["workspace_dir"])).absolute()
    _assert_neutral_workspace_path(workspace)
    manifest = json.loads(
        (workspace / "workspace_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.get("protocol_fingerprint") != protocol_fingerprint():
        raise ValueError("blind workspace manifest fingerprint drift")
    return workspace, pointer


def initialize_blind_workspace(
    benchmark_dir: str | Path,
    workspace_dir: str | Path,
    *,
    require_sealed_implementation: bool = True,
) -> dict[str, Any]:
    """Create a neutral host path without copying source-conditioned packets."""

    root = Path(benchmark_dir).resolve()
    if require_sealed_implementation:
        verify_implementation_provenance(root)
    workspace = Path(workspace_dir).absolute()
    _assert_neutral_workspace_path(workspace)
    if workspace.exists():
        raise FileExistsError(
            f"immutable blind workspace already exists: {workspace}"
        )
    if not (root / "packets" / "packet_manifest.jsonl").exists():
        raise FileNotFoundError("build blinded packets before the workspace")
    workspace.mkdir(parents=True)
    (workspace / "batches").mkdir()
    (workspace / "handoffs").mkdir()
    (workspace / "rubric").mkdir()
    shutil.copy2(
        root / "judge_instructions.md",
        workspace / "rubric" / "judge_instructions.md",
    )
    shutil.copy2(
        root / "judge_protocol.json",
        workspace / "rubric" / "judge_protocol.json",
    )
    task_instructions_source = (
        Path(__file__).resolve().parents[3]
        / "prompts"
        / "genui_single_codex_judge_task_v2.md"
    )
    shutil.copy2(
        task_instructions_source,
        workspace / "rubric" / task_instructions_source.name,
    )
    schedule = _read_jsonl(root / "judge_schedule.jsonl")
    public_schedule = [
        {
            "schedule_position": int(row["schedule_position"]),
            "packet_id": str(row["packet_id"]),
            "pilot": bool(row["pilot"]),
            "block_index": int(row["block_index"]),
            "milestone_index": int(row["milestone_index"]),
        }
        for row in schedule
    ]
    _write_jsonl(workspace / "public_schedule.jsonl", public_schedule)
    workspace_id = "b_" + protocol_fingerprint()[:20]
    manifest = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "authority": JUDGE_AUTHORITY,
        "protocol_fingerprint": protocol_fingerprint(),
        "judge_instructions_sha256": _hash_file(
            workspace / "rubric" / "judge_instructions.md"
        ),
        "task_instructions_sha256": _hash_file(
            workspace
            / "rubric"
            / "genui_single_codex_judge_task_v2.md"
        ),
        "packet_count": len(public_schedule),
        "source_conditioned_packets_released_globally": False,
        "generator_identity_present": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(workspace / "workspace_manifest.json", manifest)
    pointer = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "workspace_dir": str(workspace),
        "protocol_fingerprint": protocol_fingerprint(),
        "created_at": manifest["created_at"],
    }
    _write_json(root / POINTER_NAME, pointer)
    return {
        **manifest,
        "workspace_dir": str(workspace),
    }


def sync_frozen_protocol_to_blind_workspace(
    benchmark_dir: str | Path,
) -> dict[str, Any]:
    root = Path(benchmark_dir).resolve()
    workspace, _ = _load_pointer(root)
    source = root / "frozen_judge_protocol.json"
    if not source.exists():
        raise FileNotFoundError(source)
    destination = workspace / "rubric" / source.name
    if destination.exists():
        if _hash_file(destination) != _hash_file(source):
            raise ValueError("frozen blind rubric changed after publication")
    else:
        shutil.copy2(source, destination)
    return {
        "frozen_protocol": str(destination),
        "sha256": _hash_file(destination),
    }


def _selected_schedule(
    root: Path,
    *,
    start: int,
    count: int,
) -> list[dict[str, Any]]:
    if start < 0 or count <= 0 or count > 20:
        raise ValueError("batch start/count must select 1 to 20 packets")
    schedule = sorted(
        _read_jsonl(root / "judge_schedule.jsonl"),
        key=lambda row: int(row["schedule_position"]),
    )
    if start >= len(schedule):
        raise ValueError("batch start is outside the judging schedule")
    selected = schedule[start : start + count]
    if len(selected) != count:
        raise ValueError("requested batch extends beyond the schedule")
    return selected


def _batch_name(selected: list[dict[str, Any]]) -> str:
    return (
        f"batch_{int(selected[0]['schedule_position']):04d}_"
        f"{int(selected[-1]['schedule_position']):04d}"
    )


def _copy_packet_assets(
    source_packet: Path,
    destination_packet: Path,
    destination_assets: Path,
) -> dict[str, Any]:
    value = json.loads(source_packet.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{source_packet} is not an object")
    destination_packet.parent.mkdir(parents=True, exist_ok=True)
    for capture in value.get("screenshots", []):
        source_asset = (
            source_packet.parent / str(capture["asset"])
        ).resolve()
        _link_or_copy(source_asset, destination_assets / source_asset.name)
    overview = (
        source_packet.parent / str(value["overview_asset"])
    ).resolve()
    _link_or_copy(overview, destination_assets / overview.name)
    if destination_packet.exists():
        if _hash_file(destination_packet) != _hash_file(source_packet):
            raise FileExistsError(
                f"blind packet drift: {destination_packet}"
            )
    else:
        shutil.copy2(source_packet, destination_packet)
    return value


def _existing_host_passes(
    root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["packet_id"]), str(row["pass_type"])): row
        for row in _read_jsonl(root / "raw_judgments.jsonl")
    }


def _scrub_completed_blind_evidence(
    workspace: Path,
    batch_dir: Path,
) -> None:
    expected_parent = (workspace / "batches").resolve()
    resolved_batch = batch_dir.resolve()
    if resolved_batch.parent != expected_parent:
        raise ValueError("refusing to scrub outside blind batch workspace")
    for name in ("assets", "screenshot_only", "source_conditioned"):
        path = resolved_batch / name
        if path.is_dir():
            shutil.rmtree(path)
    result_dir = resolved_batch / "results"
    if result_dir.is_dir():
        for template in result_dir.glob("*_template.jsonl"):
            template.unlink()


def _prepare_blind_rows(
    root: Path,
    workspace: Path,
    selected: list[dict[str, Any]],
    *,
    pass_type: str,
) -> dict[str, Any]:
    if pass_type not in {"screenshot_only", "source_conditioned"}:
        raise ValueError(f"unsupported pass_type: {pass_type}")
    host_passes = _existing_host_passes(root)
    if pass_type == "source_conditioned":
        missing = [
            str(row["packet_id"])
            for row in selected
            if (str(row["packet_id"]), "screenshot_only")
            not in host_passes
        ]
        if missing:
            raise ValueError(
                "source packets remain sealed until every screenshot pass "
                f"is host-recorded; missing={len(missing)}"
            )
    batch_name = _batch_name(selected)
    batch_dir = workspace / "batches" / batch_name
    packet_dir = batch_dir / pass_type
    asset_dir = batch_dir / "assets"
    result_dir = batch_dir / "results"
    packet_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    rubric_dir = batch_dir / "rubric"
    rubric_dir.mkdir(exist_ok=True)
    rubric_source = (
        root / "frozen_judge_protocol.json"
        if (root / "frozen_judge_protocol.json").exists()
        else root / "judge_protocol.json"
    )
    shutil.copy2(
        root / "judge_instructions.md",
        rubric_dir / "judge_instructions.md",
    )
    task_instructions = (
        workspace
        / "rubric"
        / "genui_single_codex_judge_task_v2.md"
    )
    shutil.copy2(
        task_instructions,
        rubric_dir / task_instructions.name,
    )
    shutil.copy2(rubric_source, rubric_dir / rubric_source.name)
    packet_rows: list[dict[str, Any]] = []
    for scheduled in selected:
        packet_id = str(scheduled["packet_id"])
        source_packet = (
            root / "packets" / pass_type / f"{packet_id}.json"
        )
        destination_packet = packet_dir / source_packet.name
        packet_value = _copy_packet_assets(
            source_packet,
            destination_packet,
            asset_dir,
        )
        packet_rows.append(
            {
                "schedule_position": int(
                    scheduled["schedule_position"]
                ),
                "packet_id": packet_id,
                "packet_path": str(destination_packet.absolute()),
                "overview_asset": str(
                    (
                        destination_packet.parent
                        / str(packet_value["overview_asset"])
                    ).absolute()
                ),
            }
        )
    result_path = result_dir / f"{pass_type}.jsonl"
    ready_path = result_dir / f"{pass_type}_READY.json"
    template_path = result_dir / f"{pass_type}_template.jsonl"
    template_rows = [
        {
            "packet_id": row["packet_id"],
            "pass_type": pass_type,
            "protocol_fingerprint": protocol_fingerprint(),
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
        if _read_jsonl(template_path) != template_rows:
            raise ValueError(f"blind result template drift: {template_path}")
    else:
        _write_jsonl(template_path, template_rows)
    manifest_path = batch_dir / f"{pass_type}_manifest.json"
    manifest = {
        "schema_version": "genui_single_codex_blind_batch.v2",
        "workspace_id": "b_" + protocol_fingerprint()[:20],
        "pass_type": pass_type,
        "start_position": int(selected[0]["schedule_position"]),
        "end_position_inclusive": int(
            selected[-1]["schedule_position"]
        ),
        "packet_count": len(packet_rows),
        "protocol_fingerprint": protocol_fingerprint(),
        "judge_instructions_path": str(
            (rubric_dir / "judge_instructions.md").absolute()
        ),
        "task_instructions_path": str(
            (rubric_dir / task_instructions.name).absolute()
        ),
        "rubric_path": str(
            (rubric_dir / rubric_source.name).absolute()
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
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"immutable blind batch drift: {manifest_path}")
    else:
        _write_json(manifest_path, manifest)
    return {
        "batch_manifest": str(manifest_path),
        "result_path": str(result_path),
        "packet_count": len(packet_rows),
        "pass_type": pass_type,
    }


def prepare_blind_batch(
    benchmark_dir: str | Path,
    *,
    start: int,
    count: int,
    pass_type: str,
) -> dict[str, Any]:
    """Release one ordered scheduled pass into a neutral batch capsule."""

    root = Path(benchmark_dir).resolve()
    verify_implementation_provenance(root)
    workspace, _ = _load_pointer(root)
    selected = _selected_schedule(root, start=start, count=count)
    if start >= 32 and not (root / "frozen_judge_protocol.json").exists():
        raise ValueError("freeze the pilot protocol before post-pilot work")
    return _prepare_blind_rows(
        root,
        workspace,
        selected,
        pass_type=pass_type,
    )


def _import_blind_rows(
    root: Path,
    workspace: Path,
    selected: list[dict[str, Any]],
    *,
    pass_type: str,
    expected_task_id: str | None = None,
) -> dict[str, Any]:
    batch_dir = workspace / "batches" / _batch_name(selected)
    result_path = batch_dir / "results" / f"{pass_type}.jsonl"
    receipt_path = (
        batch_dir / "results" / f"{pass_type}_host_import.json"
    )
    if not result_path.exists() and receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(result_path)
    expected_ids = [str(row["packet_id"]) for row in selected]
    actual_ids = [str(row.get("packet_id") or "") for row in rows]
    if len(rows) != len(expected_ids) or actual_ids != expected_ids:
        raise ValueError(
            "blind result rows must exactly match the scheduled packet order"
        )
    normalized = [validate_judgment_pass(row) for row in rows]
    if any(row["pass_type"] != pass_type for row in normalized):
        raise ValueError("blind result contains the wrong pass_type")
    if expected_task_id is not None and any(
        row["judge_task_id"] != expected_task_id for row in normalized
    ):
        raise ValueError("blind result judge_task_id mismatch")
    existing = _existing_host_passes(root)
    saved = 0
    skipped = 0
    for row in normalized:
        key = (row["packet_id"], row["pass_type"])
        prior = existing.get(key)
        if prior is not None:
            if _canonical_json(validate_judgment_pass(prior)) != _canonical_json(
                row
            ):
                raise ValueError(
                    f"host judgment differs for {row['packet_id']} "
                    f"{row['pass_type']}"
                )
            skipped += 1
            continue
        append_judgment_pass(root, row)
        existing[key] = row
        saved += 1
    archive_path = (
        root
        / "sealed"
        / "judge_block_results"
        / _batch_name(selected)
        / f"{pass_type}.jsonl"
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        if _hash_file(archive_path) != _hash_file(result_path):
            raise ValueError(
                f"immutable blind result archive drift: {archive_path}"
            )
    else:
        shutil.copy2(result_path, archive_path)
    import_receipt = {
        "schema_version": "genui_single_codex_blind_import.v2",
        "pass_type": pass_type,
        "start_position": int(selected[0]["schedule_position"]),
        "end_position_inclusive": int(
            selected[-1]["schedule_position"]
        ),
        "result_sha256": _hash_file(result_path),
        "result_archive_sealed": True,
        "saved_pass_count": saved,
        "already_imported_count": skipped,
        "judge_task_ids": sorted(
            {str(row["judge_task_id"]) for row in normalized}
        ),
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(receipt_path, import_receipt)
    result_path.unlink()
    ready_path = (
        batch_dir / "results" / f"{pass_type}_READY.json"
    )
    if ready_path.exists():
        ready_path.unlink()
    if pass_type == "source_conditioned":
        _scrub_completed_blind_evidence(
            workspace,
            batch_dir,
        )
    return import_receipt


def import_blind_batch(
    benchmark_dir: str | Path,
    *,
    start: int,
    count: int,
    pass_type: str,
    expected_task_id: str | None = None,
) -> dict[str, Any]:
    """Validate a neutral scheduled result and persist it on the host."""

    root = Path(benchmark_dir).resolve()
    verify_implementation_provenance(root)
    workspace, _ = _load_pointer(root)
    selected = _selected_schedule(root, start=start, count=count)
    return _import_blind_rows(
        root,
        workspace,
        selected,
        pass_type=pass_type,
        expected_task_id=expected_task_id,
    )


def _selected_adjudications(
    root: Path,
    *,
    start: int,
    count: int,
) -> list[dict[str, Any]]:
    if start < 0 or count <= 0 or count > 20:
        raise ValueError("batch start/count must select 1 to 20 packets")
    pending = _read_jsonl(root / "pending_adjudications.jsonl")
    if start >= len(pending):
        raise ValueError("adjudication batch start is outside pending rows")
    selected_pending = pending[start : start + count]
    if len(selected_pending) != count:
        raise ValueError("adjudication batch extends beyond pending rows")
    schedule_count = len(_read_jsonl(root / "judge_schedule.jsonl"))
    return [
        {
            "schedule_position": schedule_count + start + index,
            "packet_id": str(row["packet_id"]),
        }
        for index, row in enumerate(selected_pending)
    ]


def prepare_blind_adjudication_batch(
    benchmark_dir: str | Path,
    *,
    start: int,
    count: int,
    pass_type: str,
) -> dict[str, Any]:
    """Release opaque third-pass packets without repeat metadata."""

    root = Path(benchmark_dir).resolve()
    verify_implementation_provenance(root)
    workspace, _ = _load_pointer(root)
    selected = _selected_adjudications(root, start=start, count=count)
    return _prepare_blind_rows(
        root,
        workspace,
        selected,
        pass_type=pass_type,
    )


def import_blind_adjudication_batch(
    benchmark_dir: str | Path,
    *,
    start: int,
    count: int,
    pass_type: str,
    expected_task_id: str | None = None,
) -> dict[str, Any]:
    root = Path(benchmark_dir).resolve()
    verify_implementation_provenance(root)
    workspace, _ = _load_pointer(root)
    selected = _selected_adjudications(root, start=start, count=count)
    return _import_blind_rows(
        root,
        workspace,
        selected,
        pass_type=pass_type,
        expected_task_id=expected_task_id,
    )


def blind_workspace_status(
    workspace_dir: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_dir).absolute()
    _assert_neutral_workspace_path(workspace)
    manifest = json.loads(
        (workspace / "workspace_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    pass_counts = {"screenshot_only": 0, "source_conditioned": 0}
    completed_packets: set[str] = set()
    partial_packets: set[str] = set()
    for batch_dir in sorted((workspace / "batches").glob("batch_*")):
        by_pass: dict[str, set[str]] = {}
        for pass_type in pass_counts:
            receipt_path = (
                batch_dir
                / "results"
                / f"{pass_type}_host_import.json"
            )
            manifest_path = batch_dir / f"{pass_type}_manifest.json"
            if receipt_path.exists() and manifest_path.exists():
                manifest_row = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                ids = {
                    str(row["packet_id"])
                    for row in manifest_row.get("packets", [])
                }
            else:
                path = batch_dir / "results" / f"{pass_type}.jsonl"
                rows = _read_jsonl(path)
                ids = {str(row.get("packet_id") or "") for row in rows}
            by_pass[pass_type] = ids
            pass_counts[pass_type] += len(ids)
        completed_packets.update(
            by_pass["screenshot_only"] & by_pass["source_conditioned"]
        )
        partial_packets.update(
            by_pass["screenshot_only"] ^ by_pass["source_conditioned"]
        )
    return {
        "workspace_id": manifest["workspace_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "screenshot_pass_count": pass_counts["screenshot_only"],
        "source_pass_count": pass_counts["source_conditioned"],
        "completed_packet_count": len(completed_packets),
        "partial_packet_count": len(partial_packets),
    }


__all__ = [
    "POINTER_NAME",
    "WORKSPACE_SCHEMA_VERSION",
    "blind_workspace_status",
    "import_blind_adjudication_batch",
    "import_blind_batch",
    "initialize_blind_workspace",
    "prepare_blind_adjudication_batch",
    "prepare_blind_batch",
    "sync_frozen_protocol_to_blind_workspace",
]
