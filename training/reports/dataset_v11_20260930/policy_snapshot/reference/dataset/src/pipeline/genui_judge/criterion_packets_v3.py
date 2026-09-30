"""Packet construction for the GenUI Anchored Criterion Judge v3."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from .criterion_protocol_v3 import (
    JUDGE_PACKET_SCHEMA_VERSION,
    JUDGE_PROTOCOL_VERSION,
    protocol_fingerprint,
    rubric_for_pass,
)


_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "schema"
    / "genui_anchored_criterion_packet_v3.schema.json"
)
_PACKET_VALIDATOR = Draft202012Validator(
    json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _hash_file(destination) != _hash_file(source):
            raise ValueError(f"asset collision: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _opaque_packet_id(ui_id: str, occurrence: int) -> str:
    payload = (
        f"{protocol_fingerprint()}|{ui_id}|{occurrence}|criterion-repeat"
    )
    return "c_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def validate_criterion_packet(packet: Mapping[str, Any]) -> None:
    errors = sorted(
        _PACKET_VALIDATOR.iter_errors(packet),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path)
        raise ValueError(
            f"criterion packet schema error at {location or '<root>'}: {first.message}"
        )


def _capture_rows_by_ui(bundle_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows = _read_jsonl(bundle_root / "capture" / "native_capture_manifest_96.jsonl")
    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ui_id = str(row.get("ui_id") or "")
        if not ui_id:
            raise ValueError("capture manifest row lacks ui_id")
        output.setdefault(ui_id, []).append(row)
    return output


def _screenshot_evidence(
    bundle_root: Path,
    ui_id: str,
    captures: Mapping[str, list[dict[str, Any]]],
    *,
    asset_dir: Path,
) -> tuple[list[dict[str, Any]], bool]:
    rows = captures.get(ui_id, [])
    candidate_failure = any(
        row.get("failure_class") == "candidate"
        and bool(row.get("required", True))
        for row in rows
    )
    visible_rows = [
        row
        for row in rows
        if bool(row.get("image_captured", row.get("ok")))
    ]
    if not visible_rows and not candidate_failure:
        raise ValueError(f"no screenshot evidence for {ui_id}")
    profile_order = {"compact": 0, "medium_700dp": 1, "expanded_900dp": 2}
    capture_order = {"initial_viewport": 0, "full_height": 1}
    result: list[dict[str, Any]] = []
    for row in sorted(
        visible_rows,
        key=lambda item: (
            0 if item.get("state_id") == "initial" else 1,
            profile_order.get(str(item.get("viewport_profile") or ""), 99),
            str(item.get("state_id") or ""),
            capture_order.get(str(item.get("capture_kind") or ""), 99),
        ),
    ):
        relative = Path(str(row.get("local_path") or ""))
        path = relative if relative.is_absolute() else bundle_root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        expected_hash = str(row.get("screenshot_sha256") or "")
        actual_hash = _hash_file(path)
        if expected_hash and expected_hash != actual_hash:
            raise ValueError(f"screenshot hash mismatch: {relative}")
        suffix = path.suffix.lower() or ".png"
        asset_name = f"{actual_hash}{suffix}"
        destination = asset_dir / asset_name
        _link_or_copy(path, destination)
        result.append(
            {
                "asset": f"../assets/{asset_name}",
                "sha256": actual_hash,
                "viewport_profile": row.get("viewport_profile"),
                "viewport_width_dp": row.get("viewport_width_dp"),
                "viewport_height_dp": row.get("viewport_height_dp"),
                "capture_kind": row.get("capture_kind"),
                "state_id": row.get("state_id"),
                "state_description": row.get("state_description"),
            }
        )
    return result, candidate_failure


def _fresh_repeat_schedule(
    ui_ids: list[str],
    *,
    seed: int,
    task_block_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if task_block_size <= 0:
        raise ValueError("task_block_size must be positive")
    first = list(ui_ids)
    second = list(ui_ids)
    random.Random(seed).shuffle(first)
    random.Random(seed ^ 0x5A17C9).shuffle(second)
    # All second occurrences are at least len(ui_ids) positions after the
    # first half.  They therefore cannot share a fresh task block with their
    # paired first occurrence for the intended 96-pair reliability run.
    ordered = [(ui_id, 0) for ui_id in first] + [
        (ui_id, 1) for ui_id in second
    ]
    public: list[dict[str, Any]] = []
    sealed: list[dict[str, Any]] = []
    first_packet_by_ui: dict[str, str] = {}
    first_position_by_ui: dict[str, int] = {}
    for position, (ui_id, occurrence) in enumerate(ordered):
        packet_id = _opaque_packet_id(ui_id, occurrence)
        if occurrence == 0:
            first_packet_by_ui[ui_id] = packet_id
            first_position_by_ui[ui_id] = position
        public.append(
            {
                "schedule_position": position,
                "packet_id": packet_id,
                "task_block_index": position // task_block_size,
            }
        )
        sealed.append(
            {
                "packet_id": packet_id,
                "ui_id": ui_id,
                "occurrence": occurrence,
                "repeat_of_packet_id": (
                    first_packet_by_ui.get(ui_id) if occurrence == 1 else None
                ),
                "schedule_position": position,
                "task_block_index": position // task_block_size,
                "pair_spacing": (
                    position - first_position_by_ui[ui_id]
                    if occurrence == 1
                    else None
                ),
            }
        )
    for row in sealed:
        if row["occurrence"] == 1:
            original = next(
                item
                for item in sealed
                if item["packet_id"] == row["repeat_of_packet_id"]
            )
            if original["task_block_index"] == row["task_block_index"]:
                raise AssertionError("repeat pair shares a fresh task block")
    return public, sealed


def build_criterion_review_packets(
    review_bundle_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20260801,
    task_block_size: int = 16,
) -> dict[str, Any]:
    """Build a fresh blinded 192-occurrence reliability workspace from 96 UIs.

    The output contains packet JSON and a public schedule.  Repeat identities
    remain only in ``sealed/criterion_v3_pair_identity_map.jsonl``.
    Screenshot assets are deduplicated by SHA-256 and hard-linked or copied
    into the workspace so the judging packet set is self-contained.
    """

    bundle_root = Path(review_bundle_dir).resolve()
    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    records = {
        str(row["ui_id"]): row
        for row in _read_jsonl(bundle_root / "samples" / "selected_genui_96.jsonl")
    }
    contracts = {
        str(row["ui_id"]): row
        for row in _read_jsonl(bundle_root / "samples" / "expected_contracts_96.jsonl")
    }
    ui_ids = sorted(records)
    if len(ui_ids) != 96 or set(ui_ids) != set(contracts):
        raise ValueError("review bundle must contain exactly 96 aligned samples/contracts")
    public_schedule, sealed_schedule = _fresh_repeat_schedule(
        ui_ids,
        seed=seed,
        task_block_size=task_block_size,
    )
    captures = _capture_rows_by_ui(bundle_root)
    asset_dir = output_root / "packets" / "assets"
    asset_dir.mkdir(parents=True)
    screenshot_by_ui: dict[str, tuple[list[dict[str, Any]], bool]] = {}
    for ui_id in ui_ids:
        screenshot_by_ui[ui_id] = _screenshot_evidence(
            bundle_root, ui_id, captures, asset_dir=asset_dir
        )
    screen_dir = output_root / "packets" / "screenshot_only"
    source_dir = output_root / "packets" / "source_conditioned"
    screen_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    sealed_by_packet = {row["packet_id"]: row for row in sealed_schedule}
    packet_manifest: list[dict[str, Any]] = []
    for scheduled in public_schedule:
        packet_id = str(scheduled["packet_id"])
        ui_id = str(sealed_by_packet[packet_id]["ui_id"])
        screenshots, candidate_failure = screenshot_by_ui[ui_id]
        common = {
            "schema_version": JUDGE_PACKET_SCHEMA_VERSION,
            "protocol_version": JUDGE_PROTOCOL_VERSION,
            "packet_id": packet_id,
            "protocol_fingerprint": protocol_fingerprint(),
            "screenshots": screenshots,
            "candidate_render_failure": candidate_failure,
        }
        screenshot_packet = {
            **common,
            "pass_type": "screenshot_only",
            "rubric": rubric_for_pass("screenshot_only"),
        }
        contract_row = contracts[ui_id]
        source_packet = {
            **common,
            "pass_type": "source_conditioned",
            "rubric": rubric_for_pass("source_conditioned"),
            "source_text": str(records[ui_id].get("response_text") or ""),
            "expected_ui_contract": contract_row["expected_ui_contract"],
            "expected_ui_contract_hash": str(
                contract_row.get("expected_ui_contract_hash") or ""
            ),
        }
        validate_criterion_packet(screenshot_packet)
        validate_criterion_packet(source_packet)
        _atomic_write_json(screen_dir / f"{packet_id}.json", screenshot_packet)
        _atomic_write_json(source_dir / f"{packet_id}.json", source_packet)
        packet_manifest.append(
            {
                **scheduled,
                "screenshot_only_packet": (
                    Path("packets") / "screenshot_only" / f"{packet_id}.json"
                ).as_posix(),
                "source_conditioned_packet": (
                    Path("packets") / "source_conditioned" / f"{packet_id}.json"
                ).as_posix(),
                "screenshot_count": len(screenshots),
                "candidate_render_failure": candidate_failure,
            }
        )
    _atomic_write_jsonl(output_root / "criterion_v3_schedule.jsonl", public_schedule)
    _atomic_write_jsonl(
        output_root / "sealed" / "criterion_v3_pair_identity_map.jsonl",
        sealed_schedule,
    )
    _atomic_write_jsonl(
        output_root / "packets" / "criterion_v3_packet_manifest.jsonl",
        packet_manifest,
    )
    _atomic_write_json(
        output_root / "criterion_v3_protocol.json",
        {
            "schema_version": "genui_anchored_criterion_protocol_manifest.v3",
            "protocol_version": JUDGE_PROTOCOL_VERSION,
            "protocol_fingerprint": protocol_fingerprint(),
            "sample_count": 96,
            "scheduled_occurrence_count": 192,
            "task_block_size": task_block_size,
            "seed": seed,
            "source_bundle_label": bundle_root.name,
            "source_bundle_manifest_sha256": (
                _hash_file(bundle_root / "REVIEW_BUNDLE_MANIFEST.json")
                if (bundle_root / "REVIEW_BUNDLE_MANIFEST.json").exists()
                else None
            ),
            "note": (
                "Repeat identity is sealed. Actual fresh LLM judgments are not "
                "produced by packet construction."
            ),
        },
    )
    asset_paths = list(asset_dir.iterdir())
    return {
        "sample_count": 96,
        "scheduled_occurrence_count": 192,
        "packet_count": len(packet_manifest),
        "unique_asset_count": len(asset_paths),
        "asset_bytes": sum(path.stat().st_size for path in asset_paths),
        "protocol_fingerprint": protocol_fingerprint(),
        "output_dir": str(output_root),
    }


__all__ = [
    "build_criterion_review_packets",
    "validate_criterion_packet",
]
