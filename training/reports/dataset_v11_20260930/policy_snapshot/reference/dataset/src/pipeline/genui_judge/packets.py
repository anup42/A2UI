"""Materialize blinded two-pass packets from native capture evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw, ImageOps

from .protocol import (
    PASS_DIMENSIONS,
    RUBRIC_DEFINITIONS,
    SCALE_ANCHORS,
    protocol_fingerprint,
)


PACKET_SCHEMA_VERSION = "genui_single_codex_blinded_packet.v2"
_PACKET_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "schema"
    / "genui_codex_packet_v2.schema.json"
)
_PACKET_VALIDATOR = Draft202012Validator(
    json.loads(_PACKET_SCHEMA_PATH.read_text(encoding="utf-8"))
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
    with path.open("r", encoding="utf-8") as handle:
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


def _validate_packet(value: Mapping[str, Any]) -> None:
    errors = sorted(
        _PACKET_VALIDATOR.iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path)
        raise ValueError(
            "invalid blinded packet"
            + (f" at {location}" if location else "")
            + f": {error.message}"
        )


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _hash_file(destination) != _hash_file(source):
            raise FileExistsError(
                f"blinded asset already exists with different bytes: {destination}"
            )
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _build_overview(
    captures: list[dict[str, Any]],
    *,
    asset_dir: Path,
    destination: Path,
) -> None:
    tiles: list[Image.Image] = []
    tile_width = 420
    tile_height = 858
    label_height = 60
    for capture in captures:
        source = asset_dir / Path(str(capture["asset"])).name
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            fitted = ImageOps.contain(
                rgb,
                (tile_width - 24, tile_height - label_height - 24),
                method=Image.Resampling.LANCZOS,
            )
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        x = (tile_width - fitted.width) // 2
        y = label_height + (tile_height - label_height - fitted.height) // 2
        tile.paste(fitted, (x, y))
        label = (
            f"{capture.get('viewport_profile')} | "
            f"{capture.get('capture_kind')} | "
            f"{capture.get('state_id')}"
        )
        draw = ImageDraw.Draw(tile)
        draw.text((14, 12), label[:70], fill="black")
        draw.rectangle(
            (0, 0, tile_width - 1, tile_height - 1),
            outline="#b0b0b0",
            width=2,
        )
        tiles.append(tile)
    if not tiles:
        tiles.append(Image.new("RGB", (tile_width, tile_height), "white"))
    columns = min(3, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    overview = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        "#e8e8e8",
    )
    for index, tile in enumerate(tiles):
        overview.paste(
            tile,
            ((index % columns) * tile_width, (index // columns) * tile_height),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    overview.save(destination, format="JPEG", quality=85, optimize=True)


def _rubric_for(pass_type: str) -> dict[str, Any]:
    dimensions = PASS_DIMENSIONS[pass_type]
    return {
        "pass_type": pass_type,
        "dimensions": [
            {
                "name": name,
                "definition": RUBRIC_DEFINITIONS[name],
            }
            for name in dimensions
        ],
        "anchors": {
            str(score): description
            for score, description in SCALE_ANCHORS.items()
        },
        "score_increment": 5,
        "confidence_policy": "Report confidence; never modify scores with it.",
        "reference_truth_policy": (
            "Treat the supplied source as reference truth. Do not judge its "
            "factual accuracy or writing quality."
        ),
    }


def _capture_index(
    benchmark_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    manifest_path = benchmark_dir / "native_capture_manifest.jsonl"
    rows = _read_jsonl(manifest_path)
    by_ui: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ui_id = str(row.get("ui_id") or "")
        if not ui_id:
            raise ValueError("capture manifest row lacks ui_id")
        by_ui.setdefault(ui_id, []).append(row)
    return by_ui


def build_blinded_packets(
    benchmark_dir: str | Path,
    *,
    allow_incomplete_capture: bool = False,
) -> dict[str, Any]:
    """Create packet JSON and opaque screenshot names without metric leakage."""

    root = Path(benchmark_dir).resolve()
    final_packet_root = root / "packets"
    if final_packet_root.exists():
        raise FileExistsError(
            "immutable packet directory already exists: "
            f"{final_packet_root}"
        )
    packet_root = root / ".packets.building"
    marker_path = packet_root / "_BUILDING.json"
    if packet_root.exists():
        marker = (
            json.loads(marker_path.read_text(encoding="utf-8"))
            if marker_path.exists()
            else {}
        )
        if marker.get("protocol_fingerprint") != protocol_fingerprint():
            raise ValueError(
                f"unrecognized packet staging directory: {packet_root}"
            )
        resolved = packet_root.resolve()
        if resolved.parent != root:
            raise ValueError("packet staging path escaped benchmark root")
        shutil.rmtree(packet_root)
    packet_root.mkdir()
    _atomic_write_json(
        marker_path,
        {
            "schema_version": PACKET_SCHEMA_VERSION,
            "protocol_fingerprint": protocol_fingerprint(),
        },
    )
    schedule = _read_jsonl(root / "judge_schedule.jsonl")
    identities = {
        row["packet_id"]: row
        for row in _read_jsonl(
            root / "sealed" / "packet_identity_map.jsonl"
        )
    }
    records = {
        row["ui_id"]: row
        for row in _read_jsonl(root / "selected_genui.jsonl")
    }
    contracts = {
        row["ui_id"]: row
        for row in _read_jsonl(root / "expected_contracts.jsonl")
    }
    captures = _capture_index(root)
    screenshot_dir = packet_root / "assets"
    screenshot_dir.mkdir(parents=True)
    screen_packet_dir = packet_root / "screenshot_only"
    source_packet_dir = packet_root / "source_conditioned"
    screen_packet_dir.mkdir()
    source_packet_dir.mkdir()
    packet_manifest: list[dict[str, Any]] = []

    for scheduled in sorted(
        schedule, key=lambda row: int(row["schedule_position"])
    ):
        packet_id = str(scheduled["packet_id"])
        identity = identities[packet_id]
        ui_id = str(identity["ui_id"])
        record = records[ui_id]
        contract = contracts[ui_id]
        capture_rows = captures.get(ui_id, [])
        evidence_rows = [
            row
            for row in capture_rows
            if bool(row.get("image_captured", row.get("ok")))
        ]
        candidate_render_failure = any(
            row.get("failure_class") == "candidate"
            and bool(row.get("required", True))
            for row in capture_rows
        )
        if not evidence_rows and not allow_incomplete_capture:
            raise ValueError(f"no native screenshot evidence for {ui_id}")
        blinded_captures: list[dict[str, Any]] = []
        profile_order = {
            "compact": 0,
            "medium_700dp": 1,
            "expanded_900dp": 2,
        }
        capture_order = {
            "initial_viewport": 0,
            "full_height": 1,
        }
        for index, capture in enumerate(
            sorted(
                evidence_rows,
                key=lambda row: (
                    0 if row.get("state_id") == "initial" else 1,
                    profile_order.get(
                        str(row.get("viewport_profile") or ""),
                        99,
                    ),
                    str(row.get("state_id") or ""),
                    capture_order.get(
                        str(row.get("capture_kind") or ""),
                        99,
                    ),
                ),
            )
        ):
            source_path = Path(str(capture["local_path"])).resolve()
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            suffix = source_path.suffix.lower() or ".png"
            asset_name = f"{packet_id}_{index:02d}{suffix}"
            destination = screenshot_dir / asset_name
            _link_or_copy(source_path, destination)
            blinded_captures.append(
                {
                    "asset": f"../assets/{asset_name}",
                    "sha256": _hash_file(destination),
                    "viewport_profile": capture.get("viewport_profile"),
                    "viewport_width_dp": capture.get("viewport_width_dp"),
                    "viewport_height_dp": capture.get("viewport_height_dp"),
                    "capture_kind": capture.get("capture_kind"),
                    "state_id": capture.get("state_id"),
                    "state_description": (
                        "Renderer default initial state"
                        if capture.get("state_id") == "initial"
                        else "Alternate internal renderer state"
                    ),
                }
            )
        overview_name = f"{packet_id}_overview.jpg"
        overview_path = screenshot_dir / overview_name
        _build_overview(
            blinded_captures,
            asset_dir=screenshot_dir,
            destination=overview_path,
        )
        common = {
            "schema_version": PACKET_SCHEMA_VERSION,
            "packet_id": packet_id,
            "protocol_fingerprint": protocol_fingerprint(),
            "screenshots": blinded_captures,
            "overview_asset": f"../assets/{overview_name}",
            "overview_sha256": _hash_file(overview_path),
            "candidate_render_failure": candidate_render_failure,
        }
        screenshot_packet = {
            **common,
            "pass_type": "screenshot_only",
            "rubric": _rubric_for("screenshot_only"),
        }
        source_packet = {
            **common,
            "pass_type": "source_conditioned",
            "rubric": _rubric_for("source_conditioned"),
            "source_text": str(record.get("response_text") or ""),
            "expected_ui_contract": contract["expected_ui_contract"],
            "expected_ui_contract_hash": contract[
                "expected_ui_contract_hash"
            ],
        }
        _validate_packet(screenshot_packet)
        _validate_packet(source_packet)
        _atomic_write_json(
            screen_packet_dir / f"{packet_id}.json",
            screenshot_packet,
        )
        _atomic_write_json(
            source_packet_dir / f"{packet_id}.json",
            source_packet,
        )
        packet_manifest.append(
            {
                "packet_id": packet_id,
                "schedule_position": scheduled["schedule_position"],
                "screenshot_only_path": str(
                    final_packet_root
                    / "screenshot_only"
                    / f"{packet_id}.json"
                ),
                "source_conditioned_path": str(
                    final_packet_root
                    / "source_conditioned"
                    / f"{packet_id}.json"
                ),
                "screenshot_count": len(blinded_captures),
                "candidate_render_failure": candidate_render_failure,
            }
        )

    manifest_path = packet_root / "packet_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in packet_manifest:
            handle.write(_canonical_json(row) + "\n")
    marker_path.unlink()
    os.replace(packet_root, final_packet_root)
    return {
        "packet_count": len(packet_manifest),
        "asset_count": len(
            list((final_packet_root / "assets").iterdir())
        ),
        "protocol_fingerprint": protocol_fingerprint(),
        "packet_manifest": str(
            final_packet_root / "packet_manifest.jsonl"
        ),
    }


def build_adjudication_packets(
    benchmark_dir: str | Path,
) -> dict[str, Any]:
    """Clone evidence under fresh opaque IDs for required third judgments."""

    root = Path(benchmark_dir).resolve()
    pending = _read_jsonl(root / "pending_adjudications.jsonl")
    created = 0
    for row in pending:
        packet_id = str(row["packet_id"])
        original_id = str(row["original_packet_id"])
        asset_dir = root / "packets" / "assets"
        cloned_assets: list[dict[str, Any]] | None = None
        cloned_overview: tuple[str, str] | None = None
        for pass_type in ("screenshot_only", "source_conditioned"):
            source = (
                root
                / "packets"
                / pass_type
                / f"{original_id}.json"
            )
            destination_dir = root / "packets" / pass_type
            destination = destination_dir / f"{packet_id}.json"
            if destination.exists():
                continue
            value = json.loads(source.read_text(encoding="utf-8"))
            value["packet_id"] = packet_id
            if cloned_assets is None:
                cloned_assets = []
                for index, capture in enumerate(value.get("screenshots", [])):
                    old = (
                        source.parent / str(capture["asset"])
                    ).resolve()
                    suffix = old.suffix.lower() or ".png"
                    name = f"{packet_id}_{index:02d}{suffix}"
                    new = asset_dir / name
                    _link_or_copy(old, new)
                    cloned_assets.append(
                        {
                            **capture,
                            "asset": f"../assets/{name}",
                            "sha256": _hash_file(new),
                        }
                    )
                old_overview = (
                    source.parent / str(value["overview_asset"])
                ).resolve()
                overview_name = f"{packet_id}_overview.jpg"
                new_overview = asset_dir / overview_name
                _link_or_copy(old_overview, new_overview)
                cloned_overview = (
                    f"../assets/{overview_name}",
                    _hash_file(new_overview),
                )
            value["screenshots"] = cloned_assets
            if cloned_overview is None:
                raise AssertionError("adjudication overview was not cloned")
            value["overview_asset"] = cloned_overview[0]
            value["overview_sha256"] = cloned_overview[1]
            _atomic_write_json(destination, value)
            created += 1
    return {
        "pending_adjudication_count": len(pending),
        "packet_files_created": created,
        "requires_fresh_task": bool(pending),
    }


__all__ = [
    "PACKET_SCHEMA_VERSION",
    "build_adjudication_packets",
    "build_blinded_packets",
]
