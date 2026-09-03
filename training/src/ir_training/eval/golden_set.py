from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.common.jsonl import read_jsonl


def load_fixed_golden_rows(
    split_path: str | Path,
    *,
    max_rows: int | None = None,
    required_rows: int | None = None,
    require_exact_rows: bool = True,
    require_unique_rows: bool = True,
) -> list[dict[str, Any]]:
    """Load a held-out set and enforce its row-count and identity contract."""

    resolved = Path(split_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Golden evaluation split is missing: {resolved}")
    rows = list(read_jsonl(resolved))
    if required_rows is not None:
        required = int(required_rows)
        if required < 1:
            raise ValueError("required_rows must be positive when set.")
        observed = len(rows) if require_exact_rows else min(
            len(rows), int(max_rows) if max_rows is not None else len(rows)
        )
        invalid = observed != required if require_exact_rows else observed < required
        if invalid:
            comparator = "exactly" if require_exact_rows else "at least"
            raise ValueError(
                f"Golden evaluation requires {comparator} {required} rows; "
                f"observed {observed} in {resolved}."
            )
    if max_rows is not None:
        maximum = int(max_rows)
        if maximum < 1:
            raise ValueError("max_rows must be positive when set.")
        rows = rows[:maximum]
    if required_rows is not None and len(rows) < int(required_rows):
        raise ValueError(
            f"max_rows selected {len(rows)} rows but required_rows={required_rows}."
        )
    if require_unique_rows:
        identities = [_row_identity(row, index) for index, row in enumerate(rows)]
        if len(set(identities)) != len(identities):
            raise ValueError(f"Golden evaluation split has duplicate identities: {resolved}")
    return rows


def validate_prepared_golden_contract(
    split_path: str | Path,
    *,
    dataset_config_path: str | Path,
    source_genui_path: str | Path,
    source_responses_path: str | Path,
    expected_genui_sha256: str,
    expected_responses_sha256: str,
    expected_dataset_config_sha256: str | None = None,
    expected_split_sha256: str | None = None,
    required_rows: int,
) -> dict[str, Any]:
    """Cryptographically bind a prepared fixed set to its source and config.

    Row-count checks alone cannot distinguish a stale or substituted 32-row
    file. This gate requires the dataset config, both immutable source files,
    the preparation manifest, and the selected output JSONL to agree.
    """

    split = Path(split_path).expanduser().resolve()
    config_path = Path(dataset_config_path).expanduser().resolve()
    source_genui = Path(source_genui_path).expanduser().resolve()
    source_responses = Path(source_responses_path).expanduser().resolve()
    required = int(required_rows)
    if required < 1:
        raise ValueError("required_rows must be positive.")
    for label, value in (
        ("expected_genui_sha256", expected_genui_sha256),
        ("expected_responses_sha256", expected_responses_sha256),
    ):
        digest = str(value).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    expected_genui = str(expected_genui_sha256).strip().lower()
    expected_responses = str(expected_responses_sha256).strip().lower()
    expected_config = str(expected_dataset_config_sha256 or "").strip().lower()
    expected_split = str(expected_split_sha256 or "").strip().lower()
    for label, digest in (
        ("expected_dataset_config_sha256", expected_config),
        ("expected_split_sha256", expected_split),
    ):
        if digest and (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest.")

    if not config_path.is_file():
        raise FileNotFoundError(f"Golden dataset config is missing: {config_path}")
    config = load_yaml(config_path)
    run = config.get("run") if isinstance(config.get("run"), dict) else {}
    filters = (
        config.get("filters") if isinstance(config.get("filters"), dict) else {}
    )
    split_config = (
        config.get("split") if isinstance(config.get("split"), dict) else {}
    )
    configured_output = resolve_path(run.get("output_dir", ""), training_root())
    if split != (configured_output / split.name).resolve():
        raise ValueError(
            "Prepared Golden split is outside the dataset config output: "
            f"{split} != {(configured_output / split.name).resolve()}"
        )
    configured_source_dir = resolve_path(
        run.get("source_run_dir", ""), training_root()
    )
    if source_genui != (configured_source_dir / "genui.jsonl").resolve():
        raise ValueError("Golden genui.jsonl path differs from dataset config.")
    if source_responses != (configured_source_dir / "responses.jsonl").resolve():
        raise ValueError("Golden responses.jsonl path differs from dataset config.")
    if str(run.get("source_genui_sha256") or "").strip().lower() != expected_genui:
        raise ValueError("Golden dataset config genui.jsonl SHA-256 pin differs.")
    if (
        str(run.get("source_responses_sha256") or "").strip().lower()
        != expected_responses
    ):
        raise ValueError("Golden dataset config responses.jsonl SHA-256 pin differs.")
    if not bool(
        int(filters.get("required_accepted_rows", 0)) == required
        and filters.get("require_exact_accepted_rows") is True
        and filters.get("require_unique_source_ids") is True
        and float(split_config.get("train", -1)) == 1.0
        and float(split_config.get("val", -1)) == 0.0
        and float(split_config.get("test", -1)) == 0.0
    ):
        raise ValueError("Golden dataset config is not an exact evaluation-only split.")

    for label, path, expected in (
        ("genui.jsonl", source_genui, expected_genui),
        ("responses.jsonl", source_responses, expected_responses),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Golden source {label} is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Golden source {label} SHA-256 mismatch: expected {expected}, "
                f"observed {actual}."
            )

    rows = load_fixed_golden_rows(
        split,
        max_rows=required,
        required_rows=required,
        require_exact_rows=True,
        require_unique_rows=True,
    )
    manifest_path = split.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared Golden manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Prepared Golden manifest must be a JSON object.")
    config_sha256 = _sha256_file(config_path)
    split_sha256 = _sha256_file(split)
    if expected_config and config_sha256 != expected_config:
        raise ValueError(
            "Golden dataset config differs from the independently pinned digest."
        )
    if expected_split and split_sha256 != expected_split:
        raise ValueError(
            "Prepared Golden split differs from the independently pinned digest."
        )
    if str(manifest.get("config_sha256") or "").lower() != config_sha256:
        raise ValueError("Prepared Golden manifest dataset-config hash mismatch.")
    if _manifest_hash_values(manifest, "source_genui_sha256s") != {expected_genui}:
        raise ValueError("Prepared Golden manifest genui.jsonl source hash mismatch.")
    if _manifest_hash_values(manifest, "source_responses_sha256s") != {
        expected_responses
    }:
        raise ValueError("Prepared Golden manifest responses.jsonl source hash mismatch.")
    outputs = (
        manifest.get("output_sha256s")
        if isinstance(manifest.get("output_sha256s"), dict)
        else {}
    )
    if str(outputs.get(split.name) or "").lower() != split_sha256:
        raise ValueError("Prepared Golden split hash is absent or stale in manifest.")
    counts = (
        manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    )
    if not bool(
        counts.get("all") == required
        and counts.get("accepted") == required
        and counts.get("train") == required
        and counts.get("val") == 0
        and counts.get("test") == 0
        and counts.get("rejected") == 0
        and manifest.get("source_group_count") == required
    ):
        raise ValueError("Prepared Golden manifest count contract mismatch.")
    return {
        "rows": len(rows),
        "dataset_config": str(config_path),
        "dataset_config_sha256": config_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "split": str(split),
        "split_sha256": split_sha256,
        "source_genui": str(source_genui),
        "source_genui_sha256": expected_genui,
        "source_responses": str(source_responses),
        "source_responses_sha256": expected_responses,
    }


def _row_identity(row: dict[str, Any], index: int) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for value in (
        row.get("source_id"),
        row.get("response_id"),
        row.get("id"),
        metadata.get("query_id"),
    ):
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"row-index:{index}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hash_values(manifest: dict[str, Any], key: str) -> set[str]:
    value = manifest.get(key)
    if not isinstance(value, dict):
        return set()
    return {str(item).strip().lower() for item in value.values() if str(item).strip()}
