from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ir_training.common.config import repo_root


TENSORBOARD_ROOT_ENV = "A2UI_TENSORBOARD_ROOT"


def resolve_tensorboard_root(
    configured: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    """Resolve the shared TensorBoard root used by training and final evals.

    ``A2UI_TENSORBOARD_ROOT`` takes precedence so an MLP job can mount and use
    ``/tensorboard`` without rewriting a checked-in config. Relative configured
    paths are anchored at the repository root, not at ``training/``.
    """

    value = os.environ.get(TENSORBOARD_ROOT_ENV) or configured or "tensorboard"
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    anchor = Path(repository_root).resolve() if repository_root else repo_root()
    return (anchor / path).resolve()


def resolve_tensorboard_run_dir(
    configured_root: str | Path | None,
    *,
    run_id: str,
    repository_root: str | Path | None = None,
) -> Path:
    return resolve_tensorboard_root(
        configured_root, repository_root=repository_root
    ) / _safe_component(run_id, fallback="run")


def log_evaluation_result(
    tensorboard_root: str | Path,
    *,
    run_id: str,
    evaluation_name: str,
    metrics: Mapping[str, Any],
    step: int = 0,
    artifacts: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    source_aggregate_path: str | Path | None = None,
    writer_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Persist one comparable evaluation result to TensorBoard and JSON.

    Scalar tags use ``evaluation/<evaluation_name>/<metric>``. The complete
    metrics, metadata, and artifact identities are retained in a deterministic
    JSON sidecar under the same TensorBoard run directory.
    """

    resolved_root = resolve_tensorboard_root(tensorboard_root)
    safe_run_id = _safe_component(run_id, fallback="run")
    safe_evaluation = _safe_component(evaluation_name, fallback="evaluation")
    resolved_step = int(step)
    if resolved_step < 0:
        raise ValueError("TensorBoard evaluation step must be non-negative.")
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping.")

    log_dir = resolved_root / safe_run_id
    records_dir = log_dir / "evaluation_records" / safe_evaluation
    records_dir.mkdir(parents=True, exist_ok=True)
    artifact_records = _artifact_records(artifacts or {})
    source_record = (
        artifact_identity(Path(source_aggregate_path))
        if source_aggregate_path is not None
        else None
    )
    scalar_metrics = flatten_scalar_metrics(metrics)
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "evaluation_name": str(evaluation_name),
        "step": resolved_step,
        "tensorboard_root": str(resolved_root),
        "tensorboard_log_dir": str(log_dir),
        "scalar_tags": {
            f"evaluation/{safe_evaluation}/{key}": value
            for key, value in scalar_metrics.items()
        },
        "metrics": _json_safe(metrics),
        "metadata": _json_safe(metadata or {}),
        "artifacts": artifact_records,
        "source_aggregate": source_record,
    }
    record_path = records_dir / f"step_{resolved_step:09d}.json"
    _atomic_write_json(record_path, record)

    factory = writer_factory or _summary_writer_factory()
    writer = factory(log_dir=str(log_dir))
    try:
        for tag, value in record["scalar_tags"].items():
            writer.add_scalar(tag, value, resolved_step)
        writer.add_text(
            f"evaluation/{safe_evaluation}/record",
            "```json\n" + json.dumps(record, indent=2, ensure_ascii=False) + "\n```",
            resolved_step,
        )
        writer.flush()
    finally:
        writer.close()
    record["record_path"] = str(record_path)
    return record


def flatten_scalar_metrics(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten finite numeric leaves for stable TensorBoard scalar tags."""

    flattened: dict[str, float] = {}
    for raw_key, value in values.items():
        key = _safe_tag_component(raw_key)
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_scalar_metrics(value, prefix=full_key))
            continue
        if isinstance(value, bool):
            flattened[full_key] = 1.0 if value else 0.0
            continue
        if isinstance(value, (int, float)):
            parsed = float(value)
            if math.isfinite(parsed):
                flattened[full_key] = parsed
    return flattened


def _summary_writer_factory() -> Callable[..., Any]:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError(
            "TensorBoard logging requires torch and tensorboard from "
            "training/requirements-training.txt."
        ) from exc
    return SummaryWriter


def _artifact_records(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for raw_name, value in artifacts.items():
        name = str(raw_name)
        if isinstance(value, (str, Path)):
            records[name] = artifact_identity(Path(value))
        else:
            records[name] = _json_safe(value)
    return records


def artifact_identity(path: str | Path) -> dict[str, Any]:
    """Return a deterministic identity for one file or directory artifact."""

    path = Path(path)
    resolved = path.expanduser().resolve()
    record: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
        "kind": "missing",
    }
    if resolved.is_file():
        record.update(
            {
                "kind": "file",
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
        )
    elif resolved.is_dir():
        files = sorted(
            candidate
            for candidate in resolved.rglob("*")
            if candidate.is_file() and not candidate.is_symlink()
        )
        entries = [
            {
                "path": candidate.relative_to(resolved).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
            for candidate in files
        ]
        manifest_bytes = json.dumps(
            entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        record.update(
            {
                "kind": "directory",
                "file_count": len(entries),
                "size_bytes": sum(int(item["size_bytes"]) for item in entries),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "hash_semantics": "sha256_of_sorted_path_size_sha256_manifest",
                "files": entries,
            }
        )
    return record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_component(value: Any, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    return normalized or fallback


def _safe_tag_component(value: Any) -> str:
    return _safe_component(value, fallback="metric")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
