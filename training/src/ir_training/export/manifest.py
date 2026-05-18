from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    output_dir: str | Path,
    base_model: str,
    adapter_type: str,
    training_data_run: str | None,
    prompt_version: str | None,
    schema_version: str | None,
    runtime: str,
    min_app_version: str,
    max_input_tokens: int,
    max_output_tokens: int,
    files: list[str | Path] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    file_entries = []
    for item in files or []:
        path = Path(item)
        if not path.exists() or not path.is_file():
            continue
        file_entries.append(
            {
                "path": str(path.relative_to(out_dir) if path.is_relative_to(out_dir) else path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "manifest_version": 1,
        "base_model": base_model,
        "adapter_type": adapter_type,
        "training_data_run": training_data_run,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "runtime": runtime,
        "min_app_version": min_app_version,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "files": file_entries,
        "metrics": metrics or {},
    }


def write_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
