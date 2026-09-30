"""Immutable implementation identity for a Single-Codex judge run."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


PROVENANCE_NAME = "judge_implementation_provenance.json"
PROVENANCE_SCHEMA_VERSION = "genui_single_codex_implementation.v2"
REPO_ROOT = Path(__file__).resolve().parents[4]


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


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout.strip()


def _implementation_paths() -> list[Path]:
    judge_dir = (
        REPO_ROOT / "dataset" / "src" / "pipeline" / "genui_judge"
    )
    paths = sorted(judge_dir.glob("*.py"))
    paths.extend(
        [
            REPO_ROOT
            / "dataset"
            / "scripts"
            / "run_single_codex_judge_v2.py",
            REPO_ROOT
            / "dataset"
            / "scripts"
            / "capture_single_codex_judge_native.py",
            REPO_ROOT
            / "dataset"
            / "scripts"
            / "watch_single_codex_judge_block.py",
            REPO_ROOT
            / "dataset"
            / "schema"
            / "genui_codex_judgment_v2.schema.json",
            REPO_ROOT
            / "dataset"
            / "schema"
            / "genui_codex_packet_v2.schema.json",
            REPO_ROOT
            / "dataset"
            / "prompts"
            / "genui_single_codex_judge_task_v2.md",
            REPO_ROOT
            / "android"
            / "app"
            / "build.gradle.kts",
            REPO_ROOT
            / "android"
            / "app"
            / "src"
            / "main"
            / "java"
            / "com"
            / "samsung"
            / "genuicraft"
            / "ui"
            / "DatasetRenderCaptureActivity.kt",
        ]
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return sorted(set(paths))


def _identity(benchmark_dir: Path) -> dict[str, Any]:
    inventory = [
        {
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _hash_file(path),
        }
        for path in _implementation_paths()
    ]
    benchmark_artifacts = []
    for name in (
        "judge_instructions.md",
        "judge_protocol.json",
        "judge_schedule.jsonl",
        "selected_genui.jsonl",
        "expected_contracts.jsonl",
        "native_provenance.json",
        "native_capture_manifest.jsonl",
        "native_capture_summary.json",
    ):
        path = benchmark_dir / name
        if path.is_file():
            benchmark_artifacts.append(
                {"name": name, "sha256": _hash_file(path)}
            )
    packet_manifest = benchmark_dir / "packets" / "packet_manifest.jsonl"
    base_packet_ids: list[str] = []
    if packet_manifest.is_file():
        for line in packet_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            packet_id = str(row.get("packet_id") or "")
            if packet_id:
                base_packet_ids.append(packet_id)
    packet_files = sorted(
        directory / f"{packet_id}.json"
        for directory in (
            benchmark_dir / "packets" / "screenshot_only",
            benchmark_dir / "packets" / "source_conditioned",
        )
        for packet_id in base_packet_ids
        if (directory / f"{packet_id}.json").is_file()
    )
    packet_inventory = [
        {
            "path": str(path.relative_to(benchmark_dir)).replace(
                "\\", "/"
            ),
            "sha256": _hash_file(path),
        }
        for path in packet_files
    ]
    return {
        "implementation_inventory": inventory,
        "benchmark_artifacts": benchmark_artifacts,
        "packet_file_count": len(packet_inventory),
        "packet_tree_sha256": hashlib.sha256(
            _canonical_json(packet_inventory).encode("utf-8")
        ).hexdigest(),
    }


def implementation_fingerprint(
    benchmark_dir: str | Path,
) -> tuple[str, dict[str, Any]]:
    root = Path(benchmark_dir).resolve()
    identity = _identity(root)
    fingerprint = hashlib.sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest()
    return fingerprint, identity


def seal_implementation_provenance(
    benchmark_dir: str | Path,
) -> dict[str, Any]:
    root = Path(benchmark_dir).resolve()
    path = root / PROVENANCE_NAME
    fingerprint, identity = implementation_fingerprint(root)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("implementation_fingerprint") != fingerprint:
            raise ValueError(
                "judge implementation changed after provenance was sealed"
            )
        return existing
    git_status = _git("status", "--porcelain=v1", "--untracked-files=all")
    dirty_paths = {
        line[3:].replace("\\", "/")
        for line in git_status.splitlines()
        if len(line) > 3
    }
    inventory_paths = {
        row["path"] for row in identity["implementation_inventory"]
    }
    value = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "implementation_fingerprint": fingerprint,
        **identity,
        "git_commit": _git("rev-parse", "HEAD"),
        "repository_dirty": bool(git_status),
        "inventory_dirty_paths": sorted(dirty_paths & inventory_paths),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "sealed_at": datetime.now(timezone.utc).isoformat(),
    }
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
    return value


def verify_implementation_provenance(
    benchmark_dir: str | Path,
) -> dict[str, Any]:
    root = Path(benchmark_dir).resolve()
    path = root / PROVENANCE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"seal judge implementation provenance first: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    fingerprint, _ = implementation_fingerprint(root)
    if value.get("implementation_fingerprint") != fingerprint:
        raise ValueError(
            "judge implementation/artifact fingerprint changed after seal"
        )
    return value


__all__ = [
    "PROVENANCE_NAME",
    "PROVENANCE_SCHEMA_VERSION",
    "implementation_fingerprint",
    "seal_implementation_provenance",
    "verify_implementation_provenance",
]
