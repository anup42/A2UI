from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class RunPaths:
    run_dir: Path
    queries_path: Path
    responses_path: Path
    genui_path: Path
    metrics_path: Path
    aggregates_path: Path
    artifacts_dir: Path
    manifest_path: Path


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with _jsonl_append_lock(self.path):
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


@contextmanager
def _jsonl_append_lock(path: Path):
    """Small cross-process lock for multi-worker JSONL appends."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("ascii", errors="ignore"))
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 1800:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.close(fd)
        finally:
            lock_path.unlink(missing_ok=True)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    def _iter():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    return _iter()


def load_existing_ids(path: Path, key: str) -> set[str]:
    existing: set[str] = set()
    for row in iter_jsonl(path):
        value = row.get(key)
        if value:
            existing.add(value)
    return existing


def load_jsonl_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        value = row.get(key)
        if value:
            out[value] = row
    return out


def get_run_paths(base_dir: Path, run_id: str, artifact_dir_name: str) -> RunPaths:
    run_dir = base_dir / run_id
    artifacts_dir = run_dir / artifact_dir_name
    genui_path = run_dir / "genui.jsonl"

    return RunPaths(
        run_dir=run_dir,
        queries_path=run_dir / "queries.jsonl",
        responses_path=run_dir / "responses.jsonl",
        genui_path=genui_path,
        metrics_path=run_dir / "metrics.jsonl",
        aggregates_path=run_dir / "aggregates.json",
        artifacts_dir=artifacts_dir,
        manifest_path=run_dir / 'run_manifest.json',
    )

