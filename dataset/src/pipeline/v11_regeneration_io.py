"""Byte-preserving, verified inputs and durable journals for opt-in relabeling."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BLOCK = 1024 * 1024
MAX_ROW_BYTES = 16 * BLOCK


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_path(value: Path) -> Path:
    path = Path(value).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Paths must not traverse links: {part}")
    return path.resolve()


def atomic_bytes(path: Path, content: bytes) -> None:
    safe_path(path)
    temporary = path.with_name(path.name + ".pending")
    safe_path(temporary)
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical({"payload": value, "sha256": digest(value)}) + b"\n")


def checked(path: Path) -> Any:
    safe_path(path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or digest(envelope.get("payload")) != envelope.get("sha256"):
        raise ValueError(f"Journal integrity mismatch: {path.name}")
    return envelope["payload"]


class SplitInput:
    """Read restored splits or the repository's checked gzip-part bundle.

    No code from the supplied directory is imported. Bundled manifest validation
    reuses the repository-owned restorer; every raw and compressed hash is also
    checked while streaming. Parts can end in the middle of a UTF-8/JSONL row.
    """

    def __init__(self, directory: Path):
        self.directory = safe_path(directory)
        self.bundle = None
        self.declared = {}
        self.identities: dict[str, Any] = {}
        if (self.directory / "bundle.json").exists():
            module_path = ROOT / "training/data/train/v11/restore.py"
            spec = importlib.util.spec_from_file_location("_v11_checked_restorer", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.bundle = module.load_manifest(self.directory)
            self.identities["bundle.json"] = file_hash(self.directory / "bundle.json")
        elif not all((self.directory / f"{split}.jsonl").is_file() for split in ("train", "val")):
            raise ValueError("Input requires train.jsonl and val.jsonl, or a checked v11 bundle")
        for name in ("manifest.json", "prompt_scaffold.json"):
            path = self.directory / name
            if path.exists():
                safe_path(path)
                self.identities[name] = file_hash(path)
                if name == "manifest.json":
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(manifest, dict):
                        raise ValueError("Input manifest must be an object")
                    self.declared = manifest.get("outputs", {})

    def _chunks(self, split: str):
        if self.bundle is None:
            path = safe_path(self.directory / f"{split}.jsonl")
            with path.open("rb") as stream:
                yield from iter(lambda: stream.read(BLOCK), b"")
            return
        item = next(item for item in self.bundle["files"] if item["name"] == f"{split}.jsonl")
        for part in item["parts"]:
            path = safe_path(self.directory / part["name"])
            if file_hash(path) != part["sha256"]:
                raise ValueError(f"Compressed input hash mismatch: {part['name']}")
            count = 0
            with gzip.open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(BLOCK), b""):
                    count += len(chunk)
                    if count > part["raw_bytes"]:
                        raise ValueError("Compressed part exceeds declared raw size")
                    yield chunk
            if count != part["raw_bytes"]:
                raise ValueError("Compressed part raw size mismatch")

    def rows(self, split: str):
        if split not in {"train", "val"}:
            raise ValueError("Only original train/val splits are permitted; no benchmark data")
        pending = b""
        hasher = hashlib.sha256()
        size = count = 0
        for chunk in self._chunks(split):
            hasher.update(chunk)
            size += len(chunk)
            pieces = (pending + chunk).split(b"\n")
            pending = pieces.pop()
            for line in pieces:
                raw = line + b"\n"
                count += 1
                yield count, raw, self._parse(raw)
            if len(pending) > MAX_ROW_BYTES:
                raise ValueError("Input JSONL row exceeds bounded size")
        if pending:
            count += 1
            yield count, pending, self._parse(pending)
        identity = {"rows": count, "bytes": size, "sha256": hasher.hexdigest()}
        declared = self.declared.get(f"{split}.jsonl", {})
        if any(key in declared and declared[key] != identity[key] for key in identity):
            raise ValueError(f"Split differs from input manifest: {split}")
        if self.bundle is not None:
            item = next(item for item in self.bundle["files"] if item["name"] == f"{split}.jsonl")
            if any(identity[key] != item[key] for key in identity):
                raise ValueError(f"Bundled split identity mismatch: {split}")
        previous = self.identities.get(f"{split}.jsonl")
        if previous is not None and previous != identity:
            raise ValueError(f"Input changed during regeneration: {split}")
        self.identities[f"{split}.jsonl"] = identity

    @staticmethod
    def _parse(raw: bytes) -> dict:
        if len(raw) > MAX_ROW_BYTES:
            raise ValueError("Input JSONL row exceeds bounded size")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Each split row must be a JSON object")  # noqa: TRY004 - external artifact schema
        return value

    def recheck_sidecars(self) -> None:
        for name, expected in self.identities.items():
            if isinstance(expected, str) and file_hash(self.directory / name) != expected:
                raise ValueError(f"Input sidecar changed: {name}")
