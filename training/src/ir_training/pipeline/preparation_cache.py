"""Reuse only complete, content-bound preparations; never trust size/mtime alone.

The cache contains small receipts pointing to completed local run directories,
not another multi-GB dataset. Removing an old run simply causes a cache miss.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

from ir_training.common.config import repo_root
from ir_training.common.progress import Progress, fingerprint_file, log


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def identity(plan: dict, source_hashes: dict) -> dict:
    root, options = repo_root(), plan["options"]
    code = set()
    for directory in (root / "training/src/ir_training", root / "dataset/src/pipeline"):
        code.update(directory.rglob("*.py"))
    code.update((root / "dataset/schema").rglob("*.json"))
    code.add(root / "training/scripts/prepare_review_training.py")
    model = Path(options["model_dir"])
    assets = {str(path.relative_to(model)).replace("\\", "/"): _sha(path)
              for path in sorted(model.rglob("*"))
              if path.is_file() and path.suffix.lower() in {".json", ".model", ".jinja", ".txt", ".vocab", ".tiktoken"}}
    packages = {}
    for name in ("transformers", "tokenizers", "jsonschema", "referencing", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "version": 1, "source_files": source_hashes, "tokenizer_directory": str(model),
        "tokenizer_assets": assets, "python": sys.version, "packages": packages,
        "implementation": {str(path.relative_to(root)).replace("\\", "/"): _sha(path) for path in sorted(code)},
        "shared_prompt": plan["shared_prompt"], "goldens": plan["goldens"],
        "options": {key: options[key] for key in ("profile", "input_dir", "source_run_dir", "seed", "max_seq_length", "max_input_tokens")},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _artifact_paths(output: Path) -> list[Path]:
    return [output / "data_audit.json", *sorted((output / "prepared").glob("*.json*"))]


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if value == "data_audit.json":
        return path
    if len(path.parts) != 2 or path.parts[0] != "prepared" or path.suffix not in {".json", ".jsonl"} or ".." in path.parts:
        raise ValueError("Invalid cached artifact path")
    return path


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".receipt-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def publish(cache: Path, output: Path, binding: dict, *, interval: float = 10) -> None:
    files = {str(path.relative_to(output)).replace("\\", "/"): fingerprint_file(path, interval=interval)["sha256"]
             for path in _artifact_paths(output)}
    receipt = output / "preparation_receipt.json"
    _atomic_json(receipt, {"identity": binding, "files": files})
    _atomic_json(cache / f"{_digest(binding)}.json", {"output": str(output), "receipt_sha256": _sha(receipt)})
    log(f"Verified preparation registered for reuse: {cache}")


def restore(cache: Path, output: Path, binding: dict, *, interval: float = 10) -> bool:
    index = cache / f"{_digest(binding)}.json"
    try:
        if not index.is_file():
            log("Preparation cache miss; running full validation and tokenization")
            return False
        pointer = json.loads(index.read_text(encoding="utf-8"))
        previous = Path(pointer["output"]).resolve(strict=True)
        receipt = previous / "preparation_receipt.json"
        if _sha(receipt) != pointer["receipt_sha256"]:
            raise ValueError("Preparation receipt changed")
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved["identity"] != binding:
            raise ValueError("Preparation identity changed")
        files = saved["files"]
        required = {"data_audit.json", "prepared/manifest.json", "prepared/train.jsonl", "prepared/val.jsonl",
                    "prepared/golden32.jsonl", "prepared/golden35.jsonl", "prepared/prompt_scaffolds.json",
                    "prepared/shared_prompt.json", "prepared/inference_prompt.json", "prepared/source_prompt_scaffolds.json"}
        if not isinstance(files, dict) or not required <= set(files):
            raise ValueError("Preparation receipt is incomplete")
        for name, digest in files.items():
            relative = _safe_relative(name)
            path = previous.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(previous) or fingerprint_file(path, interval=interval)["sha256"] != digest:
                raise ValueError(f"Cached artifact changed: {name}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log(f"Preparation cache rejected ({exc}); running full preparation")
        return False
    if (output / "prepared").exists() or (output / "data_audit.json").exists():
        raise FileExistsError("Cache restore requires a fresh preparation destination")
    temporary = Path(tempfile.mkdtemp(prefix=".prepared-reuse-", dir=output))
    try:
        for name, expected in files.items():
            source, destination = previous / name, temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with Progress(f"Reuse {name}", total=source.stat().st_size, unit="bytes", interval=interval) as progress:
                with source.open("rb") as reader, destination.open("xb") as writer:
                    for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                        writer.write(block)
                        digest.update(block)
                        progress.advance(len(block))
            if digest.hexdigest() != expected:
                raise ValueError(f"Cached artifact changed during copy: {name}")
        (temporary / "prepared").rename(output / "prepared")
        (temporary / "data_audit.json").rename(output / "data_audit.json")
        _atomic_json(output / "cache_reuse.json", {"previous_output": str(previous), "identity_sha256": _digest(binding),
                                                  "receipt_sha256": pointer["receipt_sha256"]})
    finally:
        # Only the fresh temporary child created by this invocation is removed.
        if temporary.exists():
            if temporary.is_symlink() or not temporary.resolve().is_relative_to(output.resolve()):
                raise ValueError("Refusing cleanup outside the fresh cache restore directory")
            shutil.rmtree(temporary)
    log(f"Reused verified preparation from {previous}; no source validation or tokenization repeated")
    return True
