"""Content-bound preparation artifacts, independent of old training runs.

Only preprocessing dependencies participate in the identity. Trainer, LoRA,
GPU, logging and experiment settings do not change the prepared examples.
Entries are published atomically and are never used without checking every
artifact. Version-one run-directory pointers intentionally miss once.
"""
from __future__ import annotations

import ast
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import uuid

from ir_training.common.cache_store import cache_lock
from ir_training.common.config import repo_root
from ir_training.common.progress import Progress, fingerprint_file, log


CACHE_VERSION = 2
REQUIRED_ARTIFACTS = frozenset({
    "data_audit.json", "prepared/manifest.json", "prepared/train.jsonl", "prepared/val.jsonl",
    "prepared/golden32.jsonl", "prepared/golden35.jsonl", "prepared/prompt_scaffolds.json",
    "prepared/bixby50.jsonl",
    "prepared/shared_prompt.json", "prepared/inference_prompt.json", "prepared/source_prompt_scaffolds.json",
})
# These files also contain model/training configuration and orchestration.
# Bind only their preprocessing/verification functions, never their LoRA recipe.
_FUNCTIONS = {
    "training/src/ir_training/pipeline/golden_training.py": (
        "_group_split", "_load_tokenizer", "_strict_rows", "_prepare_uncached", "_write",
    ),
    "training/scripts/prepare_review_training.py": ("sha256", "rows", "source_identity", "verify_prepared"),
}
_MODULE_ROOTS = (
    "ir_training.data.audit_filter", "ir_training.data.build_pairs", "ir_training.data.express_preparation",
    "ir_training.data.shared_prompt", "ir_training.eval.golden_set", "ir_training.eval.prepared_contract",
)
# Progress and provenance do not determine accepted examples, token lengths or
# split membership. Keeping them out permits console-log improvements.
_NON_SEMANTIC_MODULES = {"ir_training.common.progress", "ir_training.common.git"}


def _sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _module_path(root: Path, name: str) -> Path | None:
    for prefix, directory in (("ir_training", "training/src"), ("pipeline", "dataset/src")):
        if name == prefix or name.startswith(prefix + "."):
            path = root / directory / Path(*name.split("."))
            for candidate in (path.with_suffix(".py"), path / "__init__.py"):
                if candidate.is_file():
                    return candidate
    return None


def _imports(tree: ast.AST, module: str, *, package: bool = False) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parents = module.split(".") if package else module.split(".")[:-1]
                if node.level > len(parents):
                    continue
                base = ".".join([*parents[:len(parents) - node.level + 1], *base.split(".")]).strip(".")
            if base:
                names.add(base)
                names.update(base + "." + alias.name for alias in node.names if alias.name != "*")
    return names


def _implementation(root: Path) -> dict[str, str]:
    """Statically follow local preprocessing imports without importing ML code.

AST fingerprints ignore whitespace/comments and function line-number changes.
Module imports are followed transitively, including relative codec imports and
imports inside functions. Schema/catalog files are bound separately below.
"""
    result, pending, visited = {}, set(_MODULE_ROOTS), set()
    for relative, functions in _FUNCTIONS.items():
        tree = ast.parse((root / relative).read_text(encoding="utf-8-sig"))
        nodes = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in functions:
            if name not in nodes:
                raise ValueError(f"Missing preparation identity function: {relative}:{name}")
            node = nodes[name]
            result[f"{relative}:{name}"] = _digest(ast.dump(node, include_attributes=False))
            pending.update(_imports(node, ""))
    while pending:
        name = pending.pop()
        if name in visited or name in _NON_SEMANTIC_MODULES:
            continue
        visited.add(name)
        path = _module_path(root, name)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        result[path.relative_to(root).as_posix()] = _digest(ast.dump(tree, include_attributes=False))
        pending.update(_imports(tree, name, package=path.name == "__init__.py"))
    for path in sorted((root / "dataset/schema").rglob("*.json")):
        result[path.relative_to(root).as_posix()] = _sha(path)
    return dict(sorted(result.items()))


def identity(plan: dict, source_hashes: dict) -> dict:
    options = plan["options"]
    model = Path(options["model_dir"])
    assets = {path.relative_to(model).as_posix(): _sha(path)
              for path in sorted(model.rglob("*"))
              if path.is_file() and path.suffix.lower() in {".json", ".model", ".jinja", ".txt", ".vocab", ".tiktoken"}}
    packages = {}
    for name in ("transformers", "tokenizers", "jsonschema", "referencing", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "version": CACHE_VERSION, "source_files": source_hashes, "tokenizer_directory": str(model),
        "tokenizer_assets": assets, "python": sys.version, "packages": packages,
        "implementation": _implementation(repo_root()), "shared_prompt": plan["shared_prompt"], "goldens": plan["goldens"],
        "options": {key: options[key] for key in ("profile", "input_dir", "source_run_dir", "seed", "max_seq_length", "max_input_tokens")},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _namespace(cache: Path) -> Path:
    return Path(cache) / "prepared-v2"


def lock(cache: Path, binding: dict, *, interval: float = 10):
    """Serialize the whole lookup/build/publish transaction across launchers."""
    _assert_no_links(_namespace(cache))
    return cache_lock(_namespace(cache), "build-" + _digest(binding), interval=interval)


def _assert_no_links(path: Path) -> None:
    # Reject even links that currently point inside the allowed directory: a
    # retargeted link must never turn a checksum/copy into an arbitrary read.
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError(f"Symlink/junction is not a cache artifact: {candidate}")


def _artifact_paths(output: Path) -> list[Path]:
    return [output / "data_audit.json", *sorted((output / "prepared").glob("*.json*"))]


def _safe_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ValueError("Invalid cached artifact path")
    path = PurePosixPath(value)
    if value == "data_audit.json":
        return path
    if (str(path) != value or len(path.parts) != 2 or path.parts[0] != "prepared"
            or path.suffix not in {".json", ".jsonl"} or ".." in path.parts):
        raise ValueError("Invalid cached artifact path")
    return path


def _atomic_json(path: Path, value) -> None:
    _assert_no_links(path)
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


def _validated_manifest(entry: Path, binding: dict, *, interval: float) -> dict:
    _assert_no_links(entry)
    manifest = entry / "manifest.json"
    _assert_no_links(manifest)
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or saved.get("version") != CACHE_VERSION:
        raise ValueError("Unsupported preparation cache version; one rebuild is required")
    if saved.get("identity") != binding:
        raise ValueError("Preparation cache identity changed")
    files = saved.get("files")
    if not isinstance(files, dict) or not REQUIRED_ARTIFACTS <= files.keys():
        raise ValueError("Preparation cache manifest is incomplete")
    for name, digest in files.items():
        relative = _safe_relative(name)
        path = entry.joinpath(*relative.parts)
        _assert_no_links(path)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid cached artifact checksum: {name}")
        if not path.is_file() or not path.resolve().is_relative_to(entry.resolve()):
            raise ValueError(f"Missing or unsafe cached artifact: {name}")
        if fingerprint_file(path, interval=interval)["sha256"] != digest:
            raise ValueError(f"Cached artifact checksum changed: {name}")
    return saved


def _copy_checked(source: Path, destination: Path, expected: str, *, interval: float) -> None:
    _assert_no_links(source)
    _assert_no_links(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with Progress(f"Cache copy {source.name}", total=source.stat().st_size, unit="bytes", interval=interval) as progress:
        with source.open("rb") as reader, destination.open("xb") as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                writer.write(block)
                digest.update(block)
                progress.advance(len(block))
            writer.flush()
            os.fsync(writer.fileno())
    if digest.hexdigest() != expected:
        raise ValueError(f"Cached artifact changed during copy: {source.name}")


def _cleanup_temporary(temporary: Path, parent: Path) -> None:
    if temporary.exists():
        _assert_no_links(temporary)
        if temporary.parent.resolve() != parent.resolve() or not temporary.name.startswith(".prepared-"):
            raise ValueError("Refusing cleanup outside fresh cache temporary directory")
        for child in temporary.rglob("*"):
            _assert_no_links(child)
        shutil.rmtree(temporary)


def _miss(cache: Path, binding: dict) -> str:
    # Explain changed identity categories without dataset contents or trusting
    # old pointer destinations. Restrict diagnostics to a few entries.
    previous = []
    for path in list(_namespace(cache).glob("*/manifest.json"))[:20]:
        try:
            _assert_no_links(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved, dict) and isinstance(saved.get("identity"), dict):
                previous.append(saved["identity"])
        except (OSError, ValueError, TypeError):
            continue
    if previous:
        changes = min(([key for key in binding if old.get(key) != binding[key]] for old in previous), key=len)
        return "identity changed: " + ", ".join(changes or ["entry missing/incomplete"])
    if any(Path(cache).glob("*.json")):
        return "legacy run-pointer cache is not portable; one v2 rebuild is required"
    return "no complete entry for this data/tokenizer/preprocessing identity (first run or removed cache)"


def publish(cache: Path, output: Path, binding: dict, *, interval: float = 10) -> None:
    cache, output = Path(cache), Path(output)
    namespace, key = _namespace(cache), _digest(binding)
    _assert_no_links(namespace)
    _assert_no_links(output)
    namespace.mkdir(parents=True, exist_ok=True)
    entry = namespace / key
    with cache_lock(namespace, key, interval=interval):
        files = {}
        for path in _artifact_paths(output):
            _assert_no_links(path)
            relative = path.relative_to(output).as_posix()
            _safe_relative(relative)
            files[relative] = fingerprint_file(path, interval=interval)["sha256"]
        if not REQUIRED_ARTIFACTS <= files.keys():
            raise ValueError("Cannot publish incomplete prepared artifacts")
        # Reused runs must not overwrite or hard-link the immutable owned entry.
        if entry.exists():
            try:
                saved = _validated_manifest(entry, binding, interval=interval)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                _assert_no_links(entry)
                rejected = namespace / f".rejected-{key}-{uuid.uuid4().hex}"
                entry.rename(rejected)
                log(f"Preparation CACHE REPAIR: moved invalid entry aside ({exc}): {rejected}")
            else:
                if saved["files"] != files:
                    raise ValueError("Prepared output differs from the existing cache with the same identity")
                _atomic_json(output / "preparation_receipt.json", saved)
                log(f"Preparation CACHE STORED: existing verified owned entry {entry}")
                return
        saved = {"version": CACHE_VERSION, "identity": binding, "files": files}
        temporary = Path(tempfile.mkdtemp(prefix=".prepared-publish-", dir=namespace))
        try:
            for name, expected in files.items():
                _copy_checked(output / name, temporary / name, expected, interval=interval)
            _atomic_json(temporary / "manifest.json", saved)
            temporary.rename(entry)
        finally:
            _cleanup_temporary(temporary, namespace)
        _atomic_json(output / "preparation_receipt.json", saved)
    log(f"Preparation CACHE STORED: {entry}; reusable even after this run is deleted")


def restore(cache: Path, output: Path, binding: dict, *, interval: float = 10) -> bool:
    cache, output = Path(cache), Path(output)
    namespace, key = _namespace(cache), _digest(binding)
    entry = namespace / key
    installed = False
    try:
        _assert_no_links(namespace)
        _assert_no_links(output)
        with cache_lock(namespace, key, interval=interval):
            if not entry.exists():
                log(f"Preparation CACHE MISS [{key[:12]}]: {_miss(cache, binding)}; running preparation")
                return False
            saved = _validated_manifest(entry, binding, interval=interval)
            if (output / "prepared").exists() or (output / "data_audit.json").exists():
                raise FileExistsError("Cache restore requires a fresh preparation destination")
            output.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".prepared-reuse-", dir=output))
            try:
                for name, expected in saved["files"].items():
                    _copy_checked(entry / name, temporary / name, expected, interval=interval)
                (temporary / "prepared").rename(output / "prepared")
                installed = True
                (temporary / "data_audit.json").rename(output / "data_audit.json")
                _atomic_json(output / "preparation_receipt.json", saved)
                _atomic_json(output / "cache_reuse.json", {"cache_entry": str(entry), "identity_sha256": key,
                                                          "cache_version": CACHE_VERSION, "manifest_sha256": _sha(entry / "manifest.json")})
            finally:
                _cleanup_temporary(temporary, output)
    except FileExistsError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if installed:
            raise RuntimeError("Cache restore could not finish installing verified files; choose a fresh output directory") from exc
        log(f"Preparation CACHE MISS [{key[:12]}]: rejected/unavailable entry ({exc}); running preparation")
        return False
    log(f"Preparation CACHE HIT [{key[:12]}]: {entry}; filtering and preparation tokenization skipped, integrity checks retained")
    return True
