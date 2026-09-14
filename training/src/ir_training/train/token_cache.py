"""Content-bound, one-writer, memory-mapped completion-only HF training data.

Only final tensors are persisted, as Arrow IPC streams (not pickle or text
caches). No model weights, optimizer state or model-preflight result is cached.
"""
from __future__ import annotations

import ast
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import uuid

from ir_training.common.cache_store import assert_no_links, cache_lock, digest, hash_file
from ir_training.common.progress import Progress, fingerprint_file, log
from ir_training.train.source_data import iter_sft_rows, validate_sft_messages, _reject_json_constant

VERSION = 1
COLUMNS = ("input_ids", "attention_mask", "labels")


def _function_hash(function) -> str:
    # AST excludes comments/line offsets and unrelated trainer/LoRA functions.
    return digest(ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))), include_attributes=False))


def _special_value(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_special_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _special_value(item) for key, item in value.items()}
    if hasattr(value, "content"):
        return {key: getattr(value, key, None) for key in (
            "content", "single_word", "lstrip", "rstrip", "normalized", "special")}
    raise ValueError(f"Cannot fingerprint tokenizer state {type(value).__name__}; disable token caching for this tokenizer")


def tokenizer_identity(tokenizer) -> dict:
    if not type(tokenizer).__module__.startswith("transformers."):
        raise ValueError("Persistent token caching supports built-in Transformers fast tokenizers; disable token caching for custom subclasses")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not callable(getattr(backend, "to_str", None)):
        raise ValueError("Persistent token caching requires a serializable fast tokenizer; use --no-token-cache for custom/slow tokenizers")
    state = json.loads(backend.to_str())
    # HF's per-call default is no truncation/no padding. These backend fields
    # are transient mutable call state, not tokenizer behavior in this path.
    state.pop("truncation", None)
    state.pop("padding", None)
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "backend_sha256": digest(state),
        "vocabulary_sha256": digest(tokenizer.get_vocab()),
        "chat_template_sha256": digest(getattr(tokenizer, "chat_template", None)),
        "special_tokens": _special_value(getattr(tokenizer, "special_tokens_map_extended", {})),
        "token_ids": {name: getattr(tokenizer, name, None) for name in (
            "pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id")},
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "split_special_tokens": getattr(tokenizer, "split_special_tokens", False),
        # HF 5.16 renamed the fast backend and removed older encode helpers.
        # Bind the available methods, while package versions pin that API shape.
        "implementation": {name: _function_hash(getattr(tokenizer, name)) for name in (
            "__call__", "apply_chat_template", "_batch_encode_plus", "_encode_plus", "_encode")
            if callable(getattr(tokenizer, name, None))},
    }


def identity(data_files: dict, adapter, tokenizer, *, max_seq_length: int,
             input_vocab_size: int, label_vocab_size: int, interval: float = 10) -> dict:
    from ir_training.models.base import ModelAdapter
    from ir_training.train import sft

    formatter = adapter.format_example
    # Built-in adapters inherit this self-contained formatting function using
    # only chat_template_kwargs. An override can depend on arbitrary helpers or
    # instance state, so config alone cannot establish a safe cache identity.
    if getattr(formatter, "__func__", None) is not ModelAdapter.format_example:
        raise ValueError("Persistent token caching cannot fingerprint custom adapter formatters; use --no-token-cache")
    formatting_options = {"chat_template_kwargs": adapter.config.get("chat_template_kwargs") or {}}
    functions = (sft._completion_suffix_text, sft._tokenize_completion_only_row,
                 sft._tokenize_text, _validate_tensor_row, _write_tokens, _source_rows, _special_value,
                 tokenizer_identity, formatter, iter_sft_rows, validate_sft_messages, _reject_json_constant)
    return {
        "version": VERSION,
        "sources": {name: fingerprint_file(Path(path), count_rows=True, interval=interval)
                    for name, path in sorted(data_files.items())},
        "tokenizer": tokenizer_identity(tokenizer),
        # Snapshot, not references to mutable adapter options: the post-build
        # identity check must detect changes made during formatting.
        "formatting_options": json.loads(json.dumps(formatting_options, allow_nan=False)),
        "formatter": f"{formatter.__module__}.{formatter.__qualname__}",
        "implementation": {function.__qualname__: _function_hash(function) for function in functions},
        "python": list(sys.version_info[:3]),
        "packages": {name: version(name) for name in ("transformers", "tokenizers", "datasets", "pyarrow")},
        "max_seq_length": max_seq_length, "input_vocab_size": input_vocab_size,
        "label_vocab_size": label_vocab_size, "raw_token_check_policy": "all_rows",
        "columns": list(COLUMNS), "arrow_storage": "ipc_stream_int32_v1",
    }


def _validate_tensor_row(row: dict, *, max_seq_length: int, input_vocab_size: int,
                         label_vocab_size: int) -> None:
    if set(row) != set(COLUMNS):
        raise ValueError("Token cache row must contain exactly input_ids, attention_mask, labels")
    ids, mask, labels = (row[name] for name in COLUMNS)
    if not all(isinstance(values, list) and all(type(value) is int for value in values)
               for values in (ids, mask, labels)):
        raise ValueError("Token cache tensors must be lists of integer IDs (not booleans)")
    if not 2 <= len(ids) <= max_seq_length or len(mask) != len(ids) or len(labels) != len(ids):
        raise ValueError("Token cache tensor lengths are inconsistent or exceed max_seq_length")
    if any(value != 1 for value in mask) or any(not 0 <= value < input_vocab_size for value in ids):
        raise ValueError("Token cache attention mask or input vocabulary check failed")
    first = next((index for index, value in enumerate(labels) if value != -100), len(labels))
    if not 0 < first < len(labels) or labels[first:] != ids[first:]:
        raise ValueError("Token cache labels must mask the prompt and supervise the exact completion suffix")
    if any(not 0 <= value < label_vocab_size for value in labels[first:]):
        raise ValueError("Token cache completion label is outside output vocabulary")


def _source_rows(path: Path):
    yield from iter_sft_rows(path)


def greedy_text_rows(path: Path, adapter, tokenizer, count: int) -> list[dict]:
    if count < 1:
        raise ValueError("preflight.greedy_probe_rows must be positive")
    import itertools
    return [{"prompt_text": adapter.format_example(row, tokenizer=tokenizer, include_assistant=False)}
            for row in itertools.islice(_source_rows(path), count)]


def _write_tokens(directory: Path, data_files: dict, adapter, tokenizer, binding: dict, interval: float) -> dict:
    import pyarrow as pa
    from ir_training.train.sft import _completion_suffix_text, _tokenize_completion_only_row, _tokenize_text

    schema = pa.schema([(name, pa.list_(pa.int32())) for name in COLUMNS])
    summaries = {}
    for split, source in data_files.items():
        summary = {"rows": 0, "raw_rows_checked": 0, "tensor_rows_checked": 0,
                   "max_full_tokens": 0, "max_prompt_tokens": 0, "max_completion_tokens": 0}
        target = directory / f"{split}.arrow"
        with target.open("xb") as stream, pa.ipc.new_stream(stream, schema) as writer, Progress(
                f"Build token cache {split}", total=binding["sources"][split]["rows"], interval=interval) as progress:
            batch = []
            for example in _source_rows(Path(source)):
                full = adapter.format_example(example, tokenizer=tokenizer, include_assistant=True)
                prompt = adapter.format_example(example, tokenizer=tokenizer, include_assistant=False)
                completion = _completion_suffix_text(example, full, prompt)
                row = _tokenize_completion_only_row(tokenizer=tokenizer, prompt_text=prompt,
                    completion_text=completion, full_text=full, max_seq_length=binding["max_seq_length"])
                # Preserve the distinct full-string vocabulary guard: prompt +
                # completion tokenization is not equivalent for every tokenizer.
                raw_ids = _tokenize_text(tokenizer, full)
                if not raw_ids or any(type(value) is not int or not 0 <= value < binding["input_vocab_size"]
                                      for value in raw_ids):
                    raise ValueError(f"Token cache raw input vocabulary check failed at {split}[{summary['rows']}]")
                _validate_tensor_row(row, **{name: binding[name] for name in (
                    "max_seq_length", "input_vocab_size", "label_vocab_size")})
                completion_length = sum(value != -100 for value in row["labels"])
                summary["rows"] += 1
                summary["raw_rows_checked"] += 1
                summary["tensor_rows_checked"] += 1
                for key, value in (("max_full_tokens", len(row["input_ids"])),
                                   ("max_prompt_tokens", len(row["input_ids"]) - completion_length),
                                   ("max_completion_tokens", completion_length)):
                    summary[key] = max(summary[key], value)
                batch.append(row)
                progress.advance()
                if len(batch) == 128:
                    writer.write_batch(pa.RecordBatch.from_pylist(batch, schema=schema))
                    batch.clear()
            if batch:
                writer.write_batch(pa.RecordBatch.from_pylist(batch, schema=schema))
        if not summary["rows"] or summary["rows"] != binding["sources"][split]["rows"]:
            raise ValueError(f"Empty or changed source split while building token cache: {split}")
        summaries[split] = summary
    return summaries


def _json_write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _read_entry(entry: Path, binding: dict, interval: float):
    from datasets import Dataset, DatasetDict

    assert_no_links(entry)
    if not entry.is_dir():
        raise ValueError("Entry is missing, incomplete or a symlink")
    manifest_path = entry / "manifest.json"
    assert_no_links(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("identity") != binding:
        raise ValueError("Token cache identity changed")
    expected = {f"{name}.arrow" for name in binding["sources"]}
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != expected:
        raise ValueError("Incomplete token cache artifact inventory")
    if not isinstance(manifest.get("splits"), dict) or set(manifest["splits"]) != set(binding["sources"]):
        raise ValueError("Incomplete token cache validation receipt")
    actual_files = {path.name for path in entry.iterdir()}
    if actual_files != expected | {"manifest.json"}:
        raise ValueError("Unexpected token cache artifacts")
    result = {}
    for name in binding["sources"]:
        path = entry / f"{name}.arrow"
        assert_no_links(path)
        if not path.resolve().is_relative_to(entry.resolve()):
            raise ValueError("Token cache artifact escapes entry")
        if fingerprint_file(path, interval=interval)["sha256"] != manifest["files"][path.name]:
            raise ValueError(f"Token cache artifact checksum changed: {path.name}")
        summary = manifest["splits"][name]
        rows = binding["sources"][name]["rows"]
        if not isinstance(summary, dict) or rows <= 0 or any(summary.get(key) != rows for key in ("rows", "raw_rows_checked", "tensor_rows_checked")):
            raise ValueError("Token cache validation receipt is incomplete")
    # Verify every artifact before mapping any file. This also permits a
    # corrupt multi-split entry to be quarantined on Windows without live maps.
    for name in binding["sources"]:
        path = entry / f"{name}.arrow"
        rows = binding["sources"][name]["rows"]
        result[name] = Dataset.from_file(str(path), in_memory=False)
        if len(result[name]) != rows or set(result[name].column_names) != set(COLUMNS):
            raise ValueError("Token cache Arrow shape differs from its receipt")
    return DatasetDict(result), manifest


def _miss_reason(root: Path, binding: dict) -> str:
    latest = root / "latest.json"
    assert_no_links(latest)
    if not latest.is_file() or latest.is_symlink():
        return "no completed entry for these inputs (first build or legacy cache)"
    try:
        previous = json.loads(latest.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            return "previous diagnostic identity unavailable"
        changed = [key for key in binding if previous.get(key) != binding[key]]
        return "changed " + ", ".join(changed) if changed else "matching entry missing or removed"
    except (OSError, ValueError, TypeError):
        return "previous diagnostic identity unavailable"


def load_or_build(cache_dir: Path, data_files: dict, adapter, tokenizer, *, max_seq_length: int,
                  input_vocab_size: int, label_vocab_size: int, interval: float = 10,
                  lock_timeout: float = 21600):
    if set(data_files) not in ({"train"}, {"train", "validation"}):
        raise ValueError("Token cache requires train and optional validation sources")
    for name, value in (("max_seq_length", max_seq_length), ("input_vocab_size", input_vocab_size),
                        ("label_vocab_size", label_vocab_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"Token cache requires positive {name}")
    assert_no_links(Path(cache_dir))
    root = Path(cache_dir).resolve() / "tokens-v1"
    assert_no_links(root)
    root.mkdir(parents=True, exist_ok=True)
    parameters = dict(max_seq_length=max_seq_length, input_vocab_size=input_vocab_size,
                      label_vocab_size=label_vocab_size, interval=interval)
    binding = identity(data_files, adapter, tokenizer, **parameters)
    key = digest(binding)
    entry = root / key
    with cache_lock(root, key, timeout=lock_timeout, interval=interval):
        cached = None
        reason = _miss_reason(root, binding)
        if entry.exists() or entry.is_symlink():
            try:
                cached = _read_entry(entry, binding, interval)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reason = f"rejected entry: {exc}"
                assert_no_links(entry)
                if not entry.resolve().is_relative_to(root):
                    raise ValueError("Unsafe token cache entry; refusing to move or reuse it") from exc
                rejected = root / f".rejected-{key[:12]}-{uuid.uuid4().hex}"
                assert_no_links(rejected)
                entry.rename(rejected)
                log(f"TOKEN CACHE rejected artifacts retained at {rejected}")
        if cached is not None:
            status = "hit"
            log(f"TOKEN CACHE HIT {key[:16]}: no full-corpus formatting/tokenization; {entry}")
        else:
            status = "miss"
            log(f"TOKEN CACHE MISS {key[:16]}: {reason}")
            temporary = Path(tempfile.mkdtemp(prefix=f".build-{key[:12]}-", dir=root))
            try:
                # Conservative room check; actual bytes are streamed in bounded
                # batches, never a second full formatted-text or tensor copy.
                estimate = sum(item["rows"] for item in binding["sources"].values()) * max_seq_length * 12
                free = shutil.disk_usage(root).free
                log(f"TOKEN CACHE storage: worst-case tensors about {estimate:,} bytes; free={free:,}")
                if free < estimate + 64 * 1024**2:
                    raise OSError("Insufficient disk space for worst-case token cache; select a larger --token-cache-dir or --no-token-cache")
                summaries = _write_tokens(temporary, data_files, adapter, tokenizer, binding, interval)
                if identity(data_files, adapter, tokenizer, **parameters) != binding:
                    raise ValueError("Token cache inputs/tokenizer/implementation changed during build")
                manifest = {"identity": binding, "splits": summaries,
                            "files": {f"{name}.arrow": hash_file(temporary / f"{name}.arrow") for name in data_files}}
                _json_write(temporary / "manifest.json", manifest)
                temporary.rename(entry)
            finally:
                if temporary.exists():
                    assert_no_links(temporary)
                    if temporary.parent.resolve() != root or not temporary.resolve().is_relative_to(root):
                        raise ValueError("Unsafe token cache temporary cleanup path")
                    for child in temporary.rglob("*"):
                        assert_no_links(child)
                    shutil.rmtree(temporary)
            cached = _read_entry(entry, binding, interval)
            # Diagnostic only, not an authority for cache validity. Concurrent
            # different keys publish via unique temp names + atomic replacement.
            diagnostic = root / f".latest-{uuid.uuid4().hex}.json"
            _json_write(diagnostic, binding)
            os.replace(diagnostic, root / "latest.json")
            log(f"TOKEN CACHE stored {key[:16]}: ready for all ranks and later runs")
        # Detect concurrent source replacement even on hits; never trust mtime.
        if identity(data_files, adapter, tokenizer, **parameters) != binding:
            raise ValueError("Token cache inputs/tokenizer/implementation changed during reuse")
        dataset, manifest = cached
        for split, summary in manifest["splits"].items():
            log(f"TOKEN CACHE {split}: {summary}")
        return dataset, {"enabled": True, "status": status, "key": key, "directory": str(entry),
                         "manifest_sha256": hash_file(entry / "manifest.json"), "splits": manifest["splits"],
                         "raw_token_check_policy": "all_rows", "model_preflight_cached": False}
