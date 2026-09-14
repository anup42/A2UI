"""Real CPU tokenizer/Arrow regression coverage for persistent SFT tensors."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.common import cache_store
from ir_training.models.base import ModelAdapter
from ir_training.models.gemma import GemmaAdapter
from ir_training.train import sft, token_cache


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "[PAD]": 1, "hello": 2, "answer": 3, "different": 4,
         "HELLO": 5, "helloanswer": 6}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.chat_template = (
        "{{ messages[0]['content'] }} "
        "{% if messages|length > 1 %}{{ messages[-1]['content'] }}{% endif %}"
    )
    return tokenizer


def _rows():
    return [
        {"messages": [{"role": "user", "content": "hello"},
                      {"role": "assistant", "content": "answer"}],
         "metadata": {"nested": {"reviewed": True}}},
        {"messages": [{"role": "user", "content": "HELLO"},
                      {"role": "assistant", "content": "different"}],
         "metadata": {"nested": {"reviewed": "unknown"}}},
    ]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def sample(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    files = {"train": tmp_path / "input/train.jsonl", "validation": tmp_path / "input/val.jsonl"}
    _write_jsonl(files["train"], _rows())
    _write_jsonl(files["validation"], _rows()[:1])
    return {"cache_dir": tmp_path / "cache", "data_files": files,
            "adapter": GemmaAdapter("gemma-test", {"chat_template_kwargs": {"enable_thinking": False}}),
            "tokenizer": _tokenizer(), "max_seq_length": 32,
            "input_vocab_size": 7, "label_vocab_size": 7, "interval": 0.01}


def _identity(sample):
    return token_cache.identity(**{key: value for key, value in sample.items() if key != "cache_dir"})


def _entry(sample):
    return sample["cache_dir"] / "tokens-v1" / cache_store.digest(_identity(sample))


def test_cold_warm_real_memmap_no_repeated_formatting_or_tokenization(sample, monkeypatch, capsys):
    counts = {"format": 0, "tokenize": 0}
    original_format = ModelAdapter.format_example
    original_tokenize = sft._tokenize_text

    def format_counted(self, *args, **kwargs):
        counts["format"] += 1
        return original_format(self, *args, **kwargs)

    def tokenize_counted(*args, **kwargs):
        counts["tokenize"] += 1
        return original_tokenize(*args, **kwargs)

    monkeypatch.setattr(ModelAdapter, "format_example", format_counted)
    monkeypatch.setattr(sft, "_tokenize_text", tokenize_counted)
    cold, cold_report = token_cache.load_or_build(**sample)
    assert cold_report["status"] == "miss"
    assert counts == {"format": 6, "tokenize": 9}
    assert type(cold["train"].data).__name__ == "MemoryMappedTable"
    assert cold["train"][0] == {"input_ids": [2, 3], "attention_mask": [1, 1], "labels": [-100, 3]}
    assert set(cold["train"].column_names) == set(token_cache.COLUMNS)
    # Metadata deliberately has the same nested field typed bool then string.
    # Streaming records never asks Arrow to infer/store those metadata types.
    assert len(cold["train"]) == 2
    warm, warm_report = token_cache.load_or_build(**sample)
    assert warm_report["status"] == "hit"
    assert warm_report["key"] == cold_report["key"]
    assert warm["train"][:] == cold["train"][:]
    assert counts == {"format": 6, "tokenize": 9}
    assert warm_report["model_preflight_cached"] is False
    assert warm_report["raw_token_check_policy"] == "all_rows"
    assert warm_report["splits"]["train"]["raw_rows_checked"] == 2
    output = capsys.readouterr().out
    assert "TOKEN CACHE MISS" in output and "TOKEN CACHE HIT" in output


def test_identical_moved_inputs_share_key_and_cache(sample, tmp_path):
    first, report = token_cache.load_or_build(**sample)
    moved = {}
    for name, source in sample["data_files"].items():
        target = tmp_path / "another_run/prepared" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        moved[name] = target
    result, reused = token_cache.load_or_build(**dict(sample, data_files=moved))
    assert reused["status"] == "hit"
    assert reused["key"] == report["key"]
    assert result["train"][:] == first["train"][:]


def test_training_lora_gpu_and_model_path_options_do_not_change_key(sample):
    before = _identity(sample)
    sample["adapter"].config.update(learning_rate=1e-3, epochs=9, lora={"r": 99},
                                    model_source="/moved/model", device_map="cuda:7", gradient_checkpointing=False)
    assert _identity(sample) == before


@pytest.mark.parametrize("change", ["normalizer", "split_special_tokens", "template", "template_kwargs", "added_token"])
def test_tokenizer_behavior_and_format_options_change_identity(sample, change):
    from tokenizers import AddedToken, normalizers

    before = _identity(sample)
    if change == "normalizer":
        sample["tokenizer"].backend_tokenizer.normalizer = normalizers.Lowercase()
    elif change == "split_special_tokens":
        sample["tokenizer"].split_special_tokens = True
    elif change == "template":
        sample["tokenizer"].chat_template += " "
    elif change == "template_kwargs":
        sample["adapter"].config["chat_template_kwargs"]["enable_thinking"] = True
    else:
        sample["tokenizer"].add_tokens([AddedToken("EXTRA", lstrip=True, normalized=False)])
    assert _identity(sample) != before


def test_transient_backend_padding_and_truncation_do_not_change_identity(sample):
    before = _identity(sample)
    sample["tokenizer"].backend_tokenizer.enable_truncation(max_length=5)
    sample["tokenizer"].backend_tokenizer.enable_padding(length=8, pad_id=1, pad_token="[PAD]")
    assert _identity(sample) == before


@pytest.mark.parametrize("field,value", [("max_seq_length", 64), ("input_vocab_size", 8), ("label_vocab_size", 8)])
def test_effective_limits_change_cache_key(sample, field, value):
    _, original = token_cache.load_or_build(**sample)
    _, changed = token_cache.load_or_build(**dict(sample, **{field: value}))
    assert changed["status"] == "miss"
    assert changed["key"] != original["key"]


@pytest.mark.parametrize("split", ["train", "validation"])
def test_any_split_content_change_invalidates_cache(sample, split):
    _, original = token_cache.load_or_build(**sample)
    _write_jsonl(sample["data_files"][split], _rows()[1:])
    _, changed = token_cache.load_or_build(**sample)
    assert changed["status"] == "miss" and changed["key"] != original["key"]


def test_full_string_raw_vocab_is_checked_separately(sample):
    # Individually tokenized prompt/completion are known IDs 2/3, but the
    # concatenated string maps to ID 6 and must fail before publication.
    sample["tokenizer"].chat_template = (
        "{{ messages[0]['content'] }}{% if messages|length > 1 %}{{ messages[-1]['content'] }}{% endif %}")
    sample["input_vocab_size"] = 6
    with pytest.raises(ValueError, match="raw input vocabulary"):
        token_cache.load_or_build(**sample)
    assert not _entry(sample).exists()


@pytest.mark.parametrize("failure", ["overflow", "prefix", "label_vocab", "empty_split"])
def test_invalid_training_rows_never_publish(sample, failure):
    if failure == "overflow":
        sample["max_seq_length"] = 1
        match = "overflow"
    elif failure == "prefix":
        sample["tokenizer"].chat_template = (
            "{% if messages|length > 1 %}different {{ messages[-1]['content'] }}"
            "{% else %}hello {% endif %}")
        match = "exact prefix"
    elif failure == "label_vocab":
        sample["label_vocab_size"] = 3
        match = "output vocabulary"
    else:
        _write_jsonl(sample["data_files"]["validation"], [])
        match = "Empty or changed source split"
    with pytest.raises(ValueError, match=match):
        token_cache.load_or_build(**sample)
    root = sample["cache_dir"] / "tokens-v1"
    assert not list(root.glob(".build-*"))
    assert not _entry(sample).exists()


def test_builder_exception_cleans_only_its_temporary_directory_and_releases_lock(sample, monkeypatch):
    def fail_writer(directory, *_args):
        (directory / "partial.arrow").write_bytes(b"partial")
        raise RuntimeError("injected builder failure")

    monkeypatch.setattr(token_cache, "_write_tokens", fail_writer)
    key = cache_store.digest(_identity(sample))
    root = sample["cache_dir"] / "tokens-v1"
    root.mkdir(parents=True)
    preserved = root / "unrelated.txt"
    preserved.write_text("preserve", encoding="utf-8")
    with pytest.raises(RuntimeError, match="injected builder failure"):
        token_cache.load_or_build(**sample)
    assert not list(root.glob(".build-*"))
    assert preserved.read_text(encoding="utf-8") == "preserve"
    with cache_store.cache_lock(root, key, timeout=0.2, interval=0.01):
        pass


def test_formatter_options_changed_during_build_reject_publication(sample, monkeypatch):
    original = token_cache._write_tokens

    def change_format_options(directory, data_files, adapter, tokenizer, binding, interval):
        result = original(directory, data_files, adapter, tokenizer, binding, interval)
        adapter.config["chat_template_kwargs"]["enable_thinking"] = True
        return result

    monkeypatch.setattr(token_cache, "_write_tokens", change_format_options)
    initial_entry = _entry(sample)
    with pytest.raises(ValueError, match="changed during build"):
        token_cache.load_or_build(**sample)
    assert not initial_entry.exists()
    assert not list(initial_entry.parent.glob(".build-*"))


@pytest.mark.parametrize("corruption", ["checksum", "missing", "extra", "receipt", "unsafe_inventory",
                                        "manifest_list", "manifest_null", "files_null", "splits_null", "summary_null"])
def test_corrupted_or_incomplete_entries_are_rejected_and_rebuilt(sample, corruption):
    dataset, first = token_cache.load_or_build(**sample)
    # Release the Windows memory map before deliberately editing its backing file.
    del dataset
    entry = Path(first["directory"])
    if corruption == "checksum":
        with (entry / "train.arrow").open("ab") as stream:
            stream.write(b"tampered")
    elif corruption == "missing":
        (entry / "train.arrow").unlink()
    elif corruption == "extra":
        (entry / "extra.json").write_text("{}", encoding="utf-8")
    else:
        manifest_path = entry / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if corruption == "receipt":
            manifest["splits"]["train"]["raw_rows_checked"] = 1
        elif corruption == "unsafe_inventory":
            manifest["files"]["../outside.arrow"] = manifest["files"].pop("train.arrow")
        elif corruption == "manifest_list":
            manifest = []
        elif corruption == "manifest_null":
            manifest = None
        elif corruption == "files_null":
            manifest["files"] = None
        elif corruption == "splits_null":
            manifest["splits"] = None
        else:
            manifest["splits"]["train"] = None
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, repaired = token_cache.load_or_build(**sample)
    assert repaired["status"] == "miss"
    assert repaired["key"] == first["key"]
    assert len(list(entry.parent.glob(".rejected-*"))) == 1


def test_custom_tokenizer_subclasses_are_rejected(sample):
    from transformers import PreTrainedTokenizerFast

    class CustomTokenizer(PreTrainedTokenizerFast):
        pass

    custom = CustomTokenizer(tokenizer_object=sample["tokenizer"].backend_tokenizer)
    with pytest.raises(ValueError, match="custom subclasses"):
        token_cache.tokenizer_identity(custom)


def test_custom_adapter_formatters_require_explicit_uncached_path(sample):
    class CustomAdapter(GemmaAdapter):
        def format_example(self, example, tokenizer=None, include_assistant=True):
            return self.model_id + super().format_example(example, tokenizer, include_assistant)

    sample["adapter"] = CustomAdapter("runtime-dependent-prefix", sample["adapter"].config)
    with pytest.raises(ValueError, match="custom.*format|format.*custom"):
        _identity(sample)


@pytest.mark.parametrize("unsafe", ["root", "entry", "artifact"])
def test_windows_junction_paths_are_rejected_without_following_or_moving(sample, monkeypatch, unsafe):
    _, report = token_cache.load_or_build(**sample)
    entry = Path(report["directory"])
    target = {"root": sample["cache_dir"], "entry": entry, "artifact": entry / "train.arrow"}[unsafe]
    original = getattr(Path, "is_junction", lambda _path: False)

    def is_junction(path):
        return path == target or original(path)

    monkeypatch.setattr(Path, "is_junction", is_junction, raising=False)
    with pytest.raises(ValueError, match="Symlink/junction"):
        token_cache.load_or_build(**sample)
    assert entry.exists()


def test_cache_lock_timeout_is_visible_and_live_lock_is_not_removed(tmp_path, capsys):
    with cache_store.cache_lock(tmp_path, "shared", timeout=0.3, interval=0.01):
        with pytest.raises(TimeoutError, match="do not delete a live lock"):
            with cache_store.cache_lock(tmp_path, "shared", timeout=0.08, interval=0.01):
                pytest.fail("second holder must not acquire live lock")
        assert (tmp_path / ".locks/shared.lock").is_file()
    with cache_store.cache_lock(tmp_path, "shared", timeout=0.3, interval=0.01):
        pass
    assert "CACHE WAIT" in capsys.readouterr().out


def _spawned_writer(directory, data_files, adapter, tokenizer, binding, interval):
    import os
    global _original_spawned_writer, _spawn_marker_dir
    (_spawn_marker_dir / f"built-{os.getpid()}").write_text("build", encoding="utf-8")
    return _original_spawned_writer(directory, data_files, adapter, tokenizer, binding, interval)


def _spawn_cache_worker(root, train, validation, marker_dir, barrier, queue):
    import traceback
    global _original_spawned_writer, _spawn_marker_dir
    _original_spawned_writer = token_cache._write_tokens
    _spawn_marker_dir = Path(marker_dir)
    token_cache._write_tokens = _spawned_writer
    try:
        tokenizer = _tokenizer()
        barrier.wait(timeout=45)
        dataset, report = token_cache.load_or_build(
            Path(root), {"train": Path(train), "validation": Path(validation)},
            GemmaAdapter("gemma-test", {"chat_template_kwargs": {"enable_thinking": False}}), tokenizer,
            max_seq_length=32, input_vocab_size=7, label_vocab_size=7, interval=0.05, lock_timeout=45)
        queue.put({"status": report["status"], "key": report["key"], "row": dataset["train"][0]})
    except BaseException:
        queue.put({"error": traceback.format_exc()})


def test_two_spawned_processes_publish_exactly_one_entry(sample, tmp_path):
    ctx = multiprocessing.get_context("spawn")
    queue, barrier = ctx.Queue(), ctx.Barrier(2)
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    args = (str(sample["cache_dir"]), str(sample["data_files"]["train"]),
            str(sample["data_files"]["validation"]), str(marker_dir), barrier, queue)
    processes = [ctx.Process(target=_spawn_cache_worker, args=args) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert all("error" not in result for result in results), results
        assert sorted(result["status"] for result in results) == ["hit", "miss"]
        assert results[0]["key"] == results[1]["key"]
        assert results[0]["row"] == results[1]["row"]
        assert len(list(marker_dir.glob("built-*"))) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        queue.close()


@pytest.mark.parametrize("field,value", [("input_ids", [True, 3]), ("attention_mask", [1, 0]),
                                         ("labels", [-100, 2]), ("labels", [2, 3]),
                                         ("labels", [-100, -100]), ("labels", [-100])])
def test_tensor_shape_and_completion_mask_contract_is_strict(field, value):
    row = {"input_ids": [2, 3], "attention_mask": [1, 1], "labels": [-100, 3]}
    row[field] = value
    with pytest.raises(ValueError):
        token_cache._validate_tensor_row(row, max_seq_length=32, input_vocab_size=7, label_vocab_size=7)
