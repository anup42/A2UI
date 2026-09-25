"""CPU-only checks for consolidated tokenization and process resource budgets."""
from __future__ import annotations

from pathlib import Path
import inspect
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.train import gpu_profile
from ir_training.train.sft import _apply_training_data_seed, _initialize_training_seed, _tokenize_completion_only_row, _tokenize_sft_text_dataset, _validate_tokenized_sft_dataset, train_sft


class Split(list):
    column_names = ["text", "prompt_text", "completion_text"]

    def map(self, function, **kwargs):
        self.map_kwargs = kwargs
        batch = {name: [row[name] for row in self] for name in self.column_names}
        result = function(batch)
        return [dict(zip(result, values, strict=True)) for values in zip(*result.values(), strict=True)]


class Tokenizer:
    def __init__(self, overrides=None):
        self.calls = []
        self.overrides = overrides or {}

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        self.calls.append(text)
        return {"input_ids": self.overrides.get(text, [ord(char) for char in text])}


def row(prompt="PP", completion="CCC"):
    return {"prompt_text": prompt, "completion_text": completion, "text": prompt + completion}


def test_consolidated_tokenization_keeps_masked_tensors_and_all_checks(capsys, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    split = Split([row(), row("P", "C")])
    tokenizer = Tokenizer()
    result = _tokenize_sft_text_dataset({"train": split}, tokenizer, 8, input_vocab_size=128)
    # Prompt + completion + historical full-string vocab check: three calls
    # per row, not the previous seven on rank zero. No diagnostic retokenize.
    assert tokenizer.calls == ["PP", "CCC", "PPCCC", "P", "C", "PC"]
    reference = [_tokenize_completion_only_row(tokenizer=Tokenizer(), prompt_text=r["prompt_text"],
        completion_text=r["completion_text"], full_text=r["text"], max_seq_length=8) for r in split]
    assert result["train"] == reference
    _validate_tokenized_sft_dataset(dataset=result, max_rows=0, input_vocab_size=128, label_vocab_size=128)
    assert split.map_kwargs["keep_in_memory"] is True
    assert split.map_kwargs["load_from_cache_file"] is False
    assert split.map_kwargs["batch_size"] == 128
    output = capsys.readouterr().out
    assert "Tokenize train SFT text [rank 0]" in output
    assert "2/2 rows" in output
    assert "max_full_tokens=5" in output
    assert "raw_rows_checked=2" in output


def test_historical_full_string_vocab_guard_is_not_removed():
    tokenizer = Tokenizer({"PPCCC": [999]})
    with pytest.raises(ValueError, match="outside model vocabulary.*input"):
        _tokenize_sft_text_dataset({"train": Split([row()])}, tokenizer, 8, input_vocab_size=128)


def test_full_string_empty_tokenization_is_rejected():
    with pytest.raises(ValueError, match="empty tokenization"):
        _tokenize_sft_text_dataset({"train": Split([row()])}, Tokenizer({"PPCCC": []}), 8, input_vocab_size=128)


def test_actual_prompt_and_completion_ids_still_validate_against_separate_heads():
    tokenizer = Tokenizer({"CCC": [90], "PPCCC": [3]})
    result = _tokenize_sft_text_dataset({"train": Split([row()])}, tokenizer, 8, input_vocab_size=128)
    with pytest.raises(ValueError, match="label id outside model vocabulary"):
        _validate_tokenized_sft_dataset(dataset=result, max_rows=0, input_vocab_size=128, label_vocab_size=85)
    tokenizer = Tokenizer({"PP": [999], "PPCCC": [3]})
    result = _tokenize_sft_text_dataset({"train": Split([row()])}, tokenizer, 8, input_vocab_size=128)
    with pytest.raises(ValueError, match="token id outside model vocabulary"):
        _validate_tokenized_sft_dataset(dataset=result, max_rows=0, input_vocab_size=128, label_vocab_size=128)


@pytest.mark.parametrize("sample,limit,match", [(row(), 4, "overflow"), (dict(row(), text="different"), 8, "exact prefix"), (row("", "C"), 8, "nonempty")])
def test_consolidation_never_relaxes_prefix_or_overflow_checks(sample, limit, match):
    with pytest.raises(ValueError, match=match):
        _tokenize_sft_text_dataset({"train": Split([sample])}, Tokenizer(), limit, input_vocab_size=128)


def test_bounded_raw_check_count_resets_per_split_and_nonzero_rank_checks(capsys, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    tokenizer = Tokenizer()
    result = _tokenize_sft_text_dataset({name: Split([row(), row()]) for name in ("train", "validation")},
        tokenizer, 8, input_vocab_size=128, token_check_rows=1)
    assert len(result["train"]) == len(result["validation"]) == 2
    assert tokenizer.calls.count("PPCCC") == 2
    output = capsys.readouterr().out
    assert "[rank 3]" in output
    assert "actual masked sequence lengths" not in output


def test_available_cpu_count_respects_affinity_and_cgroup(monkeypatch):
    monkeypatch.setattr(gpu_profile.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(gpu_profile.os, "sched_getaffinity", lambda _: set(range(6)), raising=False)
    monkeypatch.setattr(gpu_profile.Path, "read_text", lambda _path: "400000 100000")
    assert gpu_profile.available_cpu_count() == 4
    monkeypatch.setattr(gpu_profile.Path, "read_text", lambda _path: "max 100000")
    assert gpu_profile.available_cpu_count() == 6


def test_available_cpu_count_handles_legacy_quota_and_unavailable_affinity(monkeypatch):
    monkeypatch.setattr(gpu_profile.os, "cpu_count", lambda: 128)
    def no_affinity(_):
        raise OSError("not supported")
    monkeypatch.setattr(gpu_profile.os, "sched_getaffinity", no_affinity, raising=False)
    def read(path):
        if path.name == "cpu.cfs_quota_us":
            return "200000"
        if path.name == "cpu.cfs_period_us":
            return "100000"
        raise FileNotFoundError(str(path))
    monkeypatch.setattr(gpu_profile.Path, "read_text", read)
    assert gpu_profile.available_cpu_count() == 2


def test_launch_environment_bounds_default_cpu_threads_and_preserves_overrides(monkeypatch):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("PYTHONUNBUFFERED", "0")
    environment = gpu_profile.training_environment({"cuda_visible_devices": "GPU-a,GPU-b"})
    assert environment["OMP_NUM_THREADS"] == "3"
    assert environment["MKL_NUM_THREADS"] == environment["OPENBLAS_NUM_THREADS"] == environment["NUMEXPR_NUM_THREADS"] == "1"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"


def test_auto_dataloader_budget_uses_available_cpus_and_reports_explicit_oversubscription(monkeypatch):
    monkeypatch.setattr(gpu_profile, "available_cpu_count", lambda: 16)
    inventory = {"devices": [{"visible_index": index, "launch_identifier": str(index),
        "name": "NVIDIA H100", "total_memory_bytes": 80 * 1024**3, "compute_capability": [9, 0]}
        for index in range(8)]}
    profile = gpu_profile.build_gpu_profile(inventory, model="e2b")
    assert profile["world_size"] == 8
    assert profile["cuda_visible_devices"] == "0,1,2,3,4,5,6,7"
    assert profile["microbatch"] == 1
    assert profile["gradient_accumulation_steps"] == 4
    assert profile["available_cpu_count"] == 16
    assert profile["dataloader_num_workers"] == 1
    assert profile["total_dataloader_workers"] == 8
    assert profile["cpu_oversubscribed"] is False
    profile = gpu_profile.build_gpu_profile(inventory, model="e2b", dataloader_workers=4)
    assert profile["total_dataloader_workers"] == 32
    assert profile["cpu_worker_budget_overridden"] is True
    assert profile["cpu_oversubscribed"] is True
    with pytest.raises(ValueError, match="CPU count must be positive"):
        gpu_profile.build_gpu_profile(inventory, model="e2b", cpu_count=0)


def test_initialization_seed_is_identical_across_ranks_and_resolves_config_precedence(monkeypatch, capsys):
    calls = []
    for rank in ("0", "3", "7"):
        monkeypatch.setenv("RANK", rank)
        assert _initialize_training_seed({"seed": 123}, {"seed": 456}, seed_setter=calls.append) == 123
    assert calls == [123, 123, 123]
    assert _initialize_training_seed({}, {"seed": 456}, seed_setter=calls.append) == 456
    assert _initialize_training_seed({}, {}, seed_setter=calls.append) == 42
    assert "seed=123 on rank 7 (before model/LoRA load)" in capsys.readouterr().out


@pytest.mark.parametrize("seed", [True, -1, 2**32, 3.5, "42", None])
def test_invalid_initialization_seed_fails_before_setting_rng(seed):
    calls = []
    with pytest.raises(ValueError, match="training.seed must be an integer"):
        _initialize_training_seed({"seed": seed}, {}, seed_setter=calls.append)
    assert calls == []


def test_initialization_seed_reproduces_cpu_random_model_initialization():
    import random
    import numpy as np
    import torch
    from transformers import set_seed
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        def initialized():
            _initialize_training_seed({"seed": 123}, {}, seed_setter=set_seed)
            return random.random(), np.random.random(), torch.nn.Linear(4, 3).weight.detach().clone()
        first, second = initialized(), initialized()
        assert first[:2] == second[:2]
        assert torch.equal(first[2], second[2])
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def test_seed_precedes_model_adapter_initialization_and_resume_rng_flow_is_kept():
    source = inspect.getsource(train_sft)
    seed_call = source.index("initialization_seed = _initialize_training_seed")
    for constructor in ("create_adapter(load_cfg)", "adapter.load_model()", "get_peft_model(model, configured_lora)"):
        assert seed_call < source.index(constructor)
    assert '"seed": initialization_seed' in source
    assert source.index("_apply_training_data_seed(training_cfg, training_args_kwargs, args_params)") < source.index("args_cls(**training_args_kwargs)")
    # Seeding construction does not replace Trainer's checkpoint RNG restore.
    resume_assignment = source.index('train_kwargs["resume_from_checkpoint"] = str(resolved_resume_checkpoint)')
    assert seed_call < resume_assignment < source.index("trainer.train(**train_kwargs)")


def test_data_seed_is_explicit_for_experiments_and_legacy_recipes_are_unchanged():
    kwargs = {"seed": 42}
    _apply_training_data_seed({}, kwargs, {})
    assert kwargs == {"seed": 42}
    _apply_training_data_seed({"data_seed": 19}, kwargs, {"data_seed": object()})
    assert kwargs == {"seed": 42, "data_seed": 19}


@pytest.mark.parametrize("seed", [True, -1, 2**32, 3.5, "42", None])
def test_invalid_data_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="training.data_seed must be an integer"):
        _apply_training_data_seed({"data_seed": seed}, {}, {"data_seed": object()})


def test_explicit_data_seed_requires_trainer_support():
    with pytest.raises(ValueError, match="does not support configured data_seed"):
        _apply_training_data_seed({"data_seed": 42}, {}, {})
