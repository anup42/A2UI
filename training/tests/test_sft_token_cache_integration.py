"""Real CPU Gemma4/PEFT preflight through the persistent token-cache branch."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.models.base import ModelAdapter
from ir_training.models.gemma import GemmaAdapter
from ir_training.train import sft


@pytest.fixture
def local_fixture(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    pytest.importorskip("peft", minversion="0.19.0")
    pytest.importorskip("datasets")
    tokenizers = pytest.importorskip("tokenizers")
    # This is a CPU smoke fixture even on a CUDA-capable CI worker. No distributed
    # process group, downloads, real checkpoint or Golden evaluation is used.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model_dir, data_dir = tmp_path / "model", tmp_path / "data"
    model_dir.mkdir()
    data_dir.mkdir()
    words = ["[PAD]", "[UNK]", "[EOS]", "[BOS]", "user", "assistant", ":", "prompt", "result"]
    vocabulary = {word: index for index, word in enumerate(words + [f"item{index}" for index in range(23)])}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]", bos_token="[BOS]",
        model_max_length=64,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' : ' + message['content'] + ' ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant : ' }}{% endif %}"
    )
    tokenizer.save_pretrained(model_dir)
    text_config = transformers.Gemma4TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        max_position_embeddings=64, vocab_size_per_layer_input=32, hidden_size_per_layer_input=4,
        layer_types=["full_attention", "full_attention"], num_kv_shared_layers=0,
        pad_token_id=0, bos_token_id=3, eos_token_id=2,
    )
    model = transformers.Gemma4ForCausalLM(text_config)
    model.save_pretrained(model_dir)
    for split, count in (("train", 3), ("val", 2)):
        rows = [{"messages": [{"role": "user", "content": f"prompt item{index}"},
                               {"role": "assistant", "content": f"result item{index}"}],
                 # Real archive rows carry heterogeneous renderer properties;
                 # these must never participate in Arrow schema inference.
                 "metadata": {"nested": {"flag": [True, "true", {"values": [1, "two"]}][index]}},
                 "genui_json": {"root": "r", "state": {"value": [False, "false", [1, {"enabled": True}]][index]},
                                "elements": {"r": {"props": {"value": [True, "true", {"items": []}][index]}}}}}
                for index in range(count)]
        (data_dir / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = {
        "run": {"id": "cold", "dataset_dir": str(data_dir), "output_dir": str(tmp_path / "cold")},
        "model": {"family": "gemma", "model_id": "local-random-gemma-4", "model_source": str(model_dir),
                  "tokenizer_source": str(model_dir), "tokenizer_loader": "pretrained_tokenizer_fast",
                  "model_loader": "auto_causal_lm", "dtype": "float32", "device_map": "none",
                  "attn_implementation": "eager", "max_context_tokens": 64},
        "training": {"method": "lora_sft", "trainer_backend": "hf", "allow_cpu": True, "epochs": 1,
                     "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
                     "max_seq_length": 64, "report_to": "none", "token_cache": True,
                     "token_cache_dir": str(tmp_path / "shared-tokens"), "cache_progress_seconds": 1,
                     "dataloader_num_workers": 0, "learning_rate": 2e-5, "logging_steps": 1,
                     "preflight_token_check_rows": 0},
        "lora": {"r": 2, "alpha": 2, "dropout": 0.0,
                 "target_modules": r"model\.layers\.\d+\.self_attn\.(q|v)_proj"},
        "preflight": {"rows": 1, "greedy_probe_rows": 1, "greedy_probe_new_tokens": 2,
                      "min_greedy_tokens": 1, "logit_probe_tokens": 2},
        "golden_eval": {"enabled": False},
    }
    try:
        yield config
    finally:
        torch.set_num_threads(previous_threads)


def test_train_sft_cold_warm_cache_skips_full_corpus_work_but_runs_live_gates(local_fixture, tmp_path, monkeypatch):
    import datasets

    calls = {"format": [], "tensors": [], "numeric": [], "greedy": []}
    original_formatter = ModelAdapter.format_example
    def observed_formatter(self, example, tokenizer=None, include_assistant=True):
        calls["format"].append(include_assistant)
        return original_formatter(self, example, tokenizer=tokenizer, include_assistant=include_assistant)
    monkeypatch.setattr(ModelAdapter, "format_example", observed_formatter)
    for name, key in (("_validate_tokenized_sft_dataset", "tensors"), ("_run_forward_numeric_gate", "numeric"),
                      ("_run_deterministic_greedy_gate", "greedy")):
        original = getattr(sft, name)
        def observe(*args, _original=original, _key=key, **kwargs):
            calls[_key].append(kwargs)
            return _original(*args, **kwargs)
        monkeypatch.setattr(sft, name, observe)
    def forbidden(*args, **kwargs):
        pytest.fail("Cached SFT must not load/format/tokenize the whole dataset through the old path")
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    monkeypatch.setattr(sft, "_materialize_sft_text_dataset", forbidden)
    monkeypatch.setattr(sft, "_tokenize_sft_text_dataset", forbidden)

    cold = sft.train_sft(local_fixture, preflight_only=True)
    assert cold["passed"] is True and cold["training_executed"] is False
    assert cold["token_cache"]["status"] == "miss"
    assert len(calls["format"]) == 2 * (3 + 2) + 1  # full/prompt per row, one greedy prefix
    assert len(calls["tensors"]) == 1 and len(calls["numeric"]) == len(calls["greedy"]) == 2
    assert set(calls["tensors"][0]["dataset"]["train"].column_names) == {"input_ids", "attention_mask", "labels"}
    for values in calls.values():
        values.clear()

    warm_config = deepcopy(local_fixture)
    warm_config["run"].update(id="warm", output_dir=str(tmp_path / "warm"))
    warm_config["training"].update(learning_rate=6e-5, per_device_train_batch_size=2,
                                   gradient_accumulation_steps=2, logging_steps=7)
    warm_config["lora"].update(r=4, alpha=4)
    # Runtime topology metadata is deliberately different; this CPU test does
    # not claim to create an eight-rank distributed training process group.
    warm_config["runtime"] = {"world_size": 8, "cuda_visible_devices": "0,1,2,3,4,5,6,7"}
    warm = sft.train_sft(warm_config, preflight_only=True)
    assert warm["passed"] is True and warm["training_executed"] is False
    assert warm["token_cache"]["status"] == "hit"
    assert warm["token_cache"]["key"] == cold["token_cache"]["key"]
    assert warm["token_cache"]["model_preflight_cached"] is False
    assert calls["format"] == [False]  # only the single bounded live greedy probe
    assert len(calls["tensors"]) == 1 and len(calls["numeric"]) == len(calls["greedy"]) == 2
    assert calls["tensors"][0]["max_rows"] == 0  # all cached tensors still validated
    assert all(item["max_rows"] == 1 for item in calls["numeric"] + calls["greedy"])
    assert warm["numeric_preflight"]["passed"] is True
    assert warm["numeric_preflight"]["zero_adapter_initialization"]["verified_zero_delta"] is True


def test_train_sft_without_token_cache_accepts_heterogeneous_metadata_and_runs_live_gates(local_fixture, monkeypatch):
    import datasets

    def forbidden(*args, **kwargs):
        pytest.fail("SFT must not infer a JSON/Arrow schema from heterogeneous archive metadata")
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    calls = {"numeric": [], "greedy": [], "tensors": []}
    for name, key in (("_run_forward_numeric_gate", "numeric"), ("_run_deterministic_greedy_gate", "greedy"),
                      ("_validate_tokenized_sft_dataset", "tensors")):
        original = getattr(sft, name)
        def observe(*args, _original=original, _key=key, **kwargs):
            calls[_key].append(kwargs)
            return _original(*args, **kwargs)
        monkeypatch.setattr(sft, name, observe)
    config = deepcopy(local_fixture)
    config["training"]["token_cache"] = False
    result = sft.train_sft(config, preflight_only=True)
    assert result["passed"] is True and result["training_executed"] is False
    assert result["token_cache"] == {"enabled": False, "status": "disabled"}
    assert result["numeric_preflight"]["passed"] is True
    assert result["numeric_preflight"]["zero_adapter_initialization"]["verified_zero_delta"] is True
    assert len(calls["numeric"]) == len(calls["greedy"]) == 2
    assert len(calls["tensors"]) == 1
    assert len(calls["tensors"][0]["dataset"]["train"]) == 3
    assert len(calls["tensors"][0]["dataset"]["validation"]) == 2
    assert set(calls["tensors"][0]["dataset"]["train"].column_names) == {"input_ids", "attention_mask", "labels"}
    assert not Path(config["training"]["token_cache_dir"]).exists()


@pytest.mark.parametrize("cache_mode", ["warm", "disabled"])
def test_random_cpu_one_optimizer_step_evaluates_and_saves_with_both_cache_modes(local_fixture, tmp_path, monkeypatch, cache_mode):
    """One synthetic update, not production-model training or a Golden score."""
    import datasets
    import torch
    from safetensors.torch import load_file
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    def forbidden(*args, **kwargs):
        pytest.fail("Raw heterogeneous metadata must not reach Arrow JSON inference")
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    config = deepcopy(local_fixture)
    if cache_mode == "warm":
        cold = sft.train_sft(config, preflight_only=True)
        assert cold["passed"] is True and cold["token_cache"]["status"] == "miss"
    else:
        config["training"]["token_cache"] = False
    run_id = f"synthetic-step-{cache_mode}"
    output = tmp_path / run_id
    tensorboard_root = tmp_path / "tensorboard"
    monkeypatch.setenv("A2UI_TENSORBOARD_ROOT", str(tensorboard_root))
    config["run"].update(id=run_id, output_dir=str(output))
    config["training"].update(
        max_steps=1, eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
        logging_steps=1, optim="adamw_torch", warmup_steps=0, warmup_ratio=0.0,
        report_to="tensorboard", tensorboard_root=str(tensorboard_root), tensorboard_subdir="training",
        logging_dir=str(tensorboard_root / run_id / "training"),
    )
    result = sft.train_sft(config)
    expected_status = "hit" if cache_mode == "warm" else "disabled"
    assert result["token_cache"]["status"] == expected_status
    if cache_mode == "warm":
        assert result["token_cache"]["key"] == cold["token_cache"]["key"]
    assert result["checkpoint_step"] == 1
    assert result["numeric_preflight"]["passed"] is True
    assert result["numeric_preflight"]["zero_adapter_initialization"]["verified_zero_delta"] is True

    checkpoint = output / "checkpoint-1"
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == 1
    assert any("eval_loss" in item for item in state["log_history"])
    assert (checkpoint / "optimizer.pt").is_file() and (checkpoint / "scheduler.pt").is_file()
    saved = json.loads((output / "training_metadata.json").read_text(encoding="utf-8"))
    checkpoint_saved = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
    assert saved["checkpoint_step"] == checkpoint_saved["checkpoint_step"] == 1
    assert saved["token_cache"]["status"] == checkpoint_saved["token_cache"]["status"] == expected_status
    final = Path(result["final_adapter"])
    assert (final / "adapter_config.json").is_file() and (final / "tokenizer_config.json").is_file()
    weights = load_file(str(final / "adapter_model.safetensors"))
    lora_b = [value for name, value in weights.items() if "lora_B" in name]
    # Fresh B matrices start at zero; nonzero final values prove backward and
    # an actual optimizer update, beyond preflight/model construction alone.
    assert lora_b and all(torch.isfinite(value).all() for value in weights.values())
    assert any(torch.count_nonzero(value).item() > 0 for value in lora_b)
    events = EventAccumulator(str(tensorboard_root / run_id / "training"))
    events.Reload()
    tags = events.Tags()["scalars"]
    assert "train/loss" in tags and "eval/loss" in tags
    assert any(event.step == 1 for event in events.Scalars("eval/loss"))


@pytest.mark.parametrize("protected_name", ["model", "tokenizer", "data", "output"])
@pytest.mark.parametrize("relation", ["same", "child", "parent"])
def test_sft_token_cache_rejects_overlap_with_protected_directories(tmp_path, protected_name, relation):
    protected = {name: tmp_path / name / "protected" for name in ("model", "tokenizer", "data", "output")}
    for directory in protected.values():
        directory.mkdir(parents=True)
    adapter = GemmaAdapter("fixture", {"model_source": str(protected["model"]), "tokenizer_source": str(protected["tokenizer"])})
    target = protected[protected_name]
    if relation == "child":
        target /= "cache"
    elif relation == "parent":
        target = target.parent
    with pytest.raises(ValueError, match="must not overlap"):
        sft._sft_token_cache_dir({"token_cache_dir": str(target)}, protected["data"], protected["output"], adapter)


def test_sft_token_cache_accepts_shared_sibling_but_not_a_file(tmp_path):
    model, data, output = tmp_path / "model", tmp_path / "data", tmp_path / "output"
    for directory in (model, data, output):
        directory.mkdir()
    adapter = GemmaAdapter("fixture", {"model_source": str(model), "tokenizer_source": str(model)})
    shared = tmp_path / "cache" / "tokens"
    assert sft._sft_token_cache_dir({"token_cache_dir": str(shared)}, data, output, adapter) == shared.resolve()
    file_path = tmp_path / "cache-file"
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="points to a file"):
        sft._sft_token_cache_dir({"token_cache_dir": str(file_path)}, data, output, adapter)


@pytest.mark.parametrize("previous", [None, "previous-tensorboard-directory", ""])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_tensorboard_environment_restores_previous_or_unset_on_success_and_exception(tmp_path, monkeypatch, previous, raise_inside):
    name = "TENSORBOARD_LOGGING_DIR"
    if previous is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, previous)
    desired = tmp_path / "scoped-tensorboard"
    if raise_inside:
        with pytest.raises(RuntimeError, match="synthetic constructor failure"):
            with sft._training_tensorboard_environment(desired):
                assert os.environ[name] == str(desired)
                raise RuntimeError("synthetic constructor failure")
    else:
        with sft._training_tensorboard_environment(desired):
            assert os.environ[name] == str(desired)
    assert os.environ.get(name) == previous
    assert (name in os.environ) is (previous is not None)
    assert not desired.exists()


@pytest.mark.parametrize("previous", [None, "leave-this-environment-alone", ""])
def test_tensorboard_environment_none_does_not_modify_the_environment(monkeypatch, previous):
    name = "TENSORBOARD_LOGGING_DIR"
    if previous is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, previous)
    with sft._training_tensorboard_environment(None):
        assert os.environ.get(name) == previous
        assert (name in os.environ) is (previous is not None)
    assert os.environ.get(name) == previous


def test_modern_tensorboard_callback_captures_scoped_directory_without_creating_writer(tmp_path, monkeypatch):
    pytest.importorskip("transformers", minversion="5.16.1")
    pytest.importorskip("tensorboard")
    from transformers.integrations import TensorBoardCallback

    name = "TENSORBOARD_LOGGING_DIR"
    previous = str(tmp_path / "previous")
    desired = tmp_path / "expected" / "run" / "training"
    monkeypatch.setenv(name, previous)
    with sft._training_tensorboard_environment(desired):
        callback = TensorBoardCallback()
        assert callback.logging_dir == str(desired)
        assert callback.tb_writer is None
    assert os.environ[name] == previous
    assert callback.logging_dir == str(desired)
    assert callback.tb_writer is None
    assert not desired.exists()
