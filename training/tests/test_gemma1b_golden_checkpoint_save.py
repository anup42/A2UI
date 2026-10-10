"""Serialization-only cache handling for the opt-in Gemma 3 1B best snapshot."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from ir_training.train import callbacks, sft


@pytest.mark.parametrize("save_fails", [False, True])
@pytest.mark.parametrize("original_cache", ["hybrid", None])
def test_golden_best_save_restores_exact_cache_without_enabling_it(
    tmp_path, original_cache, save_fails,
):
    config = SimpleNamespace(cache_implementation=original_cache, use_cache=False)
    observed = []

    def save_model(path):
        observed.append((config.cache_implementation, config.use_cache))
        if save_fails:
            raise ValueError("save failed")

    # Match the normal multi-GPU callback's wrapped model path too.
    model = SimpleNamespace(module=SimpleNamespace(
        generation_config=config, save_pretrained=save_model,
    ))
    saved_tokenizers = []
    kwargs = {
        "model": model,
        "tokenizer": SimpleNamespace(save_pretrained=saved_tokenizers.append),
        "checkpoint_dir": tmp_path / "best",
        "event_dir": tmp_path / "event",
        "best_info": {"metric_value": 1.0},
        "clear_generation_cache_on_save": True,
    }
    if save_fails:
        with pytest.raises(ValueError, match="save failed"):
            callbacks._save_best_golden_checkpoint(**kwargs)
    else:
        callbacks._save_best_golden_checkpoint(**kwargs)
    assert observed == [(None, False)]
    assert config.cache_implementation is original_cache
    assert config.use_cache is False
    assert bool(saved_tokenizers) is not save_fails


def test_golden_best_save_default_keeps_legacy_cache_behavior(tmp_path):
    config = SimpleNamespace(cache_implementation="hybrid", use_cache=False)
    observed = []
    model = SimpleNamespace(
        generation_config=config,
        save_pretrained=lambda _: observed.append(
            (config.cache_implementation, config.use_cache)
        ),
    )
    callbacks._save_best_golden_checkpoint(
        model=model, tokenizer=SimpleNamespace(save_pretrained=lambda _: None),
        checkpoint_dir=tmp_path / "best", event_dir=tmp_path / "event", best_info={},
    )
    assert observed == [("hybrid", False)]
    assert config.cache_implementation == "hybrid"


def test_golden_best_save_does_not_invent_missing_cache_attribute(tmp_path):
    config = SimpleNamespace(use_cache=False)
    callbacks._save_best_golden_checkpoint(
        model=SimpleNamespace(generation_config=config, save_pretrained=lambda _: None),
        tokenizer=SimpleNamespace(save_pretrained=lambda _: None),
        checkpoint_dir=tmp_path / "best", event_dir=tmp_path / "event", best_info={},
        clear_generation_cache_on_save=True,
    )
    assert vars(config) == {"use_cache": False}


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_golden_callback_forwards_explicit_save_policy(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr(sft, "build_golden_set_eval_callback", lambda **kwargs: kwargs)
    kwargs = {"clear_generation_cache_on_save": True} if enabled else {}
    result = sft._build_optional_golden_callback(
        golden_eval_cfg={"enabled": True},
        base=tmp_path, output_dir=tmp_path, adapter=object(), tokenizer=object(),
        model_cfg={"max_context_tokens": 32768}, training_cfg={"max_seq_length": 4096},
        **kwargs,
    )
    assert result["clear_generation_cache_on_save"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_golden_callback_forwards_save_policy_only_on_normal_best_path(
    tmp_path, monkeypatch, enabled,
):
    split = tmp_path / "golden.jsonl"
    split.write_text('{"id":"one"}\n', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))
    monkeypatch.setattr(callbacks, "_distributed_context", lambda: (0, 1))
    saved = []
    monkeypatch.setattr(callbacks, "_save_best_golden_checkpoint", lambda **kw: saved.append(kw))
    callback = callbacks.build_golden_set_eval_callback(
        enabled=True, split_path=split, output_dir=tmp_path / "eval",
        adapter=object(), tokenizer=object(), clear_generation_cache_on_save=enabled,
    )
    callback._record_best_if_improved(
        model=object(), state=SimpleNamespace(global_step=1, epoch=0.5),
        event_label="step_1", event_dir=tmp_path / "event",
        aggregate={"overall_score": 1.0},
    )
    assert len(saved) == 1
    assert saved[0]["clear_generation_cache_on_save"] is enabled
