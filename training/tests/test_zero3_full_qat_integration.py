"""ZeRO-3 wiring and persisted-receipt isolation; no CUDA or DeepSpeed imports."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from ir_training.common.config import load_yaml
from ir_training.qat import full_model_contract as contract
from ir_training.train import sft, zero3_checkpoint
from test_zero3_preflight import _probe3


def _base_config():
    root = Path(__file__).resolve().parents[1]
    return load_yaml(root / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml")


def test_zero3_config_only_changes_explicit_stage_and_preserves_source():
    source = _base_config()
    before = copy.deepcopy(source)
    zero2 = contract.configure_full_qat(source, distributed_backend="sharded")
    zero3 = contract.configure_full_qat(source, distributed_backend="sharded", zero_stage=3)
    assert source == before
    contract.validate_full_qat_config(zero3)
    assert zero3["training"].pop("zero_stage") == 3
    assert zero3 == zero2
    assert "zero_stage" not in zero2["training"]
    assert contract.configure_full_qat(source) == contract.configure_full_qat(source, zero_stage=2)


@pytest.mark.parametrize("stage", [3, True, "2", 2.0])
def test_ddp_cannot_accept_stage3_or_ambiguous_selector(stage):
    with pytest.raises(ValueError):
        contract.configure_full_qat(_base_config(), zero_stage=stage)


def test_zero3_receipt_round_trip_is_stage_bound(tmp_path, monkeypatch):
    config = contract.configure_full_qat(_base_config(), distributed_backend="sharded", zero_stage=3)
    config["run"]["output_dir"] = str(tmp_path / "training")
    config["training"].update(gradient_accumulation_steps=4, learning_rate=1e-5)
    path = tmp_path / "config.yaml"
    path.write_text("file-bound-fixture", encoding="utf-8")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    report = contract.write_optimizer_preflight(config, path, _probe3())
    assert contract.require_optimizer_preflight(config, path) == report
    # Same config-file hash cannot make a ZeRO-3 receipt certify ZeRO-2.
    config["training"].pop("zero_stage")
    with pytest.raises(ValueError, match="matching disposable optimizer preflight"):
        contract.require_optimizer_preflight(config, path)
    with pytest.raises(ValueError):
        contract.write_optimizer_preflight(config, path, _probe3())


@pytest.mark.parametrize("backend,stage", [("ddp", 2), ("sharded", 2), ("sharded", 3)])
def test_checked_trainer_saves_collectively_only_for_zero3(monkeypatch, backend, stage):
    calls = []

    class Base:
        def save_model(self, output_dir, _internal_call=False):
            calls.append(("parent", output_dir, _internal_call))

    monkeypatch.setenv("WORLD_SIZE", "1")
    checked = sft._build_checked_causal_lm_trainer(
        Base, {"distributed_backend": backend, "zero_stage": stage}
    )
    trainer = object.__new__(checked)
    trainer.args = SimpleNamespace(output_dir="default-output")
    trainer.processing_class = object()
    monkeypatch.setattr(zero3_checkpoint, "save_zero3_checkpoint", lambda *a, **k: calls.append((a, k)))
    trainer.save_model("checkpoint", _internal_call=True)
    if stage == 3:
        assert calls == [((trainer, "checkpoint"), {"tokenizer": trainer.processing_class})]
    else:
        assert calls == [("parent", "checkpoint", True)]


def test_zero3_save_failure_is_not_replaced_with_shard_only_fallback(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    checked = sft._build_checked_causal_lm_trainer(
        object, {"distributed_backend": "sharded", "zero_stage": 3}
    )
    trainer = object.__new__(checked)
    trainer.args = SimpleNamespace(output_dir="default-output")

    def fail(*args, **kwargs):
        raise ValueError("consolidation failed")

    monkeypatch.setattr(zero3_checkpoint, "save_zero3_checkpoint", fail)
    with pytest.raises(ValueError, match="consolidation failed"):
        trainer.save_model()


@pytest.mark.parametrize("stage", [2, 3])
def test_golden_callback_wiring_preserves_separate_eval_budget(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(sft, "build_golden_set_eval_callback", lambda **kwargs: kwargs)
    trainer = object() if stage == 3 else None
    result = sft._build_optional_golden_callback(
        golden_eval_cfg={"enabled": True, "max_input_tokens": 5120, "max_new_tokens": 2048},
        base=tmp_path, output_dir=tmp_path, adapter=object(), tokenizer=object(),
        model_cfg={"max_context_tokens": 32768}, training_cfg={"max_seq_length": 4096},
        zero3_trainer=trainer,
    )
    assert result["max_input_tokens"] == 5120
    assert result["max_new_tokens"] == 2048
    if stage == 3:
        assert result["zero3_trainer"] is trainer
    else:
        assert "zero3_trainer" not in result
