"""Storage/compute policy regressions for generic full SFT and full QAT."""
from __future__ import annotations

import copy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.qat.full_model_contract import WORKFLOW, full_parameter_autocast
from ir_training.train.precision import (
    bind_full_finetune_precision,
    full_finetune_load_config,
    full_finetune_precision_policy,
    training_precision_flags,
)


def config(method="full_finetune_sft", dtype="bfloat16", mixed="auto"):
    return {"model": {"dtype": dtype, "model_source": "local-checkpoint"},
            "training": {"method": method, "mixed_precision": mixed}}


@pytest.mark.parametrize("method", ["full_finetune_sft", "full_finetune_qat",
                                    " FULL_FINETUNE_SFT ", " FULL_FINETUNE_QAT "])
def test_supported_full_methods_preserve_fp32_storage_and_bf16_compute(method):
    cfg = config(method=method)
    before = copy.deepcopy(cfg)
    policy = full_finetune_precision_policy(cfg, resolved_dtype="bfloat16")
    assert policy == {"policy": "fp32_parameters_v1", "parameter_dtype": "float32",
                      "compute_dtype": "bfloat16"}
    load_cfg = full_finetune_load_config(cfg["model"], policy)
    assert load_cfg["dtype"] == "float32"
    assert load_cfg["model_source"] == "local-checkpoint"
    assert cfg == before  # Export lineage retains the exact requested model config.


@pytest.mark.parametrize("dtype,mixed,compute", [
    ("bfloat16", "auto", "bfloat16"), ("float16", "auto", "float16"),
    ("float32", "auto", "float32"), ("float32", "bf16", "bfloat16"),
    ("float32", "fp16", "float16"), ("bfloat16", "no", "float32"),
    ("bfloat16", False, "float32"), ("float16", "off", "float32"),
])
def test_compute_policy_matches_trainer_flags(dtype, mixed, compute):
    cfg = config(dtype=dtype, mixed=mixed)
    policy = full_finetune_precision_policy(cfg, resolved_dtype=dtype)
    assert policy["compute_dtype"] == compute
    assert training_precision_flags(dtype, cfg["training"]) == {
        "bf16": compute == "bfloat16", "fp16": compute == "float16",
    }


def test_cpu_auto_fallback_and_explicit_cpu_bf16_are_distinct():
    cfg = config()
    assert full_finetune_precision_policy(cfg, resolved_dtype="float32")["compute_dtype"] == "float32"
    cfg["training"]["mixed_precision"] = "bf16"
    assert full_finetune_precision_policy(cfg, resolved_dtype="float32")["compute_dtype"] == "bfloat16"


@pytest.mark.parametrize("cuda,bf16,expected", [(False, False, "float32"),
                                              (True, False, "float16"),
                                              (True, True, "bfloat16")])
def test_standalone_auto_precision_uses_same_hardware_fallback_as_training(monkeypatch, cuda, bf16, expected):
    from ir_training.train import precision, sft

    monkeypatch.setattr(precision, "_cuda_available", lambda: cuda)
    monkeypatch.setattr(precision, "_cuda_bf16_supported", lambda: bf16)
    monkeypatch.setattr(sft, "_cuda_available", lambda: cuda)
    monkeypatch.setattr(sft, "_cuda_bf16_supported", lambda: bf16)
    standalone = full_finetune_precision_policy(config())
    training = full_finetune_precision_policy(config(), resolved_dtype=sft._resolve_training_dtype("bfloat16"))
    assert standalone == training
    assert standalone["compute_dtype"] == expected


@pytest.mark.parametrize("method", ["lora_sft", "qat_lora_sft", "sft_lora", "qat_lora_grpo"])
def test_lora_paths_keep_their_existing_loading_and_compute_policy(method):
    cfg = config(method=method)
    assert full_finetune_precision_policy(cfg) is None
    assert full_finetune_load_config(cfg["model"], None) == cfg["model"]


def test_official_full_qat_policy_is_unchanged():
    cfg = config(method="full_finetune_qat", dtype="float32", mixed="bf16")
    cfg["run"] = {"purpose": WORKFLOW}
    assert full_finetune_precision_policy(cfg) is None


def test_binding_refuses_weights_that_were_already_rounded_during_loading():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="loaded directly"):
        bind_full_finetune_precision(model, full_finetune_precision_policy(config()))
    assert model.weight.dtype == torch.bfloat16  # No silent late promotion.


@pytest.mark.parametrize("mixed,expected", [("bf16", "bfloat16"), ("no", "float32")])
def test_shared_direct_context_executes_cpu_compute_policy(mixed, expected):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2, dtype=torch.float32)
    bind_full_finetune_precision(model, full_finetune_precision_policy(config(mixed=mixed)))
    with torch.no_grad(), full_parameter_autocast(model):
        result = model(torch.ones(1, 2))
    assert result.dtype == getattr(torch, expected)
    assert model.weight.dtype == torch.float32
    assert not torch.is_autocast_enabled("cpu")  # Context does not leak.


@pytest.mark.parametrize("compute", ["bfloat16", "float16"])
def test_generic_cuda_context_selects_requested_dtype_without_allocating_gpu(monkeypatch, compute):
    torch = pytest.importorskip("torch")
    calls = []
    monkeypatch.setattr(torch, "autocast", lambda device, **kw: calls.append((device, kw)) or nullcontext())
    model = SimpleNamespace(
        _a2ui_full_finetune_compute_dtype=compute,
        parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda"))]),
    )
    with full_parameter_autocast(model):
        pass
    assert calls == [("cuda", {"dtype": getattr(torch, compute)})]


def test_official_context_still_uses_its_own_bf16_flag(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    monkeypatch.setattr(torch, "autocast", lambda device, **kw: calls.append((device, kw)) or nullcontext())
    model = SimpleNamespace(_a2ui_full_parameter_amp=True,
                            parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda"))]))
    with full_parameter_autocast(model):
        pass
    assert calls == [("cuda", {"dtype": torch.bfloat16})]


def test_cpu_fp16_does_not_silently_use_a_different_trainer_policy():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    bind_full_finetune_precision(model, full_finetune_precision_policy(config(mixed="fp16")))
    with pytest.raises(ValueError, match="requires CUDA"):
        full_parameter_autocast(model)
