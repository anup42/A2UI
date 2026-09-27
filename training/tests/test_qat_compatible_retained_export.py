from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import build_gemma4_retained_scale_litertlm as exporter


@pytest.mark.parametrize("bits", [2, 4])
def test_bf16_rounding_counterexample_matches_qat_not_ordinary_merge(bits):
    torch = pytest.importorskip("torch")
    from ir_training.qat.fake_quant import _effective_lora_weight

    base = torch.ones((2, 4), dtype=torch.bfloat16)
    a = np.full((1, 4), -0.498, dtype=np.float32)
    b = np.ones((2, 1), dtype=np.float32)
    scales = np.ones((2, 1), dtype=np.float32)
    delta = torch.from_numpy(b) @ torch.from_numpy(a)
    qat = _effective_lora_weight(
        SimpleNamespace(base_layer=SimpleNamespace(weight=base), get_delta_weight=lambda _: delta),
        ("default",),
    )
    merged = base.clone()
    merged.add_(delta)
    old_raw, _ = exporter._quantize_projection(merged.float().numpy(), scales, bits=bits, working_set_bytes=32)
    raw, telemetry, proof = exporter._qat_compatible_projection(
        base.float().numpy(), a, b, scales, base_dtype="BF16", adapter_dtype="F32",
        scaling=1.0, bits=bits, working_set_bytes=32,
    )
    expected, _ = exporter._quantize_projection(qat.float().numpy(), scales, bits=bits, working_set_bytes=32)
    assert raw == expected != old_raw
    assert telemetry["code_histogram"] == {0: 8}
    assert proof["code_exact"] and proof["dequantized_weight_forward_exact"]
    assert proof["native_inference_parity_verified"] is False


@pytest.mark.parametrize("working_set", [32, 1048576])
@pytest.mark.parametrize("bits", [2, 4])
def test_rank64_chunked_reconstruction_matches_dense_qat(bits, working_set):
    torch = pytest.importorskip("torch")
    from ir_training.qat.fake_quant import _effective_lora_weight

    rng = np.random.default_rng(73)
    base = torch.from_numpy(rng.normal(0, 0.1, (12, 32)).astype(np.float32)).to(torch.bfloat16)
    a = rng.normal(0, 0.02, (64, 32)).astype(np.float32)
    b = rng.normal(0, 0.02, (12, 64)).astype(np.float32)
    scales = rng.uniform(0.07, 0.11, (12, 1)).astype(np.float32)
    delta = (torch.from_numpy(b) @ torch.from_numpy(a)) * 0.5
    qat = _effective_lora_weight(SimpleNamespace(
        base_layer=SimpleNamespace(weight=base), get_delta_weight=lambda _: delta,
    ), ("default",))
    expected, _ = exporter._quantize_projection(qat.float().numpy(), scales, bits=bits, working_set_bytes=working_set)
    actual, _, proof = exporter._qat_compatible_projection(
        base.float().numpy(), a, b, scales, base_dtype="BF16", adapter_dtype="F32",
        scaling=0.5, bits=bits, working_set_bytes=working_set,
    )
    assert actual == expected
    assert proof["dequantized_weight_forward_exact"]


@pytest.mark.parametrize("base_dtype,adapter_dtype", [("F32", "F32"), ("BF16", "BF16")])
def test_reconstruction_does_not_guess_other_dtype_contracts(base_dtype, adapter_dtype):
    pytest.importorskip("torch")
    with pytest.raises(exporter.RetainedScaleExportError, match="BF16 base and F32"):
        exporter._qat_compatible_projection(
            np.ones((1, 4)), np.ones((1, 4)), np.ones((1, 1)), np.ones((1, 1)),
            base_dtype=base_dtype, adapter_dtype=adapter_dtype, scaling=1, bits=2, working_set_bytes=32,
        )


def test_qat_code_gate_detects_corrupted_serialization(monkeypatch):
    pytest.importorskip("torch")
    original = exporter._quantize_projection

    def corrupt(*args, **kwargs):
        raw, report = original(*args, **kwargs)
        return bytes([raw[0] ^ 1]) + raw[1:], report

    monkeypatch.setattr(exporter, "_quantize_projection", corrupt)
    with pytest.raises(exporter.RetainedScaleExportError, match="codes differ"):
        exporter._qat_compatible_projection(
            np.ones((1, 4)), np.full((1, 4), -0.498), np.ones((1, 1)), np.ones((1, 1)),
            base_dtype="BF16", adapter_dtype="F32", scaling=1, bits=2, working_set_bytes=32,
        )


def test_changed_qat_helper_cannot_silently_change_versioned_arithmetic(monkeypatch):
    pytest.importorskip("torch")
    from ir_training.qat import fake_quant

    monkeypatch.setattr(fake_quant, "_effective_lora_weight", lambda module, _: module.base_layer.weight)
    with pytest.raises(exporter.RetainedScaleExportError, match="codes differ"):
        exporter._qat_compatible_projection(
            np.ones((1, 4)), np.full((1, 4), -0.498), np.ones((1, 1)), np.ones((1, 1)),
            base_dtype="BF16", adapter_dtype="F32", scaling=1, bits=2, working_set_bytes=32,
        )


def test_patch_uses_qat_codes_and_preserves_original_merge_verification(monkeypatch):
    torch = pytest.importorskip("torch")
    base = torch.ones((1, 4), dtype=torch.bfloat16)
    a = np.full((1, 4), -0.498, dtype=np.float32)
    b = np.ones((1, 1), dtype=np.float32)
    merged = base.clone()
    merged.add_(torch.from_numpy(b) @ torch.from_numpy(a))

    class Checkpoint:
        def __init__(self, data, dtype):
            self.data = data
            self.entries = {key: {"dtype": dtype} for key in data}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def load_float32(self, key):
            return self.data[key].copy()

    key = "model.layers.0.mlp.down_proj.weight"
    record = {"hf_weight_key": key}
    monkeypatch.setattr(exporter, "EXPECTED_MUTABLE_COUNT", 1)
    monkeypatch.setattr(exporter, "EXPECTED_MUTABLE_BITS", {2: 1})
    monkeypatch.setattr(exporter, "_schema_model", lambda _: object())
    monkeypatch.setattr(exporter, "_official_weight_alias_groups", lambda *_: [{"buffer_index": 0, "aliases": [0]}])
    monkeypatch.setattr(exporter, "_buffer_view", lambda _model, _index, section: (
        np.frombuffer(section, dtype=np.uint8)[4:5], "inline", 4, 1,
    ))
    qparams = SimpleNamespace(
        inventory={key: {"weight_shape": [1, 4], "bits": 2}},
        load_scale=lambda *_a, **_k: torch.ones((1, 1)),
    )
    official = b"HEAD\x55TAIL"
    candidate, report = exporter._quantize_and_patch(
        official,
        checkpoint=Checkpoint({key: merged.float().numpy()}, "BF16"), mappings={key: key},
        mutable_records=[record], qparams=qparams, label="trained_merged_checkpoint", working_set_bytes=32,
        base_checkpoint=Checkpoint({key: base.float().numpy()}, "BF16"), base_mappings={key: key},
        adapter_checkpoint=Checkpoint({"a": a, "b": b}, "F32"), adapter_mappings={key: {"a": "a", "b": "b"}},
        lora_scaling=1.0, qat_compatible_weights=True,
    )
    assert bytes(candidate) == b"HEAD\x00TAIL"
    assert report["verified"]
    assert report["base_lora_parity"]["scope"] == "original_supplied_merged_checkpoint"
    assert report["base_lora_parity"]["numerical_exact_count"] == 1
    assert report["qat_weight_reconstruction"]["changed_code_count_from_merged_checkpoint"] == 4
    assert report["checks"]["qat_weight_code_parity_205"]
    assert report["checks"]["qat_dequantized_weight_forward_parity_205"]


def test_plan_names_distinct_arithmetic_and_does_not_claim_native_parity(monkeypatch):
    monkeypatch.setattr(exporter, "_build_plan", lambda **_: ({"plan_passed": True, "mode": exporter.MODE}, {}))
    plan = exporter.run(
        official_litertlm="base", official_artifact_sha256="hash", checkpoint="merged", adapter_checkpoint="adapter",
        training_config="config", mobile_training_seed_manifest="seed", mobile_qparams_contract="qparams",
        zero_adapter_checkpoint="seed", output_dir="out", qat_compatible_weights=True,
    )
    assert plan["mode"] == "retained_scale_qat_compatible_weights_v1"
    assert plan["weight_arithmetic"] == exporter.QAT_WEIGHT_ARITHMETIC
    assert plan["native_inference_parity_verified"] is False
    assert plan["merged_checkpoint_role"] == "original_provenance_and_merge_verification_only"


def test_cli_forwards_explicit_qat_compatible_option(monkeypatch, capsys):
    received = {}

    def fake_run(**kwargs):
        received.update(kwargs)
        return {"plan_passed": True, "passed": True}

    monkeypatch.setattr(exporter, "run", fake_run)
    flags = ["--official-litertlm", "base", "--official-artifact-sha256", "hash", "--checkpoint", "merged",
             "--adapter-checkpoint", "adapter", "--training-config", "config", "--mobile-training-seed-manifest", "manifest",
             "--mobile-qparams-contract", "qparams", "--zero-adapter-checkpoint", "seed", "--output-dir", "out",
             "--qat-compatible-weights", "--execute"]
    assert exporter.main(flags) == 0
    assert received["qat_compatible_weights"] is True
    assert received["execute"] is True
