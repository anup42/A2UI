"""Real Gemma 4 text coverage for the parameter/persistent-buffer boundary."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.qat import full_model_contract as contract
from ir_training.train.full_parameters import enable_full_parameter_training


def _tiny_e2b_text_model():
    transformers = pytest.importorskip("transformers", minversion="5.16.1")

    # Keep the audited E2B layer/shared-KV topology while making every width and
    # vocabulary allocation tiny. Both attention kinds are needed to exercise
    # all six derived, non-persistent embedding/rotary buffers.
    layer_types = ["sliding_attention"] * 35
    for layer in (5, 11, 17, 23, 29, 34):
        layer_types[layer] = "full_attention"
    config = transformers.Gemma4TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=35,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        max_position_embeddings=64,
        sliding_window=16,
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=4,
        layer_types=layer_types,
        num_kv_shared_layers=20,
        attention_k_eq_v=False,
        use_bidirectional_attention=None,
        use_double_wide_mlp=False,
        tie_word_embeddings=False,
        use_cache=False,
    )
    return transformers.Gemma4ForCausalLM(config)


def _persistent_buffers(model):
    result = {}
    for prefix, module in model.named_modules(remove_duplicate=False):
        for local_name, tensor in module._buffers.items():
            if local_name not in module._non_persistent_buffers_set:
                result[f"{prefix}.{local_name}" if prefix else local_name] = tensor
    return result


def test_real_35_layer_gemma4_full_parameter_and_buffer_contract(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    transformers = pytest.importorskip("transformers", minversion="5.16.1")
    seed = _tiny_e2b_text_model().to(dtype=torch.bfloat16)
    bf16_state_as_fp32 = {
        name: tensor.detach().to(dtype=torch.float32).clone()
        for name, tensor in seed.state_dict().items()
    }
    seed_dir = tmp_path / "tiny-bf16-seed"
    seed.save_pretrained(seed_dir, safe_serialization=True)
    model = transformers.Gemma4ForCausalLM.from_pretrained(
        seed_dir, dtype=torch.float32
    )
    assert all(tensor.dtype == torch.float32 for tensor in model.state_dict().values())
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in bf16_state_as_fp32.items()
    )
    assert all(
        tensor.dtype == torch.float32 and not tensor.requires_grad
        for tensor in _persistent_buffers(model).values()
    )

    state_shapes = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    named_parameters = dict(model.named_parameters(remove_duplicate=False))
    all_buffers = dict(model.named_buffers(remove_duplicate=False))
    persistent = _persistent_buffers(model)
    nonpersistent_names = set(all_buffers) - set(model.state_dict())

    assert len(state_shapes) == 541
    assert len(named_parameters) == contract.EXPECTED_PARAMETER_TENSOR_COUNT == 506
    assert set(persistent) == contract.EXPECTED_PERSISTENT_BUFFER_NAMES
    assert len(persistent) == 35
    assert len(all_buffers) == 41
    assert len(nonpersistent_names) == 6
    assert all(name not in state_shapes for name in nonpersistent_names)

    original_buffers = {
        name: (tensor, tensor.detach().clone()) for name, tensor in persistent.items()
    }
    monkeypatch.setattr(contract, "seed_shapes", lambda _config: state_shapes)
    # Production preflight checks this before enabling trainability.
    report = contract.verify_full_model_inventory(model, {})
    scope = enable_full_parameter_training(model)
    assert scope["named_parameter_count"] == 506
    assert scope["unique_parameter_count"] == 506
    assert set(dict(model.named_parameters(remove_duplicate=False))) == set(named_parameters)
    for name, (identity, value) in original_buffers.items():
        current = _persistent_buffers(model)[name]
        assert current is identity
        assert current.requires_grad is False
        assert current.grad is None
        assert torch.equal(current, value)

    assert contract.verify_full_model_inventory(model, {}) == report
    assert report["verified"] is True
    assert report["state_tensor_count"] == 541
    assert report["named_parameter_count"] == 506
    assert report["persistent_buffer_count"] == 35

    saved = tmp_path / "tiny-gemma4-text"
    model.save_pretrained(saved, safe_serialization=True)
    restored = transformers.Gemma4ForCausalLM.from_pretrained(
        saved, dtype=torch.float32
    )
    restored_state = restored.state_dict()
    assert len(restored_state) == 541
    assert set(restored_state) == set(model.state_dict())
    assert all(tensor.dtype == torch.float32 for tensor in restored_state.values())
    assert all(
        torch.equal(restored_state[name], tensor)
        for name, tensor in model.state_dict().items()
    )
    assert set(_persistent_buffers(restored)) == contract.EXPECTED_PERSISTENT_BUFFER_NAMES
    assert len(dict(restored.named_buffers(remove_duplicate=False))) == 41
    assert all(name not in restored_state for name in nonpersistent_names)
    restored_report = contract.verify_full_model_inventory(restored, {})
    assert restored_report["persistent_buffer_value_sha256"] == report[
        "persistent_buffer_value_sha256"
    ]

    # A layer scalar with identical shape/value is still invalid once promoted
    # from persistent buffer to trainable Parameter registration.
    layer = restored.model.layers[0]
    scalar = layer.layer_scalar.detach().clone()
    del layer._buffers["layer_scalar"]
    layer.register_parameter("layer_scalar", torch.nn.Parameter(scalar))
    with pytest.raises(ValueError, match="506 named parameter"):
        contract.verify_full_model_inventory(restored, {})
