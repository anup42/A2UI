"""Real tiny HF/PEFT models, random weights only; no downloads or GPU required."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.common.config import load_yaml
from ir_training.models.gemma import GemmaAdapter
from ir_training.train.lora_config import build_lora_config, resolve_lora_config_targets


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("shared_kv", [False, True])
def test_real_gemma4_decoder_lora_attachment_forward_backward_and_reload(tmp_path, multimodal, shared_kv):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    peft = pytest.importorskip("peft", minversion="0.19.0")
    # Small but real E2B-style text decoder, including per-layer embeddings.
    text_config = transformers.Gemma4TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, global_head_dim=16, max_position_embeddings=64,
        vocab_size_per_layer_input=32, hidden_size_per_layer_input=4,
        layer_types=["full_attention", "full_attention"],
        num_kv_shared_layers=1 if shared_kv else 0,
    )
    if multimodal:
        config = transformers.Gemma4Config(text_config=text_config, vision_config=None, audio_config=None)
        model_class = transformers.Gemma4ForConditionalGeneration
        prefix = "model.language_model.layers."
    else:
        config, model_class, prefix = text_config, transformers.Gemma4ForCausalLM, "model.layers."
    torch.manual_seed(42)
    model = model_class(config)
    if not multimodal:
        from types import SimpleNamespace
        with pytest.raises(ValueError, match="no supported nn.Linear"):
            resolve_lora_config_targets(
                SimpleNamespace(target_modules=r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"),
                model,
            )
    recipe = load_yaml(ROOT / "configs/models/gemma4_e2b_a2ui_express_review_sft.yaml")
    lora = build_lora_config(GemmaAdapter(recipe["model"]["model_id"]), recipe["lora"])
    projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    expected = {
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.startswith(prefix)
        and name.rsplit(".", 1)[-1] in projections
    }
    assert expected and all(name.startswith(prefix) for name in expected)
    assert resolve_lora_config_targets(lora, model) == expected
    adapted = peft.get_peft_model(model, lora)
    realized = {
        name.removeprefix("base_model.model.")
        for name, module in adapted.named_modules() if hasattr(module, "lora_A")
    }
    assert realized == expected
    assert all("lora_" in name for name, parameter in adapted.named_parameters() if parameter.requires_grad)
    tokens = torch.tensor([[2, 4, 5, 6]])
    loss = adapted(input_ids=tokens, labels=tokens, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in adapted.parameters() if p.requires_grad]
    # DDP runs with find_unused_parameters=False, so no trainable adapter may
    # be silently disconnected from the loss.
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    # Materialized exact names must survive a PEFT checkpoint round trip.
    adapted.save_pretrained(tmp_path, safe_serialization=True, save_embedding_layers=False)
    reloaded = peft.PeftModel.from_pretrained(model_class(config), tmp_path, is_trainable=True)
    assert set(reloaded.peft_config["default"].target_modules) == expected
