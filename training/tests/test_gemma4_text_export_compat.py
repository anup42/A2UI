from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.export import gemma4_text_compat as compat


class Gemma4TextConfig:
    __module__ = "transformers.models.gemma4.configuration_gemma4"
    model_type = "gemma4_text"
    num_hidden_layers = 35
    hidden_size = 1536
    hidden_size_per_layer_input = 256


class Gemma4ForCausalLM:
    __module__ = "transformers.models.gemma4.modeling_gemma4"

    def __init__(self):
        self.config = Gemma4TextConfig()
        token_embedding = object()
        self.model = types.SimpleNamespace(
            embed_tokens=token_embedding,
            embed_tokens_per_layer=object(),
            get_per_layer_inputs=lambda ids, unused: (ids, unused),
        )
        self.get_input_embeddings = lambda: token_embedding
        self.lm_head = lambda value: ("head", value)

    def state_dict(self):
        required = {
            "model.embed_tokens.weight": 1,
            "model.embed_tokens_per_layer.weight": 2,
            "model.per_layer_model_projection.weight": 3,
            "model.per_layer_projection_norm.weight": 4,
            "model.norm.weight": 5,
            "lm_head.weight": 6,
        }
        for layer in range(35):
            prefix = f"model.layers.{layer}."
            for suffix in (
                "layer_scalar",
                "self_attn.q_norm.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "per_layer_input_gate.weight",
                "per_layer_projection.weight",
                "post_per_layer_input_norm.weight",
            ):
                required[prefix + suffix] = layer
        required.update(
            {
                f"model.synthetic.{index}": index
                for index in range(compat.EXPECTED_STATE_KEYS - len(required))
            }
        )
        return required


def _fake_exporter(monkeypatch):
    class Base:
        def __init__(self, model, *args, **kwargs):
            self.model = model

    class PerLayer(Base):
        pass

    exportable_module = types.SimpleNamespace(
        LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder=Base,
        LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder=Base,
        LiteRTExportableModuleForPerLayerEmbedder=PerLayer,
    )
    exportables = types.SimpleNamespace(
        get_prefill_decode_exportables=lambda config, options: (
            "old-prefill",
            "old-decode",
        ),
        get_additional_exportables=lambda config: {"old": True},
    )
    metadata = types.SimpleNamespace(get_metadata_builder=lambda config: "old-metadata")
    gemma_metadata = types.SimpleNamespace(build_llm_metadata=lambda *args: args)
    patches = types.SimpleNamespace(
        _CONTEXT_REGISTRY={"gemma4": contextlib.nullcontext},
        _MODEL_PATCH_REGISTRY={
            "gemma4": lambda model, options: contextlib.nullcontext()
        },
    )
    model_ext = types.ModuleType("litert_torch.generative.export_hf.model_ext")
    model_ext.exportables, model_ext.metadata_builder, model_ext.patches = (
        exportables,
        metadata,
        patches,
    )
    gemma = types.ModuleType("litert_torch.generative.export_hf.model_ext.gemma4")
    gemma.exportable_module, gemma.metadata_builder = exportable_module, gemma_metadata
    monkeypatch.setitem(sys.modules, model_ext.__name__, model_ext)
    monkeypatch.setitem(sys.modules, gemma.__name__, gemma)
    monkeypatch.setattr(compat, "_require_version", lambda name, expected: expected)
    return exportables, metadata, patches, gemma_metadata


def test_scoped_route_is_explicit_and_restored(monkeypatch):
    exportables, metadata, patches, gemma_metadata = _fake_exporter(monkeypatch)
    old_prefill = exportables.get_prefill_decode_exportables
    config = Gemma4TextConfig()
    options = types.SimpleNamespace(externalize_embedder=True)
    with compat.gemma4_text_export_context() as report:
        routed = exportables.get_prefill_decode_exportables(config, options)
        assert (
            routed[0].__name__
            == "LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder"
        )
        assert "per_layer_embedder" in exportables.get_additional_exportables(config)
        assert (
            metadata.get_metadata_builder(config) is gemma_metadata.build_llm_metadata
        )
        assert (
            patches._CONTEXT_REGISTRY["gemma4_text"]
            is patches._CONTEXT_REGISTRY["gemma4"]
        )
        assert config.model_type == "gemma4_text"
        assert report["config_relabelled"] is False
        assert report["litert_torch_version"] == compat.LITERT_TORCH_VERSION
    assert exportables.get_prefill_decode_exportables is old_prefill
    assert "gemma4_text" not in patches._CONTEXT_REGISTRY
    assert exportables.get_additional_exportables(config) == {"old": True}


def test_other_routes_are_unchanged(monkeypatch):
    exportables, metadata, _, _ = _fake_exporter(monkeypatch)
    other = types.SimpleNamespace(model_type="gemma4")
    options = types.SimpleNamespace(externalize_embedder=True)
    with compat.gemma4_text_export_compatibility():
        assert exportables.get_prefill_decode_exportables(other, options) == (
            "old-prefill",
            "old-decode",
        )
        assert exportables.get_additional_exportables(other) == {"old": True}
        assert metadata.get_metadata_builder(other) == "old-metadata"


def test_model_gate_requires_complete_both_embedding_and_norm_inventory():
    model = Gemma4ForCausalLM()
    compat._require_model(model)
    original = model.state_dict
    model.state_dict = lambda: {
        key: value
        for key, value in original().items()
        if key != "model.embed_tokens_per_layer.weight"
    }
    with pytest.raises(ValueError, match="both embeddings and norms"):
        compat._require_model(model)


def test_per_layer_exportable_uses_standalone_embedding_path(monkeypatch):
    exportables, _, _, _ = _fake_exporter(monkeypatch)
    model = Gemma4ForCausalLM()
    with compat.gemma4_text_export_compatibility():
        cls = exportables.get_additional_exportables(model.config)["per_layer_embedder"]
        assert cls(model).forward("tokens") == {"embeddings": ("tokens", None)}


def test_route_rejects_non_e2b_and_nonexternal_embedding(monkeypatch):
    exportables, _, _, _ = _fake_exporter(monkeypatch)
    config = Gemma4TextConfig()
    with compat.gemma4_text_export_compatibility():
        with pytest.raises(ValueError, match="externalized"):
            exportables.get_prefill_decode_exportables(
                config, types.SimpleNamespace(externalize_embedder=False)
            )
        with pytest.raises(ValueError, match="wrapper-only"):
            exportables.get_prefill_decode_exportables(
                config,
                types.SimpleNamespace(externalize_embedder=True, split_cache=True),
            )
        config.num_hidden_layers = 2
        with pytest.raises(TypeError, match="exact Gemma 4 E2B"):
            exportables.get_additional_exportables(config)


def test_real_tiny_standalone_forward_uses_final_norm_head_and_both_embeddings(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="5.16.1")
    if transformers.__version__ != compat.TRANSFORMERS_VERSION:
        pytest.skip("compatibility contract is pinned to Transformers 5.16.1")

    config = transformers.Gemma4TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        max_position_embeddings=64,
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=4,
        layer_types=["full_attention", "full_attention"],
        num_kv_shared_layers=1,
    )
    torch.manual_seed(7)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    assert isinstance(model, transformers.Gemma4ForCausalLM)
    assert model.config.model_type == "gemma4_text"
    assert not hasattr(model.model, "language_model")

    class Base(torch.nn.Module):
        def __init__(self, actual_model, export_config):
            super().__init__()
            self.model = actual_model
            self.export_config = export_config

        def adapt_inputs(
            self,
            tokens,
            embeddings,
            per_layer_embeddings,
            input_pos,
            kv_cache,
            mask,
            **kwargs,
        ):
            del tokens, input_pos, mask, kwargs
            return {
                "inputs_embeds": embeddings,
                "per_layer_inputs": per_layer_embeddings,
                "past_key_values": kv_cache,
                "use_cache": True,
            }

        def attention_kwargs(self):
            return {}

    class PerLayer(torch.nn.Module):
        def __init__(self, actual_model):
            super().__init__()
            self.model = actual_model

    upstream = types.SimpleNamespace(
        LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder=Base,
        LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder=Base,
        LiteRTExportableModuleForPerLayerEmbedder=PerLayer,
    )
    _, generate_cls, per_layer_cls = compat._standalone_classes(upstream)
    # The production gate is deliberately full-size. This test substitutes an
    # actual tiny model only after proving the exact HF class and topology.
    monkeypatch.setattr(compat, "_require_model", lambda value: None)

    token_ids = torch.tensor([[1, 2]], dtype=torch.long)
    token_embedding_calls = []
    per_layer_embedding_calls = []
    final_norm_calls = []
    head_calls = []
    hooks = [
        model.model.embed_tokens.register_forward_hook(
            lambda *args: token_embedding_calls.append(True)
        ),
        model.model.embed_tokens_per_layer.register_forward_hook(
            lambda *args: per_layer_embedding_calls.append(True)
        ),
        model.model.norm.register_forward_hook(
            lambda *args: final_norm_calls.append(True)
        ),
        model.lm_head.register_forward_hook(lambda *args: head_calls.append(True)),
    ]
    try:
        with torch.no_grad():
            # This is the unchanged generic token-embedder source used by the
            # exporter, and the compatibility per-layer section respectively.
            embeddings = model.get_input_embeddings()(token_ids)
            per_layer = per_layer_cls(model).forward(token_ids)["embeddings"]

            expected_hidden = model.model(
                inputs_embeds=embeddings,
                per_layer_inputs=per_layer,
                use_cache=False,
            ).last_hidden_state
            expected_logits = model.lm_head(expected_hidden)

            class Cache:
                def insert_dummy_cache_layers(self, unused_config):
                    cache = transformers.DynamicCache()
                    cache.remove_dummy_cache_layers = lambda other_config: cache
                    return cache

            options = types.SimpleNamespace(extra_kwargs={})
            actual = generate_cls(model, options).forward(
                embeddings, per_layer, None, Cache(), None
            )
    finally:
        for hook in hooks:
            hook.remove()

    assert torch.allclose(actual["logits"], expected_logits, atol=1e-6, rtol=1e-6)
    assert tuple(per_layer.shape) == (1, 2, 2, 4)
    assert token_embedding_calls
    assert per_layer_embedding_calls
    assert final_norm_calls
    assert head_calls
    assert model.config.model_type == "gemma4_text"
