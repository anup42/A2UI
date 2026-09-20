"""Scoped LiteRT Torch 0.9.4 support for standalone Gemma 4 text models.

The upstream Gemma 4 exporter only dispatches the multimodal ``gemma4``
wrapper.  Full-parameter training uses Transformers' honest ``gemma4_text``
configuration and ``Gemma4ForCausalLM`` topology instead.  This module adapts
that topology without changing the model type or modifying installed packages.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
from collections.abc import Iterator
from typing import Any

LITERT_TORCH_VERSION = "0.9.4"
TRANSFORMERS_VERSION = "5.16.1"
MODEL_TYPE = "gemma4_text"
EXPECTED_STATE_KEYS = 541


def _require_version(distribution: str, expected: str) -> str:
    actual = importlib.metadata.version(distribution)
    if actual != expected:
        raise RuntimeError(
            f"Standalone Gemma 4 text export requires {distribution}=={expected}; "
            f"found {actual}"
        )
    return actual


def _require_config(config: Any) -> None:
    cls = type(config)
    if (
        cls.__module__ != "transformers.models.gemma4.configuration_gemma4"
        or cls.__name__ != "Gemma4TextConfig"
        or getattr(config, "model_type", None) != MODEL_TYPE
        or getattr(config, "num_hidden_layers", None) != 35
        or getattr(config, "hidden_size", None) != 1536
        or getattr(config, "hidden_size_per_layer_input", None) != 256
    ):
        raise TypeError(
            "Compatibility route requires the exact Gemma 4 E2B Gemma4TextConfig"
        )


def _require_model(model: Any) -> None:
    _require_config(getattr(model, "config", None))
    cls = type(model)
    if (
        cls.__module__ != "transformers.models.gemma4.modeling_gemma4"
        or cls.__name__ != "Gemma4ForCausalLM"
        or hasattr(getattr(model, "model", None), "language_model")
        or not callable(
            getattr(getattr(model, "model", None), "get_per_layer_inputs", None)
        )
    ):
        raise TypeError(
            "Compatibility route requires standalone Gemma4ForCausalLM topology"
        )
    keys = set(model.state_dict())
    required = {
        "model.embed_tokens.weight",
        "model.embed_tokens_per_layer.weight",
        "model.per_layer_model_projection.weight",
        "model.per_layer_projection_norm.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for layer in range(35):
        prefix = f"model.layers.{layer}."
        required.update(
            prefix + suffix
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
            )
        )
    if model.get_input_embeddings() is not model.model.embed_tokens:
        raise ValueError("Token embedder is not bound to model.embed_tokens")
    if not hasattr(model.model, "embed_tokens_per_layer"):
        raise ValueError("Per-layer embedder is missing embed_tokens_per_layer")
    if len(keys) != EXPECTED_STATE_KEYS or not required <= keys:
        raise ValueError(
            "Standalone E2B checkpoint inventory is incomplete or unexpected: "
            f"expected {EXPECTED_STATE_KEYS} keys including both embeddings and norms, "
            f"found {len(keys)}"
        )


def _standalone_classes(upstream: Any) -> tuple[type, type, type]:
    prefill_base = (
        upstream.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder
    )
    decode_base = (
        upstream.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder
    )

    class StandaloneGemma4TextPrefill(prefill_base):
        def __init__(self, model: Any, *args: Any, **kwargs: Any) -> None:
            _require_model(model)
            super().__init__(model, *args, **kwargs)

        def forward(
            self,
            embeddings: Any,
            per_layer_embeddings: Any,
            input_pos: Any,
            kv_cache: Any,
            mask: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if self.export_config.extra_kwargs.get("apply_gpu_composites", False):
                kwargs["apply_gpu_composites"] = True
            inputs = self.adapt_inputs(
                None,
                embeddings,
                per_layer_embeddings,
                input_pos,
                kv_cache,
                mask,
                use_bool_mask=self.export_config.extra_kwargs.get(
                    "use_bool_mask", False
                ),
                **kwargs,
            )
            config = self.model.config
            inputs["past_key_values"] = inputs[
                "past_key_values"
            ].insert_dummy_cache_layers(config)
            inputs |= self.attention_kwargs()
            output = self.model.model(**inputs)
            cache = output.past_key_values.remove_dummy_cache_layers(config)
            return {"kv_cache": cache}

    class StandaloneGemma4TextGenerate(decode_base):
        def __init__(self, model: Any, *args: Any, **kwargs: Any) -> None:
            _require_model(model)
            super().__init__(model, *args, **kwargs)

        def forward(
            self,
            embeddings: Any,
            per_layer_embeddings: Any,
            input_pos: Any,
            kv_cache: Any,
            mask: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if self.export_config.extra_kwargs.get("apply_gpu_composites", False):
                kwargs["apply_gpu_composites"] = True
            inputs = self.adapt_inputs(
                None,
                embeddings,
                per_layer_embeddings,
                input_pos,
                kv_cache,
                mask,
                use_bool_mask=self.export_config.extra_kwargs.get(
                    "use_bool_mask", False
                ),
                **kwargs,
            )
            config = self.model.config
            inputs["past_key_values"] = inputs[
                "past_key_values"
            ].insert_dummy_cache_layers(config)
            inputs |= self.attention_kwargs()
            output = self.model.model(**inputs)
            cache = output.past_key_values.remove_dummy_cache_layers(config)
            return {
                "kv_cache": cache,
                "logits": self.model.lm_head(output.last_hidden_state),
            }

    class StandaloneGemma4TextPerLayerEmbedder(
        upstream.LiteRTExportableModuleForPerLayerEmbedder
    ):
        def __init__(self, model: Any) -> None:
            _require_model(model)
            super().__init__(model)

        def forward(self, token_ids: Any) -> dict[str, Any]:
            return {
                "embeddings": self.model.model.get_per_layer_inputs(token_ids, None)
            }

    # Quantization recipes match exported module/output scopes. Preserve the
    # upstream Gemma 4 names even though the Python implementations adapt the
    # standalone topology.
    StandaloneGemma4TextPrefill.__name__ = (
        "LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder"
    )
    StandaloneGemma4TextGenerate.__name__ = (
        "LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder"
    )
    StandaloneGemma4TextPerLayerEmbedder.__name__ = (
        "LiteRTExportableModuleForPerLayerEmbedder"
    )
    StandaloneGemma4TextPrefill.__qualname__ = StandaloneGemma4TextPrefill.__name__
    StandaloneGemma4TextGenerate.__qualname__ = StandaloneGemma4TextGenerate.__name__
    StandaloneGemma4TextPerLayerEmbedder.__qualname__ = (
        StandaloneGemma4TextPerLayerEmbedder.__name__
    )

    return (
        StandaloneGemma4TextPrefill,
        StandaloneGemma4TextGenerate,
        StandaloneGemma4TextPerLayerEmbedder,
    )


@contextlib.contextmanager
def gemma4_text_export_context() -> Iterator[dict[str, Any]]:
    """Temporarily install the narrow standalone-text exporter route."""
    litert_version = _require_version("litert-torch", LITERT_TORCH_VERSION)
    transformers_version = _require_version("transformers", TRANSFORMERS_VERSION)

    from litert_torch.generative.export_hf.model_ext import (
        exportables,
        metadata_builder,
        patches,
    )
    from litert_torch.generative.export_hf.model_ext.gemma4 import exportable_module
    from litert_torch.generative.export_hf.model_ext.gemma4 import (
        metadata_builder as gemma4_metadata,
    )

    classes = _standalone_classes(exportable_module)
    original_prefill = exportables.get_prefill_decode_exportables
    original_additional = exportables.get_additional_exportables
    original_metadata = metadata_builder.get_metadata_builder
    missing = object()
    old_context = patches._CONTEXT_REGISTRY.get(MODEL_TYPE, missing)
    old_model_patch = patches._MODEL_PATCH_REGISTRY.get(MODEL_TYPE, missing)
    gemma_context = patches._CONTEXT_REGISTRY.get("gemma4")
    gemma_model_patch = patches._MODEL_PATCH_REGISTRY.get("gemma4")
    if gemma_context is None or gemma_model_patch is None:
        raise RuntimeError(
            "Pinned exporter did not register the expected Gemma 4 patches"
        )

    def prefill(config: Any, export_config: Any) -> Any:
        if getattr(config, "model_type", None) != MODEL_TYPE:
            return original_prefill(config, export_config)
        _require_config(config)
        if not export_config.externalize_embedder:
            raise ValueError("Standalone Gemma 4 E2B requires externalized embedders")
        unsupported = [
            name
            for name in ("split_cache", "externalize_rope", "aot_backend")
            if getattr(export_config, name, None)
        ]
        if getattr(export_config, "litert_lm_model_type_override", None) is not None:
            unsupported.append("litert_lm_model_type_override")
        if unsupported:
            raise ValueError(
                "Standalone Gemma 4 text compatibility does not support wrapper-only "
                f"export options: {sorted(unsupported)}"
            )
        return classes[:2]

    def additional(config: Any) -> Any:
        if getattr(config, "model_type", None) != MODEL_TYPE:
            return original_additional(config)
        _require_config(config)
        return {"per_layer_embedder": classes[2]}

    def metadata(config: Any) -> Any:
        if getattr(config, "model_type", None) != MODEL_TYPE:
            return original_metadata(config)
        _require_config(config)
        return gemma4_metadata.build_llm_metadata

    exportables.get_prefill_decode_exportables = prefill
    exportables.get_additional_exportables = additional
    metadata_builder.get_metadata_builder = metadata
    patches._CONTEXT_REGISTRY[MODEL_TYPE] = gemma_context
    patches._MODEL_PATCH_REGISTRY[MODEL_TYPE] = gemma_model_patch
    try:
        yield {
            "active": True,
            "model_type": MODEL_TYPE,
            "litert_torch_version": litert_version,
            "transformers_version": transformers_version,
            "prefill_decode_route": "standalone_gemma4_text",
            "token_embedder_route": "upstream_get_input_embeddings",
            "per_layer_embedder_route": "standalone_get_per_layer_inputs",
            "metadata_route": "gemma4",
            "config_relabelled": False,
            "installed_packages_modified": False,
        }
    finally:
        exportables.get_prefill_decode_exportables = original_prefill
        exportables.get_additional_exportables = original_additional
        metadata_builder.get_metadata_builder = original_metadata
        if old_context is missing:
            patches._CONTEXT_REGISTRY.pop(MODEL_TYPE, None)
        else:
            patches._CONTEXT_REGISTRY[MODEL_TYPE] = old_context
        if old_model_patch is missing:
            patches._MODEL_PATCH_REGISTRY.pop(MODEL_TYPE, None)
        else:
            patches._MODEL_PATCH_REGISTRY[MODEL_TYPE] = old_model_patch


# Descriptive backwards-compatible spelling for direct callers/tests.
gemma4_text_export_compatibility = gemma4_text_export_context
