from __future__ import annotations

from typing import Any

from ir_training.models.base import ModelAdapter
from ir_training.models.gemma import GemmaAdapter
from ir_training.models.llama import LlamaAdapter
from ir_training.models.qwen import QwenAdapter

_ADAPTERS: dict[str, type[ModelAdapter]] = {
    GemmaAdapter.family: GemmaAdapter,
    QwenAdapter.family: QwenAdapter,
    LlamaAdapter.family: LlamaAdapter,
}


def create_adapter(model_config: dict[str, Any]) -> ModelAdapter:
    family = str(model_config.get("family") or "").strip().lower()
    model_id = str(model_config.get("model_id") or "").strip()
    if not family:
        raise ValueError("model.family is required")
    if not model_id:
        raise ValueError("model.model_id is required")
    adapter_cls = _ADAPTERS.get(family)
    if adapter_cls is None:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unsupported model family '{family}'. Supported: {supported}")
    return adapter_cls(model_id=model_id, config=model_config)


def supported_families() -> list[str]:
    return sorted(_ADAPTERS)
