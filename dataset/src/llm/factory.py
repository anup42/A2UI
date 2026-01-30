from __future__ import annotations

from typing import Dict

from .base import BaseLLMAdapter, ModelSpec
from .gemini_adapter import GeminiAdapter
from .openai_adapter import OpenAIAdapter
from .local_adapter import LocalAdapter
from .gauss_adapter import GaussAdapter


def build_adapter(spec: ModelSpec) -> BaseLLMAdapter:
    provider = spec.provider.lower()
    if provider == "openai":
        return OpenAIAdapter(spec)
    if provider == "gemini":
        return GeminiAdapter(spec)
    if provider == "gauss":
        return GaussAdapter(spec)
    if provider == "local":
        return LocalAdapter(spec)
    raise ValueError(f"Unsupported provider: {spec.provider}")


def load_model_specs(data: Dict) -> list[ModelSpec]:
    specs = []
    for item in data.get("models", []):
        specs.append(
            ModelSpec(
                name=item.get("name"),
                provider=item.get("provider"),
                model=item.get("model"),
                supports_json_mode=bool(item.get("supports_json_mode")),
                endpoint=item.get("endpoint"),
                limits=item.get("limits"),
            )
        )
    return specs
