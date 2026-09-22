from __future__ import annotations

from typing import Dict

from .base import BaseLLMAdapter, ModelSpec
from .gemini_adapter import GeminiAdapter
from .openai_adapter import OpenAIAdapter
from .azure_openai_responses_adapter import AzureOpenAIResponsesAdapter
from .local_adapter import LocalAdapter
from .openrouter_adapter import OpenRouterAdapter
from .perplexity_adapter import PerplexityAdapter


def build_adapter(spec: ModelSpec) -> BaseLLMAdapter:
    provider = spec.provider.lower()
    if provider == "openai":
        return OpenAIAdapter(spec)
    if provider == "azure_openai":
        return AzureOpenAIResponsesAdapter(spec)
    if provider == "gemini":
        return GeminiAdapter(spec)
    if provider == "openrouter":
        return OpenRouterAdapter(spec)
    if provider == "perplexity":
        return PerplexityAdapter(spec)
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
