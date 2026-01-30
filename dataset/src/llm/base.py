from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelSpec:
    name: str
    provider: str
    model: str
    supports_json_mode: bool = False
    endpoint: Optional[str] = None
    limits: Optional[dict] = None


@dataclass
class LLMResult:
    text: str
    raw: Any
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float]
    model: str
    provider: str
    error: Optional[str] = None


class BaseLLMAdapter:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> LLMResult:
        raise NotImplementedError


class LLMRateLimitError(Exception):
    code = 429

    def __init__(
        self,
        provider: str,
        model: str,
        limits: Optional[dict] = None,
        headers: Optional[dict] = None,
        message: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.limits = limits or {}
        self.headers = headers or {}
        self.message = message or "rate limit hit"
        super().__init__(self.__str__())

    def __str__(self) -> str:
        limits = self.limits if self.limits else "unset"
        headers = self.headers if self.headers else "none"
        return f"{self.message} provider={self.provider} model={self.model} limits={limits} headers={headers}"
