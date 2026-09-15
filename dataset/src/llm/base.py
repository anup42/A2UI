from __future__ import annotations

import re
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
    reasoning_text: Optional[str] = None
    reasoning_source: Optional[str] = None
    reasoning_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    completion_complete: Optional[bool] = None


def completion_metadata(payload: Any) -> tuple[Optional[str], Optional[bool]]:
    if not isinstance(payload, dict):
        return None, None
    choices = payload.get("choices") or []
    candidates = payload.get("candidates") or []
    reason = None
    if choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
    elif candidates and isinstance(candidates[0], dict):
        reason = candidates[0].get("finishReason")
    if reason is None:
        return None, None
    reason = str(reason)
    if reason.lower() in {"length", "max_tokens", "content_filter", "safety", "recitation"}:
        return reason, False
    return reason, True if reason.lower() in {"stop", "end_turn", "eos_token"} else None


def _reasoning_value_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks = [_reasoning_value_text(item) for item in value]
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "summary", "reasoning_content", "reasoning"):
            text = _reasoning_value_text(value.get(key))
            if text:
                return text
    return ""


def extract_reasoning_metadata(payload: Any) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """Extract only reasoning explicitly returned by a provider payload."""
    if not isinstance(payload, dict):
        return None, None, None

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    usage_metadata = (
        payload.get("usageMetadata") if isinstance(payload.get("usageMetadata"), dict) else {}
    )
    token_candidates = [
        usage_metadata.get("thoughtsTokenCount"),
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else None,
        (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        if isinstance(usage.get("output_tokens_details"), dict)
        else None,
    ]
    reasoning_tokens: Optional[int] = None
    for value in token_candidates:
        try:
            if value is not None:
                reasoning_tokens = max(0, int(value))
                break
        except (TypeError, ValueError):
            continue

    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            for key in ("reasoning_content", "reasoning_text", "reasoning", "analysis"):
                text = _reasoning_value_text(message.get(key))
                if text:
                    return text, f"message.{key}", reasoning_tokens

    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        thought_chunks: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict) and part.get("thought") is True:
                    text = _reasoning_value_text(part.get("text"))
                    if text:
                        thought_chunks.append(text)
        if thought_chunks:
            return "\n".join(thought_chunks), "candidate.thought_parts", reasoning_tokens

    output = payload.get("output")
    if isinstance(output, list):
        summary_chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            text = _reasoning_value_text(item.get("summary"))
            if not text:
                text = _reasoning_value_text(item.get("content"))
            if text:
                summary_chunks.append(text)
        if summary_chunks:
            return "\n".join(summary_chunks), "response.reasoning_summary", reasoning_tokens

    return None, None, reasoning_tokens


def split_reasoning_from_text(text: str) -> tuple[Optional[str], str]:
    """Separate common inline thinking blocks while preserving the final answer."""
    raw = text or ""
    chunks: list[str] = []
    patterns = (
        r"(?is)^\s*<think>\s*(.*?)\s*</think>",
        r"(?is)^\s*<\|think\|>\s*(.*?)\s*(?:<\|/think\|>|<\|end_think\|>)",
        r"(?is)^\s*<\|channel\|>\s*(?:analysis|thought|thinking)\b(.*?)(?=<\|channel\|>|<\|message\|>|$)",
        r"(?is)^\s*<\|channel>\s*(?:analysis|thought|thinking)\b(.*?)(?:<channel\|>|$)",
        r"(?is)^\s*<\|start\|>\s*(?:analysis|thought|thinking)\b(.*?)(?=<\|end\|>|<\|start\|>|$)",
    )
    cleaned = raw
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        chunks.extend(match.strip() for match in matches if match.strip())
        cleaned = re.sub(pattern, "", cleaned)
    reasoning = "\n".join(chunks).strip() or None
    return reasoning, cleaned.strip()


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
