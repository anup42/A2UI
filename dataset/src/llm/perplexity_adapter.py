from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Optional

from .base import BaseLLMAdapter, LLMResult, LLMRateLimitError


def _extract_rate_headers(headers) -> dict:
    extracted = {}
    if headers is None:
        return extracted
    for key, value in headers.items():
        lk = key.lower()
        if lk.startswith("x-ratelimit-") or lk.startswith("ratelimit-"):
            extracted[lk] = value
    return extracted


def _extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


class PerplexityAdapter(BaseLLMAdapter):
    def generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> LLMResult:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            return LLMResult(
                text="",
                raw=None,
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error="PERPLEXITY_API_KEY not set",
            )

        base_url = os.getenv("PERPLEXITY_API_BASE", "https://api.perplexity.ai").rstrip("/")
        url = f"{base_url}/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, object] = {
            "model": self.spec.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        if json_mode and self.spec.supports_json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            if hasattr(exc, "code") and getattr(exc, "code") == 429:
                err_headers = getattr(exc, "headers", None)
                raise LLMRateLimitError(
                    provider=self.spec.provider,
                    model=self.spec.model,
                    limits=self.spec.limits,
                    headers=_extract_rate_headers(err_headers),
                )
            return LLMResult(
                text="",
                raw=None,
                latency_ms=(time.time() - start) * 1000,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error=str(exc),
            )

        elapsed = (time.time() - start) * 1000
        payload = json.loads(raw)
        usage = payload.get("usage", {})
        return LLMResult(
            text=_extract_message_text(payload),
            raw=payload,
            latency_ms=elapsed,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_usd=None,
            model=self.spec.model,
            provider=self.spec.provider,
            error=None,
        )
