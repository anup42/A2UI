from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Optional

from .base import BaseLLMAdapter, LLMResult, LLMRateLimitError
from .http_transport import urlopen


def _extract_rate_headers(headers) -> dict:
    extracted = {}
    if headers is None:
        return extracted
    for key, value in headers.items():
        lk = key.lower()
        if lk.startswith("x-ratelimit-"):
            extracted[lk] = value
    return extracted


class OpenAIAdapter(BaseLLMAdapter):
    def generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> LLMResult:
        api_key = os.getenv("OPENAI_API_KEY")
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
                error="OPENAI_API_KEY not set",
            )

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
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        start = time.time()
        try:
            with urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            if hasattr(exc, "code") and getattr(exc, "code") == 429:
                headers = getattr(exc, "headers", None)
                raise LLMRateLimitError(
                    provider=self.spec.provider,
                    model=self.spec.model,
                    limits=self.spec.limits,
                    headers=_extract_rate_headers(headers),
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
        text = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = payload.get("usage", {})
        return LLMResult(
            text=text or "",
            raw=payload,
            latency_ms=elapsed,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_usd=None,
            model=self.spec.model,
            provider=self.spec.provider,
            error=None,
        )
