from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Optional

from .base import BaseLLMAdapter, LLMResult, LLMRateLimitError


class GeminiAdapter(BaseLLMAdapter):
    def generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> LLMResult:
        api_key = os.getenv("GEMINI_API_KEY")
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
                error="GEMINI_API_KEY not set",
            )

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.spec.model}:generateContent"
        params = urllib.parse.urlencode({"key": api_key})
        url = f"{endpoint}?{params}"

        body: dict[str, object] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if seed is not None:
            body["generationConfig"]["seed"] = seed

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            # Surface 429 as a retryable exception to allow backoff.
            if hasattr(exc, "code") and getattr(exc, "code") == 429:
                raise LLMRateLimitError(
                    provider=self.spec.provider,
                    model=self.spec.model,
                    limits=self.spec.limits,
                    headers={},
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
        text = ""
        candidates = payload.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
        usage = payload.get("usageMetadata", {})
        return LLMResult(
            text=text or "",
            raw=payload,
            latency_ms=elapsed,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            cost_usd=None,
            model=self.spec.model,
            provider=self.spec.provider,
            error=None,
        )
