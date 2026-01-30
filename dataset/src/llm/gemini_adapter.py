from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Optional

from .base import BaseLLMAdapter, LLMResult, LLMRateLimitError


class GeminiAdapter(BaseLLMAdapter):
    def _load_keys(self) -> list[str]:
        keys = []
        multi = os.getenv("GEMINI_API_KEYS")
        if multi:
            # Support comma-separated and newline-separated keys (e.g., "key1,\nkey2,\nkey3,")
            normalized = multi.replace("\n", ",")
            keys.extend([item.strip() for item in normalized.split(",") if item.strip()])
        single = os.getenv("GEMINI_API_KEY")
        if single and single not in keys:
            keys.append(single.strip())
        return keys

    def generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> LLMResult:
        keys = self._load_keys()
        if not keys:
            return LLMResult(
                text="",
                raw=None,
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error="GEMINI_API_KEY or GEMINI_API_KEYS not set",
            )

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.spec.model}:generateContent"

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
        start = time.time()
        raw = None
        last_error: Exception | None = None
        attempts = 0
        cycles = 3
        for _ in range(cycles):
            for api_key in keys:
                attempts += 1
                params = urllib.parse.urlencode({"key": api_key})
                url = f"{endpoint}?{params}"
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        raw = resp.read().decode("utf-8")
                    break
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    if exc.code in (429, 503):
                        continue
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
                except Exception as exc:
                    last_error = exc
                    # Timeout or transient error -> try next key
                    if isinstance(exc, TimeoutError):
                        continue
                    if "timed out" in str(exc).lower():
                        continue
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
            if raw:
                break

        if raw is None:
            if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 429:
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
                error=f"Gemini keys exhausted after {attempts} attempts: {last_error}",
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
