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

    def _read_http_error_body(self, exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read()
            if isinstance(body, bytes):
                return body.decode("utf-8", errors="replace")
            return str(body)
        except Exception:
            return ""

    def _infer_max_tokens_from_error(self, message: str) -> Optional[int]:
        lowered = message.lower()
        if "maxoutputtokens" not in lowered and "max_output_tokens" not in lowered:
            return None
        import re
        match = re.search(r"(?:maxoutputtokens|max_output_tokens)[^0-9]*([0-9]{2,})", lowered)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
        return None

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

        max_cap_env = os.getenv("GEMINI_MAX_OUTPUT_TOKENS")
        if max_cap_env:
            try:
                max_tokens = min(max_tokens, int(max_cap_env))
            except Exception:
                pass
        else:
            # Conservative cap to avoid invalid maxOutputTokens errors.
            max_tokens = min(max_tokens, 8192)

        def _build_body(use_seed: bool, token_limit: int) -> dict[str, object]:
            body: dict[str, object] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": token_limit,
                },
            }
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            if use_seed and seed is not None:
                body["generationConfig"]["seed"] = seed
            return body

        use_seed = seed is not None
        token_limit = max_tokens
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
                retry_without_seed = False
                retry_with_clamp = False
                while True:
                    body = _build_body(use_seed, token_limit)
                    data = json.dumps(body).encode("utf-8")
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
                        detail = self._read_http_error_body(exc)
                        lowered = detail.lower()
                        if exc.code in (401, 403) or "api_key_invalid" in lowered or "api key not valid" in lowered:
                            # Invalid key: try the next key in the list.
                            break
                        if exc.code in (429, 503):
                            break
                        if exc.code == 400:
                            if use_seed and "seed" in lowered and "unsupported" in lowered:
                                use_seed = False
                                if not retry_without_seed:
                                    retry_without_seed = True
                                    continue
                            limit = self._infer_max_tokens_from_error(detail)
                            if limit and limit < token_limit:
                                token_limit = limit
                                if not retry_with_clamp:
                                    retry_with_clamp = True
                                    continue
                            # If maxOutputTokens is too large but not explicit, try a smaller cap once.
                            if not retry_with_clamp and token_limit > 2048:
                                token_limit = 2048
                                retry_with_clamp = True
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
                                error=f"HTTP {exc.code}: {exc.reason} {detail}".strip(),
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
                            error=f"HTTP {exc.code}: {exc.reason} {detail}".strip(),
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
