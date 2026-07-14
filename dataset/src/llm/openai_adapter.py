from __future__ import annotations

import os
import time
from typing import Optional

from .base import BaseLLMAdapter, LLMResult, LLMRateLimitError, extract_reasoning_metadata


def _extract_rate_headers(headers) -> dict:
    extracted = {}
    if headers is None:
        return extracted
    try:
        items = headers.items()
    except Exception:
        items = []
    for key, value in items:
        lk = str(key).lower()
        if lk.startswith("x-ratelimit-") or lk.startswith("ratelimit-"):
            extracted[lk] = str(value)
    return extracted


def _is_falsey(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"0", "false", "no", "n", "off"}


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

        try:
            import httpx  # type: ignore
            from openai import OpenAI  # type: ignore
            from openai import APIStatusError, APITimeoutError, APIConnectionError  # type: ignore
        except Exception as exc:
            return LLMResult(
                text="",
                raw=None,
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error=f"openai/httpx package missing: {exc}",
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Optional explicit proxy/cert path for enterprise setups.
        # Falls back to standard env vars (HTTPS_PROXY/SSL_CERT_FILE) if unset.
        proxy_url = (os.getenv("OPENAI_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or "").strip()
        cert_path = (os.getenv("OPENAI_CA_CERT") or os.getenv("SSL_CERT_FILE") or "").strip()
        disable_ssl_verify = not _is_falsey(os.getenv("A2UI_DISABLE_SSL_VERIFY", "1"))

        transport_kwargs: dict[str, object] = {}
        if proxy_url:
            transport_kwargs["proxy"] = proxy_url
        if disable_ssl_verify:
            transport_kwargs["verify"] = False
        elif cert_path:
            transport_kwargs["verify"] = cert_path

        if transport_kwargs:
            transport = httpx.HTTPTransport(**transport_kwargs)
            http_client = httpx.Client(transport=transport, timeout=60.0)
        else:
            http_client = httpx.Client(timeout=60.0)

        base_url = (os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1").strip()
        client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

        start = time.time()
        try:
            req_kwargs: dict[str, object] = {
                "model": self.spec.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if seed is not None:
                req_kwargs["seed"] = seed
            if json_mode:
                req_kwargs["response_format"] = {"type": "json_object"}
            completion = client.chat.completions.create(**req_kwargs)
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status == 429:
                response = getattr(exc, "response", None)
                raise LLMRateLimitError(
                    provider=self.spec.provider,
                    model=self.spec.model,
                    limits=self.spec.limits,
                    headers=_extract_rate_headers(getattr(response, "headers", None)),
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
                error=f"HTTP {status}: {exc}",
            )
        except (APITimeoutError, APIConnectionError) as exc:
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
        finally:
            try:
                http_client.close()
            except Exception:
                pass

        elapsed = (time.time() - start) * 1000
        payload = completion.model_dump() if hasattr(completion, "model_dump") else {}
        reasoning_text, reasoning_source, reasoning_tokens = extract_reasoning_metadata(payload)
        text = ""
        try:
            if completion.choices and completion.choices[0].message:
                text = completion.choices[0].message.content or ""
        except Exception:
            text = ""
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return LLMResult(
            text=text or "",
            raw=payload,
            latency_ms=elapsed,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost_usd=None,
            model=self.spec.model,
            provider=self.spec.provider,
            error=None,
            reasoning_text=reasoning_text,
            reasoning_source=reasoning_source,
            reasoning_tokens=reasoning_tokens,
        )
