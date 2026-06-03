from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from .base import BaseLLMAdapter, LLMRateLimitError, LLMResult
from .http_transport import urlopen


class AzureOpenAIResponsesAdapter(BaseLLMAdapter):
    """Azure OpenAI Responses API adapter used by the Android GPT-5.4 flow."""

    def generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> LLMResult:
        api_key = (
            os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
            or ""
        ).strip()
        if not api_key:
            return self._error("AZURE_OPENAI_API_KEY/AZURE_OPENAI_SUBSCRIPTION_KEY not set")

        endpoint = self._responses_endpoint()
        body = {
            "model": self.spec.model,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max(1, min(int(max_tokens), 8192)),
            "store": False,
        }
        if system:
            body["instructions"] = system
        if json_mode:
            body["text"] = {"format": {"type": "json_object"}}

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "api-key": api_key,
            },
            method="POST",
        )

        start = time.time()
        try:
            with urlopen(request, timeout=180) as response:
                raw_text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raw_text = ""
            try:
                raw_text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code == 429:
                raise LLMRateLimitError(
                    provider=self.spec.provider,
                    model=self.spec.model,
                    limits=self.spec.limits,
                    headers=self._rate_headers(exc.headers),
                )
            usage = self._usage(raw_text)
            return LLMResult(
                text="",
                raw=self._raw_json(raw_text),
                latency_ms=(time.time() - start) * 1000,
                input_tokens=usage[0],
                output_tokens=usage[1],
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error=f"HTTP {exc.code}: {self._error_message(raw_text)}",
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

        usage = self._usage(raw_text)
        text = self._response_text(raw_text)
        if not text:
            return LLMResult(
                text="",
                raw=self._raw_json(raw_text),
                latency_ms=(time.time() - start) * 1000,
                input_tokens=usage[0],
                output_tokens=usage[1],
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error="Azure OpenAI response did not include output text.",
            )
        return LLMResult(
            text=text,
            raw=self._raw_json(raw_text),
            latency_ms=(time.time() - start) * 1000,
            input_tokens=usage[0],
            output_tokens=usage[1],
            cost_usd=None,
            model=self.spec.model,
            provider=self.spec.provider,
            error=None,
        )

    def _responses_endpoint(self) -> str:
        endpoint = (
            self.spec.endpoint
            or os.getenv("AZURE_OPENAI_RESPONSES_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or "https://genui1.openai.azure.com/openai/responses?api-version=2025-04-01-preview"
        ).strip()
        if endpoint and not endpoint.lower().startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"
        if "api-version=" not in endpoint.lower():
            separator = "&" if "?" in endpoint else "?"
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview").strip()
            endpoint = f"{endpoint}{separator}api-version={api_version}"
        return endpoint

    def _error(self, message: str) -> LLMResult:
        return LLMResult(
            text="",
            raw=None,
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=None,
            model=self.spec.model,
            provider=self.spec.provider,
            error=message,
        )

    @staticmethod
    def _raw_json(raw_text: str):
        try:
            return json.loads(raw_text)
        except Exception:
            return raw_text

    @staticmethod
    def _usage(raw_text: str) -> tuple[int, int]:
        try:
            root = json.loads(raw_text)
            usage = root.get("usage") or {}
            return (
                int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
                int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
            )
        except Exception:
            return 0, 0

    @staticmethod
    def _response_text(raw_text: str) -> str:
        try:
            root = json.loads(raw_text)
        except Exception:
            return ""
        output_text = root.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        parts: list[str] = []
        for item in root.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                text = (
                    content.get("text")
                    or content.get("output_text")
                    or content.get("refusal")
                )
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()

    @staticmethod
    def _error_message(raw_text: str) -> str:
        try:
            root = json.loads(raw_text)
            error = root.get("error") or {}
            if isinstance(error, dict):
                return str(error.get("message") or error.get("code") or raw_text)[:500]
            return str(root.get("message") or raw_text)[:500]
        except Exception:
            return raw_text[:500]

    @staticmethod
    def _rate_headers(headers) -> dict:
        out = {}
        try:
            items = headers.items()
        except Exception:
            return out
        for key, value in items:
            lower = str(key).lower()
            if lower.startswith("x-ratelimit-") or lower.startswith("ratelimit-"):
                out[lower] = str(value)
        return out
