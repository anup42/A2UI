from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from .base import BaseLLMAdapter, LLMRateLimitError, LLMResult, extract_reasoning_metadata
from .http_transport import urlopen


class AzureOpenAIResponsesAdapter(BaseLLMAdapter):
    """Azure OpenAI Responses API adapter used by the Android GPT-5.4 flow."""

    def generate_batch(
        self,
        prompts: list[str],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seeds: Optional[list[int]],
        json_mode: bool = False,
        batch_name: Optional[str] = None,
    ) -> list[LLMResult]:
        if not prompts:
            return []
        if not self._single_call_batch_enabled() or len(prompts) <= 1:
            return [
                self.generate(
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seeds[idx] if seeds and idx < len(seeds) else None,
                    json_mode=json_mode,
                )
                for idx, prompt in enumerate(prompts)
            ]

        batch_prompt = self._build_batch_prompt(prompts)
        started = time.time()
        result = self.generate(
            prompt=batch_prompt,
            system=(
                "You are a strict batch execution wrapper. For each input item, answer "
                "that item's prompt independently. Return only JSON with a results array. "
                "Each result must contain the original index and a text string."
            ),
            temperature=temperature,
            max_tokens=min(max(1, int(max_tokens)) * len(prompts), self._max_output_tokens_cap()),
            seed=seeds[0] if seeds else None,
            json_mode=True,
        )
        if result.error:
            return self._batch_error_results(prompts, result.error, started)
        parsed = self._parse_batch_response(result.text)
        if parsed is None:
            if self._single_call_batch_sequential_fallback_enabled():
                return [
                    self.generate(
                        prompt=prompt,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seeds[idx] if seeds and idx < len(seeds) else None,
                        json_mode=json_mode,
                    )
                    for idx, prompt in enumerate(prompts)
                ]
            return self._batch_error_results(
                prompts,
                "Azure OpenAI batch response was not valid results JSON.",
                started,
            )

        out: list[LLMResult] = []
        total_in = result.input_tokens
        total_out = result.output_tokens
        for idx in range(len(prompts)):
            text = parsed.get(idx, "").strip()
            out.append(
                LLMResult(
                    text=text,
                    raw=result.raw,
                    latency_ms=result.latency_ms,
                    input_tokens=total_in // max(1, len(prompts)),
                    output_tokens=total_out // max(1, len(prompts)),
                    cost_usd=None,
                    model=self.spec.model,
                    provider=self.spec.provider,
                    error=None if text else "Azure OpenAI batch response omitted this item.",
                )
            )
        return out

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
        output_cap = self._max_output_tokens_cap()
        body = {
            "model": self.spec.model,
            "input": prompt,
            "max_output_tokens": max(1, min(int(max_tokens), output_cap)),
            "store": False,
        }
        if self._include_temperature():
            body["temperature"] = temperature
        reasoning_effort = self._reasoning_effort()
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
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
            with urlopen(request, timeout=self._timeout_seconds()) as response:
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
        raw_payload = self._raw_json(raw_text)
        reasoning_text, reasoning_source, reasoning_tokens = extract_reasoning_metadata(raw_payload)
        return LLMResult(
            text=text,
            raw=raw_payload,
            latency_ms=(time.time() - start) * 1000,
            input_tokens=usage[0],
            output_tokens=usage[1],
            cost_usd=None,
            model=self.spec.model,
            provider=self.spec.provider,
            error=None,
            reasoning_text=reasoning_text,
            reasoning_source=reasoning_source,
            reasoning_tokens=reasoning_tokens,
        )

    @staticmethod
    def _build_batch_prompt(prompts: list[str]) -> str:
        payload = {
            "instructions": (
                "Process each item independently. Do not merge items. "
                "Return JSON only, exactly: {\"results\":[{\"index\":0,\"text\":\"...\"}, ...]}"
            ),
            "items": [
                {
                    "index": idx,
                    "prompt": prompt,
                }
                for idx, prompt in enumerate(prompts)
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _parse_batch_response(text: str) -> dict[int, str] | None:
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = parsed.get("results") or parsed.get("items") or parsed.get("outputs")
        else:
            return None
        if not isinstance(items, list):
            return None
        out: dict[int, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except Exception:
                continue
            value = item.get("text") or item.get("output") or item.get("response")
            if isinstance(value, str):
                out[idx] = value
        return out

    def _batch_error_results(self, prompts: list[str], error: str, started: float) -> list[LLMResult]:
        return [
            LLMResult(
                text="",
                raw=None,
                latency_ms=(time.time() - started) * 1000,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error=error,
            )
            for _ in prompts
        ]

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

    @staticmethod
    def _max_output_tokens_cap() -> int:
        raw = (
            os.getenv("AZURE_OPENAI_MAX_OUTPUT_TOKENS_CAP")
            or os.getenv("A2UI_AZURE_MAX_OUTPUT_TOKENS_CAP")
            or "16384"
        ).strip()
        try:
            return max(1, int(raw))
        except Exception:
            return 16384

    @staticmethod
    def _reasoning_effort() -> str:
        effort = (
            os.getenv("AZURE_OPENAI_REASONING_EFFORT")
            or os.getenv("OPENAI_REASONING_EFFORT")
            or ""
        ).strip().lower()
        if effort in {"", "none", "off", "false", "0", "disabled"}:
            return ""
        return effort

    def _include_temperature(self) -> bool:
        raw = (
            os.getenv("AZURE_OPENAI_INCLUDE_TEMPERATURE")
            or os.getenv("A2UI_AZURE_INCLUDE_TEMPERATURE")
            or ""
        ).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        return not self.spec.model.lower().startswith("gpt-5")

    @staticmethod
    def _timeout_seconds() -> int:
        raw = (
            os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS")
            or os.getenv("A2UI_AZURE_TIMEOUT_SECONDS")
            or "300"
        ).strip()
        try:
            return max(30, int(raw))
        except Exception:
            return 300

    @staticmethod
    def _single_call_batch_enabled() -> bool:
        return (
            os.getenv("AZURE_OPENAI_SINGLE_CALL_BATCH")
            or os.getenv("A2UI_AZURE_SINGLE_CALL_BATCH")
            or ""
        ).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _single_call_batch_sequential_fallback_enabled() -> bool:
        return (
            os.getenv("AZURE_OPENAI_BATCH_SEQUENTIAL_FALLBACK")
            or os.getenv("A2UI_AZURE_BATCH_SEQUENTIAL_FALLBACK")
            or "1"
        ).strip().lower() not in {"0", "false", "no", "n", "off"}

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
