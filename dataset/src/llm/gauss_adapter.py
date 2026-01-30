from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Optional

from .base import BaseLLMAdapter, LLMResult, LLMRateLimitError


class GaussAdapter(BaseLLMAdapter):
    def _resolve_endpoint(self, endpoint: str) -> str:
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/openapi/chat/v1/messages"):
            return endpoint
        return f"{endpoint}/openapi/chat/v1/messages"

    def _resolve_base(self, endpoint: str) -> str:
        endpoint = endpoint.rstrip("/")
        marker = "/openapi/chat/v1/messages"
        if marker in endpoint:
            return endpoint.split(marker)[0]
        return endpoint

    def list_models(self, all_models: bool = True) -> dict:
        endpoint = os.getenv("GAUSS_ENDPOINT") or self.spec.endpoint
        client_key = os.getenv("GAUSS_CLIENT_KEY")
        token = os.getenv("GAUSS_OPENAPI_TOKEN")
        email = os.getenv("GAUSS_USER_EMAIL")

        if not endpoint:
            return {"error": "GAUSS_ENDPOINT not set"}
        if not client_key:
            return {"error": "GAUSS_CLIENT_KEY not set"}
        if not token:
            return {"error": "GAUSS_OPENAPI_TOKEN not set"}

        base = self._resolve_base(endpoint)
        path = "all-models" if all_models else "models"
        url = f"{base}/openapi/chat/v1/{path}"
        headers = {
            "x-generative-ai-client": client_key,
            "x-openapi-token": token,
        }
        if email:
            headers["x-generative-ai-user-email"] = email

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            return {"error": str(exc)}

        try:
            payload = json.loads(raw)
        except Exception as exc:
            return {"error": f"gauss_models_parse_error: {exc}", "raw": raw}

        return {"models": payload}

    def generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> LLMResult:
        endpoint = os.getenv("GAUSS_ENDPOINT") or self.spec.endpoint
        client_key = os.getenv("GAUSS_CLIENT_KEY")
        token = os.getenv("GAUSS_OPENAPI_TOKEN")
        email = os.getenv("GAUSS_USER_EMAIL")
        model_id = os.getenv("GAUSS_MODEL_ID") or self.spec.model

        if not endpoint:
            return LLMResult(
                text="",
                raw=None,
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error="GAUSS_ENDPOINT not set",
            )
        if not client_key:
            return LLMResult(
                text="",
                raw=None,
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error="GAUSS_CLIENT_KEY not set",
            )
        if not token:
            return LLMResult(
                text="",
                raw=None,
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error="GAUSS_OPENAPI_TOKEN not set",
            )
        if not model_id or model_id == "GAUSS_MODEL_ID":
            return LLMResult(
                text="",
                raw=None,
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error="GAUSS_MODEL_ID not set",
            )

        url = self._resolve_endpoint(endpoint)
        headers = {
            "Content-Type": "application/json",
            "x-generative-ai-client": client_key,
            "x-openapi-token": token,
        }
        if email:
            headers["x-generative-ai-user-email"] = email

        llm_config: dict[str, object] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            llm_config["seed"] = seed

        body: dict[str, object] = {
            "modelIds": [model_id],
            "contents": [prompt],
            "isStream": False,
            "llmConfig": llm_config,
        }
        if system:
            body["systemPrompt"] = system

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            if hasattr(exc, "code") and getattr(exc, "code") == 429:
                raise
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
        try:
            payload = json.loads(raw)
        except Exception as exc:
            return LLMResult(
                text="",
                raw=raw,
                latency_ms=elapsed,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error=f"gauss_parse_error: {exc}",
            )

        if isinstance(payload, dict):
            response_code = payload.get("responseCode")
            status = payload.get("status")
            if response_code and response_code != "R20000":
                if response_code == "R40010":
                    raise LLMRateLimitError(
                        provider=self.spec.provider,
                        model=model_id,
                        limits=self.spec.limits,
                        headers={},
                        message=f"gauss rate limited ({response_code})",
                    )
                return LLMResult(
                    text="",
                    raw=payload,
                    latency_ms=elapsed,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=None,
                    model=self.spec.model,
                    provider=self.spec.provider,
                    error=f"gauss_error: {response_code} status={status}",
                )
            text = payload.get("content", "") or ""
        else:
            text = ""

        return LLMResult(
            text=text,
            raw=payload,
            latency_ms=elapsed,
            input_tokens=0,
            output_tokens=0,
            cost_usd=None,
            model=model_id,
            provider=self.spec.provider,
            error=None,
        )
