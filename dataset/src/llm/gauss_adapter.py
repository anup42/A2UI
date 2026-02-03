from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
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

    def _candidate_endpoints(self, endpoint: str) -> list[str]:
        endpoint = endpoint.rstrip("/")
        base = self._resolve_base(endpoint)
        primary = self._resolve_endpoint(endpoint)
        candidates = [primary]
        alt_messages = f"{base}/openapi/chat/v1/messages"
        alt_with_models = f"{base}/openapi/chat/v1/messages-with-models"
        for url in (alt_messages, alt_with_models):
            if url not in candidates:
                candidates.append(url)
        return candidates

    def _resolve_max_new_tokens(self, requested: int) -> tuple[int, int | None]:
        limit: int | None = None
        env_limit = os.getenv("GAUSS_MAX_NEW_TOKENS")
        if env_limit:
            try:
                limit = int(env_limit)
            except ValueError:
                limit = None
        spec_limits = self.spec.limits if isinstance(self.spec.limits, dict) else {}
        spec_limit = spec_limits.get("max_new_tokens") if spec_limits else None
        if spec_limit is not None and limit is None:
            try:
                limit = int(spec_limit)
            except (TypeError, ValueError):
                limit = None
        if limit is None:
            # Safe default based on Gauss docs/examples.
            limit = 2048
        if requested > limit:
            return limit, limit
        return requested, limit

    def _clamp_temperature(self, value: float) -> tuple[float, bool]:
        # Gauss docs specify 0 < temperature < 1
        clamped = min(max(value, 0.01), 0.99)
        return clamped, clamped != value

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
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = ""
            return {
                "error": f"gauss_models_http_error: {exc.code} {exc.reason}",
                "body": body[:1000],
            }
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

        max_new_tokens, max_limit = self._resolve_max_new_tokens(max_tokens)
        temp_value, temp_clamped = self._clamp_temperature(temperature)

        llm_config: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temp_value,
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
        start = time.time()
        raw = None
        last_error: Optional[Exception] = None
        attempted: list[str] = []
        prompt_preview = prompt.replace("\n", " ")[:200]
        request_debug = {
            "modelIds": [model_id],
            "prompt_preview": prompt_preview,
            "prompt_length": len(prompt),
            "temperature": temp_value,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "system": bool(system),
        }
        if max_limit is not None and max_new_tokens != max_tokens:
            request_debug["max_new_tokens_clamped_from"] = max_tokens
        if temp_clamped:
            request_debug["temperature_clamped_from"] = temperature
        for candidate in self._candidate_endpoints(endpoint):
            attempted.append(candidate)
            req = urllib.request.Request(candidate, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                if code == 404:
                    continue
                if code == 429:
                    raise
                body = ""
                try:
                    body = exc.read().decode("utf-8")
                except Exception:
                    body = ""
                return LLMResult(
                    text="",
                    raw=None,
                    latency_ms=(time.time() - start) * 1000,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=None,
                    model=self.spec.model,
                    provider=self.spec.provider,
                    error=(
                        f"gauss_http_error: {code} {getattr(exc, 'reason', '')}; "
                        f"body={body[:1000]}; request={request_debug}; "
                        f"endpoint={candidate}"
                    ),
                )
            except Exception as exc:
                last_error = exc
                if hasattr(exc, "code") and getattr(exc, "code") == 404:
                    continue
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
                    error=f"{exc}; request={request_debug}; endpoint={candidate}",
                )

        if raw is None:
            error_text = "gauss_error: 404 Not Found"
            if last_error is not None and hasattr(last_error, "code"):
                error_text = f"gauss_error: {getattr(last_error, 'code')} {last_error}"
            return LLMResult(
                text="",
                raw=None,
                latency_ms=(time.time() - start) * 1000,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error=f"{error_text}. Tried: {', '.join(attempted)}. request={request_debug}",
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
