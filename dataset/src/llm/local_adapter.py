from __future__ import annotations

import json
import hashlib
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from typing import Optional

from .base import BaseLLMAdapter, LLMResult, completion_metadata, extract_reasoning_metadata, split_reasoning_from_text
from .http_transport import urlopen


class LocalAdapter(BaseLLMAdapter):
    _endpoint_pool_lock = threading.Lock()
    _endpoint_pool_index = 0

    def __init__(self, spec):
        super().__init__(spec)
        self._model_path: Optional[str] = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    @staticmethod
    def _is_http_endpoint(endpoint: str) -> bool:
        lowered = endpoint.lower()
        return lowered.startswith("http://") or lowered.startswith("https://")

    @staticmethod
    def _normalize_http_endpoint(endpoint: str) -> str:
        normalized = (endpoint or "").strip()
        if not normalized:
            return ""
        trimmed = normalized.rstrip("/")
        lowered = trimmed.lower()
        if lowered.endswith("/v1/chat/completions"):
            return trimmed
        if lowered.endswith("/chat/completions"):
            return trimmed
        if lowered.endswith("/v1/models"):
            return f"{trimmed[:-len('/models')]}/chat/completions"
        if lowered.endswith("/v1"):
            return f"{trimmed}/chat/completions"
        if re.match(r"^https?://[^/]+$", trimmed, re.IGNORECASE):
            return f"{trimmed}/v1/chat/completions"
        return normalized

    @classmethod
    def _endpoint_pool(cls, fallback: str) -> list[str]:
        raw = (os.environ.get("LOCAL_VLLM_ENDPOINTS") or "").strip()
        endpoints: list[str] = []
        if raw:
            for part in re.split(r"[,;\s]+", raw):
                endpoint = cls._normalize_http_endpoint(part)
                if endpoint and cls._is_http_endpoint(endpoint) and endpoint not in endpoints:
                    endpoints.append(endpoint)

        fallback_endpoint = cls._normalize_http_endpoint(fallback)
        if not endpoints and fallback_endpoint and cls._is_http_endpoint(fallback_endpoint):
            endpoints.append(fallback_endpoint)
        return endpoints

    @classmethod
    def _select_http_endpoint(cls, fallback: str) -> str:
        endpoints = cls._endpoint_pool(fallback)
        if not endpoints:
            return fallback
        if len(endpoints) == 1:
            return endpoints[0]
        with cls._endpoint_pool_lock:
            endpoint = endpoints[cls._endpoint_pool_index % len(endpoints)]
            cls._endpoint_pool_index += 1
        return endpoint

    @staticmethod
    def _is_truthy(value: Optional[str]) -> bool:
        if value is None:
            return False
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _model_is_qwen_or_deepseek(self) -> bool:
        model_name = (self.spec.model or "").lower()
        return "qwen" in model_name or "deepseek" in model_name

    def _model_is_large_reasoning_family(self) -> bool:
        model_name = (self.spec.model or "").lower()
        return any(name in model_name for name in ("qwen", "deepseek", "gemma"))

    def _should_strip_thinking(self) -> bool:
        if self._model_is_large_reasoning_family():
            return True
        return self._is_truthy(os.environ.get("LOCAL_VLLM_STRIP_THINKING"))

    @staticmethod
    def _strip_thinking_text(text: str) -> str:
        # Only explicit complete model delimiters count as reasoning. Ordinary
        # prose such as "Thinking: ..." is valid source content and stays intact.
        _, cleaned = split_reasoning_from_text(text or "")
        return cleaned

    @staticmethod
    def _env_float(name: str, default: float | None = None) -> float | None:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    @staticmethod
    def _env_int(name: str, default: int | None = None) -> int | None:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except Exception:
            return default

    @classmethod
    def _context_retry_max_tokens(cls, message: str, requested_max_tokens: int) -> int | None:
        lowered = message.lower()
        if "maximum context length" not in lowered:
            return None
        context_match = re.search(
            r"maximum context length (?:is|of)\s+(\d+)",
            message,
            re.IGNORECASE,
        )
        if not context_match:
            context_match = re.search(
                r"maximum context length.*?(\d+)\s+tokens",
                message,
                re.IGNORECASE,
            )
        prompt_match = re.search(
            r"prompt contains at least\s+(\d+)\s+input tokens",
            message,
            re.IGNORECASE,
        )
        if not prompt_match:
            prompt_match = re.search(
                r"\((\d+)\s+in\s+(?:the\s+)?(?:messages|prompt|input)",
                message,
                re.IGNORECASE,
            )
        if not prompt_match:
            prompt_match = re.search(
                r"(\d+)\s+(?:input|prompt|message)\s+tokens",
                message,
                re.IGNORECASE,
            )
        if not context_match or not prompt_match:
            return None
        context_tokens = int(context_match.group(1))
        prompt_tokens = int(prompt_match.group(1))
        safety = cls._env_int("LOCAL_VLLM_CONTEXT_SAFETY_TOKENS", 32) or 0
        adjusted = context_tokens - prompt_tokens - max(0, safety)
        min_retry = cls._env_int("LOCAL_VLLM_MIN_RETRY_OUTPUT_TOKENS", 256) or 1
        if adjusted < min_retry or adjusted >= requested_max_tokens:
            return None
        return max(1, adjusted)

    @staticmethod
    def _exception_message(exc: Exception) -> str:
        message = str(exc)
        if hasattr(exc, "read"):
            try:
                body_text = exc.read().decode("utf-8", errors="replace")  # type: ignore[attr-defined]
                if body_text:
                    message = f"{message}: {body_text[:2000]}"
            except Exception:
                pass
        return message

    @staticmethod
    def _is_transient_vllm_server_error(message: str) -> bool:
        if not message:
            return False
        lowered = message.lower()
        transient_markers = (
            "connection refused",
            "connection reset",
            "connection aborted",
            "remote end closed connection",
            "remote disconnected",
            "temporarily unavailable",
            "service unavailable",
            "http error 500",
            "500 internal server error",
            "enginedeaderror",
            "enginecore encountered",
            "asyncllm output_handler failed",
        )
        return any(marker in lowered for marker in transient_markers)

    def _strict_offline_mode(self) -> bool:
        raw = os.environ.get("LOCAL_STRICT_OFFLINE")
        if raw is not None and raw.strip():
            return self._is_truthy(raw)
        # Default to strict offline for large local reasoning models.
        return self._model_is_large_reasoning_family()

    def _resolve_model_path(self) -> Optional[str]:
        strict_offline = self._strict_offline_mode()
        endpoint = os.path.expandvars(os.path.expanduser((self.spec.endpoint or "").strip()))
        if endpoint and not self._is_http_endpoint(endpoint):
            return endpoint

        model_name = (self.spec.model or "").lower()
        candidates: list[str] = []
        if "qwen" in model_name:
            candidates.append(os.environ.get("QWEN_MODEL_PATH", ""))
        if "deepseek" in model_name:
            candidates.append(os.environ.get("DEEPSEEK_MODEL_PATH", ""))
        if "gemma" in model_name:
            candidates.append(os.environ.get("GEMMA4_MODEL_PATH", ""))
        candidates.append(os.environ.get("LOCAL_MODEL_PATH", ""))
        candidates.extend(self._model_root_candidates(self.spec.model or ""))

        # Allow model field itself to be a local path.
        if self.spec.model and Path(os.path.expandvars(os.path.expanduser(self.spec.model))).exists():
            candidates.insert(0, self.spec.model)
        # If model looks like a Hugging Face repo id, allow loading from hub.
        if self.spec.model and "/" in self.spec.model and not strict_offline:
            candidates.append(self.spec.model)

        for raw in candidates:
            normalized = os.path.expandvars(os.path.expanduser((raw or "").strip()))
            if normalized:
                return normalized
        return None

    @staticmethod
    def _model_id_variants(model_id: str) -> list[str]:
        raw = (model_id or "").strip().strip("/")
        if not raw:
            return []
        leaf = raw.rsplit("/", 1)[-1]
        variants = [
            raw,
            raw.replace("/", "--"),
            raw.replace("/", "__"),
            raw.replace("/", "_"),
            leaf,
            leaf.replace("_", "-"),
            leaf.replace("-", "_"),
        ]
        lowered = [item.lower() for item in variants]
        upper_b = [item.replace("-31b-", "-31B-") for item in variants + lowered]
        deduped: list[str] = []
        for item in variants + lowered + upper_b:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    @classmethod
    def _model_root_candidates(cls, model_id: str) -> list[str]:
        root_values = [
            os.environ.get("MODEL_ROOT", ""),
            os.environ.get("LOCAL_MODEL_ROOT", ""),
            os.environ.get("A2UI_MODEL_ROOT", ""),
            os.environ.get("GEMMA4_MODEL_ROOT", ""),
            os.environ.get("QWEN_MODEL_ROOT", ""),
            os.environ.get("DEEPSEEK_MODEL_ROOT", ""),
        ]
        roots: list[Path] = []
        for raw in root_values:
            for part in (raw or "").split(os.pathsep):
                normalized = os.path.expandvars(os.path.expanduser(part.strip()))
                if normalized:
                    roots.append(Path(normalized))

        candidates: list[str] = []
        for root in roots:
            if not root.exists():
                continue
            if (root / "config.json").is_file():
                candidates.append(str(root))
            exact = root / Path((model_id or "").strip("/"))
            if exact.exists():
                candidates.append(str(exact))
            for variant in cls._model_id_variants(model_id):
                path = root / Path(variant)
                if path.exists():
                    candidates.append(str(path))
        return candidates

    def _local_input_device(self):
        if self._torch is None:
            return None
        try:
            for param in self._model.parameters():
                if getattr(param, "device", None) is not None and param.device.type != "meta":
                    return param.device
        except Exception:
            pass
        if getattr(self._model, "device", None) is not None:
            return self._model.device
        return self._torch.device("cpu")

    def _ensure_local_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        strict_offline = self._strict_offline_mode()
        local_files_only = strict_offline or self._is_truthy(os.environ.get("LOCAL_FILES_ONLY"))
        if strict_offline:
            # Prevent any accidental network fallback through HF hub.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        model_path = self._resolve_model_path()
        if not model_path:
            model_name = (self.spec.model or "").lower()
            if "qwen" in model_name:
                raise RuntimeError("QWEN_MODEL_PATH is not set for local Qwen model")
            if "deepseek" in model_name:
                raise RuntimeError("DEEPSEEK_MODEL_PATH is not set for local DeepSeek model")
            if "gemma" in model_name:
                raise RuntimeError("GEMMA4_MODEL_PATH is not set for local Gemma model")
            raise RuntimeError("LOCAL_MODEL_PATH is not set for provider=local model")
        if strict_offline and not Path(model_path).exists():
            raise RuntimeError(
                "Strict offline mode requires a local model directory path. "
                "Set QWEN_MODEL_PATH / DEEPSEEK_MODEL_PATH / GEMMA4_MODEL_PATH / LOCAL_MODEL_PATH to an existing folder."
            )

        self._model_path = model_path
        # Allocator tuning is opt-in because some CUDA/PyTorch stacks crash with
        # expandable_segments (invalid argument in CUDACachingAllocator).
        local_alloc_conf = (os.environ.get("LOCAL_CUDA_ALLOC_CONF") or "").strip()
        if local_alloc_conf and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = local_alloc_conf
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Local Transformers inference requires torch+transformers; "
                "install them in this environment."
            ) from exc

        self._torch = torch
        dtype_name = os.environ.get("LOCAL_MODEL_DTYPE", "float16").strip().lower()
        torch_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_name, torch.float16)
        device_map = os.environ.get("LOCAL_MODEL_DEVICE_MAP", "auto").strip() or "auto"
        load_in_4bit = self._is_truthy(os.environ.get("LOCAL_MODEL_LOAD_IN_4BIT"))
        attn_impl = (os.environ.get("LOCAL_MODEL_ATTN_IMPL") or "").strip()
        offload_dir = (os.environ.get("LOCAL_MODEL_OFFLOAD_DIR") or ".offload").strip()
        max_memory_per_gpu = (os.environ.get("LOCAL_MODEL_MAX_MEMORY") or "").strip()
        gpu_mem_util = (os.environ.get("LOCAL_MODEL_GPU_MEMORY_UTILIZATION") or "0.88").strip()
        cpu_memory = (os.environ.get("LOCAL_MODEL_CPU_MEMORY") or "64GiB").strip()

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "local_files_only": local_files_only,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        if torch.cuda.is_available():
            max_memory: dict[Any, str] | None = None
            if max_memory_per_gpu:
                max_memory = {idx: max_memory_per_gpu for idx in range(torch.cuda.device_count())}
            elif self._model_is_large_reasoning_family():
                # Auto-apply per-GPU memory caps for large local models unless explicitly overridden.
                try:
                    frac = float(gpu_mem_util)
                except Exception:
                    frac = 0.88
                frac = min(max(frac, 0.50), 0.98)
                max_memory = {}
                for idx in range(torch.cuda.device_count()):
                    total_gib = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
                    cap_gib = max(1, int(total_gib * frac))
                    max_memory[idx] = f"{cap_gib}GiB"
            if max_memory:
                max_memory["cpu"] = cpu_memory
                model_kwargs["max_memory"] = max_memory
                model_kwargs["offload_folder"] = offload_dir
                Path(offload_dir).mkdir(parents=True, exist_ok=True)

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except Exception as exc:
                raise RuntimeError("LOCAL_MODEL_LOAD_IN_4BIT=1 requires bitsandbytes installed") from exc
            bnb_dtype = torch_dtype if torch_dtype in (torch.float16, torch.bfloat16) else torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bnb_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=local_files_only,
            )
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
                local_files_only=local_files_only,
            )
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self._model.eval()

    def _http_generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool,
    ) -> LLMResult:
        endpoint = self._select_http_endpoint((self.spec.endpoint or "").strip())
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
                error="Local endpoint not configured",
            )
        local_max_output_raw = (os.environ.get("LOCAL_VLLM_MAX_OUTPUT_TOKENS") or "").strip()
        if local_max_output_raw:
            try:
                local_max_output = max(1, int(local_max_output_raw))
                if max_tokens > local_max_output:
                    max_tokens = local_max_output
            except Exception:
                pass

        thinking_setting = (os.environ.get("LOCAL_VLLM_ENABLE_THINKING") or "").strip()
        thinking_enabled = self._is_truthy(thinking_setting)
        send_template_kwargs_raw = os.environ.get("LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS")
        if send_template_kwargs_raw is None:
            # Qwen commonly relies on per-request chat-template kwargs. Gemma vLLM
            # builds often reject this field unless they expose the matching flags.
            send_template_kwargs = "qwen" in (self.spec.model or "").lower()
        else:
            send_template_kwargs = self._is_truthy(send_template_kwargs_raw)

        if (
            thinking_enabled
            and not send_template_kwargs
            and "gemma" in (self.spec.model or "").lower()
        ):
            marker = "<|think|>"
            if system:
                if not system.lstrip().startswith(marker):
                    system = f"{marker}\n{system}"
            else:
                system = marker

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        served_model_override = (os.environ.get("LOCAL_VLLM_SERVED_MODEL") or "").strip()
        request_model = served_model_override or self.spec.model
        use_hf_generation_config = self._is_truthy(
            os.environ.get("LOCAL_VLLM_USE_HF_GENERATION_CONFIG")
        )
        force_sampling_overrides = self._is_truthy(
            os.environ.get("LOCAL_VLLM_FORCE_SAMPLING_OVERRIDES")
        )
        body: dict[str, object] = {
            "model": request_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if not use_hf_generation_config or force_sampling_overrides:
            body["temperature"] = temperature
            default_top_p = 0.95 if "gemma" in (self.spec.model or "").lower() else None
            default_top_k = 64 if "gemma" in (self.spec.model or "").lower() else None
            top_p = self._env_float("LOCAL_VLLM_TOP_P", default_top_p)
            top_k = self._env_int("LOCAL_VLLM_TOP_K", default_top_k)
            repetition_penalty = self._env_float(
                "LOCAL_VLLM_REPETITION_PENALTY",
                1.0 if "gemma" in (self.spec.model or "").lower() else None,
            )
            if top_p is not None:
                body["top_p"] = top_p
            if top_k is not None:
                body["top_k"] = top_k
            if repetition_penalty is not None:
                body["repetition_penalty"] = repetition_penalty
        if seed is not None:
            body["seed"] = seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if send_template_kwargs and thinking_setting:
            # Leave the model's default intact unless reasoning is explicitly set.
            body["chat_template_kwargs"] = {"enable_thinking": thinking_enabled}

        timeout_s = 60.0
        timeout_raw = (os.environ.get("LOCAL_VLLM_TIMEOUT_SECONDS") or "").strip()
        if timeout_raw:
            try:
                timeout_s = max(1.0, float(timeout_raw))
            except Exception:
                timeout_s = 60.0
        elif self._model_is_large_reasoning_family():
            timeout_s = 600.0

        request_attempts: list[dict[str, Any]] = []

        def post_once(request_body: dict[str, object]) -> str:
            controls = {key: value for key, value in request_body.items() if key != "messages"}
            request_meta = {"attempt_index": len(request_attempts), "controls": controls,
                            "messages_sha256": hashlib.sha256(json.dumps(request_body.get("messages"), ensure_ascii=False).encode("utf-8")).hexdigest()}
            request_attempts.append(request_meta)
            started = time.monotonic()
            data = json.dumps(request_body).encode("utf-8")
            if self._is_truthy(os.environ.get("LOCAL_VLLM_LOG_REQUESTS")):
                print(
                    "Local vLLM POST "
                    f"endpoint={endpoint} "
                    f"model={request_body.get('model')} "
                    f"max_tokens={request_body.get('max_tokens')}",
                    file=sys.stderr,
                    flush=True,
                )
            req = urllib.request.Request(
                endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(req, timeout=timeout_s) as resp:
                    response_text = resp.read().decode("utf-8")
                request_meta["status"] = "completed"
                return response_text
            except Exception as exc:
                request_meta["status"] = "failed"
                request_meta["error"] = str(exc)[:500]
                raise
            finally:
                request_meta["latency_ms"] = (time.monotonic() - started) * 1000

        def post_with_server_retries(request_body: dict[str, object]) -> str:
            retry_enabled = self._is_truthy(
                os.environ.get("LOCAL_VLLM_RETRY_CONNECTION_ERRORS", "1")
            )
            if not retry_enabled:
                return post_once(request_body)

            interval_s = self._env_float("LOCAL_VLLM_RETRY_INTERVAL_SECONDS", 10.0) or 10.0
            max_wait_s = self._env_float("LOCAL_VLLM_RETRY_MAX_SECONDS", 0.0) or 0.0
            interval_s = max(1.0, interval_s)
            first_failure_at: float | None = None
            attempt = 0

            while True:
                try:
                    return post_once(request_body)
                except Exception as exc:
                    message = self._exception_message(exc)
                    if not self._is_transient_vllm_server_error(message):
                        raise
                    now = time.time()
                    if first_failure_at is None:
                        first_failure_at = now
                    elapsed_s = now - first_failure_at
                    if max_wait_s > 0 and elapsed_s >= max_wait_s:
                        raise
                    attempt += 1
                    if attempt == 1 or attempt % 6 == 0:
                        print(
                            "Local vLLM server unavailable for current request; "
                            f"retrying same sample in {interval_s:.0f}s "
                            f"(attempt={attempt}, elapsed={elapsed_s:.0f}s): {message[:500]}",
                            file=sys.stderr,
                            flush=True,
                        )
                    time.sleep(interval_s)

        start = time.time()
        try:
            raw = post_with_server_retries(body)
        except Exception as exc:
            message = self._exception_message(exc)
            retry_succeeded = False
            retry_max_tokens = self._context_retry_max_tokens(message, int(body["max_tokens"]))
            if retry_max_tokens is not None:
                body["max_tokens"] = retry_max_tokens
                try:
                    raw = post_with_server_retries(body)
                except Exception as retry_exc:
                    message = self._exception_message(retry_exc)
                else:
                    raw_payload = json.loads(raw)
                    raw_payload.setdefault("a2ui_retry", {})["reduced_max_tokens"] = retry_max_tokens
                    raw = json.dumps(raw_payload)
                    retry_max_tokens = None
                    retry_succeeded = True
            if not retry_succeeded:
                if retry_max_tokens is not None:
                    message = f"{message}. Retried with max_tokens={retry_max_tokens} but request still failed."
                lower_msg = message.lower()
                if "cudacachingallocator.cpp" in lower_msg and "invalid argument" in lower_msg:
                    message = (
                        f"{message}. Hint: your CUDA allocator config is incompatible with this stack. "
                        "Unset PYTORCH_CUDA_ALLOC_CONF (and LOCAL_CUDA_ALLOC_CONF), restart, and retry."
                    )
                if "maximum context length" in lower_msg:
                    message = (
                        f"{message}. Hint: lower LOCAL_STAGE3_PROMPT_MAX_TOKENS, "
                        "A2UI_GENUI_PROMPT_MAX_TOKENS, or LOCAL_VLLM_MAX_OUTPUT_TOKENS, "
                        "or increase VLLM_MAX_MODEL_LEN if the model/server supports it."
                    )
                return LLMResult(
                    text="",
                    raw={"a2ui_request_attempts": request_attempts},
                    latency_ms=(time.time() - start) * 1000,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=None,
                    model=self.spec.model,
                    provider=self.spec.provider,
                    error=message,
                )

        elapsed = (time.time() - start) * 1000
        payload = json.loads(raw)
        payload["a2ui_request_attempts"] = request_attempts
        text = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(text, list):
            text = "\n".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
        text = text if isinstance(text, str) else ""
        reason, complete = completion_metadata(payload)
        incomplete_inline = (
            (text.lstrip().startswith("<think>") and "</think>" not in text)
            or (text.lstrip().startswith("<|think|>") and not any(marker in text for marker in ("<|/think|>", "<|end_think|>")))
        )
        reasoning_text, reasoning_source, reasoning_tokens = extract_reasoning_metadata(payload)
        if not reasoning_text:
            inline_reasoning, cleaned_text = split_reasoning_from_text(text)
            if inline_reasoning:
                reasoning_text = inline_reasoning
                reasoning_source = "message.inline_thinking"
                text = cleaned_text
        if self._should_strip_thinking():
            text = self._strip_thinking_text(text)
        usage = payload.get("usage", {})
        completion_error = None
        if complete is False:
            completion_error = f"incomplete_completion: finish_reason={reason}"
        elif incomplete_inline:
            completion_error = "incomplete_completion: unclosed_reasoning_block"
            complete = False
        elif not text.strip():
            completion_error = "incomplete_completion: empty_final_content"
            complete = False
        return LLMResult(
            text=text or "",
            raw=payload,
            latency_ms=elapsed,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_usd=None,
            model=self.spec.model,
            provider=self.spec.provider,
            error=completion_error,
            finish_reason=reason,
            completion_complete=complete,
            reasoning_text=reasoning_text,
            reasoning_source=reasoning_source,
            reasoning_tokens=reasoning_tokens,
        )

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

        endpoint = self._select_http_endpoint((self.spec.endpoint or "").strip())
        if not self._is_http_endpoint(endpoint):
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

        default_parallelism = min(len(prompts), 4)
        parallelism = self._env_int(
            "LOCAL_VLLM_BATCH_PARALLELISM",
            self._env_int("LOCAL_VLLM_PARALLEL_REQUESTS", default_parallelism),
        )
        parallelism = max(1, min(len(prompts), int(parallelism or default_parallelism)))

        if parallelism <= 1:
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

        results: list[LLMResult | None] = [None] * len(prompts)
        start = time.time()

        def _generate_one(idx: int, prompt: str) -> tuple[int, LLMResult]:
            return idx, self.generate(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seeds[idx] if seeds and idx < len(seeds) else None,
                json_mode=json_mode,
            )

        with ThreadPoolExecutor(
            max_workers=parallelism,
            thread_name_prefix=f"local-vllm-batch-{batch_name or 'request'}",
        ) as executor:
            future_to_idx = {
                executor.submit(_generate_one, idx, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_idx, result = future.result()
                    results[result_idx] = result
                except Exception as exc:
                    results[idx] = LLMResult(
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

        return [
            result
            if result is not None
            else LLMResult(
                text="",
                raw=None,
                latency_ms=(time.time() - start) * 1000,
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error="local batch worker did not return a result",
            )
            for result in results
        ]

    def _local_prompt(self, prompt: str, system: Optional[str], json_mode: bool) -> str:
        if json_mode:
            prompt = f"{prompt}\n\nReturn only a valid JSON object."
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        chunks = []
        if system:
            chunks.append(f"System:\n{system.strip()}")
        chunks.append(f"User:\n{prompt.strip()}")
        chunks.append("Assistant:")
        return "\n\n".join(chunks)

    def _local_generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool,
    ) -> LLMResult:
        start = time.time()
        try:
            self._ensure_local_model_loaded()
            assert self._tokenizer is not None and self._model is not None and self._torch is not None
            full_prompt = self._local_prompt(prompt, system, json_mode)
            max_input_tokens_raw = (os.environ.get("LOCAL_MODEL_MAX_INPUT_TOKENS") or "").strip()
            max_input_tokens = 0
            if max_input_tokens_raw:
                try:
                    max_input_tokens = max(0, int(max_input_tokens_raw))
                except Exception:
                    max_input_tokens = 0
            elif self._model_is_large_reasoning_family():
                max_input_tokens = 81920

            tokenizer_kwargs: dict[str, Any] = {"return_tensors": "pt"}
            if max_input_tokens > 0:
                tokenizer_kwargs["truncation"] = True
                tokenizer_kwargs["max_length"] = max_input_tokens

            encoded = self._tokenizer(full_prompt, **tokenizer_kwargs)
            if "input_ids" not in encoded:
                raise RuntimeError("Tokenizer output missing input_ids")
            prompt_len = int(encoded["input_ids"].shape[-1])

            input_device = self._local_input_device()
            if input_device is None:
                raise RuntimeError("Unable to resolve local model input device")
            encoded = {k: v.to(input_device) for k, v in encoded.items()}

            if seed is not None:
                self._torch.manual_seed(seed)
                if self._torch.cuda.is_available():
                    self._torch.cuda.manual_seed_all(seed)

            max_new_tokens_raw = (os.environ.get("LOCAL_MODEL_MAX_NEW_TOKENS") or "").strip()
            max_new_tokens_cap = 0
            if max_new_tokens_raw:
                try:
                    max_new_tokens_cap = max(1, int(max_new_tokens_raw))
                except Exception:
                    max_new_tokens_cap = 0
            elif self._model_is_large_reasoning_family():
                max_new_tokens_cap = 8192
            if max_new_tokens_cap > 0:
                max_tokens = min(int(max_tokens), max_new_tokens_cap)

            do_sample = bool(temperature and temperature > 0.0)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
            }
            use_cache_raw = os.environ.get("LOCAL_MODEL_USE_CACHE", "")
            if use_cache_raw.strip():
                gen_kwargs["use_cache"] = self._is_truthy(use_cache_raw)
            elif self._model_is_large_reasoning_family():
                # Lower peak memory for long prompts.
                gen_kwargs["use_cache"] = False
            if do_sample:
                gen_kwargs["temperature"] = max(float(temperature), 1e-5)
                default_top_p = 0.95 if "gemma" in (self.spec.model or "").lower() else 0.95
                default_top_k = 64 if "gemma" in (self.spec.model or "").lower() else None
                top_p = self._env_float("LOCAL_VLLM_TOP_P", default_top_p)
                top_k = self._env_int("LOCAL_VLLM_TOP_K", default_top_k)
                repetition_penalty = self._env_float(
                    "LOCAL_VLLM_REPETITION_PENALTY",
                    1.0 if "gemma" in (self.spec.model or "").lower() else None,
                )
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p
                if top_k is not None:
                    gen_kwargs["top_k"] = top_k
                if repetition_penalty is not None:
                    gen_kwargs["repetition_penalty"] = repetition_penalty
            if self._tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = self._tokenizer.pad_token_id
            elif self._tokenizer.eos_token_id is not None:
                gen_kwargs["pad_token_id"] = self._tokenizer.eos_token_id

            with self._torch.no_grad():
                output_ids = self._model.generate(**encoded, **gen_kwargs)
            generated_ids = output_ids[0][prompt_len:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            reasoning_text, cleaned_text = split_reasoning_from_text(text)
            if reasoning_text:
                text = cleaned_text

            elapsed = (time.time() - start) * 1000
            return LLMResult(
                text=text,
                raw={"mode": "local_transformers", "model_path": self._model_path},
                latency_ms=elapsed,
                input_tokens=prompt_len,
                output_tokens=int(generated_ids.shape[-1]),
                cost_usd=None,
                model=self.spec.model,
                provider=self.spec.provider,
                error=None,
                reasoning_text=reasoning_text,
                reasoning_source="generated.inline_thinking" if reasoning_text else None,
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

    def generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> LLMResult:
        endpoint = os.path.expandvars(os.path.expanduser((self.spec.endpoint or "").strip()))
        strict_offline = self._strict_offline_mode()
        allow_http_in_offline = self._is_truthy(os.environ.get("LOCAL_ALLOW_HTTP_ENDPOINT"))
        if endpoint and self._is_http_endpoint(endpoint) and not (strict_offline and not allow_http_in_offline):
            return self._http_generate(prompt, system, temperature, max_tokens, seed, json_mode)
        return self._local_generate(prompt, system, temperature, max_tokens, seed, json_mode)
