from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any
from typing import Optional

from .base import BaseLLMAdapter, LLMResult
from .http_transport import urlopen


class LocalAdapter(BaseLLMAdapter):
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
        cleaned = re.sub(r"(?is)<think>.*?</think>", "", text or "")
        cleaned = re.sub(
            r"(?is)<\|think\|>.*?(?=<\|end_think\|>|<\|end\|>|\{|\[|$)",
            "",
            cleaned,
        )
        cleaned = re.sub(r"(?is)<\|end_think\|>|<\|think\|>", "", cleaned)
        cleaned = re.sub(r"(?is)<\|channel\|>\s*(analysis|thought|thinking)\b.*?(?=<\|channel\|>|<\|message\|>|$)", "", cleaned)
        cleaned = re.sub(r"(?is)<\|start\|>\s*(analysis|thought|thinking)\b.*?(?=<\|end\|>|<\|start\|>|$)", "", cleaned)
        cleaned = re.sub(r"(?is)^\s*(analysis|thought|thinking)\s*:\s*.*?(?=\n\s*(final|assistant)\s*:|$)", "", cleaned)
        cleaned = re.sub(r"(?is)^\s*final\s*:\s*", "", cleaned.strip())
        return cleaned.strip()

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
        endpoint = (self.spec.endpoint or "").strip()
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

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        served_model_override = (os.environ.get("LOCAL_VLLM_SERVED_MODEL") or "").strip()
        request_model = served_model_override or self.spec.model
        body: dict[str, object] = {
            "model": request_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
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
        if self._is_truthy(os.environ.get("LOCAL_VLLM_ENABLE_THINKING")):
            body["chat_template_kwargs"] = {"enable_thinking": True}

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        timeout_s = 60.0
        timeout_raw = (os.environ.get("LOCAL_VLLM_TIMEOUT_SECONDS") or "").strip()
        if timeout_raw:
            try:
                timeout_s = max(1.0, float(timeout_raw))
            except Exception:
                timeout_s = 60.0
        elif self._model_is_large_reasoning_family():
            timeout_s = 600.0

        start = time.time()
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            message = str(exc)
            lower_msg = message.lower()
            if "cudacachingallocator.cpp" in lower_msg and "invalid argument" in lower_msg:
                message = (
                    f"{message}. Hint: your CUDA allocator config is incompatible with this stack. "
                    "Unset PYTORCH_CUDA_ALLOC_CONF (and LOCAL_CUDA_ALLOC_CONF), restart, and retry."
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
                error=message,
            )

        elapsed = (time.time() - start) * 1000
        payload = json.loads(raw)
        text = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        if self._should_strip_thinking():
            text = self._strip_thinking_text(text)
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
