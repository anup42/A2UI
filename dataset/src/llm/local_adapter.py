from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any
from typing import Optional

from .base import BaseLLMAdapter, LLMResult


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

    def _resolve_model_path(self) -> Optional[str]:
        endpoint = os.path.expandvars(os.path.expanduser((self.spec.endpoint or "").strip()))
        if endpoint and not self._is_http_endpoint(endpoint):
            return endpoint

        model_name = (self.spec.model or "").lower()
        candidates: list[str] = []
        if "qwen" in model_name:
            candidates.append(os.environ.get("QWEN_MODEL_PATH", ""))
        if "deepseek" in model_name:
            candidates.append(os.environ.get("DEEPSEEK_MODEL_PATH", ""))
        candidates.append(os.environ.get("LOCAL_MODEL_PATH", ""))

        # Allow model field itself to be a local path.
        if self.spec.model and Path(os.path.expandvars(os.path.expanduser(self.spec.model))).exists():
            candidates.insert(0, self.spec.model)
        # If model looks like a Hugging Face repo id, allow loading from hub.
        if self.spec.model and "/" in self.spec.model:
            candidates.append(self.spec.model)

        for raw in candidates:
            normalized = os.path.expandvars(os.path.expanduser((raw or "").strip()))
            if normalized:
                return normalized
        return None

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
        model_path = self._resolve_model_path()
        if not model_path:
            model_name = (self.spec.model or "").lower()
            if "qwen" in model_name:
                raise RuntimeError("QWEN_MODEL_PATH is not set for local Qwen model")
            if "deepseek" in model_name:
                raise RuntimeError("DEEPSEEK_MODEL_PATH is not set for local DeepSeek model")
            raise RuntimeError("LOCAL_MODEL_PATH is not set for provider=local model")

        self._model_path = model_path
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
        cpu_memory = (os.environ.get("LOCAL_MODEL_CPU_MEMORY") or "64GiB").strip()

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        if max_memory_per_gpu and torch.cuda.is_available():
            max_memory = {idx: max_memory_per_gpu for idx in range(torch.cuda.device_count())}
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
            )
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
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

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, object] = {
            "model": self.spec.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
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

        elapsed = (time.time() - start) * 1000
        payload = json.loads(raw)
        text = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
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
            encoded = self._tokenizer(full_prompt, return_tensors="pt")
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

            do_sample = bool(temperature and temperature > 0.0)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(float(temperature), 1e-5)
                gen_kwargs["top_p"] = 0.95
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
        if endpoint and self._is_http_endpoint(endpoint):
            return self._http_generate(prompt, system, temperature, max_tokens, seed, json_mode)
        return self._local_generate(prompt, system, temperature, max_tokens, seed, json_mode)
