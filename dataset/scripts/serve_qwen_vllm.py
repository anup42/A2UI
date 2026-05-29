#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ARCHITECTURE_ALIASES = {
    # Qwen3.6 A3B checkpoints currently advertise the newer Transformers
    # architecture name, while vLLM 0.11.x exposes the compatible Qwen3 MoE
    # causal-LM implementation in its model registry.
    "Qwen3_5MoeForConditionalGeneration": "Qwen3MoeForCausalLM",
    "Qwen3_5MoeForCausalLM": "Qwen3MoeForCausalLM",
}

QWEN36_TEXT_ROPE_PARAMETERS = {
    # vLLM's Qwen3.6 recipe documents these text RoPE parameters for Qwen3.6
    # long-context serving. They also avoid the older Qwen3 MoE path reading an
    # incompatible mrope_section from the newer Qwen3.6 config.
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
}

QWEN36_TEXT_CONFIG_COMPAT_OVERRIDES = {
    # vLLM 0.11.x's Qwen3 MoE implementation expects the older MoE sparsity
    # cadence key. Qwen3.6 uses the newer Qwen3.5 text config shape and may not
    # expose this attribute after AutoConfig materialization.
    "decoder_sparse_step": 1,
    "rope_parameters": QWEN36_TEXT_ROPE_PARAMETERS,
    "rope_scaling": QWEN36_TEXT_ROPE_PARAMETERS,
}


def _vllm_supports_flag(flag: str) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except Exception:
        return False
    return flag in ((result.stdout or "") + (result.stderr or ""))


def _auto_architecture_override(model_path: str, requested: str) -> list[str] | None:
    if requested.lower() in {"", "0", "false", "none", "off"}:
        return None
    if requested.lower() != "auto":
        return [item.strip() for item in requested.split(",") if item.strip()]

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not inspect model config for architecture override: {exc}", flush=True)
        return None

    architectures = config.get("architectures") or []
    mapped = [ARCHITECTURE_ALIASES[arch] for arch in architectures if arch in ARCHITECTURE_ALIASES]
    return mapped or None


def _deep_update(target: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _auto_hf_overrides(model_path: str, architectures: list[str] | None) -> dict:
    overrides: dict = {}
    if architectures:
        overrides["architectures"] = architectures
    if architectures and "Qwen3MoeForCausalLM" in architectures:
        overrides["text_config"] = QWEN36_TEXT_CONFIG_COMPAT_OVERRIDES
        print(
            "Applying Qwen3.6 text config compatibility override: "
            f"{QWEN36_TEXT_CONFIG_COMPAT_OVERRIDES}",
            flush=True,
        )
    return overrides


def _model_path_with_config_overlay(model_path: str, overrides: dict) -> str:
    source = Path(model_path).resolve()
    config_path = source / "config.json"
    if not config_path.is_file():
        return str(source)

    overlay = Path(tempfile.mkdtemp(prefix="a2ui_qwen_vllm_model_"))
    for child in source.iterdir():
        target = overlay / child.name
        if child.name == "config.json":
            continue
        try:
            target.symlink_to(child, target_is_directory=child.is_dir())
        except Exception:
            if child.is_dir():
                shutil.copytree(child, target, symlinks=True)
            else:
                shutil.copy2(child, target)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    original_architectures = config.get("architectures")
    original_text_config = dict(config.get("text_config") or {})
    _deep_update(config, overrides)
    (overlay / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    text_config = config.get("text_config") or {}
    print(
        "Using temporary vLLM config overlay: "
        f"architectures {original_architectures} -> {config.get('architectures')}; "
        "text_config keys "
        f"{sorted(original_text_config.keys())} -> {sorted(text_config.keys())}; "
        f"decoder_sparse_step={text_config.get('decoder_sparse_step')}; "
        f"rope_scaling={text_config.get('rope_scaling')}; "
        f"path={overlay}",
        flush=True,
    )
    return str(overlay)


def _install_runtime_shims(env: dict[str, str]) -> None:
    shim_dir = Path(tempfile.mkdtemp(prefix="a2ui_qwen_vllm_shims_"))
    sitecustomize = shim_dir / "sitecustomize.py"
    sitecustomize.write_text(
        """
# Auto-installed by A2UI's vLLM launcher.
# Bridges small Transformers/vLLM API gaps for newer Qwen checkpoints on
# clusters pinned to older manylinux-compatible vLLM wheels.
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        @property
        def all_special_tokens_extended(self):
            return list(getattr(self, "all_special_tokens", []) or [])

        PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended
except Exception:
    pass

try:
    # Newer Qwen3.5/3.6 text configs can omit legacy attributes expected by
    # vLLM 0.11.x's Qwen3 MoE model class. Config-file overlays are not enough
    # when Transformers drops unknown keys while materializing the config
    # object, so patch the config class directly.
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )

    if not hasattr(Qwen3_5MoeTextConfig, "decoder_sparse_step"):
        def _a2ui_get_decoder_sparse_step(self):
            return self.__dict__.get("decoder_sparse_step", 1)

        def _a2ui_set_decoder_sparse_step(self, value):
            self.__dict__["decoder_sparse_step"] = value

        Qwen3_5MoeTextConfig.decoder_sparse_step = property(
            _a2ui_get_decoder_sparse_step,
            _a2ui_set_decoder_sparse_step,
        )
        print(
            "A2UI vLLM shim: added Qwen3_5MoeTextConfig.decoder_sparse_step",
            flush=True,
        )
except Exception as exc:
    print(f"A2UI vLLM shim: Qwen3.5/3.6 config patch not installed: {exc}", flush=True)

try:
    # vLLM 0.11.x accepts Qwen3.6 mrope_section through rope_scaling, but its
    # Qwen3 MoE path does not pass the matching partial_rotary_factor into
    # get_rope(). Newer vLLM builds handle this internally. Keep this shim
    # scoped to rope configs that explicitly provide partial_rotary_factor.
    import vllm.model_executor.layers.rotary_embedding as _a2ui_rope

    if not getattr(_a2ui_rope.get_rope, "_a2ui_qwen36_partial_patch", False):
        _a2ui_original_get_rope = _a2ui_rope.get_rope

        def _a2ui_get_rope(*args, **kwargs):
            rope_scaling = kwargs.get("rope_scaling")
            partial_rotary_factor = kwargs.get("partial_rotary_factor", 1.0)
            if (
                isinstance(rope_scaling, dict)
                and "partial_rotary_factor" in rope_scaling
                and partial_rotary_factor == 1.0
            ):
                kwargs["partial_rotary_factor"] = float(
                    rope_scaling["partial_rotary_factor"]
                )
            return _a2ui_original_get_rope(*args, **kwargs)

        _a2ui_get_rope._a2ui_qwen36_partial_patch = True
        _a2ui_rope.get_rope = _a2ui_get_rope
        print(
            "A2UI vLLM shim: enabled rope_scaling partial_rotary_factor patch",
            flush=True,
        )
except Exception as exc:
    print(f"A2UI vLLM shim: rope patch not installed: {exc}", flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(shim_dir) if not current else f"{shim_dir}{os.pathsep}{current}"
    print(f"Installed vLLM runtime shims: {sitecustomize}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a Qwen-compatible vLLM OpenAI API server.")
    parser.add_argument("--model-path", required=True, help="Local Hugging Face model directory.")
    parser.add_argument("--served-model-name", required=True, help="Model name exposed through /v1 APIs.")
    parser.add_argument("--gpus", type=int, default=1, help="Tensor-parallel GPU count.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--swap-space", type=int, default=None, help="CPU swap space in GiB.")
    parser.add_argument(
        "--quantization-mode",
        default=os.environ.get("VLLM_QUANTIZATION_MODE", "none"),
        choices=["none", "int8", "bnb-int8", "bitsandbytes-int8", "bnb-4bit", "bitsandbytes-4bit", "4bit", "fp8"],
        help="Optional vLLM weight quantization mode.",
    )
    parser.add_argument("--kv-cache-dtype", default=os.environ.get("VLLM_KV_CACHE_DTYPE", ""))
    parser.add_argument("--cpu-offload-gb", default=os.environ.get("VLLM_CPU_OFFLOAD_GB", ""))
    parser.add_argument("--max-num-seqs", default=os.environ.get("VLLM_MAX_NUM_SEQS", ""))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--enable-reasoning", action="store_true")
    parser.add_argument("--reasoning-parser", default="qwen3")
    parser.add_argument(
        "--architecture-override",
        default=os.environ.get("VLLM_ARCHITECTURE_OVERRIDE", "auto"),
        help="Comma-separated architecture override, 'auto', or 'none'.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    _install_runtime_shims(env)

    model_path = args.model_path
    arch_override = _auto_architecture_override(args.model_path, args.architecture_override)
    hf_overrides = _auto_hf_overrides(args.model_path, arch_override)
    hf_overrides_supported = _vllm_supports_flag("--hf-overrides")
    if hf_overrides:
        print(f"Applying vLLM HF overrides: {hf_overrides}", flush=True)
        force_overlay = os.environ.get("VLLM_FORCE_CONFIG_OVERLAY", "auto").lower()
        overlay_required = "text_config" in hf_overrides and force_overlay in {"auto", "1", "true", "yes", "on"}
        if overlay_required or not hf_overrides_supported:
            model_path = _model_path_with_config_overlay(args.model_path, hf_overrides)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        args.served_model_name,
        "--tensor-parallel-size",
        str(max(1, args.gpus)),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.swap_space is not None:
        cmd += ["--swap-space", str(args.swap_space)]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    quantization_mode = str(args.quantization_mode or "none").lower()
    if quantization_mode in {"int8", "bnb-int8", "bitsandbytes-int8"}:
        cmd += [
            "--quantization",
            "bitsandbytes",
            "--load-format",
            "bitsandbytes",
            "--model-loader-extra-config",
            '{"load_in_8bit":true,"load_in_4bit":false}',
        ]
    elif quantization_mode in {"bnb-4bit", "bitsandbytes-4bit", "4bit"}:
        cmd += [
            "--quantization",
            "bitsandbytes",
            "--load-format",
            "bitsandbytes",
            "--model-loader-extra-config",
            '{"load_in_8bit":false,"load_in_4bit":true}',
        ]
    elif quantization_mode == "fp8":
        cmd += ["--quantization", "fp8"]
    elif quantization_mode not in {"", "none", "off", "false", "0"}:
        raise SystemExit(f"Unsupported --quantization-mode: {args.quantization_mode}")
    if args.kv_cache_dtype:
        cmd += ["--kv-cache-dtype", args.kv_cache_dtype]
    if args.cpu_offload_gb:
        cmd += ["--cpu-offload-gb", str(args.cpu_offload_gb)]
    if args.max_num_seqs:
        cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if hf_overrides and hf_overrides_supported and model_path == args.model_path:
        cmd += ["--hf-overrides", json.dumps(hf_overrides)]
    if args.enable_reasoning:
        if _vllm_supports_flag("--enable-reasoning"):
            cmd.append("--enable-reasoning")
        if _vllm_supports_flag("--reasoning-parser"):
            cmd += ["--reasoning-parser", args.reasoning_parser]
        else:
            print("vLLM build does not expose --reasoning-parser; continuing without it.", flush=True)

    print("Starting vLLM:", " ".join(cmd), flush=True)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
