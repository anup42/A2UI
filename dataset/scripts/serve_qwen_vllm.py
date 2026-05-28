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


def _model_path_with_config_overlay(model_path: str, architectures: list[str]) -> str:
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
    original = config.get("architectures")
    config["architectures"] = architectures
    (overlay / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        f"Using temporary vLLM config overlay: architectures {original} -> {architectures}; path={overlay}",
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
    hf_overrides_supported = _vllm_supports_flag("--hf-overrides")
    if arch_override:
        print(f"Applying vLLM architecture override: {arch_override}", flush=True)
        if not hf_overrides_supported:
            model_path = _model_path_with_config_overlay(args.model_path, arch_override)

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
    if arch_override and hf_overrides_supported:
        cmd += ["--hf-overrides", json.dumps({"architectures": arch_override})]
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
