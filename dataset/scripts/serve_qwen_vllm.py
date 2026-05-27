#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys


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
    args = parser.parse_args()

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model_path,
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
