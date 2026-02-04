from __future__ import annotations

import argparse
import os
import sys
from typing import List


def _build_command(
    model_path: str,
    served_model_name: str,
    gpus: int,
    host: str,
    port: int,
    dtype: str,
    gpu_mem_util: float,
    max_model_len: int | None,
    trust_remote_code: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(gpus),
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch vLLM OpenAI-compatible server for Qwen3-Coder-30B-A3B-Instruct."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local folder path for the model (or set QWEN_MODEL_PATH).",
    )
    parser.add_argument(
        "--served-model-name",
        default="qwen3-coder-30b-a3b-instruct",
        help="Model name exposed via the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="Tensor-parallel GPU count (4 or 8 recommended for V100).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help=(
            "Optional comma-separated GPU ids to pin (e.g. '0,1,2,3'). "
            "If omitted and CUDA_VISIBLE_DEVICES is unset, defaults to 0..gpus-1."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Compute dtype for V100 (float16 recommended).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to use (0-1).",
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=None,
        help="Optional CPU swap space in GB for KV cache spillover.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional max model length override.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable if the model requires remote code.",
    )
    args = parser.parse_args()

    model_path = args.model_path or os.environ.get("QWEN_MODEL_PATH")
    if not model_path:
        raise SystemExit("Missing model path. Use --model-path or set QWEN_MODEL_PATH.")

    if args.gpus not in (4, 8):
        print(
            f"Warning: gpus={args.gpus} is unusual for V100; expected 4 or 8.",
            file=sys.stderr,
        )

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.gpus))

    cmd = _build_command(
        model_path=model_path,
        served_model_name=args.served_model_name,
        gpus=args.gpus,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        gpu_mem_util=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )
    if args.swap_space:
        cmd += ["--swap-space", str(args.swap_space)]
    print("Launching vLLM server:")
    print(" ".join(cmd))
    os.execvpe(cmd[0], cmd, os.environ)


if __name__ == "__main__":
    main()
