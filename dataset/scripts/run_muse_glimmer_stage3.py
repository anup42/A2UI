#!/usr/bin/env python3
"""Serve Muse Glimmer with DFlash and run the existing A2UI Express Stage 3.

Run ``plan`` on any machine. ``servers`` needs a Linux H100 host with the
Muse-capable SGLang build; ``probe`` and ``generate`` use its local endpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "meta-models/Muse-Glimmer-30B"
DRAFT_ID = "meta-models/Muse-Glimmer-30B-assistant"
MODEL_CONFIG = "muse_glimmer_30b_sglang_reasoning_dflash"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "servers", "probe", "generate"))
    parser.add_argument("--gpus", type=int, choices=(4, 8), required=True)
    parser.add_argument("--gpu-ids", default=os.environ.get("MUSE_GPU_IDS"))
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), default=int(os.environ.get("MUSE_TP_SIZE", "1")))
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("MUSE_BASE_PORT", "30000")))
    parser.add_argument("--host", default=os.environ.get("MUSE_HOST", "127.0.0.1"))
    parser.add_argument("--endpoints", help="Comma-separated chat-completion URLs for a separate client host")
    parser.add_argument("--model-path", default=os.environ.get("MUSE_MODEL_PATH", MODEL_ID))
    parser.add_argument("--draft-model-path", default=os.environ.get("MUSE_DRAFT_MODEL_PATH", DRAFT_ID))
    parser.add_argument("--served-model", default=os.environ.get("MUSE_SERVED_MODEL", "muse-glimmer"))
    parser.add_argument("--context-length", type=positive_int, default=32768)
    parser.add_argument("--output-tokens", type=positive_int, default=12288)
    parser.add_argument("--prompt-tokens", type=positive_int, default=16000)
    parser.add_argument("--safety-tokens", type=int, default=512)
    parser.add_argument("--requests-per-server", type=positive_int, default=8)
    parser.add_argument("--max-running-requests", type=positive_int, default=16)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--dflash-block-size", type=positive_int, default=16)
    parser.add_argument("--reasoning-strength", choices=("low", "medium", "high", "xhigh"), default="high")
    parser.add_argument("--run-id", help="Existing run containing Stage 1 queries and Stage 2 responses")
    parser.add_argument("--source-run-id", help="Read Stage 1/2 from this run and write Muse Stage 3 to --run-id")
    parser.add_argument("--max-genui-total", type=positive_int, help="Cap newly generated Stage 3 records")
    parser.add_argument("--wait-seconds", type=positive_int, default=1800)
    parser.add_argument("--log-dir", type=Path, default=Path(tempfile.gettempdir()) / "a2ui_muse_glimmer")
    args = parser.parse_args(argv)
    if args.gpus % args.tp:
        parser.error("--tp must divide --gpus")
    if args.requests_per_server > args.max_running_requests:
        parser.error("--requests-per-server must be <= --max-running-requests")
    if not 0.5 <= args.mem_fraction_static <= 0.95:
        parser.error("--mem-fraction-static must be between 0.5 and 0.95")
    if args.safety_tokens < 0 or args.context_length <= args.output_tokens + args.safety_tokens + 500:
        parser.error("context length must exceed output tokens + safety tokens + 500")
    if not 1 <= args.base_port <= 65535 - (args.gpus // args.tp):
        parser.error("base port is out of range for the selected replicas")
    if args.mode == "generate" and not args.run_id:
        parser.error("generate requires --run-id")
    for label, value in (("--run-id", args.run_id), ("--source-run-id", args.source_run_id)):
        if value and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) or value in {".", ".."}):
            parser.error(f"{label} must be a single safe directory name")
    return args


def gpu_groups(args: argparse.Namespace) -> list[tuple[str, ...]]:
    ids = (
        [item for item in re.split(r"[,;\s]+", args.gpu_ids.strip()) if item]
        if args.gpu_ids else [str(index) for index in range(args.gpus)]
    )
    if len(ids) != args.gpus or len(set(ids)) != len(ids) or not all(item.isdigit() for item in ids):
        raise ValueError(f"Expected {args.gpus} distinct numeric GPU IDs; got {ids}")
    return [tuple(ids[index:index + args.tp]) for index in range(0, len(ids), args.tp)]


def endpoints(args: argparse.Namespace) -> list[str]:
    if args.endpoints:
        values = [item.rstrip("/") for item in re.split(r"[,;\s]+", args.endpoints) if item]
        if len(values) != args.gpus // args.tp or len(set(values)) != len(values):
            raise ValueError(f"Expected {args.gpus // args.tp} distinct endpoints")
        normalized = []
        for value in values:
            if not value.startswith(("http://", "https://")):
                raise ValueError(f"Endpoint must be an HTTP URL: {value}")
            if value.endswith("/chat/completions"):
                normalized.append(value)
            elif value.endswith("/v1"):
                normalized.append(value + "/chat/completions")
            else:
                normalized.append(value + "/v1/chat/completions")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Endpoints must be distinct after URL normalization")
        return normalized
    return [
        f"http://127.0.0.1:{args.base_port + index}/v1/chat/completions"
        for index in range(args.gpus // args.tp)
    ]


def server_command(args: argparse.Namespace, index: int) -> list[str]:
    return [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model_path,
        "--served-model-name", args.served_model,
        "--host", args.host,
        "--port", str(args.base_port + index),
        "--tp-size", str(args.tp),
        "--dtype", "bfloat16",
        "--language-model-only",  # Stage 3 supplies text; save vision VRAM for KV cache.
        "--context-length", str(args.context_length),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--max-running-requests", str(args.max_running_requests),
        "--reasoning-parser", "muse",
        "--tool-call-parser", "muse",
        "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", args.draft_model_path,
        "--speculative-dflash-block-size", str(args.dflash_block_size),
        "--enable-metrics",
    ]


def client_env(args: argparse.Namespace) -> dict[str, str]:
    prompt_cap = min(args.prompt_tokens, args.context_length - args.output_tokens - args.safety_tokens)
    parallelism = args.gpus // args.tp * args.requests_per_server
    env = os.environ.copy()
    env.update({
        "LOCAL_VLLM_ENDPOINTS": ",".join(endpoints(args)),
        "LOCAL_VLLM_SERVED_MODEL": args.served_model,
        "LOCAL_ALLOW_HTTP_ENDPOINT": "1",
        "LOCAL_STRICT_OFFLINE": "0",  # Local HTTP endpoints, not local Transformers loading.
        "LOCAL_VLLM_ENABLE_THINKING": "1",
        "LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS": "1",
        "LOCAL_VLLM_REASONING_STRENGTH": args.reasoning_strength,
        "LOCAL_VLLM_STRIP_THINKING": "1",
        "LOCAL_VLLM_USE_HF_GENERATION_CONFIG": "0",
        "LOCAL_VLLM_FORCE_SAMPLING_OVERRIDES": "1",
        "LOCAL_VLLM_TOP_P": "0.95",
        "LOCAL_VLLM_TOP_K": "64",
        "LOCAL_VLLM_MAX_OUTPUT_TOKENS": str(args.output_tokens),
        "LOCAL_VLLM_MIN_RETRY_OUTPUT_TOKENS": str(args.output_tokens),
        "LOCAL_VLLM_TIMEOUT_SECONDS": "900",
        "LOCAL_VLLM_RETRY_MAX_SECONDS": "180",
        "LOCAL_VLLM_RETRY_RESULT_ERRORS": "1",
        "LOCAL_VLLM_BATCH_PARALLELISM": str(parallelism),
        "LOCAL_VLLM_PARALLEL_REQUESTS": str(parallelism),
        "A2UI_GENUI_MAX_TOKENS": str(args.output_tokens),
        "A2UI_GENUI_PROMPT_MAX_TOKENS": str(prompt_cap),
        "LOCAL_STAGE3_PROMPT_MAX_TOKENS": str(prompt_cap),
        "STAGE3_CONTEXT_SAFETY_TOKENS": str(args.safety_tokens),
        "STAGE3_RESPECT_CONFIG_PROMPT_MAX": "1",
        "VLLM_MAX_MODEL_LEN": str(args.context_length),  # Stage 3's generic local-server context control.
        "A2UI_STAGE3_PROMPT_FILE": "prompts/genui_gen_mobile_a2ui_express_v1.md",
        "A2UI_GENUI_TEMPERATURE": "1.0",
        "A2UI_GENUI_REPAIR_TEMPERATURE": "1.0",
        "A2UI_GENUI_FINAL_REGEN_TEMPERATURE": "1.0",
        "A2UI_CALL_SLEEP_SECONDS": "0",
        "A2UI_MAX_ATTEMPTS": "2",
        "A2UI_MAX_REPAIR_ATTEMPTS": "1",
        "STAGE3_FINAL_REGEN_ATTEMPTS": "1",
    })
    return env


def plan(args: argparse.Namespace) -> None:
    groups = gpu_groups(args)
    print(f"Muse Glimmer Stage 3: {args.gpus} H100 GPUs, {len(groups)} BF16 replicas, TP={args.tp}")
    print(f"DFlash draft={args.draft_model_path} block={args.dflash_block_size}; reasoning={args.reasoning_strength}")
    print(f"Context={args.context_length}, output={args.output_tokens}, prompt cap={client_env(args)['LOCAL_STAGE3_PROMPT_MAX_TOKENS']}")
    print(f"Client concurrency={len(groups) * args.requests_per_server} ({args.requests_per_server}/server)")
    for index, group in enumerate(groups):
        command = server_command(args, index)
        print(f"GPU {','.join(group)}: CUDA_VISIBLE_DEVICES={','.join(group)} {shlex.join(['python', *command[1:]])}")
    print("Endpoints: " + ", ".join(endpoints(args)))


def validate_local_gpus(args: argparse.Namespace) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Server launch requires Linux with NVIDIA H100 GPUs")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("nvidia-smi must list the selected GPUs") from exc
    available = set(result.stdout.split())
    selected = {gpu for group in gpu_groups(args) for gpu in group}
    if not selected <= available:
        raise RuntimeError(f"Selected GPUs {sorted(selected - available)} were not reported by nvidia-smi")
    for path in (args.model_path, args.draft_model_path):
        if not (Path(path) / "config.json").is_file():
            raise RuntimeError(f"Download the model once and pass its directory with config.json: {path}")


def serve(args: argparse.Namespace) -> None:
    validate_local_gpus(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    logs = []
    try:
        for index, group in enumerate(gpu_groups(args)):
            log_path = args.log_dir / f"replica_{index}_gpu_{'_'.join(group)}.log"
            log_file = log_path.open("ab")
            logs.append(log_file)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(group)
            command = server_command(args, index)
            print(f"Starting replica {index} on GPU {env['CUDA_VISIBLE_DEVICES']}; log={log_path}", flush=True)
            processes.append(subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log_file,
                                              stderr=subprocess.STDOUT, start_new_session=True))
        print("Servers are starting. Use 'probe' in another shell, then 'generate'.", flush=True)
        while True:
            for index, process in enumerate(processes):
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(f"Replica {index} exited with code {exit_code}; inspect {args.log_dir}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopping Muse Glimmer replicas", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log_file in logs:
            log_file.close()


def read_json(url: str, payload: dict | None = None, timeout: float = 10) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"{url}: HTTP {exc.code}: {details}") from exc


def probe(args: argparse.Namespace) -> None:
    urls = endpoints(args)
    deadline = time.monotonic() + args.wait_seconds
    pending = set(urls)
    while pending:
        for url in tuple(pending):
            models_url = url.rsplit("/chat/completions", 1)[0] + "/models"
            try:
                ids = [entry.get("id") for entry in read_json(models_url, timeout=5).get("data", [])]
            except (OSError, ValueError, RuntimeError):
                continue
            if args.served_model not in ids:
                raise RuntimeError(f"{models_url} serves {ids}, expected {args.served_model}")
            pending.remove(url)
        if pending:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for {len(pending)} endpoints: {sorted(pending)}")
            time.sleep(3)

    for url in urls:
        base_url = url.rsplit("/v1/chat/completions", 1)[0]
        # Check the resolved SGLang settings, not only the command we intended
        # to launch: generate can also connect to separately managed servers.
        try:
            server = read_json(base_url + "/server_info", timeout=10)
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
            server = read_json(base_url + "/get_server_info", timeout=10)
        if (str(server.get("speculative_algorithm") or "").upper() != "DFLASH"
                or not server.get("speculative_draft_model_path")
                or server.get("reasoning_parser") != "muse"
                or server.get("tool_call_parser") != "muse"):
            raise RuntimeError(
                f"{base_url} must have DFLASH, a draft model, and Muse reasoning/tool parsers"
            )
        payload = read_json(url, {
            "model": args.served_model,
            "messages": [{"role": "user", "content": "Compute 19 times 23. Reply with the number only."}],
            "max_tokens": 2048,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "chat_template_kwargs": {"reasoning_strength": args.reasoning_strength},
        }, timeout=180)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content")
        if (choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip()
                or not isinstance(reasoning, str) or not reasoning.strip()
                or any(marker in content for marker in ("<|start|>", "<|message|>", "<|eom|>"))):
            raise RuntimeError(f"{url} did not return separate, complete Muse reasoning and final content")
        print(f"Ready: {url} (DFlash active; reasoning and final answer separated)", flush=True)


def stage3_command(args: argparse.Namespace, response_count: int) -> list[str]:
    command = [
        sys.executable, str(REPO_ROOT / "dataset" / "src" / "main.py"),
        "--stage", "3", "--model", MODEL_CONFIG, "--run_id", args.run_id,
        "--genui_batch_size", str(args.gpus // args.tp * args.requests_per_server),
        "--rate_limit_qps", "0",
        "--max_genui_total", str(args.max_genui_total or response_count),
    ]
    if args.source_run_id:
        command.extend(("--stage3_source_run_id", args.source_run_id))
    return command


def generate(args: argparse.Namespace) -> None:
    runs_root = REPO_ROOT / "dataset" / "data" / "runs"
    source_dir = runs_root / (args.source_run_id or args.run_id)
    output_dir = runs_root / args.run_id
    queries_path = source_dir / "queries.jsonl"
    responses_path = source_dir / "responses.jsonl"
    if not queries_path.is_file() or not responses_path.is_file():
        raise RuntimeError(f"Stage 3 needs queries.jsonl and responses.jsonl in {source_dir}")
    existing_output = output_dir / "genui.jsonl"
    if existing_output.exists():
        with existing_output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                model = (row.get("gen") or {}).get("model")
                if model != MODEL_ID:
                    raise RuntimeError(
                        f"{existing_output} contains model={model!r}; choose a separate --run-id for Muse"
                    )
    with responses_path.open("r", encoding="utf-8") as handle:
        response_count = sum(bool(line.strip()) for line in handle)
    if response_count == 0:
        raise RuntimeError(f"No Stage 2 responses in {responses_path}")
    probe(args)  # Fail before any dataset writes if a replica or parser is wrong.
    command = stage3_command(args, response_count)
    print(f"Generating A2UI Express Stage 3 for {args.run_id}: {shlex.join(command)}", flush=True)
    raise SystemExit(subprocess.call(command, cwd=REPO_ROOT, env=client_env(args)))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.mode == "plan":
            plan(args)
        elif args.mode == "servers":
            serve(args)
        elif args.mode == "probe":
            probe(args)
        else:
            generate(args)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
