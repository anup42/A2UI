#!/usr/bin/env python3
"""Serve Muse Glimmer with DFlash and run A2UI dataset generation.

Run ``plan`` on any machine. ``servers`` needs a Linux H100 host with the
Muse-capable SGLang build; ``probe``, ``generate`` (Stage 3 only), and
``cycle`` (Stages 1, 2, 3) use its local endpoints.
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
    parser.add_argument("mode", choices=("plan", "servers", "probe", "generate", "cycle"))
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
    parser.add_argument("--run-id", help="Dataset run to generate or resume")
    parser.add_argument("--source-run-id", help="Read Stage 1/2 from this run and write Muse Stage 3 to --run-id")
    parser.add_argument("--max-genui-total", type=positive_int, help="Cap newly generated Stage 3 records")
    parser.add_argument("--total", "--max-generation-total", dest="total", type=positive_int,
                        help="Final query/response/GenUI count for cyclic generation")
    parser.add_argument("--cycle-size", "--generation-cycle-size", dest="cycle_size", type=positive_int,
                        help="Records per stage before advancing to the next stage")
    parser.add_argument("--queries-per-intent", type=positive_int,
                        help="Optional Stage 1 intent quota; raised to cover --total when needed")
    parser.add_argument("--stage1-batch-size", type=positive_int, default=8)
    parser.add_argument("--query-output-tokens", type=positive_int, default=8192)
    parser.add_argument("--response-output-tokens", type=positive_int, default=8192)
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
    if args.mode in {"generate", "cycle"} and not args.run_id:
        parser.error(f"{args.mode} requires --run-id")
    if args.mode == "cycle":
        if not args.total or not args.cycle_size:
            parser.error("cycle requires --total and --cycle-size")
        if args.source_run_id or args.max_genui_total:
            parser.error("cycle writes all stages to one run; omit --source-run-id and --max-genui-total")
    if (args.mode == "cycle" or args.total or args.cycle_size) and (
        args.context_length <= max(args.query_output_tokens, args.response_output_tokens) + args.safety_tokens + 500
    ):
        parser.error("context length must exceed Stage 1/2 output budgets + safety tokens + 500")
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
        "--language-model-only",  # Dataset stages supply text; save vision VRAM for KV cache.
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


def client_env(args: argparse.Namespace, stage: int = 3) -> dict[str, str]:
    prompt_cap = min(args.prompt_tokens, args.context_length - args.output_tokens - args.safety_tokens)
    parallelism = args.gpus // args.tp * args.requests_per_server
    output_budget = {1: args.query_output_tokens, 2: args.response_output_tokens, 3: args.output_tokens}[stage]
    env = os.environ.copy()
    env.update({
        "LOCAL_VLLM_ENDPOINTS": ",".join(endpoints(args)),
        "LOCAL_VLLM_SERVED_MODEL": args.served_model,
        "LOCAL_ALLOW_HTTP_ENDPOINT": "1",
        "LOCAL_STRICT_OFFLINE": "0",  # Local HTTP endpoints, not local Transformers loading.
        "LOCAL_VLLM_ENABLE_THINKING": "1",
        "LOCAL_MUSE_REQUIRE_REASONING": "1" if stage == 3 else "0",
        "LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS": "1",
        "LOCAL_VLLM_REASONING_STRENGTH": args.reasoning_strength,
        "LOCAL_VLLM_STRIP_THINKING": "1",
        "LOCAL_VLLM_USE_HF_GENERATION_CONFIG": "0",
        "LOCAL_VLLM_FORCE_SAMPLING_OVERRIDES": "1",
        "LOCAL_VLLM_TOP_P": "0.95",
        "LOCAL_VLLM_TOP_K": "64",
        "LOCAL_VLLM_MAX_OUTPUT_TOKENS": str(output_budget),
        "LOCAL_VLLM_MIN_RETRY_OUTPUT_TOKENS": str(output_budget),
        "LOCAL_VLLM_TIMEOUT_SECONDS": "900",
        "LOCAL_VLLM_RETRY_MAX_SECONDS": "180",
        "LOCAL_VLLM_RETRY_RESULT_ERRORS": "1",
        "LOCAL_VLLM_BATCH_PARALLELISM": str(parallelism),
        "LOCAL_VLLM_PARALLEL_REQUESTS": str(parallelism),
        "A2UI_STAGE1_PROMPT_FILE": "prompts/query_gen_gemma_v3_diverse_openings.md",
        "A2UI_STAGE1_INTENT_BATCH_SIZE": str(parallelism),
        "A2UI_STAGE1_INTENT_CYCLE_SIZE": "0",
        "A2UI_QUERY_MAX_TOKENS": str(args.query_output_tokens),
        "A2UI_RESPONSE_MAX_TOKENS": str(args.response_output_tokens),
        "A2UI_QUERY_TEMPERATURE": "1.0",
        "A2UI_RESPONSE_TEMPERATURES": "1.0",
        "DATASET_OFFLINE_MODE": "1",
        "STAGE2_KEEP_UNRESOLVED_MEDIA": "1",
        "STAGE2_REAL_ASSET_RETRY_ENABLED": "0",
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
    print(f"Muse Glimmer dataset: {args.gpus} H100 GPUs, {len(groups)} BF16 replicas, TP={args.tp}")
    print(f"DFlash draft={args.draft_model_path} block={args.dflash_block_size}; reasoning={args.reasoning_strength}")
    print(f"Context={args.context_length}, output={args.output_tokens}, prompt cap={client_env(args)['LOCAL_STAGE3_PROMPT_MAX_TOKENS']}")
    print(f"Client concurrency={len(groups) * args.requests_per_server} ({args.requests_per_server}/server)")
    if args.total and args.cycle_size:
        print(f"Cycle: Stage 1 -> 2 -> 3, {args.cycle_size} records per stage, total={args.total}")
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
        print("Servers are starting. Use 'cycle' or Stage 3-only 'generate' in another shell.", flush=True)
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
    command = stage_command(args, 3, args.max_genui_total or response_count)
    if args.source_run_id:
        command.extend(("--stage3_source_run_id", args.source_run_id))
    return command


def stage_command(args: argparse.Namespace, stage: int, remaining: int,
                  queries_per_intent: int | None = None) -> list[str]:
    parallelism = args.gpus // args.tp * args.requests_per_server
    command = [
        sys.executable, str(REPO_ROOT / "dataset" / "src" / "main.py"),
        "--stage", str(stage), "--model", MODEL_CONFIG, "--run_id", args.run_id,
        "--rate_limit_qps", "0",
    ]
    if stage == 1:
        if queries_per_intent is None:
            raise ValueError("Stage 1 needs queries_per_intent")
        command.extend(("--max_queries_total", str(remaining),
                        "--k_queries_per_intent", str(queries_per_intent),
                        "--stage1_batch_size", str(args.stage1_batch_size)))
    elif stage == 2:
        command.extend(("--max_responses_total", str(remaining),
                        "--stage2_batch_size", str(parallelism),
                        "--stage2_response_batch_size", "1"))
    elif stage == 3:
        command.extend(("--max_genui_total", str(remaining),
                        "--genui_batch_size", str(parallelism)))
    else:
        raise ValueError(f"Unsupported generation stage: {stage}")
    return command


def run_config() -> dict:
    dataset_root = REPO_ROOT / "dataset"
    config_path = dataset_root / "configs" / "run.yaml"
    if config_path.is_file():
        from yaml import safe_load
        return (safe_load(config_path.read_text(encoding="utf-8")) or {}).get("run", {})
    return {}


def run_layout(run_id: str) -> tuple[Path, Path]:
    dataset_root = REPO_ROOT / "dataset"
    config = run_config()
    output_root = Path(config.get("output_dir", "data/runs"))
    intents_path = Path(config.get("intents_file", "intents.info"))
    if not output_root.is_absolute():
        output_root = dataset_root / output_root
    if not intents_path.is_absolute():
        intents_path = dataset_root / intents_path
    return output_root / run_id, intents_path


def count_records(path: Path, require_muse: bool = False,
                  require_reasoning: bool = False) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            if require_muse:
                gen = row.get("gen")
                model = gen.get("model") if isinstance(gen, dict) else None
                if model != MODEL_ID:
                    raise RuntimeError(
                        f"{path}:{line_number} contains model={model!r}; choose a separate --run-id for Muse"
                    )
                validation = row.get("validation")
                source_rejected = (
                    row.get("record_status") == "quality_rejected"
                    and isinstance(validation, dict)
                    and validation.get("generation_attempted") is False
                )
                if require_reasoning and not source_rejected:
                    trace = row.get("reasoning_text")
                    source = gen.get("reasoning_source") if isinstance(gen, dict) else None
                    if not isinstance(trace, str) or not trace.strip() or source != "message.reasoning_content":
                        raise RuntimeError(
                            f"{path}:{line_number} has no saved Muse to=self reasoning; "
                            "Stage 3 cannot count this row for a reasoning-complete run"
                        )
            count += 1
    return count


def cycle(args: argparse.Namespace) -> None:
    config = run_config()
    if (int(config.get("n_responses_per_query", 1)) != 1
            or int(config.get("genui_candidates_per_response", 1)) != 1):
        raise RuntimeError("Cyclic one-to-one counts require one response per query and one GenUI candidate")
    run_dir, intents_path = run_layout(args.run_id)
    if not intents_path.is_file():
        raise RuntimeError(f"Missing Stage 1 intents file: {intents_path}")
    intent_count = sum(bool(line.strip()) for line in intents_path.read_text(encoding="utf-8").splitlines())
    if intent_count == 0:
        raise RuntimeError(f"No Stage 1 intents in {intents_path}")
    queries_per_intent = max((args.total + intent_count - 1) // intent_count,
                             args.queries_per_intent or 0)
    paths = {1: run_dir / "queries.jsonl", 2: run_dir / "responses.jsonl", 3: run_dir / "genui.jsonl"}
    counts = {
        stage: count_records(path, require_muse=True, require_reasoning=stage == 3)
        for stage, path in paths.items()
    }
    probe(args)  # Check the model, reasoning, and DFlash before any dataset write.
    print(f"Cyclic Muse generation: run={args.run_id} total={args.total} "
          f"cycle_size={args.cycle_size} queries_per_intent={queries_per_intent}", flush=True)
    while min(counts.values()) < args.total:
        completed_floor = min(counts.values())
        target = min(args.total, (completed_floor // args.cycle_size + 1) * args.cycle_size)
        print(f"Cycle target={target}: queries={counts[1]} responses={counts[2]} genui={counts[3]}", flush=True)
        for stage in (1, 2, 3):
            while counts[stage] < target:
                remaining = target - counts[stage]
                command = stage_command(args, stage, remaining, queries_per_intent)
                print(f"Stage {stage}: need {remaining}; {shlex.join(command)}", flush=True)
                exit_code = subprocess.call(command, cwd=REPO_ROOT, env=client_env(args, stage))
                if exit_code != 0:
                    raise RuntimeError(f"Stage {stage} exited with code {exit_code}; resume the same run after fixing it")
                updated = count_records(paths[stage], require_muse=True,
                                        require_reasoning=stage == 3)
                print(f"Stage {stage}: {updated}/{target} (+{updated - counts[stage]})", flush=True)
                if updated <= counts[stage]:
                    raise RuntimeError(f"Stage {stage} made no progress toward {target}; inspect the run before resuming")
                counts[stage] = updated
    print(f"Cyclic Muse generation complete: queries={counts[1]} responses={counts[2]} "
          f"genui={counts[3]}", flush=True)


def generate(args: argparse.Namespace) -> None:
    source_dir, _ = run_layout(args.source_run_id or args.run_id)
    output_dir, _ = run_layout(args.run_id)
    queries_path = source_dir / "queries.jsonl"
    responses_path = source_dir / "responses.jsonl"
    if not queries_path.is_file() or not responses_path.is_file():
        raise RuntimeError(f"Stage 3 needs queries.jsonl and responses.jsonl in {source_dir}")
    existing_output = output_dir / "genui.jsonl"
    count_records(existing_output, require_muse=True, require_reasoning=True)
    response_count = count_records(responses_path)
    if response_count == 0:
        raise RuntimeError(f"No Stage 2 responses in {responses_path}")
    probe(args)  # Fail before any dataset writes if a replica or parser is wrong.
    command = stage3_command(args, response_count)
    print(f"Generating A2UI Express Stage 3 for {args.run_id}: {shlex.join(command)}", flush=True)
    exit_code = subprocess.call(command, cwd=REPO_ROOT, env=client_env(args))
    if exit_code == 0:
        count_records(existing_output, require_muse=True, require_reasoning=True)
    raise SystemExit(exit_code)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.mode == "plan":
            plan(args)
        elif args.mode == "servers":
            serve(args)
        elif args.mode == "probe":
            probe(args)
        elif args.mode == "cycle":
            cycle(args)
        else:
            generate(args)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
