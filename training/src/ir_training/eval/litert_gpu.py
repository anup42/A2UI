"""Bounded, target-blind LiteRT-LM GPU generation using the official Python API.

LiteRT-LM GPU is not CUDA DDP. The pinned Python API has no device ordinal
selector, so this runner intentionally owns ONE engine instead of launching
several workers which might all select the first OpenCL/WebGPU adapter.
"""
from __future__ import annotations

import csv
import importlib.metadata
import inspect
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.common.progress import Progress, fingerprint_file, log
from ir_training.generation_policy import closing_sentinel_end, sha256_text

PROTOCOL_VERSION = "a2ui_external_generation_v1"
RUNTIME_VERSION = "0.17.0"
EVENT_PREFIX = "A2UI_LITERT_EVENT "


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _nvidia_query(fields: str, *, processes: bool = False) -> list[list[str]]:
    flag = "--query-compute-apps" if processes else "--query-gpu"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"{flag}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("NVIDIA GPU evidence unavailable: nvidia-smi failed") from exc
    if result.returncode:
        raise RuntimeError(f"NVIDIA GPU evidence unavailable: {result.stderr.strip()}")
    return [list(map(str.strip, row)) for row in csv.reader(result.stdout.splitlines()) if row]


def nvidia_inventory() -> list[dict[str, str]]:
    rows = _nvidia_query("index,uuid,name,memory.total")
    if not rows or any(len(row) != 4 for row in rows):
        raise RuntimeError("No NVIDIA GPUs were reported by nvidia-smi")
    return [dict(zip(("index", "uuid", "name", "memory_total_mib"), row)) for row in rows]


def allowed_gpu_uuids() -> set[str] | None:
    """The orchestrator binds permitted physical GPUs; this is not affinity."""
    raw = os.environ.get("A2UI_LITERT_ALLOWED_GPU_UUIDS")
    if raw is None:
        return None
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("A2UI_LITERT_ALLOWED_GPU_UUIDS must be a JSON array of GPU UUIDs") from exc
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or not value.startswith("GPU-") for value in values)
            or len(values) != len(set(values))):
        raise ValueError("A2UI_LITERT_ALLOWED_GPU_UUIDS must contain unique, nonempty physical GPU UUIDs")
    return set(values)


def validate_gpu_allocation(evidence: dict[str, Any]) -> None:
    devices = evidence.get("devices") or []
    if not devices or any(not device.get("gpu_uuid") for device in devices):
        raise RuntimeError("GPU allocation evidence is missing device UUIDs")
    allowed = allowed_gpu_uuids()
    observed = {device["gpu_uuid"] for device in devices}
    if allowed is not None and not observed.issubset(allowed):
        raise RuntimeError(
            f"Native LiteRT-LM selected unallocated GPU(s) {sorted(observed - allowed)}; "
            f"allowed={sorted(allowed)}. CUDA_VISIBLE_DEVICES does not control this runtime's "
            "OpenCL/WebGPU selection; use GPU-isolated execution or an allocated native default GPU."
        )


def _read_self_status() -> str:
    return Path("/proc/self/status").read_text(encoding="utf-8")


def self_pid_identity() -> dict[str, Any]:
    """Read only this process's kernel-reported PID namespace chain.

    NSpid is ordered from the namespace mounting procfs to the innermost one.
    A container-mounted procfs can omit the host PID; no global process search,
    process-name matching, GPU-memory delta or environment-supplied PID is used.
    """
    owner = os.getpid()
    identity: dict[str, Any] = {"local_pid": owner, "candidate_pids": [owner],
                               "namespace_source": "/proc/self/status:NSpid", "namespace_status": "unavailable"}
    try:
        status = _read_self_status()
    except OSError as exc:
        identity["namespace_error"] = type(exc).__name__
        return identity
    lines = [line.partition(":")[2].split() for line in status.splitlines() if line.startswith("NSpid:")]
    if not lines:
        identity["namespace_status"] = "missing"
        return identity
    if len(lines) != 1 or not lines[0] or any(not value.isdecimal() for value in lines[0]):
        identity["namespace_status"] = "invalid"
        return identity
    chain = [int(value) for value in lines[0]]
    if any(value <= 0 for value in chain) or chain[-1] != owner:
        identity["namespace_status"] = "invalid_self_identity"
        return identity
    identity.update(candidate_pids=list(dict.fromkeys(chain)), namespace_status="verified", namespace_pids=chain)
    return identity


def gpu_process_evidence() -> dict[str, Any]:
    """Require an allocation for this process, not another training job's GPU.

    This observes allocation, not every individual delegated kernel. Engine GPU
    selection and native logs are retained separately; no all-op GPU claim is made.
    """
    identity = self_pid_identity()
    owner = identity["local_pid"]
    candidates = {str(pid) for pid in identity["candidate_pids"]}
    rows = _nvidia_query("pid,gpu_uuid,used_gpu_memory", processes=True)
    devices = []
    matched_pids = set()
    for row in rows:
        if len(row) != 3 or row[0] not in candidates:
            continue
        try:
            memory = float(row[2])
        except ValueError:
            continue
        if math.isfinite(memory) and memory > 0:
            devices.append({"gpu_uuid": row[1], "allocated_mib": memory})
            matched_pids.add(int(row[0]))
    if len(matched_pids) > 1:
        raise RuntimeError(
            f"Ambiguous NVIDIA PID-namespace evidence: multiple self PID candidates {sorted(matched_pids)} "
            "have GPU allocations. Refusing to attribute another process's allocation; use a host-PID-visible "
            "runtime container or ask the cluster administrator to align NVIDIA process reporting."
        )
    if not devices:
        raise RuntimeError(
            "LiteRT-LM GPU requested, but nvidia-smi reports no GPU allocation "
            f"for runner PID {owner} / verified self candidates {sorted(candidates)}. "
            "Refusing to label CPU/unknown execution as GPU. Check native runner.log and driver/OpenCL/Vulkan libraries. "
            "If NVIDIA reports host PIDs hidden from /proc/self/status NSpid, use an administrator-approved "
            "host-PID-visible runtime container; no unrelated PID or memory-delta fallback is allowed."
        )
    evidence = {"pid": owner, "devices": devices, "evidence": "nvidia-smi process GPU allocation",
                "pid_namespace_identity": identity, "nvidia_smi_pid": next(iter(matched_pids)),
                "all_operations_gpu_verified": False}
    validate_gpu_allocation(evidence)
    return evidence


def runtime_preflight(*, gpu_workers: int = 1, runtime: Any = None) -> dict[str, Any]:
    """Cheap prerequisite gate; it is NOT a model/format/kernel compatibility test."""
    if gpu_workers != 1:
        raise ValueError(
            "The pinned LiteRT-LM Python GPU API has no device-index selector. "
            "gpu_workers must be 1; CUDA_VISIBLE_DEVICES does not establish OpenCL/WebGPU "
            "device isolation. Training/HF evaluation can still use all GPUs."
        )
    if sys.platform != "linux":
        raise RuntimeError("The built-in NVIDIA-evidence GPU runner requires Linux")
    try:
        installed = importlib.metadata.version("litert-lm-api")
        if installed != RUNTIME_VERSION:
            raise RuntimeError(f"Install litert-lm-api=={RUNTIME_VERSION}; found {installed}")
        if runtime is None:
            import litert_lm as runtime
    except ImportError as exc:
        raise RuntimeError(f"Install litert-lm-api=={RUNTIME_VERSION} in the runtime environment") from exc
    required_engine = {"backend", "max_num_tokens", "cache_dir"}
    if not required_engine.issubset(inspect.signature(runtime.Engine).parameters):
        raise RuntimeError("Installed LiteRT-LM Engine API differs from the pinned runtime contract")
    required_session = {"apply_prompt_template", "sampler_config", "max_output_tokens"}
    if not required_session.issubset(inspect.signature(runtime.Engine.create_session).parameters):
        raise RuntimeError("LiteRT-LM create_session must support raw prefill and max_output_tokens")
    for name in ("tokenize", "detokenize", "create_session"):
        if not callable(getattr(runtime.Engine, name, None)):
            raise RuntimeError(f"LiteRT-LM Engine lacks {name}")  # noqa: TRY004 - dependency ABI error
    if not callable(getattr(getattr(runtime, "Backend", None), "GPU", None)):
        raise RuntimeError("LiteRT-LM runtime lacks Backend.GPU")  # noqa: TRY004 - dependency ABI error
    for name in ("run_prefill", "run_decode_async", "cancel_process"):
        if not callable(getattr(getattr(runtime, "Session", None), name, None)):
            raise RuntimeError(f"LiteRT-LM Session lacks {name}")  # noqa: TRY004 - dependency ABI error
    # The pinned Engine calls this same loader in __init__. Importing the Python
    # package alone does not establish that its native library/ABI can load.
    ffi = getattr(runtime, "_ffi", None)
    if ffi is None:
        from importlib import import_module

        ffi = import_module("litert_lm._ffi")
    try:
        native = ffi._get_lib()
        if not callable(getattr(native, "litert_lm_engine_settings_create", None)):
            raise RuntimeError("LiteRT-LM native engine symbol is unavailable")  # noqa: TRY004
    except Exception as exc:
        raise RuntimeError("Pinned LiteRT-LM native library could not load; repair runtime dependencies before training") from exc
    inventory = nvidia_inventory()
    allowed = allowed_gpu_uuids()
    if allowed is not None and not allowed.issubset({device["uuid"] for device in inventory}):
        raise RuntimeError("The job's allowed NVIDIA GPU UUIDs are absent from the runtime inventory")
    return {
        "status": "prerequisites_passed", "runtime_version": installed,
        "requested_backend": "gpu", "gpu_workers": 1, "gpu_inventory": inventory,
        "allowed_gpu_uuids": sorted(allowed) if allowed is not None else None,
        "model_kernel_tested": False, "device_selection": "native runtime chooses GPU",
        "multi_gpu_supported": False, "runner_cpu_fallback_allowed": False,
        "native_helpers_may_use_cpu": True,
        "native_library_loaded": True,
        "pid_namespace_identity": self_pid_identity(),
    }


def build_gpu_requests(
    rows: list[dict[str, Any]], *, model: Path, model_config: dict[str, Any],
    max_input_tokens: int, max_new_tokens: int, mtp_enabled: bool,
) -> list[dict[str, Any]]:
    from ir_training.eval.external_runner import (
        _assert_target_not_exposed,
        _request_from_golden_row,
    )
    from ir_training.models.registry import create_adapter

    adapter = create_adapter(model_config.get("model", model_config))
    tokenizer = adapter.load_tokenizer()
    if model_config.get("prepared_evaluation_contract") is not None:
        from ir_training.eval.prepared_contract import (
            verify_loaded_evaluation_tokenizer,
        )

        verify_loaded_evaluation_tokenizer(tokenizer, model_config.get("model", model_config),
                                            model_config["prepared_evaluation_contract"])
    requests = []
    with Progress("Format LiteRT-LM Golden requests", total=len(rows)) as progress:
        for row in rows:
            request = _request_from_golden_row(
                row, model=model, max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens, mtp_enabled=mtp_enabled,
            )
            formatted = adapter.format_example(row, tokenizer=tokenizer, include_assistant=False)
            ids = list(tokenizer(formatted, add_special_tokens=False)["input_ids"])
            if len(ids) > max_input_tokens:
                raise ValueError(f"Golden row {row.get('id')} exceeds max_input_tokens; never truncate Golden input")
            _assert_target_not_exposed(row, {"prompt": formatted})
            request.update(formatted_prompt=formatted, expected_input_ids=ids,
                           prompt_sha256=sha256_text(formatted))
            requests.append(request)
            progress.advance()
    return requests


def _event(phase: str, **fields: Any) -> None:
    print(EVENT_PREFIX + json.dumps({"phase": phase, **fields}), flush=True)


def prepare_native_prefill(engine: Any, request: dict[str, Any]) -> tuple[str, list[int], dict[str, Any]]:
    """Account for v0.17.0's automatic BOS even in raw-session mode.

    runtime/core/session_utils.cc::ApplyPromptTemplates prepends the native BOS
    on the first turn independently of apply_prompt_template. Supply the exact
    remaining text, and compare the resulting complete token sequence to HF.
    """
    prompt = request["formatted_prompt"]
    bos_id = engine.bos_token_id
    bos_string = engine.detokenize([bos_id]) if bos_id is not None and bos_id >= 0 else ""
    prefill = prompt
    if bos_string:
        if request["expected_input_ids"][0] != bos_id or not prompt.startswith(bos_string):
            raise RuntimeError(
                f"Native BOS parity failed for {request['id']}: the raw runtime inserts its BOS; "
                "the training prompt must contain the identical leading BOS. Check packaged metadata."
            )
        prefill = prompt[len(bos_string):]
        native_ids = [bos_id, *engine.tokenize(prefill)]
    else:
        native_ids = list(engine.tokenize(prefill))
    if native_ids != request["expected_input_ids"]:
        raise RuntimeError(
            f"Native tokenizer parity failed for {request['id']}: HF={len(request['expected_input_ids'])}, "
            f"LiteRT={len(native_ids)} tokens. Check packaged tokenizer/BOS/chat-template metadata; "
            "do not silently compare different prompts."
        )
    return prefill, native_ids, {"native_bos_inserted": bool(bos_string), "native_bos_token_id": bos_id,
                                 "prefill_text_sha256": sha256_text(prefill)}


def _validate_requests(rows: list[dict[str, Any]], model: Path) -> None:
    if not rows:
        raise ValueError("No LiteRT-LM generation requests supplied")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Request IDs must be nonempty and unique")
    for row in rows:
        if row.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Invalid request protocol version")
        if Path(row.get("model_path", "")).resolve() != model:
            raise ValueError("Request model path does not match the selected package")
        if any(key in row for key in ("completion", "completion_targets", "genui_json", "expected_output")):
            raise ValueError("Expected Golden completions must never be sent to the runtime")
        if row.get("do_sample") is not False or type(row.get("mtp_enabled")) is not bool:
            raise ValueError("Runner requires greedy decoding and an explicit Boolean MTP policy")
        prompt = row.get("formatted_prompt")
        if not isinstance(prompt, str) or not prompt or sha256_text(prompt) != row.get("prompt_sha256"):
            raise ValueError("A hash-bound, adapter-formatted prompt is required")
        tokens = row.get("expected_input_ids")
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Expected input token IDs are required for native tokenizer parity")
        for key in ("max_input_tokens", "max_new_tokens"):
            if type(row.get(key)) is not int:
                raise ValueError(f"{key} must be a positive integer")
            _positive(row[key], key)
        if len(tokens) > row["max_input_tokens"]:
            raise ValueError("Golden request exceeds max_input_tokens")
    if len({(row["max_input_tokens"], row["max_new_tokens"], row["mtp_enabled"]) for row in rows}) != 1:
        raise ValueError("All requests in a worker must use the same context/output/MTP settings")


def run_gpu_worker(
    *, model_path: str | Path, requests_path: str | Path, outputs_path: str | Path,
    cache_dir: str | Path | None = None, runtime: Any = None,
    evidence_reader: Callable[[], dict[str, Any]] = gpu_process_evidence,
) -> dict[str, Any]:
    """Called in an isolated child; parent enforces native-operation timeouts."""
    model = Path(model_path).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    output = Path(outputs_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite runtime results: {output}")
    rows = list(read_jsonl(requests_path))
    _validate_requests(rows, model)
    prerequisites = runtime_preflight(runtime=runtime)
    if runtime is None:
        import litert_lm as runtime
    first = rows[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    _event("engine_load", rows=len(rows), runtime_version=RUNTIME_VERSION, backend="gpu")
    with Progress("Load LiteRT-LM GPU engine", unit="stage"):
        engine = runtime.Engine(
            str(model), backend=runtime.Backend.GPU(),
            max_num_tokens=first["max_input_tokens"] + first["max_new_tokens"],
            cache_dir=str(Path(cache_dir).resolve()) if cache_dir else None,
            enable_speculative_decoding=first["mtp_enabled"],
        )
    started_all = time.monotonic()
    with engine, output.open("x", encoding="utf-8") as stream:
        if engine.backend.get_name() != "gpu":
            raise RuntimeError("LiteRT-LM engine did not retain the explicitly selected GPU backend")
        for index, request in enumerate(rows):
            _event("case_start", id=request["id"], completed=index, total=len(rows))
            prefill, native_ids, bos_evidence = prepare_native_prefill(engine, request)
            started = time.monotonic()
            chunks: list[str] = []
            stopped_on_envelope = False
            with Progress(f"LiteRT-LM case {index + 1}/{len(rows)} {request['id']}", unit="stage"):  # noqa: SIM117
                with engine.create_session(
                    apply_prompt_template=False,
                    sampler_config=runtime.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0, seed=42),
                    max_output_tokens=request["max_new_tokens"],
                ) as session:
                    session.run_prefill([prefill])
                    evidence = evidence_reader()
                    validate_gpu_allocation(evidence)
                    for chunk in session.run_decode_async():
                        if len(chunk.texts) != 1 or not isinstance(chunk.texts[0], str):
                            raise RuntimeError("Unexpected LiteRT-LM response structure")
                        chunks.append(chunk.texts[0])
                        if closing_sentinel_end("".join(chunks)) is not None:
                            session.cancel_process()
                            stopped_on_envelope = True
                            break
            raw = "".join(chunks)
            elapsed = time.monotonic() - started
            # Streaming responses do not expose exact generated token IDs in
            # 0.17.0. Never invent an exact EOS reason/token throughput count.
            result = {
                "id": request["id"], "generated_text": raw,
                "raw_completion": raw, "runtime_version": RUNTIME_VERSION,
                "requested_backend": "gpu", "engine_backend": engine.backend.get_name(),
                "gpu_evidence": evidence, "generation_seconds": elapsed,
                "input_tokens": len(native_ids), "prompt_sha256": request["prompt_sha256"],
                "input_ids_sha256": sha256_text(json.dumps(native_ids)),
                "tokenizer_parity_passed": True, "mtp_enabled": request["mtp_enabled"],
                "max_new_tokens": request["max_new_tokens"],
                "stop_reason": "closing_sentinel" if stopped_on_envelope else "native_stop_or_token_limit",
                "exact_output_token_ids_available": False,
                **bos_evidence,
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            overall = time.monotonic() - started_all
            log(f"LiteRT-LM completed {index + 1}/{len(rows)}; case={request['id']}; "
                f"{elapsed:.1f}s; ETA {overall / (index + 1) * (len(rows) - index - 1):.0f}s")
            _event("case_done", id=request["id"], completed=index + 1, total=len(rows))
    _event("finished", completed=len(rows), total=len(rows))
    return {**prerequisites, "status": "passed", "row_count": len(rows), "model_kernel_tested": True}


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_bounded_worker(
    command: list[str], *, log_path: Path, timeout_seconds: float = 7200,
    case_timeout_seconds: float = 600, load_timeout_seconds: float = 1800,
) -> None:
    """Tee native logs live; kill only our child on a hard load/case deadline."""
    for name, value in (("timeout_seconds", timeout_seconds), ("case_timeout_seconds", case_timeout_seconds),
                        ("load_timeout_seconds", load_timeout_seconds)):
        _positive(value, name)
    started = phase_started = time.monotonic()
    phase, current_id = "engine_load", None
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Inherit the evaluator's process group: the outer stage supervisor can
    # terminate the evaluator AND this native child. Do not create an orphanable
    # nested session. This worker owns native threads, not distributed children.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", env=env)
    lines: queue.Queue[str | None] = queue.Queue()

    def read_lines() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader = None
    try:
        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        with log_path.open("x", encoding="utf-8") as log_stream, Progress("LiteRT-LM GPU worker", unit="stage"):
            finished = False
            while not finished:
                now = time.monotonic()
                phase_limit = case_timeout_seconds if phase == "case_start" else load_timeout_seconds
                if now - started > timeout_seconds or now - phase_started > phase_limit:
                    raise TimeoutError(f"LiteRT-LM GPU timed out in {phase}, case={current_id}; see {log_path}. "
                                       "Partial outputs retained; no automatic retry or CPU fallback.")
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    finished = True
                    continue
                print(line, end="", flush=True)
                log_stream.write(line)
                log_stream.flush()
                if line.startswith(EVENT_PREFIX):
                    event = json.loads(line[len(EVENT_PREFIX):])
                    phase, current_id = event["phase"], event.get("id")
                    phase_started = time.monotonic()
            process.wait(timeout=max(0.1, timeout_seconds - (time.monotonic() - started)))
            if process.returncode:
                raise RuntimeError(f"LiteRT-LM GPU runner exited {process.returncode}; see {log_path}")
            if phase != "finished":
                raise RuntimeError("LiteRT-LM runner exited without its completion event")
    finally:
        _stop_child(process)
        if reader is not None and reader.ident is not None:
            reader.join(timeout=2)
        if process.stdout:
            process.stdout.close()


def run_litert_gpu_generation(
    *, model_path: str | Path, split_path: str | Path, output_dir: str | Path,
    model_config: dict[str, Any], max_input_tokens: int = 4096, max_new_tokens: int = 2048,
    required_rows: int | None = None, mtp_enabled: bool = False,
    cache_dir: str | Path | None = None, timeout_seconds: float = 7200,
    case_timeout_seconds: float = 600, load_timeout_seconds: float = 1800, gpu_workers: int = 1,
    runtime_python: str | Path | None = None,
) -> dict[str, Any]:
    from ir_training.eval.external_runner import (
        aggregate_external_runtime_metrics,
        merge_external_outputs,
    )
    from ir_training.eval.golden_set import load_fixed_golden_rows

    for name, value in (("max_input_tokens", max_input_tokens), ("max_new_tokens", max_new_tokens)):
        _positive(value, name)
    if gpu_workers != 1:
        raise ValueError("Built-in LiteRT-LM API supports one GPU engine, not CUDA device sharding")
    model, split, out = (Path(value).expanduser().resolve() for value in (model_path, split_path, output_dir))
    if not model.is_file():
        raise FileNotFoundError(model)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a fresh LiteRT-LM evaluation directory: {out}")
    rows = load_fixed_golden_rows(split, required_rows=required_rows, require_exact_rows=True)
    requests = build_gpu_requests(rows, model=model, model_config=model_config,
                                  max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
                                  mtp_enabled=mtp_enabled)
    out.mkdir(parents=True, exist_ok=True)
    requests_path, outputs_path = out / "runner_requests.jsonl", out / "runner_outputs.jsonl"
    write_jsonl(requests_path, requests)
    model_hash = fingerprint_file(model)["sha256"]
    cache_base = Path(cache_dir).expanduser().resolve() if cache_dir else out / "runtime_cache"
    # Variants commonly share the basename model.litertlm. Never assume native
    # compiler cache filenames bind complete package bytes or runtime versions.
    effective_cache = cache_base / RUNTIME_VERSION / model_hash
    effective_cache.mkdir(parents=True, exist_ok=True)
    worker_script = Path(__file__).resolve().parents[3] / "scripts" / "run_litertlm_gpu.py"
    command = [str(runtime_python or sys.executable), "-u", str(worker_script), "--worker", "--model", str(model),
               "--requests", str(requests_path), "--outputs", str(outputs_path),
               "--cache-dir", str(effective_cache)]
    runner_log = out / "runner.log"
    run_bounded_worker(command, log_path=runner_log, timeout_seconds=timeout_seconds,
                       case_timeout_seconds=case_timeout_seconds, load_timeout_seconds=load_timeout_seconds)
    output_rows = list(read_jsonl(outputs_path))
    for row in output_rows:
        if row.get("engine_backend") != "gpu" or not (row.get("gpu_evidence") or {}).get("devices"):
            raise RuntimeError("Runtime output is missing GPU execution evidence")
        if row.get("tokenizer_parity_passed") is not True:
            raise RuntimeError("Runtime output is missing native tokenizer parity")
        validate_gpu_allocation(row["gpu_evidence"])
    predictions = merge_external_outputs(rows, output_rows)
    predictions_path = out / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "model_path": str(model), "model_sha256": model_hash,
        "split_path": str(split), "row_count": len(rows), "runtime_version": RUNTIME_VERSION,
        "runtime_cache_base": str(cache_base), "runtime_cache_dir": str(effective_cache),
        "requested_backend": "gpu", "gpu_workers": 1, "multi_gpu_supported": False,
        "mtp_enabled": mtp_enabled, "max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens,
        "command": command, "requests_path": str(requests_path), "runner_outputs_path": str(outputs_path),
        "predictions_path": str(predictions_path), "runner_log_path": str(runner_log), "returncode": 0,
        "runtime_metrics": aggregate_external_runtime_metrics(output_rows),
        "gpu_execution": {
            "requested_backend": "gpu", "engine_backend": "gpu", "allocation_observed": True,
            "tokenizer_parity_passed": True, "all_operations_gpu_verified": False,
            "gpu_uuids": sorted({device["gpu_uuid"] for row in output_rows
                                 for device in row["gpu_evidence"]["devices"]}),
        },
        "evidence_scope": "Explicit GPU engine with per-process NVIDIA allocations; not all operations or GPU utilization certified",
    }
    manifest_path = out / "external_runner_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
