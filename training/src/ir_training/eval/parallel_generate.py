"""Isolated, GPU-sharded Golden generation without DDP or rank collectives.

The caller verifies the prepared contract once. Fresh child interpreters load
one model per assigned GPU, retain incremental outputs, and exit. Only a fully
validated, original-order merge is published for scoring. A dead worker cannot
leave peers waiting in a collective, and CUDA failures are never auto-retried.
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ir_training.common.jsonl import write_jsonl
from ir_training.common.progress import Progress, log
from ir_training.eval.generate import (
    build_prediction_record,
    generate_predictions,
    prediction_source_context_hash,
)
from ir_training.generation_policy import sha256_text


def read_rows_strict(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{number}")  # noqa: TRY004 - malformed file content
            rows.append(row)
    return rows


def shard_indices(count: int, workers: int) -> list[list[int]]:
    if count < 1 or workers < 1 or workers > count:
        raise ValueError("Golden sharding requires 1 <= workers <= case count")
    return [list(range(rank, count, workers)) for rank in range(workers)]


def resolve_generation_devices(selection: str = "auto", *, require_gpu: bool = False) -> list[str]:
    """Return scheduler-safe physical IDs/UUIDs, never unmask outside allocation."""
    if selection.lower() == "cpu":
        if require_gpu:
            raise ValueError("--devices cpu conflicts with --require-gpu")
        return []
    import torch

    from ir_training.train.gpu_profile import detect_cuda_devices, select_devices

    if not torch.cuda.is_available():
        if require_gpu or selection.lower() not in {"auto", "all"}:
            raise RuntimeError("CUDA Golden evaluation requested but no CUDA-visible GPU is available")
        log("Golden generation: no CUDA GPU detected; using explicit reported CPU fallback")
        return []
    return [device["launch_identifier"] for device in select_devices(detect_cuda_devices(), selection)]


def merge_shards(rows: list[dict], shards: list[list[int]], paths: list[Path], output_path: Path) -> int:
    """Fail on missing, duplicate, reordered or substituted cases before publish."""
    if len(shards) != len(paths):
        raise ValueError("Golden shard path count does not match assignments")
    by_index: dict[int, dict] = {}
    for indices, path in zip(shards, paths, strict=True):
        predictions = read_rows_strict(path)
        if len(predictions) != len(indices):
            raise ValueError(f"Incomplete Golden shard {path}: {len(predictions)} of {len(indices)} cases")
        for index, prediction in zip(indices, predictions, strict=True):
            if index in by_index or not 0 <= index < len(rows):
                raise ValueError(f"Invalid or duplicate Golden shard index: {index}")
            expected = build_prediction_record(rows[index], "")
            if (prediction.get("id") != expected.get("id")
                    or prediction.get("source_context_sha256") != expected["source_context_sha256"]
                    or prediction_source_context_hash(prediction) != expected["source_context_sha256"]
                    or sha256_text(str(prediction.get("response_text", ""))) != expected["response_text_sha256"]
                    or sha256_text(str(prediction.get("expected", ""))) != expected["expected_sha256"]):
                raise ValueError(f"Golden shard source identity mismatch at case index {index}")
            if not isinstance(prediction.get("generated_text"), str):
                raise ValueError(f"Golden shard has no generated text at case index {index}")  # noqa: TRY004 - malformed shard content
            by_index[index] = prediction
    if set(by_index) != set(range(len(rows))):
        raise ValueError("Golden shards do not cover the complete cohort")
    partial = output_path.with_name(output_path.name + ".merging")
    count = write_jsonl(partial, (by_index[index] for index in range(len(rows))))
    partial.replace(output_path)
    return count


def _stop_workers(processes: list[subprocess.Popen]) -> None:
    # All children are owned by this invocation; never touch unrelated jobs.
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def wait_for_workers(processes: list[subprocess.Popen], *, timeout_seconds: float, started: float) -> None:
    """Watch every child, including ranks after an earlier slow rank."""
    try:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Generation timeout must be a finite positive number")
        while True:
            codes = [process.poll() for process in processes]
            failures = [(rank, code) for rank, code in enumerate(codes) if code not in (None, 0)]
            if failures:
                raise RuntimeError(f"Golden GPU worker failed: {failures}; partial outputs retained, no scores published")
            if all(code == 0 for code in codes):
                return
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(f"Golden GPU generation exceeded {timeout_seconds:g}s; workers stopped, partial outputs retained")
            time.sleep(0.2)
    finally:
        _stop_workers(processes)


def generate_predictions_parallel(
    config: dict[str, Any], split_path: str | Path, output_path: str | Path,
    max_rows: int | None = None, *, max_input_tokens: int | None = None,
    max_new_tokens: int | None = None, adapter_checkpoint: str | Path | None = None,
    apply_qat: bool = False, devices: str = "auto", require_gpu: bool = False,
    timeout_seconds: float = 3600, performance_metrics: dict | None = None,
) -> int:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Launch Golden evaluation once, not with torchrun; it owns isolated GPU workers")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Generation timeout must be a finite positive number")
    if max_rows is not None and (type(max_rows) is not int or max_rows < 1):
        raise ValueError("max_rows must be a positive integer")
    for name, value in (("max_input_tokens", max_input_tokens), ("max_new_tokens", max_new_tokens)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    rows = read_rows_strict(split_path)
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        raise ValueError("Cannot generate an empty Golden cohort")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Golden generation requires unique, nonempty case IDs")
    selected = resolve_generation_devices(devices, require_gpu=require_gpu)
    worker_devices = selected[:len(rows)] or [None]
    shards = shard_indices(len(rows), len(worker_devices))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to reuse completed Golden predictions: {output}; use a fresh evaluation attempt")
    work = output.parent / (output.stem + "_workers")
    # Existing partials must never be confused with a new checkpoint's results.
    work.mkdir(exist_ok=False)
    from ir_training.train.gpu_profile import available_cpu_count

    threads = max(1, min(4, available_cpu_count() // len(worker_devices)))
    src = str(Path(__file__).resolve().parents[2])
    dataset_src = str(Path(__file__).resolve().parents[4] / "dataset" / "src")
    processes = []
    paths = []
    started = time.monotonic()
    log(f"Golden generation: {len(rows)} cases, {len(worker_devices)} isolated worker(s), "
        f"GPU IDs={selected}, CPU threads/worker={threads}, timeout={timeout_seconds:g}s")
    try:
        with Progress("Golden parallel generation (load + cases)", unit="stage"):
            for rank, (identifier, indices) in enumerate(zip(worker_devices, shards, strict=True)):
                directory = work / f"rank_{rank:03d}"
                directory.mkdir()
                shard = directory / "input.jsonl"
                write_jsonl(shard, (rows[index] for index in indices))
                prediction = directory / "predictions.jsonl"
                paths.append(prediction)
                job = {
                    "config": config, "split_path": str(shard), "output_path": str(prediction),
                    "max_rows": len(indices), "max_input_tokens": max_input_tokens,
                    "max_new_tokens": max_new_tokens,
                    "adapter_checkpoint": str(Path(adapter_checkpoint).resolve()) if adapter_checkpoint is not None else None,
                    "apply_qat": apply_qat, "gpu": identifier is not None,
                }
                job_path = directory / "job.json"
                job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
                environment = dict(os.environ)
                # Scope before interpreter/torch import, preserving UUID/MIG mappings.
                environment.update(CUDA_VISIBLE_DEVICES=identifier or "", A2UI_EVAL_WORKER=str(rank),
                    OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                    TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
                    PYTHONPATH=os.pathsep.join((src, dataset_src, environment.get("PYTHONPATH", ""))))
                for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK"):
                    environment.pop(key, None)
                processes.append(subprocess.Popen(
                    [sys.executable, "-u", "-m", "ir_training.eval.parallel_generate", "--worker-job", str(job_path)],
                    env=environment,
                ))
            wait_for_workers(processes, timeout_seconds=timeout_seconds, started=started)
        generated = merge_shards(rows, shards, paths, output)
    except BaseException:
        _stop_workers(processes)
        raise
    elapsed = time.monotonic() - started
    predictions = read_rows_strict(output)
    tokens = sum(row.get("runtime", {}).get("output_tokens", 0) for row in predictions)
    report = {
        "generation_runtime_wall_seconds": elapsed,
        "generation_runtime_wall_output_tokens_per_second": tokens / elapsed if elapsed else 0,
        "generation_runtime_worker_count": len(worker_devices),
        "generation_runtime_gpu_count": len(selected[:len(rows)]),
        "generation_runtime_case_count": generated,
    }
    (work / "execution.json").write_text(json.dumps({**report, "gpu_identifiers": selected,
        "shards": shards, "scope": "model load + generation + exact ordered merge; no scoring"}, indent=2), encoding="utf-8")
    if performance_metrics is not None:
        performance_metrics.update(report)
    log(f"Golden generation complete: {generated} cases, wall={elapsed:.1f}s, "
        f"aggregate tokens/s={report['generation_runtime_wall_output_tokens_per_second']:.2f}")
    return generated


def run_worker(job_path: str | Path) -> None:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    config = copy.deepcopy(job.pop("config"))
    gpu = job.pop("gpu")
    model = config.setdefault("model", {})
    if gpu:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Golden worker must see exactly its one assigned CUDA GPU")
        torch.cuda.set_device(0)
        model.update(device_map={"": 0}, inference_device="auto")
    else:
        model.update(device_map=None, inference_device="cpu")
    count = generate_predictions(config, **job)
    if count != job["max_rows"]:
        raise RuntimeError(f"Golden worker generated {count} cases, expected {job['max_rows']}")
    if gpu:
        predictions = read_rows_strict(job["output_path"])
        if any(not str(row.get("runtime", {}).get("inference_device", "")).startswith("cuda") for row in predictions):
            raise RuntimeError("GPU Golden worker unexpectedly ran inference on a non-CUDA device")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Internal isolated Golden generation worker")
    parser.add_argument("--worker-job", required=True)
    run_worker(parser.parse_args().worker_job)
