"""Rank-local, metadata-only diagnostics; never a passing preflight receipt.

Keep the maximum of observed PyTorch peaks: DeepSpeed can reset the underlying
counters during setup, so a final counter alone can lose earlier evidence.
Even this cannot recover transients reset between samples. Device-wide free
memory is sampled, not a continuous minimum. Subtracting a historical allocator
peak from a different-time free minimum is not a non-PyTorch memory estimate.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ir_training.common.progress import log
from ir_training.train.sharded_contract import deepspeed_config_sha256

_COUNTERS = (
    "baseline_allocated_bytes", "baseline_reserved_bytes", "peak_allocated_bytes",
    "peak_reserved_bytes", "device_total_bytes", "device_free_bytes_min",
)


def memory_gate_checks(memory: dict, local_rank: int | None = None) -> dict[str, bool]:
    """Name every predicate in the existing strict sharded memory gate."""
    device = memory.get("device")
    checks = {
        "cuda_local_rank_only": memory.get("cuda_local_rank_only") is True,
        "nccl_collectives_certified": memory.get("nccl_collectives_certified") is True,
        "cuda_device_name": isinstance(device, str) and re.fullmatch(r"cuda:[0-9]+", device) is not None,
        "local_rank_matches": local_rank is None or device == f"cuda:{local_rank}",
        "nonnegative_integer_counters": all(type(memory.get(key)) is int and memory[key] >= 0
                                             for key in _COUNTERS),
    }
    if not checks["nonnegative_integer_counters"]:
        return checks
    allocated, reserved = memory["peak_allocated_bytes"], memory["peak_reserved_bytes"]
    total, free = memory["device_total_bytes"], memory["device_free_bytes_min"]
    fraction = memory.get("peak_reserved_fraction")
    finite_fraction = type(fraction) in (int, float) and math.isfinite(fraction)
    checks.update(
        positive_total=total > 0,
        peak_allocated_at_least_baseline=allocated >= memory["baseline_allocated_bytes"],
        peak_reserved_at_least_baseline=reserved >= memory["baseline_reserved_bytes"],
        peak_reserved_at_least_allocated=reserved >= allocated,
        baseline_reserved_at_least_allocated=memory["baseline_reserved_bytes"] >= memory["baseline_allocated_bytes"],
        reserved_not_above_total=reserved <= total,
        free_not_above_total=free <= total,
        finite_reported_reserved_fraction=finite_fraction,
        unchanged_reserved_limit=memory.get("max_reserved_fraction") == 0.90,
        reported_reserved_fraction_matches=(total > 0 and finite_fraction
            and math.isclose(fraction, reserved / total, rel_tol=1e-12, abs_tol=1e-12)),
        peak_reserved_below_90_percent=total > 0 and reserved / total < 0.90,
        sampled_free_above_10_percent=total > 0 and free / total > 0.10,
    )
    return checks


class ShardedProbeDiagnostics:
    """Persist incremental CPU metadata even if a later gate/worker fails.

    No tensor values, snapshots, full gradients, or optimizer state_dict copies
    are requested. Catchable errors are reported; SIGKILL/host OOM/driver hangs
    can only leave the last atomically written sample, not a guaranteed final.
    """

    def __init__(self, output_dir: str | Path, *, device: Any, rank: int,
                 local_rank: int, world_size: int, accumulation_steps: int,
                 config: dict, sequence_length: int):
        self.device = device
        self.path = Path(output_dir).resolve() / f"sharded_preflight_diagnostics_rank{rank}.json"
        self.started = time.monotonic()
        self.samples: list[dict] = []
        self.partition_snapshots: list[dict] = []
        self.minimum_free: int | None = None
        self.events: list[dict] = []
        self.nccl_collectives_certified = False
        self.validation_attempt: dict | None = None
        self.identity = {
            "schema_version": 1, "kind": "sharded_preflight_diagnostics",
            "diagnostic_only": True, "certifies_training": False,
            "rank": rank, "local_rank": local_rank, "world_size": world_size,
            "pid": os.getpid(), "hostname": socket.gethostname(), "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gradient_accumulation_steps": accumulation_steps,
            "microbatch": 1, "effective_batch": accumulation_steps * world_size,
            "sequence_length": sequence_length,
            "config_sha256": deepspeed_config_sha256(config),
            "thresholds": {"reserved_fraction_strictly_below": 0.90,
                           "sampled_free_fraction_strictly_above": 0.10},
            "measurement_notes": [
                "Report retains maxima of observed allocator peaks; diagnostics never reset them.",
                "DeepSpeed may reset peak counters internally. Transients reset between samples remain unobserved.",
                "Free minimum is discrete synchronized device-wide sampling, not a continuous per-process minimum.",
                "Non-PyTorch estimate uses contemporaneous driver used minus current reserved, not allocator peaks.",
                "That estimate includes other processes, contexts, NCCL/driver memory and sampling races; it is not attribution.",
                "Partition/view logical byte counts can alias and must not be added as disjoint allocations.",
            ],
        }

    @contextmanager
    def capture_deepspeed_setup_events(self):
        """Capture the pinned optimizer's per-rank CPU/CUDA flatten decision.

        This observes ordinary log records only, without patching DeepSpeed or
        intercepting tensor operations. Enable INFO on its logger temporarily;
        restore its level and remove our handler even when Trainer raises.
        """
        diagnostics = self

        class SetupEvents(logging.Handler):
            def emit(self, record):
                match = re.fullmatch(
                    r"Flattening param group ([0-9]+) on (\S+) \((sufficient|insufficient) memory\)",
                    record.getMessage(),
                )
                if match:
                    diagnostics.events.append({
                        "phase": "deepspeed_parameter_flatten",
                        "elapsed_seconds": time.monotonic() - diagnostics.started,
                        "group_index": int(match[1]), "flatten_device": match[2],
                        "reported_memory": match[3], "source": "DeepSpeed INFO log",
                    })

        logger = logging.getLogger("DeepSpeed")
        original_level = logger.level
        handler = SetupEvents(level=logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(min(logger.getEffectiveLevel(), logging.INFO))
        try:
            yield
        finally:
            logger.removeHandler(handler)
            handler.close()
            logger.setLevel(original_level)

    def sample(self, phase: str, *, microsteps: int = 0, optimizer_steps: int = 0) -> dict:
        import torch

        torch.cuda.synchronize(self.device)
        allocated = int(torch.cuda.memory_allocated(self.device))
        reserved = int(torch.cuda.memory_reserved(self.device))
        free, driver_total = map(int, torch.cuda.mem_get_info(self.device))
        properties = torch.cuda.get_device_properties(self.device)
        stats = torch.cuda.memory_stats(self.device)
        sample = {
            "phase": phase, "elapsed_seconds": time.monotonic() - self.started,
            "microsteps": microsteps, "optimizer_steps": optimizer_steps,
            "allocated_bytes": allocated, "reserved_bytes": reserved,
            "reserved_unallocated_bytes": reserved - allocated,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
            "device_total_bytes": int(properties.total_memory),
            "driver_total_bytes": driver_total, "device_free_bytes": free,
            "non_pytorch_used_estimate_bytes": driver_total - free - reserved,
            "allocator_stats": {key: stats.get(key) for key in (
                "active_bytes.all.current", "inactive_split_bytes.all.current",
                "allocated_bytes.all.peak", "reserved_bytes.all.peak", "num_alloc_retries", "num_ooms",
            )},
        }
        self.identity["gpu_name"] = str(getattr(properties, "name", "unavailable"))
        self.identity["gpu_uuid"] = str(getattr(properties, "uuid", "unavailable"))
        if self.samples:
            previous = self.samples[-1]
            decreased = [key for key in ("peak_allocated_bytes", "peak_reserved_bytes")
                         if sample[key] < previous[key]]
            if decreased:
                self.events.append({
                    "phase": "allocator_peak_counter_decreased", "observed_at": phase,
                    "previous_phase": previous["phase"], "counters": decreased,
                    "note": "An intervening peak-counter reset is indicated; earlier observed peaks are retained.",
                })
        self.minimum_free = free if self.minimum_free is None else min(self.minimum_free, free)
        self.samples.append(sample)
        self.publish("running")
        return sample

    def memory_report(self, baseline_allocated: int, baseline_reserved: int) -> dict:
        if not self.samples:
            raise RuntimeError("Sharded diagnostics has no CUDA memory sample")
        final = self.samples[-1]
        peak_reserved = max(item["peak_reserved_bytes"] for item in self.samples)
        peak_allocated = max(item["peak_allocated_bytes"] for item in self.samples)
        total = final["device_total_bytes"]
        return {
            "device": str(self.device), "cuda_local_rank_only": True,
            "nccl_collectives_certified": self.nccl_collectives_certified,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "device_total_bytes": total, "device_free_bytes_min": self.minimum_free,
            "peak_reserved_fraction": peak_reserved / total if total > 0 else None,
            "device_free_fraction_min": self.minimum_free / total if total > 0 else None,
            "max_reserved_fraction": 0.90,
            "device_free_bytes_min_is_sampled": True,
            "final_allocated_bytes": final["allocated_bytes"],
            "final_reserved_bytes": final["reserved_bytes"],
            "final_free_bytes": final["device_free_bytes"],
        }

    def publish(self, status: str, *, memory: dict | None = None,
                error: BaseException | None = None, emit: bool = False) -> None:
        checks = memory_gate_checks(memory, self.identity["local_rank"]) if memory is not None else {}
        if status == "pending_validation":
            # A later exception sample can change peaks/free memory. Preserve
            # the exact inputs and predicates presented to the validator too.
            self.validation_attempt = {
                "memory": dict(memory), "checks": dict(checks),
                "failed_conditions": [key for key, passed in checks.items() if not passed],
            }
        record = {
            **self.identity, "status": status,
            "phase": self.samples[-1]["phase"] if self.samples else "before_first_sample",
            "memory": memory, "memory_gate_checks": checks,
            "failed_conditions": [key for key, passed in checks.items() if not passed],
            "samples": self.samples, "partition_snapshots": self.partition_snapshots,
            "events": self.events,
            "validation_attempt": self.validation_attempt,
            "error": {"type": type(error).__name__, "message": str(error)} if error else None,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        if emit:
            # Complete gate counters and exact conditions for every rank; the
            # full per-parameter partition inventory stays in the JSON file.
            log("Sharded preflight memory: " + json.dumps({
                "rank": self.identity["rank"], "local_rank": self.identity["local_rank"],
                "status": status, "memory": memory, "checks": checks,
                "failed_conditions": record["failed_conditions"], "error": record["error"],
                "diagnostic_path": str(self.path),
            }, sort_keys=True, allow_nan=False))

    def record_failure(self, error: BaseException, *, baseline_allocated: int,
                       baseline_reserved: int, microsteps: int, optimizer_steps: int) -> None:
        """Best effort only; never mask the exception which stopped preflight."""
        self.events.append({"phase": "failure_origin",
                            "last_sample_phase": self.samples[-1]["phase"] if self.samples else None,
                            "error": repr(error)})
        try:
            self.sample("exception", microsteps=microsteps, optimizer_steps=optimizer_steps)
        except Exception as diagnostic_error:  # noqa: BLE001 - broken CUDA context reporting only
            self.events.append({"phase": "exception_sample", "error": repr(diagnostic_error)})
        try:
            memory = self.memory_report(baseline_allocated, baseline_reserved) if self.samples else None
            self.publish("failed", memory=memory, error=error, emit=True)
        except Exception as diagnostic_error:  # noqa: BLE001 - preserve original CUDA/disk failure
            log(f"Sharded diagnostic write failed rank={self.identity['rank']}: {diagnostic_error!r}; original={error!r}")
