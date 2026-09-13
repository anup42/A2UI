"""CPU-only spawned workers with deterministic order and bounded input buffering."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import math
import multiprocessing
import os
from pathlib import Path


def resolve_prepare_workers(requested: int = 0) -> int:
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ValueError("--prepare-workers must be zero (auto) or a positive integer")
    if requested:
        return requested
    cpus = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    # Respect common container CPU quotas as well as scheduler affinity.
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        for directory in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
            try:
                quota = int((Path(directory) / "cpu.cfs_quota_us").read_text())
                period = int((Path(directory) / "cpu.cfs_period_us").read_text())
                if quota > 0 and period > 0:
                    cpus = min(cpus, max(1, math.ceil(quota / period)))
            except (OSError, ValueError):
                pass
    workers = min(16, max(1, cpus - 2 if cpus > 4 else cpus))
    try:
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        try:
            limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            available = min(available, max(0, limit - used))
        except (OSError, ValueError):
            pass
        workers = min(workers, max(1, available // (1024 ** 3)))
    except (AttributeError, OSError, ValueError):
        if os.name == "nt":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in
                              ("total_physical", "available_physical", "total_page", "available_page",
                               "total_virtual", "available_virtual", "available_extended")]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                workers = min(workers, max(1, status.available_physical // (1024 ** 3)))
    return workers


def _batch(function, values):
    return [function(value) for value in values]


def ordered_bounded_map(function, values, *, workers: int = 1, initializer=None, initargs=(), batch_size: int = 16):
    """At most 2 * workers batches are resident; duplicate decisions stay ordered."""
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    if workers == 1:
        if initializer is not None:
            initializer(*initargs)
        yield from map(function, values)
        return
    iterator, pending = iter(values), deque()
    # Never fork a parent that might already have imported CUDA or tokenizers.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initializer, initargs=initargs) as pool:
        try:
            while True:
                while len(pending) < workers * 2:
                    batch = list(islice(iterator, batch_size))
                    if not batch:
                        break
                    pending.append(pool.submit(_batch, function, batch))
                if not pending:
                    break
                yield from pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()
