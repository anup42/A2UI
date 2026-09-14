"""Small shared cache primitives; never delete a configured cache root."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import time

from ir_training.common.progress import log


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def hash_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def assert_no_links(path: Path) -> None:
    """Reject symlinks AND Windows junctions before any resolve/write/move."""
    for candidate in (Path(path), *Path(path).parents):
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError(f"Symlink/junction is not a safe cache path: {candidate}")


@contextmanager
def cache_lock(root: Path, key: str, *, timeout: float = 21600, interval: float = 10):
    """One OS-lock writer across ranks/processes, with bounded visible waiting.

    Kernel locks are released on process exit; an abandoned lock FILE is not
    a stale lock and must never be forcibly removed to bypass a live writer.
    Shared multi-host paths require a filesystem providing inter-host locks.
    """
    from filelock import FileLock, Timeout

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", key):
        raise ValueError("Unsafe cache lock key")
    if not all(math.isfinite(value) and value > 0 for value in (timeout, interval)):
        raise ValueError("Cache lock timeout and progress interval must be finite and positive")
    assert_no_links(Path(root))
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / ".locks"
    assert_no_links(directory)
    directory.mkdir(exist_ok=True)
    lock_path = directory / f"{key}.lock"
    assert_no_links(lock_path)
    if not lock_path.resolve().is_relative_to(root):
        raise ValueError("Cache lock path escapes its cache root")
    lock = FileLock(str(lock_path))
    started = time.monotonic()
    waited = False
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(
                f"CACHE WAIT timed out after {timeout:g}s: {lock_path}. "
                "Check the writer's logs and shared filesystem locking; do not delete a live lock."
            )
        try:
            lock.acquire(timeout=min(interval, remaining))
            break
        except Timeout:
            waited = True
            log(f"CACHE WAIT: another process is building/verifying {key[:20]}; "
                f"elapsed {time.monotonic() - started:.0f}s; lock={lock_path}")
    try:
        if waited:
            log(f"CACHE WAIT finished: acquired {key[:20]}; rechecking completed entry")
        yield
    finally:
        lock.release()
