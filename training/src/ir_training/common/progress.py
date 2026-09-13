"""Flushed console progress and heartbeats during blocking preparation steps."""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
from pathlib import Path
import threading
import time


_PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with _PRINT_LOCK:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


class Progress:
    def __init__(self, label: str, *, total: int | None = None, unit: str = "rows", interval: float = 10):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Progress interval must be positive")
        self.label, self.total, self.unit, self.interval = label, total, unit, interval
        self.count = 0
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread = None

    def advance(self, count: int = 1) -> None:
        self.count += count

    def _emit(self, status: str) -> None:
        elapsed = max(time.monotonic() - self.started, 0.001)
        if self.unit == "stage":
            log(f"{self.label}: {status}; elapsed {elapsed:.1f}s")
            return
        rate = self.count / elapsed
        detail = f"{self.count:,}" + (f"/{self.total:,}" if self.total is not None else "")
        eta = ""
        if self.total is not None and rate:
            eta = f"; ETA {max(0, self.total - self.count) / rate:.0f}s"
        log(f"{self.label}: {status}; {detail} {self.unit}; {rate:.1f} {self.unit}/s; elapsed {elapsed:.1f}s{eta}")

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            self._emit("running")

    def __enter__(self):
        self.started = time.monotonic()
        self._emit("starting")
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, *_):
        self._stop.set()
        self._thread.join(timeout=2)
        self._emit("failed" if exc_type else "finished")


def fingerprint_file(path: Path, *, count_rows: bool = False, interval: float = 10) -> dict:
    """Hash actual bytes; optionally count nonblank JSONL lines in the same pass."""
    digest, rows = hashlib.sha256(), 0
    with Progress(f"Hash {path}", total=path.stat().st_size, unit="bytes", interval=interval) as progress:
        with path.open("rb") as stream:
            blocks = stream if count_rows else iter(lambda: stream.read(8 * 1024 * 1024), b"")
            for block in blocks:
                digest.update(block)
                progress.advance(len(block))
                if count_rows and block.strip():
                    rows += 1
    return {"sha256": digest.hexdigest(), "rows": rows if count_rows else None}
