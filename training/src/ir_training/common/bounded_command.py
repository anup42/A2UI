"""Stream subprocess output with a hard deadline, heartbeat and group teardown."""
from __future__ import annotations

import codecs
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from ir_training.common.config import repo_root
from ir_training.common.progress import log as progress_log


def _stop(process: subprocess.Popen) -> None:
    # POSIX process groups include torchrun ranks and nested evaluator workers.
    # Kill the group even if its parent exited but descendants hold stdout open.
    for hard in (False, True):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL if hard else signal.SIGTERM)
            elif process.poll() is None:
                (process.kill if hard else process.terminate)()
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
        if os.name != "posix":
            return
        # The parent can exit before its ranks; final group SIGKILL is harmless
        # when all children already exited and cannot target another session.


def run_bounded_command(command: list[str], log: Path, environment: dict[str, str], *,
                        timeout_seconds: float = 172800, progress_seconds: float = 10) -> None:
    """Hard total-stage timeout, not an idle-output timeout (long kernels are valid)."""
    if not all(math.isfinite(x) and x > 0 for x in (timeout_seconds, progress_seconds)):
        raise ValueError("Command timeout and progress interval must be positive and finite")
    log.parent.mkdir(parents=True, exist_ok=True)
    progress_log(f"Running {log.stem}; deadline {timeout_seconds:g}s; log: {log}")
    env = {**environment, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    events: queue.Queue = queue.Queue(maxsize=256)
    started = last_progress = time.monotonic()
    with log.open("w", encoding="utf-8", newline="") as stream:
        process = subprocess.Popen(command, cwd=repo_root(), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, bufsize=0, start_new_session=os.name == "posix")
        finished = threading.Event()

        def reader():
            try:
                while not finished.is_set():
                    block = process.stdout.read(4096)
                    while not finished.is_set():
                        try:
                            events.put(block, timeout=.2)
                            break
                        except queue.Full:
                            pass
                    if not block:
                        return
            except (OSError, ValueError):
                if not finished.is_set():
                    events.put(b"")

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                now = time.monotonic()
                if now - started >= timeout_seconds:
                    raise TimeoutError(f"Stage {log.stem} exceeded {timeout_seconds:g}s. Partial logs retained: {log}")
                if now - last_progress >= progress_seconds:
                    message = f"Stage {log.stem}: elapsed {now-started:.0f}s; process={process.pid}; deadline={timeout_seconds:g}s"
                    progress_log(message)
                    stream.write(message + "\n")
                    stream.flush()
                    last_progress = now
                try:
                    block = events.get(timeout=min(.5, max(.01, timeout_seconds - (now - started))))
                except queue.Empty:
                    continue
                text = decoder.decode(block, final=not block)
                stream.write(text)
                stream.flush()
                sys.stdout.write(text)
                sys.stdout.flush()
                if not block:
                    break
            code = process.wait(timeout=max(.01, timeout_seconds - (time.monotonic() - started)))
            if code:
                raise RuntimeError(f"Stage failed with exit {code}; inspect {log}")
        except BaseException:
            _stop(process)
            raise
        finally:
            finished.set()
            process.stdout.close()
            thread.join(timeout=2)
