"""No models or GPUs: real subprocess streaming and bounded CPU utility checks."""
import io
import os
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.common.parallel import ordered_bounded_map, resolve_prepare_workers
from ir_training.common.progress import Progress, fingerprint_file
from ir_training.pipeline.golden_training import _run_command
from ir_training.pipeline.preparation_cache import _safe_relative


def test_child_output_is_live_not_waited_until_exit(tmp_path, monkeypatch):
    seen = threading.Event()
    class Console(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if "first" in self.getvalue():
                seen.set()
            return result
    console = Console()
    monkeypatch.setattr(sys, "stdout", console)
    failures = []
    def run():
        try:
            _run_command([sys.executable, "-c", "import sys,time; print('first', end='\\r', flush=True); time.sleep(2); print('second'); print('stderr', file=sys.stderr)"], tmp_path / "stream.log", dict(os.environ))
        except BaseException as exc:
            failures.append(exc)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert seen.wait(timeout=10)
        assert worker.is_alive(), "The console must receive output before the child exits"
        assert b"first\r" in (tmp_path / "stream.log").read_bytes()
    finally:
        worker.join(timeout=10)
    assert not worker.is_alive() and not failures
    assert b"first\rsecond" in (tmp_path / "stream.log").read_bytes()
    assert "stderr" in console.getvalue()


def test_child_failure_retains_console_and_file_diagnostics(tmp_path, capsys):
    with pytest.raises(RuntimeError, match="exit 7"):
        _run_command([sys.executable, "-c", "import sys; print('diagnostic', file=sys.stderr); sys.exit(7)"], tmp_path / "failed.log", dict(os.environ))
    assert "diagnostic" in capsys.readouterr().out
    assert "diagnostic" in (tmp_path / "failed.log").read_text()


def test_progress_heartbeats_during_blocking_work(capsys):
    with Progress("blocked fixture", total=10, interval=0.02) as progress:
        progress.advance(2)
        time.sleep(0.07)
    output = capsys.readouterr().out
    assert "starting" in output and "running" in output and "finished" in output
    assert "ETA" in output and "2/10" in output


def test_hash_counts_nonblank_lines_and_final_unterminated_row(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{}\n\n  \n{}')
    report = fingerprint_file(path, count_rows=True)
    assert report["rows"] == 2
    import hashlib
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_cpu_worker_override_and_spawn_order():
    assert resolve_prepare_workers(3) == 3
    assert 1 <= resolve_prepare_workers() <= 16
    with pytest.raises(ValueError):
        resolve_prepare_workers(-1)
    assert list(ordered_bounded_map(abs, range(-35, 0), workers=2, batch_size=4)) == list(map(abs, range(-35, 0)))


@pytest.mark.parametrize("path", ["../outside", "/absolute.json", "prepared/../../outside.json", "prepared/code.py", "C:/outside.json"])
def test_cache_artifact_paths_cannot_escape_run(path):
    with pytest.raises(ValueError):
        _safe_relative(path)
