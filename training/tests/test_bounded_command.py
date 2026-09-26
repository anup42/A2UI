import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.common.bounded_command import run_bounded_command


def test_subprocess_streams_partial_lines(tmp_path):
    path = tmp_path / "log.txt"
    run_bounded_command([sys.executable, "-c", "print('hello', end='')"], path, dict(os.environ), timeout_seconds=10)
    assert path.read_text() == "hello"


def test_silent_process_has_bounded_timeout_and_logs_progress(tmp_path):
    path = tmp_path / "log.txt"
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="Partial logs retained"):
        run_bounded_command([sys.executable, "-c", "import time; time.sleep(60)"], path, dict(os.environ),
                            timeout_seconds=.8, progress_seconds=.2)
    assert time.monotonic() - start < 10
    assert "elapsed" in path.read_text()


def test_nested_command_can_suppress_heartbeat_without_hiding_child_output(tmp_path):
    path = tmp_path / "nested.log"
    run_bounded_command(
        [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(.15); print('done')"],
        path,
        dict(os.environ),
        timeout_seconds=2,
        progress_seconds=.02,
        emit_heartbeat=False,
    )
    assert path.read_text() == "start\ndone\n"


def test_nonzero_exit_fails(tmp_path):
    with pytest.raises(RuntimeError, match="exit 3"):
        run_bounded_command([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "error.log", dict(os.environ))
