"""Exercise launch planning and GPU grouping with a file-only mock child."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "dataset" / "scripts"
BASH = str(Path("C:/Program Files/Git/bin/bash.exe")) if os.name == "nt" else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH or not Path(BASH).exists(), reason="bash unavailable")


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


def run(script, args=(), **env):
    return subprocess.run([BASH, shell_path(SCRIPTS / script), *args],
                          cwd=ROOT, env={**os.environ, **env}, text=True,
                          capture_output=True, timeout=25)


@pytest.mark.parametrize("tp,replicas", [(1, 8), (2, 4), (4, 2), (8, 1)])
def test_plan_is_cpu_only_and_selects_replica_layout(tp, replicas):
    result = run("run_gemma4_h100x8.sh", ["plan"], H100_TP_SIZE=str(tp))
    assert result.returncode == 0, result.stderr
    assert f"replicas={replicas} tensor_parallel_per_replica={tp}" in result.stdout
    assert f"client_parallelism={replicas * 32}" in result.stdout
    assert "assets_offline=1 keep_asset_references=1" in result.stdout
    assert "queries_per_prompt=8" in result.stdout
    assert "stage3_prompt_cap=11776 safety_reserve=512 repairs=1 final_regenerations=1 transport_attempts=2" in result.stdout


def test_plan_rejects_stale_endpoint_count():
    result = run("run_gemma4_h100x8.sh", ["plan"], H100_TP_SIZE="2", LOCAL_VLLM_ENDPOINTS="http://localhost:8000/v1/chat/completions")
    assert result.returncode == 2
    assert "Expected 4 endpoints" in result.stderr


def test_plan_recomputes_prompt_budget_for_larger_completion():
    result = run("run_gemma4_h100x8.sh", ["plan"], A2UI_GENUI_MAX_TOKENS="8192")
    assert result.returncode == 0, result.stderr
    assert "stage3_prompt_cap=7680" in result.stdout


def test_plan_rejects_output_that_exhausts_context():
    result = run("run_gemma4_h100x8.sh", ["plan"], VLLM_MAX_MODEL_LEN="4096")
    assert result.returncode == 2
    assert "Context must exceed" in result.stderr


@pytest.mark.parametrize("script", ["run_gemma4_h100x8.sh", "run_gemma4_vllm_replicas.sh", "run_gemma4_vllm_python.sh"])
def test_shell_syntax(script):
    result = subprocess.run([BASH, "-n", shell_path(SCRIPTS / script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_replica_groups_and_ports_use_two_gpus_each(tmp_path):
    child = tmp_path / "mock.sh"
    child.write_text('printf "%s:%s\\n" "$CUDA_VISIBLE_DEVICES" "$A2UI_VLLM_GPUS" > "$CAPTURE_DIR/$VLLM_PORT.txt"\n', encoding="utf8")
    endpoints = tmp_path / "endpoints.env"
    result = run("run_gemma4_vllm_replicas.sh", REPLICA_GPU_IDS="0,1,2,3,4,5,6,7",
                 REPLICA_TP_SIZE="2", A2UI_VLLM_SERVER_SCRIPT=shell_path(child),
                 REPLICA_LOG_DIR=shell_path(tmp_path / "logs"), REPLICA_ENV_FILE=shell_path(endpoints),
                 CAPTURE_DIR=shell_path(tmp_path), REPLICA_START_DELAY_SECONDS="0", VLLM_BASE_PORT="18000")
    assert result.returncode == 0, result.stderr
    for index in range(4):
        assert (tmp_path / f"{18000 + index}.txt").read_text().strip() == f"{2 * index},{2 * index + 1}:2"
    assert endpoints.read_text().count("/v1/chat/completions") == 4


@pytest.mark.parametrize("gpus,tp", [("0,0", "1"), ("0,1,2", "2")])
def test_invalid_gpu_assignment_fails_before_child(tmp_path, gpus, tp):
    child = tmp_path / "mock.sh"
    child.write_text('echo unexpected > "$CAPTURE_DIR/launched"\n', encoding="utf8")
    result = run("run_gemma4_vllm_replicas.sh", REPLICA_GPU_IDS=gpus, REPLICA_TP_SIZE=tp,
                 A2UI_VLLM_SERVER_SCRIPT=shell_path(child), REPLICA_LOG_DIR=shell_path(tmp_path / "logs"),
                 CAPTURE_DIR=shell_path(tmp_path))
    assert result.returncode == 2
    assert not (tmp_path / "launched").exists()


def test_replica_parent_allows_child_cleanup_to_finish(tmp_path):
    child = tmp_path / "mock.sh"
    child.write_text('trap \'sleep 3; echo stopped > "$CAPTURE_DIR/stopped"; exit 0\' TERM\n'
                     'while true; do sleep 0.1; done\n', encoding="utf8")
    result = run("run_gemma4_vllm_replicas.sh", REPLICA_GPU_IDS="0", REPLICA_TP_SIZE="1",
                 A2UI_VLLM_SERVER_SCRIPT=shell_path(child), REPLICA_LOG_DIR=shell_path(tmp_path / "logs"),
                 REPLICA_ENV_FILE=shell_path(tmp_path / "endpoints.env"), CAPTURE_DIR=shell_path(tmp_path),
                 REPLICA_START_DELAY_SECONDS="0.5", REPLICA_WAIT_FOREVER="0")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "stopped").read_text().strip() == "stopped"
