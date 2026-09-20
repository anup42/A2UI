"""Opt-in, real two-GPU integration gate for the ZeRO-2 Trainer lane."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_real_two_gpu_sharded_preflight_train_and_save(tmp_path):
    if os.environ.get("A2UI_RUN_SHARDED_GPU_TESTS") != "1":
        pytest.skip("set A2UI_RUN_SHARDED_GPU_TESTS=1 to run the real ZeRO-2 test")
    if not sys.platform.startswith("linux"):
        pytest.skip("the reviewed sharded runtime is Linux/CUDA only")

    # Under explicit opt-in, missing/broken dependencies are test failures.
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("the real sharded test requires at least two visible CUDA GPUs")

    worker = Path(__file__).with_name("sharded_gpu_worker.py")
    repo_training = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_training / "src") + (
        os.pathsep + existing_path if existing_path else ""
    )
    output_dir = tmp_path / "sharded-gpu"
    for phase in ("preflight", "train"):
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nproc_per_node=2", str(worker), "--output-dir", str(output_dir),
                   "--phase", phase]
        try:
            completed = subprocess.run(
                command, cwd=repo_training.parent, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(f"two-rank {phase} worker exceeded 300 seconds\n{exc.stdout or ''}")
        assert completed.returncode == 0, completed.stdout
    receipt = output_dir / "gpu_test_receipt.json"
    assert receipt.is_file(), completed.stdout
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["world_size"] == 2 and result["optimizer_steps"] == 2
    assert result["saved_matches_live_graph"] is True
    assert result["all_saved_tensors_fp32"] is True
    assert result["bounded_generation_verified"] is True
