"""No CUDA hardware required: backend selection, restoration and fatal diagnostics."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.train import cuda_runtime as runtime
from ir_training.train.sft import _build_checked_causal_lm_trainer


def fake_torch():
    flags = {"cudnn": True, "flash": True, "mem_efficient": False, "math": True}
    backend = SimpleNamespace(**{f"{name}_sdp_enabled": (lambda key=name: flags[key]) for name in flags})
    backend.enable_cudnn_sdp = lambda value: flags.update(cudnn=value)
    return SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test-cuda"),
                           backends=SimpleNamespace(cuda=backend, cudnn=SimpleNamespace(version=lambda: 123)),
                           cuda=SimpleNamespace(is_available=lambda: False)), flags


@pytest.mark.parametrize("fail", [False, True])
def test_cudnn_only_disabled_and_all_state_restored(fail):
    torch, flags = fake_torch()
    original = dict(flags)
    try:
        with runtime.sdpa_policy({}, torch_module=torch) as report:
            assert flags == {**original, "cudnn": False}
            assert report["effective"] == flags
            assert runtime.active_attention_policy() == report
            if fail:
                raise RuntimeError("synthetic backward failure")
    except RuntimeError as exc:
        assert fail and str(exc) == "synthetic backward failure"
    assert flags == original
    assert runtime.active_attention_policy() == {}


def test_explicit_opt_out_does_not_change_backends():
    torch, flags = fake_torch()
    with runtime.sdpa_policy({"training": {"disable_cudnn_sdpa": False}}, torch_module=torch):
        assert flags["cudnn"] is True


@pytest.mark.parametrize("value", [None, "false", 0, 1])
def test_nonboolean_backend_setting_rejected(value):
    torch, flags = fake_torch()
    with (
        pytest.raises(ValueError, match="must be a boolean"),
        runtime.sdpa_policy({"training": {"disable_cudnn_sdpa": value}}, torch_module=torch),
    ):
        pytest.fail("bad option accepted")
    assert flags["cudnn"] is True


def test_older_torch_without_cudnn_sdp_api_is_supported():
    torch, _ = fake_torch()
    del torch.backends.cuda.cudnn_sdp_enabled
    del torch.backends.cuda.enable_cudnn_sdp
    with runtime.sdpa_policy({}, torch_module=torch) as report:
        assert report["effective"]["cudnn"] is None


def test_uncontrollable_enabled_cudnn_fails_closed():
    torch, _ = fake_torch()
    del torch.backends.cuda.enable_cudnn_sdp
    with (
        pytest.raises(RuntimeError, match="Cannot disable"),
        runtime.sdpa_policy({}, torch_module=torch),
    ):
        pytest.fail("unsafe backend accepted")


def test_memory_diagnostic_survives_failed_cuda_context_and_never_synchronizes():
    torch, _ = fake_torch()
    def failed_info(device):
        raise RuntimeError("cudaErrorContained")
    torch.cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 3,
        get_device_name=lambda device: "H100", mem_get_info=failed_info,
        **{key: lambda device: 42 for key in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved")})
    report = runtime.cuda_memory_snapshot(torch_module=torch)
    assert report["device"] == 3 and report["memory_allocated_bytes"] == 42
    assert "cudaErrorContained" in report["free_memory_error"]


def test_allocator_environment_is_reported_without_changing_settings(monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    expected = {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}
    torch, _ = fake_torch()
    with runtime.sdpa_policy({}, torch_module=torch) as report:
        assert report["allocator_environment"] == expected
    assert runtime.cuda_memory_snapshot(torch_module=torch)["allocator_environment"] == expected
    assert runtime.allocator_environment() == expected


def test_failed_training_step_reports_shapes_and_never_retries(monkeypatch, capsys):
    calls = []
    class BaseTrainer:
        def __init__(self):
            self.state = SimpleNamespace(global_step=395)
        def training_step(self, *args, **kwargs):
            calls.append(1)
            raise RuntimeError("mha_graph.execute failed")
    from ir_training.train import sft
    monkeypatch.setattr(sft, "cuda_memory_snapshot", lambda: {"rank": "1", "free_bytes": 1024})
    trainer = _build_checked_causal_lm_trainer(BaseTrainer)()
    inputs = {"input_ids": SimpleNamespace(shape=(2, 4096)), "labels": SimpleNamespace(shape=(2, 4096))}
    with pytest.raises(RuntimeError, match="Do not retry.*contained CUDA") as caught:
        trainer.training_step(None, inputs)
    assert calls == [1]
    assert str(caught.value.__cause__) == "mha_graph.execute failed"
    output = capsys.readouterr().out
    assert '"optimizer_step": 395' in output and "(2, 4096)" in output and '"rank": "1"' in output
