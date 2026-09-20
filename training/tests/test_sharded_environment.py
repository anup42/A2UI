"""Cheap fail-closed NVTX checks, with no GPU/model allocation in unit tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ir_training.train import sharded_environment as environment
from ir_training.train.sharded_contract import validate_sharded_runtime


@pytest.fixture
def nvtx_runtime(tmp_path, monkeypatch):
    prefix = tmp_path / "sharded-venv"
    module_path = prefix / "lib/python3.12/site-packages/nvtx/__init__.py"
    calls = []

    class Domain:
        def push_range(self, *, message, category):
            calls.append(("push", message, category))

        def pop_range(self):
            calls.append(("pop",))

    domain = Domain()

    def get_domain(name):
        assert name == "DeepSpeed"
        return domain

    nvtx = SimpleNamespace(__file__=str(module_path), enabled=lambda: True,
                           get_domain=get_domain)
    accelerator = SimpleNamespace(
        device_name=lambda: "cuda", supports_nvtx_domain=True,
        range_push=lambda message, *, domain: nvtx.get_domain(domain).push_range(
            message=message, category=None),
        range_pop=lambda *, domain: nvtx.get_domain(domain).pop_range(),
    )
    wrappers = SimpleNamespace(
        enable_nvtx=True,
        _range_push=lambda accelerator, message: accelerator.range_push(message, domain="DeepSpeed"),
        _range_pop=lambda accelerator: accelerator.range_pop(domain="DeepSpeed"),
    )
    modules = {
        "nvtx": nvtx,
        "deepspeed.accelerator": SimpleNamespace(get_accelerator=lambda: accelerator),
        "deepspeed.utils.nvtx": wrappers,
    }
    imported = []
    original_import = environment.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name in modules:
            imported.append(name)
            return modules[name]
        return original_import(name, *args, **kwargs)

    versions = {"nvtx": "0.2.15", "deepspeed": "0.19.7",
                "transformers": "5.16.1", "accelerate": "1.15.0"}
    distribution = SimpleNamespace(locate_file=lambda _: module_path)
    monkeypatch.setattr(environment.importlib, "import_module", import_module)
    monkeypatch.setattr(environment.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(environment.importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(environment.sys, "prefix", str(prefix))
    monkeypatch.setattr(environment.sys, "base_prefix", str(tmp_path / "system"))
    return SimpleNamespace(
        nvtx=nvtx, accelerator=accelerator, wrappers=wrappers, calls=calls,
        versions=versions, distribution=distribution, imported=imported,
        prefix=prefix, module_path=module_path,
    )


def test_probe_exercises_direct_and_actual_deepspeed_range_wrapper_contract(nvtx_runtime):
    report = environment.probe_nvtx_compatibility()
    assert report["passed"] is True
    assert report["venv_local"] is True
    assert report["module"] == str(nvtx_runtime.module_path.resolve())
    assert report["deepspeed_range_push_pop"] is True
    assert report["model_loaded"] is False
    assert report["training_executed"] is False
    assert nvtx_runtime.calls == [
        ("push", "a2ui-sharded-nvtx-direct-probe", None), ("pop",),
        ("push", "a2ui-sharded-deepspeed-nvtx-probe", None), ("pop",),
    ]
    assert nvtx_runtime.imported == ["nvtx", "deepspeed.accelerator", "deepspeed.utils.nvtx"]


def test_old_dummy_domain_signature_fails_even_with_good_version_metadata(nvtx_runtime):
    class OldDummyDomain:
        def push_range(self, attributes):
            raise AssertionError("Unsupported keyword must be rejected before this body")

    nvtx_runtime.nvtx.get_domain = lambda _: OldDummyDomain()
    with pytest.raises(RuntimeError, match="unexpected keyword argument 'message'") as error:
        environment.probe_nvtx_compatibility()
    assert "--ignore-installed --no-deps nvtx==0.2.15" in str(error.value)
    assert nvtx_runtime.imported == ["nvtx"]


def test_correct_global_version_still_must_shadow_inside_venv(nvtx_runtime, tmp_path):
    nvtx_runtime.nvtx.__file__ = str(tmp_path / "system/nvtx/__init__.py")
    with pytest.raises(RuntimeError, match="inherited from outside the active venv"):
        environment.probe_nvtx_compatibility()
    assert not nvtx_runtime.calls


def test_distribution_metadata_must_describe_the_imported_package(nvtx_runtime, tmp_path):
    nvtx_runtime.distribution.locate_file = lambda _: tmp_path / "other/nvtx/__init__.py"
    with pytest.raises(RuntimeError, match="does not match the selected distribution"):
        environment.probe_nvtx_compatibility()


def test_probe_rejects_system_interpreter(nvtx_runtime, monkeypatch):
    monkeypatch.setattr(environment.sys, "base_prefix", str(nvtx_runtime.prefix))
    with pytest.raises(RuntimeError, match="dedicated Python venv"):
        environment.probe_nvtx_compatibility()


@pytest.mark.parametrize("version", ["0.2.14", "0.2.16"])
def test_probe_rejects_unreviewed_nvtx_version(nvtx_runtime, version):
    nvtx_runtime.versions["nvtx"] = version
    with pytest.raises(RuntimeError, match="Expected nvtx 0.2.15"):
        environment.probe_nvtx_compatibility()
    assert not nvtx_runtime.calls


@pytest.mark.parametrize("disabled", ["nvtx", "deepspeed"])
def test_disabling_instrumentation_cannot_pass_probe(nvtx_runtime, disabled):
    if disabled == "nvtx":
        nvtx_runtime.nvtx.enabled = lambda: False
    else:
        nvtx_runtime.wrappers.enable_nvtx = False
    with pytest.raises(RuntimeError, match="disabled"):
        environment.probe_nvtx_compatibility()


@pytest.mark.parametrize("change", ["cpu", "legacy_range_path"])
def test_deepspeed_must_use_cuda_domain_path(nvtx_runtime, change):
    if change == "cpu":
        nvtx_runtime.accelerator.device_name = lambda: "cpu"
    else:
        nvtx_runtime.accelerator.supports_nvtx_domain = False
    with pytest.raises(RuntimeError, match="CUDA NVTX domain path"):
        environment.probe_nvtx_compatibility()


def test_actual_deepspeed_wrapper_failure_is_not_hidden_by_direct_probe(nvtx_runtime):
    def broken_push(*_args, **_kwargs):
        raise TypeError("DeepSpeed-specific incompatible range API")

    nvtx_runtime.wrappers._range_push = broken_push
    with pytest.raises(RuntimeError, match="DeepSpeed-specific incompatible range API"):
        environment.probe_nvtx_compatibility()
    assert nvtx_runtime.calls == [("push", "a2ui-sharded-nvtx-direct-probe", None), ("pop",)]


def test_version_gate_provides_scoped_nvtx_repair_before_torch(nvtx_runtime):
    nvtx_runtime.versions["nvtx"] = "0.2.14"
    with pytest.raises(RuntimeError, match="requires reviewed nvtx 0.2.15") as error:
        validate_sharded_runtime()
    assert "dedicated sharded venv" in str(error.value)
    assert "--ignore-installed --no-deps nvtx==0.2.15" in str(error.value)
    assert not nvtx_runtime.imported


def test_missing_nvtx_is_actionable(nvtx_runtime, monkeypatch):
    def version(name):
        if name == "nvtx":
            raise environment.importlib.metadata.PackageNotFoundError(name)
        return nvtx_runtime.versions[name]

    monkeypatch.setattr(environment.importlib.metadata, "version", version)
    with pytest.raises(RuntimeError, match="nvtx 0.2.15; it is not installed") as error:
        validate_sharded_runtime()
    assert "--ignore-installed --no-deps nvtx==0.2.15" in str(error.value)


@pytest.mark.parametrize(("world_size", "accumulation"), [(2, 16), (4, 8), (8, 4)])
def test_quick_check_builds_same_accumulation_configuration(world_size, accumulation, monkeypatch):
    monkeypatch.setattr("ir_training.train.sharded_contract.validate_sharded_runtime",
                        lambda: {"nvtx": {"passed": True}})
    report = environment.check_sharded_environment(world_size=world_size, effective_batch=32)
    assert report["accumulation_configuration"] == {
        "world_size": world_size, "per_device_train_batch_size": 1, "effective_batch": 32,
        "trainer": accumulation, "accelerate_plugin": accumulation, "deepspeed": accumulation,
    }
    assert report["distributed_engine_created"] is False
    assert report["memory_preflight_passed"] is False


@pytest.mark.parametrize(("world_size", "effective_batch"), [(1, 32), (3, 32), (True, 32),
                                                          (4, 31), (4, 0), (4, True)])
def test_bad_batch_arguments_fail_before_runtime_probe(world_size, effective_batch, monkeypatch):
    def forbidden_probe():
        raise AssertionError("Invalid static arguments must fail before environment imports")

    monkeypatch.setattr("ir_training.train.sharded_contract.validate_sharded_runtime", forbidden_probe)
    with pytest.raises(ValueError):
        environment.check_sharded_environment(world_size=world_size, effective_batch=effective_batch)


def _cli():
    path = Path(__file__).resolve().parents[1] / "scripts/check_sharded_training_env.py"
    spec = importlib.util.spec_from_file_location("check_sharded_training_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quick_cli_passes_arguments_and_does_not_claim_memory_fit(monkeypatch, capsys):
    cli = _cli()
    seen = []
    monkeypatch.setattr(cli, "check_sharded_environment", lambda **kwargs: seen.append(kwargs) or
                        {"passed": True, "model_loaded": False})
    assert cli.main(["--world-size", "4", "--effective-batch", "32"]) == 0
    assert seen == [{"world_size": 4, "effective_batch": 32}]
    output = capsys.readouterr().out
    assert json.loads(output[:output.index("Environment check passed")])["model_loaded"] is False
    assert "full model/memory preflight is still required" in output


def test_quick_cli_returns_nonzero_on_incompatible_environment(monkeypatch, capsys):
    cli = _cli()

    def failure(**_kwargs):
        raise RuntimeError("bad NVTX package")

    monkeypatch.setattr(cli, "check_sharded_environment", failure)
    assert cli.main([]) == 1
    output = capsys.readouterr()
    assert "bad NVTX package" in output.err
    assert not output.out


def test_nvtx_pin_is_exclusive_to_sharded_dependency_overlay():
    root = Path(__file__).resolve().parents[1]
    assert "nvtx==0.2.15" in (root / "requirements-full-parameter-qat-sharded.txt").read_text()
    for filename in ("requirements-full-parameter-qat.txt", "requirements-gemma4-qat.txt"):
        assert "nvtx" not in (root / filename).read_text().lower()


@pytest.mark.parametrize("backend", ["sharded", "ddp", "lora"])
def test_sft_environment_gate_precedes_seed_reads_and_never_runs_for_ddp_or_lora(backend, monkeypatch):
    from ir_training.common.config import load_yaml
    from ir_training.qat.full_model_contract import configure_full_qat
    from ir_training.train import sft

    root = Path(__file__).resolve().parents[1]
    base = load_yaml(root / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml")
    config = base if backend == "lora" else configure_full_qat(base, distributed_backend=backend)
    calls = []

    def environment_failure():
        calls.append("environment")
        raise RuntimeError("environment sentinel")

    def seed_sentinel(*_args, **_kwargs):
        calls.append("seed")
        raise RuntimeError("seed sentinel")

    monkeypatch.setattr("ir_training.train.sharded_contract.validate_sharded_runtime", environment_failure)
    monkeypatch.setattr(sft, "verify_configured_mobile_training_seed", seed_sentinel)
    monkeypatch.setattr(sft, "_stabilize_torch_runtime", lambda: None)
    # Invoke the real SFT entry body without its GPU-only attention decorator.
    with pytest.raises(RuntimeError, match="environment sentinel" if backend == "sharded" else "seed sentinel"):
        sft.train_sft.__wrapped__(config, preflight_only=True)
    assert calls == (["environment"] if backend == "sharded" else ["seed"])
