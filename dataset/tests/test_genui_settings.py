"""Model/runtime selection and fresh dashboard preparation regression tests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline.genui_dashboard import Controller
from pipeline.genui_setup import settings
from pipeline.v11_regeneration_io import file_hash
from test_chatgpt_stage3 import (
    FakeCodex,
    codex,
    finish,
    genui,
    pilot,
    rpc_client,
    runner,
)
from test_openai_stage3_batch import inputs


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path / "genui"
    pilot.prepare(inputs(tmp_path), output, samples=1, transport="chatgpt", pilot_profile="fidelity_v2")
    return output


def options(prepared):
    return {"input_dir": str(prepared.parent / "input"), "output_dir": str(prepared.parent / "fresh"),
            "samples": 1, "seed": 123, "selection_profile": "random", "model": "gpt-6-luna",
            "reasoning_effort": "low", "reasoning_summary": "auto", "timeout": 90}


def test_latest_codex_is_selected_and_explicit_pin_is_respected(monkeypatch):
    monkeypatch.setattr(codex.os, "get_exec_path", lambda: ["old", "new", "other", "broken"])
    monkeypatch.setattr(codex.shutil, "which", lambda name, path=None: (path or name) + "/codex")
    versions = {"old/codex": "0.133.0", "new/codex": "0.166.0-alpha.10", "other/codex": "0.166.0-alpha.9"}
    def version(args, **kwargs):
        return SimpleNamespace(stdout="codex-cli " + versions.get(args[0], "unknown"))
    monkeypatch.setattr(codex.subprocess, "run", version)
    assert codex.resolve_codex() == ("new/codex", "codex-cli 0.166.0-alpha.10")
    assert codex.resolve_codex("old") == ("old/codex", "codex-cli 0.133.0")
    versions["other/codex"] = "0.166.0"
    assert codex.resolve_codex()[0] == "other/codex"
    with pytest.raises(codex.CodexError, match="Install Codex"):
        codex.resolve_codex("missing")


def test_current_codex_uses_verified_restricted_profile(tmp_path):
    client, calls = rpc_client(tmp_path)
    client.version = "codex-cli 0.166.0-alpha.5.1"
    original = client.rpc
    def rpc(method, params=None, **kwargs):
        result = original(method, params, **kwargs)
        if method == "thread/start":
            result["activePermissionProfile"] = {"id": "genui"}
        return result
    client.rpc = rpc
    finish(client)
    config = {"model": "gpt-6.1-sol", "reasoning_effort": "ultra", "max_output_tokens": 8192}
    assert client.generate(runner.request_body(config, "source", "policy"))["status"] == "completed"
    start = next(p for name, p in calls if name == "thread/start")
    turn = next(p for name, p in calls if name == "turn/start")
    assert start["permissions"] == turn["permissions"] == "genui"
    assert "sandbox" not in start and "sandboxPolicy" not in turn
    assert start["config"]["permissions.genui.filesystem"] == {str(tmp_path): "read"}
    assert start["config"]["permissions.genui.network.enabled"] is False


def test_permission_profile_not_applied_fails_before_model_turn(tmp_path):
    client, calls = rpc_client(tmp_path)
    client.version = "codex-cli 0.166.0"
    with pytest.raises(codex.CodexError, match="restricted permission profile"):
        client.generate(runner.request_body({"model": "gpt-6.1-sol", "reasoning_effort": "low", "max_output_tokens": 1024}, "source", "policy"))
    assert not any(name == "turn/start" for name, _ in calls)


def test_account_model_metadata_and_hidden_models(tmp_path):
    client, _ = rpc_client(tmp_path)
    client.rpc = lambda *a, **kw: {"data": [
        {"model": "new-model", "displayName": "New model", "isDefault": True,
         "defaultReasoningEffort": "max", "supportedReasoningEfforts": [{"reasoningEffort": "max"}, {"reasoningEffort": "ultra"}]},
        {"model": "hidden", "hidden": True}, {"model": "image-only", "inputModalities": ["image"]}], "nextCursor": None}
    assert client.models() == [{"model": "new-model", "display_name": "New model", "is_default": True,
                                "default_reasoning_effort": "max", "reasoning_efforts": ["max", "ultra"]}]


def test_prepare_from_dashboard_is_offline_new_run_and_preserves_archive(prepared):
    before = {str(p): file_hash(p) for p in prepared.rglob("*") if p.is_file()}
    controller = Controller(prepared, client_factory=FakeCodex)
    config = options(prepared)
    assert controller.model_catalog()["codex_version"] == FakeCodex.version
    assert len(controller.model_catalog()["models"]) == 2
    controller.action("prepare", config)
    controller.worker.join(timeout=60)
    assert not controller.worker.is_alive() and controller.error is None
    assert controller.evidence.directory == Path(config["output_dir"])
    contract = genui.load(controller.evidence.directory)
    assert contract["config"]["model"] == "gpt-6-luna" and contract["config"]["reasoning_effort"] == "low"
    assert contract["config"]["pilot_profile"] == "fidelity_v2" and contract["config"]["generation_timeout"] == 90
    assert controller.run_timeout == 90 and controller.snapshot()["status"] == "prepared"
    assert controller.client.calls == 0
    assert not (controller.evidence.directory / "generation.json").exists()
    assert before == {str(p): file_hash(p) for p in prepared.rglob("*") if p.is_file()}
    reopened = Controller(controller.evidence.directory, client_factory=FakeCodex)
    assert reopened.run_timeout == 90
    with pytest.raises(ValueError, match="bound by offline preparation"):
        controller.action("run", {"model": "other-model"})
    seen = []
    client = FakeCodex()
    original_generate = client.generate
    def generate(body, **kwargs):
        seen.append(kwargs["timeout"])
        return original_generate(body, **kwargs)
    client.generate = generate
    genui.run(controller.evidence.directory, client=client)
    assert seen == [90]
    controller.close()


@pytest.mark.parametrize("change", [{"samples": True}, {"samples": 0}, {"timeout": 29}, {"timeout": 3601},
    {"seed": -1}, {"reasoning_summary": "raw"}, {"selection_profile": "failure_mix_v1"},
    {"pilot_profile": "legacy"}, {"reasoning_effort": "unsupported"}])
def test_invalid_settings_fail_before_writing(prepared, change):
    config = {**options(prepared), **change}
    with pytest.raises(ValueError):
        settings(config, prepared)
    assert not Path(config["output_dir"]).exists()


def test_existing_and_nested_outputs_and_unavailable_models_rejected(prepared):
    controller = Controller(prepared, client_factory=FakeCodex)
    config = options(prepared)
    for path in (prepared, prepared / "nested", prepared.parent / "input/child"):
        with pytest.raises(ValueError):
            controller.action("prepare", {**config, "output_dir": str(path)})
    with pytest.raises(codex.CodexError, match="unavailable"):
        controller.action("prepare", {**config, "model": "not-advertised"})
    with pytest.raises(codex.CodexError, match="unavailable"):
        controller.action("prepare", {**config, "reasoning_effort": "ultra"})
    assert not Path(config["output_dir"]).exists() and controller.worker is None


def test_extended_effort_only_changes_chatgpt_preparation(prepared):
    new = prepared.parent / "max-run"
    pilot.prepare(prepared.parent / "input", new, samples=1, transport="chatgpt", reasoning_effort="ultra")
    assert genui.load(new)["config"]["reasoning_effort"] == "ultra"
    with pytest.raises(ValueError, match="reasoning effort"):
        pilot.prepare(prepared.parent / "input", prepared.parent / "batch", samples=1, reasoning_effort="ultra")
    assert not (prepared.parent / "batch").exists()


def test_preparation_failure_retains_current_run(prepared, monkeypatch):
    controller = Controller(prepared, client_factory=FakeCodex)
    def fail(**kwargs):
        raise ValueError("Source integrity failed")
    monkeypatch.setattr(pilot, "prepare", fail)
    controller.action("prepare", options(prepared))
    controller.worker.join(timeout=5)
    assert controller.evidence.directory == prepared
    assert "Source integrity failed" in controller.error and controller.client.calls == 0
