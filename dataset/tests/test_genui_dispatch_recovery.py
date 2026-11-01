"""Offline Windows dispatch recovery; no generation, prompt or fidelity bypass."""
from __future__ import annotations

import errno
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline import chatgpt_stage3 as genui
from pipeline import openai_stage3_batch as pilot
from pipeline import v11_regeneration_io as journal
from test_chatgpt_stage3 import FakeCodex
from test_openai_stage3_batch import SOURCE, inputs, row


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path / "prepared"
    source = inputs(tmp_path, [row(f"s-{i}", SOURCE.replace("Alder", f"Alder{i}")) for i in range(7)])
    pilot.prepare(source, output, samples=7, transport="chatgpt", pilot_profile="fidelity_v2", generation_workers=3)
    return output


def windows_error(code):
    exc = PermissionError(errno.EACCES, "private path or provider details")
    exc.winerror = code
    return exc


@pytest.mark.parametrize("code", [5, 32, 33])
def test_windows_sharing_retries_identical_atomic_bytes(tmp_path, monkeypatch, code):
    replace = journal.os.replace
    attempts = []
    def denied_first(source, target):
        attempts.append(source.read_bytes())
        if len(attempts) < 4:
            raise windows_error(code)
        return replace(source, target)
    monkeypatch.setattr(journal.os, "replace", denied_first)
    monkeypatch.setattr(genui.time, "sleep", lambda _: None)
    target = tmp_path / "generation.json"
    genui.save(target, {"completed": 156})
    assert journal.checked(target) == {"completed": 156}
    assert len(attempts) == 4 and len(set(attempts)) == 1


@pytest.mark.parametrize("code,expected_attempts", [(5, 10), (32, 10), (33, 10), (112, 1), (None, 1)])
def test_persistent_or_unrelated_io_failure_is_not_bypassed(tmp_path, monkeypatch, code, expected_attempts):
    calls = []
    def denied(*_):
        calls.append(1)
        raise windows_error(code)
    monkeypatch.setattr(journal.os, "replace", denied)
    monkeypatch.setattr(genui.time, "sleep", lambda _: None)
    with pytest.raises(OSError):
        genui.atomic_bytes(tmp_path / "progress.json", b"{}\n")
    assert len(calls) == expected_attempts
    assert not (tmp_path / "progress.json").exists()


def test_local_write_retry_never_reissues_generation(prepared, monkeypatch):
    replace = journal.os.replace
    denied = set()
    def occasionally_locked(source, target):
        if target.name in {"generation.json", "progress.json"} and target.name not in denied:
            denied.add(target.name)
            raise windows_error(5)
        return replace(source, target)
    monkeypatch.setattr(journal.os, "replace", occasionally_locked)
    client = FakeCodex()
    assert genui.run(prepared, client=client, workers=1)["status"] == "generated"
    assert client.calls == 7 and denied == {"generation.json", "progress.json"}
    assert genui.run(prepared, client=client, workers=1)["status"] == "generated"
    assert client.calls == 7


def stop_after_two(output, monkeypatch, *, failure=None):
    failure = failure or ValueError("offline stop")
    client = FakeCodex()
    original = genui.load
    def stopped(path):
        if client.calls == 2:
            raise failure
        return original(path)
    with monkeypatch.context() as patch:
        patch.setattr(genui, "load", stopped)
        with pytest.raises((ValueError, RuntimeError)):
            genui.run(output, client=client, workers=1)
    assert client.calls == 2
    return {p: p.read_bytes() for p in (output / "chatgpt_records").glob("*.json")}


def test_dispatch_diagnostic_is_saved_and_resume_reuses_results(prepared, monkeypatch):
    saved = stop_after_two(prepared, monkeypatch, failure=TypeError("PRIVATE SECRET"))
    state = journal.checked(prepared / "generation.json")
    assert state["dispatch_error"]["type"] == "TypeError"
    assert state["dispatch_error"]["frames"]
    assert "PRIVATE SECRET" not in json.dumps(state)
    assert "TypeError" in json.loads((prepared / "progress.json").read_text())["pause_reason"]
    client = FakeCodex()
    assert genui.run(prepared, client=client, workers=1)["status"] == "generated"
    assert client.calls == 5 and all(p.read_bytes() == contents for p, contents in saved.items())
    state = journal.checked(prepared / "generation.json")
    assert "dispatch_error" not in state and len(state["dispatch_errors"]) == 1


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    current = pilot.fingerprint()
    prior = {**current, **pilot.CHATGPT_DISPATCH_PREDECESSOR}
    output = tmp_path / "legacy"
    with monkeypatch.context() as patch:
        patch.setattr(pilot, "fingerprint", lambda: prior)
        pilot.prepare(inputs(tmp_path, [row(f"s-{i}", SOURCE.replace("Alder", f"Alder{i}")) for i in range(7)]),
                      output, samples=7, transport="chatgpt", pilot_profile="fidelity_v2")
        stop_after_two(output, patch)
    return output


def test_explicit_runtime_upgrade_preserves_contract_and_saved_outputs(legacy):
    before = {p: p.read_bytes() for p in legacy.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Bound code"):
        genui.run(legacy, client=FakeCodex())
    result = genui.upgrade_dispatch_runtime(legacy)
    assert result["preserved_outputs"] == 2 and result["remaining"] == 5
    assert all(p.read_bytes() == value for p, value in before.items())
    # The Batch API/default loader still rejects the old code contract.
    with pytest.raises(ValueError, match="Bound code"):
        pilot.load(legacy)
    assert genui.upgrade_dispatch_runtime(legacy)["status"] == "runtime_upgrade_already_registered"
    client = FakeCodex()
    assert genui.run(legacy, client=client)["status"] == "generated" and client.calls == 5
    genui.collect(legacy)
    assert pilot.CHATGPT_RUNTIME_RECEIPT in journal.checked(legacy / "chatgpt_collection.json")["files"]
    assert all(p.read_bytes() == value for p, value in before.items()
               if p.name not in {"generation.json", "progress.json", "regeneration.lock"})


@pytest.mark.parametrize("path", ["dataset/prompts/genui_gen_mobile_a2ui_express_v1.md",
                                 "dataset/src/pipeline/v11_regeneration_guard.py"])
def test_runtime_upgrade_cannot_accept_prompt_or_fidelity_drift(legacy, monkeypatch, path):
    current = pilot.fingerprint()
    monkeypatch.setattr(pilot, "fingerprint", lambda: {**current, path: "changed"})
    with pytest.raises(ValueError, match="limited to the reviewed"):
        genui.upgrade_dispatch_runtime(legacy)
    assert not (legacy / pilot.CHATGPT_RUNTIME_RECEIPT).exists()


def test_runtime_upgrade_cannot_rebind_requests_or_modified_saved_outputs(legacy):
    genui.upgrade_dispatch_runtime(legacy)
    target = next((legacy / "chatgpt_records").glob("*.json"))
    original = target.read_bytes()
    genui.atomic_bytes(target, b"modified")
    with pytest.raises(ValueError, match="Preserved ChatGPT output changed"):
        genui.load(legacy)
    genui.atomic_bytes(target, original)
    genui.atomic_bytes(legacy / "requests.jsonl", b"modified")
    with pytest.raises(ValueError, match="Prepared pilot file changed"):
        genui.load(legacy)


def test_registered_upgrade_does_not_allow_further_code_changes(legacy, monkeypatch):
    genui.upgrade_dispatch_runtime(legacy)
    current = pilot.fingerprint()
    path = next(iter(pilot.CHATGPT_DISPATCH_PREDECESSOR))
    monkeypatch.setattr(pilot, "fingerprint", lambda: {**current, path: "another change"})
    with pytest.raises(ValueError, match="receipt changed"):
        genui.load(legacy)


def test_runtime_upgrade_rejects_uncertain_attempts(legacy):
    state = journal.checked(legacy / "generation.json")
    next(iter(state["attempts"].values()))["status"] = "inflight"
    journal.save(legacy / "generation.json", state)
    with pytest.raises(ValueError, match="uncertain/in-flight"):
        genui.upgrade_dispatch_runtime(legacy)
    assert not (legacy / pilot.CHATGPT_RUNTIME_RECEIPT).exists()


def test_runtime_upgrade_rejects_unknown_predecessor(legacy):
    contract = journal.checked(legacy / "contract.json")
    contract["code_hashes"][next(iter(pilot.CHATGPT_DISPATCH_PREDECESSOR))] = "unknown old implementation"
    journal.save(legacy / "contract.json", contract)
    with pytest.raises(ValueError, match="limited to the reviewed"):
        genui.upgrade_dispatch_runtime(legacy)
