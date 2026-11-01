"""Offline GenUI transport, crash recovery, strict replay and dashboard checks."""
from __future__ import annotations

import http.client
import io
import json
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline import chatgpt_stage3 as genui
from pipeline import genui_codex_client as codex
from pipeline import openai_stage3_batch as pilot
from pipeline import openai_stage3_batch_api as batch
from pipeline import openai_stage3_batch_runner as runner
from pipeline.genui_dashboard import ASSETS, Controller, Evidence, make_server
from pipeline.v11_regeneration_io import checked, file_hash, save
from test_openai_stage3_batch import GOOD, SOURCE, inputs


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "client", lambda: pytest.fail("No API client permitted"))
    output = tmp_path / "genui"
    pilot.prepare(inputs(tmp_path), output, samples=1, transport="chatgpt", pilot_profile="fidelity_v2")
    return output


class FakeCodex:
    version = "codex-cli test"

    def __init__(self, auth_home=None, text=GOOD):
        self.calls, self.text, self.closed = 0, text, False

    def account(self):
        return {"signed_in": True, "type": "chatgpt", "plan": "test"}

    def check_model(self, model, effort):
        assert model == "gpt-6.1-sol" and effort == "medium"

    def limits(self):
        return {"primary": {"usedPercent": 20, "windowDurationMins": 300}}

    def login(self, device=False):
        return {"type": "chatgpt", "authUrl": "https://auth.openai.com/authorize?state=test"}

    def generate(self, body, *, on_thread, **kwargs):
        self.calls += 1
        on_thread("thread-" + str(self.calls))
        return {"status": "completed", "model": body["model"], "thread_id": "thread-" + str(self.calls),
                "turn_id": "turn-" + str(self.calls), "text": self.text, "reasoning_summaries": ["Brief summary"],
                "usage": {"inputTokens": 100, "outputTokens": 40, "cachedInputTokens": 20, "reasoningOutputTokens": 10}}

    def close(self):
        self.closed = True


def test_chatgpt_preparation_preserves_batch_defaults_and_no_price_claim(prepared, tmp_path):
    contract = genui.load(prepared)
    assert contract["config"]["transport"] == "chatgpt"
    estimate = json.loads((prepared / "estimate.json").read_text())
    assert estimate["billing"] == "chatgpt_subscription" and estimate["cost_usd"] is None
    assert estimate["estimated_input_tokens"] > 0
    assert "budget_estimate_usd" not in estimate and "prompt_caching" not in estimate
    with pytest.raises(ValueError, match="cannot be submitted"):
        batch.submit(prepared, max_estimated_cost_usd=100)
    for operation in (batch.status, batch.collect):
        with pytest.raises(ValueError, match="not an API Batch"):
            operation(prepared)
    assert not (prepared / "submission.json").exists()
    original = tmp_path / "batch"
    pilot.prepare(tmp_path / "input", original, samples=1)
    assert "transport" not in pilot.load(original)["config"]
    with pytest.raises(ValueError, match="cannot be repurposed"):
        genui.run(original, client=FakeCodex())
    assert not (original / "regeneration.lock").exists()


def test_actual_canonical_replay_and_sealed_collection(prepared):
    client = FakeCodex()
    result = genui.run(prepared, client=client)
    assert result["status"] == "generated" and client.calls == 1
    assert genui.run(prepared, client=client)["status"] == "generated" and client.calls == 1
    report = genui.collect(prepared)
    assert report["accepted"] == 1 and report["training_input_ready"] is False
    assert report["transport"] == "chatgpt" and "batch_status" not in report
    assert report["usage"]["input_tokens"] == 100 and report["usage"]["cost_usd"] is None
    accepted = runner.read_rows(prepared / "pilot_accepted.jsonl")[0]
    assert accepted["response_text"] == SOURCE
    assert "chatgpt_stage3" in accepted["metadata"] and "openai_stage3_batch" not in accepted["metadata"]
    assert genui.collect(prepared) == report
    assert genui.run(prepared, client=client)["status"] == "already_collected" and client.calls == 1
    (prepared / "pilot_accepted.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        genui.collect(prepared)


def test_fidelity_rejection_does_not_trigger_regeneration(prepared):
    client = FakeCodex(text=GOOD.replace("$120", "$999"))
    genui.run(prepared, client=client)
    report = genui.collect(prepared)
    assert report["accepted"] == 0 and report["rejected_or_missing"] == 1
    assert client.calls == 1


def test_unknown_attempt_cannot_be_retried_even_after_restart(prepared):
    class Lost(FakeCodex):
        def generate(self, body, *, on_thread, **kwargs):
            self.calls += 1
            on_thread("unknown-thread")
            # Durable intent exists BEFORE a provider call can have happened.
            state = checked(prepared / "generation.json")
            assert next(iter(state["attempts"].values()))["thread_id"] == "unknown-thread"
            raise TimeoutError("secret provider details must not reach saved error")
    client = Lost()
    assert genui.run(prepared, client=client)["status"] == "paused"
    assert Evidence(prepared).samples()[0]["reasons"] == ["TimeoutError"]
    with pytest.raises(ValueError, match="uncertain"):
        genui.run(prepared, client=client)
    with pytest.raises(ValueError, match="uncertain"):
        genui.collect(prepared)
    assert genui.run(prepared, client=client, skip_uncertain=True)["status"] == "generated"
    assert client.calls == 1
    assert genui.collect(prepared)["accepted"] == 0
    assert "secret provider" not in "".join(p.read_text() for p in (prepared / "chatgpt_records").glob("*.json"))
    assert len(list((prepared / "chatgpt_records").glob("*.uncertain.json"))) == 1


def test_crash_after_response_write_reuses_durable_record(prepared):
    client = FakeCodex()
    genui.run(prepared, client=client)
    state = checked(prepared / "generation.json")
    for attempt in state["attempts"].values():
        attempt.pop("record_sha256")
        attempt["status"] = "inflight"
    save(prepared / "generation.json", state)
    assert genui.run(prepared, client=client)["status"] == "generated"
    assert client.calls == 1
    assert genui.collect(prepared)["accepted"] == 1


def test_modified_result_and_request_binding_fail_closed(prepared):
    genui.run(prepared, client=FakeCodex())
    state = checked(prepared / "generation.json")
    identifier = next(iter(state["attempts"]))
    path = prepared / "chatgpt_records" / (identifier + ".json")
    record = checked(path)
    record["request_sha256"] = "wrong"
    save(path, record)
    state["attempts"][identifier]["record_sha256"] = file_hash(path)
    save(prepared / "generation.json", state)
    with pytest.raises(ValueError, match="binding mismatch"):
        genui.collect(prepared)


@pytest.mark.parametrize("usage", [{}, {"inputTokens": -1}, {"inputTokens": 1, "outputTokens": 2, "cachedInputTokens": 3, "reasoningOutputTokens": 0}])
def test_usage_never_fabricated(usage):
    with pytest.raises(ValueError, match="usage"):
        genui.token_usage(usage)
    report = genui.usage_report([{"status": "completed"}])
    assert report["records_without_usage"] == 1 and report["cost_usd"] is None


def rpc_client(tmp_path):
    client = object.__new__(codex.CodexClient)
    client.workspace = SimpleNamespace(name=str(tmp_path))
    client.home = tmp_path / "auth"
    client._events = queue.Queue()
    calls = []
    def rpc(method, params=None, **kwargs):
        calls.append((method, params))
        if method == "account/read":
            return {"account": {"type": "chatgpt", "email": "not-exposed@example.com", "planType": "test"}}
        if method == "thread/start":
            return {"thread": {"id": "t"}, "model": "gpt-6.1-sol", "modelProvider": "openai", "instructionSources": []}
        if method == "turn/start":
            return {"turn": {"id": "u"}}
        return {}
    client.rpc = rpc
    return client, calls


def event(client, method, **params):
    client._events.put({"method": method, "params": {"threadId": "t", "turnId": "u", **params}})


def finish(client, text=GOOD, status="completed"):
    event(client, "item/completed", item={"type": "agentMessage", "id": "answer", "phase": "final_answer", "text": text})
    event(client, "turn/completed", turn={"id": "u", "status": status, "error": None})


def test_codex_protocol_uses_genui_isolation_and_only_summary(tmp_path):
    client, calls = rpc_client(tmp_path)
    event(client, "item/completed", item={"type": "reasoning", "id": "r", "summary": ["Brief summary"], "content": ["PRIVATE RAW"]})
    event(client, "thread/tokenUsage/updated", tokenUsage={"total": {"inputTokens": 100, "outputTokens": 40, "cachedInputTokens": 0, "reasoningOutputTokens": 5}})
    finish(client)
    config = {"model": "gpt-6.1-sol", "reasoning_effort": "medium", "max_output_tokens": 8192, "reasoning_summary": "auto"}
    result = client.generate(runner.request_body(config, "untrusted source", "canonical policy"))
    assert result["reasoning_summaries"] == ["Brief summary"] and "PRIVATE RAW" not in json.dumps(result)
    start = next(params for method, params in calls if method == "thread/start")
    assert start["ephemeral"] and start["baseInstructions"] == "canonical policy"
    assert start["config"]["features.shell_tool"] is False and start["sandbox"] == "read-only"
    assert not any(key.startswith("model_providers.") for key in start["config"])
    turn = next(params for method, params in calls if method == "turn/start")
    assert turn["input"] == [{"type": "text", "text": "untrusted source"}]
    assert turn["summary"] == "concise" and turn["approvalPolicy"] == "never"
    assert turn["sandboxPolicy"]["access"]["includePlatformDefaults"] is False
    assert "email" not in client.account()


@pytest.mark.parametrize("failure", ["tool", "reroute", "ambiguous", "interrupted", "empty", "disconnected"])
def test_codex_incomplete_or_unexpected_output_is_never_accepted(tmp_path, failure):
    client, _ = rpc_client(tmp_path)
    if failure == "tool":
        event(client, "item/started", item={"type": "commandExecution", "id": "cmd"})
    if failure == "reroute":
        event(client, "model/rerouted")
    if failure == "ambiguous":
        event(client, "item/completed", item={"type": "agentMessage", "id": "extra", "phase": "final_answer", "text": "extra"})
    if failure == "disconnected":
        event(client, "genui/disconnected")
    finish(client, text="" if failure == "empty" else GOOD, status="interrupted" if failure == "interrupted" else "completed")
    with pytest.raises(codex.CodexError):
        client._completion("t", "u", "gpt-6.1-sol", .1, None)


def test_model_catalog_is_checked_without_generation(tmp_path):
    client, _ = rpc_client(tmp_path)
    client.rpc = lambda *args, **kwargs: {"data": [{"model": "test-model", "supportedReasoningEfforts": [{"reasoningEffort": "medium"}]}], "nextCursor": None}
    client.check_model("test-model", "medium")
    with pytest.raises(codex.CodexError, match="unavailable"):
        client.check_model("different-model", "medium")


def test_oauth_url_validation(tmp_path):
    client, _ = rpc_client(tmp_path)
    seen = []
    def rpc(method, params):
        seen.append((method, params))
        return {"type": "chatgpt", "authUrl": "https://auth.openai.com/authorize", "loginId": "not-forwarded"}
    client.rpc = rpc
    assert client.login() == {"type": "chatgpt", "authUrl": "https://auth.openai.com/authorize"}
    assert seen == [("account/login/start", {"type": "chatgpt"})]
    client.rpc = lambda *args: {"authUrl": "https://attacker.invalid/"}
    with pytest.raises(codex.CodexError, match="sign-in URL"):
        client.login()


def test_auth_home_never_uses_repo_or_existing_custom_config(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        codex.CodexClient(ROOT / ".genui-test")
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="dedicated"):
        codex.CodexClient(tmp_path)


@pytest.mark.parametrize("nested", ["same", "child", "parent"])
def test_auth_directory_cannot_be_packaged_with_generation_data(tmp_path, nested):
    output = tmp_path / "output"
    auth = {"same": output, "child": output / "auth", "parent": tmp_path}[nested]
    with pytest.raises(ValueError, match="directories separate"):
        Controller(output, auth_home=auth)
    with pytest.raises(ValueError, match="directories separate"):
        genui.run(output, auth_home=auth)
    assert not output.exists()


def test_genui_client_identity_and_no_credential_environment_inheritance(tmp_path, monkeypatch):
    calls, captured = [], {}
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret-not-forwarded")
    monkeypatch.setenv("CODEX_SESSION_ID", "fixture-session-not-forwarded")
    monkeypatch.setattr(codex.shutil, "which", lambda name: "codex.exe")
    monkeypatch.setattr(codex.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="codex-cli test"))
    process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda: None, wait=lambda **_: 0)
    def popen(args, **kwargs):
        captured.update(args=args, **kwargs)
        return process
    monkeypatch.setattr(codex.subprocess, "Popen", popen)
    monkeypatch.setattr(codex.CodexClient, "_read", lambda self: None)
    monkeypatch.setattr(codex.CodexClient, "rpc", lambda self, method, params=None: calls.append((method, params)) or {})
    # Codex installs bundled skills here itself. A subsequent sign-in client
    # must be able to reuse its auth home without enabling custom user skills.
    (tmp_path / "skills/.system").mkdir(parents=True)
    client = codex.CodexClient(tmp_path)
    try:
        assert calls[0] == ("initialize", {"clientInfo": {"name": "genui", "title": "GenUI", "version": codex.CLIENT_VERSION}})
        assert captured["env"]["CODEX_HOME"] == str(tmp_path)
        assert "OPENAI_API_KEY" not in captured["env"] and "CODEX_SESSION_ID" not in captured["env"]
        assert 'forced_login_method="chatgpt"' in captured["args"]
        assert json.loads(process.stdin.getvalue())["method"] == "initialized"
    finally:
        client.close()


def test_unavailable_model_fails_before_an_attempt(prepared):
    class Unavailable(FakeCodex):
        def check_model(self, model, effort):
            raise codex.CodexError("Model unavailable")
    client = Unavailable()
    with pytest.raises(codex.CodexError, match="unavailable"):
        genui.run(prepared, client=client)
    assert client.calls == 0 and checked(prepared / "generation.json")["attempts"] == {}


def test_dashboard_uses_same_strict_decisions_and_preserves_archive(prepared):
    genui.run(prepared, client=FakeCodex())
    genui.collect(prepared)
    before = {str(p): file_hash(p) for p in prepared.rglob("*") if p.is_file()}
    evidence = Evidence(prepared)
    snapshot = evidence.snapshot()
    assert snapshot["brand"] == "GenUI" and snapshot["accepted"] == 1
    sample = evidence.detail(0)
    assert sample["source"] == SOURCE and sample["output"] == GOOD and sample["status"] == "accepted"
    assert before == {str(p): file_hash(p) for p in prepared.rglob("*") if p.is_file()}
    (prepared / "report.json").write_text('{"accepted":999}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        evidence.snapshot()


def test_dashboard_cannot_read_outside_its_run(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (tmp_path / "secret.json").write_text('{"private":"data"}', encoding="utf-8")
    with pytest.raises(ValueError, match="inside"):
        Evidence(output).read("../secret.json")


def test_dashboard_http_csrf_origin_and_file_boundaries(prepared):
    controller = Controller(prepared, client_factory=FakeCodex)
    server = make_server(controller, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(path, *, method="GET", headers=None, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = (response.status, response.read(), dict(response.getheaders()))
        connection.close()
        return result
    try:
        status, data, headers = request("/")
        assert status == 200 and b"GenUI" in data and "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert request("/api/state", headers={"Origin": "https://attacker.invalid"})[0] == 403
        assert request("/api/state", headers={"Host": "attacker.invalid"})[0] == 403
        assert request("/api/auth/login", method="POST", body="{}")[0] == 403
        assert request("/../contract.json")[0] == 404
        status, data, _ = request("/api/state")
        snapshot = json.loads(data)
        assert status == 200 and snapshot["auth"]["checked"] is False
        authorized = {"X-GenUI-Token": snapshot["csrf"], "Content-Type": "application/json"}
        assert request("/api/auth/login", method="POST", headers=authorized, body="{}")[0] == 200
        assert request("/api/run", method="POST", headers=authorized, body='{"model":"override"}')[0] == 409
        assert request("/api/sample?index=-1")[0] == 409
        assert b"GenUI" in request("/app.js")[1]
        assert "innerHTML" not in (ASSETS / "app.js").read_text(encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
        controller.close()
