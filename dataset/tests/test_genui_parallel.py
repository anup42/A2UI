"""Bounded parallel generation, cache reuse, transport isolation and durable recovery."""
from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline import chatgpt_stage3 as genui
from pipeline import genui_codex_client as codex
from pipeline import openai_stage3_batch as pilot
from pipeline import openai_stage3_batch_api as batch
from pipeline import openai_stage3_batch_runner as runner
from pipeline.genui_dashboard import Controller, Evidence
from pipeline.genui_setup import settings
from pipeline.v11_regeneration_io import checked, digest
from test_chatgpt_stage3 import FakeCodex, rpc_client
from test_genui_settings import options
from test_openai_stage3_batch import GOOD, SOURCE, inputs, row


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "client", lambda: pytest.fail("No paid API calls permitted"))
    source = inputs(tmp_path, [row(f"source-{i}", SOURCE.replace("Alder", f"Alder{i}")) for i in range(7)])
    output = tmp_path / "parallel"
    pilot.prepare(source, output, samples=7, transport="chatgpt", pilot_profile="fidelity_v2", generation_workers=3)
    return output


class ParallelCodex(FakeCodex):
    def __init__(self, output, *, fail=False, barrier_size=3):
        super().__init__()
        self.output, self.fail = output, fail
        self.barrier = threading.Barrier(barrier_size)
        self.lock, self.first_done = threading.Lock(), False
        self.live, self.peak, self.bodies, self.completed = 0, 0, [], []
        self.keys = {digest(r["body"]): r["custom_id"] for r in runner.read_rows(output / "requests.jsonl")}

    def generate(self, body, *, on_thread, on_event, **kwargs):
        identifier = self.keys[digest(body)]
        source = re.search(r"Alder(\d+)", body["input"][1]["content"]).group(1)
        with self.lock:
            self.calls += 1
            ordinal = self.calls
            assert ordinal == 1 or self.first_done, "Parallel work began before the real first sample completed"
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.bodies.append(body)
        state = checked(self.output / "generation.json")
        assert state["attempts"][identifier]["status"] == "inflight"
        on_thread("thread-" + source)
        assert checked(self.output / "generation.json")["attempts"][identifier]["thread_id"] == "thread-" + source
        try:
            if ordinal == 1:
                progress = json.loads((self.output / "progress.json").read_text())
                assert progress["cache_phase"] == "warming" and progress["active_workers"] == 1
            if 2 <= ordinal <= 1 + self.barrier.parties:
                self.barrier.wait(timeout=30)
                if self.fail and ordinal == 2:
                    on_event({"usage": {"inputTokens": 80, "outputTokens": 5, "cachedInputTokens": 40, "reasoningOutputTokens": 1}})
                    raise TimeoutError("Uncertain model result")
                if self.fail:
                    deadline = time.monotonic() + 10
                    while checked(self.output / "generation.json")["status"] != "paused":
                        assert time.monotonic() < deadline
                        time.sleep(.01)
            usage = {"inputTokens": 100 + int(source), "outputTokens": 40, "cachedInputTokens": 50,
                     "reasoningOutputTokens": 10, "cacheWriteInputTokens": 20}
            with self.lock:
                self.completed.append(identifier)
            return {"status": "completed", "model": body["model"], "thread_id": "thread-" + source,
                    "turn_id": "turn-" + source, "text": GOOD.replace("Alder", "Alder" + source), "usage": usage}
        finally:
            with self.lock:
                self.live -= 1
                if ordinal == 1:
                    self.first_done = True


def test_real_parallelism_warms_once_binds_each_result_and_replays_strictly(prepared):
    client = ParallelCodex(prepared)
    result = genui.run(prepared, client=client)
    assert result["status"] == "generated" and client.calls == 7 and client.peak == 3
    assert len({digest(body["input"][0]) for body in client.bodies}) == 1
    assert len({body["model"] for body in client.bodies}) == 1
    snapshot = Evidence(prepared).snapshot()
    assert snapshot["workers"] == 3 and snapshot["active_workers"] == 0 and snapshot["cache_phase"] == "idle"
    assert genui.run(prepared, client=client)["status"] == "generated" and client.calls == 7
    report = genui.collect(prepared)
    assert report["accepted"] == 7 and report["training_input_ready"] is False
    assert report["usage"]["input_tokens"] == 721
    assert report["usage"]["cached_input_tokens"] == 350 and report["usage"]["cache_write_input_tokens"] == 140
    assert report["usage"]["reported_cache_hit_rate"] == pytest.approx(350 / 721)
    assert report["usage"]["records_without_usage"] == report["usage"]["records_without_cache_write_usage"] == 0
    assert genui.collect(prepared) == report


def test_failure_drains_inflight_preserves_usage_and_resume_never_retries(prepared):
    client = ParallelCodex(prepared, fail=True)
    assert genui.run(prepared, client=client)["status"] == "paused"
    assert client.calls == 4 and client.live == 0
    progress = json.loads((prepared / "progress.json").read_text())
    assert progress["completed"] == 3 and progress["uncertain"] == 1 and progress["active_workers"] == 0
    assert progress["usage"]["records_with_partial_usage"] == 1
    assert progress["usage"]["records_without_usage"] == 0
    attempted = set(checked(prepared / "generation.json")["attempts"])
    with pytest.raises(ValueError, match="uncertain"):
        genui.run(prepared, client=client)
    with pytest.raises(ValueError, match="uncertain"):
        genui.collect(prepared)
    resumed = FakeCodex()
    assert genui.run(prepared, client=resumed, workers=1, skip_uncertain=True)["status"] == "generated"
    assert resumed.calls == 3 and client.calls == 4
    state = checked(prepared / "generation.json")
    assert state["execution_history"][-1]["workers"] == 1
    assert sum(a["status"] == "skipped" for a in state["attempts"].values()) == 1
    assert len(attempted) == 4 and len(state["attempts"]) == 7


def test_quota_exhaustion_dispatches_nothing_and_can_resume(prepared):
    class Exhausted(FakeCodex):
        def limits(self):
            return {"primary": {"usedPercent": 100}, "secondary": {"usedPercent": 20}}
    client = Exhausted()
    assert genui.run(prepared, client=client)["status"] == "paused" and client.calls == 0
    assert checked(prepared / "generation.json")["attempts"] == {}
    assert "allowance exhausted" in Evidence(prepared).snapshot()["pause_reason"]
    healthy = FakeCodex()
    assert genui.run(prepared, client=healthy, workers=1)["status"] == "generated"
    assert healthy.calls == 7


def test_dispatch_error_still_saves_already_started_result(prepared, monkeypatch):
    release, started = threading.Event(), threading.Event()
    class Delayed(FakeCodex):
        def generate(self, body, **kwargs):
            if self.calls == 1:
                started.set()
                assert release.wait(timeout=15)
            return super().generate(body, **kwargs)
    original = genui.load
    checks = 0
    def changed(path):
        nonlocal checks
        checks += 1
        if checks == 5:  # two initial loads, warm source, first parallel source, next dispatch
            assert started.wait(timeout=15)
            release.set()
            raise ValueError("Bound code changed")
        return original(path)
    monkeypatch.setattr(genui, "load", changed)
    client = Delayed()
    with pytest.raises(ValueError, match="Bound code changed"):
        genui.run(prepared, client=client)
    state = checked(prepared / "generation.json")
    assert state["status"] == "paused" and client.calls == 2
    assert all(a["status"] == "completed" for a in state["attempts"].values())
    assert len(list((prepared / "chatgpt_records").glob("*.json"))) == 2


@pytest.mark.parametrize("workers", [0, 9, True, 2.5, "2"])
def test_bad_worker_limits_fail_before_any_dispatch(prepared, workers):
    with pytest.raises(ValueError, match="workers"):
        genui.run(prepared, client=FakeCodex(), workers=workers)
    assert not (prepared / "generation.json").exists()
    with pytest.raises(ValueError, match="workers"):
        settings({**options(prepared), "workers": workers}, prepared)


def test_parallel_options_only_apply_to_chatgpt_and_survive_dashboard_reload(prepared):
    source = prepared.parent / "input"
    with pytest.raises(ValueError, match="only to the ChatGPT"):
        pilot.prepare(source, prepared.parent / "batch", samples=1, generation_workers=2)
    legacy = prepared.parent / "serial"
    pilot.prepare(source, legacy, samples=1, transport="chatgpt")
    assert "generation_workers" not in genui.load(legacy)["config"]
    assert genui.run(legacy, client=FakeCodex())["workers"] == 1
    controller = Controller(prepared, client_factory=FakeCodex)
    assert controller.snapshot()["workers"] == 3 and controller.defaults["workers"] == 3
    controller.action("prepare", {**options(prepared), "workers": 4})
    controller.worker.join(timeout=60)
    assert controller.error is None
    assert genui.load(controller.evidence.directory)["config"]["generation_workers"] == 4
    assert Controller(controller.evidence.directory, client_factory=FakeCodex).snapshot()["workers"] == 4
    controller.close()


def test_mismatched_cache_prefix_fails_before_model_generation(prepared, monkeypatch):
    contract = genui.load(prepared)
    requests = runner.read_rows(prepared / "requests.jsonl")
    requests[1]["body"]["input"][0]["content"][0]["text"] += " changed prefix"
    client = FakeCodex()
    with pytest.raises(ValueError, match="identical shared instruction prefix"):
        genui._dispatch(prepared, contract, genui._state(prepared, contract), {}, requests, client, 60, 3)
    assert client.calls == 0


def test_cache_write_usage_is_optional_and_never_fabricated():
    raw = {"inputTokens": 100, "outputTokens": 40, "cachedInputTokens": 60, "reasoningOutputTokens": 10}
    usage = genui.usage_report([{"usage": raw}, {"usage": {**raw, "cacheWriteInputTokens": 30}}, {}])
    assert usage["records_without_usage"] == 1 and usage["records_without_cache_write_usage"] == 2
    assert usage["cache_write_input_tokens"] == 30 and usage["reported_cache_hit_rate"] == .6
    for invalid in (-1, True, 101):
        with pytest.raises(ValueError, match="cache-write usage"):
            genui.token_usage({**raw, "cacheWriteInputTokens": invalid})


def test_interleaved_notifications_reach_only_their_registered_thread():
    events = codex._ThreadEvents()
    a, b = events.subscribe("a"), events.subscribe("b")
    for identity in ("b", "a", "b", "a"):
        events.put({"method": "item/completed", "params": {"threadId": identity}})
    assert [a.get_nowait()["params"]["threadId"] for _ in range(2)] == ["a", "a"]
    assert [b.get_nowait()["params"]["threadId"] for _ in range(2)] == ["b", "b"]
    for target in (a, b):
        with pytest.raises(queue.Empty):
            target.get_nowait()
    events.put({"method": "genui/disconnected"})
    assert a.get_nowait()["method"] == b.get_nowait()["method"] == "genui/disconnected"
    with pytest.raises(codex.CodexError, match="disconnected"):
        events.subscribe("c")


def test_slow_consumer_overflow_fails_closed_without_poisoning_other_threads():
    events = codex._ThreadEvents()
    a, b = events.subscribe("a"), events.subscribe("b")
    for _ in range(4097):
        events.put({"method": "item/agentMessage/delta", "params": {"threadId": "a"}})
    assert a.get_nowait()["method"] == "genui/eventOverflow"
    events.put({"method": "turn/completed", "params": {"threadId": "b"}})
    assert b.get_nowait()["method"] == "turn/completed"
    events.unsubscribe("a")
    assert events.subscribe("a").empty()


def test_parallel_codex_turns_keep_prompts_results_and_usage_isolated(tmp_path):
    client, _ = rpc_client(tmp_path)
    barrier, lock, starts = threading.Barrier(2), threading.Lock(), []
    def rpc(method, params=None, **kwargs):
        if method == "account/read":
            return {"account": {"type": "chatgpt"}}
        if method == "thread/start":
            with lock:
                starts.append(params)
                identity = str(len(starts))
            return {"thread": {"id": identity}, "model": "gpt-6.1-sol", "modelProvider": "openai"}
        if method == "turn/start":
            identity = params["threadId"]
            barrier.wait(timeout=10)
            for name, extra in [
                ("thread/tokenUsage/updated", {"tokenUsage": {"total": {"inputTokens": int(identity)}}}),
                ("item/completed", {"item": {"id": "answer", "type": "agentMessage", "text": params["input"][0]["text"]}}),
                ("turn/completed", {"turn": {"id": "turn-" + identity, "status": "completed"}}),
            ]:
                client._events.put({"method": name, "params": {"threadId": identity, "turnId": "turn-" + identity, **extra}})
            return {"turn": {"id": "turn-" + identity}}
        return {}
    client.rpc = rpc
    config = {"model": "gpt-6.1-sol", "reasoning_effort": "low", "max_output_tokens": 1024}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.generate, runner.request_body(config, source, "same shared policy")) for source in ("source A", "source B")]
        results = [f.result(timeout=10) for f in futures]
    assert [r["text"] for r in results] == ["source A", "source B"]
    assert all(r["usage"]["inputTokens"] == int(r["thread_id"]) for r in results)
    assert len({r["thread_id"] for r in results}) == 2
    assert starts[0]["baseInstructions"] == starts[1]["baseInstructions"] == "same shared policy"
    assert starts[0]["cwd"] == starts[1]["cwd"]
    assert client._events._queues == {}


def test_duplicate_thread_id_does_not_unsubscribe_another_worker(tmp_path):
    client, calls = rpc_client(tmp_path)
    existing = client._events.subscribe("t")
    config = {"model": "gpt-6.1-sol", "reasoning_effort": "low", "max_output_tokens": 1024}
    with pytest.raises(codex.CodexError, match="Duplicate"):
        client.generate(runner.request_body(config, "source", "policy"))
    assert client._events._queues["t"] is existing
    assert not any(name in {"turn/start", "thread/unsubscribe"} for name, _ in calls)
