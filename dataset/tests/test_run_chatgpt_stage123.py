"""Offline source generation, fidelity and durable turn regressions."""

import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_chatgpt_stage123 as coordinator


class ScriptedAdapter:
    def __init__(self, stage, *outputs):
        self.stage = stage
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, prompt, *, system):
        self.calls.append((prompt, system))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(text=output, model=coordinator.MODEL, raw={
            "thread_id": f"thread-{self.stage}", "turn_id": f"turn-{self.stage}-{len(self.calls)}",
        })


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "sources.sqlite"
    with closing(sqlite3.connect(path)) as db:
        coordinator.initialize(db)
    return path


def numerical_plan():
    return {"slot": 2, "shape": "numerical", "domain": "rail journeys", "task": "Show working.",
            "supplied_literals": ["11 minutes", "24 minutes"], "answer_literals": ["35 minutes"]}


def write_query(database, plan, text):
    output = json.dumps([{"slot": plan["slot"], "query_text": text}])
    adapter = ScriptedAdapter(1, output)
    return coordinator.query_batch(database, adapter, [plan], seed=621)


def test_fresh_query_and_answer_keep_distinct_turn_provenance_on_resume(database):
    plan = numerical_plan()
    text = "How much time for two activities of 11 minutes and 24 minutes? Show the addition."
    source = "Add 11 minutes and 24 minutes to get a total of 35 minutes."
    query, = write_query(database, plan, text)
    assert query["source"] == "generated"
    assert query["gen"] == {"model": coordinator.MODEL, "thread_id": "thread-1", "turn_id": "turn-1-1"}
    adapter = ScriptedAdapter(2, source)
    row, hold = coordinator.answer(database, adapter, query, plan, set(), "{query_text}")
    assert hold is None and row["response_text"] == source
    assert row["metadata"]["query"] == query
    assert row["metadata"]["stage2"] == {"model": coordinator.MODEL, "thread_id": "thread-2", "turn_id": "turn-2-1"}
    resume1, resume2 = ScriptedAdapter(1), ScriptedAdapter(2)
    assert coordinator.query_batch(database, resume1, [plan], 621) == [query]
    assert coordinator.answer(database, resume2, query, plan, set(), "{query_text}") == (row, None)
    assert resume1.calls == resume2.calls == []


def test_query_does_not_accept_numeric_substring(database):
    assert write_query(database, numerical_plan(), "Combine 111 minutes and 24 minutes.") == []


@pytest.mark.parametrize("answer_text", [
    "Add 11 minutes and 24 minutes to get a total of 135 minutes.",
    "Add 12 minutes and 23 minutes to get a total of 35 minutes.",
])
def test_stage2_holds_wrong_numeric_answer_after_bounded_retry(database, answer_text):
    plan = numerical_plan()
    query, = write_query(database, plan, "Combine 11 minutes and 24 minutes and show the working.")
    adapter = ScriptedAdapter(2, answer_text, answer_text)
    row, hold = coordinator.answer(database, adapter, query, plan, set(), "{query_text}")
    assert row is None and hold
    assert len(adapter.calls) == 2
    assert adapter.calls[0] != adapter.calls[1]


def test_stage2_holds_unsupplied_external_destination(database):
    url = "https://www.loc.gov/"
    plan = {"slot": 1, "shape": "reference", "domain": "library", "task": "Retain the URL.",
            "supplied_literals": [url], "answer_literals": [url]}
    query, = write_query(database, plan, f"Where can I start looking for library resources? Keep {url}")
    source = f"Start with the library catalogue at {url} and open the new library offer at https://other.invalid/."
    result, hold = coordinator.answer(database, ScriptedAdapter(2, source, source), query, plan, set(), "{query_text}")
    assert result is None and "unsupplied URLs" in hold


def test_uncertain_delivery_and_prompt_drift_never_issue_a_second_turn(database):
    adapter = ScriptedAdapter(1, TimeoutError("connection interrupted"))
    with pytest.raises(TimeoutError):
        coordinator.turn(database, "query-1", adapter, "system", "original")
    with pytest.raises(ValueError, match="uncertain"):
        coordinator.turn(database, "query-1", adapter, "system", "original")
    with pytest.raises(ValueError, match="changed"):
        coordinator.turn(database, "query-1", adapter, "system", "altered")
    assert len(adapter.calls) == 1


def test_parallel_claims_for_the_same_turn_dispatch_once(database):
    active, release = Event(), Event()

    class BlockingAdapter(ScriptedAdapter):
        def generate(self, prompt, *, system):
            active.set()
            assert release.wait(5)
            return super().generate(prompt, system=system)

    adapter = BlockingAdapter(1, "durable reply")
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(coordinator.turn, database, "q", adapter, "system", "same")
        try:
            assert active.wait(5)
            with pytest.raises(ValueError, match="uncertain"):
                coordinator.turn(database, "q", adapter, "system", "same")
        finally:
            release.set()
        assert pending.result()["text"] == "durable reply"
    assert len(adapter.calls) == 1


def test_coordinator_replaces_prescreen_holds_and_verifies_shards_on_restart(tmp_path, monkeypatch):
    import chatgpt_fresh_labels
    from pipeline.genui_quality import validation_v5_4

    class OfflineClient:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

        def account(self):
            return {"signed_in": True}

        def limits(self):
            return {}

        def check_model(self, model, effort):
            assert (model, effort) == (coordinator.MODEL, coordinator.EFFORT)

    questions = ["Which train leaves first?", "Compare these two breakfast options.", "Plan a short museum visit."]
    requested = []

    def queries(database, adapter, plans, seed, attempt=0):
        requested.extend(plan["slot"] for plan in plans)
        return [{"slot": n, "query_id": f"q{n}", "query_text": questions[n], "intent": "generic",
                 "shape": "short_answer"} for n in [plan["slot"] for plan in plans]]

    def answer(database, adapter, query, plan, exclusions, prompt_template):
        return {"id": query["query_id"], "response_text": query["query_text"], "metadata": {"query": query}}, None

    def labels(rows, path, client, **options):
        count = len(rows)
        held = int(count == 2)
        path.mkdir()
        archive = path / "evidence.tar.gz"
        archive.write_bytes(b"sealed fixture")
        result = {"dispatched": count - held, "accepted": count - held, "held": held,
                  "archive_sha256": coordinator.file_hash(archive)}
        coordinator.save(path / "result.json", result)
        return result

    monkeypatch.setattr(coordinator, "ChatGPTSourceClient", OfflineClient)
    monkeypatch.setattr(coordinator, "CodexClient", OfflineClient)
    monkeypatch.setattr(coordinator, "_benchmark_exclusions", lambda: (set(), set(), {}))
    monkeypatch.setattr(coordinator.pilot, "fingerprint", lambda: {"test": "fixture"})
    monkeypatch.setattr(validation_v5_4, "ensure_v5_4_validation_ready", lambda: None)
    monkeypatch.setattr(coordinator, "query_batch", queries)
    monkeypatch.setattr(coordinator, "answer", answer)
    monkeypatch.setattr(chatgpt_fresh_labels, "label_shard", labels)
    args = SimpleNamespace(output_dir=tmp_path, target=2, seed=42, shard_size=2, workers=2, codex_executable=None)
    coordinator.run(args)
    state = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert (state["stage"], state["queries"], state["generated"], state["accepted"]) == ("complete", 3, 2, 2)
    assert requested == [0, 1, 2]
    coordinator.run(args)
    assert requested == [0, 1, 2]
    (tmp_path / "s00000" / "evidence.tar.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="evidence changed"):
        coordinator.run(args)
    assert requested == [0, 1, 2]
