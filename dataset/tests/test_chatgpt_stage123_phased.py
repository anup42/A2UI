"""Offline checks for switching at an exact label count and preserving phase barriers."""

import json
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_chatgpt_stage123_phased as phased

original = phased.original


def source_brief(slot, _seed):
    return {"slot": slot, "shape": "short_answer", "domain": "test", "context": "fixture", "task": "Ask a question.",
            "supplied_literals": [], "answer_literals": []}


def saved_reply(text, slot):
    return SimpleNamespace(text=text, model=original.MODEL, raw={"thread_id": f"thread-{slot}", "turn_id": f"turn-{slot}"})


class SourceAdapter:
    def __init__(self, stage, reply):
        self.stage, self.reply = stage, reply

    def generate(self, prompt, *, system):
        return self.reply(prompt)


class NoQuotaLimit:
    def limits(self):
        return {}


def test_answer_prefix_moves_only_task_fields():
    template = (original.ROOT / "dataset/prompts/muse_stage2_quality_v1.md").read_text(encoding="utf-8")
    reordered = phased.shared_prefix_template(template)
    assert Counter(filter(None, template.splitlines())) == Counter(filter(None, reordered.splitlines()))
    fixed, task = reordered.split("## Task\n")
    for key in ("{query_text}", "{intent}", "{tags}"):
        assert key in task and key not in fixed
    assert "## " not in task
    with pytest.raises(ValueError, match="unique movable"):
        phased.shared_prefix_template(template.replace("{tags}", ""))


def test_exact_transition_reserve_and_resume(tmp_path, monkeypatch):
    events = []
    interrupt = [True]
    plan = {"target": 70, "shard_size": 4, "seed": 1234}

    def query(n):
        return {"slot": n, "query_id": f"query-{n}", "query_text": f"Describe scenario {n}.",
                    "intent": "test", "shape": "short_answer"}

    def source(n):
        return {"id": f"query-{n}", "response_text": f"Scenario {n} has its own answer.", "metadata": {"query": query(n)}}

    def queries(database, adapter, plans, seed, attempt):
        events.extend(("query", p["slot"]) for p in plans)
        return [query(p["slot"]) for p in plans]

    def answer(database, adapter, q, brief, excluded, template):
        index = q["slot"]
        events.append(("answer", index))
        assert ("## Task" in template.split("## Complete, readable source")[-1]) == (index >= 9)
        return (None, "source failed") if index == 11 else (source(index), None)

    def label(rows, directory, client, **kwargs):
        slots = [r["metadata"]["query"]["slot"] for r in rows]
        if slots[0] >= 9 and interrupt[0]:
            interrupt[0] = False
            raise RuntimeError("simulated stop before labels")
        events.extend(("label", n) for n in slots)
        directory.mkdir(exist_ok=True)
        archive = directory / "evidence.tar.gz"
        archive.write_bytes(str(slots).encode())
        result = {"dispatched": len(rows), "accepted": len(rows)-1, "held": 1, "archive_sha256": phased.file_hash(archive)}
        phased.save(directory / "result.json", result)
        return result

    monkeypatch.setattr(original, "query_batch", queries)
    monkeypatch.setattr(original, "answer", answer)
    monkeypatch.setattr(original, "_benchmark_exclusions", lambda: (set(), set(), {}))
    monkeypatch.setattr(phased.labels, "_screen", lambda rows: (rows, []))
    monkeypatch.setattr(phased.labels, "label_shard", label)

    class Client:
        def limits(self):
            return {}

    with closing(sqlite3.connect(tmp_path / "sources.sqlite")) as db, ThreadPoolExecutor(4) as pool:
        original.initialize(db)
        # The legacy run sealed slots 0..3 and already saved all answers for slots 4..7.
        with db:
            db.executemany("INSERT INTO items VALUES (?,?,?,NULL)",
                           [(n, original.packed(query(n)), original.packed(source(n))) for n in range(8)])
            label([source(n) for n in range(4)], tmp_path / "s00000", None)
            db.execute("INSERT INTO shards VALUES (0,4,3,1)")

        def resume():
            schedule = phased.scheduling_contract(tmp_path, db, plan, 9, 13)
            phased.PhasedRun(tmp_path, db, plan, schedule, 4, Client(), Client(), pool).execute()

        with pytest.raises(RuntimeError, match="simulated stop"):
            resume()
        bulk = phased.checked(tmp_path / "phased_sources.json")
        assert (bulk["generated"], bulk["start"], bulk["target"], bulk["reserve"]) == (9, 9, 61, 3)
        assert phased.checked(tmp_path / "phased_answers_done.json")["count"] == 61
        phases = [kind for kind, slot in events if slot >= 9]
        assert phases == sorted(phases, key=("query", "answer").index)
        assert not any(kind in {"query", "answer"} and slot < 8 for kind, slot in events)
        assert not any(kind == "answer" and slot > 70 for kind, slot in events)
        before = len(events)
        resume()
        assert {kind for kind, _ in events[before:]} == {"label"}
        labelled = [slot for kind, slot in events if kind == "label"]
        assert len(labelled) == len(set(labelled)) == 70 and 11 not in labelled
        assert db.execute("SELECT sum(generated),sum(accepted+held) FROM shards").fetchone() == (70, 70)
        before = len(events)
        resume()
        assert len(events) == before
        with db:
            db.execute("UPDATE items SET query=? WHERE slot=11", (original.packed(query(42)),))
        with pytest.raises(ValueError, match="Frozen Stage 1"):
            resume()


def test_offline_reconciliation_accepts_valid_candidates_and_preserves_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(original, "brief", source_brief)
    database = tmp_path / "sources.sqlite"
    schedule = {"legacy_stop": 4}
    bulk = {"schedule_sha256": phased.digest(schedule), "start": 4, "stop": 20}
    phased.save(tmp_path / "phased_sources.json", bulk)
    first = [
        {"slot": 4, "query_text": "Where is the nearest library?"},
        {"slot": 5, "query_text": "Compare the two ticket prices."},
        {"slot": 6, "query_text": "Where is the nearest library?"},
        {"slot": 7, "query_text": "Visit the café at https://example.test/."},
    ]

    def save_group(lo, response):
        return original.query_batch(database, SourceAdapter(1, lambda _: saved_reply(response, lo)),
                                    [source_brief(n, 1) for n in range(lo, lo + 4)], 1)

    with closing(sqlite3.connect(database)) as db:
        original.initialize(db)
        first_query, *_ = save_group(4, json.dumps(first))
        with db:
            db.execute("INSERT INTO items(slot,query) VALUES (?,?)", (4, original.packed(first_query)))
        assert save_group(8, "malformed response") == []

        def time_out(_):
            raise phased.CodexError(phased.COMPLETION_TIMEOUT)

        with pytest.raises(phased.CodexError, match="timed out"):
            original.query_batch(database, SourceAdapter(1, time_out), [source_brief(n, 1) for n in range(12, 16)], 1)
        journal = db.execute("SELECT * FROM turns ORDER BY key").fetchall()
        with pytest.raises(ValueError, match="Explicitly name"):
            phased.reconcile_queries(tmp_path, db, schedule, 1, [])
        assert db.execute("SELECT count(*) FROM items").fetchone() == (1,)
        report = phased.reconcile_queries(tmp_path, db, schedule, 1, ["q-12-4-0"])
        assert report["saved_candidates"] == 3
        assert report["accepted_slots"] == [5]
        assert report["held_slots"]["6"] == "near-duplicate saved query"
        assert report["held_slots"]["7"] == "invalid saved query"
        assert list(report["held_slots"])[2:] == ["8", "9", "10", "11"]
        assert report["skipped_requests"] == {"q-12-4-0": [12, 13, 14, 15]}
        assert db.execute("SELECT * FROM turns ORDER BY key").fetchall() == journal
        assert db.execute("SELECT slot FROM items WHERE query IS NOT NULL ORDER BY slot").fetchall() == [(4,), (5,)]
        assert phased.reconcile_queries(tmp_path, db, schedule, 1, ["q-12-4-0"]) == report
        phased.check_unresolved(db)


@pytest.mark.parametrize("stage", [1, 2])
def test_a_source_timeout_holds_only_its_work_and_keeps_siblings(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(original, "brief", source_brief)
    monkeypatch.setattr(original, "_benchmark_exclusions", lambda: (set(), set(), {}))
    calls = []
    database = tmp_path / "sources.sqlite"

    def generate_queries(prompt):
        slots = [row["slot"] for row in json.loads(prompt)]
        calls.append(slots[0])
        if slots[0] == 0:
            raise phased.CodexError(phased.COMPLETION_TIMEOUT)
        return saved_reply(json.dumps([{"slot": n, "query_text": f"Describe item {n}."} for n in slots]), slots[0])

    def generate_answer(prompt):
        slot = int(prompt)
        calls.append(slot)
        if slot == 0:
            raise phased.CodexError(phased.COMPLETION_TIMEOUT)
        return saved_reply(f"Answer for slot {slot}.", slot)

    def answer(database, adapter, query, plan, excluded, template):
        slot = query["slot"]
        reply = original.turn(database, f"a-{slot}-0", adapter, "system", str(slot))
        return {"id": f"answer-{slot}", "response_text": reply["text"], "metadata": {"query": query}}, None

    with closing(sqlite3.connect(database)) as db, ThreadPoolExecutor(2) as pool:
        original.initialize(db)
        plan = {"target": 20, "seed": 1, "shard_size": 4}
        job = phased.PhasedRun(tmp_path, db, plan, {"legacy_stop": 0}, 2, NoQuotaLimit(), None, pool)
        if stage == 1:
            assert job.qgen.timeout == 900
            job.qgen = SourceAdapter(1, generate_queries)
            job.queries(0, 8, "test")
            job.queries(0, 8, "resume")
            assert sorted(calls) == [0, 4]
            assert db.execute("SELECT count(query),count(hold) FROM items").fetchone() == (4, 4)
            key = "q-0-4-0"
        else:
            assert job.agen.timeout == 900
            for slot in (0, 1):
                query = {"slot": slot, "query_id": f"q{slot}", "query_text": f"Describe item {slot}."}
                with db:
                    db.execute("INSERT INTO items(slot,query) VALUES (?,?)", (slot, original.packed(query)))
            job.agen = SourceAdapter(2, generate_answer)
            monkeypatch.setattr(original, "answer", answer)
            assert job.answers(0, 2, 2, "", "test") == 2
            assert job.answers(0, 2, 2, "", "resume") == 0
            assert sorted(calls) == [0, 1]
            assert db.execute("SELECT count(source),count(hold) FROM items").fetchone() == (1, 1)
            key = "a-0-0"
        assert db.execute("SELECT key FROM phased_skipped_turns").fetchall() == [(key,)]
        assert db.execute("SELECT key FROM turns WHERE reply IS NULL").fetchall() == [(key,)]
        phased.check_unresolved(db)


def test_schedule_upgrade_keeps_its_original_binding(tmp_path):
    plan = {"target": 20, "shard_size": 4}
    with closing(sqlite3.connect(":memory:")) as db:
        original.initialize(db)
        old = phased.scheduling_contract(tmp_path, db, plan, 5, 4)
        old["scheduler_sha256"] = min(phased.PRE_TIMEOUT_SCHEDULERS)
        receipt = tmp_path / "phased_schedule.json"
        phased.save(receipt, old)
        sealed = receipt.read_bytes()
        with pytest.raises(ValueError, match="offline query reconciliation"):
            phased.scheduling_contract(tmp_path, db, plan, 5, 4)
        assert phased.scheduling_contract(tmp_path, db, plan, 5, 4, register_recovery=True) == old
        assert phased.scheduling_contract(tmp_path, db, plan, 5, 4) == old
        assert receipt.read_bytes() == sealed
        with pytest.raises(ValueError, match="schedule changed"):
            phased.scheduling_contract(tmp_path, db, plan, 5, 8, register_recovery=True)
