"""Offline checks for switching at an exact label count and preserving phase barriers."""

import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_chatgpt_stage123_phased as phased

original = phased.original


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
