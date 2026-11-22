"""Continue a pinned ChatGPT dataset run with separate query, answer and label phases."""

import argparse
import hashlib
import json
import math
import re
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import chatgpt_fresh_labels as labels
import run_chatgpt_stage123 as original
from pipeline.genui_codex_client import CodexError
from pipeline.genui_quality.validation_v5_4 import ensure_v5_4_validation_ready
from pipeline.v11_regeneration_io import checked, digest, file_hash, save

SOURCE_TIMEOUT = 900
# This diagnostic is emitted only by the pinned client's completion deadline.
COMPLETION_TIMEOUT = "Generation timed out; no automatic retry"
PRE_TIMEOUT_SCHEDULERS = {
    "129f8feb47b58af41aaa1204d4c5bd56030dee67d06e0b8292f727a6b26a6e62",
    "f69e96558be6be6eea4cc21e3f716468e9ab07561b434b53fcd63e7320bb5fde",
}  # The released 040c4b79 scheduler with LF or CRLF.


def shared_prefix_template(template):
    """Move just the variable task block behind all the original source fidelity rules."""
    before, marker, rest = template.partition("## Task\n")
    task, following, rules = rest.partition("\n## ")
    if not marker or not following or any(task.count(field) != 1 or field in before + rules
                                          for field in ("{query_text}", "{intent}", "{tags}")):
        raise ValueError("The source template has no unique movable task block")
    return before + "## " + rules.rstrip() + "\n\n" + marker + task.strip() + "\n"


def check_predecessor(root):
    plan = checked(root / "plan.json")
    drivers = [Path(original.__file__), Path(labels.__file__), original.ROOT / "dataset/src/chatgpt_sources.py"]
    if (plan["stage3"] != original.pilot.fingerprint()
            or plan["driver"] != {p.name: file_hash(p) for p in drivers}
            or (plan["model"], plan["effort"]) != (original.MODEL, original.EFFORT)):
        raise ValueError("Run with the predecessor's exact generator, evaluator, prompts and model")
    return plan


def scheduling_contract(root, db, plan, switch, label_size, *, register_recovery=False):
    if not 0 <= switch < plan["target"] or not 1 <= label_size <= 1000:
        raise ValueError("Invalid transition count or Stage 3 group size")
    contract = {"plan_sha256": digest(plan), "scheduler_sha256": file_hash(Path(__file__)),
                "switch_after": switch, "label_size": label_size}
    for start, generated, accepted, held in db.execute("SELECT * FROM shards ORDER BY start"):
        if generated or held:
            path = root / f"s{start:05d}"
            receipt = checked(path / "result.json")
            if ((generated, accepted, held) != (receipt["dispatched"], receipt["accepted"], receipt["held"])
                    or file_hash(path / "evidence.tar.gz") != receipt["archive_sha256"]):
                raise ValueError(f"Completed shard evidence changed at {start}")
    receipt_path = root / "phased_schedule.json"
    if receipt_path.exists():
        receipt = checked(receipt_path)
        expected = {**contract, "legacy_stop": receipt["legacy_stop"]}
        if receipt != expected:
            old = receipt["scheduler_sha256"]
            if old not in PRE_TIMEOUT_SCHEDULERS or receipt != {**expected, "scheduler_sha256": old}:
                raise ValueError("Phased schedule changed on resume")
            upgrade = root / "phased_timeout_runtime.json"
            binding = {"original_schedule": digest(receipt), "scheduler_sha256": contract["scheduler_sha256"],
                       "source_timeout_seconds": SOURCE_TIMEOUT}
            if register_recovery and not upgrade.exists():
                save(upgrade, binding)
            if not upgrade.exists() or checked(upgrade) != binding:
                raise ValueError("Register this scheduler with offline query reconciliation before resuming")
        return receipt
    starts = [r[0] for r in db.execute("SELECT start FROM shards ORDER BY start")]
    step = plan["shard_size"]
    if starts != list(range(0, len(starts) * step, step)):
        raise ValueError("Expected contiguous completed legacy shards")
    generated = db.execute("SELECT coalesce(sum(generated),0) FROM shards").fetchone()[0]
    if generated > switch:
        raise ValueError("The requested transition count has already been passed")
    if db.execute("SELECT 1 FROM turns WHERE reply IS NULL LIMIT 1").fetchone():
        raise ValueError("Review uncertain source turns before switching schedules")
    for directory in root.glob("s[0-9][0-9][0-9][0-9][0-9]"):
        if int(directory.name[1:]) not in starts and any(p.name != "regeneration.lock" for p in directory.iterdir()):
            raise ValueError(f"Finish the existing Stage 3 work in {directory.name} before switching")
    contract["legacy_stop"] = len(starts) * step
    save(receipt_path, contract)
    return contract


def init_recovery_tables(db):
    db.execute("CREATE TABLE IF NOT EXISTS phased_skipped_turns "
               "(key TEXT PRIMARY KEY REFERENCES turns(key), binding TEXT NOT NULL, reason TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS phased_reconciliations (id TEXT PRIMARY KEY, report BLOB NOT NULL)")


def record_skip(db, key, slots, *, stage, reason):
    row = db.execute("SELECT binding,request,reply FROM turns WHERE key=?", (key,)).fetchone()
    if row is None or row[2] is not None:
        raise ValueError(f"Only an unresolved dispatched turn can be skipped: {key}")
    binding, request = row[0], original.unpacked(row[1])
    if digest(request) != binding or request["stage"] != stage:
        raise ValueError(f"Invalid skipped request binding: {key}")
    db.execute("INSERT OR IGNORE INTO phased_skipped_turns VALUES (?,?,?)", (key, binding, reason))
    if db.execute("SELECT binding FROM phased_skipped_turns WHERE key=?", (key,)).fetchone() != (binding,):
        raise ValueError(f"Changed skipped request: {key}")
    hold = f"{reason}: {key}"
    if stage == 1:
        db.executemany("INSERT OR IGNORE INTO items(slot,hold) VALUES (?,?)", [(slot, hold) for slot in slots])
    else:
        db.executemany("UPDATE items SET hold=? WHERE slot=? AND hold IS NULL AND source IS NULL",
                       [(hold, slot) for slot in slots])
    return hold


def check_unresolved(db):
    unknown = db.execute("SELECT t.key FROM turns t LEFT JOIN phased_skipped_turns s ON t.key=s.key "
                         "WHERE t.reply IS NULL AND s.key IS NULL LIMIT 1").fetchone()
    if unknown:
        raise ValueError(f"Review unresolved turn {unknown[0]} before live dispatch")
    if db.execute("SELECT 1 FROM phased_skipped_turns s LEFT JOIN turns t ON t.key=s.key "
                  "WHERE t.key IS NULL OR s.binding!=t.binding OR t.reply IS NOT NULL LIMIT 1").fetchone():
        raise ValueError("Skipped request evidence changed")


def admit_query(db, index, done, query):
    slot = query["slot"]
    if slot in done or index.has_near_duplicate(query["query_text"]):
        return False
    db.execute("INSERT INTO items(slot,query) VALUES (?,?)", (slot, original.packed(index.add(query))))
    done.add(slot)
    return True


class OfflineQueryAdapter:
    stage = 1

    def generate(self, *_args, **_kwargs):
        raise RuntimeError("Offline reconciliation cannot generate queries")


def query_slots(key, bulk):
    match = re.fullmatch(r"q-(\d+)-(\d+)-([01])", key)
    if not match:
        raise ValueError(f"Unexpected query key {key}")
    start, count, attempt = map(int, match.groups())
    if (not bulk["start"] <= start < bulk["stop"] or count != (1 if attempt else min(4, bulk["stop"] - start))
            or (not attempt and (start - bulk["start"]) % 4)):
        raise ValueError(f"Query group outside its prepared schedule: {key}")
    return range(start, start + count), attempt


def reconcile_queries(root, db, schedule, seed, skip):
    bulk = checked(root / "phased_sources.json")
    if bulk["schedule_sha256"] != digest(schedule) or (root / "phased_queries_done.json").exists():
        raise ValueError("Offline reconciliation requires an unfinished, bound Stage 1")
    init_recovery_tables(db)
    known = {key for key, in db.execute("SELECT key FROM phased_skipped_turns")}
    unknown = {key for key, in db.execute("SELECT key FROM turns WHERE reply IS NULL")}
    if set(skip) - unknown or (unknown - known) != (set(skip) - known):
        raise ValueError(f"Explicitly name each unresolved request to skip: {sorted(unknown-known)}")
    for key in skip:
        query_slots(key, bulk)
    turns = []
    manifests = []
    for key, binding, request, blob in db.execute("SELECT key,binding,request,reply FROM turns ORDER BY rowid"):
        if not key.startswith("q-") or int(key.split("-")[1]) < bulk["start"]:
            continue
        slots, attempt = query_slots(key, bulk)
        if digest(original.unpacked(request)) != binding:
            raise ValueError(f"Changed request journal for {key}")
        turns.append((key, slots, attempt, blob))
        manifests.append((key, binding, hashlib.sha256(blob).hexdigest() if blob else None))
    identity = digest([digest(schedule), digest(bulk), sorted(skip), manifests])
    saved = db.execute("SELECT report FROM phased_reconciliations WHERE id=?", (identity,)).fetchone()
    if saved:
        report = original.unpacked(saved[0])
    else:
        index, existing = original.QueryQualityIndex(), set()
        for slot, query in db.execute("SELECT slot,query FROM items ORDER BY slot"):
            existing.add(slot)
            if query:
                index.add(original.unpacked(query))
        done = existing.copy()
        candidates, held, accepted = set(), {}, []
        with db:
            for key, slots, attempt, blob in turns:
                plans = [original.brief(n, seed) for n in slots]
                if blob is None:
                    # Verify the same prompt binding the live generator would use.
                    try:
                        original.query_batch(root / "sources.sqlite", OfflineQueryAdapter(), plans, seed, attempt)
                    except ValueError as exc:
                        if str(exc) != f"Delivery uncertain for {key}. Review the existing attempt before resuming.":
                            raise
                    else:
                        raise ValueError(f"Expected unresolved request {key}")
                    continue
                if all(slot in existing for slot in slots):
                    continue
                reply = original.unpacked(blob)
                if reply.get("model") != original.MODEL or not reply.get("thread_id") or not reply.get("turn_id"):
                    raise ValueError(f"Missing saved query provenance: {key}")
                try:
                    entries = json.loads(reply["text"])
                except ValueError:
                    entries = []
                identifiable = {e["slot"] for e in entries if isinstance(e, dict) and type(e.get("slot")) is int
                                and e["slot"] in slots and isinstance(e.get("query_text"), str) and e["query_text"].strip()
                                } if isinstance(entries, list) else set()
                candidates.update(identifiable - existing)
                for slot in slots:
                    if slot not in done:
                        held[slot] = "invalid saved query" if slot in identifiable else "no usable query in saved reply"
                for query in original.query_batch(root / "sources.sqlite", OfflineQueryAdapter(), plans, seed, attempt):
                    slot = query["slot"]
                    if slot not in done:
                        if admit_query(db, index, done, query):
                            accepted.append(slot)
                            held.pop(slot, None)
                        else:
                            held[slot] = "near-duplicate saved query"
            db.executemany("INSERT INTO items(slot,hold) VALUES (?,?)", sorted(held.items()))
            skipped = {}
            for key in sorted(skip):
                slots, _ = query_slots(key, bulk)
                record_skip(db, key, slots, stage=1, reason="reviewed unresolved request")
                skipped[key] = list(slots)
            check_unresolved(db)
            report = {"reconciliation_sha256": identity, "saved_candidates": len(candidates),
                      "accepted_slots": sorted(accepted), "held_slots": {str(n): reason for n, reason in sorted(held.items())},
                      "skipped_requests": skipped, "previous_items": len(existing)}
            db.execute("INSERT INTO phased_reconciliations VALUES (?,?)", (identity, original.packed(report)))
    path = root / f"phased_query_recovery_{identity[:12]}.json"
    if path.exists():
        if checked(path) != report:
            raise ValueError("Query recovery receipt changed")
    else:
        save(path, report)
    return report


class PhasedRun:
    def __init__(self, root, db, plan, schedule, workers, client, label_client, pool):
        self.root, self.db, self.plan, self.schedule = root, db, plan, schedule
        self.database = root / "sources.sqlite"
        self.workers, self.client, self.label_client, self.pool = workers, client, label_client, pool
        self.qgen, self.agen = [original.ChatGPTSourceAdapter(client, stage=n, model=original.MODEL,
                                                            effort=original.EFFORT, timeout=SOURCE_TIMEOUT)
                                for n in (1, 2)]
        self.index, self.seen = original.QueryQualityIndex(), set()
        for query, source in db.execute("SELECT query,source FROM items ORDER BY slot"):
            if query:
                self.index.add(original.unpacked(query))
            if source:
                self.seen.update(original.pilot.source_hashes(original.unpacked(source)))
        _, self.excluded, _ = original._benchmark_exclusions()
        self.template = (original.ROOT / "dataset/prompts/muse_stage2_quality_v1.md").read_text(encoding="utf-8")
        self.source_budget = plan["target"] + max(1, plan["target"] // 20)
        self.last_progress = 0.0
        self.active_bulk = None
        init_recovery_tables(db)
        check_unresolved(db)
        db.execute("CREATE TABLE IF NOT EXISTS phased_labels (start INTEGER PRIMARY KEY, stop INTEGER NOT NULL)")
        cursor = schedule["legacy_stop"]
        for start, stop in db.execute("SELECT start,stop FROM phased_labels ORDER BY start"):
            if start != cursor or stop <= start or not db.execute("SELECT 1 FROM shards WHERE start=?", (start,)).fetchone():
                raise ValueError("The saved label ranges are inconsistent")
            cursor = stop

    def totals(self):
        return self.db.execute("SELECT coalesce(sum(generated),0),coalesce(sum(accepted),0),"
                               "coalesce(sum(held),0) FROM shards").fetchone()

    def progress(self, stage, detail, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_progress < 2:
            return
        if not force:
            self.check_space()
        queries, responses, source_holds = self.db.execute("SELECT count(query),count(source),count(hold) FROM items").fetchone()
        generated, accepted, held = self.totals()
        if stage in {"Stage 1", "Stage 2"} and self.active_bulk:
            start, target = self.active_bulk["start"], self.active_bulk["target"]
            primary, discarded = self.db.execute(
                "SELECT count(CASE WHEN slot<? THEN query END),count(hold) FROM items WHERE slot>=? AND slot<?",
                (start + target, start, self.active_bulk["stop"])).fetchone()
            if discarded > self.active_bulk["reserve"]:
                raise ValueError("Source holds exceed the reserved slots. Review the run before dispatching more.")
            if stage == "Stage 1":
                detail = f"{primary:,}/{target:,} primary queries saved · {self.active_bulk['reserve']} reserve slots"
        state = {"stage": stage, "detail": detail, "workers": self.workers, "target": self.plan["target"],
                     "queries": queries, "responses": responses, "source_holds": source_holds, "generated": generated,
                     "accepted": accepted, "held": held, "updated_at": time.time()}
        original.atomic_bytes(self.root / "progress.json", original.canonical(state))
        self.last_progress = now

    def check_space(self):
        if shutil.disk_usage(self.root).free < 512 * 1024**2:
            raise ValueError("Less than 512 MiB is free. Resume after freeing space.")

    def queries(self, start, stop, detail):
        self.check_space()
        self.progress("Stage 1", detail, force=True)
        done = {r[0] for r in self.db.execute("SELECT slot FROM items WHERE slot>=? AND slot<?", (start, stop))}
        seed = self.plan["seed"]
        for attempt in (0, 1):
            size = 1 if attempt else 4
            groups = [range(n, min(n + size, stop)) for n in range(start, stop, size)]
            groups = [group for group in groups if any(n not in done for n in group)]

            def ask(group, attempt=attempt):
                try:
                    return original.query_batch(self.database, self.qgen, [original.brief(n, seed) for n in group],
                                                seed, attempt), ()
                except CodexError as exc:
                    if exc.args != (COMPLETION_TIMEOUT,):
                        raise
                    key = f"q-{group[0]}-{len(group)}-{attempt}"
                    with closing(sqlite3.connect(self.database, timeout=30)) as db, db:
                        record_skip(db, key, group, stage=1, reason="source completion timed out")
                    return [], group

            for found, skipped in original.dispatch(self.pool, ask, groups, self.client, self.workers):
                done.update(skipped)
                with self.db:
                    for query in found:
                        admit_query(self.db, self.index, done, query)
                self.progress("Stage 1", detail)
        with self.db:
            self.db.executemany("INSERT INTO items(slot,hold) VALUES (?,'invalid or duplicate query')",
                                [(n,) for n in range(start, stop) if n not in done])

    def answers(self, start, stop, limit, template, detail):
        self.check_space()
        self.progress("Stage 2", detail, force=True)
        pending = self.db.execute("SELECT slot,query FROM items WHERE slot>=? AND slot<? AND query IS NOT NULL "
                                  "AND source IS NULL AND hold IS NULL ORDER BY slot LIMIT ?", (start, stop, limit)).fetchall()

        def respond(item):
            slot, query = item
            try:
                row, hold = original.answer(self.database, self.agen, original.unpacked(query),
                                            original.brief(slot, self.plan["seed"]), self.excluded, template)
            except CodexError as exc:
                if exc.args != (COMPLETION_TIMEOUT,):
                    raise
                with closing(sqlite3.connect(self.database, timeout=30)) as db, db:
                    keys = db.execute("SELECT key FROM turns WHERE key IN (?,?) AND reply IS NULL",
                                      (f"a-{slot}-0", f"a-{slot}-1")).fetchall()
                    if len(keys) != 1:
                        raise ValueError(f"Cannot identify the timed-out answer turn for slot {slot}") from exc
                    hold = record_skip(db, keys[0][0], [slot], stage=2, reason="source completion timed out")
                row = None
            return slot, row, hold

        for slot, row, hold in original.dispatch(self.pool, respond, pending, self.client, self.workers):
            if row:
                hashes = original.pilot.source_hashes(row)
                if hashes & self.seen:
                    row, hold = None, "duplicate source answer"
                self.seen.update(hashes)
            with self.db:
                self.db.execute("UPDATE items SET source=?,hold=? WHERE slot=?", (original.packed(row) if row else None, hold, slot))
            self.progress("Stage 2", detail)
        return len(pending)

    def sources(self, start, stop, limit):
        records = self.db.execute("SELECT slot,source FROM items WHERE source IS NOT NULL AND slot>=? AND slot<? "
                                  "ORDER BY slot LIMIT ?", (start, stop, limit))
        return [(slot, original.unpacked(blob)) for slot, blob in records]

    def label(self, start, stop, sources, *, bulk):
        self.check_space()
        rows = [row for _, row in sources]
        detail = "Phased labels" if bulk else f"First {self.schedule['switch_after']:,} labels"
        self.progress("Stage 3", f"{detail} · source slots {start + 1}–{stop}", force=True)
        result = {"accepted": 0, "held": 0, "dispatched": 0}
        if rows:
            result = labels.label_shard(rows, self.root / f"s{start:05d}", self.label_client, workers=self.workers,
                                        model=original.MODEL, reasoning_effort=original.EFFORT, seed=self.plan["seed"])
            if result["accepted"] + result["held"] != len(rows) or (bulk and result["dispatched"] != len(rows)):
                raise ValueError("Stage 3 evidence differs from the screened source queue")
        with self.db:
            self.db.execute("INSERT INTO shards VALUES (?,?,?,?)", (start, result["dispatched"], result["accepted"], result["held"]))
            self.db.execute("INSERT INTO phased_labels VALUES (?,?)", (start, stop))
        if len(rows) >= 16 and not result["accepted"]:
            raise RuntimeError("No training-eligible outputs in this shard. Review its results before continuing.")

    def cursor(self):
        return self.db.execute("SELECT coalesce(max(stop),?) FROM phased_labels", (self.schedule["legacy_stop"],)).fetchone()[0]

    def warmup(self):
        switch = self.schedule["switch_after"]
        while self.totals()[0] < switch:
            start = self.cursor()
            count = min(self.plan["shard_size"], switch - self.totals()[0], self.source_budget - start)
            if count <= 0:
                raise ValueError("Source replacement budget exhausted before the transition")
            stop = start + count
            detail = f"Completing the first {switch:,} labels · source slots {start + 1}–{stop}"
            self.queries(start, stop, detail)
            self.answers(start, stop, count, self.template, detail)
            self.label(start, stop, self.sources(start, stop, count), bulk=False)

    def bulk_plan(self):
        path = self.root / "phased_sources.json"
        if path.exists():
            plan = checked(path)
            if plan["schedule_sha256"] != digest(self.schedule):
                raise ValueError("Bulk sources belong to another schedule")
            return plan
        generated, accepted, held = self.totals()
        if generated != self.schedule["switch_after"]:
            raise ValueError("The transition must occur exactly at the requested output count")
        start = self.cursor()
        remaining = self.plan["target"] - generated
        # Stage 2 sometimes rejects a perfectly valid query's answer. Prepare a small
        # query-only reserve up front so replacement answers never trigger a phase reversal.
        observed_loss = max(0, start - generated)
        reserve = min(self.source_budget - start - remaining,
                      self.plan["shard_size"] + math.ceil(2 * remaining * observed_loss / max(start, 1)))
        if reserve < 0:
            raise ValueError("Insufficient remaining source budget")
        plan = {"schedule_sha256": digest(self.schedule), "start": start, "target": remaining, "reserve": reserve,
                    "stop": start + remaining + reserve, "generated": generated, "accepted": accepted, "held": held}
        save(path, plan)
        return plan

    def bulk_sources(self, bulk):
        self.active_bulk = bulk
        start, stop, target = bulk["start"], bulk["stop"], bulk["target"]
        query_barrier = self.root / "phased_queries_done.json"
        binding = digest(bulk)
        if not query_barrier.exists():
            self.queries(start, stop, f"All {target:,} primary queries and {bulk['reserve']} reserve slots before Stage 2")
        records = self.db.execute("SELECT slot,query FROM items WHERE slot>=? AND slot<? ORDER BY slot", (start, stop))
        queries = [(slot, hashlib.sha256(blob).hexdigest() if blob else None) for slot, blob in records]
        valid = sum(sha is not None for _, sha in queries)
        if len(queries) != stop - start or valid < target:
            raise ValueError("Incomplete query pool or too many held queries. Inspect before generating more.")
        query_identity = {"bulk_sha256": binding, "valid_queries": valid, "queries_sha256": digest(queries)}
        if not query_barrier.exists():
            save(query_barrier, query_identity)
        elif checked(query_barrier) != query_identity:
            raise ValueError("Frozen Stage 1 query pool changed")
        source_barrier = self.root / "phased_answers_done.json"
        if not source_barrier.exists():
            template = shared_prefix_template(self.template)
            while True:
                count = self.db.execute("SELECT count(source) FROM items WHERE slot>=? AND slot<?", (start, stop)).fetchone()[0]
                if count > target:
                    raise ValueError("Extra sources in the phased answer queue")
                if count < target:
                    done = self.answers(start, stop, target - count, template, f"All {target:,} source answers before Stage 3")
                    if not done:
                        raise ValueError("Eligible queries exhausted. Inspect source holds before continuing.")
                    continue
                sources = self.sources(start, stop, target)
                _, held = labels._screen([row for _, row in sources])
                holds = {record["id"]: record["reasons"] for record in held}
                if not holds:
                    break
                with self.db:
                    self.db.executemany("UPDATE items SET source=NULL,hold=? WHERE slot=?",
                                        [(", ".join(holds[row["id"]]), slot) for slot, row in sources if row["id"] in holds])
            save(source_barrier, {"bulk_sha256": binding, "sources_sha256": digest(sources), "count": target})
        sources = self.sources(start, stop, target + 1)
        if len(sources) != target or checked(source_barrier) != {
            "bulk_sha256": binding, "sources_sha256": digest(sources), "count": target
        }:
            raise ValueError("Frozen Stage 2 source queue changed")
        return sources

    def execute(self):
        self.warmup()
        bulk = self.bulk_plan()
        sources = self.bulk_sources(bulk)
        size = self.schedule["label_size"]
        cursor = self.cursor()
        pending = [(slot, row) for slot, row in sources if slot >= cursor]
        if self.totals()[0] + len(pending) != self.plan["target"]:
            raise ValueError("The remaining source queue does not match the output budget")
        for pos in range(0, len(pending), size):
            chunk = pending[pos:pos + size]
            stop = chunk[-1][0] + 1
            self.label(cursor, stop, chunk, bulk=True)
            cursor = stop
        self.progress("complete", "All labels generated and assessed", force=True)


def run(args):
    root = args.output_dir.resolve()
    original.validate_auth_location(root)
    workers = original.worker_count(args.workers)
    with original._exclusive(root), closing(sqlite3.connect(root / "sources.sqlite", timeout=30)) as db:
        plan = check_predecessor(root)
        schedule = scheduling_contract(root, db, plan, args.switch_after, args.label_size,
                                       register_recovery=args.reconcile_queries)
        ensure_v5_4_validation_ready()
        if args.reconcile_queries:
            report = reconcile_queries(root, db, schedule, plan["seed"], args.skip_unresolved_query)
            print(json.dumps({"saved_candidates": report["saved_candidates"], "accepted": len(report["accepted_slots"]),
                              "held_slots": report["held_slots"], "skipped_requests": report["skipped_requests"],
                              "reconciliation_sha256": report["reconciliation_sha256"]}))
            return
        with closing(original.ChatGPTSourceClient(executable=args.codex_executable)) as client, closing(
                original.CodexClient(executable=args.codex_executable)) as label_client, ThreadPoolExecutor(workers) as pool:
            if not client.account()["signed_in"]:
                raise ValueError("GenUI requires ChatGPT sign-in")
            client.check_model(original.MODEL, original.EFFORT)
            job = PhasedRun(root, db, plan, schedule, workers, client, label_client, pool)
            try:
                job.execute()
            except Exception as exc:
                job.progress("paused", str(exc)[:400], force=True)
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--switch-after", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--label-size", type=int, default=256)
    parser.add_argument("--codex-executable")
    parser.add_argument("--reconcile-queries", action="store_true", help="Validate saved bulk Stage 1 replies offline and exit")
    parser.add_argument("--skip-unresolved-query", action="append", default=[], metavar="KEY",
                        help="During offline reconciliation, mark this unresolved query request skipped without retrying")
    args = parser.parse_args()
    if args.skip_unresolved_query and not args.reconcile_queries:
        parser.error("--skip-unresolved-query requires --reconcile-queries")
    run(args)
