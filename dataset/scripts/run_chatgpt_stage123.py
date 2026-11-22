"""Generate fresh queries, source answers and Express labels with a signed-in GenUI account."""

import argparse
import itertools
import json
import random
import re
import shutil
import sqlite3
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from chatgpt_sources import ChatGPTSourceAdapter, ChatGPTSourceClient
from pipeline import openai_stage3_batch as pilot
from pipeline.chatgpt_stage3 import atomic_bytes, worker_count
from pipeline.genui_codex_client import CodexClient, validate_auth_location
from pipeline.source_quality import QueryQualityIndex, assess_source_quality
from pipeline.v11_regeneration import _benchmark_exclusions, _exclusive
from pipeline.v11_regeneration_io import canonical, checked, digest, file_hash, save

MODEL = "gpt-6.2-sol"
EFFORT = "medium"
DOMAINS = (
    "rail and bus journeys", "flight and airport logistics", "hotels and trip itineraries",
    "weather-dependent outdoor plans", "calendar and meeting coordination", "sports and fitness",
    "recipes and groceries", "restaurants and dietary preferences", "home maintenance",
    "concerts and local events", "library and reading plans", "classes and workshops",
    "product comparisons", "appointments and service bookings", "deliveries and order tracking",
    "shared household tasks", "museums and accessible visits", "work projects and handovers",
    "gardening", "community volunteering", "travel packing", "subscriptions and mobile plans",
    "study timetables", "hobbies and creative projects", "music and podcasts",
)
SHAPES = (
    "entity_list", "comparison", "numerical", "schedule", "entity_list",
    "procedure", "multi_section", "reference", "short_answer", "qualified_options",
)
GUIDANCE = {
    "entity_list": "Supply facts about 2–4 distinct named options with comparable attributes and ask for a useful shortlist.",
    "comparison": "Compare plausible alternatives using supplied evidence and a meaningful tradeoff.",
    "numerical": "Use both supplied quantities exactly and ask to show the calculation.",
    "schedule": "Supply times or dates and ask for a feasible sequence under one real constraint.",
    "procedure": "Ask for concrete sequential steps with a prerequisite, optional step, or exception.",
    "multi_section": "Ask for related information that naturally breaks into two or three distinct sections.",
    "reference": "Supply the assigned resource URL verbatim and ask to keep it as a reference or next action. Choose a task relevant to that resource.",
    "short_answer": "Ask a terse, natural mobile question that warrants a direct brief answer. Prefer everyday wording.",
    "qualified_options": "Supply named alternatives with exceptions such as included, optional, unavailable or only after a time.",
}
RESOURCES = (
    ("public park trip planning", "https://www.nps.gov/"),
    ("weather planning", "https://www.weather.gov/"),
    ("museum visits", "https://www.si.edu/"),
    ("finding library resources", "https://www.loc.gov/"),
    ("space education", "https://www.nasa.gov/"),
    ("public transport in London", "https://tfl.gov.uk/"),
)
QUERY_SYSTEM = """Write original, realistic user requests for a mobile assistant. Return a JSON array with
one object per supplied slot, each containing its integer slot and query_text. Work in English. Mix natural
short requests with conversational questions and supplied examples. Vary names, settings, sentence openings,
constraints, and the information the user needs. Never mention dataset generation, slots, layout, or the
requested data shape in the query. Use supplied scenario facts, or explicitly hypothetical details for
time-sensitive information. No external research, real personal data, or destinations other than supplied URLs.
Do not reuse a storyline from another slot. Include each slot's supplied literals exactly in that query."""
ANSWER_RULES = """Write only the answer a mobile assistant would give. Follow the task and preserve
the supplied entities, quantities, order, units, negations, and caveats. Keep each option's fields together
and named consistently. Use natural bullets, a compact rectangular table, steps or prose as appropriate.
Keep simple tasks short. Do not repeat a table as prose. For arithmetic show operands and the final value
with its unit. Use only destinations supplied by the user and keep citations distinct from actions.
Never invent current availability or external verification."""


def brief(slot, seed):
    # Shuffle each balanced block separately so adjacent shards cover the full mix.
    order = list(SHAPES)
    random.Random(f"{seed}:shape:{slot // len(order)}").shuffle(order)
    shape = order[slot % len(order)]
    rng = random.Random(f"{seed}:scenario:{slot}")
    domain = rng.choice(DOMAINS)
    task = GUIDANCE[shape]
    literals, answer_literals = [], []
    if shape == "numerical":
        a, b = sorted(rng.sample(range(11, 120), 2), reverse=True)
        unit = rng.choice(("minutes", "points", "credits", "items", "kilometers"))
        total = rng.choice((True, False))
        task += " Ask for the total." if total else " Ask how much more the larger quantity is."
        literals = [f"{a} {unit}", f"{b} {unit}"]
        answer_literals = [f"{a + b if total else a - b} {unit}"]
    if shape == "reference":
        domain, url = rng.choice(RESOURCES)
        literals = [url]
        answer_literals = [url]
    context = rng.choice(("a last-minute change", "a first-time visitor", "a group deciding together",
                          "an experienced user seeking a quick detail", "a tight time window", "a budget limit",
                          "accessibility needs", "two conflicting preferences", "a follow-up to an earlier plan",
                          "limited connectivity", "a straightforward everyday question"))
    return {"slot": slot, "shape": shape, "domain": domain, "context": context, "task": task,
            "supplied_literals": literals, "answer_literals": answer_literals}


def packed(value):
    return zlib.compress(canonical(value))


def unpacked(blob):
    return json.loads(zlib.decompress(blob))


def initialize(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS turns (key TEXT PRIMARY KEY, binding TEXT NOT NULL,
                                          request BLOB NOT NULL, reply BLOB);
        CREATE TABLE IF NOT EXISTS items (slot INTEGER PRIMARY KEY, query BLOB, source BLOB, hold TEXT);
        CREATE TABLE IF NOT EXISTS shards (start INTEGER PRIMARY KEY, generated INTEGER, accepted INTEGER, held INTEGER);
    """)


def turn(database, key, adapter, system, prompt):
    request = {"model": MODEL, "effort": EFFORT, "stage": adapter.stage, "system": system, "prompt": prompt}
    binding = digest(request)
    with closing(sqlite3.connect(database, timeout=30)) as db:
        saved = db.execute("SELECT binding,reply FROM turns WHERE key=?", (key,)).fetchone()
        if saved:
            if saved[0] != binding:
                raise ValueError(f"Prompt changed for saved turn {key}")
            if saved[1] is None:
                raise ValueError(f"Delivery uncertain for {key}. Review the existing attempt before resuming.")
            return unpacked(saved[1])
        with db:
            db.execute("INSERT INTO turns VALUES (?,?,?,NULL)", (key, binding, packed(request)))
        result = adapter.generate(prompt, system=system)
        reply = {"text": result.text, "thread_id": result.raw["thread_id"], "turn_id": result.raw["turn_id"],
                 "model": result.model, "usage": result.raw.get("usage")}
        with db:
            db.execute("UPDATE turns SET reply=? WHERE key=?", (packed(reply), key))
        return reply


def query_batch(database, adapter, plans, seed, attempt=0):
    slots = [p["slot"] for p in plans]
    inputs = [{k: v for k, v in p.items() if k != "answer_literals"} for p in plans]
    prompt = json.dumps(inputs, ensure_ascii=False)
    if attempt:
        prompt += "\nGive the requested new, distinct queries. Check every literal, ID and the JSON array shape."
    key = f"q-{slots[0]}-{len(slots)}-{attempt}"
    reply = turn(database, key, adapter, QUERY_SYSTEM, prompt)
    try:
        entries = json.loads(reply["text"])
    except ValueError:
        entries = []
    found = {}
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            slot, text = entry.get("slot"), entry.get("query_text")
            if type(slot) is not int or slot not in slots or not isinstance(text, str):
                continue
            found[slot] = text.strip() if slot not in found else ""
    queries = []
    for plan in plans:
        slot = plan["slot"]
        text = found.get(slot, "")
        urls = extract_urls(text)
        allowed = {s for s in plan["supplied_literals"] if s.startswith("https://")}
        if (len(text.split()) >= 3 and len(text) <= 4000 and urls == allowed
                and all(has_literal(text, s) for s in plan["supplied_literals"])):
            queries.append({"query_id": f"sol-{seed}-{slot:05d}", "query_text": text,
                            "intent": plan["domain"], "shape": plan["shape"], "source": "generated",
                            "slot": slot, "gen": {k: reply[k] for k in ("thread_id", "turn_id", "model")}})
    return queries


def extract_urls(text):
    return {url.rstrip(".,);]") for url in re.findall(r"https?://[^\s<>\x22]+", text)}


def has_literal(text, literal):
    return re.search(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", text, re.IGNORECASE) is not None


def dispatch(pool, function, items, client, workers):
    pending = iter(items)
    active = set()
    next_limit_check = 0.0
    while True:
        for _ in range(workers - len(active)):
            item = next(pending, None)
            if item is None:
                break
            if time.monotonic() >= next_limit_check:
                limits = client.limits()
                if any((limits.get(window) or {}).get("usedPercent", 0) >= 100 for window in ("primary", "secondary")):
                    raise RuntimeError("ChatGPT allowance exhausted. Resume after the window resets.")
                next_limit_check = time.monotonic() + 5
            active.add(pool.submit(function, item))
        if not active:
            return
        finished, active = wait(active, return_when=FIRST_COMPLETED)
        for future in finished:
            yield future.result()


def answer(database, adapter, query, plan, exclusions, prompt_template):
    text = query["query_text"]
    prompt = prompt_template.replace("{query_text}", text).replace("{intent}", query["intent"]).replace("{tags}", query["shape"])
    reasons = []
    for attempt in range(2):
        feedback = "\nCheck these issues with the earlier answer: " + ", ".join(reasons) if reasons else ""
        reply = turn(database, f"a-{plan['slot']}-{attempt}", adapter, ANSWER_RULES, prompt + feedback)
        source = reply["text"].strip()
        quality = assess_source_quality(query, source)
        row = {"id": query["query_id"], "response_text": source,
               "metadata": {"assigned_split": "train", "query_text": text, "query": query,
                            "source_quality": quality, "stage2": {k: reply[k] for k in ("thread_id", "turn_id", "model")}}}
        reasons = []
        if quality["status"] == "failed" or quality.get("training_eligibility") == "exclude":
            reasons.append("source contract")
        for gate in (pilot.guard.assess_source, pilot.fidelity_v2.assess_source):
            reasons.extend(gate(row)["reasons"])
        if extract_urls(source) - extract_urls(text):
            reasons.append("unsupplied URLs")
        if plan.get("shape") == "numerical":
            for literal in plan["supplied_literals"]:
                operand = literal.split()[0]
                if not re.search(r"(?<![\d.])" + re.escape(operand) + r"(?![\d.])", source):
                    reasons.append(f"show the supplied operand {literal}")
        for literal in plan["answer_literals"]:
            if not has_literal(source, literal):
                reasons.append(f"include the verified result {literal}")
        if pilot.source_hashes(row) & exclusions:
            reasons.append("held-out source overlap")
        if not reasons:
            return row, None
    return None, ", ".join(reasons)


def run(args):
    from chatgpt_fresh_labels import label_shard
    from pipeline.genui_quality.validation_v5_4 import ensure_v5_4_validation_ready

    workers = worker_count(args.workers)
    root = args.output_dir.resolve()
    validate_auth_location(root)
    root.mkdir(parents=True, exist_ok=True)
    database = root / "sources.sqlite"
    with _exclusive(root), closing(sqlite3.connect(database, timeout=30)) as db:
        config = {"target": args.target, "seed": args.seed, "shard_size": args.shard_size,
                  "model": MODEL, "effort": EFFORT, "stage3": pilot.fingerprint(),
                  "driver": {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("chatgpt_fresh_labels.py"),
                                                              ROOT / "dataset/src/chatgpt_sources.py")}}
        receipt = root / "plan.json"
        if receipt.exists():
            if checked(receipt) != config:
                raise ValueError("A different configuration or code snapshot owns this run directory")
        else:
            save(receipt, config)
        initialize(db)

        def progress(stage, detail):
            queries, responses, source_holds = db.execute(
                "SELECT count(query),count(source),count(hold) FROM items").fetchone()
            generated, accepted, held = db.execute(
                "SELECT coalesce(sum(generated),0),coalesce(sum(accepted),0),coalesce(sum(held),0) FROM shards").fetchone()
            atomic_bytes(root / "progress.json", canonical({"stage": stage, "detail": detail, "target": args.target,
                         "workers": workers, "queries": queries, "responses": responses, "source_holds": source_holds,
                         "generated": generated, "accepted": accepted, "held": held, "updated_at": time.time()}))

        try:
            progress("preflight", "Checking evaluator, source exclusions and subscription model")
            ensure_v5_4_validation_ready()
            _, excluded, _ = _benchmark_exclusions()
            idx = QueryQualityIndex()
            seen = set()
            for query_blob, source_blob in db.execute("SELECT query,source FROM items ORDER BY slot"):
                if query_blob:
                    idx.add(unpacked(query_blob))
                if source_blob:
                    seen.update(pilot.source_hashes(unpacked(source_blob)))
            template = (ROOT / "dataset/prompts/muse_stage2_quality_v1.md").read_text(encoding="utf-8")
            with closing(ChatGPTSourceClient(executable=args.codex_executable)) as client, closing(
                    CodexClient(executable=args.codex_executable)) as label_client, ThreadPoolExecutor(workers) as pool:
                if not client.account()["signed_in"]:
                    raise ValueError("GenUI requires ChatGPT sign-in")
                client.check_model(MODEL, EFFORT)
                qgen, agen = [ChatGPTSourceAdapter(client, stage=n, model=MODEL, effort=EFFORT) for n in (1, 2)]
                source_budget = args.target + max(1, args.target // 20)
                for start in itertools.count(0, args.shard_size):
                    completed = db.execute("SELECT generated,accepted,held FROM shards WHERE start=?", (start,)).fetchone()
                    shard = root / f"s{start:05d}"
                    if completed:
                        if completed[0] or completed[2]:
                            saved = checked(shard / "result.json")
                            if (completed != (saved["dispatched"], saved["accepted"], saved["held"])
                                    or file_hash(shard / "evidence.tar.gz") != saved["archive_sha256"]):
                                raise ValueError(f"Completed shard evidence changed at slot {start}")
                        continue
                    generated = db.execute("SELECT coalesce(sum(generated),0) FROM shards").fetchone()[0]
                    if generated == args.target:
                        break
                    prior_slots = db.execute("SELECT count(*) FROM items WHERE slot<?", (start,)).fetchone()[0]
                    slots = min(args.shard_size, args.target - generated, source_budget - prior_slots)
                    if slots <= 0:
                        raise RuntimeError("Source replacement budget exhausted. Inspect source holds before continuing.")
                    if shutil.disk_usage(root).free < 512 * 1024**2:
                        raise ValueError("Paused before the next shard because less than 512 MiB of disk space is free")
                    stop = start + slots
                    plans = {n: brief(n, args.seed) for n in range(start, stop)}
                    detail = f"Source slots {start + 1} through {stop}"
                    progress("Stage 1", detail)
                    done = {r[0] for r in db.execute("SELECT slot FROM items WHERE slot>=? AND slot<?", (start, stop))}
                    def accept_queries(queries, done=done, detail=detail):
                        for query in queries:
                            slot = query["slot"]
                            if slot in done:
                                continue
                            if idx.has_near_duplicate(query["query_text"]):
                                continue
                            annotated = idx.add(query)
                            with db:
                                db.execute("INSERT INTO items(slot,query) VALUES (?,?)", (slot, packed(annotated)))
                            done.add(slot)
                        progress("Stage 1", detail)

                    groups = [[plans[n] for n in range(i, min(i + 4, stop))] for i in range(start, stop, 4)]
                    groups = [group for group in groups if any(p["slot"] not in done for p in group)]
                    for queries in dispatch(pool, lambda group: query_batch(database, qgen, group, args.seed), groups, client, workers):
                        accept_queries(queries)
                    repair = [[plans[n]] for n in sorted(plans.keys() - done)]
                    for queries in dispatch(pool, lambda group: query_batch(database, qgen, group, args.seed, 1), repair, client, workers):
                        accept_queries(queries)
                    with db:
                        db.executemany("INSERT INTO items(slot,hold) VALUES (?,'invalid or duplicate query')",
                                       [(n,) for n in plans.keys() - done])
                    progress("Stage 2", detail)
                    pending = [(n, unpacked(q)) for n, q in db.execute(
                        "SELECT slot,query FROM items WHERE slot>=? AND slot<? AND source IS NULL AND hold IS NULL ORDER BY slot",
                        (start, stop))]
                    def respond(item, plans=plans):
                        n, query = item
                        return n, *answer(database, agen, query, plans[n], excluded, template)

                    for n, source, hold in dispatch(pool, respond, pending, client, workers):
                        if source:
                            identities = pilot.source_hashes(source)
                            if identities & seen:
                                source, hold = None, "duplicate source answer"
                            seen.update(identities)
                        with db:
                            db.execute("UPDATE items SET source=?,hold=? WHERE slot=?", (packed(source) if source else None, hold, n))
                        progress("Stage 2", detail)
                    rows = [unpacked(r[0]) for r in db.execute(
                        "SELECT source FROM items WHERE slot>=? AND slot<? AND source IS NOT NULL ORDER BY slot", (start, stop))]
                    progress("Stage 3", detail)
                    result = label_shard(rows, shard, label_client, workers=workers, model=MODEL,
                                         reasoning_effort=EFFORT, seed=args.seed) if rows else {"accepted": 0, "held": 0, "dispatched": 0}
                    if result["accepted"] + result["held"] != len(rows):
                        raise ValueError("Incomplete Stage 3 shard")
                    with db:
                        db.execute("INSERT INTO shards VALUES (?,?,?,?)", (start, result["dispatched"], result["accepted"], result["held"]))
                    progress("validated shard", detail)
                    if stop - start >= 16 and not result["accepted"]:
                        raise RuntimeError(f"No training-eligible examples in {detail}. Review this shard before continuing.")
            progress("complete", "Stage 3 generation target reached and all outputs assessed")
        except Exception as exc:
            progress("paused", str(exc)[:400])
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=6221122)
    parser.add_argument("--codex-executable")
    args = parser.parse_args()
    if not 1 <= args.shard_size <= 1000 or args.target < 1:
        parser.error("Use a positive target and a shard size from 1 to 1000")
    run(args)
