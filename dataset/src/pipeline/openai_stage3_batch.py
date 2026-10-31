"""Opt-in, train-only OpenAI Batch pilot; no changes to Muse or training defaults."""
from __future__ import annotations

import hashlib
import heapq
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

from pipeline import openai_stage3_batch_runner as runner
from pipeline import v11_regeneration as existing
from pipeline import v11_regeneration_guard as guard
from pipeline.source_quality import assess_source_quality
from pipeline.training_augmentation import admission_errors
from pipeline.v11_regeneration_io import (
    ROOT,
    SplitInput,
    atomic_bytes,
    canonical,
    checked,
    digest,
    file_hash,
    safe_path,
    save,
)

VERSION = "openai-stage3-batch-v1"
# Per million tokens, Batch short-context prices verified 2026-10-31.
PRICES = {"gpt-6.1-sol": {"input": 1.0, "output": 5.0, "cached_input": 0.05}}
PRICE_URL = "https://developers.openai.com/api/docs/pricing"


def fingerprint() -> dict:
    paths = {Path(__file__), ROOT / "dataset/scripts/openai_stage3_batch.py",
             ROOT / "training/data/train/v11/restore.py",
             ROOT / "training/src/ir_training/pipeline/golden_training.py",
             ROOT / "dataset/scripts/capture_android_run_screenshots.py",
             ROOT / "dataset/tests/fixtures/flat_expr_parity_vectors_v5_4.json",
             ROOT / "dataset/tests/fixtures/a2ui_express_conformance_v1.json",
             ROOT / "android/app/src/test/resources/flat_expr_parity_vectors_v5_4.json",
             ROOT / "android/app/src/test/resources/a2ui_express_conformance_v1.json",
             guard.SOURCE_HOLDS_PATH, ROOT / "training/data/quality/v11_review_holds_20261001.json"}
    for directory in ("dataset/src/pipeline", "dataset/src/llm", "dataset/src/utils",
                      "training/src/ir_training/data", "training/src/ir_training/eval"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update((ROOT / "android/app/src/main/java/com/samsung/genuicraft/renderer").rglob("*.kt"))
    for directory, glob in (("dataset/schema", "*.json"), ("dataset/prompts", "*.md"),
                            ("dataset/configs", "*.yaml"), ("dataset/configs", "*.json")):
        paths.update((ROOT / directory).rglob(glob))
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sorted(paths)}


def source_aliases(row: dict) -> set[str]:
    meta = row.get("metadata") or {}
    return {str(value) for value in (existing._row_id(row), row.get("source_id"),
            row.get("response_id"), meta.get("source_id"), meta.get("response_id"), meta.get("ui_id")) if value}


def source_hashes(row: dict) -> set[str]:
    return {digest(" ".join(text.split())) for text in (guard.source_text(row), existing._generation_source(row))}


def query_for(row: dict) -> dict:
    return {"query_text": (row.get("metadata") or {}).get("query_text", "Render the supplied response faithfully.")}


def select(source: SplitInput, samples: int, seed: int) -> tuple[list[dict], dict, dict]:
    excluded_ids, excluded_sources, benchmark_files = existing._benchmark_exclusions()
    counts = Counter()
    # Validation sources/aliases are excluded even if renamed in training.
    for _, _, row in source.rows("val"):
        excluded_ids.update(source_aliases(row))
        excluded_sources.update(source_hashes(row))
        counts["validation_rows_excluded"] += 1
    heap, ids, seen_sources = [], set(), set()
    for line, raw, row in source.rows("train"):
        counts["train_rows_scanned"] += 1
        row_id = existing._row_id(row)
        if row_id in ids:
            raise ValueError("Duplicate training row identity")
        ids.add(row_id)
        if (row.get("metadata") or {}).get("assigned_split", "train") != "train":
            raise ValueError("Training input contains a non-training split")
        aliases, hashes = source_aliases(row), source_hashes(row)
        if aliases & excluded_ids or hashes & excluded_sources:
            counts["heldout_overlap_excluded"] += 1
            continue
        if hashes & seen_sources:
            counts["duplicate_sources_excluded"] += 1
            continue
        seen_sources.update(hashes)
        priority = int(digest({"seed": seed, "id": row_id}), 16)
        if len(heap) == samples and priority >= -heap[0][0]:
            continue
        if guard.assess_source(row).get("eligible") is not True:
            counts["sampled_source_holds_excluded"] += 1
            continue
        source_quality = assess_source_quality(query_for(row), existing._generation_source(row))
        if source_quality["status"] == "failed" or source_quality.get("training_eligibility") == "exclude":
            counts["sampled_source_contract_excluded"] += 1
            continue
        entry = {"source_id": row_id, "line": line, "original_split": "train", "row": row,
                 "original_row_sha256": hashlib.sha256(raw).hexdigest(),
                 "response_id": "p-" + digest(row)[:24]}
        item = (-priority, row_id, entry)
        if len(heap) < samples:
            heapq.heappush(heap, item)
        else:
            heapq.heapreplace(heap, item)
    source.recheck_sidecars()
    if len(heap) != samples:
        raise ValueError(f"Only {len(heap)} eligible distinct training sources; requested {samples}")
    return [entry for _, _, entry in sorted(heap, reverse=True)], dict(counts), benchmark_files


def estimate(requests: list[dict], config: dict) -> dict:
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(config["model"])
        method = "tiktoken_model_encoding_estimate_plus_framing"
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
        method = "o200k_base_proxy_estimate_plus_framing_not_exact_model_tokenizer"
    per_request = [sum(len(encoding.encode(message["content"], disallowed_special=()))
                       for message in req["body"]["input"]) + 32 for req in requests]
    if max(per_request) > 64000:
        raise ValueError("Prompt exceeds pilot input budget; source was not truncated")
    total = sum(per_request)
    rates = PRICES[config["model"]]
    output_cap = len(requests) * config["max_output_tokens"]
    # Estimates, not an API billing cap. Allow tokenizer variance and cache writes.
    budget = (total * 1.25 * rates["input"] * 1.25 + output_cap * rates["output"]) / 1e6
    return {"model": config["model"], "requests": len(requests), "pricing_as_of": "2026-10-31",
            "pricing_url": PRICE_URL, "batch_usd_per_million_tokens": rates,
            "token_estimation_method": method, "estimated_input_tokens": total,
            "max_estimated_input_tokens_per_request": max(per_request),
            "output_token_cap_including_reasoning": output_cap,
            "scenarios": [{"average_output_tokens_including_reasoning": count,
                           "total_tokens_estimated": total + len(requests) * count,
                           "batch_cost_usd_estimated": round((total * rates["input"] + len(requests) * count * rates["output"]) / 1e6, 4)}
                          for count in (2048, 4096, config["max_output_tokens"]) if count <= config["max_output_tokens"]],
            "budget_estimate_usd": round(budget, 4),
            "limits": "Estimates, not a provider spending cap; no cached-read discount assumed. "
                      "Budget includes 25% input token margin and 25% cache-write allowance. "
                      "Excludes taxes/regional premiums; actual usage includes reasoning tokens. "
                      "Batch reduces price, not token usage. No paid retries."}


def prepare(input_dir: Path, output_dir: Path, *, samples=1000, seed=123,
            model="gpt-6.1-sol", reasoning_effort="medium", max_output_tokens=8192) -> dict:
    from pipeline.genui_quality.validation_v5_4 import ensure_v5_4_validation_ready

    if not 1 <= samples <= 1000 or seed < 0 or not 1024 <= max_output_tokens <= 32768:
        raise ValueError("Pilot requires 1..1000 samples, nonnegative seed and 1024..32768 output tokens")
    if model not in PRICES or reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError("Unsupported pilot model or reasoning effort")
    output, source = safe_path(output_dir), SplitInput(input_dir)
    if output == source.directory or source.directory in output.parents or output in source.directory.parents:
        raise ValueError("Pilot output must be separate from its source dataset")
    if output.exists():
        raise ValueError("Prepare requires a fresh output directory; use status/collect for an existing pilot")
    # Check the canonical evaluator's release dependencies BEFORE any paid job.
    ensure_v5_4_validation_ready()
    probe = guard.assess_candidate(
        {"id": "openai-stage3-local-evaluator-probe", "response_text": "Meeting at 11:00."},
        '<a2ui>root=Text("Meeting at 11:00.")</a2ui>')
    if probe.get("eligible") is not True:
        raise ValueError(f"Canonical evaluator preflight failed: {probe.get('reasons')}")
    config = {"model": model, "reasoning_effort": reasoning_effort,
              "max_output_tokens": max_output_tokens, "seed": seed, "samples": samples}
    code_hashes = fingerprint()
    selected, counts, benchmark_files = select(source, samples, seed)
    output.mkdir(parents=True, exist_ok=False)
    runner.write_rows(output / "selected.jsonl", selected)
    runner.write_rows(output / "responses.jsonl", (
        {"query_id": entry["response_id"], "response_id": entry["response_id"], "n_idx": 1,
         "response_text": existing._generation_source(entry["row"])} for entry in selected))
    runner.write_rows(output / "queries.jsonl", (
        {"query_id": entry["response_id"], **query_for(entry["row"])} for entry in selected))
    adapter = runner.CaptureAdapter(config)
    runner.run(output, output / "capture", config, adapter)
    bindings = runner.captured_bindings(output / "capture")
    if set(bindings) != set(adapter.requests) or set(bindings.values()) != {entry["response_id"] for entry in selected}:
        raise ValueError("Stage 3 did not capture exactly one complete request per selected source")
    requests = list(adapter.requests.values())
    runner.write_rows(output / "requests.jsonl", requests)
    if (output / "requests.jsonl").stat().st_size >= 200_000_000:
        raise ValueError("Batch input exceeds 200 MB; prepare a smaller pilot")
    costs = estimate(requests, config)
    atomic_bytes(output / "estimate.json", canonical(costs) + b"\n")
    if fingerprint() != code_hashes:
        raise ValueError("Code/prompt/schema changed during preparation")
    files = {name: file_hash(output / name) for name in
             ("requests.jsonl", "selected.jsonl", "responses.jsonl", "queries.jsonl", "estimate.json")}
    contract = {"version": VERSION, "config": config, "files": files, "bindings": bindings,
                "input_identities": source.identities, "source_directory": str(source.directory),
                "counts": counts, "benchmark_files": benchmark_files, "code_hashes": code_hashes,
                "guard": guard.guard_identity(), "environment": runner.ENVIRONMENT}
    save(output / "contract.json", contract)
    return {"status": "prepared_offline_no_api_calls", "output_dir": str(output), **costs}


def load(output: Path) -> dict:
    contract = checked(output / "contract.json")
    if contract["version"] != VERSION or contract["code_hashes"] != fingerprint() or contract["guard"] != guard.guard_identity():
        raise ValueError("Bound code/prompt/schema/policy changed; use the original code for this pilot")
    for name, expected in contract["files"].items():
        if file_hash(safe_path(output / name)) != expected:
            raise ValueError(f"Prepared pilot file changed: {name}")
    for name, expected in contract["benchmark_files"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("Pinned benchmark exclusion policy changed")
    return contract


def index_results(records: list[dict], expected: set[str]) -> dict:
    result = {}
    for record in records:
        key = record.get("custom_id")
        if not isinstance(key, str) or key not in expected or key in result:
            raise ValueError("Unknown or duplicate Batch custom_id")
        result[key] = record
    return result


def validate_results(output: Path, contract: dict, records: list[dict], batch_status: str) -> dict:
    from ir_training.data.express_preparation import prepare_row, serialize_checked
    from ir_training.data.reference_binding import bind_source_target_references

    from pipeline.ir_formats import express

    config = contract["config"]
    requests = {row["custom_id"]: row for row in runner.read_rows(output / "requests.jsonl")}
    indexed = index_results(records, set(requests))
    adapter = runner.ReplayAdapter(config, requests, indexed)
    # A fresh disposable directory permits safe local revalidation after interruption.
    directory = Path(tempfile.mkdtemp(prefix="v-", dir=output))
    replay = directory / "stage3"
    runner.run(output, replay, config, adapter)
    if adapter.seen != set(requests):
        raise ValueError("Canonical Stage 3 replay did not consume all prepared requests")
    generated = {}
    if (replay / "genui.jsonl").exists():
        for row in runner.read_rows(replay / "genui.jsonl"):
            key = row["response_id"]
            if key in generated or key not in contract["bindings"].values():
                raise ValueError("Ambiguous canonical Stage 3 candidate")
            generated[key] = row
    by_source = {value: key for key, value in contract["bindings"].items()}
    accepted, assessments = [], []
    for entry in runner.read_rows(output / "selected.jsonl"):
        source_row, response_id = entry["row"], entry["response_id"]
        custom_id = by_source[response_id]
        candidate = generated.get(response_id)
        assessment = {"source_id": entry["source_id"], "custom_id": custom_id, "eligible": False}
        try:
            runner.response_text(indexed.get(custom_id, {}), config)
            if candidate is None:
                raise ValueError("canonical_stage3_rejected")
            reasons = admission_errors(candidate, existing._generation_source(source_row))
            if reasons:
                raise ValueError(";".join(reasons))
            # Stage 3 retains opaque tokens in its Express text, alongside the
            # canonical reference map. Restore through the proven binding gate
            # before rebinding to the ORIGINAL training row's placeholder names.
            parsed = serialize_checked(candidate["a2ui_express"], "root-first")
            bound = bind_source_target_references(candidate["response_text"], parsed.graph, candidate)
            target = existing._candidate_target(source_row, {
                "a2ui_express": express.encode(bound.graph, shorten_ids=False)})
            assessment.update(guard.assess_candidate(source_row, target))
            if assessment["eligible"]:
                updated = deepcopy(source_row)
                updated["completion"] = updated["a2ui_express"] = target
                updated["completion_targets"] = {"a2ui_express_v1": target}
                if not updated.get("messages") or updated["messages"][-1].get("role") != "assistant":
                    raise ValueError("missing_original_assistant_target")
                updated["messages"][-1]["content"] = target
                updated.setdefault("metadata", {})["openai_stage3_batch"] = {
                    "contract_sha256": digest(contract), "custom_id": custom_id,
                    "original_row_sha256": entry["original_row_sha256"], "original_split": "train",
                    "source_factuality_verified": False}
                prepared, _, _ = prepare_row(updated, "root-first")
                if guard.source_text(prepared) != guard.source_text(source_row) or existing._row_id(prepared) != entry["source_id"]:
                    raise ValueError("immutable_source_identity_changed")
                accepted.append(prepared)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            assessment.update(eligible=False, reasons=[str(exc)])
        assessments.append(assessment)
    # Recheck before publication; the accepted file is a pilot subset, never a full dataset.
    load(output)
    runner.write_rows(output / "pilot_accepted.jsonl", accepted)
    runner.write_rows(output / "assessments.jsonl", assessments)
    runner.write_rows(output / "rejected.jsonl", (item for item in assessments if not item["eligible"]))
    report = {"status": "pilot_validated", "batch_status": batch_status, "requested": len(requests),
              "results_received": len(indexed), "missing_results": len(requests) - len(indexed),
              "accepted": len(accepted), "rejected_or_missing": len(requests) - len(accepted),
              "training_input_ready": False, "scope": "train-only pilot subset; review before dataset composition",
              "stage3_artifacts": replay.relative_to(output).as_posix(),
              "rejection_reasons": dict(Counter(reason for item in assessments if not item["eligible"]
                                                 for reason in item.get("reasons", []))),
              "contract_sha256": digest(contract)}
    atomic_bytes(output / "report.json", canonical(report) + b"\n")
    return report
