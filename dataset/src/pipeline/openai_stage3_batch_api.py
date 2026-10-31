"""Explicit paid submission and resumable collection; no implicit generation retries."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from pipeline import openai_stage3_batch as pilot
from pipeline.openai_stage3_batch_runner import read_rows
from pipeline.v11_regeneration import _exclusive
from pipeline.v11_regeneration_io import (
    atomic_bytes,
    canonical,
    checked,
    digest,
    file_hash,
    safe_path,
    save,
)

TERMINAL = {"completed", "failed", "expired", "cancelled"}


def client():
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the environment; never pass it as a CLI argument")
    # Ignore legacy adapter/base-URL/SSL overrides. A timed-out create must not
    # silently submit another billable job via SDK retries.
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1",
                  max_retries=0, timeout=120.0)


def _dict(value):
    return value.model_dump(mode="json")


def _submission(output, contract):
    state = checked(output / "submission.json")
    if state.get("contract_sha256") != digest(contract):
        raise ValueError("Submission belongs to a different prepared pilot")
    return state


def _verify_batch(batch, state):
    if (batch.get("input_file_id") != state.get("input_file_id")
            or batch.get("endpoint") != "/v1/responses"
            or batch.get("completion_window") != "24h"
            or (batch.get("metadata") or {}).get("pilot_contract") != state["contract_sha256"]):
        raise ValueError("Remote Batch identity does not match this pilot")


def submit(output_dir: Path, max_estimated_cost_usd: float, *, api=None) -> dict:
    output = safe_path(output_dir)
    with _exclusive(output):
        contract = pilot.load(output)
        estimate = json.loads((output / "estimate.json").read_text(encoding="utf-8"))
        if not math.isfinite(max_estimated_cost_usd) or max_estimated_cost_usd <= 0:
            raise ValueError("Cost allowance must be finite and positive")
        if estimate["budget_estimate_usd"] > max_estimated_cost_usd:
            raise ValueError("Estimated capped-output cost exceeds --max-estimated-cost-usd; inspect estimate.json")
        state_path = output / "submission.json"
        state = _submission(output, contract) if state_path.exists() else {"contract_sha256": digest(contract)}
        if state.get("batch_id"):
            return {"status": "already_submitted", "batch_id": state["batch_id"]}
        if state.get("create_attempted"):
            raise ValueError("Previous Batch creation outcome is unknown. Do not resubmit. Recover with status --batch-id from the OpenAI Batch dashboard")
        api = api or client()
        if not state.get("input_file_id"):
            with (output / "requests.jsonl").open("rb") as stream:
                uploaded = api.files.create(file=stream, purpose="batch")
            state["input_file_id"] = uploaded.id
        state.update(create_attempted=True, max_estimated_cost_usd=max_estimated_cost_usd)
        save(state_path, state)  # Durable intent before the potentially billable call.
        try:
            batch = _dict(api.batches.create(input_file_id=state["input_file_id"], endpoint="/v1/responses",
                         completion_window="24h", metadata={"pilot_contract": digest(contract), "workflow": pilot.VERSION}))
            _verify_batch(batch, state)
        except Exception as exc:
            raise RuntimeError("Batch creation outcome is unknown; recover using status --batch-id. No automatic resubmission") from exc
        state["batch_id"] = batch["id"]
        save(state_path, state)
        save(output / "remote_status.json", batch)
        return {"status": batch["status"], "batch_id": batch["id"], "estimated_budget_usd": estimate["budget_estimate_usd"]}


def _status(output, contract, api, batch_id=None):
    state = _submission(output, contract)
    if state.get("batch_id") and batch_id and state["batch_id"] != batch_id:
        raise ValueError("Cannot replace an existing Batch association")
    identifier = state.get("batch_id") or batch_id
    if not identifier:
        raise ValueError("No recorded Batch ID; use status --batch-id to recover an uncertain submission")
    batch = _dict(api.batches.retrieve(identifier))
    _verify_batch(batch, state)
    if batch.get("id") != identifier:
        raise ValueError("Remote Batch ID mismatch")
    state["batch_id"] = identifier
    save(output / "submission.json", state)
    save(output / "remote_status.json", batch)
    return batch


def status(output_dir: Path, batch_id=None, *, api=None) -> dict:
    output = safe_path(output_dir)
    with _exclusive(output):
        batch = _status(output, pilot.load(output), api or client(), batch_id)
        return {key: batch.get(key) for key in ("id", "status", "request_counts", "errors", "output_file_id", "error_file_id")}


def usage_report(records: list[dict], config: dict) -> dict:
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0}
    missing = 0
    for record in records:
        body = (record.get("response") or {}).get("body") or {}
        usage = body.get("usage")
        if not isinstance(usage, dict):
            missing += 1
            continue
        values = {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                  "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
                  "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}
        if (any(type(value) is not int or value < 0 for value in values.values())
                or values["cached_input_tokens"] > values["input_tokens"]
                or values["reasoning_tokens"] > values["output_tokens"]):
            raise ValueError("Malformed API usage evidence")
        for key, value in values.items():
            totals[key] += value
    prices = pilot.PRICES[config["model"]]
    cost = ((totals["input_tokens"] - totals["cached_input_tokens"]) * prices["input"]
            + totals["cached_input_tokens"] * prices["cached_input"] + totals["output_tokens"] * prices["output"]) / 1e6
    return {**totals, "records_without_usage": missing,
            "batch_cost_usd_from_reported_tokens_estimated": round(cost, 6),
            "scope": "Includes rejected/incomplete responses with reported usage; reasoning is already inside output_tokens. "
                     "Not an invoice; cache-write charges, regional premiums and taxes are not included.",
            "pricing_as_of": "2026-10-31", "pricing_url": pilot.PRICE_URL}


def collect(output_dir: Path, *, api=None) -> dict:
    output = safe_path(output_dir)
    with _exclusive(output):
        contract = pilot.load(output)
        receipt_path = output / "collection.json"
        if receipt_path.exists():
            receipt = checked(receipt_path)
            if receipt["contract_sha256"] != digest(contract):
                raise ValueError("Collection contract changed")
            for name, expected in receipt["files"].items():
                if file_hash(safe_path(output / name)) != expected:
                    raise ValueError("Collected evidence changed")
            return json.loads((output / "report.json").read_text(encoding="utf-8"))
        api = api or client()
        batch = _status(output, contract, api)
        if batch["status"] not in TERMINAL:
            return {"status": batch["status"], "message": "Batch is still running; collect later. No resubmission."}
        download_path = output / "downloads.json"
        downloads = checked(download_path) if download_path.exists() else {"batch_id": batch["id"], "files": {}}
        if downloads["batch_id"] != batch["id"]:
            raise ValueError("Download Batch identity mismatch")
        records = []
        for key in ("output_file_id", "error_file_id"):
            identifier = batch.get(key)
            if not identifier:
                continue
            name = key + ".jsonl"
            entry = downloads["files"].get(name)
            if entry:
                if entry["id"] != identifier or file_hash(output / name) != entry["sha256"]:
                    raise ValueError("Downloaded Batch evidence changed")
            else:
                content = api.files.content(identifier).content
                if not isinstance(content, bytes):
                    raise ValueError("Expected binary Batch output")
                atomic_bytes(output / name, content)
                downloads["files"][name] = {"id": identifier, "sha256": file_hash(output / name)}
                save(download_path, downloads)
            records.extend(read_rows(output / name))
        # Validate IDs before computing usage or replaying: results arrive unordered.
        pilot.index_results(records, set(contract["bindings"]))
        usage = usage_report(records, contract["config"])
        usage["requested_results_not_returned"] = len(contract["bindings"]) - len(records)
        atomic_bytes(output / "usage.json", canonical(usage) + b"\n")
        report = pilot.validate_results(output, contract, records, batch["status"])
        report["usage"] = usage
        atomic_bytes(output / "report.json", canonical(report) + b"\n")
        files = {name: file_hash(output / name) for name in
                 ("pilot_accepted.jsonl", "assessments.jsonl", "rejected.jsonl", "report.json", "usage.json")}
        files.update({name: item["sha256"] for name, item in downloads["files"].items()})
        # Seal canonical audit artifacts too; repeated collect is read-only.
        for path in (output / report["stage3_artifacts"]).rglob("*"):
            if path.is_file():
                files[path.relative_to(output).as_posix()] = file_hash(path)
        save(receipt_path, {"contract_sha256": digest(contract), "batch_id": batch["id"], "files": files})
        return report
