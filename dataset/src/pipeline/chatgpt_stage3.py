"""Separate, resumable GenUI/ChatGPT generation over canonical pilot prompts."""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

from pipeline import openai_stage3_batch as pilot
from pipeline import openai_stage3_batch_runner as runner
from pipeline.genui_codex_client import (
    CLIENT_VERSION,
    SETTINGS,
    CodexClient,
    CodexError,
    validate_auth_location,
)
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

VERSION = "genui-chatgpt-stage3-v1"


def load(output):
    contract = pilot.load(output)
    if contract["config"].get("transport") != "chatgpt":
        raise ValueError("Prepare a fresh ChatGPT run; existing API Batch runs cannot be repurposed")
    return contract


def response_text(record, config):
    if (record.get("transport") != "chatgpt" or record.get("status") != "completed"
            or record.get("model") != config["model"] or not record.get("thread_id")
            or not record.get("turn_id") or not isinstance(record.get("text"), str)
            or not record["text"].strip()):
        raise ValueError("chatgpt_result_missing_failed_or_incomplete")
    return record["text"]


def token_usage(raw):
    if raw is None:
        return None
    mapping = {"inputTokens": "input_tokens", "outputTokens": "output_tokens",
               "cachedInputTokens": "cached_input_tokens", "reasoningOutputTokens": "reasoning_tokens"}
    if not isinstance(raw, dict) or any(type(raw.get(key)) is not int or raw[key] < 0 for key in mapping):
        raise ValueError("Malformed Codex token usage")
    if raw["cachedInputTokens"] > raw["inputTokens"] or raw["reasoningOutputTokens"] > raw["outputTokens"]:
        raise ValueError("Inconsistent Codex token usage")
    return {name: raw[key] for key, name in mapping.items()}


class ReplayAdapter(runner.CaptureAdapter):
    def __init__(self, config, requests, records):
        super().__init__(config)
        self.requests, self.records, self.seen = requests, records, set()
        self.pending = defaultdict(deque)
        for identifier, request in requests.items():
            self.pending[runner.request_id(request["body"])].append(identifier)

    def generate(self, *, prompt, system, **kwargs):
        body = runner.request_body(self.config, prompt, system)
        pending = self.pending.get(runner.request_id(body))
        if not pending:
            raise ValueError("ChatGPT replay prompt drift or duplicate call")
        identifier = pending.popleft()
        if identifier in self.seen or self.requests[identifier]["body"] != body:
            raise ValueError("ChatGPT replay prompt binding changed")
        self.seen.add(identifier)
        record = self.records.get(identifier, {})
        try:
            text = response_text(record, self.config)
            if record.get("request_sha256") != digest(self.requests[identifier]):
                raise ValueError("ChatGPT result/request binding changed")
            usage = token_usage(record.get("usage")) or {}
        except ValueError as exc:
            return runner.result(self.config, error=str(exc), raw={"custom_id": identifier, "transport": "chatgpt"})
        return runner.result(self.config, text=text, usage={**usage,
            "output_tokens_details": {"reasoning_tokens": usage.get("reasoning_tokens")}},
            raw={"custom_id": identifier, "transport": "chatgpt", "thread_id": record["thread_id"],
                 "turn_id": record["turn_id"]})


def _state(output, contract):
    path = output / "generation.json"
    state = checked(path) if path.exists() else {"version": VERSION, "contract_sha256": digest(contract),
        "client_version": CLIENT_VERSION, "settings_sha256": digest(SETTINGS), "attempts": {},
        "created_at": time.time(), "status": "ready"}
    if (state.get("contract_sha256") != digest(contract) or state.get("version") != VERSION
            or state.get("client_version") != CLIENT_VERSION or state.get("settings_sha256") != digest(SETTINGS)):
        raise ValueError("ChatGPT run contract/runtime changed")
    if set(state["attempts"]) - set(contract["bindings"]):
        raise ValueError("Unknown ChatGPT request identity")
    return state


def records(output, contract, state):
    requests = {r["custom_id"]: r for r in runner.read_rows(output / "requests.jsonl")}
    found = []
    for identifier, attempt in state["attempts"].items():
        path = safe_path(output / "chatgpt_records" / (identifier + ".json"))
        if path.parent != output / "chatgpt_records":
            raise ValueError("Invalid result identity")
        if not path.exists():
            if attempt.get("record_sha256"):
                raise ValueError("Recorded ChatGPT result is missing")
            continue
        record = checked(path)
        if attempt.get("record_sha256") and file_hash(path) != attempt["record_sha256"]:
            raise ValueError("ChatGPT result changed")
        if (record.get("custom_id") != identifier or record.get("contract_sha256") != digest(contract)
                or record.get("request_sha256") != digest(requests[identifier])):
            raise ValueError("ChatGPT source/request/result binding mismatch")
        if record.get("status") == "completed":
            response_text(record, contract["config"])
            token_usage(record.get("usage"))
        elif record.get("status") not in {"unknown", "skipped"}:
            raise ValueError("Unexpected ChatGPT result state")
        found.append(record)
    return found


def usage_report(found):
    totals = dict.fromkeys(("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"), 0)
    missing = 0
    for record in found:
        usage = token_usage(record.get("usage"))
        if usage is None:
            missing += 1
        else:
            for key, value in usage.items():
                totals[key] += value
    return {**totals, "records_without_usage": missing, "billing": "chatgpt_subscription", "cost_usd": None,
            "scope": "Reported Codex tokens only; reasoning is included in output. Subscription limits apply. Not API Batch pricing or an invoice."}


def _progress(output, state, requested, found, active=None):
    counts = Counter(item["status"] for item in found)
    atomic_bytes(output / "progress.json", canonical({"status": state["status"], "requested": requested,
        "completed": counts["completed"], "skipped": counts["skipped"], "uncertain": counts["unknown"],
        "attempted": len(state["attempts"]), "active_custom_id": active, "updated_at": time.time(),
        "started_at": state["created_at"], "usage": usage_report(found)}) + b"\n")


def run(output_dir: Path, *, client=None, auth_home=None, timeout=600, skip_uncertain=False):
    if not 30 <= timeout <= 3600:
        raise ValueError("Per-sample timeout must be 30..3600 seconds")
    output = safe_path(output_dir)
    validate_auth_location(output, auth_home)
    load(output)  # Refuse other transports before even creating a lock file.
    owned_client = client is None
    with _exclusive(output):
        contract = load(output)
        if (output / "chatgpt_collection.json").exists():
            return {"status": "already_collected"}
        state = _state(output, contract)
        found = records(output, contract, state)
        indexed = {item["custom_id"]: item for item in found}
        requests = {item["custom_id"]: item for item in runner.read_rows(output / "requests.jsonl")}
        (output / "chatgpt_records").mkdir(exist_ok=True)
        # A complete durable response can be recovered after a crash before its
        # journal acknowledgement. Never issue a second model turn for it.
        for identifier, attempt in state["attempts"].items():
            record = indexed.get(identifier)
            if record and record["status"] in {"completed", "skipped"}:
                attempt.update(status=record["status"], record_sha256=file_hash(output / "chatgpt_records" / (identifier + ".json")))
                continue
            if not skip_uncertain:
                raise ValueError("Previous generation outcome is uncertain; inspect it, then use --skip-uncertain to skip it without regeneration")
            record = dict(record or {}, custom_id=identifier, transport="chatgpt", status="skipped",
                          previous_status="unknown", contract_sha256=digest(contract), request_sha256=digest(requests[identifier]))
            path = output / "chatgpt_records" / (identifier + ".json")
            if path.exists():
                save(output / "chatgpt_records" / (identifier + ".uncertain.json"), indexed[identifier])
            save(path, record)
            indexed[identifier] = record
            attempt.update(status="skipped", record_sha256=file_hash(path))
        save(output / "generation.json", state)
        pending = [item for key, item in requests.items() if key not in indexed]
        if not pending:
            state["status"] = "generated"
            save(output / "generation.json", state)
            _progress(output, state, len(requests), list(indexed.values()))
            return {"status": "generated", "requests": len(requests)}
        client = client or CodexClient(auth_home)
        try:
            if not client.account()["signed_in"]:
                raise ValueError("Sign in with ChatGPT through GenUI before generation")
            client.check_model(contract["config"]["model"], contract["config"]["reasoning_effort"])
            if state.get("codex_version", client.version) != client.version:
                raise ValueError("Codex CLI version changed during a prepared generation run")
            state["codex_version"] = client.version
            for request in pending:
                load(output)
                identifier = request["custom_id"]
                state["status"] = "generating"
                attempt = state["attempts"][identifier] = {"status": "inflight", "started_at": time.time()}
                save(output / "generation.json", state)  # Durable intent BEFORE any model request.
                _progress(output, state, len(requests), list(indexed.values()), identifier)
                def on_thread(thread_id, attempt=attempt):
                    attempt["thread_id"] = thread_id
                    save(output / "generation.json", state)
                try:
                    reply = client.generate(request["body"], timeout=timeout, on_thread=on_thread)
                    response_text(dict(reply, transport="chatgpt"), contract["config"])
                    token_usage(reply.get("usage"))
                except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
                    # Provider may already have consumed allowance. Never retry
                    # an error/timeout automatically or label it a quality fail.
                    reply = {"status": "unknown", "error": "generation_outcome_uncertain",
                             "diagnostic": str(exc) if isinstance(exc, CodexError) else type(exc).__name__}
                record = dict(reply, transport="chatgpt", custom_id=identifier,
                              contract_sha256=digest(contract), request_sha256=digest(request), finished_at=time.time())
                path = output / "chatgpt_records" / (identifier + ".json")
                save(path, record)
                indexed[identifier] = record
                attempt.update(status=record["status"], record_sha256=file_hash(path))
                if record.get("diagnostic"):
                    attempt["diagnostic"] = record["diagnostic"]
                state["status"] = "paused" if record["status"] == "unknown" else "generating"
                save(output / "generation.json", state)
                _progress(output, state, len(requests), list(indexed.values()))
                if state["status"] == "paused":
                    return {"status": "paused", "reason": "Generation outcome uncertain; no retry was made", "custom_id": identifier}
            state["status"] = "generated"
            save(output / "generation.json", state)
            _progress(output, state, len(requests), list(indexed.values()))
            return {"status": "generated", "requests": len(requests)}
        finally:
            if owned_client:
                client.close()


def collect(output_dir: Path):
    output = safe_path(output_dir)
    load(output)
    with _exclusive(output):
        contract = load(output)
        receipt_path = output / "chatgpt_collection.json"
        if receipt_path.exists():
            receipt = checked(receipt_path)
            if receipt["contract_sha256"] != digest(contract):
                raise ValueError("Collection contract changed")
            for name, sha in receipt["files"].items():
                if file_hash(safe_path(output / name)) != sha:
                    raise ValueError("Collected GenUI evidence changed")
            return json.loads((output / "report.json").read_text(encoding="utf-8"))
        state = _state(output, contract)
        found = records(output, contract, state)
        if {item["custom_id"] for item in found} != set(contract["bindings"]) or any(item["status"] == "unknown" for item in found):
            raise ValueError("Generation is incomplete or uncertain; no partial publication")
        state["status"] = "validating"
        save(output / "generation.json", state)
        _progress(output, state, len(contract["bindings"]), found)
        report = pilot.validate_results(output, contract, found, "completed")
        report["usage"] = usage_report(found)
        atomic_bytes(output / "usage.json", canonical(report["usage"]) + b"\n")
        atomic_bytes(output / "report.json", canonical(report) + b"\n")
        state["status"] = "validated"
        save(output / "generation.json", state)
        _progress(output, state, len(contract["bindings"]), found)
        names = ["report.json", "usage.json", "assessments.jsonl", "pilot_accepted.jsonl", "rejected.jsonl", "generation.json", "progress.json"]
        paths = [output / name for name in names]
        paths += list((output / "chatgpt_records").glob("*.json"))
        paths += [p for p in (output / report["stage3_artifacts"]).rglob("*") if p.is_file()]
        save(receipt_path, {"version": VERSION, "contract_sha256": digest(contract),
                           "files": {p.relative_to(output).as_posix(): file_hash(p) for p in paths}})
        return report
