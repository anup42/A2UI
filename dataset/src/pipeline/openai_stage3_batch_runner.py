"""Offline capture/replay adapters around the unchanged production Stage 3 runner."""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path

from llm.base import BaseLLMAdapter, LLMResult, ModelSpec

from pipeline.v11_regeneration_io import ROOT, canonical, digest

DATASET = ROOT / "dataset"
ENVIRONMENT = {
    "DATASET_OFFLINE_MODE": "1",
    "STAGE3_PROMPT_MODE": "system_prefix",
    "STAGE3_FINAL_REGEN_ATTEMPTS": "0",
    "GENUI_STAGE3_ROLLING_DISPATCH": "0",
    "LOCAL_VLLM_RETRY_RESULT_ERRORS": "0",
}


@contextmanager
def offline_environment():
    previous = {key: os.environ.get(key) for key in ENVIRONMENT}
    os.environ.update(ENVIRONMENT)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def request_body(config: dict, prompt: str, system: str | None) -> dict:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return {"model": config["model"], "input": messages, "store": False,
            "max_output_tokens": config["max_output_tokens"],
            "reasoning": {"effort": config["reasoning_effort"]}}


def request_id(body: dict) -> str:
    return "stage3-" + digest(body)[:48]


def result(config, *, text="", error=None, raw=None, usage=None):
    usage = usage or {}
    # The canonical runner's audit contract consumes Chat-style finish metadata.
    # The original Responses payload is retained separately, unchanged.
    audit_raw = {"choices": [{"finish_reason": "stop" if not error else "length"}],
                 "batch_evidence": raw}
    return LLMResult(text=text, raw=audit_raw, latency_ms=0,
                     input_tokens=usage.get("input_tokens", 0),
                     output_tokens=usage.get("output_tokens", 0), cost_usd=None,
                     model=config["model"], provider="openai", error=error,
                     completion_complete=not error, finish_reason="stop" if not error else "length",
                     reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"))


class CaptureAdapter(BaseLLMAdapter):
    def __init__(self, config):
        super().__init__(ModelSpec("openai-stage3-batch", "openai", config["model"]))
        self.config = config
        self.requests = {}

    def generate(self, *, prompt, system, **kwargs):
        body = request_body(self.config, prompt, system)
        # Different source URLs can produce the SAME masked prompt. Preserve a
        # separate request/source binding for each; remote results still join by ID.
        custom_id = request_id(body) + f"-{len(self.requests):04d}"
        self.requests[custom_id] = {"custom_id": custom_id, "method": "POST",
                                    "url": "/v1/responses", "body": body}
        return result(self.config, error="offline_batch_capture", raw={"custom_id": custom_id})


def response_text(record: dict, config: dict) -> str:
    if record.get("error"):
        raise ValueError("batch_request_error")
    response = record.get("response") or {}
    body = response.get("body") or {}
    if response.get("status_code") != 200 or body.get("status") != "completed" or body.get("error"):
        raise ValueError("response_failed_or_incomplete")
    if body.get("model") != config["model"]:
        raise ValueError("response_model_mismatch")
    messages = []
    for item in body.get("output", []):
        if item.get("type") == "reasoning":
            continue
        if item.get("type") != "message" or item.get("role") != "assistant" or item.get("status") != "completed":
            raise ValueError("unexpected_response_output")
        for content in item.get("content", []):
            if content.get("type") != "output_text" or not isinstance(content.get("text"), str):
                raise ValueError("refusal_or_unexpected_content")
            messages.append(content["text"])
    if len(messages) != 1 or not messages[0].strip():
        raise ValueError("missing_or_ambiguous_output_text")
    return messages[0]


class ReplayAdapter(CaptureAdapter):
    def __init__(self, config, requests, records):
        super().__init__(config)
        self.requests = requests
        self.records = records
        self.seen = set()
        self.pending = defaultdict(deque)
        for custom_id, request in requests.items():
            self.pending[request_id(request["body"])].append(custom_id)

    def generate(self, *, prompt, system, **kwargs):
        body = request_body(self.config, prompt, system)
        pending = self.pending.get(request_id(body))
        if not pending:
            raise ValueError("Stage 3 replay prompt drift or duplicate call")
        custom_id = pending.popleft()
        if self.requests[custom_id]["body"] != body or custom_id in self.seen:
            raise ValueError("Stage 3 replay prompt drift or duplicate call")
        self.seen.add(custom_id)
        record = self.records.get(custom_id)
        try:
            if record is None:
                raise ValueError("missing_batch_result")
            text = response_text(record, self.config)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return result(self.config, error=str(exc), raw={"custom_id": custom_id})
        body = record["response"]["body"]
        return result(self.config, text=text, usage=body.get("usage"),
                      raw={"custom_id": custom_id, "response_id": body.get("id")})


def run(output: Path, directory: Path, config: dict, adapter: BaseLLMAdapter) -> None:
    from utils.rate_limit import RateLimiter

    from pipeline.cache import PromptCache
    from pipeline.stage3_genui import run_stage3

    directory.mkdir(parents=True)
    logger = logging.getLogger("dataset.openai_stage3_batch.offline")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    with offline_environment():
        run_stage3(
            queries_path=output / "queries.jsonl", responses_path=output / "responses.jsonl",
            prompt_path=DATASET / "prompts/genui_gen_mobile_a2ui_express_v1.md",
            prompt_supplement_path=DATASET / "prompts/v11_regeneration_v1.md",
            adapter=adapter, genui_path=directory / "genui.jsonl",
            schema_path=DATASET / "schema/canonical_ui_graph_v1.schema.json",
            artifacts_dir=directory / "artifacts", candidates_per_response=1,
            max_repair_attempts=0, max_tokens=config["max_output_tokens"],
            prompt_max_tokens=64000, seed=config["seed"], rate_limiter=RateLimiter(0),
            cache=PromptCache(directory / "cache.jsonl", enabled=False), logger=logger,
            batch_size=1, max_attempts=1, metric_version="v5_4",
            ir_formats=["a2ui_express_v1"], grpo_entity_extraction=True,
            phase_invocation_id="openai-stage3-batch-v1",
        )


def captured_bindings(directory: Path) -> dict:
    bindings = {}
    for path in (directory / "artifacts").glob("error_*.json"):
        error = json.loads(path.read_text(encoding="utf-8"))
        if error.get("error") != "offline_batch_capture":
            raise ValueError(f"Stage 3 preparation failed: {error.get('error')}")
        custom_id = error["raw_payload"]["batch_evidence"]["custom_id"]
        response_id = error["response_id"]
        if custom_id in bindings or response_id in bindings.values():
            raise ValueError("Ambiguous captured source binding")
        bindings[custom_id] = response_id
    return bindings


def write_rows(path: Path, rows) -> None:
    from pipeline.v11_regeneration_io import atomic_bytes
    atomic_bytes(path, b"".join(canonical(row) + b"\n" for row in rows))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected JSONL objects")
    return rows
