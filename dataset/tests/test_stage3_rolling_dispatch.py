"""CPU-only coverage of local HTTP Stage 3 dispatch; no server is contacted."""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from threading import Event, Lock

import pytest

DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from llm.base import BaseLLMAdapter, LLMResult, ModelSpec
from pipeline.cache import PromptCache
from pipeline.ir_formats import A2UI_EXPRESS_V1, encode_from_flat_spec
from pipeline.stage3_genui import run_stage3
from utils.rate_limit import RateLimiter


class Adapter(BaseLLMAdapter):
    """Deterministic local-HTTP-shaped fixture, also used by aggregate tests."""

    def __init__(self, *, before_generate=None, failures=()):
        super().__init__(ModelSpec("fake-http", "local", "fake-model",
                                   endpoint="http://unused.invalid/v1/chat/completions"))
        self.before_generate = before_generate
        self.failures = set(failures)
        self.started = []
        self.active = 0
        self.peak_active = 0
        self.lock = Lock()

    def generate(self, prompt, system, temperature, max_tokens, seed, json_mode=False):
        marker = int(re.search(r"MARKER_(\d+)", prompt).group(1))
        with self.lock:
            self.started.append(marker)
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            if self.before_generate is not None:
                self.before_generate(marker)
            spec = {
                "root": "screen", "state": {}, "elements": {
                    "screen": {"type": "Stack", "props": {"direction": "vertical"},
                               "children": ["title", "body"]},
                    "title": {"type": "Text", "props": {"text": "Trip", "variant": "h2"},
                              "children": []},
                    "body": {"type": "Text", "props": {
                        "text": f"Train at 09:30 costs $24. Direct service MARKER_{marker}."},
                        "children": []},
                },
            }
            text = encode_from_flat_spec(spec, A2UI_EXPRESS_V1)
            return LLMResult(text=text, raw={"text": text}, latency_ms=1.0,
                             input_tokens=20, output_tokens=30, cost_usd=None,
                             model=self.spec.model, provider=self.spec.provider,
                             error="fixture generation failure" if marker in self.failures else None)
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture(autouse=True)
def rolling_environment(monkeypatch):
    monkeypatch.setenv("GENUI_STAGE3_ROLLING_DISPATCH", "1")
    monkeypatch.setenv("LOCAL_ALLOW_HTTP_ENDPOINT", "1")
    monkeypatch.setenv("LOCAL_VLLM_STRICT_OFFLINE", "0")
    monkeypatch.setenv("LOCAL_VLLM_RETRY_RESULT_ERRORS", "0")
    monkeypatch.setenv("LOCAL_VLLM_BATCH_PARALLELISM", "2")
    monkeypatch.setenv("GENUI_STAGE3_ROLLING_BUFFER", "2")
    monkeypatch.setenv("STAGE3_FINAL_REGEN_ATTEMPTS", "0")
    monkeypatch.delenv("GENUI_STAGE3_AGGREGATE_WORK_DIR", raising=False)


def _run(tmp_path, adapter, *, markers=(0, 1, 2), max_total=None, output_name="genui.jsonl", cache=None):
    responses = tmp_path / "responses.jsonl"
    queries = tmp_path / "queries.jsonl"
    responses.write_text("".join(json.dumps({
        "response_id": f"r_{marker}", "query_id": f"q_{marker}", "n_idx": 1,
        "response_text": f"Trip train at 09:30 costs $24. Direct service MARKER_{marker}.",
    }) + "\n" for marker in markers), encoding="utf-8")
    queries.write_text("".join(json.dumps({"query_id": f"q_{marker}", "intent": "travel", "tags": []})
                               + "\n" for marker in dict.fromkeys(markers)), encoding="utf-8")
    output = tmp_path / output_name
    run_stage3(
        queries_path=queries, responses_path=responses, adapter=adapter,
        prompt_path=DATASET_ROOT / "prompts/genui_gen_mobile_a2ui_express_v1.md",
        schema_path=DATASET_ROOT / "schema/canonical_ui_graph_v1.schema.json",
        genui_path=output, artifacts_dir=tmp_path / "artifacts",
        candidates_per_response=1, max_repair_attempts=0, max_tokens=2048,
        prompt_max_tokens=None, batch_size=2, seed=1, rate_limiter=RateLimiter(0),
        cache=cache if cache is not None else PromptCache(tmp_path / "cache.jsonl", enabled=False),
        logger=logging.getLogger("stage3-rolling-test"), max_total=max_total, max_attempts=1,
        metric_version="legacy", ir_formats=[A2UI_EXPRESS_V1],
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert all(row["record_status"] == "accepted" for row in rows)
    return rows


def test_rolling_refills_before_slowest_request_finishes(tmp_path):
    third_started = Event()

    def order(marker):
        if marker == 0:
            assert third_started.wait(10), "dispatcher waited for the slowest request before refilling"
        elif marker == 2:
            third_started.set()

    adapter = Adapter(before_generate=order)
    rows = _run(tmp_path, adapter)
    assert third_started.is_set()
    assert adapter.peak_active == 2
    assert sorted(adapter.started) == [0, 1, 2]
    assert {row["response_id"] for row in rows} == {"r_0", "r_1", "r_2"}


def test_rolling_never_submits_more_than_max_total(tmp_path):
    adapter = Adapter()
    rows = _run(tmp_path, adapter, markers=range(8), max_total=2)
    assert len(rows) == 2
    assert sorted(adapter.started) == [0, 1]
    assert adapter.peak_active <= 2


def test_rolling_skips_duplicate_inflight_and_resumed_ids(tmp_path):
    first = Adapter()
    rows = _run(tmp_path, first, markers=(0, 0, 1, 2), max_total=2)
    assert sorted(first.started) == [0, 1]
    assert len(rows) == 2
    resumed = Adapter()
    rows = _run(tmp_path, resumed, markers=(0, 0, 1, 2), max_total=1)
    assert resumed.started == [2]
    assert len({row["ui_id"] for row in rows}) == len(rows) == 3


def test_rolling_failed_request_releases_slot_without_counting_as_output(tmp_path):
    adapter = Adapter(failures=(0,))
    rows = _run(tmp_path, adapter, max_total=2)
    assert sorted(adapter.started) == [0, 1, 2]
    assert {row["response_id"] for row in rows} == {"r_1", "r_2"}
    errors = list((tmp_path / "artifacts").glob("error_*.json"))
    assert any("fixture generation failure" in path.read_text(encoding="utf-8") for path in errors)


def test_rolling_retries_only_transiently_failed_item_and_preserves_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_VLLM_RETRY_RESULT_ERRORS", "1")
    monkeypatch.setenv("LOCAL_VLLM_RETRY_MAX_SECONDS", "30")
    monkeypatch.setattr("pipeline.stage3_genui.time.sleep", lambda _: None)

    class RetryAdapter(Adapter):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            with self.lock:
                if self.started.count(0) == 1 and "MARKER_0" in result.text:
                    result.error = "missing_muse_reasoning_content"
            return result

    adapter = RetryAdapter()
    rows = _run(tmp_path, adapter, markers=(0, 1), max_total=2)
    assert sorted(adapter.started) == [0, 0, 1]
    retried = next(row for row in rows if row["response_id"] == "r_0")
    assert len(retried["generation_attempts"]) == 2
    assert len(rows) == 2


def test_rolling_preserves_cache_hits_without_new_requests(tmp_path):
    cache = PromptCache(tmp_path / "cache.jsonl", enabled=True)
    _run(tmp_path, Adapter(), markers=(0, 1), cache=cache)
    cached_adapter = Adapter()
    rows = _run(tmp_path, cached_adapter, markers=(0, 1), output_name="cached.jsonl", cache=cache)
    assert cached_adapter.started == []
    assert len(rows) == 2
