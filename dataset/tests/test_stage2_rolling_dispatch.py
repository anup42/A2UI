"""Offline regression coverage for the Stage 2 local-HTTP rolling scheduler."""
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
from pipeline import stage2_responses as stage2
from pipeline.cache import PromptCache
from utils.rate_limit import RateLimiter


class Adapter(BaseLLMAdapter):
    def __init__(self, *, before_generate=None, failures=()):
        super().__init__(ModelSpec("fake-http", "local", "fake-model",
                                   endpoint="http://unused.invalid/v1/chat/completions"))
        self.before_generate = before_generate
        self.failures = set(failures)
        self.started = []
        self.active = 0
        self.peak_active = 0
        self.batch_calls = 0
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
            batch_match = re.search(r"Return exactly (\d+) distinct responses", prompt)
            count = int(batch_match.group(1)) if batch_match else 1
            texts = [f"Result MARKER_{marker}: example {index} is available." for index in range(count)]
            text = json.dumps(texts) if batch_match else texts[0]
            return LLMResult(text=text, raw={"text": text}, latency_ms=1.0,
                             input_tokens=20, output_tokens=30, cost_usd=None,
                             model=self.spec.model, provider=self.spec.provider,
                             error="fixture generation failure" if marker in self.failures else None)
        finally:
            with self.lock:
                self.active -= 1

    def generate_batch(self, *, prompts, system, temperature, max_tokens, seeds,
                       json_mode=False, batch_name=None):
        self.batch_calls += 1
        return [self.generate(prompt, system, temperature, max_tokens, seed, json_mode)
                for prompt, seed in zip(prompts, seeds)]


@pytest.fixture(autouse=True)
def rolling_environment(monkeypatch):
    for name, value in {
        "STAGE2_ROLLING_DISPATCH": "1", "LOCAL_ALLOW_HTTP_ENDPOINT": "1",
        "LOCAL_VLLM_STRICT_OFFLINE": "0", "LOCAL_VLLM_BATCH_PARALLELISM": "2",
        "LOCAL_VLLM_PARALLEL_REQUESTS": "2", "DATASET_OFFLINE_MODE": "1",
        "INTERNET": "0", "STAGE2_ASSET_WORKERS": "1",
        "STAGE2_REAL_ASSET_RETRY_ENABLED": "0", "STAGE2_COMMONS_ENRICHMENT_ENABLED": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(stage2, "_load_local_icon_context", lambda *_: None)

    def no_network(*args, **kwargs):
        pytest.fail("rolling scheduler fixture must not access the network")

    monkeypatch.setattr(stage2, "urlopen", no_network)


def _run(tmp_path, adapter, *, markers=(0, 1, 2), max_total=None,
         n_per_query=1, batch_size=1, output_name="responses.jsonl", cache=None):
    queries = tmp_path / "queries.jsonl"
    queries.write_text("".join(json.dumps({
        "query_id": f"q_{marker}", "query_text": f"Explain MARKER_{marker}.",
        "intent": "generic", "tags": [],
    }) + "\n" for marker in markers), encoding="utf-8")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Query: {query_text}\nIntent: {intent}\nTags: {tags}", encoding="utf-8")
    output = tmp_path / output_name
    stage2.run_stage2(
        queries_path=queries, prompt_path=prompt, batch_prompt_path=None, adapter=adapter,
        responses_path=output, n_per_query=n_per_query, batch_size=batch_size,
        query_batch_size=2, group_by_intent=False, batch_fallback_per_query=False,
        temperatures=[0.1], max_tokens=200, seed=1, rate_limiter=RateLimiter(0),
        cache=cache if cache is not None else PromptCache(tmp_path / "cache.jsonl", enabled=False),
        logger=logging.getLogger("stage2-rolling-test"), max_total=max_total, max_attempts=1,
    )
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


def test_rolling_refills_while_slowest_request_is_pending(tmp_path):
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
    assert adapter.batch_calls == 0
    assert sorted(adapter.started) == [0, 1, 2]
    assert {row["query_id"] for row in rows} == {"q_0", "q_1", "q_2"}


@pytest.mark.parametrize("n_per_query,batch_size,max_total", [(1, 1, 2), (3, 3, 2), (2, 2, 3)])
def test_rolling_reserves_output_slots_and_honors_max_total(tmp_path, n_per_query, batch_size, max_total):
    adapter = Adapter()
    rows = _run(tmp_path, adapter, markers=range(8), max_total=max_total,
                n_per_query=n_per_query, batch_size=batch_size)
    assert len(rows) == max_total
    assert len({row["response_id"] for row in rows}) == max_total
    assert adapter.peak_active <= 2
    assert len(adapter.started) <= max_total
    assert adapter.batch_calls == 0


def test_rolling_failure_releases_reserved_slot_and_continues(tmp_path):
    adapter = Adapter(failures=(0,))
    rows = _run(tmp_path, adapter, max_total=2)
    assert sorted(adapter.started) == [0, 1, 2]
    assert {row["query_id"] for row in rows} == {"q_1", "q_2"}


def test_rolling_resume_and_cache_do_not_regenerate_completed_work(tmp_path):
    cache = PromptCache(tmp_path / "cache.jsonl", enabled=True)
    first = Adapter()
    assert len(_run(tmp_path, first, max_total=1, cache=cache)) == 1
    assert first.started == [0]
    resumed = Adapter()
    rows = _run(tmp_path, resumed, max_total=2, cache=cache)
    assert sorted(resumed.started) == [1, 2]
    assert len({row["response_id"] for row in rows}) == len(rows) == 3
    cached = Adapter()
    assert len(_run(tmp_path, cached, output_name="cached.jsonl", cache=cache)) == 3
    assert cached.started == []


def test_disabling_rolling_keeps_existing_batch_route(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE2_ROLLING_DISPATCH", "0")
    adapter = Adapter()
    assert len(_run(tmp_path, adapter, markers=(0, 1))) == 2
    assert adapter.batch_calls == 1
