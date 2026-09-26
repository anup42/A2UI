"""CPU-only coverage for live Muse HTTP timing and persisted run logs."""

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm.base import ModelSpec  # noqa: E402
from llm.local_adapter import LocalAdapter  # noqa: E402
from utils.logging import setup_logger  # noqa: E402


PROMPT = "Private source prompt that must not appear in timing logs"
COMPLETION = "Private generated completion"
REASONING = "Private generated reasoning"


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": COMPLETION, "reasoning_content": REASONING},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 20},
        }).encode()


@pytest.fixture
def muse_adapter(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("LOCAL_VLLM_", "LOCAL_MUSE_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LOCAL_VLLM_ENABLE_THINKING", "1")
    monkeypatch.setenv("LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS", "1")
    monkeypatch.setenv("LOCAL_VLLM_SERVED_MODEL", "muse-latency-test")
    monkeypatch.setenv("LOCAL_VLLM_RETRY_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("LOCAL_VLLM_RETRY_MAX_SECONDS", "180")
    monkeypatch.setenv("LOCAL_MUSE_REQUIRE_REASONING", "1")
    return LocalAdapter(ModelSpec(
        "muse", "local", "meta-models/Muse-Glimmer-30B", endpoint="http://unused.test/v1"
    ))


@pytest.fixture
def run_log(tmp_path):
    logger = logging.getLogger("dataset")
    original_handlers, original_level = list(logger.handlers), logger.level
    setup_logger(tmp_path)
    try:
        yield tmp_path / "run.log"
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers = original_handlers
        logger.setLevel(original_level)


def fake_clock(monkeypatch, readings, sleeps=None):
    ticks = iter(readings)
    # Replace this module's clock object without changing logging's own clock.
    monkeypatch.setattr("llm.local_adapter.time", SimpleNamespace(
        monotonic=lambda: next(ticks),
        time=lambda: 500.0,  # Retry policy may still use wall time.
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    ))


def generate(adapter):
    return adapter._http_generate(PROMPT, None, 1.0, 1024, 42, False)


def timing_lines(path):
    text = path.read_text(encoding="utf-8")
    assert all(secret not in text for secret in (PROMPT, COMPLETION, REASONING))
    return [line for line in text.splitlines() if "Muse HTTP " in line]


@pytest.mark.parametrize("strength,expected", [(None, "high"), ("medium", "medium"), ("xhigh", "xhigh")])
def test_muse_http_latency_persists_effective_reasoning_and_uses_dataset_logger(
    monkeypatch, muse_adapter, run_log, caplog, strength, expected
):
    if strength is not None:
        monkeypatch.setenv("LOCAL_VLLM_REASONING_STRENGTH", strength)
    monkeypatch.setattr("llm.local_adapter.urlopen", lambda *args, **kwargs: Response())
    fake_clock(monkeypatch, [10.0, 11.0, 13.0, 14.0])

    result = generate(muse_adapter)

    assert result.error is None
    assert result.latency_ms == 4000.0
    attempt = result.raw["a2ui_request_attempts"][0]
    assert attempt["latency_ms"] == 2000.0
    assert attempt["controls"]["chat_template_kwargs"] == {"reasoning_strength": expected}
    lines = timing_lines(run_log)
    assert len(lines) == 1
    assert "attempt=0 status=completed model=muse-latency-test" in lines[0]
    assert f"reasoning_strength={expected} latency_ms=2000.0" in lines[0]
    assert f"messages_sha256={attempt['messages_sha256']}" in lines[0]
    assert "dataset.llm.local_adapter" in lines[0]
    assert any(record.name == "dataset.llm.local_adapter" and "Muse HTTP attempt=0" in record.message
               for record in caplog.records)


def test_muse_http_retries_log_each_failed_and_successful_attempt(monkeypatch, muse_adapter, run_log):
    responses = iter([ConnectionError("connection refused"), Response()])

    def urlopen(*args, **kwargs):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("llm.local_adapter.urlopen", urlopen)
    sleeps = []
    fake_clock(monkeypatch, [10.0, 11.0, 12.0, 15.0, 18.0, 19.0], sleeps)

    result = generate(muse_adapter)

    assert result.error is None
    assert result.latency_ms == 9000.0
    assert sleeps == [1.0]
    attempts = result.raw["a2ui_request_attempts"]
    assert [item["latency_ms"] for item in attempts] == [1000.0, 3000.0]
    lines = timing_lines(run_log)
    assert len(lines) == 2
    assert "attempt=0 status=failed" in lines[0] and "latency_ms=1000.0" in lines[0]
    assert "attempt=1 status=completed" in lines[1] and "latency_ms=3000.0" in lines[1]
    for line in lines:
        assert "reasoning_strength=high" in line
        assert f"messages_sha256={attempts[0]['messages_sha256']}" in line


def test_muse_failed_result_keeps_monotonic_latency(monkeypatch, muse_adapter, run_log):
    def fail(*args, **kwargs):
        raise RuntimeError("request denied")

    monkeypatch.setattr("llm.local_adapter.urlopen", fail)
    fake_clock(monkeypatch, [10.0, 11.0, 13.0, 14.0])

    result = generate(muse_adapter)

    assert result.error == "request denied"
    assert result.latency_ms == 4000.0
    lines = timing_lines(run_log)
    assert len(lines) == 1
    assert "attempt=0 status=failed" in lines[0]
    assert "latency_ms=2000.0" in lines[0]


def test_muse_unspecified_server_reasoning_is_not_reported_as_high(monkeypatch, muse_adapter, run_log):
    monkeypatch.setenv("LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS", "0")
    monkeypatch.setattr("llm.local_adapter.urlopen", lambda *args, **kwargs: Response())
    fake_clock(monkeypatch, [10.0, 11.0, 13.0, 14.0])

    result = generate(muse_adapter)

    assert "chat_template_kwargs" not in result.raw["a2ui_request_attempts"][0]["controls"]
    assert "reasoning_strength=server_default" in timing_lines(run_log)[0]
