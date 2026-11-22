import sys
from pathlib import Path
from threading import Event

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatgpt_sources import ChatGPTSourceAdapter, ChatGPTSourceClient
from pipeline.genui_codex_client import CodexClient


def test_source_client_preserves_thread_isolation_and_turn_options(monkeypatch):
    monkeypatch.setattr(CodexClient, "rpc", lambda self, method, params, timeout: params)
    client = object.__new__(ChatGPTSourceClient)
    params = {"developerInstructions": "Return one final Express block.", "permissions": "genui",
              "approvalPolicy": "never", "model": "test"}
    actual = client.rpc("thread/start", params)
    assert "source data" in actual["developerInstructions"]
    assert "Do not call tools" in actual["developerInstructions"]
    assert {**actual, "developerInstructions": params["developerInstructions"]} == params
    assert client.rpc("turn/start", params) is params


def reply(model, text):
    return {"model": model, "status": "completed", "thread_id": "t", "turn_id": "u", "text": text,
            "usage": {"inputTokens": 40, "outputTokens": 20, "cachedInputTokens": 32, "reasoningOutputTokens": 5}}


@pytest.mark.parametrize(("stage", "instruction"), [(1, "distinct user queries"), (2, "source responses")])
def test_adapter_binds_stage_model_effort_and_usage(stage, instruction):
    class FakeClient:
        def generate(self, body, *, timeout):
            assert timeout == 123
            assert body["model"] == "fixture-model"
            assert body["reasoning"] == {"effort": "medium"}
            prefix = body["input"][0]["content"][0]["text"]
            assert instruction in prefix and "fixture system" in prefix
            assert body["input"][1]["content"] == "fixture prompt"
            return reply(body["model"], "fixture output")

    adapter = ChatGPTSourceAdapter(FakeClient(), stage=stage, model="fixture-model", effort="medium", timeout=123)
    result = adapter.generate("fixture prompt", system="fixture system")
    assert result.text == "fixture output" and result.provider == "chatgpt"
    assert result.input_tokens == 40 and result.output_tokens == 20 and result.reasoning_tokens == 5
    assert result.raw["usage"]["cachedInputTokens"] == 32 and result.cost_usd is None


def test_parallel_sources_return_ordered_results_and_isolate_failure():
    release = Event()

    class FakeClient:
        def generate(self, body, *, timeout):
            prompt = body["input"][1]["content"]
            if prompt == "first":
                assert release.wait(5)
            if prompt == "second":
                release.set()
                return reply("unexpected-model", prompt)
            return reply(body["model"], prompt)

    adapter = ChatGPTSourceAdapter(FakeClient(), stage=2, model="fixture-model", effort="medium", workers=2)
    results = adapter.generate_batch(["first", "second", "third"])
    assert [item.text for item in results] == ["first", "", "third"]
    assert results[1].error and results[1].completion_complete is False
    assert not results[0].error and not results[2].error
