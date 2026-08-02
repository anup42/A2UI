from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
SRC = DATASET_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm.base import BaseLLMAdapter, LLMResult, ModelSpec
from pipeline.cache import PromptCache
from pipeline.ir_formats import A2UI_EXPRESS_V1, encode_from_flat_spec
from pipeline.stage3_genui import run_stage3
from utils.rate_limit import RateLimiter


SPEC = {
    "root": "screen",
    "state": {},
    "elements": {
        "screen": {"type": "Stack", "props": {"direction": "vertical"}, "children": ["title", "body", "summary", "details"]},
        "title": {"type": "Text", "props": {"text": "Trip", "variant": "h2"}, "children": []},
        "body": {"type": "Text", "props": {"text": "Train at 09:30 costs $24."}, "children": []},
        "summary": {"type": "Card", "props": {"title": "Itinerary"}, "children": ["summary_text"]},
        "summary_text": {"type": "Text", "props": {"text": "BLR to LKO"}, "children": []},
        "details": {"type": "Text", "props": {"text": "Direct service."}, "children": []},
    },
}


class _ExpressAdapter(BaseLLMAdapter):
    def __init__(self, *, wrong_output: bool = False):
        super().__init__(ModelSpec("fake", "fake", "fake-model", supports_json_mode=False))
        self.wrong_output = wrong_output
        self.json_modes: list[bool] = []

    def generate(self, prompt, system, temperature, max_tokens, seed, json_mode=False):
        self.json_modes.append(bool(json_mode))
        text = json.dumps(SPEC) if self.wrong_output else encode_from_flat_spec(SPEC, A2UI_EXPRESS_V1)
        return LLMResult(
            text=text,
            raw={"text": text},
            latency_ms=7.0,
            input_tokens=20,
            output_tokens=30,
            cost_usd=None,
            model=self.spec.model,
            provider=self.spec.provider,
        )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _run(tmp_path: Path, adapter: _ExpressAdapter) -> list[dict]:
    responses = tmp_path / "responses.jsonl"
    queries = tmp_path / "queries.jsonl"
    output = tmp_path / "genui.jsonl"
    _write_jsonl(responses, [{"response_id": "r1", "query_id": "q_trip", "n_idx": 1, "response_text": "Trip train at 09:30 costs $24."}])
    _write_jsonl(queries, [{"query_id": "q_trip", "intent": "travel", "tags": []}])
    run_stage3(
        queries_path=queries,
        responses_path=responses,
        prompt_path=DATASET_ROOT / "prompts" / "genui_gen_mobile_a2ui_express_v1.md",
        adapter=adapter,
        genui_path=output,
        schema_path=DATASET_ROOT / "schema" / "canonical_ui_graph_v1.schema.json",
        artifacts_dir=tmp_path / "artifacts",
        candidates_per_response=1,
        max_repair_attempts=0,
        max_tokens=2048,
        prompt_max_tokens=None,
        seed=1,
        rate_limiter=RateLimiter(0),
        cache=PromptCache(tmp_path / "cache.jsonl", enabled=False),
        logger=logging.getLogger("stage3-test"),
        batch_size=1,
        max_attempts=1,
        metric_version="legacy",
        ir_formats=[A2UI_EXPRESS_V1],
    )
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


def test_stage3_generates_only_express_and_disables_json_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE3_FINAL_REGEN_ATTEMPTS", "0")
    adapter = _ExpressAdapter()
    rows = _run(tmp_path, adapter)

    assert {row["ui_id"] for row in rows} == {"u_trip_01"}
    assert rows[0]["source_format"] == A2UI_EXPRESS_V1
    assert rows[0]["record_status"] == "accepted"
    assert rows[0]["semantic_hash"]
    assert rows[0]["model_completion_raw"]
    assert rows[0]["model_native_output_normalized"].startswith("<a2ui>")
    assert adapter.json_modes == [False]


def test_wrong_native_format_is_rejected_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE3_FINAL_REGEN_ATTEMPTS", "0")
    rows = _run(tmp_path, _ExpressAdapter(wrong_output=True))
    assert len(rows) == 1
    assert rows[0]["record_status"] == "format_rejected"
    assert "genui_json" not in rows[0]
    assert not rows[0]["validation"]["schema_valid_strict"]
