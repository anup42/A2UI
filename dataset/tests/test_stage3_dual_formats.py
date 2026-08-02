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
from pipeline.dual_format_pilot import select_pair
from pipeline.ir_formats import A2UI_EXPRESS_V1, COMPACT_IR_V2, encode_from_flat_spec
from pipeline.stage3_genui import run_stage3
from utils.rate_limit import RateLimiter


SPEC = {
    "root": "screen",
    "state": {},
    "elements": {
        "screen": {
            "type": "Stack",
            "props": {"direction": "vertical", "gap": "md"},
            "children": ["title", "card"],
        },
        "title": {"type": "Text", "props": {"text": "Trip", "variant": "h2"}, "children": []},
        "card": {"type": "Card", "props": {}, "children": ["body", "button"]},
        "body": {"type": "Text", "props": {"text": "Train at 09:30 costs $24."}, "children": []},
        "button": {
            "type": "Button",
            "props": {"label": "Continue"},
            "children": [],
            "on": {"press": {"action": "emitEvent", "params": {"name": "continue"}}},
        },
    },
}


class _FormatAdapter(BaseLLMAdapter):
    def __init__(self, *, wrong_express: bool = False):
        super().__init__(ModelSpec("fake", "fake", "fake-model", supports_json_mode=True))
        self.wrong_express = wrong_express
        self.json_modes: list[bool] = []

    def generate(self, prompt, system, temperature, max_tokens, seed, json_mode=False):
        self.json_modes.append(bool(json_mode))
        format_context = (system or "") + "\n" + prompt
        if "A2UI Express" in format_context:
            text = json.dumps(SPEC) if self.wrong_express else encode_from_flat_spec(SPEC, A2UI_EXPRESS_V1)
        else:
            text = json.dumps(encode_from_flat_spec(SPEC, COMPACT_IR_V2), separators=(",", ":"))
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


def _run(tmp_path: Path, adapter: BaseLLMAdapter) -> list[dict]:
    responses = tmp_path / "responses.jsonl"
    queries = tmp_path / "queries.jsonl"
    output = tmp_path / "genui.jsonl"
    _write_jsonl(
        responses,
        [{"response_id": "r1", "query_id": "q_trip", "n_idx": 1, "response_text": "Trip train at 09:30 costs $24."}],
    )
    _write_jsonl(queries, [{"query_id": "q_trip", "intent": "travel", "tags": []}])
    run_stage3(
        queries_path=queries,
        responses_path=responses,
        prompt_path=DATASET_ROOT / "prompts" / "genui_gen_mobile_flatspec_v11.md",
        adapter=adapter,
        genui_path=output,
        schema_path=DATASET_ROOT / "schema" / "genui_flatspec.schema.json",
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
    )
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


def test_stage3_generates_and_records_both_native_formats(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE3_FINAL_REGEN_ATTEMPTS", "0")
    adapter = _FormatAdapter()
    rows = _run(tmp_path, adapter)

    assert {row["ui_id"] for row in rows} == {"u_trip_01_cir2", "u_trip_01_exp1"}
    assert {row["source_format"] for row in rows} == {COMPACT_IR_V2, A2UI_EXPRESS_V1}
    assert all(row["record_status"] == "accepted" for row in rows)
    assert all(row["semantic_hash"] for row in rows)
    assert all(row["model_completion_raw"] for row in rows)
    assert all(row["model_native_output_normalized"] for row in rows)
    assert [False, True] == sorted(adapter.json_modes)

    report = json.loads((tmp_path / "dual_format_pilot_report.json").read_text(encoding="utf-8"))
    assert report["complete_pair_count"] == 1
    assert report["comparable_pair_count"] == 1
    assert sum(bool(row["dual_format_selection"]["selected"]) for row in rows) == 1


def test_wrong_native_format_is_rejected_without_flatspec_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE3_FINAL_REGEN_ATTEMPTS", "0")
    rows = _run(tmp_path, _FormatAdapter(wrong_express=True))
    by_format = {row["source_format"]: row for row in rows}

    assert by_format[COMPACT_IR_V2]["record_status"] == "accepted"
    rejected = by_format[A2UI_EXPRESS_V1]
    assert rejected["record_status"] == "format_rejected"
    assert "genui_json" not in rejected
    assert not rejected["validation"]["schema_valid_strict"]


def test_selection_is_quality_first_then_tokens_within_tolerance():
    rows = [
        {
            "ui_id": "u_1_cir2",
            "source_format": COMPACT_IR_V2,
            "record_status": "accepted",
            "validation": {"schema_valid_strict": True},
            "metrics": {"genui_quality_v5_4": 90.0},
            "format_metrics": {"estimated_tokens": 100},
        },
        {
            "ui_id": "u_1_exp1",
            "source_format": A2UI_EXPRESS_V1,
            "record_status": "accepted",
            "validation": {"schema_valid_strict": True},
            "metrics": {"genui_quality_v5_4": 89.5},
            "format_metrics": {"estimated_tokens": 60},
        },
    ]
    within = select_pair(rows, quality_tolerance=1.0)
    strict = select_pair(rows, quality_tolerance=0.1)
    assert within["selected_format"] == A2UI_EXPRESS_V1
    assert strict["selected_format"] == COMPACT_IR_V2
