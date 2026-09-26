"""Exercise real Stage 2 routing with stub completions; never run model inference."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.base import LLMResult, ModelSpec
from pipeline.cache import PromptCache
from pipeline.common import render_prompt
from pipeline.muse_source_prompt import (
    MUSE_MODEL,
    PROMPT_VERSION,
    select_stage2_prompt,
)
from pipeline.source_quality import (
    assess_source_quality,
    parse_actions,
    source_contract_prompt,
)
from pipeline.stage2_responses import _clean_url, _extract_asset_urls, run_stage2
from utils.hashing import hash_text

GENERIC = "Generic unchanged: {query_text}; intent={intent}; tags={tags}"
MUSE = ModelSpec("registered-muse", "local", MUSE_MODEL)


@pytest.mark.parametrize("spec", [
    ModelSpec("other", "local", "another-model"),
    ModelSpec("muse-name-only", "local", "another-model"),
    ModelSpec("remote", "openai", MUSE_MODEL),
])
def test_non_muse_template_is_byte_unchanged(spec):
    assert select_stage2_prompt(GENERIC, spec) == GENERIC


def test_muse_replaces_conflicting_generic_and_keeps_quality_requirements():
    template = select_stage2_prompt("Require live data; exactly 3-8 rows", MUSE)
    assert "Require live data" not in template
    assert "exactly 3-8 rows" not in template
    for required in (
        PROMPT_VERSION, "{query_text}", "{intent}", "{tags}", "weighted means",
        "strictly under/after", "date/weekday", "every duration and break",
        "Rankings", "no evidence of browsing", "pl.u-...", "all-zero list IDs",
        "Action:", "12 or 21", "Media is optional", "Internal audit",
        "not private reasoning", "not real listings",
    ):
        assert required in template


def test_documented_action_grammar_reaches_source_quality_guard():
    template = select_stage2_prompt(GENERIC, MUSE)
    example = next(line.strip() for line in template.splitlines() if line.strip().startswith("Action:"))
    response = example.replace("accurate label", "Open playlist").replace(
        "complete_destination", "https://music.apple.com/playlist/pl.u-..."
    )
    assert parse_actions(response) == [{"label": "Open playlist",
                                        "destination": "https://music.apple.com/playlist/pl.u-..."}]
    quality = assess_source_quality({"query_text": "Plan a playlist."}, response)
    assert quality["status"] == "failed"
    assert any(finding["code"] == "truncated_http_destination" for finding in quality["findings"])


@pytest.mark.parametrize("value", [
    "https://en.wikipedia.org/wiki/Plan_(film)",
    "(https://en.wikipedia.org/wiki/Plan_(film)).",
    "['https://en.wikipedia.org/wiki/Plan_(film)'].",
    '<https://en.wikipedia.org/wiki/Plan_(film)>',
])
def test_clean_url_preserves_balanced_delimiters(value):
    assert _clean_url(value) == "https://en.wikipedia.org/wiki/Plan_(film)"


def test_asset_extraction_retains_balanced_parentheses_in_all_routes():
    url = "https://images.unsplash.com/photo_(test)"
    assert _extract_asset_urls(f"Media: Image={url}") == [url]
    assert _extract_asset_urls(f"Images:\n[Reference]({url}).") == [url]


@pytest.mark.parametrize("is_muse,query_batch_size", [(True, 1), (True, 2), (False, 1), (False, 2)])
def test_run_stage2_routing_cache_and_provenance(tmp_path, monkeypatch, is_muse, query_batch_size):
    import pipeline.stage2_responses as stage2

    for name, value in {
        "DATASET_OFFLINE_MODE": "1", "INTERNET": "0", "STAGE2_ASSET_WORKERS": "1",
        "STAGE2_REAL_ASSET_RETRY_ENABLED": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(stage2, "_load_local_icon_context", lambda *_: None)
    monkeypatch.setattr(stage2, "_commons_enrichment_enabled", lambda: False)
    monkeypatch.setattr(stage2, "urlopen", Mock(side_effect=AssertionError("network forbidden")))
    queries = [
        {"query_id": f"q_{index}", "query_text": f"Compute {index} plus 2.",
         "intent": "calculation", "tags": ["math", "sum"]}
        for index in (1, 2)
    ]
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text("".join(json.dumps(query) + "\n" for query in queries), encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text(GENERIC, encoding="utf-8")
    spec = MUSE if is_muse else ModelSpec("other", "local", "another-model")
    calls = []

    def complete(prompt):
        calls.append(prompt)
        index = 1 if "Compute 1 plus 2." in prompt else 2
        return LLMResult(f"Total: {index + 2}", {}, 0, 0, 0, None, spec.model, spec.provider)

    single = Mock(side_effect=lambda **kwargs: complete(kwargs["prompt"]))
    batch = Mock(side_effect=lambda **kwargs: [complete(prompt) for prompt in kwargs["prompts"]])
    adapter = SimpleNamespace(spec=spec, generate=single, generate_batch=batch)
    cache = PromptCache(tmp_path / "cache.jsonl", enabled=True)
    responses_path = tmp_path / "responses.jsonl"

    def run(output):
        run_stage2(
            queries_path, prompt_path, None, adapter, output, 1, 1, query_batch_size,
            False, False, [.1], 200, 13, Mock(), cache, logging.getLogger("muse-stage2-test"),
            max_total=2, max_attempts=1,
        )

    run(responses_path)
    assert len(calls) == 2
    assert batch.call_count == (1 if query_batch_size == 2 else 0)
    assert single.call_count == (2 if query_batch_size == 1 else 0)
    rows = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    template = select_stage2_prompt(GENERIC, spec)
    expected = [render_prompt(template, query_text=query["query_text"], intent="calculation",
                              tags="math, sum") + source_contract_prompt(query) for query in queries]
    assert sorted(calls) == sorted(expected)
    cache_rows = [json.loads(line) for line in (tmp_path / "cache.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["prompt_hash"] for row in cache_rows} == {hash_text(f"{spec.name}:{p}") for p in expected}
    for row in rows:
        assert row["response_text"] in {"Total: 3", "Total: 4"}
        assert "source_quality" in row
        if is_muse:
            assert row["gen"]["prompt_version"] == PROMPT_VERSION
            assert row["gen"]["prompt_template_sha256"] == hash_text(template)
            assert row["gen"]["prompt_sha256"] in {hash_text(prompt) for prompt in expected}
        else:
            assert "prompt_version" not in row["gen"]
    # A second output uses the same actual prompt cache, not the adapter.
    run(tmp_path / "cached_responses.jsonl")
    assert len(calls) == 2


@pytest.mark.parametrize("is_muse,invalid_media,expected_calls", [
    (True, False, 1), (True, True, 2), (False, False, 2),
])
def test_visual_media_retries_are_optional_only_for_muse(
    tmp_path, monkeypatch, is_muse, invalid_media, expected_calls,
):
    import pipeline.stage2_responses as stage2

    for name, value in {
        "DATASET_OFFLINE_MODE": "1", "INTERNET": "0", "STAGE2_ASSET_WORKERS": "1",
        "STAGE2_REAL_ASSET_RETRY_ENABLED": "1", "STAGE2_REAL_ASSET_RETRY_MAX_ATTEMPTS": "2",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(stage2, "_load_local_icon_context", lambda *_: None)
    monkeypatch.setattr(stage2, "_commons_enrichment_enabled", lambda: False)
    monkeypatch.setattr(stage2, "urlopen", Mock(side_effect=AssertionError("network forbidden")))
    query = {"query_id": "q_1", "query_text": "Plan an evening at home.", "intent": "entertainment", "tags": []}
    (tmp_path / "queries.jsonl").write_text(json.dumps(query) + "\n", encoding="utf-8")
    (tmp_path / "prompt.md").write_text(GENERIC, encoding="utf-8")
    spec = MUSE if is_muse else ModelSpec("other", "local", "another-model")
    valid = "Evening plan\nRead a book for 30 minutes, then take a break."
    first = valid + "\nMedia: Icon=https://picsum.photos/icon.svg" if invalid_media else valid
    generate = Mock(side_effect=[LLMResult(text, {}, 0, 0, 0, None, spec.model, spec.provider)
                                for text in (first, valid)])
    adapter = SimpleNamespace(spec=spec, generate=generate)
    run_stage2(
        tmp_path / "queries.jsonl", tmp_path / "prompt.md", None, adapter,
        tmp_path / "responses.jsonl", 1, 1, 1, False, False, [.1], 200, 13,
        Mock(), PromptCache(tmp_path / "cache.jsonl", enabled=False),
        logging.getLogger("muse-stage2-media-test"), max_total=1, max_attempts=1,
    )
    assert generate.call_count == expected_calls
    row = json.loads((tmp_path / "responses.jsonl").read_text(encoding="utf-8"))
    assert row["response_text"] == valid
    if invalid_media:
        retry = generate.call_args.kwargs["prompt"]
        assert "random/placeholder media hosts are not allowed" in retry
        assert "use icon-only media" not in retry
        assert "omit unsupported media when no valid reference is available" in retry
        assert "Inline Media lines are optional" in retry
        assert row["gen"]["prompt_sha256"] == hash_text(retry)
    elif not is_muse:
        assert "use icon-only media" in generate.call_args.kwargs["prompt"]
