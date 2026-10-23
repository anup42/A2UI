"""The correction supplement is opt-in and its example is executable."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.generation_audit import graph_acceptance_errors
from pipeline.ir_formats import compile_express_to_wire, decode_express_completion


def test_v11_correction_example_compiles_and_keeps_entity_fields():
    guidance = (ROOT / "prompts/v11_regeneration_v1.md").read_text(encoding="utf-8")
    examples = re.findall(r"(?ms)^<a2ui>\n.*?^</a2ui>", guidance)
    assert len(examples) == 1
    graph = decode_express_completion(examples[0])
    assert graph_acceptance_errors(graph) == []
    assert compile_express_to_wire(examples[0])
    elements = graph["elements"]
    assert elements["cedar"]["props"]["title"] == "Hotel: Cedar"
    assert elements["elm"]["props"]["title"] == "Hotel: Elm"
    assert elements["priceC"]["props"]["text"] == "Price: INR 4200/night"
    assert elements["priceE"]["props"]["text"] == "Price: INR 5300/night"
    assert elements["cedar"]["children"] == ["cityC", "priceC", "ratingC"]
    assert elements["elm"]["children"] == ["cityE", "priceE", "ratingE"]


def test_v11_guidance_does_not_add_a_second_source_placeholder_or_relax_fidelity():
    guidance = (ROOT / "prompts/v11_regeneration_v1.md").read_text(encoding="utf-8")
    assert "{response_text}" not in guidance
    for requirement in (
        "same entity, section, event, and action context",
        "Do not split",
        "Plain steps, checklists, ingredients",
        "of notification",
        "Do not silently repair contradictions",
        "do not truncate or summarize",
    ):
        assert requirement in guidance


def test_stage3_supplement_is_opt_in_hash_bound_and_preserves_canonical_prompt(tmp_path, monkeypatch):
    from llm.base import BaseLLMAdapter, LLMResult, ModelSpec
    from pipeline.cache import PromptCache
    from pipeline.muse_prompt import MUSE_MODEL, compose_stage3_prompt
    from pipeline.stage3_genui import _prepare_prompt_context, run_stage3
    from pipeline.training_fidelity import compose_training_fidelity_prompt
    from utils.rate_limit import RateLimiter

    monkeypatch.setenv("STAGE3_FINAL_REGEN_ATTEMPTS", "0")
    monkeypatch.setenv("DATASET_OFFLINE_MODE", "1")
    monkeypatch.setenv("LOCAL_MUSE_REQUIRE_REASONING", "0")
    monkeypatch.delenv("STAGE3_PROMPT_MODE", raising=False)
    monkeypatch.delenv("GEMINI_STAGE3_PROMPT_MODE", raising=False)
    source = "Meeting at 11:00. Availability is not confirmed."
    target = '<a2ui>root=Text(' + json.dumps(source) + ')</a2ui>'
    spec = ModelSpec("muse", "local", MUSE_MODEL)

    class Adapter(BaseLLMAdapter):
        def __init__(self):
            super().__init__(spec)
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResult(text=target, raw={"choices": [{"finish_reason": "stop"}]},
                             latency_ms=1, input_tokens=10, output_tokens=20, cost_usd=None,
                             model=spec.model, provider=spec.provider,
                             finish_reason="stop", completion_complete=True)

    adapter = Adapter()
    prompt = ROOT / "prompts/genui_gen_mobile_a2ui_express_v1.md"
    supplement = ROOT / "prompts/v11_regeneration_v1.md"
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps({"query_id": "q1", "response_id": "r1", "n_idx": 1,
                                     "response_text": source}) + "\n", encoding="utf-8")
    cache = PromptCache(tmp_path / "cache.jsonl", enabled=True)
    logger = logging.getLogger(__name__)
    common = {
        "queries_path": None, "responses_path": responses, "prompt_path": prompt,
        "adapter": adapter, "schema_path": ROOT / "schema/canonical_ui_graph_v1.schema.json",
        "candidates_per_response": 1, "max_repair_attempts": 0, "max_tokens": 2048,
        "prompt_max_tokens": 16000, "seed": 1, "rate_limiter": RateLimiter(0),
        "cache": cache, "logger": logger, "batch_size": 1, "max_attempts": 1,
        "metric_version": "v5_4", "ir_formats": ["a2ui_express_v1"],
    }
    original_system, _ = _prepare_prompt_context(
        compose_training_fidelity_prompt(compose_stage3_prompt(prompt.read_text(encoding="utf-8"), spec)),
        adapter, logger,
    )
    for name, extra in (("default", {}), ("opt_in", {
        "prompt_supplement_path": supplement, "grpo_entity_extraction": True,
    })):
        out = tmp_path / name
        out.mkdir()
        run_stage3(**common, genui_path=out / "genui.jsonl", artifacts_dir=out / "artifacts", **extra)
    assert len(adapter.calls) == 2  # An old default-prompt cache hit cannot supply the opt-in result.
    assert adapter.calls[0]["system"] == original_system
    assert "Opt-in v11 correction guidance" not in adapter.calls[0]["system"]
    assert "Opt-in v11 correction guidance" in adapter.calls[1]["system"]
    assert "Pinned catalog signatures" in adapter.calls[1]["system"]
    default = json.loads((tmp_path / "default/genui.jsonl").read_text(encoding="utf-8"))
    corrected = json.loads((tmp_path / "opt_in/genui.jsonl").read_text(encoding="utf-8"))
    assert "prompt_supplement" not in default["gen"]
    assert "entity_extraction_policy" not in default["gen"]
    assert corrected["gen"]["entity_extraction_policy"] == "2.0.0"
    assert "grpo_entity_extraction" not in default["generation_reward_v5_4"]["evidence"]
    assert "grpo_entity_extraction" in corrected["generation_reward_v5_4"]["evidence"]
    assert "grpo_entity_extraction" in corrected["render_artifact_quality_v5_4"]["evidence"]
    assert corrected["gen"]["prompt_supplement"] == {
        "prompt_file": supplement.name,
        "prompt_sha256": hashlib.sha256(supplement.read_bytes()).hexdigest(),
    }
    assert default["response_text"] == corrected["response_text"] == source


def test_stage3_rejects_ambiguous_supplement_before_generation(tmp_path):
    import pytest
    from llm.base import BaseLLMAdapter, ModelSpec
    from pipeline.cache import PromptCache
    from pipeline.stage3_genui import run_stage3
    from utils.rate_limit import RateLimiter

    path = tmp_path / "supplement.md"
    for text in ("", "Second source: {response_text}"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="prompt supplement"):
            run_stage3(queries_path=None, responses_path=tmp_path / "nonexistent.jsonl",
                       prompt_path=ROOT / "prompts/genui_gen_mobile_a2ui_express_v1.md",
                       adapter=BaseLLMAdapter(ModelSpec("test", "local", "fixture")),
                       genui_path=tmp_path / "genui.jsonl", schema_path=ROOT / "schema/canonical_ui_graph_v1.schema.json",
                       artifacts_dir=tmp_path / "artifacts", candidates_per_response=1,
                       max_repair_attempts=0, max_tokens=2048, prompt_max_tokens=16000, seed=1,
                       rate_limiter=RateLimiter(0), cache=PromptCache(tmp_path / "cache", enabled=False),
                       logger=logging.getLogger(__name__), prompt_supplement_path=path)
    assert not (tmp_path / "artifacts").exists()


def test_stage3_entity_policy_rejects_nonbool_and_incompatible_scoring_before_io():
    import pytest
    from pipeline.stage3_genui import run_stage3

    common = {
        "queries_path": None, "responses_path": None, "prompt_path": None,
        "adapter": None, "genui_path": None, "schema_path": None, "artifacts_dir": None,
        "candidates_per_response": 1, "max_repair_attempts": 0, "max_tokens": 100,
        "prompt_max_tokens": 100, "seed": 1, "rate_limiter": None, "cache": None, "logger": None,
    }
    with pytest.raises(ValueError, match="explicit boolean"):
        run_stage3(**common, grpo_entity_extraction="true")
    with pytest.raises(ValueError, match="requires v5_4"):
        run_stage3(**common, grpo_entity_extraction=True, metric_version="legacy")
