"""Compatibility counters must be labelled without becoming semantic gates."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.metrics import (
    aggregate_metrics,
    compute_ui_metrics,
    count_characters,
    lexical_token_estimate,
    metric_diagnostic_metadata,
)


def test_card_title_counter_limitation_is_explained_without_changing_values():
    program = '<a2ui>\nroot=Card([body],title="Summary")\nbody=Text("An explanation.")\n</a2ui>'
    metrics = compute_ui_metrics("# Summary\nAn explanation.", program)
    assert metrics["section_heading_coverage"] == 0
    assert all(isinstance(value, (int, float)) for value in metrics.values())
    metadata = metric_diagnostic_metadata("a2ui_express_v1")
    assert "Card titles" in metadata["legacy_metrics"]["section_heading_coverage"]
    assert metadata["legacy_metrics"]["automatic_semantic_rejection_supported"] is False
    assert metadata["stage3_warning_counters"]["warning_is_proof_of_content_loss"] is False


def test_size_units_distinguish_characters_words_and_real_model_tokens():
    text = "é  AB\nC"
    assert count_characters(text) == 4
    assert lexical_token_estimate(text) == 3
    assert len(text) == 7 and len(text.encode("utf-8")) == 8
    sizes = metric_diagnostic_metadata()["size_and_token_fields"]
    assert sizes["output_tokens_json"]["unit"] == "non_whitespace_characters"
    assert sizes["output_tokens_json"]["model_token_count"] is False
    assert sizes["format_metrics.estimated_tokens"]["model_token_count"] is False
    assert sizes["format_metrics.completion_tokens"]["unavailable_value"] is None


def test_aggregate_exposes_metadata_and_preserves_legacy_numeric_aliases():
    program = '<a2ui>\nroot=Text("Hello")\n</a2ui>'
    row = {"response_text": "Hello", "a2ui_express": program,
           "target_format": "a2ui_express_v1",
           "metrics": {"output_tokens_json": 12, "output_chars_json": 12}}
    aggregate = aggregate_metrics([row])
    assert aggregate["output_tokens_json_avg"] == aggregate["output_chars_json_avg"] == 12
    assert aggregate["metric_diagnostics"]["legacy_metrics"]["diagnostic_only"]
    assert "metric_diagnostics" not in row
    assert "genui_quality_v5_4" in aggregate
