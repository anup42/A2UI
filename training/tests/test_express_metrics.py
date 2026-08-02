from __future__ import annotations

from ir_training.eval.metrics import score_prediction


def test_repaired_candidate_is_not_counted_as_raw_native_valid() -> None:
    repaired = '<a2ui>\nroot=Text("Recovered")\n</a2ui>'
    raw = repaired.replace("</a2ui>", "")
    metrics = score_prediction("Recovered", None, raw, repaired)
    assert metrics["native_syntax_valid"] is False
    assert metrics["native_catalog_valid"] is False
    assert metrics["raw_standard_a2ui_valid"] is False
    assert metrics["repaired_syntax_valid"] is True
    assert metrics["repaired_catalog_valid"] is True
    assert metrics["repaired_standard_a2ui_valid"] is True
    assert metrics["repaired_canonical_semantic_valid"] is True
    assert metrics["canonical_semantic_valid"] is True
