"""No-generation policy regressions; no model or original corpus needed."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/audits"))
from full_data_offline_triage_20260913 import (
    Families, KEEP_STATUSES, REPAIR_STATUS, WARNINGS, base_screen, duplicate_reason,
    preliminary_classification,
)


def clean(**changes):
    return {"reserved": 0, "valid": 1, "url_error": 0,
            "sequence_tokens": 4096, "target_tokens": 2048,
            **dict.fromkeys(WARNINGS, 0), **changes}


def test_golden_precedes_other_rejections():
    assert preliminary_classification(clean(mojibake=1), "not_screened",
        reserved_family=True, validation_overlap=True) == ("REJECT", "reserved_golden_family")


def test_validation_overlap_never_becomes_repair():
    assert preliminary_classification(clean(), REPAIR_STATUS,
        validation_overlap=True, repaired_budget_pass=True) == (
            "REJECT", "validation_family_seen_in_original_training")


def test_confirmed_defect_rejected_despite_absence_of_flags():
    assert preliminary_classification(clean(), "already_symbolic_closed",
        confirmed_defect=True) == ("REJECT", "confirmed_content_defect")


@pytest.mark.parametrize("status", sorted(KEEP_STATUSES))
def test_closed_symbolic_or_no_reference_is_only_provisional_keep(status):
    assert preliminary_classification(clean(), status) == ("KEEP", status)


def test_repair_requires_completed_post_transform_budget_check():
    with pytest.raises(ValueError, match="post-transform"):
        preliminary_classification(clean(), REPAIR_STATUS)
    assert preliminary_classification(clean(), REPAIR_STATUS, repaired_budget_pass=True) == ("REPAIR", REPAIR_STATUS)
    assert preliminary_classification(clean(), REPAIR_STATUS, repaired_budget_pass=False) == (
        "REJECT", "outside_selected_post_repair_token_budget")


@pytest.mark.parametrize("flag", WARNINGS)
def test_quality_warnings_cannot_be_disguised_as_url_repairs(flag):
    assert preliminary_classification(clean(**{flag: 1}), REPAIR_STATUS,
        repaired_budget_pass=True) == ("REJECT", "unresolved_quality_warning")


def test_missing_or_unknown_url_probe_fails_closed():
    with pytest.raises(ValueError, match="Missing/unknown"):
        preliminary_classification(clean(), "not_screened")
    assert preliminary_classification(clean(), "ambiguous_not_counted") == (
        "REJECT", "unresolved_url_grounding")


@pytest.mark.parametrize("changes", [{"sequence_tokens": 4097}, {"target_tokens": 2049},
                                     {"sequence_tokens": None}, {"target_tokens": None}])
def test_selected_budget_is_not_silently_truncated(changes):
    row = clean(**changes)
    assert not base_screen(row)
    assert preliminary_classification(row, "not_screened") == (
        "REJECT", "outside_selected_original_token_budget")


def test_transitive_family_binding_across_hash_types_and_near_pairs():
    families = Families()
    families.union(["raw1", "norm1", "masked"])
    families.union(["raw2", "norm2", "masked"])
    families.union(["raw3", "norm3", None])
    families.union(["raw2", "raw3"])
    assert len({families.find(value) for value in ("raw1", "raw2", "raw3", "norm3", "masked")}) == 1
    assert families.find("independent") != families.find("raw1")


def test_dedup_original_semantics_and_effective_text_without_losing_alternatives():
    original, effective = {}, {}
    assert duplicate_reason(("src1", "sem1"), ("norm1", "text1"), "train:1", original, effective) == ("", "")
    assert duplicate_reason(("src1", "sem1"), ("norm1", "text2"), "train:2", original, effective) == (
        "duplicate_exact_source_semantic_target", "train:1")
    assert duplicate_reason(("src2", "sem2"), ("norm1", "text1"), "train:3", original, effective) == (
        "duplicate_effective_source_target_text", "train:1")
    assert duplicate_reason(("src1", "different_layout"), ("norm1", "different_text"), "train:4", original, effective) == ("", "")
