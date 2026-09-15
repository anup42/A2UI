from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "dataset" / "src")]

from ir_training.data.build_pairs import prepare_dataset
from ir_training.data.ir_targets import semantic_hash
from ir_training.data.reference_binding import bind_source_target_references, ground_preprocessed_references
from ir_training.data.repairs import repair_graph
from ir_training.data.url_preprocess import preprocess_training_urls, restore_url_placeholders
from pipeline.ir_formats import decode_express_completion, encode_express_completion


RAW = '<a2ui>\nroot=Column([photo,open])\nphoto=Image("[ICON_URL_1]",alt="Hotel")\nopen=Button("Book",onPress=openUrl("[URL_1]"))\n</a2ui>'
SOURCE = 'Media: Icon=<https://assets.invalid/hotel.svg>\nAction: [Button: Book] <https://hotel.invalid/book>'
MAP = {"[ICON_URL_1]": "https://assets.invalid/hotel.svg", "[URL_1]": "https://hotel.invalid/book"}


def record():
    restored = restore_url_placeholders(decode_express_completion(RAW), MAP)
    return {"response_id": "r1", "query_id": "q1", "ui_id": "u1", "record_status": "accepted",
            "source_format": "a2ui_express_v1", "response_text": SOURCE,
            "a2ui_express": RAW, "canonical_graph": restored,
            "canonical_graph_hash": semantic_hash(restored),
            "model_native_output_normalized": encode_express_completion(restored)}


def prepare(tmp_path, row, source=SOURCE, response_quality=None, **config):
    run = tmp_path / "run"
    run.mkdir()
    (run / "genui.jsonl").write_text(json.dumps(row) + "\n", encoding="utf8")
    (run / "responses.jsonl").write_text(json.dumps({"response_id": "r1", "response_text": source, "source_quality": response_quality}) + "\n", encoding="utf8")
    out = tmp_path / "prepared"
    cfg = {"run": {"source_run_dir": str(run), "output_dir": str(out)}, "split": {"train": 1, "val": 0, "test": 0}, **config}
    manifest = prepare_dataset(cfg)
    accepted = [json.loads(line) for line in (out / "all.jsonl").read_text(encoding="utf8").splitlines()]
    rejected = [json.loads(line) for line in (out / "rejected.jsonl").read_text(encoding="utf8").splitlines()]
    return manifest, accepted, rejected


def test_historical_hashed_alignment_preserves_actions_and_needs_no_asset_bytes(tmp_path):
    row = record()
    original = copy.deepcopy(row)
    manifest, accepted, rejected = prepare(tmp_path, row)
    assert manifest["counts"]["accepted"] == 1, rejected
    target = accepted[0]
    assert "openUrl" in target["completion"]
    assert target["reference_binding"]["method"] == "hashed_graph_reference_alignment"
    mapping = target["metadata"]["url_preprocessing"]["url_map"]
    assert restore_url_placeholders(target["canonical_graph"], mapping) == row["canonical_graph"]
    assert manifest["url_preprocessing"]["asset_bytes_required"] is False
    assert row == original


def test_restoration_rejects_nonreference_changes_even_with_new_valid_hash():
    row = record()
    row["canonical_graph"]["elements"]["open"]["props"]["label"] = "Pay now"
    row["canonical_graph_hash"] = semantic_hash(row["canonical_graph"])
    row["model_native_output_normalized"] = encode_express_completion(row["canonical_graph"])
    with pytest.raises(ValueError, match="nonreference_content_changed"):
        bind_source_target_references(SOURCE, decode_express_completion(RAW), row)


def test_restoration_rejects_hash_mismatch():
    row = record()
    row["canonical_graph_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash_mismatch"):
        bind_source_target_references(SOURCE, decode_express_completion(RAW), row)


def test_placeholder_suffix_is_not_used_to_guess_destination():
    with pytest.raises(ValueError, match="no_verified_restoration"):
        bind_source_target_references(SOURCE, decode_express_completion(RAW), {})


def test_explicit_map_and_source_visible_placeholders_are_supported():
    raw = decode_express_completion(RAW)
    bound = bind_source_target_references(SOURCE, raw, {"reference_map": MAP})
    assert bound.graph == record()["canonical_graph"]
    masked_source = SOURCE.replace(MAP["[URL_1]"], "[URL_1]").replace(MAP["[ICON_URL_1]"], "[ICON_URL_1]")
    bound = bind_source_target_references(masked_source, raw, {})
    assert bound.graph == raw
    assert bound.evidence["method"] == "source_visible_placeholders"


def test_new_url_cannot_rebind_existing_opaque_token():
    result = preprocess_training_urls("Existing [SOURCE_URL_1], new https://one.invalid/a", {"text": "[SOURCE_URL_1]"})
    assert "[SOURCE_URL_1]" not in result.url_map
    assert result.canonical_graph["text"] == "[SOURCE_URL_1]"
    assert "[SOURCE_URL_2]" in result.url_map


def test_explicit_map_cannot_override_recorded_graph_identity():
    row = record()
    row["reference_map"] = {**MAP, "[URL_1]": "https://different.invalid"}
    with pytest.raises(ValueError, match="explicit_reference_restoration_hash_mismatch"):
        bind_source_target_references(SOURCE, decode_express_completion(RAW), row)


def test_declared_scenario_family_survives_import(tmp_path):
    row = record()
    row["scenario_family_id"] = "booking_hotel"
    row["source_quality"] = {"status": "checks_passed", "training_eligibility": "eligible"}
    _, accepted, rejected = prepare(tmp_path, row, filters={"require_source_contract_checks": True})
    assert not rejected
    assert accepted[0]["scenario_family_id"] == "booking_hotel"
    assert accepted[0]["source_quality"]["training_eligibility"] == "eligible"


def test_generic_url_placeholder_is_not_removed_but_malformed_one_is_rejected():
    graph = decode_express_completion('<a2ui>root=Button("Open",onPress=openUrl("[URL_1]"))</a2ui>')
    assert not repair_graph(graph).changes
    graph["elements"]["root"]["on"]["press"]["params"]["url"] = "[URL_URL_1]"
    assert repair_graph(graph).changes[0].kind == "unsafe_url_action_removed"


def test_import_rejects_action_removal_instead_of_teaching_inert_button(tmp_path):
    raw = '<a2ui>root=Button("Open",onPress=openUrl("javascript:alert(1)"))</a2ui>'
    _, accepted, rejected = prepare(tmp_path, {"response_id": "r1", "a2ui_express": raw}, source="Open")
    assert not accepted
    assert "semantic_action_removal" in rejected[0]["reason"]


def test_unknown_target_reference_is_rejected():
    with pytest.raises(ValueError, match="unbound_target"):
        ground_preprocessed_references("Show [IMAGE_URL_1]", {"url": "[IMAGE_URL_2]"}, [], {})


def test_declared_asset_identity_is_exposed_without_loading_asset():
    result = ground_preprocessed_references("Show image", {"url": "[IMAGE_ASSET_1]"}, [{"url": "[IMAGE_ASSET_1]"}], {})
    assert "Available asset references: [IMAGE_ASSET_1]" in result


def test_role_aliases_preserve_known_source_identity():
    mapping = {"[SOURCE_URL_1]": {"url": "https://one.invalid"}, "[ACTION_URL_1]": {"url": "https://one.invalid"}}
    result = ground_preprocessed_references("Reference [SOURCE_URL_1]", {"url": "[ACTION_URL_1]"}, [], mapping)
    assert "[ACTION_URL_1] = [SOURCE_URL_1]" in result


def test_failed_source_contract_is_excluded(tmp_path):
    row = record()
    row["source_quality"] = {"status": "failed", "training_eligibility": "exclude"}
    _, accepted, rejected = prepare(tmp_path, row)
    assert not accepted
    assert rejected[0]["reason"] == "source_quality_failed"


def test_later_failed_source_audit_cannot_be_hidden_by_old_stage3_pass(tmp_path):
    row = record()
    row["source_quality"] = {"status": "checks_passed", "training_eligibility": "eligible"}
    _, accepted, rejected = prepare(tmp_path, row, response_quality={"status": "failed", "training_eligibility": "exclude"})
    assert not accepted
    assert rejected[0]["reason"] == "source_quality_failed"


def test_conflicting_source_contract_identities_are_quarantined(tmp_path):
    row = record()
    row["source_quality"] = {"status": "checks_passed", "training_eligibility": "eligible", "contract_sha256": "old"}
    _, accepted, rejected = prepare(tmp_path, row, response_quality={**row["source_quality"], "contract_sha256": "new"})
    assert not accepted
    assert rejected[0]["reason"] == "source_quality_metadata_conflict"


@pytest.mark.parametrize("metadata", [
    {"record_status": "quality_rejected"},
    {"training_acceptance": {"blocking_reasons": ["unsupported_external_action"]}},
])
def test_semantically_rejected_target_cannot_enter_training(tmp_path, metadata):
    _, accepted, rejected = prepare(tmp_path, {**record(), **metadata})
    assert not accepted
    assert rejected[0]["reason"] == "quality_rejected"


def test_strict_semantic_admission_requires_explicit_eligibility(tmp_path):
    row = {**record(), "training_acceptance": {"eligible": False, "review_reasons": ["table_content_difference"]}}
    _, accepted, rejected = prepare(tmp_path, row, filters={"require_semantic_acceptance": True})
    assert not accepted
    assert rejected[0]["reason"] == "semantic_acceptance_review_required"


def test_disabling_url_serialization_cannot_disable_destination_grounding(tmp_path):
    raw = '<a2ui>root=Button("Book",onPress=openUrl("https://invented.invalid/book"))</a2ui>'
    _, accepted, rejected = prepare(tmp_path, {"response_id": "r1", "a2ui_express": raw},
                                    source="Display hotel information.", url_preprocessing={"enabled": False})
    assert not accepted
    assert "unbound_target_references" in rejected[0]["reason"]


def test_legacy_source_review_is_explicit_and_strict_contract_mode_is_optional(tmp_path):
    manifest, accepted, rejected = prepare(tmp_path, record(), filters={"require_source_contract_checks": True})
    assert not accepted
    assert rejected[0]["reason"] == "source_contract_checks_required"
