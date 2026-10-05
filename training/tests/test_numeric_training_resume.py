"""Numerical resume compatibility only; no model, optimizer, or torch load."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.train.numeric_resume import verify_numeric_training_resume


def _config(**qat):
    return {"qat": {"enabled": True, "ste_gradient": "clipped", **qat}}


def _write(path, metadata):
    (path / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _metadata():
    return {"qat": {"numeric_contract": {"weight_ste_rule": "rounded_code_range_v1"}}}


def test_current_clipped_rule_can_resume(tmp_path):
    _write(tmp_path, _metadata())
    report = verify_numeric_training_resume(tmp_path, _config())
    assert report == {"checked": True, "verified": True, "weight_ste_rule": "rounded_code_range_v1"}


@pytest.mark.parametrize("metadata", [
    {}, {"qat": None}, {"qat": {"numeric_contract": {"ste_gradient": "clipped"}}},
    {"qat": {"numeric_contract": {"weight_ste_rule": "floating_endpoint_range_v0"}}},
    {"qat": {"numeric_contract": []}},
])
def test_legacy_or_changed_clipped_rule_cannot_resume(tmp_path, metadata):
    _write(tmp_path, metadata)
    with pytest.raises(ValueError, match="weight_ste_rule.*absent or changed"):
        verify_numeric_training_resume(tmp_path, _config())


@pytest.mark.parametrize("config", [
    {}, {"qat": {"enabled": False, "ste_gradient": "clipped"}},
    _config(ste_gradient="identity"), _config(weight_bits=32, activation_bits=32),
    _config(weight_bits=32, activation_quantizer="gemma_mobile_srq"),
])
def test_unchanged_paths_do_not_require_new_metadata(tmp_path, config):
    # Deliberately absent metadata: these paths retain their existing checks.
    assert verify_numeric_training_resume(tmp_path, config)["checked"] is False


def test_module_override_to_low_bit_requires_the_rule(tmp_path):
    _write(tmp_path, {})
    with pytest.raises(ValueError, match="weight_ste_rule"):
        verify_numeric_training_resume(tmp_path, _config(
            weight_bits=32, activation_bits=32, module_quant_configs={"q_proj": 4}
        ))


def test_legacy_activation_helper_requires_the_rule(tmp_path):
    _write(tmp_path, {})
    with pytest.raises(ValueError, match="weight_ste_rule"):
        verify_numeric_training_resume(tmp_path, _config(weight_bits=32, activation_bits=8))


def test_schema_selected_clipped_rule_is_checked(tmp_path):
    schema = tmp_path / "quantization.json"
    schema.write_text(json.dumps({"ste_gradient": "clipped"}), encoding="utf-8")
    _write(tmp_path, {})
    with pytest.raises(ValueError, match="weight_ste_rule"):
        verify_numeric_training_resume(tmp_path, {"qat": {"enabled": True, "schema_path": str(schema)}})


def test_matching_full_precision_can_resume_without_qat(tmp_path):
    precision = {"policy": "fp32_trainable_v1", "master_dtype": "float32", "amp": "bfloat16"}
    _write(tmp_path, {"full_finetune_precision": precision})
    report = verify_numeric_training_resume(tmp_path, {}, expected_full_precision=precision)
    assert report["verified"] is True
    assert report["full_finetune_precision"] == precision


@pytest.mark.parametrize("actual", [None, {}, {"policy": "legacy_bf16"}, {"version": True}])
def test_missing_or_changed_full_precision_cannot_resume(tmp_path, actual):
    _write(tmp_path, {"full_finetune_precision": actual})
    with pytest.raises(ValueError, match="full_finetune_precision.*absent or changed"):
        verify_numeric_training_resume(tmp_path, {}, expected_full_precision={"version": 1})


def test_both_numeric_policies_are_checked(tmp_path):
    precision = {"policy": "fp32_trainable_v1"}
    metadata = _metadata()
    metadata["full_finetune_precision"] = precision
    _write(tmp_path, metadata)
    report = verify_numeric_training_resume(tmp_path, _config(), expected_full_precision=precision)
    assert report["full_finetune_precision"] == precision
    assert report["weight_ste_rule"] == "rounded_code_range_v1"


@pytest.mark.parametrize("content", ["[]", "null", "{broken"])
def test_invalid_metadata_fails_with_restart_guidance(tmp_path, content):
    (tmp_path / "training_metadata.json").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Start a new run.*weight-only"):
        verify_numeric_training_resume(tmp_path, _config())


def test_missing_metadata_fails_with_restart_guidance(tmp_path):
    with pytest.raises(ValueError, match="Start a new run.*weight-only"):
        verify_numeric_training_resume(tmp_path, _config())
