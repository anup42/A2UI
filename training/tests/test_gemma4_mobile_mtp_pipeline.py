from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ir_training.common.config import load_yaml
from ir_training.export.litertlm_mtp import (
    LiteRTLMMTPPackagingError,
    _replace_same_size_section,
    compare_target_layout,
)
from ir_training.pipeline.gemma4_mobile_mtp import build_pipeline_plan


def test_mobile_mtp_pipeline_plan_is_qat_and_plan_only():
    config_path = ROOT / "configs" / "pipelines" / "gemma4_e2b_mobile_mtp.yaml"
    config = load_yaml(config_path)
    plan = build_pipeline_plan(config, config_path=config_path)

    assert plan["training"]["qat_profile"] == "gemma4_e2b_mobile_observable_wna8o8_approx"
    assert plan["training"]["best_checkpoint_required"] is True
    assert plan["package"]["mtp_enabled"] is True
    assert plan["package"]["mtp_model_type"] == "tf_lite_mtp_drafter"
    assert plan["android_gpu"]["mtp_flag"] is True
    assert any(item["code"] == "missing_base_package" for item in plan["validation"]["issues"])


def test_replace_same_size_section_preserves_prefix_and_suffix(tmp_path):
    source = tmp_path / "base.litertlm"
    source.write_bytes(b"prefix" + b"target" + b"suffix")
    output = tmp_path / "out.litertlm"
    _replace_same_size_section(
        source,
        {"begin_offset": 6, "size": 6},
        b"packed",
        output,
    )
    assert output.read_bytes() == b"prefix" + b"packed" + b"suffix"


def test_replace_same_size_section_rejects_size_change(tmp_path):
    source = tmp_path / "base.litertlm"
    source.write_bytes(b"prefix" + b"target" + b"suffix")
    with pytest.raises(LiteRTLMMTPPackagingError, match="header rewrite"):
        _replace_same_size_section(
            source,
            {"begin_offset": 6, "size": 6},
            b"larger-target",
            tmp_path / "out.litertlm",
        )


def test_compare_target_layout_ignores_buffer_indices_but_requires_tflite_graphs():
    pytest.importorskip("flatbuffers")
    pytest.importorskip("ai_edge_litert.schema_py_generated")
    from build_fresh_random_quantized_graph import _build_fresh_tflite

    records = [
        {
            "ordinal": 0,
            "operator": "FULLY_CONNECTED",
            "bits": 8,
            "shape": [4, 3],
            "type_value": 9,
        }
    ]
    graph = _build_fresh_tflite(records, seed=7)
    result = compare_target_layout(graph, graph)
    assert result["available"] is True
    assert result["ok"] is True
    assert result["structural_match"] is True
    assert result["quantization_layout_match"] is True
    assert result["buffer_storage_match"] is True
