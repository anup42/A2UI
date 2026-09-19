"""Opt-in export selection preserves the default four-variant workflow."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from ir_training.pipeline import checkpoint_export as ce
from ir_training.pipeline import deployment_export as de
from test_checkpoint_export import inputs, write_json


def test_mixed248_is_not_implicitly_added_to_training_deployment(tmp_path):
    assert tuple(de.deployment_variants("e2b")) == de.VARIANTS
    plan = ce.build_checkpoint_export_plan(inputs(tmp_path))
    assert tuple(plan["variants"]) == de.VARIANTS
    assert "--variants" not in plan["probe_command"]


def test_selected_mixed248_plan_has_no_other_conversion_training_runtime_or_mtp(tmp_path):
    options = replace(inputs(tmp_path), variants=("w248",))
    plan = ce.run_checkpoint_export(options)
    assert [s["name"] for s in plan["stages"]] == ["exporter_preflight", "merge", "export_w248"]
    assert plan["probe_command"][-2:] == ["--variants", "w248"]
    assert plan["variants"]["w248"]["weight_bits"] == [2, 4, 8]
    assert plan["variants"]["w248"]["official_graph"] is False
    assert plan["variants"]["w248"]["official_qat"] is False
    assert plan["mtp_exported"] is False and plan["training_executed"] is False
    assert plan["quality_evaluation_performed"] is False
    assert not options.output_dir.exists()


@pytest.mark.parametrize("selection", [(), ("w248", "w248"), ("w2",), "w248"])
def test_invalid_selection_rejected(tmp_path, selection):
    with pytest.raises(ValueError, match="variants"):
        ce.build_checkpoint_export_plan(replace(inputs(tmp_path), variants=selection))


def test_270m_cannot_receive_e2b_mixed248_policy(tmp_path):
    with pytest.raises(ValueError, match="E2B-only"):
        ce.build_checkpoint_export_plan(replace(inputs(tmp_path, "270m"), variants=("w248",)))


def test_mixed248_requires_explicit_experimental_ack_before_any_output(tmp_path):
    options = replace(inputs(tmp_path), variants=("w248",), allow_experimental_formats=False)
    with pytest.raises(ValueError, match="allow-experimental-formats"):
        ce.run_checkpoint_export(options, execute=True)
    assert not options.output_dir.exists()


def test_preflight_must_cover_mixed248_not_just_old_w4(tmp_path):
    plan = ce.build_checkpoint_export_plan(replace(inputs(tmp_path), variants=("w248",)))
    report_path = Path(plan["output_dir"]) / "exporter_preflight.json"
    write_json(report_path, {"status": "passed", "profile": "e2b", "variants": ["w4"]})
    with pytest.raises(ValueError, match="requested variants"):
        ce._validate_stage(plan, "exporter_preflight")
    write_json(report_path, {"status": "passed", "profile": "e2b", "variants": ["w248"]})
    assert ce._validate_stage(plan, "exporter_preflight") == {}


def test_checkpoint_cli_accepts_mixed248_selection_as_read_only_plan(tmp_path):
    options = inputs(tmp_path)
    script = Path(ce.__file__).resolve().parents[3] / "scripts/export_checkpoint_litertlm.py"
    result = subprocess.run([
        sys.executable, str(script), "--profile", "e2b", "--fit-dir", str(options.fit_dir),
        "--output-dir", str(options.output_dir), "--exporter-python", sys.executable,
        "--variants", "w248",
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert list(plan["variants"]) == ["w248"]
    assert not options.output_dir.exists()


def test_probe_cannot_silently_ignore_convert_variant_flag():
    script = Path(ce.__file__).resolve().parents[3] / "scripts/deployment_export.py"
    result = subprocess.run([
        sys.executable, str(script), "probe", "--profile", "e2b", "--variant", "w248",
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "probe takes --variants" in result.stderr
