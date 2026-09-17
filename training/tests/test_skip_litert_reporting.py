"""Export-only reports disclose skipped runtime tests without inventing scores."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.pipeline import result_tables as module


COHORTS = {"golden32": 32, "golden35": 35, "bixby50": 50}


def state(*, skip=True):
    return {
        "status": "complete",
        "plan": {"skip_litert_evaluation": skip,
                 "training": {"goldens": {key: {"rows": count} for key, count in COHORTS.items()}}},
        "results": {
            f"{label}_{cohort}": {"row_count": count, "aggregate": {
                module.REWARD_METRIC: 73.2, module.UNIQUE_REWARD_METRIC: 72.0,
                module.STRICT_METRIC: .9}}
            for label in module.DEPLOYMENT_LABELS[:3] for cohort, count in COHORTS.items()},
        "exports": {label: {"artifact": f"/exports/{label}/model.litertlm"}
                    for label in module.LITERT_LABELS},
    }


def model_rows(text):
    return [row for row in text.splitlines()
            if any(row.startswith(f"| {label} ") for label in module.DEPLOYMENT_LABELS)]


def test_export_only_discloses_nine_evaluations_and_twelve_skips():
    snapshot = state()
    before = deepcopy(snapshot)
    report = module.render_deployment_results(snapshot)
    rows = model_rows(report)
    assert len(rows) == 21
    assert len([row for row in rows if "evaluated" in row]) == 9
    skipped = [row for row in rows if "skipped by request" in row]
    assert len(skipped) == 12
    for row in skipped:
        assert row.count("n/a") == 4
        assert "unknown" not in row and "0.0000" not in row and "failed" not in row
    assert "Native runtime not validated" in report
    assert "Vulkan preflight and LiteRT-LM inference/scoring were skipped by request" in report
    assert "Artifact validation does not establish runtime compatibility or model quality" in report
    assert report.count("exported; artifact validated") == 4
    assert "Bixby50 uses source-response scoring; no reference IR" in report
    assert snapshot == before


def test_skipped_mode_never_prints_stray_native_scores():
    snapshot = state()
    snapshot["results"]["w4_bixby50"] = {"row_count": 50,
        "aggregate": {module.REWARD_METRIC: 99.1234, module.STRICT_METRIC: 1}}
    report = module.render_deployment_results(snapshot)
    assert "99.1234" not in report
    assert "skipped by request" in next(row for row in model_rows(report)
                                       if row.startswith("| w4 ") and "Bixby50" in row)


@pytest.mark.parametrize("skip", [False, None, "true"])
def test_no_explicit_flag_keeps_native_missing_evidence_behavior(skip):
    report = module.render_deployment_results(state(skip=skip))
    assert len([row for row in model_rows(report) if "not reported" in row]) == 12
    assert "skipped by request" not in report and "export-only" not in report
    assert "LiteRT export artifacts" not in report


def test_export_failure_is_separate_from_skipped_evaluation():
    snapshot = state()
    snapshot.update(status="failed", active_stage="export_w16", error="conversion failed")
    snapshot["exports"] = {"w32": snapshot["exports"]["w32"]}
    report = module.render_deployment_results(snapshot)
    assert report.count("skipped by request") == 13  # mode disclosure + twelve slots
    assert "| W16" in report and any(row.startswith("| W16") and "failed" in row
                                     for row in report.splitlines())
    assert report.count("exported; artifact validated") == 1
    assert any(row.startswith("| W8") and "not reported" in row for row in report.splitlines())
    assert "Stopped at: export_w16" in report
    assert "Failure: conversion failed" in report


def test_unstarted_export_mode_does_not_claim_artifacts_exist():
    snapshot = state()
    snapshot.update(status="failed", active_stage="training", exports={}, results={})
    report = module.render_deployment_results(snapshot)
    assert "exported; artifact validated" not in report
    assert len([row for row in model_rows(report) if "not reported" in row]) == 9
    assert len([row for row in model_rows(report) if "skipped by request" in row]) == 12


def test_old_manifest_without_mode_field_remains_full_evaluation_report():
    snapshot = state()
    snapshot["plan"].pop("skip_litert_evaluation")
    snapshot["results"].update({
        f"{label}_{cohort}": {"row_count": count, "aggregate": {module.REWARD_METRIC: 5}}
        for label in module.LITERT_LABELS for cohort, count in COHORTS.items()})
    report = module.render_deployment_results(snapshot)
    assert report.count("| evaluated") == 21
    assert "skipped by request" not in report
