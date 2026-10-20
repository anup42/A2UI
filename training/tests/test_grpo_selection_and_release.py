"""CPU-only promotion, scorer binding, and post-export release regressions."""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
from ir_training.train.grpo_experiment import (
    GUARDS,
    SELECTOR,
    QualityMonitor,
    QualityPolicy,
    file_sha256,
    verify_comparable_aggregates,
    verify_expected_scorer,
    verify_quality_gate,
)

ROOT = Path(__file__).resolve().parents[2]


def aggregate(score=50.0, guard=0.75):
    return {
        SELECTOR: score, **dict.fromkeys(GUARDS, guard),
        "golden_set_sha256": "source-cohort", "golden_set_rows": 32,
        "unique_source_count": 31,
        "metric_identity_v5_4": {"metric_fingerprint": "a" * 64,
                                "reward_pipeline_fingerprint": "b" * 64},
    }


def selection_probe(tmp_path, eligibility):
    """Run the exact production save-decision method without model imports."""
    source = ROOT / "training/src/ir_training/train/callbacks.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "_record_best_if_improved")
    writes = []
    namespace = {
        "Any": Any, "Path": Path, "json": json,
        "metric_for_best_model": SELECTOR, "zero3_trainer": None,
        "greater_is_better": True, "save_best_checkpoint": True,
        "resolved_best_checkpoint_dir": tmp_path / "best",
        "resolved_output_dir": tmp_path, "tokenizer": None,
        "clear_generation_cache_on_save": False,
        "_finite_float": float, "_metric_improved": lambda current, best, greater: current > best,
        "_distributed_context": lambda: (0, 1),
        "_save_best_golden_checkpoint": lambda **kw: writes.append(kw["best_info"]["step"]),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    selector = SimpleNamespace(best_metric_value=-math.inf, checkpoint_eligibility=eligibility,
                               evaluation_count=0, best_metric_step=None)
    selector.save = MethodType(namespace["_record_best_if_improved"], selector)
    return selector, writes


def test_best_eligible_survives_higher_scoring_unsafe_candidate(tmp_path):
    monitor = QualityMonitor(aggregate(), QualityPolicy())
    selector, writes = selection_probe(tmp_path, monitor.promotion)
    for step, metrics in ((25, aggregate(51)), (50, aggregate(60, .74)), (75, aggregate(55))):
        selector.save(model=None, state=SimpleNamespace(global_step=step, epoch=0),
                      event_label=f"step{step}", event_dir=tmp_path, aggregate=metrics)
    assert writes == [25, 75]
    assert selector.best_metric_step == 75
    assert selector.best_metric_value == 55


def test_unconfigured_sft_selector_remains_scalar_only(tmp_path):
    selector, writes = selection_probe(tmp_path, None)
    for step, metrics in ((25, aggregate(51)), (50, aggregate(60, .74)), (75, aggregate(55))):
        selector.save(model=None, state=SimpleNamespace(global_step=step, epoch=0),
                      event_label=f"step{step}", event_dir=tmp_path, aggregate=metrics)
    assert writes == [25, 50]
    assert selector.best_metric_step == 50


@pytest.mark.parametrize("decision", [None, {}, {"passed": 1}, {"passed": "true"}])
def test_eligibility_hook_fails_closed_without_overwriting(tmp_path, decision):
    selector, writes = selection_probe(tmp_path, lambda metrics: decision)
    with pytest.raises(ValueError, match="explicit passed boolean"):
        selector.save(model=None, state=SimpleNamespace(global_step=25, epoch=0),
                      event_label="step25", event_dir=tmp_path, aggregate=aggregate(55))
    assert writes == [] and selector.best_metric_step is None


@pytest.mark.parametrize("field", ["metric_fingerprint", "reward_pipeline_fingerprint"])
def test_baseline_and_candidate_cannot_mix_scorers(field):
    before, after = aggregate(), aggregate(51)
    after["metric_identity_v5_4"][field] = "c" * 64
    with pytest.raises(ValueError, match="scorer mismatch"):
        verify_comparable_aggregates(before, after, require_scorer_identity=True)


def test_new_contract_rejects_missing_identity_but_legacy_is_explicit():
    before, after = aggregate(), aggregate(51)
    del before["metric_identity_v5_4"]
    with pytest.raises(ValueError, match="lacks canonical scorer"):
        verify_comparable_aggregates(before, after, require_scorer_identity=True)
    verify_comparable_aggregates(before, after)
    after["unique_source_count"] = 30
    with pytest.raises(ValueError, match="cohort mismatch"):
        verify_comparable_aggregates(before, after)


def test_quality_gate_binds_current_scorer_identity(tmp_path, monkeypatch):
    import ir_training.train.grpo_experiment as experiment
    monkeypatch.setattr(experiment, "expected_scorer_identity", lambda config: aggregate()["metric_identity_v5_4"])
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"grpo": {"quality_control": {"enabled": True},
                                         "reward_implementation_binding": "sha256_v1"}}))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    adapter = checkpoint / "adapter_model.safetensors"
    adapter.write_bytes(b"test-only")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(aggregate()))
    selected = checkpoint / "aggregate_metrics.json"
    selected.write_text(json.dumps(aggregate(55)))
    gate = {
        "schema_version": 2, **QualityMonitor(aggregate(), QualityPolicy()).promotion(aggregate(55)),
        "training_config_sha256": file_sha256(config), "selected_adapter_sha256": file_sha256(adapter),
        "selected_aggregate_sha256": file_sha256(selected), "baseline_path": str(baseline),
        "baseline_sha256": file_sha256(baseline), "scorer_identity": aggregate()["metric_identity_v5_4"],
    }
    gate_path = tmp_path / "grpo_quality_gate.json"
    gate_path.write_text(json.dumps(gate))
    assert len(verify_quality_gate(tmp_path, checkpoint, config)) == 3
    gate["scorer_identity"]["metric_fingerprint"] = "0" * 64
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="canonical scorer identities"):
        verify_quality_gate(tmp_path, checkpoint, config)


def test_matching_historical_evaluations_cannot_masquerade_as_current_scorer():
    before, after = aggregate(), aggregate(55)
    verify_comparable_aggregates(before, after, require_scorer_identity=True)
    with pytest.raises(ValueError, match="configured reward scorer"):
        verify_expected_scorer(after, {"metric_fingerprint": "f" * 64,
                                       "reward_pipeline_fingerprint": "b" * 64})


def native_fixture():
    from ir_training.eval.qat_grpo_native import NATIVE_QUALITY_POLICY, native
    binding = {"config": {"grpo": {"native_quality_policy": dict(NATIVE_QUALITY_POLICY)}},
               "reward_implementation": aggregate()["metric_identity_v5_4"],
               "model_sha256": "c" * 64}
    results = {cohort: {
        "count": count, "generation_reward_v5_4_avg": 80., "schema_valid_strict_rate": .9,
        "cap_v5_4_avg": .8, "fully_root_reachable_v5_4_avg": 1.,
        "metric_identity_v5_4": aggregate()["metric_identity_v5_4"],
    } for cohort, count in native.COHORTS.items()}
    return binding, results


def test_native_quality_and_execution_success_are_distinct():
    from ir_training.eval.qat_grpo_native import evaluate_native_quality_release
    binding, before = native_fixture()
    after = copy.deepcopy(before)
    assert evaluate_native_quality_release(binding, after, before)["accepted"] is True
    after["bixby50"]["generation_reward_v5_4_avg"] -= 1
    gate = evaluate_native_quality_release(binding, after, before)
    assert gate["accepted"] is False and gate["status"] == "failed"
    assert "regressed:bixby50/generation_reward_v5_4_avg" in gate["issues"]
    binding["config"]["grpo"].pop("native_quality_policy")
    gate = evaluate_native_quality_release(binding, after, before)
    assert gate["accepted"] is None and gate["status"] == "not_configured"


@pytest.mark.parametrize("tamper", ["count", "missing_cohort", "nan", "missing_metric", "scorer", "relaxed"])
def test_native_release_invalid_evidence_fails_closed(tamper):
    from ir_training.eval.qat_grpo_native import evaluate_native_quality_release
    binding, before = native_fixture()
    after = copy.deepcopy(before)
    if tamper == "count":
        after["golden35"]["count"] = 34
    elif tamper == "missing_cohort":
        after.pop("golden35")
    elif tamper == "nan":
        after["golden35"]["generation_reward_v5_4_avg"] = float("nan")
    elif tamper == "missing_metric":
        after["golden35"].pop("cap_v5_4_avg")
    elif tamper == "scorer":
        after["golden35"]["metric_identity_v5_4"]["metric_fingerprint"] = "f" * 64
    elif tamper == "relaxed":
        binding["config"]["grpo"]["native_quality_policy"]["max_guard_drop"] = .1
    with pytest.raises(ValueError):
        evaluate_native_quality_release(binding, after, before)


@pytest.mark.parametrize("tamper", [None, "missing", "mapping", "metric", "pipeline"])
def test_new_grpo_provenance_binds_reward_implementation(tmp_path, tamper):
    from ir_training.train.qat_grpo_contract import verify_qat_grpo_provenance
    from ir_training.train.qat_grpo_reward import reward_implementation_identity
    from test_qat_grpo_export import _fixture

    config, metadata, _, _ = _fixture(tmp_path)
    config["grpo"]["reward_implementation_binding"] = "sha256_v1"
    identity = reward_implementation_identity(ROOT / "dataset/configs/genui_metric_v5_4.yaml")
    metadata["grpo"]["reward"]["implementation"] = identity
    if tamper == "missing":
        metadata["grpo"]["reward"].pop("implementation")
    elif tamper:
        identity[{"mapping": "mapping_sha256", "metric": "metric_fingerprint",
                  "pipeline": "reward_pipeline_fingerprint"}[tamper]] = "0" * 64
    report = verify_qat_grpo_provenance(config, metadata)
    assert report["verified"] is (tamper is None), report
    assert report["checks"]["reward_implementation_bound"] is (tamper is None)


@pytest.mark.parametrize("family", ["e2b", "270m"])
@pytest.mark.parametrize("tamper", [None, "missing_rank", "changed_bytes", "failed_bound_report", "removed_stage"])
def test_native_inspector_accepts_and_verifies_real_runtime_probe_stage(tmp_path, monkeypatch, family, tamper):
    from ir_training.eval.qat_grpo_native import inspect_qat_grpo_run, native
    from test_qat_grpo_native import fixture, write

    run, _ = fixture(tmp_path, monkeypatch, family, runtime_preflight=True)
    report_path = run / "runtime_preflight/grpo_runtime_probe.rank1.json"
    manifest_path = run / "qat_grpo_manifest.json"
    manifest = native.read_json(manifest_path)
    if tamper == "missing_rank":
        report_path.unlink()
    elif tamper in {"changed_bytes", "failed_bound_report"}:
        report = native.read_json(report_path)
        report["passed"] = False
        write(report_path, report)
        if tamper == "failed_bound_report":
            receipt = manifest["completed"]["runtime_preflight"]
            original_path = "/training/grpo-run/runtime_preflight/grpo_runtime_probe.rank1.json"
            receipt["files"][original_path] = native.file_sha256(report_path)
            write(run / "stage_receipts/runtime_preflight.json", receipt)
            write(manifest_path, manifest)
    elif tamper == "removed_stage":
        manifest["completed"].pop("runtime_preflight")
        write(manifest_path, manifest)
    if tamper is None:
        binding = inspect_qat_grpo_run(run)
        assert "runtime_preflight" in binding["receipt_bindings"]
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            inspect_qat_grpo_run(run)
