"""Bounded CPU regressions; no training, optimizer or real model conversion."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.qat.numeric_preflight import (
    LEGACY_POLICY,
    OFFICIAL_MOBILE_WORKFLOW,
    RETAINED_MOBILE_POLICY,
    numeric_preflight_provenance,
    resolve_numeric_policy,
)
from ir_training.qat.workflow import validate_qat_config
from ir_training.train import sft


def official_config() -> dict:
    config = load_yaml(ROOT / "configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml")
    config["run"]["purpose"] = OFFICIAL_MOBILE_WORKFLOW
    config["preflight"]["numeric_policy"] = RETAINED_MOBILE_POLICY
    return config


def numeric_probe(loss: float, *, qat: bool = False) -> dict:
    return {
        "completion_loss": loss, "row_losses": [loss], "rows_checked": 1,
        "completion_tokens": 128, "logits_finite": True,
        "top1_probe_ids": list(range(95)) + list(range(200, 233)) if qat else list(range(128)),
        "top_token_probe_sha256": "b" * 64 if qat else "a" * 64,
    }


def greedy_probe(*, qat: bool = False) -> dict:
    sequence = list(range(90, 98)) if qat else list(range(1, 9))
    repeats = 2 if qat else 1
    return {
        "rows_checked": 1, "repeats": repeats, "min_new_tokens": 8,
        "max_new_tokens": 32, "deterministic": True,
        "generated_token_ids": [[sequence.copy()] for _ in range(repeats)],
        "generated_token_counts": [[8] for _ in range(repeats)],
    }


def official_evidence(config: dict | None = None) -> dict:
    config = config or official_config()
    report = sft._compare_initial_numeric_reports(
        numeric_probe(1.32057861), numeric_probe(0.62379508, qat=True),
        preflight_cfg=config["preflight"], qat_enabled=True,
    )
    report["greedy_generation"] = sft._compare_initial_greedy_reports(
        greedy_probe(), greedy_probe(qat=True),
        preflight_cfg=config["preflight"], qat_enabled=True,
    )
    report["zero_adapter_initialization"] = {
        "verified_zero_delta": True, "wrapper_count": 205,
        "adapter_pair_count": 205, "nonzero_or_invalid_pairs": [],
    }
    report["adapter_initialization_mode"] = "fresh_zero_adapter"
    return report


def test_observed_h100_values_are_diagnostic_only_for_explicit_official_policy():
    config = official_config()
    assert resolve_numeric_policy(config) == RETAINED_MOBILE_POLICY
    assert validate_qat_config(config) == []
    report = official_evidence(config)
    assert report["passed"] is True
    assert report["min_top1_probe_match"] == 0.90  # Not reduced to accept this run.
    assert report["top1_probe_match_fraction"] == pytest.approx(0.742188, abs=1e-6)
    assert report["top_token_probe_equal"] is False
    assert report["loss_increase"] == pytest.approx(0.62379508 - 1.32057861)
    assert report["loss_ratio"] == pytest.approx(0.62379508 / 1.32057861)
    assert report["cross_mode_comparison"] == "diagnostic"
    assert all(report["mandatory_checks"].values())
    greedy = report["greedy_generation"]
    assert greedy["min_baseline_qat_greedy_prefix_tokens"] == 8
    assert greedy["baseline_qat_min_common_prefix_tokens"] == 0
    assert greedy["baseline_qat_positional_match_fraction"] == 0
    assert greedy["qat_greedy_deterministic"] is True
    assert numeric_preflight_provenance(config, report)["verified"] is True


def test_legacy_h100_top1_gate_is_unchanged():
    config = official_config()
    config["run"].pop("purpose")
    config["preflight"].pop("numeric_policy")
    assert resolve_numeric_policy(config) == LEGACY_POLICY
    with pytest.raises(RuntimeError, match="numeric parity failed"):
        sft._compare_initial_numeric_reports(
            numeric_probe(1.32057861), numeric_probe(0.62379508, qat=True),
            preflight_cfg=config["preflight"], qat_enabled=True,
        )


@pytest.mark.parametrize("baseline,qat", [(1.0, 2.0), (0.1, 14.9)])
def test_loss_delta_and_ratio_are_diagnostic_but_absolute_bound_remains(baseline, qat):
    preflight = official_config()["preflight"]
    report = sft._compare_initial_numeric_reports(
        numeric_probe(baseline), numeric_probe(qat), preflight_cfg=preflight, qat_enabled=True,
    )
    assert report["loss_ratio"] > 1.10
    assert report["passed"] is True
    preflight.pop("numeric_policy")
    with pytest.raises(RuntimeError, match="numeric parity failed"):
        sft._compare_initial_numeric_reports(
            numeric_probe(baseline), numeric_probe(qat), preflight_cfg=preflight, qat_enabled=True,
        )


@pytest.mark.parametrize("side", ["baseline", "qat_on"])
@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_losses_still_fail(side, loss):
    probes = {"baseline": numeric_probe(1.0), "qat_on": numeric_probe(1.0)}
    probes[side]["completion_loss"] = loss
    with pytest.raises(RuntimeError, match="non-finite"):
        sft._compare_initial_numeric_reports(
            **probes, preflight_cfg=official_config()["preflight"], qat_enabled=True,
        )


@pytest.mark.parametrize("change", [
    {"completion_loss": 15.01}, {"completion_loss": -1},
    {"row_losses": [float("nan")]}, {"logits_finite": False},
    {"top1_probe_ids": []}, {"top1_probe_ids": [1]}, {"rows_checked": 0},
    {"completion_tokens": 127},
])
def test_invalid_numeric_evidence_or_excessive_loss_still_fails(change):
    qat = numeric_probe(1.0)
    qat.update(change)
    with pytest.raises(RuntimeError, match="numeric safety gate failed"):
        sft._compare_initial_numeric_reports(
            numeric_probe(1.0), qat, preflight_cfg=official_config()["preflight"], qat_enabled=True,
        )


@pytest.mark.parametrize("tamper", ["nondeterministic", "false_flag", "one_repeat", "missing_row", "empty", "short"])
def test_greedy_requires_real_repeated_determinism_and_probe_integrity(tamper):
    qat = greedy_probe(qat=True)
    if tamper == "nondeterministic":
        qat["generated_token_ids"][1][0][0] = 999  # Keep the summary flag true.
    elif tamper == "false_flag":
        qat["deterministic"] = False
    elif tamper == "one_repeat":
        qat["repeats"] = 1
        qat["generated_token_ids"].pop()
        qat["generated_token_counts"].pop()
    elif tamper == "missing_row":
        qat["generated_token_ids"][1] = []
    else:
        sequence = [] if tamper == "empty" else [1]
        qat["generated_token_ids"] = [[sequence], [sequence]]
        qat["generated_token_counts"] = [[len(sequence)], [len(sequence)]]
    with pytest.raises(RuntimeError, match="deterministic|greedy safety gate"):
        sft._compare_initial_greedy_reports(
            greedy_probe(), qat, preflight_cfg=official_config()["preflight"], qat_enabled=True,
        )


def test_valid_early_eos_is_liveness_not_cross_mode_similarity():
    qat = greedy_probe(qat=True)
    qat.update(generated_token_ids=[[[99]], [[99]]], generated_token_counts=[[1], [1]],
               generation_diagnostics=[[{"stop_reason": "eos_token"}]] * 2)
    report = sft._compare_initial_greedy_reports(
        greedy_probe(), qat, preflight_cfg=official_config()["preflight"], qat_enabled=True,
    )
    assert report["passed"] is True
    assert report["baseline_qat_min_common_prefix_tokens"] == 0


@pytest.mark.parametrize("section,key,value", [
    ("preflight", "numeric_policy", None),
    ("preflight", "numeric_policy", "typo"),
    ("preflight", "numeric_policy", LEGACY_POLICY),
    ("run", "purpose", "other_workflow"),
    ("preflight", "require_zero_adapter_parity", False),
    ("preflight", "require_initial_loss_gate", False),
    ("preflight", "require_greedy_determinism", False),
    ("preflight", "max_initial_completion_loss", float("nan")),
    ("preflight", "max_initial_completion_loss", float("inf")),
    ("preflight", "max_initial_completion_loss", 16.0),
    ("preflight", "max_initial_completion_loss", 0),
    ("preflight", "rows", 0), ("preflight", "logit_probe_tokens", 0),
    ("qat", "enabled", False), ("qat", "expected_effective_lora_modules", 204),
    ("qat", "expected_frozen_activation_modules", 69),
    ("qat", "activation_quantizer", "legacy"),
    ("qat", "fixed_scale_required", False),
    ("qat", "fixed_activation_scale_required", False),
])
def test_policy_config_is_explicit_scoped_and_fail_closed(section, key, value):
    config = official_config()
    if value is None:
        config[section].pop(key)
    else:
        config[section][key] = value
    with pytest.raises(ValueError):
        resolve_numeric_policy(config)
    assert "invalid_numeric_preflight_policy" in {item.code for item in validate_qat_config(config)}
    assert numeric_preflight_provenance(config, official_evidence())["verified"] is False


@pytest.mark.parametrize("tamper", ["missing_policy", "legacy_policy", "missing_gates", "false_gate",
                                    "nonfinite", "excessive_loss", "missing_greedy", "forged_determinism",
                                    "nonzero_adapter", "wrong_zero_scope", "no_raw_probes",
                                    "nonboolean_gate", "extra_gate", "greedy_policy"])
def test_v2_provenance_requires_correct_policy_and_successful_mandatory_evidence(tamper):
    report = copy.deepcopy(official_evidence())
    if tamper == "missing_policy":
        report.pop("policy")
    elif tamper == "legacy_policy":
        report["policy"] = LEGACY_POLICY
    elif tamper == "missing_gates":
        report.pop("mandatory_checks")
    elif tamper == "false_gate":
        report["mandatory_checks"]["absolute_qat_loss_bounded"] = False
    elif tamper == "nonfinite":
        report["qat_on"]["completion_loss"] = float("nan")
    elif tamper == "excessive_loss":
        report["qat_on"]["completion_loss"] = 100.0
    elif tamper == "missing_greedy":
        report.pop("greedy_generation")
    elif tamper == "forged_determinism":
        report["greedy_generation"]["qat_on"]["generated_token_ids"][1][0][0] = 999
    elif tamper == "nonzero_adapter":
        report["zero_adapter_initialization"]["nonzero_or_invalid_pairs"] = ["bad"]
    elif tamper == "wrong_zero_scope":
        report["zero_adapter_initialization"]["wrapper_count"] = 204
    elif tamper == "nonboolean_gate":
        report["mandatory_checks"]["finite_logits"] = 1
    elif tamper == "extra_gate":
        report["mandatory_checks"]["invented_gate"] = True
    elif tamper == "greedy_policy":
        report["greedy_generation"].pop("policy")
    else:
        report.pop("qat_on")
    assert numeric_preflight_provenance(official_config(), report)["verified"] is False


def test_legacy_metadata_keeps_old_provenance_semantics_but_cannot_claim_new_policy():
    assert numeric_preflight_provenance({}, {"passed": True})["verified"] is True
    assert numeric_preflight_provenance({}, official_evidence())["verified"] is False


@pytest.mark.parametrize("tamper", [None, "zero", "hash", "state", "scope", "nonfinite", "greedy"])
def test_explicit_resume_requires_verified_loaded_adapter_and_all_safety_gates(tmp_path, monkeypatch, tamper):
    from ir_training.train import mobile_resume

    config = official_config()
    config["training"].update(resume_policy=mobile_resume.POLICY, refuse_resume=False,
                              resume_from_checkpoint=str(tmp_path))
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"saved adapter")
    state = {"verified": True, "global_step": 5694, "continuation": {"policy": mobile_resume.POLICY}}
    monkeypatch.setattr(mobile_resume, "verify_continuation", lambda *_args: state)
    numeric = official_evidence(config)
    numeric["adapter_initialization_mode"] = "resumed_checkpoint"
    numeric["zero_adapter_initialization"]["verified_zero_delta"] = False
    numeric["resume_state"] = copy.deepcopy(state)
    numeric["resume_adapter"] = {
        "checkpoint": str(tmp_path), "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "adapter_pair_count": 205, "finite_adapter_pairs": 205, "nonzero_adapter_pairs": 205,
    }
    if tamper == "zero":
        numeric["resume_adapter"]["nonzero_adapter_pairs"] = 0
    elif tamper == "hash":
        numeric["resume_adapter"]["adapter_sha256"] = "a" * 64
    elif tamper == "state":
        numeric["resume_state"]["global_step"] -= 1
    elif tamper == "scope":
        numeric["resume_adapter"]["adapter_pair_count"] = 204
    elif tamper == "nonfinite":
        numeric["qat_on"]["completion_loss"] = float("nan")
    elif tamper == "greedy":
        numeric["greedy_generation"]["qat_on"]["deterministic"] = False
    report = numeric_preflight_provenance(config, numeric)
    assert report["verified"] is (tamper is None)
    assert "zero_adapter_initialization_verified" not in report["checks"]
    assert "resumed_adapter_initialization_verified" in report["checks"]


@pytest.mark.parametrize("consumer", ["merge", "export"])
@pytest.mark.parametrize("tamper", [None, "missing_policy", "config_policy", "false_gate",
                                    "nonfinite", "excessive_loss", "greedy", "zero",
                                    "lora_scope", "frozen_scope", "qparams"])
def test_real_merge_and_export_provenance_enforce_v2_policy_and_existing_gates(
    tmp_path, monkeypatch, consumer, tamper,
):
    # Reuse the existing bounded 205/70 provenance fixture, not a model export.
    import yaml
    from test_mobile_srq_provenance import _export_fixture, exporter, merge_lora

    for module in (exporter, merge_lora):
        monkeypatch.setattr(module, "_golden_selection_binding", lambda *_a: {"verified": True})
    monkeypatch.setattr(merge_lora, "_portable_launcher_contract_matches", lambda *_a, **_kw: True)
    adapter, config_path, qparams, metadata = _export_fixture(tmp_path, strict=True)
    config = official_config()
    metadata["numeric_preflight"] = official_evidence(config)
    numeric = metadata["numeric_preflight"]
    if tamper == "missing_policy":
        numeric.pop("policy")
    elif tamper == "config_policy":
        config["preflight"].pop("numeric_policy")
    elif tamper == "false_gate":
        numeric["mandatory_checks"]["finite_logits"] = False
    elif tamper == "nonfinite":
        numeric["qat_on"]["completion_loss"] = float("inf")
    elif tamper == "excessive_loss":
        numeric["qat_on"]["completion_loss"] = 100
    elif tamper == "greedy":
        numeric["greedy_generation"]["qat_on"]["generated_token_ids"][1][0][0] = 999
    elif tamper == "zero":
        numeric["zero_adapter_initialization"]["verified_zero_delta"] = False
    elif tamper == "lora_scope":
        metadata["qat"]["wrapped_effective_lora_count"] = 204
    elif tamper == "frozen_scope":
        metadata["qat"]["frozen_activation_bindings"].pop("frozen_0")
    elif tamper == "qparams":
        metadata["qat"]["retained_qparams_bindings"]["module_0"]["input_activation_scale"] = 9.0
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    metadata["training_config_sha256"] = config_sha
    (adapter / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if consumer == "export":
        report = exporter._best_adapter_provenance_report(
            adapter, config_path, qparams=qparams,
            seed_report={"manifest_sha256": "c" * 64, "transformation_plan_sha256": "d" * 64},
        )
    else:
        report = merge_lora._verify_qat_training_metadata(
            adapter, training_config_sha256=config_sha, training_method="qat_lora_sft",
            mobile_training_seed={"required": False}, training_config=config,
            mobile_qparams={"verified": True, "contract_sha256": "a" * 64,
                            "scale_storage_sha256": "b" * 64, "inventory_sha256": "e" * 64,
                            "tensor_count": 278, "inventory": qparams.inventory},
        )
    assert report["verified"] is (tamper is None), report["checks"]
    if tamper is None and consumer == "merge":
        assert "initial_numeric_parity_passed" not in report["checks"]
        assert "deterministic_greedy_prefix_passed" not in report["checks"]
        assert report["checks"]["numeric_preflight_mandatory_gates_passed"] is True


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_forward_gate_checks_all_logits_not_only_completion_loss(nonfinite):
    from types import SimpleNamespace

    import torch

    class Tokenizer:
        pad_token_id = 0

        def pad(self, features, **kwargs):
            return {key: torch.tensor([item[key] for item in features])
                    for key in ("input_ids", "attention_mask")}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, **kwargs):
            logits = torch.zeros((*input_ids.shape, 8))
            # Even unused final-position logits must be finite.
            logits[0, -1, 0] = nonfinite
            return SimpleNamespace(logits=logits)

    model = Model()
    with pytest.raises(RuntimeError, match="non-finite logits"):
        sft._run_forward_numeric_gate(
            model=model, split=[{"input_ids": [1, 2, 3], "labels": [-100, 2, 3],
                                 "attention_mask": [1, 1, 1]}],
            tokenizer=Tokenizer(), input_vocab_size=8, label_vocab_size=8,
            max_position_embeddings=32, max_rows=1,
        )
    assert model.training is True
