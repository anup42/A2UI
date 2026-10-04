from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from ir_training.export.merge_lora import (
    _adapter_file_records,
    _verify_qat_training_metadata,
)
from ir_training.train.grpo_runtime import GRPOHealthMonitor, HealthThresholds
from ir_training.train.qat_grpo_contract import (
    file_identity,
    validate_qat_grpo_config,
    verify_qat_grpo_provenance,
    verify_sft_adapter_lineage,
)
from ir_training.train.qat_grpo_reward import QAT_GRPO_REWARD_VERSION


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path):
    source = tmp_path / "sft"
    source.mkdir()
    _write(source / "adapter_config.json", {"peft_type": "LORA"})
    (source / "adapter_model.safetensors").write_bytes(b"bounded fixture weights")
    config = {
        "run": {"purpose": "sft"},
        "model": {"model_id": "google/gemma-3-270m-it"},
        "training": {"method": "qat_lora_sft"},
        "lora": {"r": 16, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj"], "modules_to_save": []},
        "qat": {"enabled": True, "weight_bits": 8, "activation_bits": 32, "quantizer": "ste_ai_edge",
                "quantize_embeddings": True, "only_base_layers": True, "effective_merged_weight": True},
        "golden_eval": {"metric_for_best_model": "generation_reward_v5_4_avg"},
        "preflight": {"max_initial_completion_loss": 15.0, "greedy_probe_rows": 1,
                      "greedy_probe_new_tokens": 8, "min_greedy_tokens": 8},
    }
    source_config = tmp_path / "sft.yaml"
    source_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_meta = {
        "training_metadata_version": 4, "training_config_sha256": file_identity(source_config)["sha256"],
        "training": config["training"], "lora": config["lora"], "git_commit": "a" * 40,
        "qat": {"enabled": True, "effective_merged_weight_qat_enabled": True,
                "wrapped_effective_lora_count": 1, "uncovered_lora_adapter_linear_names": [],
                "spec": {"effective_merged_weight": True}},
        "adapter_checkpoints": [{"role": "best_golden", "files": _adapter_file_records(source)}],
    }
    _write(source / "training_metadata.json", source_meta)
    config = copy.deepcopy(config)
    config["run"]["purpose"] = "qat_lora_grpo_v1"
    config["training"]["method"] = "qat_lora_grpo"
    config["grpo"] = {"family": "270m", "sft_checkpoint": str(source), "sft_training_config": str(source_config),
                      "health": {"window_steps": 2}}
    config_path = tmp_path / "grpo.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    initial = verify_sft_adapter_lineage(config)
    assert initial["verified"], initial
    numeric = {"adapter_initialization_mode": "sft_adapter",
               "source_adapter": {"checkpoint": str(source),
                                  "adapter_sha256": file_identity(source / "adapter_model.safetensors")["sha256"],
                                  "adapter_pair_count": 1, "finite_adapter_pairs": 1, "nonzero_adapter_pairs": 1},
               "zero_adapter_initialization": {"required": False, "reason": "continued_from_trained_sft_adapter", "verified_zero_delta": False}}
    forward = {"completion_loss": 1.0, "row_losses": [1.0], "logits_finite": True,
               "rows_checked": 1, "completion_tokens": 1, "top1_probe_ids": [1]}
    greedy = {"rows_checked": 1, "min_new_tokens": 8, "max_new_tokens": 8,
              "repeats": 2, "generated_token_ids": [[[1] * 8], [[1] * 8]],
              "generated_token_counts": [[8], [8]], "deterministic": True}
    numeric.update(passed=True, qat_enabled=True, baseline=forward, qat_on=copy.deepcopy(forward),
                   greedy_generation={"passed": True, "qat_enabled": True, "baseline": copy.deepcopy(greedy), "qat_on": greedy},
                   backward={"status": "passed", "kind": "completion_log_likelihood_backward", "loss": 1.0,
                             "finite_parameters_and_gradients": True, "nonzero_gradient_tensors": 1,
                             "trainable_tensors": 2, "optimizer_steps": 0,
                             "gradient_l2_norm": 0.1, "gradient_max_abs": 0.1})
    preflight_path = tmp_path / "preflight.json"
    _write(preflight_path, {"schema_version": 1, "training_config_sha256": file_identity(config_path)["sha256"],
                            "source_lineage": initial, "numeric_preflight": numeric})
    thresholds = HealthThresholds(window_steps=2)
    monitor = GRPOHealthMonitor(thresholds)
    row = {"frac_reward_zero_std": 0.0, "completions/clipped_ratio": 0.0, "grad_norm": 0.1,
           "sampled_parameter_max_delta": 0.01, "nonfinite_parameters_or_gradients": 0, "loss": 0.2}
    monitor.observe(1, row)
    state = monitor.observe(2, row)
    health = {"passed_window": True, "optimizer_steps": 2, "last_step": 2, "window_updates": 2,
              "window_metrics": list(monitor.rows), "last_metrics": state["metrics"],
              "thresholds": asdict(thresholds), "aggregates": state["aggregates"]}
    metadata = copy.deepcopy(source_meta)
    metadata.update(training=copy.deepcopy(config["training"]), training_config_sha256=file_identity(config_path)["sha256"],
                    checkpoint_step=2, checkpoint_role="best_golden", golden_eval=config["golden_eval"],
                    best_golden_eval={"metric": "generation_reward_v5_4_avg", "metric_value": 75.0, "step": 2},
                    numeric_preflight=numeric,
                    grpo={"schema_version": 1, "algorithm": "grpo", "family": "270m",
                          "reward": {"version": QAT_GRPO_REWARD_VERSION,
                                     "config": file_identity(ROOT.parent / "dataset/configs/genui_metric_v5_4.yaml")},
                          "source_lineage": verify_sft_adapter_lineage(config, numeric),
                          "preflight": file_identity(preflight_path), "health": health})
    candidate = tmp_path / "grpo_best"
    candidate.mkdir()
    (candidate / "adapter_model.safetensors").write_bytes(b"GRPO trained fixture weights")
    _write(candidate / "adapter_config.json", {"peft_type": "LORA"})
    metadata["adapter_checkpoints"] = [{"role": "best_golden", "files": _adapter_file_records(candidate)}]
    _write(candidate / "training_metadata.json", metadata)
    return config, metadata, candidate, config_path


def test_genuine_grpo_metadata_reuses_qat_merge_provenance(tmp_path):
    config, metadata, candidate, config_path = _fixture(tmp_path)
    report = verify_qat_grpo_provenance(config, metadata)
    assert report["verified"], report
    report = _verify_qat_training_metadata(
        candidate, training_config_sha256=file_identity(config_path)["sha256"],
        training_method="qat_lora_grpo", mobile_training_seed={"required": False}, training_config=config)
    assert report["verified"], report
    assert report["checks"]["qat_grpo_provenance_verified"]


@pytest.mark.parametrize("tamper", ["source_weights", "source_config", "source_method", "zero_init", "scope",
                                     "preflight_bytes", "preflight_hash", "zero_gradients", "zero_updates",
                                     "flat_rewards", "clipping", "nonfinite", "step", "thresholds", "family",
                                     "backward_zero", "backward_nonfinite", "missing_forward", "greedy_changed",
                                     "reward_version", "reward_config"])
def test_grpo_export_rejects_stale_or_ineffective_training_evidence(tmp_path, tamper):
    config, metadata, _, _ = _fixture(tmp_path)
    source = Path(config["grpo"]["sft_checkpoint"])
    if tamper == "source_weights":
        (source / "adapter_model.safetensors").write_bytes(b"different source")
    elif tamper == "source_config":
        config["qat"]["weight_bits"] = 4
    elif tamper == "source_method":
        config["grpo"]["sft_training_config"] = str(tmp_path / "grpo.yaml")
    elif tamper == "zero_init":
        metadata["numeric_preflight"]["source_adapter"]["nonzero_adapter_pairs"] = 0
    elif tamper == "scope":
        metadata["numeric_preflight"]["source_adapter"]["adapter_pair_count"] = 2
    elif tamper == "preflight_bytes":
        _write(tmp_path / "preflight.json", {})
    elif tamper == "preflight_hash":
        metadata["grpo"]["preflight"]["sha256"] = "0" * 64
    elif tamper == "step":
        metadata["checkpoint_step"] = 1
    elif tamper == "thresholds":
        metadata["grpo"]["health"]["thresholds"]["window_steps"] = 1
    elif tamper == "family":
        config["grpo"]["family"] = "e2b"
    elif tamper == "reward_version":
        metadata["grpo"]["reward"]["version"] = "arbitrary_reward"
    elif tamper == "reward_config":
        metadata["grpo"]["reward"]["config"]["sha256"] = "0" * 64
    elif tamper == "backward_zero":
        metadata["numeric_preflight"]["backward"]["gradient_l2_norm"] = 0.0
    elif tamper == "backward_nonfinite":
        metadata["numeric_preflight"]["backward"]["loss"] = float("nan")
    elif tamper == "missing_forward":
        metadata["numeric_preflight"].pop("qat_on")
    elif tamper == "greedy_changed":
        metadata["numeric_preflight"]["greedy_generation"]["qat_on"]["generated_token_ids"][1][0][0] = 2
    else:
        field, value = {"zero_gradients": ("grad_norm", 0.0), "zero_updates": ("sampled_parameter_max_delta", 0.0),
                        "flat_rewards": ("frac_reward_zero_std", 1.0), "clipping": ("completions/clipped_ratio", 1.0),
                        "nonfinite": ("loss", float("nan"))}[tamper]
        for row in metadata["grpo"]["health"]["window_metrics"]:
            row[field] = value
    assert verify_qat_grpo_provenance(config, metadata)["verified"] is False


def test_merger_rejects_relabelled_sft_without_grpo_provenance(tmp_path):
    config, metadata, candidate, config_path = _fixture(tmp_path)
    metadata.pop("grpo")
    _write(candidate / "training_metadata.json", metadata)
    report = _verify_qat_training_metadata(
        candidate, training_config_sha256=file_identity(config_path)["sha256"],
        training_method="qat_lora_grpo", mobile_training_seed={"required": False}, training_config=config)
    assert not report["verified"]


def test_270m_topology_keeps_official_precision_and_model_contract(tmp_path):
    import build_checkpoint_official_topology as exporter
    config, _, _, config_path = _fixture(tmp_path)
    report = exporter._training_scope_report(config_path, family="gemma3_270m", official_base_model_id="google/gemma-3-270m-it")
    assert report["supported_projection_only_transplant"], report
    assert "qat_lora_sft" not in report["checks"]
    assert report["checks"]["qat_lora_grpo_config"]
    config["qat"]["activation_bits"] = 8
    with pytest.raises(ValueError, match="270M GRPO"):
        validate_qat_grpo_config(config)


@pytest.mark.parametrize("consumer", ["merge", "export"])
@pytest.mark.parametrize("tamper", [None, "lora_scope", "frozen_scope", "scale", "grpo_provenance"])
def test_e2b_grpo_preserves_common_205_70_and_scale_gates(tmp_path, monkeypatch, consumer, tamper):
    # Isolate the already-tested new GRPO evidence verifier to prove its branch
    # cannot waive any of the existing retained-scale checks in either consumer.
    from ir_training.qat.numeric_preflight import RETAINED_MOBILE_POLICY
    from ir_training.train import qat_grpo_contract as contract
    from test_mobile_srq_provenance import _export_fixture, exporter, merge_lora

    adapter, config_path, qparams, metadata = _export_fixture(tmp_path, strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("training", {})["method"] = "qat_lora_grpo"
    metadata.setdefault("training", {})["method"] = "qat_lora_grpo"
    metadata["numeric_preflight"]["zero_adapter_initialization"] = {
        "required": False, "reason": "continued_from_trained_sft_adapter", "verified_zero_delta": False}
    monkeypatch.setattr(contract, "verify_qat_grpo_provenance", lambda *_a: {"verified": tamper != "grpo_provenance"})
    for module in (merge_lora, exporter):
        monkeypatch.setattr(module, "numeric_preflight_provenance", lambda *_a: {"verified": True, "policy": RETAINED_MOBILE_POLICY})
        monkeypatch.setattr(module, "_golden_selection_binding", lambda *_a: {"verified": True})
    if tamper == "lora_scope":
        metadata["qat"]["wrapped_effective_lora_count"] = 204
    elif tamper == "frozen_scope":
        metadata["qat"]["frozen_activation_bindings"].pop("frozen_0")
    elif tamper == "scale":
        metadata["qat"]["retained_qparams_bindings"]["module_0"]["input_activation_scale"] = 9.0
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    digest = file_identity(config_path)["sha256"]
    metadata["training_config_sha256"] = digest
    _write(adapter / "training_metadata.json", metadata)
    if consumer == "merge":
        result = merge_lora._verify_qat_training_metadata(
            adapter, training_config_sha256=digest, training_method="qat_lora_grpo",
            mobile_training_seed={"required": False}, training_config=config,
            mobile_qparams={"verified": True, "contract_sha256": "a" * 64, "scale_storage_sha256": "b" * 64,
                            "inventory_sha256": "e" * 64, "tensor_count": 278, "inventory": qparams.inventory})
    else:
        result = exporter._best_adapter_provenance_report(
            adapter, config_path, qparams=qparams,
            seed_report={"manifest_sha256": "c" * 64, "transformation_plan_sha256": "d" * 64})
    assert result["verified"] is (tamper is None), result
