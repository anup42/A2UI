"""Opt-in objective evidence composes with the unchanged QAT export gates."""
import copy
import json

import pytest
import yaml
from ir_training.train.qat_grpo_contract import (
    file_identity,
    verify_qat_grpo_provenance,
)
from ir_training.train.qat_objectives import implementation_identity
from test_qat_grpo_export import _fixture
from test_qat_posttrain_data import CharacterTokenizer, build_rehearsal_bank, row


def rehearsal_fixture(tmp_path):
    config, metadata, _, config_path = _fixture(tmp_path)
    config["grpo"].update(objective="grpo_sft", rehearsal={"weight": .1},
                          max_prompt_length=1000, max_completion_length=2000)
    config["training"]["max_seq_length"] = 3000
    config["model"]["max_context_tokens"] = 4000
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    metadata["training_config_sha256"] = file_identity(config_path)["sha256"]
    preflight_path = tmp_path / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["training_config_sha256"] = metadata["training_config_sha256"]
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    _, manifest = build_rehearsal_bank([row()], CharacterTokenizer(),
        max_prompt_length=1000, max_completion_length=2000, max_sequence_length=3000)
    path = tmp_path / "rehearsal.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    metadata["grpo"].update(algorithm="grpo_sft", objective="grpo_sft",
        objective_implementation=implementation_identity("grpo_sft"), preflight=file_identity(preflight_path),
        rehearsal={"weight": .1, "policy": "completion_ce_per_sample_v1", "admission": file_identity(path)})
    return config, metadata, path


def test_rehearsal_export_reuses_existing_grpo_and_numeric_gates(tmp_path):
    config, metadata, _ = rehearsal_fixture(tmp_path)
    report = verify_qat_grpo_provenance(config, metadata)
    assert report["verified"], report
    for key in ("qat_forward_probes_passed", "qat_backward_probe_passed", "source_sft_lineage_verified",
                "selected_step_health_window_passed", "rehearsal_admission_bound"):
        assert report["checks"][key]


@pytest.mark.parametrize("tamper", ["objective", "implementation", "weight", "policy", "artifact_bytes",
    "kind", "rows", "limits", "metric", "backward", "health"])
def test_rehearsal_cannot_bypass_objective_fidelity_or_existing_export_gates(tmp_path, tamper):
    config, metadata, path = rehearsal_fixture(tmp_path)
    if tamper == "objective":
        metadata["grpo"]["algorithm"] = "grpo"
    elif tamper == "implementation":
        metadata["grpo"]["objective_implementation"] = {}
    elif tamper in {"weight", "policy"}:
        metadata["grpo"]["rehearsal"][tamper] = 0
    elif tamper == "artifact_bytes":
        path.write_text("{}", encoding="utf-8")
    elif tamper == "backward":
        metadata["numeric_preflight"]["backward"]["gradient_l2_norm"] = 0
    elif tamper == "health":
        metadata["grpo"]["health"]["passed_window"] = False
    else:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        field, value = {"kind": ("kind", "qat_dpo"), "rows": ("rows", 0),
            "limits": ("token_limits", {"max_prompt_length": 1001, "max_completion_length": 2000,
                                        "max_sequence_length": 3000}),
            "metric": ("metric_fingerprint", "0" * 64)}[tamper]
        manifest[field] = value
        path.write_text(json.dumps(manifest), encoding="utf-8")
        metadata["grpo"]["rehearsal"]["admission"] = file_identity(path)
    assert not verify_qat_grpo_provenance(config, metadata)["verified"]


def test_plain_grpo_does_not_require_rehearsal_evidence(tmp_path):
    config, metadata, _, _ = _fixture(tmp_path)
    original = copy.deepcopy(config)
    assert verify_qat_grpo_provenance(config, metadata)["verified"]
    assert config == original
