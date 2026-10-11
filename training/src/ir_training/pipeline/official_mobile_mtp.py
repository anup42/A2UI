"""Opt-in drafter phase; never changes the retained-mobile target recipe."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from ir_training.common.config import load_yaml, training_root
from ir_training.pipeline.golden_training import sha256

PURPOSE = "official_mobile_mtp_training_v1"
OPTION_DEFAULTS = {"train_mtp": False, "mtp_epochs": 1, "mtp_steps": None,
                   "mtp_learning_rate": 1e-5}
EXPORT_GATES = frozenset({
    "training_scope_supported", "training_provenance_verified", "complete_23_weight_mapping",
    "converter_inventory_count", "converter_quantization_layout", "converter_constants_transferred_exactly",
    "non_mapped_state_byte_exact", "official_graph_structure", "official_quantization_layout",
    "official_execution_topology_ignoring_buffer_indices", "official_layout_ignoring_buffer_indices",
    "execution_contract_complete", "official_execution_contract",
    "official_execution_contract_ignoring_buffer_indices", "section_size_unchanged",
    "package_outside_mtp_byte_exact", "fine_tuned_target_section_byte_exact",
})


def normalize_options(values: dict) -> None:
    """Strip disabled defaults so historical plans/config hashes are unchanged."""
    if type(values["train_mtp"]) is not bool:
        raise ValueError("train_mtp must be a boolean")
    if not values["train_mtp"]:
        if any(values[key] != default for key, default in OPTION_DEFAULTS.items()):
            raise ValueError("MTP options require --train-mtp")
        for key in OPTION_DEFAULTS:
            values.pop(key)
        return
    for name in ("mtp_epochs", "mtp_steps"):
        value = values[name]
        if value is None and name == "mtp_steps":
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    rate = values["mtp_learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("mtp_learning_rate must be positive and finite")


def extend_plan(plan: dict) -> dict:
    output = Path(plan["options"]["output_dir"])
    paths = plan["paths"]
    paths.update({key: str(output / relative) for key, relative in {
        "mtp_config": "configs/mtp_training.yaml",
        "mtp_training_dir": "mtp_training",
        "mtp_checkpoint": "mtp_training/best_checkpoint",
        "mtp_training_report": "mtp_training/mtp_drafter_training_report.json",
        "mtp_export_dir": "retained_scale_export/drafter",
        "mtp_export_report": "retained_scale_export/drafter/mtp_checkpoint_official_topology_report.json",
        "target_litertlm": "retained_scale_export/target_with_official_mtp.litertlm",
        "mtp_android_report": "android_gpu/mtp_on/android_litertlm_gpu_parity_report.json",
    }.items()})
    plan["stages"].insert(plan["stages"].index("export"), "mtp_training")
    plan["mtp"] = {
        "training": True, "inference": True, "official_section_preserved_but_unused": False,
        "phase": "separate_after_target_selection_and_merge", "target_frozen": True,
        "assistant_trainable_matrices": 23, "private_google_recipe_recovered": False,
        "optimizer_resume_supported": False, "training_gpus": 1,
        "native_acceptance_quality_and_speed_verified": False,
    }
    plan["format"]["mtp_layout_policy"] = "replace only 23 assistant matrices and scales; preserve target and other package bytes"
    plan["speed_parity"] = "unverified until both target-only and MTP-on Android acceptance/throughput gates pass"
    plan["native_quality_validation"] = "trained-MTP Golden/Bixby Android evaluator not yet supported; no native quality promotion"
    return plan


def drafter_config(plan: dict) -> dict:
    if plan.get("mtp", {}).get("training") is not True:
        raise ValueError("Drafter configuration requires --train-mtp")
    values, paths = plan["options"], plan["paths"]
    config = copy.deepcopy(load_yaml(training_root() / "configs/models/gemma4_e2b_mtp_drafter_qat.yaml"))
    target = load_yaml(Path(paths["config"]))
    config["run"].update(id=Path(values["output_dir"]).name, purpose=PURPOSE,
                         dataset_dir=paths["prepared"], output_dir=paths["mtp_training_dir"])
    config["target"].update(model_id_or_path=paths["merged"], tokenizer_id_or_path=values["model_dir"],
                            training_config=paths["config"], adapter_checkpoint=paths["best_checkpoint"])
    config["assistant"]["best_checkpoint_dir"] = paths["mtp_checkpoint"]
    config["training"].update(epochs=values["mtp_epochs"], max_steps=values["mtp_steps"] or 0,
        learning_rate=values["mtp_learning_rate"], max_seq_length=values["max_seq_length"],
        seed=values["seed"], require_cuda=True, overflow_policy="error",
        chat_template_kwargs=target["model"].get("chat_template_kwargs") or {},
        tensorboard_root=values["tensorboard_root"], tensorboard_subdir="mtp_training")
    return config


def configure_export(config: dict, plan: dict) -> None:
    paths = plan["paths"]
    config["pipeline"]["mtp"].update(enabled=True, train_assistant=True, weight_source="trained",
        # The target intermediate retains the official drafter template. The
        # existing MTP exporter then patches that template into the final file.
        preserve_official_section=True, training_config=paths["mtp_config"],
        trained_checkpoint=paths["mtp_checkpoint"], target_intermediate_litertlm=paths["target_litertlm"],
        exact_topology_output_dir=paths["mtp_export_dir"])


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")  # noqa: TRY004 -- invalid file content
    return value


def _path(value, expected: str | Path) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).resolve() == Path(expected).resolve()


def verify_mtp_training(plan: dict) -> list[Path]:
    from ir_training.train.mtp_drafter import (
        _checkpoint_manifest,
        official_mobile_source_bindings,
    )

    paths = plan["paths"]
    config_path = Path(paths["mtp_config"])
    config = load_yaml(config_path)
    if config != drafter_config(plan):
        raise ValueError("MTP config differs from the opted-in plan")
    report_path = Path(paths["mtp_training_report"])
    report = _json(report_path)
    checkpoint = Path(paths["mtp_checkpoint"])
    metadata_path = checkpoint / "mtp_drafter_training_metadata.json"
    metadata = _json(metadata_path)
    bindings = official_mobile_source_bindings(config)
    if (report.get("training_executed") is not True or report.get("ready") is not True
            or type(report.get("global_steps")) is not int or report["global_steps"] <= 0
            or not isinstance(report.get("best_selection_loss"), (int, float))
            or not math.isfinite(report["best_selection_loss"])
            or metadata.get("training_config_sha256") != sha256(config_path)
            or metadata.get("role") != "best" or metadata.get("target_frozen") is not True
            or not _path(metadata.get("target_model_id_or_path"), paths["merged"])):
        raise ValueError("MTP training lacks completed, finite, selected-target provenance")
    for evidence in (report, metadata):
        qat = evidence.get("target_qat") or {}
        if (evidence.get("source_bindings") != bindings or not bindings
                or qat.get("enabled") is not True or qat.get("true_fake_quant") is not True
                or qat.get("wrapped_effective_lora_count") != 205
                or qat.get("frozen_activation_module_count") != 70):
            raise ValueError("MTP conditioning target/data binding or retained QAT coverage differs")
    # The existing export verifier checks the exact 23-matrix scope, QAT numeric
    # contract and every checkpoint file. Bind all files here before that stage.
    files = [item for item in sorted(checkpoint.iterdir()) if item.is_file()]
    if not any(item.name.startswith("model") and item.suffix == ".safetensors" for item in files):
        raise ValueError("MTP best checkpoint has no weights")
    declared = metadata.get("checkpoint_files")
    actual = _checkpoint_manifest(checkpoint)
    if declared != actual:
        raise ValueError("MTP checkpoint files differ from their training manifest")
    return [config_path, report_path, *files]


def verify_mtp_export(plan: dict) -> list[Path]:
    verify_mtp_training(plan)
    paths = plan["paths"]
    target = _json(Path(paths["export_report"]))
    if (target.get("passed") is not True or target.get("executed") is not True
            or not _path(target.get("output_litertlm"), paths["target_litertlm"])
            or target.get("output_sha256") != sha256(Path(paths["target_litertlm"]))):
        raise ValueError("Retained target report does not bind the intermediate MTP package")
    report = _json(Path(paths["mtp_export_report"]))
    gates, boundary = report.get("gates") or {}, report.get("package_boundary") or {}
    if (report.get("final_artifact_gate_pass") is not True or report.get("assistant_trained") is not True
            or report.get("assistant_weight_source") != "trained_checkpoint"
            or not EXPORT_GATES.issubset(gates) or not all(value is True for value in gates.values())
            or report.get("official_artifact_sha256_expected") != plan["format"]["official_sha256"]
            or report.get("official_artifact_sha256_observed") != plan["format"]["official_sha256"]
            or boundary.get("output_sha256") != sha256(Path(paths["litertlm"]))
            or boundary.get("bytes_outside_mtp_verified_unchanged") is not True
            or (report.get("training_provenance") or {}).get("verified") is not True):
        raise ValueError("Trained MTP export failed graph/layout/target-preservation or live file gates")
    for value, expected in ((report.get("package_input"), paths["target_litertlm"]),
                            (report.get("package_output"), paths["litertlm"]),
                            (boundary.get("output"), paths["litertlm"]),
                            (report.get("official_artifact"), plan["options"]["official_litertlm"]),
                            (report.get("assistant_checkpoint"), paths["mtp_checkpoint"]),
                            (report.get("assistant_training_config"), paths["mtp_config"])):
        if not _path(value, expected):
            raise ValueError("MTP exporter report belongs to another input/checkpoint/output")
    metadata = _json(Path(paths["mtp_checkpoint"]) / "mtp_drafter_training_metadata.json")
    if report["training_provenance"].get("metadata") != metadata:
        raise ValueError("MTP export provenance differs from the trained checkpoint")
    return [Path(paths[key]) for key in ("export_report", "target_litertlm", "mtp_export_report", "litertlm")]
