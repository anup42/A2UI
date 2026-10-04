"""Synthetic-byte native bindings; no model weights or Android are executed."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from ir_training.eval import android_native_quality as native
from ir_training.eval.qat_grpo_native import (
    _270M_GATES,
    _E2B_GATES,
    inspect_qat_grpo_run,
)
from ir_training.export.merge_lora import _adapter_file_records
from ir_training.pipeline.official_mobile import NO_OP_CHECKS, SELECTOR
from ir_training.qat.mobile_training_seed import OFFICIAL_LITERTLM_SHA256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path, monkeypatch, family):
    run = tmp_path / "copied-run"
    run.mkdir()
    origin = "/training/grpo-run"
    names = {"prepared": "prepared", "config": "configs/grpo.yaml", "best_checkpoint": "training/best_golden_checkpoint",
             "merged": "merged_best_hf", "export_report": "export/report.json", "litertlm": "export/model.litertlm",
             "no_op_export_report": "pretraining_noop_export.json"}
    paths = {key: run / value for key, value in names.items()}
    old = lambda path: origin + "/" + path.relative_to(run).as_posix()
    source_name = "gemma4_e2b_mobile_seed_ir_qat_sft.yaml" if family == "e2b" else "gemma3_270m_ir_qat_sft.yaml"
    config = yaml.safe_load((ROOT / "configs/models" / source_name).read_text(encoding="utf-8"))
    config["run"].update(purpose="qat_lora_grpo_v1", dataset_dir=old(paths["prepared"]))
    config["training"].update(method="qat_lora_grpo", max_seq_length=4096)
    config["model"]["max_output_tokens"] = 2048
    config["grpo"] = {"family": family, "sft_checkpoint": "/external/sft-adapter", "sft_training_config": "/external/sft.yaml"}
    config["golden_eval"].update(dataset_dir=old(paths["prepared"]), split="golden32", max_rows=32, required_rows=32,
                               require_exact_rows=True, require_unique_rows=True,
                               split_path=old(paths["prepared"] / "golden32.jsonl"), metric_for_best_model=SELECTOR,
                               metric_version="v5_4", max_input_tokens=5120)
    write(paths["config"], config)
    config_sha = native.file_sha256(paths["config"])
    write(paths["config"].parent / "preparation_report.json", {"training_config_sha256": config_sha})
    write(paths["prepared"] / "prompt_scaffolds.json", [{"scaffold": "synthetic"}])
    write(paths["prepared"] / "manifest.json", {
        "validation": {"strict_express": True, "wire_schema": True, "semantic_roundtrip": True, "root_reachability": 1.0},
        "scaffold_count": 1, "prompt_scaffolds_sha256": native.file_sha256(paths["prepared"] / "prompt_scaffolds.json")})
    write(paths["best_checkpoint"] / "adapter_config.json", {"peft_type": "LORA"})
    (paths["best_checkpoint"] / "adapter_model.safetensors").write_bytes(b"synthetic adapter fixture")
    write(paths["best_checkpoint"] / "tokenizer.json", {"synthetic": True})
    adapter_files = _adapter_file_records(paths["best_checkpoint"])
    metadata = {"checkpoint_role": "best_golden", "checkpoint_step": 20, "training_config_sha256": config_sha,
                "training": {"method": "qat_lora_grpo"}, "golden_eval": config["golden_eval"],
                "best_golden_eval": {"metric": SELECTOR, "metric_value": 80.0, "step": 20},
                "adapter_checkpoints": [{"role": "best_golden", "files": adapter_files}]}
    write(paths["best_checkpoint"] / "training_metadata.json", metadata)
    merge = {"training_method": "qat_lora_grpo", "training_config_sha256": config_sha,
             "adapter_dir": old(paths["best_checkpoint"]), "merged_model_dir": old(paths["merged"]), "adapter_files": adapter_files,
             "training_run_metadata": {"path": old(paths["best_checkpoint"] / "training_metadata.json"),
                                       "sha256": native.file_sha256(paths["best_checkpoint"] / "training_metadata.json"),
                                       "verified": True, "checks": {"qat_grpo_provenance_verified": True},
                                       "qat_grpo_provenance": {"verified": True}}}
    write(paths["merged"] / "qat_mtp_merge_metadata.json", merge)
    (paths["merged"] / "model.safetensors").write_bytes(b"synthetic merged fixture")
    paths["litertlm"].parent.mkdir(parents=True)
    paths["litertlm"].write_bytes(b"synthetic official layout fixture")
    pinned = (OFFICIAL_LITERTLM_SHA256 if family == "e2b" else
              yaml.safe_load((ROOT / "configs/pipelines/gemma3_270m_qat_litertlm.yaml").read_text())["pipeline"]["exact_topology"]["official_artifact_sha256"])
    report = {"checkpoint": old(paths["merged"]), "merge_provenance": {"verified": True, "metadata": merge}}
    if family == "e2b":
        report.update(mode="retained_scale_qat_compatible_weights_v1", passed=True, executed=True, plan_passed=True,
                      official_artifact_sha256=pinned, output_sha256=native.file_sha256(paths["litertlm"]),
                      output_litertlm=old(paths["litertlm"]), adapter_checkpoint=old(paths["best_checkpoint"]),
                      adapter_identity={"verified": True, "files": adapter_files,
                                        "training_metadata_sha256": native.file_sha256(paths["best_checkpoint"] / "training_metadata.json")},
                      resolved_training_config_identity={"verified": True, "path": old(paths["config"]), "sha256": config_sha},
                      gates={key: True for key in _E2B_GATES})
        write(paths["no_op_export_report"], {"passed": True, "gate_status": "PASSED", "mode": "retained_scale_pretraining_noop_v1",
              "checks": {key: True for key in NO_OP_CHECKS},
              "resolved_training_config_identity": {"verified": True, "path": old(paths["config"]), "sha256": config_sha}})
    else:
        report.update(family="gemma3_270m", model_type="TF_LITE_PREFILL_DECODE", official_artifact_sha256_expected=pinned,
                      official_artifact_sha256_observed=pinned, final_artifact_gate_pass=True, exact_official_graph_ops_layout_match=True,
                      official_inventory_count=127, mapping_count=127, gates={key: True for key in _270M_GATES},
                      package_boundary={"output": old(paths["litertlm"]), "output_sha256": native.file_sha256(paths["litertlm"]),
                                        "bytes_outside_selected_section_verified_unchanged": True})
    write(paths["export_report"], report)
    kinds = {"golden32": "explicit_repeated_case", "golden35": "fixed_strict_subset", "bixby50": "source_only_holdout"}
    def prepared_contract(dataset, split, **kwargs):
        assert kwargs["max_sequence"] == 4096 and kwargs["max_prompt"] == 5120
        assert kwargs["expected_kind"] == kinds[split.stem]
        return {"split_path": str(split.resolve()), "benchmark_kind": kinds[split.stem],
                "reference_available": split.stem != "bixby50", "split_sha256": native.file_sha256(split)}
    monkeypatch.setattr(native, "verify_golden_preparation", prepared_contract)
    for name, count in native.COHORTS.items():
        rows = [{"id": f"{name}-{index}", "response_text": f"source {name} {index}", "reference_available": name != "bixby50"} for index in range(count)]
        split, directory = paths["prepared"] / f"{name}.jsonl", run / "evaluations" / f"best_{name}"
        write_rows(split, rows)
        saved = prepared_contract(paths["prepared"], split, max_sequence=4096, max_prompt=5120, expected_kind=kinds[name])
        saved["split_path"] = old(split)
        aggregate = {SELECTOR: 80.0, "generation_reward_v5_4_avg": 80.0}
        write(directory / "aggregate_metrics.json", aggregate)
        write_rows(directory / "predictions.jsonl", rows)
        write_rows(directory / "scored_predictions.jsonl", rows)
        write(directory / "evaluation_result.json", {"row_count": count, "qat_applied": True, "prepared_contract": saved,
              "checkpoint": old(paths["best_checkpoint"]), "checkpoint_kind": "adapter", "aggregate": aggregate,
              "tensorboard_record": f"/tensorboard/external-{name}.json"})
    write(run / "assets_verified.json", {"verified": True})
    stages = ["assets", "prepare", "configure", *(["no_op_export"] if family == "e2b" else []),
              "training", *[f"best_{name}" for name in native.COHORTS], "merge", "export"]
    plan = {"schema_version": 1, "workflow": "qat_lora_grpo_v1",
            "options": {"output_dir": origin, "family": family, "model_dir": "/external/base",
                        "source_safetensors": "/external/mobile.safetensors", "official_litertlm": "/external/official.litertlm",
                        "max_seq_length": 4096, "max_input_tokens": 5120, "max_new_tokens": 2048},
            "paths": {key: old(value) for key, value in paths.items()}, "stages": [*stages, "native_quality"],
            "official_artifact_sha256": pinned, "mtp": {"training": False, "inference": False},
            "selection": {"cohort": "golden32", "metric": SELECTOR, "golden35_used": False, "bixby50_used": False}}
    write(run / "qat_grpo_plan.json", plan)
    digest = native.file_sha256(run / "qat_grpo_plan.json")
    roots = {"assets": [run / "assets_verified.json"], "prepare": list(paths["prepared"].glob("*")),
             "configure": list(paths["config"].parent.glob("*")), "training": list(paths["best_checkpoint"].glob("*")),
             "merge": list(paths["merged"].glob("*")), "export": [paths["export_report"], paths["litertlm"]],
             "no_op_export": [paths["no_op_export_report"]]}
    completed = {}
    for stage in stages:
        files = roots.get(stage) or list((run / "evaluations" / stage).glob("*"))
        receipt = {"stage": stage, "plan_sha256": digest, "files": {old(path): native.file_sha256(path) for path in files}}
        if stage.startswith("best_"):
            receipt["files"][f"/tensorboard/external-{stage[5:]}.json"] = "c" * 64
        if stage == "assets" and family == "e2b":
            receipt["files"]["/external/mobile.safetensors"] = "d" * 64
        write(run / "stage_receipts" / f"{stage}.json", receipt)
        completed[stage] = receipt
    write(run / "qat_grpo_manifest.json", {"workflow": plan["workflow"], "plan": plan,
          "status": "awaiting_native_evaluation", "completed": completed})
    return run, paths


@pytest.mark.parametrize("family", ["e2b", "270m"])
def test_native_binds_both_families_after_cross_host_relocation(tmp_path, monkeypatch, family):
    run, paths = fixture(tmp_path, monkeypatch, family)
    binding = native.inspect_completed_run(run)
    assert binding["paths"]["litertlm"] == paths["litertlm"]
    assert binding["max_input_tokens"] == 5120
    assert len(binding["cohorts"]["bixby50"]["rows"]) == 50
    assert binding["cohorts"]["bixby50"]["prepared_contract"]["reference_available"] is False
    moved = tmp_path / "another-host-copy"
    shutil.copytree(run, moved)
    assert inspect_qat_grpo_run(moved)["paths"]["best_checkpoint"] == moved / "training/best_golden_checkpoint"


@pytest.mark.parametrize("family", ["e2b", "270m"])
@pytest.mark.parametrize("tamper", ["status", "missing_stage", "receipt_plan", "config", "adapter", "tokenizer", "artifact",
                                     "export_gate", "export_family", "hf_checkpoint", "hf_coverage", "hf_qat", "source_only", "external_receipt"])
def test_native_rejects_stale_or_incomplete_bindings(tmp_path, monkeypatch, family, tamper):
    run, paths = fixture(tmp_path, monkeypatch, family)
    manifest_path = run / "qat_grpo_manifest.json"
    state = native.read_json(manifest_path)
    if tamper == "status":
        state["status"] = "running"
        state["active_stage"] = "training"
    elif tamper == "missing_stage":
        state["completed"].pop("merge")
    elif tamper == "receipt_plan":
        state["completed"]["export"]["plan_sha256"] = "0" * 64
    else:
        selected = {"config": paths["config"], "adapter": paths["best_checkpoint"] / "adapter_model.safetensors",
                    "tokenizer": paths["best_checkpoint"] / "tokenizer.json", "artifact": paths["litertlm"]}.get(tamper)
        if selected:
            selected.write_bytes(b"changed")
        elif tamper.startswith("export_"):
            value = native.read_json(paths["export_report"])
            if tamper == "export_gate":
                value["gates"].pop(next(iter(value["gates"])))
            else:
                value["official_artifact_sha256" if family == "e2b" else "family"] = "wrong"
            write(paths["export_report"], value)
            stage = "export"
        elif tamper == "external_receipt":
            state["completed"]["prepare"]["files"]["/external/unapproved"] = "f" * 64
            write(run / "stage_receipts/prepare.json", state["completed"]["prepare"])
        else:
            stage = "best_bixby50" if tamper == "source_only" else "best_golden32"
            path = run / "evaluations" / stage / "evaluation_result.json"
            value = native.read_json(path)
            if tamper == "hf_checkpoint":
                value["checkpoint"] = "/training/grpo-run/training/other"
            elif tamper == "hf_qat":
                value["qat_applied"] = False
            elif tamper == "hf_coverage":
                value["row_count"] = 31
            else:
                value["prepared_contract"]["reference_available"] = True
            write(path, value)
        # Rebind deliberately malformed report bytes so schema guards, not only
        # receipt hash checking, reject the report tampering cases.
        if tamper.startswith(("export_", "hf_")) or tamper == "source_only":
            record = state["completed"][stage]
            for old_path in record["files"]:
                try:
                    current = native.relocate_run_path(old_path, original_root="/training/grpo-run", current_root=run)
                except ValueError:
                    continue
                record["files"][old_path] = native.file_sha256(current)
            write(run / "stage_receipts" / f"{stage}.json", record)
    write(manifest_path, state)
    with pytest.raises((ValueError, FileNotFoundError)):
        inspect_qat_grpo_run(run)


@pytest.mark.parametrize("family", ["e2b", "270m"])
def test_native_accepts_current_native_worker_and_completed_run(tmp_path, monkeypatch, family):
    run, _ = fixture(tmp_path, monkeypatch, family)
    path = run / "qat_grpo_manifest.json"
    state = native.read_json(path)
    state.update(status="running", active_stage="native_quality")
    write(path, state)
    assert inspect_qat_grpo_run(run)["plan"]["options"]["family"] == family
    state.update(status="complete", active_stage=None)
    state["completed"]["native_quality"] = {"stage": "native_quality"}
    write(path, state)
    assert inspect_qat_grpo_run(run)["plan"]["options"]["family"] == family
