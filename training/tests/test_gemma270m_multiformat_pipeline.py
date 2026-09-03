from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.eval.tensorboard_logging import log_evaluation_result
from ir_training.pipeline.gemma270m_multiformat import (
    FORMAT_ORDER,
    Gemma270MMultiformatError,
    _export_format,
    _require_litertlm_runner_ready,
    _require_training_leakage_gate,
    _use_official_q8_for_w8,
    _with_checkpoint_step,
    build_pipeline_plan,
    write_final_scorecard,
)

CONFIG_PATH = (
    ROOT / "configs" / "pipelines" / "gemma3_270m_a2ui_express_multiformat.yaml"
)


def _plan() -> dict:
    return build_pipeline_plan(load_yaml(CONFIG_PATH), config_path=CONFIG_PATH)


class _FakeWriter:
    def __init__(self, **_kwargs):
        pass

    def add_scalar(self, *_args):
        pass

    def add_text(self, *_args):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _precision_graph(lane: str) -> dict:
    histograms = {
        "w32": ({"FLOAT32": 10}, {}, 0),
        "w16": ({"FLOAT16": 8, "FLOAT32": 2}, {}, 0),
        "w8": (
            {"INT8": 8, "FLOAT32": 2},
            {"INT8|scales=8|zero_points=8|quantized_dimension=0": 8},
            8,
        ),
        "w4": (
            {"INT4": 8, "FLOAT32": 2},
            {"INT4|scales=64|zero_points=64|quantized_dimension=0": 8},
            8,
        ),
    }
    tensor_types, layouts, quantized = histograms[lane]
    return {
        "available": True,
        "summary": {
            "tensor_type_histogram": tensor_types,
            "quantization_layout_histogram": layouts,
            "quantized_tensor_count": quantized,
            "constant_tensor_type_histogram": tensor_types,
            "constant_quantization_layout_histogram": layouts,
            "constant_quantized_tensor_count": quantized,
        },
    }


def _attach_evaluation_evidence(
    plan: dict,
    *,
    aggregate_path: Path,
    artifact: Path,
    evaluation_name: str,
    artifact_field: str,
    score: float,
    tensorboard_root: Path,
    package_report: Path | None = None,
) -> None:
    metrics = {"count": 32, "generation_reward_v5_4_avg": score}
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(metrics), encoding="utf-8")
    record = log_evaluation_result(
        tensorboard_root,
        run_id=plan["run_id"],
        evaluation_name=evaluation_name,
        metrics=metrics,
        step=500,
        artifacts={
            artifact_field: artifact,
            "golden_split": Path(plan["golden"]["prepared_split"]),
        },
        metadata={
            "rows": 32,
            "split": plan["golden"]["prepared_split"],
            "metric_version": plan["golden"]["metric_version"],
        },
        source_aggregate_path=aggregate_path,
        writer_factory=_FakeWriter,
    )
    result = {
        "row_count": 32,
        "aggregate": metrics,
        "tensorboard_record": record["record_path"],
        ("checkpoint" if artifact_field == "checkpoint" else "model"): str(
            artifact.resolve()
        ),
    }
    (aggregate_path.parent / "evaluation_result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    if package_report is not None:
        lane = evaluation_name.removeprefix("litertlm_")
        package_report.parent.mkdir(parents=True, exist_ok=True)
        package_report.write_text(
            json.dumps(
                {
                    "path": str(artifact.resolve()),
                    "file_size": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "sections": [{"sha256": "a" * 64}],
                    "weights": {"hashed": True},
                    "graphs": [_precision_graph(lane)],
                }
            ),
            encoding="utf-8",
        )


def test_plan_binds_exact_golden32_and_repository_tensorboard_root(monkeypatch):
    monkeypatch.delenv("A2UI_TENSORBOARD_ROOT", raising=False)
    plan = _plan()

    assert plan["validation"]["config_ok"] is True
    assert plan["validation"]["ok"] is False
    assert plan["litertlm_runner"]["placeholder"] is True
    assert plan["golden"]["required_rows"] == 32
    assert plan["golden"]["source_report"]["rows"] == 32
    assert plan["golden"]["source_report"]["unique_rows"] == 32
    assert plan["golden"]["source_report"]["sha256"] == (
        "596f56a15a63d904423970c9768797ba795ff0528a04da37dddcba280a4f4e17"
    )
    assert plan["golden"]["responses_report"]["sha256"] == (
        "27406fa78d486ded4e9fcb2a78e987aa54d9f2ee8fe5db926874a1b2ff827089"
    )
    assert Path(plan["training"]["tensorboard_root"]) == ROOT.parent / "tensorboard"
    assert Path(plan["training"]["tensorboard_run_dir"]) == (
        ROOT.parent
        / "tensorboard"
        / "gemma3_270m_a2ui_express_multiformat_001"
        / "training"
    )
    assert plan["mtp"] == {
        "enabled": False,
        "status": "not_applicable_for_gemma3_270m",
    }


def test_plan_honors_mlp_tensorboard_mount_without_rewriting_yaml(
    monkeypatch, tmp_path: Path
):
    mounted = tmp_path / "mounted-tensorboard"
    monkeypatch.setenv("A2UI_TENSORBOARD_ROOT", str(mounted))

    plan = _plan()

    assert plan["validation"]["config_ok"] is True
    assert Path(plan["training"]["tensorboard_root"]) == mounted.resolve()
    assert Path(plan["training"]["tensorboard_run_dir"]) == (
        mounted.resolve()
        / "gemma3_270m_a2ui_express_multiformat_001"
        / "training"
    )


def test_run_id_isolates_every_mutable_path_and_rejects_unsafe_values():
    first = build_pipeline_plan(
        load_yaml(CONFIG_PATH), config_path=CONFIG_PATH, run_id_override="trial_001"
    )
    second = build_pipeline_plan(
        load_yaml(CONFIG_PATH), config_path=CONFIG_PATH, run_id_override="trial_002"
    )

    assert first["output_dir"] != second["output_dir"]
    assert first["training"]["output_dir"] != second["training"]["output_dir"]
    assert first["training"]["tensorboard_run_dir"] != second["training"]["tensorboard_run_dir"]
    assert first["formats"]["w8"]["artifact"] != second["formats"]["w8"]["artifact"]
    assert first["scorecard"]["path"] != second["scorecard"]["path"]
    assert "trial_001" in first["checkpoint_evaluation"]["output_dir"]

    for unsafe in ("../reuse", "nested/run", "unsafe name", "trailing.", "NUL"):
        with pytest.raises(Gemma270MMultiformatError, match="safe path component"):
            build_pipeline_plan(
                load_yaml(CONFIG_PATH), config_path=CONFIG_PATH, run_id_override=unsafe
            )


def test_plan_has_four_truthful_litertlm_precision_contracts():
    plan = _plan()

    assert tuple(plan["formats"]) == FORMAT_ORDER
    assert plan["formats"]["w32"]["quantization_recipe"] == "none"
    assert plan["formats"]["w16"]["quantization_recipe"] == "none"
    assert (
        "--experimental_use_fp16=True"
        in plan["formats"]["w16"]["export_config"]["export"]["extra_flags"]
    )
    assert plan["formats"]["w8"]["quantization_recipe"] == "dynamic_wi8_afp32"
    assert plan["formats"]["w8"]["qat_aligned"] is True
    assert plan["formats"]["w4"]["quantization_recipe"] == "dynamic_wi4b32_afp32"
    assert plan["formats"]["w4"]["qat_aligned"] is False
    assert plan["formats"]["w16"]["requires_explicit_opt_in"] is True
    assert plan["formats"]["w4"]["requires_explicit_opt_in"] is True
    assert "--quantization_recipe=none" in plan["formats"]["w32"]["export_command"]
    assert (
        "--quantization_recipe=dynamic_wi8_afp32"
        in plan["formats"]["w8"]["export_command"]
    )


def test_every_variant_uses_the_same_golden_and_tensorboard_run():
    plan = _plan()
    expected_split = plan["golden"]["prepared_split"]

    for name in FORMAT_ORDER:
        command = plan["formats"][name]["evaluation"]["command"]
        assert command[command.index("--split") + 1] == expected_split
        assert command[command.index("--required-rows") + 1] == "32"
        assert command[command.index("--run-id") + 1] == plan["run_id"]
        assert command[command.index("--evaluation-name") + 1] == f"litertlm_{name}"
        assert command[command.index("--runner-config") + 1].endswith(
            "litertlm_external_runner.example.yaml"
        )
        inspection = plan["formats"][name]["package_validation"]["command"]
        assert "--include-hashes" in inspection
        assert inspection[inspection.index("--output") + 1].endswith(
            "package_inspection.json"
        )


def test_placeholder_runner_blocks_only_litertlm_execution(tmp_path: Path):
    placeholder = _plan()

    assert placeholder["validation"]["config_ok"] is True
    assert placeholder["validation"]["ok"] is False
    with pytest.raises(Gemma270MMultiformatError, match="placeholder"):
        _require_litertlm_runner_ready(placeholder)

    runner = tmp_path / "runner.yaml"
    runner.write_text(
        yaml.safe_dump(
            {
                "protocol": "a2ui_external_generation_v1",
                "command": [
                    "python",
                    "litertlm_runner.py",
                    "--model",
                    "{model_path}",
                    "--requests",
                    "{requests_path}",
                    "--outputs",
                    "{outputs_path}",
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    ready = build_pipeline_plan(
        load_yaml(CONFIG_PATH), config_path=CONFIG_PATH, runner_config_override=runner
    )

    assert ready["litertlm_runner"]["ready"] is True
    assert ready["validation"]["ok"] is True
    _require_litertlm_runner_ready(ready)


def test_plan_rejects_tampered_golden_pin():
    config = copy.deepcopy(load_yaml(CONFIG_PATH))
    config["pipeline"]["golden"]["source_sha256"] = "0" * 64

    plan = build_pipeline_plan(config, config_path=CONFIG_PATH)

    assert plan["validation"]["ok"] is False
    assert any(
        issue["code"] == "golden_source_sha256_mismatch"
        for issue in plan["validation"]["issues"]
    )


def test_plan_rejects_precision_recipe_drift():
    config = copy.deepcopy(load_yaml(CONFIG_PATH))
    config["pipeline"]["formats"]["w4"]["quantization_recipe"] = "dynamic_wi4_afp32"

    plan = build_pipeline_plan(config, config_path=CONFIG_PATH)

    assert plan["validation"]["ok"] is False
    assert any(
        issue["code"] == "w4_recipe_mismatch" for issue in plan["validation"]["issues"]
    )


def test_leakage_gate_uses_response_content_not_reused_run_ids(tmp_path: Path):
    dataset_dir = tmp_path / "prepared-training"
    dataset_dir.mkdir()
    (dataset_dir / "train.jsonl").write_text(
        json.dumps(
            {
                "source_id": "q_000001",
                "response_id": "r_000001_01",
                "response_text": "A distinct training response.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "val.jsonl").write_text(
        json.dumps({"response_text": "Another distinct validation response."}) + "\n",
        encoding="utf-8",
    )
    training_config = load_yaml(
        ROOT / "configs" / "models" / "gemma3_270m_a2ui_express_qat.yaml"
    )
    training_config["run"]["dataset_dir"] = str(dataset_dir)
    temporary_training_config = tmp_path / "training.yaml"
    temporary_training_config.write_text(
        yaml.safe_dump(training_config, sort_keys=False), encoding="utf-8"
    )
    pipeline_config = copy.deepcopy(load_yaml(CONFIG_PATH))
    pipeline_config["pipeline"]["training"]["config"] = str(temporary_training_config)

    reused_id_plan = build_pipeline_plan(pipeline_config, config_path=CONFIG_PATH)

    assert reused_id_plan["golden"]["training_leakage"]["checked"] is True
    assert reused_id_plan["golden"]["training_leakage"]["ok"] is True
    assert reused_id_plan["golden"]["training_leakage"]["overlap_count"] == 0

    golden_row = json.loads(
        Path(reused_id_plan["golden"]["training_leakage"]["signature_source"])
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    (dataset_dir / "train.jsonl").write_text(
        json.dumps(
            {
                "source_id": "different-source-id",
                "response_id": "different-response-id",
                "response_text": golden_row["response_text"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    leaked_plan = build_pipeline_plan(pipeline_config, config_path=CONFIG_PATH)

    assert leaked_plan["validation"]["ok"] is False
    assert leaked_plan["golden"]["training_leakage"]["overlap_count"] == 1
    assert any(
        issue["code"] == "golden_training_content_overlap"
        for issue in leaked_plan["validation"]["issues"]
    )
    with pytest.raises(Gemma270MMultiformatError, match="normalized response content"):
        _require_training_leakage_gate(leaked_plan)


@pytest.mark.parametrize("name", ("w16", "w4"))
def test_experimental_formats_require_explicit_opt_in(name: str):
    plan = _plan()

    with pytest.raises(Gemma270MMultiformatError, match="experimental"):
        _export_format(plan, name, allow_experimental=False)


def test_official_topology_is_scoped_to_w8_only():
    plan = _plan()

    assert plan["official_q8"]["scope"] == "w8_only"
    assert plan["official_q8"]["enabled"] is False
    command = plan["official_q8"]["command"]
    assert command[command.index("--family") + 1] == "gemma3_270m"
    assert command[command.index("--official-base-model-id") + 1] == (
        "google/gemma-3-270m-it"
    )


def test_official_q8_routes_evaluation_inspection_and_scorecard_artifact(
    tmp_path: Path,
):
    plan = _plan()
    public_artifact = plan["formats"]["w8"]["artifact"]
    official_artifact = tmp_path / "official-q8.litertlm"
    official_artifact.write_bytes(b"official q8")
    plan["official_q8"]["artifact"] = str(official_artifact)

    _use_official_q8_for_w8(plan)

    w8 = plan["formats"]["w8"]
    assert public_artifact != str(official_artifact)
    assert w8["artifact"] == str(official_artifact)
    assert w8["artifact_source"] == "official_q8_topology"
    assert w8["evaluation"]["command"][
        w8["evaluation"]["command"].index("--model") + 1
    ] == str(official_artifact)
    assert w8["package_validation"]["artifact"] == str(official_artifact)
    assert str(official_artifact) in w8["package_validation"]["command"]
    assert public_artifact not in w8["package_validation"]["command"]


def test_scorecard_requires_and_compares_checkpoint_and_all_variants(tmp_path: Path):
    plan = _plan()
    plan["scorecard"]["path"] = str(tmp_path / "scorecard.json")
    tensorboard_root = tmp_path / "tensorboard"
    plan["training"]["tensorboard_root"] = str(tensorboard_root)
    plan["training"]["tensorboard_evaluation_run_dir"] = str(
        tensorboard_root / plan["run_id"]
    )
    checkpoint = tmp_path / "best-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (checkpoint / "training_metadata.json").write_text(
        json.dumps({"best_golden_eval": {"step": 500}}), encoding="utf-8"
    )
    plan["training"]["best_checkpoint"] = str(checkpoint)
    checkpoint_aggregate = tmp_path / "checkpoint" / "aggregate_metrics.json"
    _attach_evaluation_evidence(
        plan,
        aggregate_path=checkpoint_aggregate,
        artifact=checkpoint,
        evaluation_name="checkpoint",
        artifact_field="checkpoint",
        score=0.8,
        tensorboard_root=tensorboard_root,
    )
    plan["checkpoint_evaluation"]["aggregate"] = str(checkpoint_aggregate)

    for index, name in enumerate(FORMAT_ORDER):
        aggregate = tmp_path / name / "aggregate_metrics.json"
        artifact = tmp_path / f"{name}.litertlm"
        artifact.write_bytes(name.encode("ascii"))
        package_report = tmp_path / name / "package_inspection.json"
        _attach_evaluation_evidence(
            plan,
            aggregate_path=aggregate,
            artifact=artifact,
            evaluation_name=f"litertlm_{name}",
            artifact_field="litertlm",
            score=0.7 - (index * 0.1),
            tensorboard_root=tensorboard_root,
            package_report=package_report,
        )
        plan["formats"][name]["evaluation"]["aggregate"] = str(aggregate)
        plan["formats"][name]["artifact"] = str(artifact)
        plan["formats"][name]["package_validation"]["report"] = str(package_report)

    scorecard = write_final_scorecard(plan)

    assert scorecard["complete"] is True
    assert [item["name"] for item in scorecard["results"]] == [
        "checkpoint",
        "litertlm_w32",
        "litertlm_w16",
        "litertlm_w8",
        "litertlm_w4",
    ]
    assert scorecard["results"][1]["score_delta_vs_checkpoint"] == pytest.approx(-0.1)
    checkpoint_result = scorecard["results"][0]
    assert checkpoint_result["artifact_exists"] is True
    assert checkpoint_result["artifact_type"] == "directory"
    assert checkpoint_result["artifact_identity_complete"] is True
    assert checkpoint_result["artifact_file_count"] == 3
    assert checkpoint_result["artifact_size_bytes"] > 0
    assert len(checkpoint_result["artifact_sha256"]) == 64
    assert len(checkpoint_result["evidence"]["artifact"]["sha256"]) == 64
    assert {item["path"] for item in checkpoint_result["artifact_key_files"]} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_metadata.json",
    }
    assert Path(scorecard["path"]).is_file()

    package_report = Path(plan["formats"]["w4"]["package_validation"]["report"])
    saved_package = json.loads(package_report.read_text(encoding="utf-8"))
    stale_package = copy.deepcopy(saved_package)
    stale_package["sha256"] = "0" * 64
    package_report.write_text(json.dumps(stale_package), encoding="utf-8")
    invalid = write_final_scorecard(plan, require_complete=False)
    assert invalid["results"][-1]["status"] == "evidence_invalid"
    assert "inspection" in invalid["results"][-1]["evidence_error"]
    package_report.write_text(json.dumps(saved_package), encoding="utf-8")

    swapped_package = copy.deepcopy(saved_package)
    swapped_package["graphs"] = [_precision_graph("w8")]
    package_report.write_text(json.dumps(swapped_package), encoding="utf-8")
    invalid = write_final_scorecard(plan, require_complete=False)
    assert invalid["results"][-1]["status"] == "evidence_invalid"
    assert "w4 precision mismatch" in invalid["results"][-1]["evidence_error"]
    package_report.write_text(json.dumps(saved_package), encoding="utf-8")

    checkpoint_result_path = checkpoint_aggregate.parent / "evaluation_result.json"
    checkpoint_result_payload = json.loads(
        checkpoint_result_path.read_text(encoding="utf-8")
    )
    checkpoint_record_path = Path(checkpoint_result_payload["tensorboard_record"])
    saved_record = json.loads(checkpoint_record_path.read_text(encoding="utf-8"))
    wrong_step_record = copy.deepcopy(saved_record)
    wrong_step_record["step"] = 499
    checkpoint_record_path.write_text(json.dumps(wrong_step_record), encoding="utf-8")
    invalid = write_final_scorecard(plan, require_complete=False)
    assert invalid["results"][0]["status"] == "evidence_invalid"
    assert "checkpoint step" in invalid["results"][0]["evidence_error"]

    wrong_golden_record = copy.deepcopy(saved_record)
    wrong_golden_record["artifacts"]["golden_split"]["sha256"] = "0" * 64
    checkpoint_record_path.write_text(
        json.dumps(wrong_golden_record), encoding="utf-8"
    )
    invalid = write_final_scorecard(plan, require_complete=False)
    assert invalid["results"][0]["status"] == "evidence_invalid"
    assert "Golden-split identity" in invalid["results"][0]["evidence_error"]


def test_checkpoint_step_is_loaded_from_selected_checkpoint_provenance(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training_metadata.json").write_text(
        json.dumps({"best_golden_eval": {"step": 1750}}), encoding="utf-8"
    )

    command = _with_checkpoint_step(
        ["python", "evaluate.py", "--step", "<checkpoint-step>"], checkpoint
    )

    assert command[-1] == "1750"


def test_scorecard_rejects_metrics_logged_for_different_artifact(tmp_path: Path):
    plan = _plan()
    plan["scorecard"]["path"] = str(tmp_path / "scorecard.json")
    tensorboard_root = tmp_path / "tensorboard"
    plan["training"]["tensorboard_evaluation_run_dir"] = str(
        tensorboard_root / plan["run_id"]
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"before")
    (checkpoint / "training_metadata.json").write_text(
        json.dumps({"best_golden_eval": {"step": 500}}), encoding="utf-8"
    )
    plan["training"]["best_checkpoint"] = str(checkpoint)
    aggregate = tmp_path / "checkpoint-eval" / "aggregate_metrics.json"
    _attach_evaluation_evidence(
        plan,
        aggregate_path=aggregate,
        artifact=checkpoint,
        evaluation_name="checkpoint",
        artifact_field="checkpoint",
        score=0.8,
        tensorboard_root=tensorboard_root,
    )
    plan["checkpoint_evaluation"]["aggregate"] = str(aggregate)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"after")

    scorecard = write_final_scorecard(plan, require_complete=False)

    checkpoint_result = scorecard["results"][0]
    assert checkpoint_result["status"] == "evidence_invalid"
    assert "identity mismatch" in checkpoint_result["evidence_error"]


def test_scorecard_rejects_checkpoint_without_provenance(tmp_path: Path):
    plan = _plan()
    plan["scorecard"]["path"] = str(tmp_path / "scorecard.json")
    checkpoint = tmp_path / "incomplete-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter weights")
    plan["training"]["best_checkpoint"] = str(checkpoint)
    aggregate = tmp_path / "checkpoint-metrics.json"
    aggregate.write_text(
        json.dumps({"generation_reward_v5_4_avg": 0.8}), encoding="utf-8"
    )
    plan["checkpoint_evaluation"]["aggregate"] = str(aggregate)

    for name in FORMAT_ORDER:
        format_aggregate = tmp_path / f"{name}-metrics.json"
        format_aggregate.write_text(
            json.dumps({"generation_reward_v5_4_avg": 0.7}), encoding="utf-8"
        )
        artifact = tmp_path / f"{name}.litertlm"
        artifact.write_bytes(name.encode("ascii"))
        plan["formats"][name]["evaluation"]["aggregate"] = str(format_aggregate)
        plan["formats"][name]["artifact"] = str(artifact)

    with pytest.raises(Gemma270MMultiformatError, match="Checkpoint provenance is missing"):
        write_final_scorecard(plan)
