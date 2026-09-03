from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.export.merge_lora import (
    _REQUIRED_PORTABLE_PREFLIGHTS,
    _portable_launcher_contract_matches,
)
from ir_training.pipeline.gemma4_e2b_multiformat import (
    Gemma4E2BMultiformatPipelineError,
    _materialize_mtp_training_config,
    build_pipeline_plan,
    run_pipeline,
)


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_e2b_multiformat_plan_keeps_official_and_public_lanes_distinct():
    config_path = (
        ROOT
        / "configs"
        / "pipelines"
        / "gemma4_e2b_a2ui_express_multiformat.yaml"
    )
    plan = build_pipeline_plan(
        load_yaml(config_path),
        config_path=config_path,
        run_id_override="unit_e2b_multiformat",
    )

    assert plan["validation"]["config_ok"] is True
    assert plan["golden"]["required_rows"] == 32
    assert plan["training"]["periodic_golden_evaluation"] == (
        "every Trainer eval event"
    )
    assert plan["paths"]["tensorboard_root"] == str(
        (ROOT.parent / "tensorboard").resolve()
    )
    assert plan["official_qat"]["official_format"] is True
    assert plan["official_qat"]["format"] == (
        "retained_mobile_w2_w4_w8_a8_code_only"
    )
    assert plan["official_qat"]["mtp"]["enabled"] is True
    blocker_codes = {item["code"] for item in plan["validation"]["blockers"]}
    assert "training_model_source_missing" in blocker_codes
    assert "training_seed_manifest_missing" in blocker_codes
    assert "training_qparams_contract_missing" in blocker_codes
    assert "training_train_split_missing" in blocker_codes
    assert "training_validation_split_missing" in blocker_codes
    assert "mtp_training" not in plan["execute_all_stage_order"]
    assert not Path(plan["paths"]["output_root"]).exists()

    variants = plan["public_variants"]
    assert set(variants) == {"w32", "w16", "w8", "w4"}
    assert all(item["official_format"] is False for item in variants.values())
    assert variants["w32"]["quantization_recipe"] == "none"
    assert variants["w16"]["quantization_recipe"] == "none"
    assert variants["w16"]["support"] == "experimental_public_fp16"
    assert "--experimental_use_fp16=True" in variants["w16"][
        "converter_command"
    ]
    assert variants["w8"]["quantization_recipe"] == "dynamic_wi8_afp32"
    assert variants["w4"]["artifact_kind"] == "mixed_w4_w8"
    assert variants["w4"]["quantization_recipe"] == "gemma4_mixed48_b32"
    assert "not_pure_int4" in variants["w4"]["support"]
    assert plan["evaluation"]["score_names"] == [
        "checkpoint",
        "litertlm_w32",
        "litertlm_w16",
        "litertlm_w8",
        "litertlm_w4",
        "litertlm_official_qat",
    ]
    for item in [*variants.values(), plan["official_qat"]]:
        inspection = item["package_validation"]
        assert "--include-hashes" in inspection["command_template"]
        assert inspection["report"].endswith("package_inspection.json")
        assert inspection["command_template"][
            inspection["command_template"].index("--output") + 1
        ] == inspection["report"]


def test_e2b_multiformat_mtp_can_be_disabled_without_affecting_public_formats():
    config_path = (
        ROOT
        / "configs"
        / "pipelines"
        / "gemma4_e2b_a2ui_express_multiformat.yaml"
    )
    plan = build_pipeline_plan(
        load_yaml(config_path),
        config_path=config_path,
        run_id_override="unit_e2b_no_mtp",
        mtp_enabled_override=False,
    )

    assert plan["official_qat"]["mtp"]["enabled"] is False
    assert "mtp_training" not in plan["execute_all_stage_order"]
    assert "--mtp-enabled" not in plan["official_qat"]["evaluation"][
        "command_template"
    ]
    assert all(
        "--mtp-enabled" not in item["evaluation"]["command_template"]
        for item in plan["public_variants"].values()
    )


def test_e2b_multiformat_execute_all_adds_only_opted_in_trained_mtp():
    config_path = (
        ROOT
        / "configs"
        / "pipelines"
        / "gemma4_e2b_a2ui_express_multiformat.yaml"
    )
    config = copy.deepcopy(load_yaml(config_path))
    config["pipeline"]["mtp"].update(
        {"enabled": True, "weight_source": "trained", "train_assistant": True}
    )

    plan = build_pipeline_plan(
        config,
        config_path=config_path,
        run_id_override="unit_e2b_trained_mtp",
    )

    assert "mtp_training" in plan["execute_all_stage_order"]
    mtp = plan["official_qat"]["mtp"]
    assert Path(mtp["trained_checkpoint"]).is_relative_to(
        Path(plan["paths"]["trainer_run_root"])
    )
    assert Path(mtp["training_config"]).parent == (
        Path(plan["paths"]["trainer_run_root"]) / "launch"
    )
    assert len(mtp["resolved_training_config_sha256"]) == 64
    assert "trained_mtp_is_not_private_google_recipe" in {
        item["code"] for item in plan["validation"]["warnings"]
    }


def test_e2b_trained_mtp_config_is_materialized_inside_unique_run(tmp_path: Path):
    config_path = (
        ROOT
        / "configs"
        / "pipelines"
        / "gemma4_e2b_a2ui_express_multiformat.yaml"
    )
    config = copy.deepcopy(load_yaml(config_path))
    config["pipeline"]["output_root"] = str(tmp_path / "pipeline")
    config["pipeline"]["training"]["runs_root"] = str(tmp_path / "runs")
    config["pipeline"]["mtp"].update(
        {"enabled": True, "weight_source": "trained", "train_assistant": True}
    )
    plan = build_pipeline_plan(
        config,
        config_path=config_path,
        run_id_override="unit_materialized_mtp",
    )

    _materialize_mtp_training_config(plan)

    mtp = plan["official_qat"]["mtp"]
    resolved_path = Path(mtp["training_config"])
    resolved = load_yaml(resolved_path)
    assert resolved["run"]["id"] == "unit_materialized_mtp"
    assert resolved["run"]["output_dir"] == mtp["run_root"]
    assert resolved["assistant"]["best_checkpoint_dir"] == mtp[
        "trained_checkpoint"
    ]
    assert resolved["training"]["tensorboard_subdir"] == "mtp_training"
    assert hashlib.sha256(resolved_path.read_bytes()).hexdigest() == mtp[
        "resolved_training_config_sha256"
    ]


def test_e2b_multiformat_rejects_pure_int4_claim_for_public_w4():
    config_path = (
        ROOT
        / "configs"
        / "pipelines"
        / "gemma4_e2b_a2ui_express_multiformat.yaml"
    )
    config = copy.deepcopy(load_yaml(config_path))
    config["pipeline"]["public_variants"]["w4"]["artifact_kind"] = "int4"

    plan = build_pipeline_plan(
        config,
        config_path=config_path,
        run_id_override="unit_invalid_int4_claim",
    )

    assert plan["validation"]["config_ok"] is False
    assert "invalid_w4_export_contract" in {
        item["code"] for item in plan["validation"]["errors"]
    }


def test_e2b_multiformat_runner_override_replaces_checked_in_placeholder(
    tmp_path: Path,
):
    config_path = (
        ROOT
        / "configs"
        / "pipelines"
        / "gemma4_e2b_a2ui_express_multiformat.yaml"
    )
    runner = tmp_path / "real_runner.yaml"
    runner.write_text(
        yaml.safe_dump(
            {
                "protocol": "a2ui_external_generation_v1",
                "command": [
                    "litertlm-runner",
                    "--model",
                    "{model_path}",
                    "--requests",
                    "{requests_path}",
                    "--outputs",
                    "{outputs_path}",
                ],
            },
            sort_keys=False,
        )
    )
    plan = build_pipeline_plan(
        load_yaml(config_path),
        config_path=config_path,
        run_id_override="unit_e2b_runner_override",
        runner_config_override=runner,
    )

    assert plan["evaluation"]["runner_config"] == str(runner.resolve())
    assert plan["validation"]["config_ok"] is True
    assert "litertlm_runner_is_placeholder" not in {
        item["code"] for item in plan["validation"]["blockers"]
    }
    for item in [*plan["public_variants"].values(), plan["official_qat"]]:
        command = item["evaluation"]["command_template"]
        assert command[command.index("--runner-config") + 1] == str(runner.resolve())


def test_e2b_mtp_training_stage_requires_explicit_trained_mode(tmp_path: Path):
    config_path = (
        ROOT
        / "configs"
        / "pipelines"
        / "gemma4_e2b_a2ui_express_multiformat.yaml"
    )
    config = copy.deepcopy(load_yaml(config_path))
    config["pipeline"]["output_root"] = str(tmp_path / "outputs")
    config["pipeline"]["training"]["runs_root"] = str(tmp_path / "runs")

    with pytest.raises(
        Gemma4E2BMultiformatPipelineError,
        match="weight_source=trained",
    ):
        run_pipeline(
            config,
            config_path=config_path,
            run_id_override="unit_mtp_guard",
            stages=("mtp_training",),
        )


def test_e2b_execute_all_fails_before_reserving_run_when_inputs_missing(
    tmp_path: Path,
):
    config_path = (
        ROOT
        / "configs"
        / "pipelines"
        / "gemma4_e2b_a2ui_express_multiformat.yaml"
    )
    config = copy.deepcopy(load_yaml(config_path))
    output_base = tmp_path / "outputs"
    config["pipeline"]["output_root"] = str(output_base)
    config["pipeline"]["training"]["runs_root"] = str(tmp_path / "runs")

    with pytest.raises(
        Gemma4E2BMultiformatPipelineError,
        match="external artifacts/configuration",
    ):
        run_pipeline(
            config,
            config_path=config_path,
            run_id_override="unit_missing_inputs",
            execute_all=True,
        )

    assert not (output_base / "unit_missing_inputs").exists()


def test_portable_launcher_accepts_exact_golden32_and_central_tensorboard(
    tmp_path: Path, monkeypatch,
):
    script = ROOT / "scripts" / "run_gemma4_mobile_qat.py"
    spec = importlib.util.spec_from_file_location("portable_e2b_golden32", script)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    config = copy.deepcopy(
        load_yaml(
            ROOT
            / "configs"
            / "models"
            / "gemma4_e2b_a2ui_express_official_qat.yaml"
        )
    )
    seed = tmp_path / "seed"
    seed.mkdir()
    packed = seed / "model.safetensors"
    packed.write_bytes(b"exact-packed-source")
    monkeypatch.setattr(
        launcher,
        "OFFICIAL_MOBILE_SAFETENSORS_SHA256",
        hashlib.sha256(packed.read_bytes()).hexdigest(),
    )
    manifest = seed / "mobile_training_seed_manifest.json"
    manifest.write_text(json.dumps({"source": {"safetensors": str(packed)}}))
    qparams = seed / "mobile_qparams.json"
    qparams.write_text("{}")
    config["model"].update(
        {
            "model_source": str(seed),
            "mobile_training_seed_manifest": str(manifest),
            "mobile_qparams_contract": str(qparams),
        }
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text('{"id":"train"}\n')
    (dataset / "val.jsonl").write_text('{"id":"val"}\n')
    config["run"]["dataset_dir"] = str(dataset)
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "all.jsonl").write_text(
        "".join(json.dumps({"id": f"g-{index}"}) + "\n" for index in range(32))
    )
    config["golden_eval"].pop("dataset_config", None)
    config["golden_eval"].pop("source_genui_sha256", None)
    config["golden_eval"].pop("source_responses_sha256", None)
    config["golden_eval"]["dataset_dir"] = str(golden)
    config["training"]["tensorboard_root"] = str(tmp_path / "configured_tensorboard")
    monkeypatch.setenv("A2UI_TENSORBOARD_ROOT", str(tmp_path / "mlp_tensorboard"))
    source = tmp_path / "training.yaml"
    import yaml

    source.write_text(yaml.safe_dump(config, sort_keys=False))
    plan, resolved, paths = launcher.build_launch_plan(
        source,
        run_id="golden32_contract",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )

    assert plan["checks"]["contract_ok"] is True
    assert "golden_eval" in plan["bound_artifacts"]
    assert "golden100" not in plan["bound_artifacts"]
    assert plan["golden_eval_contract"]["required_rows"] == 32
    assert plan["golden_eval_contract"]["require_exact_rows"] is True
    assert resolved["training"]["logging_dir"] == str(
        (tmp_path / "mlp_tensorboard" / "golden32_contract" / "training").resolve()
    )
    assert paths["golden_output_dir"].name == "golden32"


def test_portable_launcher_leakage_uses_response_content_not_run_local_ids(
    tmp_path: Path,
):
    script = ROOT / "scripts" / "run_gemma4_mobile_qat.py"
    spec = importlib.util.spec_from_file_location("portable_e2b_leakage", script)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text(
        json.dumps({"source_id": "q_000001", "response_text": "Train response"})
        + "\n"
    )
    (dataset / "val.jsonl").write_text(
        json.dumps({"source_id": "q_000002", "response_text": "Validation response"})
        + "\n"
    )
    golden = tmp_path / "golden"
    golden.mkdir()
    golden_path = golden / "all.jsonl"
    config = {
        "model": {
            "model_source": str(tmp_path / "missing-model"),
            "mobile_training_seed_manifest": str(tmp_path / "missing-manifest"),
            "mobile_qparams_contract": str(tmp_path / "missing-qparams"),
        },
        "run": {"dataset_dir": str(dataset)},
        "golden_eval": {
            "dataset_dir": str(golden),
            "split": "all",
            "required_rows": 1,
        },
    }

    # q_000001 is only a run-local ID; different content must be allowed.
    golden_path.write_text(
        json.dumps({"source_id": "q_000001", "response_text": "Golden response"})
        + "\n"
    )
    issue_codes = {
        item["code"] for item in launcher._artifact_contract_issues(config)
    }
    assert "golden_eval_training_content_overlap" not in issue_codes

    # A different ID with the same whitespace-normalized response is leakage.
    golden_path.write_text(
        json.dumps(
            {"source_id": "golden-different-id", "response_text": " Train   response "}
        )
        + "\n"
    )
    issue_codes = {
        item["code"] for item in launcher._artifact_contract_issues(config)
    }
    assert "golden_eval_training_content_overlap" in issue_codes


def test_merge_provenance_accepts_new_exact_golden_role_and_rejects_weakening(
    tmp_path: Path,
):
    config_sha = "a" * 64
    bound: dict[str, dict[str, object]] = {}
    for role in (
        "mobile_seed_manifest",
        "mobile_qparams_contract",
        "official_packed_source",
        "training_train",
        "training_val",
        "golden_eval",
        "golden_source_genui",
        "golden_source_responses",
        "resolved_training_config",
    ):
        path = tmp_path / f"{role}.bin"
        path.write_bytes(role.encode())
        bound[role] = _identity(path)
    bound["resolved_training_config"]["sha256"] = config_sha

    reports = []
    for gate in sorted(_REQUIRED_PORTABLE_PREFLIGHTS):
        gate_log = tmp_path / f"{gate}.log"
        gate_log.write_text("passed")
        reports.append(
            {
                "id": gate,
                "passed": True,
                "log": {"present": True, **_identity(gate_log)},
                "declared_outputs": [],
            }
        )
    launch_plan = {
        "mode": "fresh_run_only_no_resume",
        "run_id": "new-golden-contract",
        "checks": {"contract_ok": True},
        "bound_artifacts": bound,
        "golden_eval_contract": {
            "artifact_role": "golden_eval",
            "required_rows": 32,
            "max_rows": 32,
            "require_exact_rows": True,
            "require_unique_rows": True,
            "metric_for_best_model": "generation_reward_v5_4_avg",
            "source_genui_sha256": bound["golden_source_genui"]["sha256"],
            "source_responses_sha256": bound["golden_source_responses"][
                "sha256"
            ],
        },
    }
    launch_path = tmp_path / "launch_plan.json"
    launch_path.write_text(json.dumps(launch_plan))
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "run_id": "new-golden-contract",
                "all_passed": True,
                "reports": reports,
            }
        )
    )
    metadata = {
        "run_id": "new-golden-contract",
        "golden_eval": {
            "required_rows": 32,
            "max_rows": 32,
            "require_exact_rows": True,
            "require_unique_rows": True,
            "metric_for_best_model": "generation_reward_v5_4_avg",
        },
        "launcher_provenance": {
            "launch_plan": {"present": True, **_identity(launch_path)},
            "preflight_report": {"present": True, **_identity(preflight_path)},
        },
    }

    assert _portable_launcher_contract_matches(
        metadata, training_config_sha256=config_sha
    )
    metadata["golden_eval"]["require_exact_rows"] = False
    assert not _portable_launcher_contract_matches(
        metadata, training_config_sha256=config_sha
    )
