from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_gemma4_mobile_qat.py"
SPEC = importlib.util.spec_from_file_location("portable_gemma4_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
PACKED_SOURCE_BYTES = b"portable-test-official-packed-source"


@pytest.fixture(autouse=True)
def _use_tiny_packed_source_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        launcher,
        "OFFICIAL_MOBILE_SAFETENSORS_SHA256",
        hashlib.sha256(PACKED_SOURCE_BYTES).hexdigest(),
    )


def _source_config(
    tmp_path: Path,
    *,
    unsafe: bool = False,
    config_path: Path | None = None,
) -> Path:
    import yaml

    source = launcher.load_yaml(
        config_path
        or ROOT
        / "configs"
        / "models"
        / "gemma4_e2b_mobile_seed_ir_qat_sft.yaml"
    )
    config = copy.deepcopy(source)
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    packed_source = seed_dir / "official_model.safetensors"
    packed_source.write_bytes(PACKED_SOURCE_BYTES)
    (seed_dir / "mobile_training_seed_manifest.json").write_text(
        json.dumps({"source": {"safetensors": str(packed_source)}}),
        encoding="utf-8",
    )
    (seed_dir / "mobile_qparams.json").write_text("{}", encoding="utf-8")
    config["model"]["model_source"] = str(seed_dir)
    config["model"]["mobile_training_seed_manifest"] = str(
        seed_dir / "mobile_training_seed_manifest.json"
    )
    config["model"]["mobile_qparams_contract"] = str(
        seed_dir / "mobile_qparams.json"
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for name in ("train.jsonl", "val.jsonl"):
        (dataset_dir / name).write_text(
            json.dumps({"id": f"sample-{name}"}) + "\n", encoding="utf-8"
        )
    config["run"]["dataset_dir"] = str(dataset_dir)
    golden_dir = tmp_path / "golden100"
    golden_dir.mkdir()
    (golden_dir / "all.jsonl").write_text(
        "".join(
            json.dumps({"id": f"golden-{index}"}) + "\n" for index in range(100)
        ),
        encoding="utf-8",
    )
    config["golden_eval"]["dataset_dir"] = str(golden_dir)
    config["qat"].update(
        {
            "scale_mode": "retained_mobile",
            "fixed_scale_required": True,
            "fixed_activation_scale_required": True,
            "effective_lora_only": True,
            "ste_gradient": "clipped",
        }
    )
    if unsafe:
        config["qat"]["scale_mode"] = "dynamic_absmax"
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_plan_is_dry_run_unique_and_uses_torchrun(tmp_path: Path) -> None:
    source = _source_config(tmp_path)
    runs_root = tmp_path / "runs"

    plan, resolved, paths = launcher.build_launch_plan(
        source,
        run_id="portable_test_001",
        runs_root=runs_root,
        num_gpus=3,
    )

    assert not paths["run_root"].exists()
    assert plan["mode"] == "fresh_run_only_no_resume"
    assert plan["checks"]["contract_ok"] is True
    assert plan["num_gpus"] == 3
    assert plan["training_command"][:3] == [
        launcher.sys.executable,
        "-m",
        "torch.distributed.run",
    ]
    assert "--nproc_per_node=3" in plan["training_command"]
    assert plan["training_command"][-1] == str(paths["resolved_config"])
    assert resolved["run"]["output_dir"] == str(paths["training_dir"])
    assert resolved["training"]["logging_dir"] == str(paths["tensorboard_dir"])
    assert resolved["golden_eval"]["output_dir"] == str(
        paths["golden_output_dir"]
    )
    assert resolved["golden_eval"]["best_checkpoint_dir"] == str(
        paths["best_checkpoint_dir"]
    )
    scale_gate = next(
        item for item in plan["preflights"] if item["id"] == "scale_preserving_qat"
    )
    assert "validate_gemma4_mobile_scale_preserving_qat.py" in " ".join(
        scale_gate["command"]
    )
    assert "--strict" in scale_gate["command"]
    assert "--source-safetensors" in scale_gate["command"]
    packed_source = Path(
        plan["bound_artifacts"]["official_packed_source"]["path"]
    )
    assert scale_gate["command"][
        scale_gate["command"].index("--source-safetensors") + 1
    ] == str(packed_source)
    numeric_gate = next(
        item for item in plan["preflights"] if item["id"] == "model_numeric_preflight"
    )
    assert numeric_gate["command"][:3] == [
        launcher.sys.executable,
        "-m",
        "torch.distributed.run",
    ]
    assert "--preflight-only" in numeric_gate["command"]
    numeric_gate = next(
        item for item in plan["preflights"] if item["id"] == "model_numeric_preflight"
    )
    assert "--preflight-only" in numeric_gate["command"]
    assert "torch.distributed.run" in numeric_gate["command"]
    assert plan["training_limit"] == {
        "mode": "num_train_epochs",
        "max_optimizer_steps": None,
        "configured_max_steps": None,
        "num_train_epochs": 2,
        "max_steps_overrides_epochs": False,
        "valid": True,
    }


def test_pinned_repeated_golden32_uses_unique_source_metric(tmp_path: Path) -> None:
    import yaml

    source = _source_config(tmp_path)
    config = launcher.load_yaml(source)
    golden32 = ROOT / "data/eval/golden32_archive_repeat_v1"
    config["golden_eval"].update(
        dataset_dir=str(golden32),
        split="golden32",
        split_path=str(golden32 / "golden32.jsonl"),
        max_rows=32,
        required_rows=32,
        metric_for_best_model="unique_source_generation_reward_v5_4_avg",
        metric_log_prefix="golden32",
    )
    source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    plan, _, _ = launcher.build_launch_plan(
        source,
        run_id="pinned_repeated_golden32",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )

    assert plan["checks"]["contract_ok"] is True
    contract = plan["golden_eval_contract"]
    assert contract["benchmark_kind"] == "explicit_repeated_case"
    assert contract["required_rows"] == 32
    assert contract["unique_source_count"] == 31
    assert (
        contract["metric_for_best_model"]
        == "unique_source_generation_reward_v5_4_avg"
    )
    evidence = plan["bound_artifacts"]["golden_benchmark_evidence"]
    assert evidence["path"] == str((golden32 / "benchmark_manifest.json").resolve())
    assert evidence["sha256"] == contract["benchmark_evidence_sha256"]


def test_repeated_golden32_rejects_ordinary_mean_metric(tmp_path: Path) -> None:
    import yaml

    source = _source_config(tmp_path)
    config = launcher.load_yaml(source)
    config["golden_eval"].update(
        dataset_dir=str(ROOT / "data/eval/golden32_archive_repeat_v1"),
        split="golden32",
        max_rows=32,
        required_rows=32,
        metric_for_best_model="generation_reward_v5_4_avg",
    )
    source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    plan, _, _ = launcher.build_launch_plan(
        source,
        run_id="repeated_wrong_metric",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )

    assert plan["checks"]["contract_ok"] is False
    assert "wrong_best_metric" in {
        issue["code"] for issue in plan["checks"]["issues"]
    }


def test_unbound_duplicate_cohort_is_not_a_general_bypass(tmp_path: Path) -> None:
    import yaml

    source = _source_config(tmp_path)
    config = launcher.load_yaml(source)
    golden = tmp_path / "unbound-repeat"
    golden.mkdir()
    rows = [{"id": f"case-{index}"} for index in range(31)]
    rows.append(dict(rows[0]))
    (golden / "all.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config["golden_eval"].update(
        dataset_dir=str(golden),
        max_rows=32,
        required_rows=32,
        metric_for_best_model="unique_source_generation_reward_v5_4_avg",
    )
    source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    plan, _, _ = launcher.build_launch_plan(
        source,
        run_id="unbound_duplicate",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )

    issue_codes = {issue["code"] for issue in plan["checks"]["issues"]}
    assert "invalid_golden_selection_contract" in issue_codes
    assert "golden_eval_contract_invalid" in issue_codes


def test_three_step_smoke_config_is_bounded_visible_and_non_promotable(
    tmp_path: Path,
) -> None:
    smoke_path = (
        ROOT
        / "configs"
        / "models"
        / "gemma4_e2b_mobile_seed_ir_qat_sft_smoke_3_steps.yaml"
    )
    source = _source_config(tmp_path, config_path=smoke_path)

    plan, resolved, _ = launcher.build_launch_plan(
        source,
        run_id="bounded_smoke_3_steps",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )

    assert plan["checks"]["contract_ok"] is True
    assert plan["training_limit"]["mode"] == "max_optimizer_steps"
    assert plan["training_limit"]["max_optimizer_steps"] == 3
    assert plan["training_limit"]["max_steps_overrides_epochs"] is True
    assert plan["run_intent"] == {
        "purpose": "bounded_retained_scale_pipeline_smoke",
        "quality_promotion_eligible": False,
        "serialization_validation_eligible": True,
    }
    assert resolved["training"]["max_steps"] == 3
    assert resolved["training"]["logging_steps"] == 1
    assert resolved["training"]["eval_steps"] == 3
    assert resolved["training"]["save_steps"] == 3
    assert resolved["training"]["save_total_limit"] == 1
    assert resolved["golden_eval"]["max_new_tokens"] == 128
    assert resolved["run"]["purpose"] == "bounded_retained_scale_pipeline_smoke"
    assert resolved["run"]["quality_promotion_eligible"] is False
    assert resolved["run"]["serialization_validation_eligible"] is True


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "3", None])
def test_plan_rejects_invalid_max_steps(tmp_path: Path, invalid: object) -> None:
    import yaml

    source = _source_config(tmp_path)
    config = launcher.load_yaml(source)
    config["training"]["max_steps"] = invalid
    source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    plan, _, _ = launcher.build_launch_plan(
        source,
        run_id="invalid_bounded_limit",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )

    assert plan["training_limit"]["valid"] is False
    assert "invalid_max_steps" in {
        item["code"] for item in plan["checks"]["issues"]
    }


def test_plan_rejects_bounded_run_with_eval_save_after_max_steps(
    tmp_path: Path,
) -> None:
    import yaml

    source = _source_config(tmp_path)
    config = launcher.load_yaml(source)
    config["training"].update({"max_steps": 3, "eval_steps": 4, "save_steps": 4})
    source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    plan, _, _ = launcher.build_launch_plan(
        source,
        run_id="bounded_bad_cadence",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )

    assert "bounded_run_cadence_exceeds_max_steps" in {
        item["code"] for item in plan["checks"]["issues"]
    }


def test_plan_rejects_absmax_scale_mode(tmp_path: Path) -> None:
    plan, _, _ = launcher.build_launch_plan(
        _source_config(tmp_path, unsafe=True),
        run_id="unsafe_scale_test",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )

    assert plan["checks"]["contract_ok"] is False
    assert "wrong_scale_mode" in {
        item["code"] for item in plan["checks"]["issues"]
    }


@pytest.mark.parametrize(
    ("section", "key", "value", "expected_code"),
    [
        ("preflight", "min_top1_probe_match", 0.5, "weak_top1_numeric_gate"),
        (
            "preflight",
            "require_greedy_determinism",
            False,
            "missing_greedy_generation_gate",
        ),
        ("training", "save_steps", 250, "checkpoint_eval_cadence_mismatch"),
    ],
)
def test_plan_rejects_weakened_numeric_and_provenance_gates(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    expected_code: str,
) -> None:
    import yaml

    source = _source_config(tmp_path)
    config = launcher.load_yaml(source)
    config[section][key] = value
    source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    plan, _, _ = launcher.build_launch_plan(
        source,
        run_id=f"weakened_{key}",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )

    assert plan["checks"]["contract_ok"] is False
    assert expected_code in {
        issue["code"] for issue in plan["checks"]["issues"]
    }


def test_copied_seed_requires_and_binds_local_packed_source_override(
    tmp_path: Path,
) -> None:
    source = _source_config(tmp_path)
    manifest = tmp_path / "seed" / "mobile_training_seed_manifest.json"
    manifest.write_text(
        json.dumps({"source": {"safetensors": str(tmp_path / "old-pc" / "missing.safetensors")}}),
        encoding="utf-8",
    )

    missing_plan, _, _ = launcher.build_launch_plan(
        source,
        run_id="copied_seed_missing_source",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )
    assert missing_plan["checks"]["contract_ok"] is False
    assert "packed_source_not_materialized" in {
        issue["code"] for issue in missing_plan["checks"]["issues"]
    }

    local_source = tmp_path / "copied-official-model.safetensors"
    local_source.write_bytes(PACKED_SOURCE_BYTES)
    plan, _, _ = launcher.build_launch_plan(
        source,
        run_id="copied_seed_local_source",
        runs_root=tmp_path / "runs",
        num_gpus=1,
        source_safetensors=local_source,
    )

    assert plan["checks"]["contract_ok"] is True
    assert plan["packed_source"] == {
        "origin": "command_line_override",
        "path": str(local_source.resolve()),
        "error": None,
    }
    identity = plan["bound_artifacts"]["official_packed_source"]
    assert identity["path"] == str(local_source.resolve())
    assert identity["size_bytes"] == len(PACKED_SOURCE_BYTES)
    assert identity["sha256"] == hashlib.sha256(PACKED_SOURCE_BYTES).hexdigest()
    scale_gate = next(
        item for item in plan["preflights"] if item["id"] == "scale_preserving_qat"
    )
    assert scale_gate["command"][
        scale_gate["command"].index("--source-safetensors") + 1
    ] == str(local_source.resolve())


def test_packed_source_override_rejects_wrong_official_identity(
    tmp_path: Path,
) -> None:
    wrong_source = tmp_path / "wrong-model.safetensors"
    wrong_source.write_bytes(b"not-the-pinned-official-model")

    plan, _, _ = launcher.build_launch_plan(
        _source_config(tmp_path),
        run_id="wrong_packed_identity",
        runs_root=tmp_path / "runs",
        num_gpus=1,
        source_safetensors=wrong_source,
    )

    assert plan["checks"]["contract_ok"] is False
    assert "packed_source_identity_mismatch" in {
        issue["code"] for issue in plan["checks"]["issues"]
    }


def test_reserve_refuses_existing_run_without_modifying_it(tmp_path: Path) -> None:
    plan, resolved, paths = launcher.build_launch_plan(
        _source_config(tmp_path),
        run_id="do_not_resume",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )
    paths["run_root"].mkdir(parents=True)
    marker = paths["run_root"] / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(launcher.PortableTrainingLaunchError, match="resume/reuse"):
        launcher._reserve_run(plan, resolved, paths)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not paths["resolved_config"].exists()


def test_reserve_binds_resolved_config_and_detects_post_gate_edit(
    tmp_path: Path,
) -> None:
    plan, resolved, paths = launcher.build_launch_plan(
        _source_config(tmp_path),
        run_id="bind_resolved_config",
        runs_root=tmp_path / "runs",
        num_gpus=1,
    )

    launcher._reserve_run(plan, resolved, paths)

    written_plan = json.loads(paths["launch_plan"].read_text(encoding="utf-8"))
    identity = written_plan["bound_artifacts"]["resolved_training_config"]
    assert identity["path"] == str(paths["resolved_config"].resolve())
    assert identity["size_bytes"] == paths["resolved_config"].stat().st_size
    assert len(identity["sha256"]) == 64
    assert launcher._verify_bound_artifacts(written_plan) == []

    paths["resolved_config"].write_text(
        paths["resolved_config"].read_text(encoding="utf-8")
        + "\n# unauthorized post-gate edit\n",
        encoding="utf-8",
    )
    issues = launcher._verify_bound_artifacts(written_plan)

    assert issues == [
        {
            "role": "resolved_training_config",
            "code": "identity_changed_after_plan",
            "path": str(paths["resolved_config"].resolve()),
            "expected_size_bytes": identity["size_bytes"],
            "observed_size_bytes": paths["resolved_config"].stat().st_size,
            "expected_sha256": identity["sha256"],
            "observed_sha256": launcher._sha256_file(paths["resolved_config"]),
        }
    ]


def test_preflight_report_binds_gate_log_and_declared_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_dir = tmp_path / "launch"
    paths = {
        "launch_dir": launch_dir,
        "preflight_report": launch_dir / "preflight_report.json",
    }
    output_path = launch_dir / "preflight" / "gate.json"
    plan = {
        "run_id": "preflight_artifact_binding",
        "preflights": [
            {
                "id": "synthetic_gate",
                "command": ["synthetic-validator", "--output", str(output_path)],
            }
        ],
    }

    def fake_run_live(command: list[str], *, cwd: Path, log_path: Path) -> int:
        del command, cwd
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("gate passed\n", encoding="utf-8")
        output_path.write_text('{"passed": true}\n', encoding="utf-8")
        return 0

    monkeypatch.setattr(launcher, "_run_live", fake_run_live)
    report = launcher._run_preflights(plan, paths=paths)

    assert report["all_passed"] is True
    gate = report["reports"][0]
    assert gate["log"]["present"] is True
    assert gate["log"]["sha256"] == launcher._sha256_file(
        Path(gate["log"]["path"])
    )
    assert gate["declared_outputs"] == [
        {
            "present": True,
            **launcher._file_identity(output_path),
        }
    ]
    persisted = json.loads(paths["preflight_report"].read_text(encoding="utf-8"))
    assert persisted == report


def test_preflight_fails_closed_when_declared_output_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_dir = tmp_path / "launch"
    missing_output = launch_dir / "preflight" / "missing.json"
    paths = {
        "launch_dir": launch_dir,
        "preflight_report": launch_dir / "preflight_report.json",
    }
    plan = {
        "run_id": "missing_preflight_output",
        "preflights": [
            {
                "id": "synthetic_gate",
                "command": ["synthetic-validator", "--output", str(missing_output)],
            }
        ],
    }

    def fake_run_live(command: list[str], *, cwd: Path, log_path: Path) -> int:
        del command, cwd
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("claimed success\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(launcher, "_run_live", fake_run_live)
    report = launcher._run_preflights(plan, paths=paths)

    assert report["all_passed"] is False
    gate = report["reports"][0]
    assert gate["exit_code"] == 0
    assert gate["passed"] is False
    assert gate["declared_outputs"] == [
        {"present": False, "path": str(missing_output.resolve())}
    ]


def test_main_default_only_prints_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _source_config(tmp_path)
    run_root = tmp_path / "runs" / "plan_only"

    exit_code = launcher.main(
        [
            "--config",
            str(source),
            "--run-id",
            "plan_only",
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Plan only" in captured.out
    assert not run_root.exists()
    payload = json.loads(captured.out.split("\nPlan only:", 1)[0])
    assert payload["checks"]["contract_ok"] is True


def test_host_gpu_profile_binds_ddp_without_changing_retained_qat(tmp_path: Path, monkeypatch) -> None:
    from ir_training.train.gpu_profile import build_gpu_profile
    source = _source_config(tmp_path)
    inventory = {"version": 1, "inherited_cuda_visible_devices": "GPU-0,GPU-1,GPU-2,GPU-3", "visible_gpu_count": 4,
        "devices": [{"visible_index": index, "launch_identifier": f"GPU-{index}", "uuid": f"GPU-{index}",
                     "name": "NVIDIA H100 80GB", "total_memory_bytes": 80 * 1024**3, "compute_capability": [9, 0]}
                    for index in range(4)]}
    profile = build_gpu_profile(inventory, model="e2b")
    monkeypatch.delenv("A2UI_TENSORBOARD_ROOT", raising=False)
    plan, resolved, _ = launcher.build_launch_plan(source, run_id="gpu_bound", runs_root=tmp_path / "runs", num_gpus=4, host_gpu_profile=profile)
    assert plan["checks"]["contract_ok"]
    assert "--nproc_per_node=4" in plan["training_command"]
    assert plan["cuda_visible_devices"] == inventory["inherited_cuda_visible_devices"]
    assert resolved["runtime"]["world_size"] == 4
    assert resolved["training"]["per_device_train_batch_size"] == 1
    assert resolved["training"]["gradient_accumulation_steps"] == 8
    assert resolved["training"]["expected_effective_batch_size"] == 32
    assert resolved["qat"] == launcher.load_yaml(source)["qat"]
