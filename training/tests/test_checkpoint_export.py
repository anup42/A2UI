"""Export-only orchestration with synthetic artifacts, not real model conversion."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from ir_training.pipeline import checkpoint_export as ce
from ir_training.pipeline import deployment_export as de


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def inputs(tmp_path, profile="e2b"):
    fit = tmp_path / "trained run/fit"
    base = tmp_path / "base model"
    data = tmp_path / "prepared data"
    write_json(
        base / "config.json", {"model_type": "gemma4" if profile == "e2b" else "gemma3"}
    )
    write_json(data / "manifest.json", {})
    (data / "train.jsonl").write_text("{}\n", encoding="utf-8")
    config = {
        "model": {"model_source": str(base)},
        "run": {"dataset_dir": str(data)},
        "golden_eval": {"max_input_tokens": 4096, "max_new_tokens": 2048},
    }
    write_json(fit / "training_config.yaml", config)
    digest = de.file_sha256(fit / "training_config.yaml")
    write_json(
        fit / "preparation_report.json",
        {
            "training_config_sha256": digest,
            "model_files": {"config.json": de.file_sha256(base / "config.json")},
        },
    )
    write_json(
        fit / "training/best_golden_checkpoint/training_metadata.json",
        {
            "training_config_sha256": digest,
            "checkpoint_step": 1000,
            "checkpoint_kind": "lora_adapter" if profile == "e2b" else "full_model",
        },
    )
    return ce.CheckpointExportOptions(
        profile=profile,
        fit_dir=fit,
        output_dir=tmp_path / "exports",
        training_python=Path(sys.executable),
        exporter_python=Path(sys.executable),
        allow_experimental_formats=True,
    )


def change_config(options, update):
    path = options.fit_dir / "training_config.yaml"
    config = json.loads(path.read_text())
    update(config)
    write_json(path, config)
    for bound in (
        options.fit_dir / "preparation_report.json",
        options.fit_dir / "training/best_golden_checkpoint/training_metadata.json",
    ):
        value = json.loads(bound.read_text())
        value["training_config_sha256"] = de.file_sha256(path)
        write_json(bound, value)


def fake_runner(calls, *, fail=None, corrupt=None):
    """Stand in for child processes, exercising real manifest/precision validation."""

    def run(argv, logfile, env, **kwargs):
        stage = logfile.stem
        calls.append((stage, argv, env, kwargs))
        logfile.parent.mkdir(parents=True, exist_ok=True)
        logfile.write_text(f"Synthetic {stage}\n", encoding="utf-8")
        if stage == fail:
            raise RuntimeError(f"simulated failure at {stage}")
        value = lambda key: argv[argv.index(key) + 1]
        profile = value("--profile")
        if argv[3] == "probe":
            write_json(
                Path(value("--report")), {"status": "passed", "profile": profile}
            )
        elif argv[3] == "prepare":
            merged = Path(value("--output-dir"))
            write_json(merged / "config.json", {})
            write_json(
                merged / "deployment_source.json",
                {
                    "profile": profile,
                    "source_checkpoint": value("--checkpoint"),
                    "merged_model_dir": str(merged),
                    "training_config_sha256": de.file_sha256(
                        Path(value("--training-config"))
                    ),
                    "merged_files": {
                        "config.json": de.file_sha256(merged / "config.json")
                    },
                    "official_retained_scale_export": False,
                },
            )
        else:
            folder = Path(value("--output-dir"))
            folder.mkdir(parents=True)
            artifact = folder / "model.litertlm"
            artifact.write_bytes(b"synthetic test artifact, not a usable model")
            variant = value("--variant")
            dtype = {"w32": "FLOAT32", "w16": "FLOAT16", "w8": "INT8", "w4": "INT4"}[
                variant
            ]
            package = {
                "graphs": [
                    {
                        "available": True,
                        "buffers": [
                            {"index": 0, "logical_size": 0},
                            {"index": 1, "logical_size": 32},
                        ],
                        "subgraphs": [
                            {
                                "tensors": [
                                    {
                                        "type_name": "FLOAT32",
                                        "buffer": 0,
                                        "shape": [1, 8],
                                    },
                                    {"type_name": dtype, "buffer": 1, "shape": [8, 8]},
                                ],
                                "operators": [
                                    {
                                        "builtin_name": "FULLY_CONNECTED",
                                        "inputs": [0, 1],
                                        "outputs": [],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
            inspection = folder / "package_inspection.json"
            write_json(inspection, package)
            write_json(
                folder / "export_manifest.json",
                {
                    "profile": profile,
                    "variant": variant,
                    "artifact": str(artifact),
                    "sha256": de.file_sha256(artifact),
                    "inspection_sha256": de.file_sha256(inspection),
                    "actual_precision": de.inspect_variant_precision(
                        package, profile, variant
                    ),
                    "source_manifest_sha256": de.file_sha256(
                        Path(value("--model-dir")) / "deployment_source.json"
                    ),
                },
            )
            if corrupt == variant:
                artifact.write_bytes(b"changed after export")

    return run


@pytest.mark.parametrize("profile", ["e2b", "270m"])
def test_read_only_plan_never_calls_training_or_inference(tmp_path, profile):
    options = inputs(tmp_path, profile)
    calls = []
    plan = ce.run_checkpoint_export(options, command_runner=fake_runner(calls))
    assert (
        plan["status"] == "plan_only" and not calls and not options.output_dir.exists()
    )
    assert plan["source_checkpoint"] == str(
        options.fit_dir / "training/best_golden_checkpoint"
    )
    assert [stage["name"] for stage in plan["stages"]] == [
        "exporter_preflight",
        "merge",
        "export_w32",
        "export_w16",
        "export_w8",
        "export_w4",
    ]
    assert all(
        Path(stage["command"][2]).name == "deployment_export.py"
        for stage in plan["stages"]
    )
    assert {stage["command"][3] for stage in plan["stages"]} == {
        "probe",
        "prepare",
        "convert",
    }
    assert (
        plan["runtime_evaluation_deferred"]
        and not plan["requires_merged_hf_and_real_litertlm_evaluation"]
    )
    assert not plan["official_retained_scale_export"] and not plan["mtp_exported"]


@pytest.mark.parametrize("profile", ["e2b", "270m"])
def test_exports_all_four_and_reports_no_quality_scores(tmp_path, capsys, profile):
    options = inputs(tmp_path, profile)
    calls = []
    state = ce.run_checkpoint_export(
        options, execute=True, command_runner=fake_runner(calls)
    )
    assert state["status"] == "exported_not_evaluated"
    assert (
        not state["training_executed"]
        and not state["quality_evaluation_performed"]
        and not state["runtime_gpu_tested"]
    )
    assert list(state["artifacts"]) == ["w32", "w16", "w8", "w4"]
    assert len(state["completed"]) == len(calls) == 6
    assert state["active_stage"] is None
    assert state == json.loads(
        (options.output_dir / "checkpoint_export_manifest.json").read_text()
    )
    for name, argv, env, kwargs in calls:
        assert env["CUDA_VISIBLE_DEVICES"] == "" and env["HF_HUB_OFFLINE"] == "1"
        assert kwargs == {"timeout_seconds": 172800, "progress_seconds": 10}
        assert (options.output_dir / "logs" / f"{name}.log").is_file()
    text = capsys.readouterr().out
    assert all(name in text for name in ("W32", "W16", "W8", "W4", "not evaluated"))
    assert not (options.output_dir / "tensorboard").exists()


@pytest.mark.parametrize("fail", ["exporter_preflight", "merge", "export_w16"])
def test_failure_stops_and_retains_completed_outputs(tmp_path, capsys, fail):
    options = inputs(tmp_path)
    calls = []
    with pytest.raises(RuntimeError, match="simulated failure"):
        ce.run_checkpoint_export(
            options, execute=True, command_runner=fake_runner(calls, fail=fail)
        )
    state = json.loads(
        (options.output_dir / "checkpoint_export_manifest.json").read_text()
    )
    assert state["status"] == "failed" and state["active_stage"] == fail
    assert calls[-1][0] == fail and fail not in state["completed"]
    if fail == "export_w16":
        assert list(state["artifacts"]) == ["w32"]
        assert Path(state["artifacts"]["w32"]["artifact"]).is_file()
        assert not (options.output_dir / "variants/w8").exists()
        assert "W16 | fp16 | - | failed" in capsys.readouterr().out


def test_tampered_export_cannot_be_recorded_as_success(tmp_path):
    options = inputs(tmp_path)
    calls = []
    with pytest.raises(ValueError, match="artifact path/hash"):
        ce.run_checkpoint_export(
            options, execute=True, command_runner=fake_runner(calls, corrupt="w8")
        )
    state = json.loads(
        (options.output_dir / "checkpoint_export_manifest.json").read_text()
    )
    assert state["status"] == "failed" and list(state["artifacts"]) == ["w32", "w16"]
    assert calls[-1][0] == "export_w8"


def test_child_success_without_evidence_is_rejected(tmp_path):
    options = inputs(tmp_path)
    with pytest.raises(FileNotFoundError):
        ce.run_checkpoint_export(
            options, execute=True, command_runner=lambda *a, **k: None
        )
    state = json.loads(
        (options.output_dir / "checkpoint_export_manifest.json").read_text()
    )
    assert state["status"] == "failed" and not state["completed"]


@pytest.mark.parametrize(
    "stage,field,value",
    [
        ("exporter_preflight", "status", "failed"),
        ("exporter_preflight", "profile", "270m"),
        ("merge", "source_checkpoint", "another checkpoint"),
        ("merge", "training_config_sha256", "changed"),
        ("merge", "merged_files", {}),
        ("merge", "official_retained_scale_export", True),
    ],
)
def test_mismatched_stage_evidence_stops_exports(tmp_path, stage, field, value):
    options = inputs(tmp_path)
    calls = []
    runner = fake_runner(calls)

    def invalid_report(argv, logfile, env, **kwargs):
        runner(argv, logfile, env, **kwargs)
        if logfile.stem == stage:
            report = options.output_dir / (
                "exporter_preflight.json"
                if stage == "exporter_preflight"
                else "merged_hf/deployment_source.json"
            )
            document = json.loads(report.read_text())
            document[field] = value
            write_json(report, document)

    with pytest.raises(ValueError, match="matching|dense deployment"):
        ce.run_checkpoint_export(options, execute=True, command_runner=invalid_report)
    assert calls[-1][0] == stage and not (options.output_dir / "variants").exists()


@pytest.mark.parametrize(
    "exception,status", [(KeyboardInterrupt, "interrupted"), (TimeoutError, "failed")]
)
def test_interruption_or_timeout_is_recorded(tmp_path, exception, status):
    options = inputs(tmp_path)

    def stop(*args, **kwargs):
        raise exception("test stop")

    with pytest.raises(exception):
        ce.run_checkpoint_export(options, execute=True, command_runner=stop)
    assert (
        json.loads(
            (options.output_dir / "checkpoint_export_manifest.json").read_text()
        )["status"]
        == status
    )


def test_experimental_acknowledgement_before_any_write(tmp_path):
    options = replace(inputs(tmp_path), allow_experimental_formats=False)
    assert ce.run_checkpoint_export(options)["status"] == "plan_only"
    with pytest.raises(ValueError, match="--allow-experimental-formats"):
        ce.run_checkpoint_export(options, execute=True)
    assert not options.output_dir.exists()


@pytest.mark.parametrize(
    "name",
    [
        "training_config.yaml",
        "preparation_report.json",
        "training/best_golden_checkpoint/training_metadata.json",
    ],
)
def test_missing_source_files_fail_before_output(tmp_path, name):
    options = inputs(tmp_path)
    (options.fit_dir / name).unlink()
    with pytest.raises(ValueError, match="missing"):
        ce.run_checkpoint_export(options)
    assert not options.output_dir.exists()


@pytest.mark.parametrize("value", [True, "seed_manifest.json"])
def test_rejects_qat_or_mobile_seed(tmp_path, value):
    options = inputs(tmp_path)
    if value is True:
        change_config(options, lambda c: c.update(qat={"enabled": True}))
    else:
        change_config(
            options, lambda c: c["model"].update(mobile_training_seed_manifest=value)
        )
    with pytest.raises(ValueError, match="not retained-scale/QAT"):
        ce.run_checkpoint_export(options)


def test_rejects_profile_mismatch_and_modified_config(tmp_path):
    options = inputs(tmp_path)
    with pytest.raises(ValueError, match="model_type"):
        ce.run_checkpoint_export(replace(options, profile="270m"))
    with (options.fit_dir / "training_config.yaml").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="original training_config"):
        ce.run_checkpoint_export(options)


@pytest.mark.parametrize(
    "field,value",
    [("checkpoint_step", 0), ("checkpoint_step", True), ("checkpoint_kind", "unknown")],
)
def test_rejects_incomplete_checkpoint_metadata(tmp_path, field, value):
    options = inputs(tmp_path)
    path = options.fit_dir / "training/best_golden_checkpoint/training_metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    write_json(path, metadata)
    with pytest.raises(ValueError, match="provenance|saved dense"):
        ce.run_checkpoint_export(options)


def test_original_report_binding_is_required(tmp_path):
    options = inputs(tmp_path)
    write_json(
        options.fit_dir / "preparation_report.json",
        {"training_config_sha256": "changed"},
    )
    with pytest.raises(ValueError, match="preparation report"):
        ce.run_checkpoint_export(options)


@pytest.mark.parametrize(
    "section,field", [("model", "model_source"), ("run", "dataset_dir")]
)
def test_original_model_and_data_must_remain_available(tmp_path, section, field):
    options = inputs(tmp_path)
    change_config(
        options, lambda c: c[section].update({field: str(tmp_path / "missing")})
    )
    with pytest.raises(ValueError, match=field):
        ce.run_checkpoint_export(options)


@pytest.mark.parametrize(
    "relative",
    [
        "trained run/fit/exports",
        "base model/exports",
        "prepared data/exports",
        "trained run",
    ],
)
def test_output_cannot_overlap_inputs(tmp_path, relative):
    options = replace(inputs(tmp_path), output_dir=tmp_path / relative)
    with pytest.raises(ValueError, match="must not overlap"):
        ce.run_checkpoint_export(options)


def test_existing_output_never_overwritten(tmp_path):
    options = inputs(tmp_path)
    options.output_dir.mkdir()
    marker = options.output_dir / "keep.txt"
    marker.write_text("user result")
    with pytest.raises(ValueError, match="NEW directory"):
        ce.run_checkpoint_export(options, execute=True)
    assert marker.read_text() == "user result"


@pytest.mark.parametrize(
    "field,value",
    [
        ("stage_timeout_seconds", 0),
        ("stage_timeout_seconds", float("nan")),
        ("progress_seconds", -1),
        ("progress_seconds", float("inf")),
    ],
)
def test_invalid_deadlines(tmp_path, field, value):
    with pytest.raises(ValueError, match="positive and finite"):
        ce.run_checkpoint_export(replace(inputs(tmp_path), **{field: value}))


def test_cache_capacity_uses_bound_training_budgets(tmp_path):
    options = inputs(tmp_path)
    change_config(options, lambda c: c["golden_eval"].update(max_new_tokens=5000))
    with pytest.raises(ValueError, match="cache_length"):
        ce.run_checkpoint_export(options)
    plan = ce.run_checkpoint_export(replace(options, cache_length=10000))
    assert plan["limits"] == {
        "cache_length": 10000,
        "max_input_tokens": 4096,
        "max_new_tokens": 5000,
    }


def test_explicit_checkpoint_and_interpreter_paths(tmp_path):
    options = inputs(tmp_path)
    final = options.fit_dir / "training/final_adapter"
    write_json(
        final / "training_metadata.json",
        json.loads(
            (
                options.fit_dir
                / "training/best_golden_checkpoint/training_metadata.json"
            ).read_text()
        ),
    )
    interpreter = tmp_path / "isolated venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    plan = ce.run_checkpoint_export(
        replace(options, checkpoint=final, exporter_python=interpreter)
    )
    assert plan["source_checkpoint"] == str(final)
    assert plan["probe_command"][0] == str(interpreter)
    assert plan["prepare_command"][0] == sys.executable
    with pytest.raises(ValueError, match="absolute Python"):
        ce.run_checkpoint_export(replace(options, exporter_python=Path("python")))


def test_cli_help_dry_run_and_failure_exit_codes(tmp_path):
    options = inputs(tmp_path)
    script = (
        Path(__file__).resolve().parents[1] / "scripts/export_checkpoint_litertlm.py"
    )
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert help_result.returncode == 0 and "--checkpoint" in help_result.stdout
    argv = [
        sys.executable,
        str(script),
        "--profile",
        "e2b",
        "--fit-dir",
        str(options.fit_dir),
        "--output-dir",
        str(options.output_dir),
        "--exporter-python",
        sys.executable,
    ]
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads(result.stdout)["status"] == "plan_only"
        and not options.output_dir.exists()
    )
    failed = subprocess.run(
        [*argv, "--execute"], capture_output=True, text=True, timeout=30, check=False
    )
    assert failed.returncode == 2 and "--allow-experimental-formats" in failed.stderr
    assert not options.output_dir.exists()
