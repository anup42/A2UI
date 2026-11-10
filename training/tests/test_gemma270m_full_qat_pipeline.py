"""Exercise the dedicated 270M launcher without training or converting a model."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cli(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run_gemma3_270m_full_qat_pipeline.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("gemma270m_full_qat_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def invocation(tmp_path):
    model = tmp_path / "sft-checkpoint"
    inputs = tmp_path / "dataset"
    output = tmp_path / "qat-run"
    model.mkdir()
    inputs.mkdir()
    write_json(model / "config.json", {
        "model_type": "gemma3_text", "hidden_size": 640, "num_hidden_layers": 18,
    })
    write_json(model / "tokenizer_config.json", {})
    (model / "model.safetensors").write_bytes(b"fixture weights never loaded")
    for split in ("train", "val"):
        (inputs / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
    return output, [
        "--model-dir", str(model), "--input-dir", str(inputs), "--output-dir", str(output),
        "--exporter-python", sys.executable,
    ]


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def argument(command, flag):
    return command[command.index(flag) + 1]


def mock_probe(command):
    report = Path(argument(command, "--report"))
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, {"status": "passed", "profile": "270m", "variants": ["w8"]})


def test_default_command_plans_full_qat_and_holdouts_without_side_effects(
    cli, invocation, monkeypatch, capsys,
):
    output, argv = invocation

    def unexpected(*args, **kwargs):
        pytest.fail("planning must neither launch training nor start an exporter")

    monkeypatch.setattr(cli, "run_pipeline", unexpected)
    monkeypatch.setattr(cli, "run_bounded_command", unexpected)

    assert cli.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "plan_only"
    assert Path(report["artifact"]) == output / "litertlm_w8/model.litertlm"
    assert not output.exists()
    training = report["training"]
    options = training["options"]
    assert options["profile"] == "270m" and options["qat"] is True
    assert options["evaluate_golden35"] is True and options["evaluate_bixby50"] is True
    assert options["max_seq_length"] == options["max_input_tokens"] == 5120
    assert options["max_new_tokens"] == 2048
    assert options["epochs"] == 1 and options["learning_rate"] == 5e-6
    assert options["eval_steps"] == options["golden_every_steps"] == 100
    assert {key: value["rows"] for key, value in training["goldens"].items()} == {
        "golden32": 32, "golden35": 35, "bixby50": 50,
    }


@pytest.mark.parametrize("resume", [False, True])
def test_cpu_probe_precedes_training_and_export_follows_complete_evaluation(
    cli, invocation, monkeypatch, resume, capsys,
):
    output, argv = invocation
    calls = []
    if resume:
        output.mkdir()
        write_json(output / "pipeline_manifest.json", {})

    def run_command(command, log, environment, **kwargs):
        assert Path(command[0]) == Path(sys.executable)
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"
        if "--report" in command:
            assert calls == []
            assert Path(argument(command, "--model-dir")) == Path(argument(argv, "--model-dir"))
            calls.append("probe")
            mock_probe(command)
            return
        assert calls == ["probe", "train"]
        assert "--execute" in command
        assert Path(argument(command, "--run-dir")) == output
        calls.append("export")
        export = output / "litertlm_w8"
        export.mkdir()
        artifact = export / "model.litertlm"
        data = b"fixture package"
        artifact.write_bytes(data)
        write_json(export / "w8_export_report.json", {
            "status": "exported", "artifact": str(artifact),
            "artifact_sha256": hashlib.sha256(data).hexdigest(),
            "physical_weight_precision": {"verified": True},
        })

    def run_training(options, *, execute, continue_run):
        assert calls == ["probe"]
        assert execute is True
        assert continue_run is resume
        assert options.output_dir == output and options.qat is True
        output.mkdir(exist_ok=True)
        calls.append("train")
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run_bounded_command", run_command)
    monkeypatch.setattr(cli, "run_pipeline", run_training)
    launch = [*argv, "--execute", *(["--continue-run"] if resume else [])]

    assert cli.main(launch) == 0
    assert calls == ["probe", "train", "export"]
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "complete"
    assert Path(report["evaluation_scorecard"]) == output / "evaluation_scorecard.json"
    assert Path(report["artifact"]) == output / "litertlm_w8/model.litertlm"


@pytest.mark.parametrize("probe_error", [True, False])
def test_pipeline_stops_before_next_stage_on_failure(cli, invocation, monkeypatch, probe_error, capsys):
    _, argv = invocation
    calls = []

    def run_command(command, log, environment, **kwargs):
        assert "--report" in command
        calls.append("probe")
        if probe_error:
            raise RuntimeError("exporter unavailable")
        mock_probe(command)

    def run_training(*args, **kwargs):
        calls.append("train")
        return {"status": "failed"}

    monkeypatch.setattr(cli, "run_bounded_command", run_command)
    monkeypatch.setattr(cli, "run_pipeline", run_training)

    assert cli.main([*argv, "--execute"]) != 0
    assert calls == (["probe"] if probe_error else ["probe", "train"])
    if probe_error:
        assert "exporter unavailable" in capsys.readouterr().err
