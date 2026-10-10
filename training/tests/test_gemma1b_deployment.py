"""CPU-only 1B orchestration fixtures; never convert, train, or run a GPU model."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import test_golden_deployment as fixtures
from ir_training.pipeline import golden_deployment as deployment

options = fixtures.options
export_validator = fixtures.export_validator


def one_b_options(options):
    fixtures.dump(options.base.model_dir / "config.json", {
        "model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"],
        "hidden_size": 1152, "intermediate_size": 6912, "num_hidden_layers": 26,
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
        "vocab_size": 262144, "tie_word_embeddings": True,
    })
    return replace(options, base=replace(options.base, profile="1b", max_input_tokens=5120),
                   allow_experimental_formats=False)


def test_one_b_plan_is_full_qat_w8_only_and_read_only(options):
    options = one_b_options(options)
    plan = deployment.run_deployment(options)
    assert plan["status"] == "plan_only"
    assert plan["workflow"] == "gemma3_1b_full_qat_w8_deployment_v1"
    assert list(plan["export"]["variants"]) == ["w8"]
    assert plan["training"]["options"]["qat"] is True
    assert plan["training"]["options"]["max_seq_length"] == 4096
    assert plan["training"]["options"]["max_input_tokens"] == 5120
    assert not options.base.output_dir.exists()
    assert not plan["mtp_exported"]


def test_one_b_disallows_unreviewed_dense_tuning(options):
    with pytest.raises(ValueError, match="screening workflow"):
        deployment.build_deployment_plan(replace(one_b_options(options), tune=True))


@pytest.mark.parametrize("skip", [False, True])
def test_one_b_executes_shared_checkpoint_export_and_three_native_cohorts(
    options, export_validator, monkeypatch, skip,
):
    options = replace(one_b_options(options), skip_litert_evaluation=skip)
    original_evaluation = fixtures.evaluation

    def qat_evaluation(path, count, artifact, **kwargs):
        result = original_evaluation(path, count, artifact, **kwargs)
        if not kwargs.get("litert"):
            result["qat_applied"] = True
            fixtures.dump(path / "evaluation_result.json", result)
        return result

    monkeypatch.setattr(fixtures, "evaluation", qat_evaluation)
    calls = []
    result = deployment.run_deployment(
        options, execute=True, command_runner=fixtures.mock_runner(calls),
        pipeline_runner=fixtures.fake_pipeline, writer_factory=fixtures.Writer,
        gpu_probe=fixtures.inventory,
    )
    assert result["status"] == "complete"
    assert set(result["exports"]) == {"w8"}
    assert len(result["results"]) == (9 if skip else 12)
    native_calls = [argv for argv, _, _ in calls if "--builtin-gpu" in argv]
    assert len(native_calls) == (0 if skip else 3)
    for argv in native_calls:
        assert argv[argv.index("--max-input-tokens") + 1] == "5120"
    conversions = [argv for argv, _, _ in calls if "convert" in argv]
    assert len(conversions) == 1
    assert conversions[0][conversions[0].index("--variant") + 1] == "w8"
    # Recovery verifies existing results and never restarts training/conversion.
    before = len(calls)
    resumed = deployment.run_deployment(
        replace(options, resume_run=True), execute=True,
        command_runner=fixtures.mock_runner(calls), pipeline_runner=fixtures.fake_pipeline,
        writer_factory=fixtures.Writer, gpu_probe=fixtures.inventory,
    )
    assert resumed["status"] == "complete" and len(calls) == before


@pytest.mark.parametrize("flag", [None, False])
def test_one_b_requires_qat_evaluation_evidence(tmp_path, flag):
    path = tmp_path / "evaluation"
    artifact = tmp_path / "checkpoint"
    result = fixtures.evaluation(path, 32, artifact)
    if flag is not None:
        result["qat_applied"] = flag
        fixtures.dump(path / "evaluation_result.json", result)
    with pytest.raises(ValueError, match="must apply QAT"):
        deployment._evaluation(path, 32, artifact=artifact, required_qat=True)
    # The new evidence requirement does not change dense/legacy validation.
    deployment._evaluation(path, 32, artifact=artifact)


def test_one_b_entrypoint_defaults_and_rejects_other_profiles(options):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_golden_deployment as cli
    argv = ["--model-dir", str(options.base.model_dir), "--output-dir", str(options.base.output_dir),
            "--exporter-python", sys.executable, "--skip-litert-evaluation"]
    parsed = cli.build_deployment_parser(required_profile="1b").parse_args(argv)
    assert parsed.profile == "1b"
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, "--profile", "270m"], required_profile="1b")
    assert exc.value.code == 2


def test_one_b_resume_rejects_missing_qat_evidence(options, export_validator, monkeypatch):
    # Missing evidence stops the run before preparing any deployment checkpoint.
    options = one_b_options(options)
    calls = []
    with pytest.raises(ValueError, match="must apply QAT"):
        deployment.run_deployment(
            options, execute=True, command_runner=fixtures.mock_runner(calls),
            pipeline_runner=fixtures.fake_pipeline, writer_factory=fixtures.Writer,
            gpu_probe=fixtures.inventory,
        )
    assert not any("prepare" in argv or "convert" in argv for argv, _, _ in calls)
    state = json.loads((options.base.output_dir / "deployment_manifest.json").read_text())
    assert state["status"] == "failed"
