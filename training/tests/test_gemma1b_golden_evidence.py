"""The standalone 1B pipeline cannot publish QAT-off evaluation as QAT-on."""
import json
from dataclasses import replace

import pytest
import test_golden_training_pipeline as fixtures
from ir_training.pipeline import golden_training as workflow
from test_gemma3_1b_qat_export import model_config

options = fixtures.options


def one_b_options(options):
    (options.model_dir / "config.json").write_text(json.dumps(model_config()), encoding="utf-8")
    return replace(options, profile="1b", max_input_tokens=5120)


def test_standalone_one_b_requires_qat_evidence(options, monkeypatch):
    options = one_b_options(options)
    with pytest.raises(ValueError, match="QAT-on evidence"):
        workflow.run_pipeline(options, execute=True, tokenizer_loader=lambda *_: fixtures.FixtureTokenizer(),
                              command_runner=fixtures.fake_runner(options, monkeypatch, []))
    assert not (options.output_dir / "evaluation_scorecard.json").exists()


def test_standalone_one_b_preserves_qat_evidence_on_resume(options, monkeypatch):
    options = one_b_options(options)
    calls = []
    fake = fixtures.fake_runner(options, monkeypatch, calls)

    def run(argv, logfile, environment):
        fake(argv, logfile, environment)
        if "--evaluation-name" in argv:
            from pathlib import Path
            output = Path(argv[argv.index("--output-dir") + 1]) / "evaluation_result.json"
            value = json.loads(output.read_text(encoding="utf-8"))
            value["qat_applied"] = True
            output.write_text(json.dumps(value), encoding="utf-8")

    result = workflow.run_pipeline(options, execute=True, tokenizer_loader=lambda *_: fixtures.FixtureTokenizer(), command_runner=run)
    assert result["status"] == "complete"
    before = len(calls)
    continued = workflow.run_pipeline(options, execute=True, continue_run=True, command_runner=run)
    assert continued["status"] == "complete" and len(calls) == before
    # Reproduce a historical receipt with a correctly hashed but QAT-off result;
    # the dedicated evidence gate must catch it, not just the generic file hash.
    name = next(path for path in result["completed"]["best_golden32"]["files"] if path.endswith("evaluation_result.json"))
    from pathlib import Path
    path = Path(name)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["qat_applied"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    result["completed"]["best_golden32"]["files"][name] = workflow.sha256(path)
    (options.output_dir / "pipeline_manifest.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="QAT-on evidence"):
        workflow.run_pipeline(options, execute=True, continue_run=True, command_runner=run)


def test_one_b_rejects_training_context_beyond_architecture(options):
    with pytest.raises(ValueError, match="Training sequence budget"):
        workflow.build_plan(replace(one_b_options(options), max_seq_length=32769))
