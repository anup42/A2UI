"""DPO-only pilot schedules preserve SFT provenance and all legacy defaults."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import test_qat_posttrain_pipeline as shared
from test_qat_posttrain_pipeline import (
    cli_module,
    grpo_continuation,
    preference_file,
    prepared,
    profile,
    workflow,
    write,
)

options = shared.options

DPO_SCHEDULE_FIELDS = (
    "dpo_learning_rate", "dpo_max_steps", "dpo_warmup_steps",
    "dpo_warmup_ratio", "dpo_golden_every_steps",
)


def source_with_warmup(options, **warmup):
    source = json.loads(options.sft_config.read_text(encoding="utf-8"))
    source["training"].update(warmup)
    write(options.sft_config, source)
    metadata = options.sft_checkpoint / "training_metadata.json"
    write(metadata, {"training_config_sha256": workflow.sha256(options.sft_config)})
    return options


def dpo_options(options, **changes):
    return replace(options, objective="dpo", dpo_preferences=preference_file(options), **changes)


@pytest.mark.parametrize("objective", ["grpo", "grpo_sft", "dpo"])
def test_absent_dpo_schedule_fields_preserve_legacy_config_exactly(options, objective):
    source_with_warmup(options, warmup_steps=500, warmup_ratio=0.1)
    current = workflow.build_plan(replace(
        options, objective=objective,
        dpo_preferences=preference_file(options) if objective == "dpo" else None,
    ))
    legacy = copy.deepcopy(current)
    for key in DPO_SCHEDULE_FIELDS:
        legacy["options"].pop(key)
    current_config = workflow.training_config(current, profile(), prepared())
    assert current_config == workflow.training_config(legacy, profile(), prepared())
    assert current_config["training"]["warmup_steps"] == 500
    assert current_config["training"]["warmup_ratio"] == 0.1


def test_pilot_overrides_are_resolved_without_changing_sft_or_other_contracts(options):
    source_with_warmup(options, warmup_steps=500, warmup_ratio=0.1)
    before = options.sft_config.read_bytes()
    original = workflow.build_plan(dpo_options(options))
    lr, steps = (1e-6, 100) if options.family == "e2b" else (2e-6, 200)
    changed = workflow.build_plan(dpo_options(
        options, dpo_learning_rate=lr, dpo_max_steps=steps,
        dpo_warmup_ratio=0.03, dpo_golden_every_steps=25,
    ))
    old_config = workflow.training_config(original, profile(), prepared())
    config = workflow.training_config(changed, profile(), prepared())
    assert config["training"]["learning_rate"] == lr
    assert config["training"]["max_steps"] == steps
    for key in ("learning_rate", "max_steps", "golden_every_steps"):
        assert changed["options"][key] == original["options"][key]
    assert changed["dpo_training_overrides"] == {
        "learning_rate": lr, "max_steps": steps, "golden_every_steps": 25,
        "warmup_steps": 3 if options.family == "e2b" else 6,
    }
    assert changed["options"]["dpo_warmup_ratio"] == 0.03
    assert config["training"]["warmup_steps"] == (3 if options.family == "e2b" else 6)
    assert "warmup_ratio" not in config["training"]
    assert config["training"]["eval_steps"] == config["training"]["save_steps"] == 25
    expected_training = copy.deepcopy(old_config["training"])
    expected_training.pop("warmup_ratio")
    expected_training.update(learning_rate=lr, max_steps=steps, warmup_steps=int(steps * 0.03),
                             eval_steps=25, save_steps=25)
    assert config["training"] == expected_training
    for key in config.keys() - {"training"}:
        assert config[key] == old_config[key]
    for key in ("source_config_sha256", "source_adapter_files", "stages", "selection", "mtp",
                "preference_files", "official_artifact_sha256", "format"):
        assert changed[key] == original[key]
    assert options.sft_config.read_bytes() == before
    assert changed["source_config_sha256"] == workflow.sha256(options.sft_config)
    assert not options.output_dir.exists()


@pytest.mark.parametrize("override,expected", [
    ({"dpo_warmup_steps": 0}, 0),
    ({"dpo_warmup_steps": 4}, 4),
    ({"dpo_warmup_ratio": 0.0}, 0),
    ({"dpo_warmup_ratio": 0.015}, 2),
])
def test_explicit_warmup_clears_inherited_schedule_and_rounds_up(options, override, expected):
    source_with_warmup(options, warmup_steps=500, warmup_ratio=0.1)
    plan = workflow.build_plan(dpo_options(options, dpo_max_steps=100, **override))
    config = workflow.training_config(plan, profile(), prepared())
    assert config["training"]["warmup_steps"] == expected
    assert "warmup_ratio" not in config["training"]


@pytest.mark.parametrize("changes", [
    {"dpo_learning_rate": 0}, {"dpo_learning_rate": -1}, {"dpo_learning_rate": True},
    {"dpo_learning_rate": float("nan")}, {"dpo_learning_rate": float("inf")},
    {"dpo_max_steps": 19}, {"dpo_max_steps": True}, {"dpo_max_steps": 100.5},
    {"dpo_golden_every_steps": 19}, {"dpo_golden_every_steps": True},
    {"dpo_golden_every_steps": 25.5},
    {"dpo_warmup_steps": -1}, {"dpo_warmup_steps": True}, {"dpo_warmup_steps": 1.5},
    {"dpo_warmup_steps": 100, "dpo_max_steps": 100},
    {"dpo_warmup_ratio": -0.1}, {"dpo_warmup_ratio": True},
    {"dpo_warmup_ratio": float("nan")}, {"dpo_warmup_ratio": float("inf")},
    {"dpo_warmup_ratio": 1.0}, {"dpo_warmup_ratio": 0.999, "dpo_max_steps": 100},
    {"dpo_warmup_ratio": 0.03, "dpo_warmup_steps": 0},
])
def test_invalid_dpo_schedule_fails_before_outputs(options, changes):
    with pytest.raises(ValueError):
        workflow.build_plan(dpo_options(options, **changes))
    assert not options.output_dir.exists()


@pytest.mark.parametrize("objective", ["grpo", "grpo_sft"])
@pytest.mark.parametrize("key,value", [
    ("dpo_learning_rate", 1e-6), ("dpo_max_steps", 100),
    ("dpo_warmup_steps", 0), ("dpo_warmup_ratio", 0.03),
    ("dpo_golden_every_steps", 25),
])
def test_dpo_schedule_cannot_change_other_objectives(options, objective, key, value):
    with pytest.raises(ValueError, match="dpo"):
        workflow.build_plan(replace(options, objective=objective, **{key: value}))


def capture_cli_plan(monkeypatch, previous):
    cli = cli_module()
    monkeypatch.setattr(grpo_continuation, "options_from_previous_run", lambda *args: previous)
    captured = []
    monkeypatch.setattr(cli, "run_pipeline", lambda opt, **kwargs: captured.append(workflow.build_plan(opt)) or {})
    return cli, captured


def test_rerun_cli_applies_pilot_overrides_and_keeps_original_sft_identity(options, monkeypatch):
    source_with_warmup(options, warmup_steps=500)
    original = replace(options, learning_rate=5e-7, max_steps=400, seed=17)
    cli, captured = capture_cli_plan(monkeypatch, original)
    lr, steps = ("1e-6", "100") if options.family == "e2b" else ("2e-6", "200")
    assert cli.main([
        "--rerun-from", "old", "--output-dir", str(options.output_dir),
        "--objective", "dpo", "--dpo-preferences", str(preference_file(options)),
        "--dpo-learning-rate", lr, "--dpo-max-steps", steps,
        "--dpo-warmup-ratio", "0.03", "--dpo-golden-every-steps", "25",
    ]) == 0
    values = captured[0]["options"]
    assert (values["learning_rate"], values["max_steps"], values["golden_every_steps"]) == (
        original.learning_rate, original.max_steps, original.golden_every_steps,
    )
    config = workflow.training_config(captured[0], profile(), prepared())
    assert (config["training"]["learning_rate"], config["training"]["max_steps"],
            config["training"]["eval_steps"]) == (float(lr), int(steps), 25)
    assert values["sft_checkpoint"] == str(original.sft_checkpoint)
    assert values["sft_config"] == str(original.sft_config)
    assert values["seed"] == 17 and values["effective_batch"] == original.effective_batch


@pytest.mark.parametrize("initial,flag,value,expected_steps", [
    ({"dpo_warmup_steps": 4}, "--dpo-warmup-ratio", "0.03", 3),
    ({"dpo_warmup_ratio": 0.03}, "--dpo-warmup-steps", "0", 0),
])
def test_rerun_can_replace_the_other_inherited_warmup_type(options, monkeypatch, initial, flag, value, expected_steps):
    previous = dpo_options(options, dpo_max_steps=100, **initial)
    cli, captured = capture_cli_plan(monkeypatch, previous)
    assert cli.main(["--rerun-from", "old", "--output-dir", str(options.output_dir), flag, value]) == 0
    actual = captured[0]["options"]
    assert actual["dpo_warmup_steps" if flag.endswith("ratio") else "dpo_warmup_ratio"] is None
    config = workflow.training_config(captured[0], profile(), prepared())
    assert config["training"]["warmup_steps"] == expected_steps


def test_rerun_still_rejects_generic_hyperparameter_flags(options, monkeypatch):
    cli, captured = capture_cli_plan(monkeypatch, options)
    for flag, value in (("--learning-rate", "1e-6"), ("--max-steps", "100"), ("--golden-every-steps", "25")):
        with pytest.raises(SystemExit):
            cli.main(["--rerun-from", "old", "--output-dir", str(options.output_dir), flag, value])
    assert captured == []


def test_rerun_rejects_both_explicit_warmup_types(options, monkeypatch):
    cli, captured = capture_cli_plan(monkeypatch, dpo_options(options))
    with pytest.raises((SystemExit, ValueError)):
        cli.main(["--rerun-from", "old", "--output-dir", str(options.output_dir),
                  "--dpo-warmup-steps", "0", "--dpo-warmup-ratio", "0.03"])
    assert captured == []


@pytest.mark.parametrize("flag,value", [
    ("--dpo-learning-rate", "1e-6"), ("--dpo-max-steps", "100"),
    ("--dpo-warmup-steps", "0"), ("--dpo-warmup-ratio", "0.03"),
    ("--dpo-golden-every-steps", "25"),
])
def test_continue_run_remains_immutable(options, monkeypatch, flag, value):
    cli, captured = capture_cli_plan(monkeypatch, options)
    with pytest.raises(SystemExit):
        cli.main(["--continue-run", "old", flag, value])
    assert captured == []


def test_switching_from_dpo_clears_all_schedule_overrides(options, monkeypatch):
    source_with_warmup(options, warmup_steps=500)
    previous = dpo_options(options, learning_rate=5e-7, max_steps=400,
                           dpo_learning_rate=2e-6, dpo_max_steps=100,
                           dpo_warmup_ratio=0.03, dpo_golden_every_steps=25)
    saved = workflow.build_plan(previous)
    write(previous.output_dir / workflow.PLAN_NAME, saved)
    previous = grpo_continuation.options_from_previous_run(
        previous.output_dir, previous.output_dir.parent / "fresh",
    )
    cli, captured = capture_cli_plan(monkeypatch, previous)
    cli.main(["--rerun-from", "old", "--output-dir", str(options.output_dir), "--objective", "grpo"])
    values = captured[0]["options"]
    assert all(values[key] is None for key in DPO_SCHEDULE_FIELDS)
    assert values["learning_rate"] == 5e-7 and values["max_steps"] == 400
    assert "dpo_training_overrides" not in captured[0]
    config = workflow.training_config(captured[0], profile(), prepared())
    assert config["training"]["learning_rate"] == 5e-7
    assert config["training"]["max_steps"] == 400
    assert config["training"]["warmup_steps"] == 500
    assert config["training"]["eval_steps"] == options.golden_every_steps


def test_direct_dpo_cli_also_accepts_explicit_pilot_schedule(options, monkeypatch):
    cli, captured = capture_cli_plan(monkeypatch, options)
    args = ["--family", options.family, "--objective", "dpo",
            "--dpo-preferences", str(preference_file(options)), "--dpo-learning-rate", "1e-6",
            "--dpo-max-steps", "100", "--dpo-warmup-steps", "3", "--dpo-golden-every-steps", "25"]
    for field in ("model_dir", "sft_config", "sft_checkpoint", "output_dir", "official_litertlm",
                  "exporter_python", "input_dir", "source_safetensors"):
        args += [f"--{field.replace('_', '-')}", str(getattr(options, field))]
    assert cli.main(args) == 0
    config = workflow.training_config(captured[0], profile(), prepared())
    assert config["training"]["max_steps"] == 100
    assert config["training"]["warmup_steps"] == 3


def test_saved_dpo_plan_round_trips_requested_schedule(options):
    original = dpo_options(options, dpo_learning_rate=2e-6, dpo_max_steps=100,
                           dpo_warmup_ratio=0.03, dpo_golden_every_steps=25)
    plan = workflow.build_plan(original)
    write(original.output_dir / workflow.PLAN_NAME, plan)
    replay = grpo_continuation.options_from_previous_run(original.output_dir, original.output_dir.parent / "fresh")
    assert replay.learning_rate == options.learning_rate and replay.max_steps == options.max_steps
    assert replay.dpo_learning_rate == 2e-6 and replay.dpo_max_steps == 100
    assert replay.dpo_warmup_ratio == 0.03 and replay.dpo_golden_every_steps == 25


@pytest.mark.parametrize("tampering", ["different", "missing"])
def test_stored_resolved_schedule_must_match_requested_overrides(options, tampering):
    plan = workflow.build_plan(dpo_options(options, dpo_max_steps=100, dpo_warmup_ratio=0.03))
    if tampering == "different":
        plan["dpo_training_overrides"]["warmup_steps"] = 99
    else:
        plan.pop("dpo_training_overrides")
    with pytest.raises(ValueError):
        workflow.training_config(plan, profile(), prepared())
