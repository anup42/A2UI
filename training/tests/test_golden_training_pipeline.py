"""End-to-end orchestration fixtures: real CPU data, never real model training."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.common.jsonl import write_jsonl
from ir_training.data.chat_templates import build_messages, build_prompt
from ir_training.eval.golden_set import load_fixed_golden_rows
from ir_training.pipeline import golden_training as workflow
from ir_training.pipeline.golden_training import GoldenTrainingOptions


class FixtureTokenizer:
    name_or_path = "fixture-tokenizer-not-a-real-model"
    chat_template = "fixture-role-prefix"
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def get_vocab(self):
        return {"fixture": 0}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        return "".join(f"{item['role']}:\n{item['content']}\n" for item in messages) + ("assistant:\n" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": list(range(len(text.split())))}


def row(identity, source=None):
    source = source or f"Independent fixture response for {identity}"
    completion = f'<a2ui>\nroot=Text("{identity}")\n</a2ui>'
    return {"id": identity, "source_id": identity, "response_id": f"response-{identity}",
            "response_text": source, "completion": completion, "metadata": {"query_id": identity},
            "messages": build_messages("Historical short fixture prompt", source, completion, target_format="a2ui_express_v1"),
            "prompt": build_prompt("Historical short fixture prompt", source, target_format="a2ui_express_v1"),
            "intent_bucket": "fixture"}


@pytest.fixture
def options(tmp_path):
    model, inputs = tmp_path / "model", tmp_path / "inputs"
    model.mkdir()
    inputs.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture only, never loaded")
    for split in ("train", "val"):
        write_jsonl(inputs / f"{split}.jsonl", [row(f"{split}-{index}") for index in range(2)])
    return GoldenTrainingOptions(model, tmp_path / "unique-experiment", input_dir=inputs, steps=20, prepare_workers=1)


def prepare(options):
    return workflow.run_pipeline(options, prepare_only=True, tokenizer_loader=lambda *_: FixtureTokenizer())


def test_plan_has_no_runtime_or_output_side_effects(options, monkeypatch):
    monkeypatch.setattr(workflow, "_load_tokenizer", lambda *a: pytest.fail("must not load tokenizer"))
    result = workflow.run_pipeline(options)
    assert result["status"] == "plan_only"
    assert not options.output_dir.exists()
    assert result["goldens"]["golden32"]["rows"] == 32
    assert result["goldens"]["golden35"]["rows"] == 35
    command = workflow.configure_command(result)
    assert command[command.index("--run-id") + 1] == options.output_dir.name
    assert "--golden35-file" in command and "--devices" in command
    assert result["options"]["token_cache"] is True
    expected_cache = options.output_dir.parent / ".golden-preparation-cache"
    assert Path(result["options"]["preparation_cache_dir"]) == expected_cache
    assert Path(command[command.index("--token-cache-dir") + 1]) == expected_cache / "tokens"
    assert "--token-cache" in command
    assert not expected_cache.exists()


def test_token_cache_flags_are_bound_and_forwarded(options, tmp_path):
    changed = replace(options, token_cache=False, token_cache_dir=tmp_path / "persistent-tokens")
    plan = workflow.build_plan(changed)
    command = workflow.configure_command(plan)
    assert plan["options"]["token_cache"] is False
    assert "--no-token-cache" in command and "--token-cache" not in command
    assert command[command.index("--token-cache-dir") + 1] == str(changed.token_cache_dir)
    shared = workflow.build_plan(replace(options, preparation_cache_dir=tmp_path / "persistent-data"))
    assert Path(shared["options"]["token_cache_dir"]) == tmp_path / "persistent-data/tokens"


@pytest.mark.parametrize("name", ["preparation_cache_dir", "token_cache_dir"])
@pytest.mark.parametrize("location", ["model", "source", "output", "parent"])
def test_cache_locations_cannot_overlap_protected_directories(options, name, location):
    protected = {"model": options.model_dir, "source": options.input_dir,
                 "output": options.output_dir, "parent": options.output_dir.parent}[location]
    cache = protected if location == "parent" else protected / "cache"
    with pytest.raises(ValueError, match="must be outside"):
        workflow.build_plan(replace(options, **{name: cache}))
    assert not options.output_dir.exists()


@pytest.mark.parametrize("script_name", ["run_golden_training", "run_golden_experiments"])
@pytest.mark.parametrize("enabled", [True, False])
def test_cli_token_cache_switches_are_parseable(options, monkeypatch, capsys, script_name, enabled):
    spec = importlib.util.spec_from_file_location(f"fixture_{script_name}", ROOT / "scripts" / f"{script_name}.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    command = [script_name, "--model-dir", str(options.model_dir), "--input-dir", str(options.input_dir),
               "--output-dir", str(options.output_dir), "--token-cache-dir", str(options.output_dir.parent / "tokens")]
    if not enabled:
        command.append("--no-token-cache")
    if script_name.endswith("experiments"):
        command.extend(["--trial-steps", "20"])
    monkeypatch.setattr(sys, "argv", command)
    assert script.main() == 0
    plan = json.loads(capsys.readouterr().out)
    values = plan["options"] if "options" in plan else plan["trials"][0]["plan"]["options"]
    assert values["token_cache"] is enabled
    assert Path(values["token_cache_dir"]) == options.output_dir.parent / "tokens"


def test_prepare_only_real_benchmarks_share_one_prompt_and_keep_all_members(options):
    state = prepare(options)
    assert state["status"] == "prepared" and list(state["completed"]) == ["prepare"]
    assert not (options.output_dir / "fit").exists()
    manifest = json.loads((options.output_dir / "prepared/manifest.json").read_text(encoding="utf-8"))
    assert manifest["scaffold_count"] == 1
    assert manifest["shared_prompt"] == state["plan"]["shared_prompt"]
    for cohort, count in (("golden32", 32), ("golden35", 35)):
        rows = load_fixed_golden_rows(options.output_dir / f"prepared/{cohort}.jsonl", required_rows=count)
        assert len(rows) == count
        assert manifest["splits"][cohort]["quarantined_rows"] == 0


def test_query_and_response_grouping_is_transitive_and_source_preserving():
    rows = [row(f"case-{index}") for index in range(30)]
    rows[1]["source_id"] = rows[0]["source_id"]
    rows[2] = row("case-2", rows[1]["response_text"])
    before = json.dumps(rows, sort_keys=True)
    result = workflow._group_split(rows, seed=42)
    assert json.dumps(rows, sort_keys=True) == before
    assert sum(map(len, result.values())) == 30
    memberships = [name for name, values in result.items() if any(value["id"] == "case-0" for value in values)]
    grouped = result[memberships[0]]
    assert {"case-0", "case-1", "case-2"} <= {value["id"] for value in grouped}
    assert result["val"]


def test_preparation_filters_reserved_sources_from_both_input_splits(options):
    for split, index in (("train", 0), ("val", 1)):
        gold_path = ROOT / workflow.GOLDENS["golden35"][0].removeprefix("training/")
        gold = load_fixed_golden_rows(gold_path, required_rows=35)[index]
        write_jsonl(options.input_dir / f"{split}.jsonl", [row(f"{split}-clean"), gold])
    prepare(options)
    audit = json.loads((options.output_dir / "data_audit.json").read_text(encoding="utf-8"))
    for split in ("train", "val"):
        assert audit["filtering"][split]["quarantine_reasons"]["reserved_evaluation_source"] == 1
        assert audit["prepared_counts"][split]["accepted_rows"] == 1


def test_existing_outputs_and_source_drift_are_rejected(options):
    prepare(options)
    with pytest.raises(FileExistsError):
        prepare(options)
    write_jsonl(options.input_dir / "train.jsonl", [row("changed-source")])
    with pytest.raises(ValueError, match="artifact changed"):
        workflow.run_pipeline(options, prepare_only=True, continue_run=True)


def test_raw_stage3_source_is_materialized_filtered_and_group_split(options, tmp_path):
    source = tmp_path / "stage3"
    source.mkdir()
    records = [{"ui_id": f"ui-{index}", "query_id": f"query-{index}", "response_id": f"response-{index}",
                "source_format": "a2ui_express_v1", "completion": row(f"raw-{index}")["completion"]}
               for index in range(20)]
    responses = [{"response_id": item["response_id"], "query_id": item["query_id"],
                  "response_text": f"Independent raw source number {index}", "intent_bucket": "fixture"}
                 for index, item in enumerate(records)]
    write_jsonl(source / "genui.jsonl", records)
    write_jsonl(source / "responses.jsonl", responses)
    result = prepare(replace(options, input_dir=None, source_run_dir=source))
    assert result["status"] == "prepared"
    audit = json.loads((options.output_dir / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["source_materialization"] is not None
    assert sum(audit["prepared_counts"][name]["accepted_rows"] for name in ("train", "val")) == 20
    assert audit["prepared_counts"]["val"]["accepted_rows"] > 0


def test_changed_raw_benchmark_sidecar_is_rejected_before_tokenizer(options, tmp_path):
    plan = workflow.build_plan(options)
    sidecar = tmp_path / "changed_manifest.json"
    sidecar.write_text("{}", encoding="utf-8")
    plan["goldens"]["golden35"]["benchmark_manifest_path"] = str(sidecar)
    with pytest.raises(ValueError, match="benchmark manifest differs"):
        workflow.prepare_data(plan, tokenizer_loader=lambda *_: pytest.fail("must reject before tokenization"))


def test_atomic_state_publish_failure_preserves_previous_record(tmp_path, monkeypatch):
    record = tmp_path / "pipeline_manifest.json"
    workflow._write(record, {"status": "prepared"})
    original = record.read_bytes()
    def fail_replace(*args):
        raise OSError("fixture disk publication failure")
    monkeypatch.setattr(workflow.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        workflow._write(record, {"status": "complete"})
    assert record.read_bytes() == original
    assert list(tmp_path.iterdir()) == [record]


def test_missing_stage_outputs_cannot_be_marked_complete(tmp_path):
    with pytest.raises(FileNotFoundError, match="required output"):
        workflow._bindings([tmp_path / "evaluation_result.json"])


def _script_module(name):
    spec = importlib.util.spec_from_file_location(f"workflow_fixture_{name}", ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_runner(options, monkeypatch, calls, *, fail_evaluation=None, fail_training=False):
    def run(command, log, environment):
        calls.append(command)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Fully mocked GPU stage; no model was loaded.\n", encoding="utf-8")
        if "prepare_review_training.py" in command[1]:
            inventory = {"version": 1, "inherited_cuda_visible_devices": None, "visible_gpu_count": 2,
                         "devices": [{"visible_index": index, "launch_identifier": str(index), "uuid": f"GPU-{index}",
                                      "name": "NVIDIA H100 80GB", "total_memory_bytes": 80 * 1024 ** 3, "compute_capability": [9, 0]} for index in range(2)]}
            with monkeypatch.context() as context:
                context.setattr("ir_training.train.gpu_profile.detect_cuda_devices", lambda: inventory)
                context.setattr(sys, "argv", command[1:])
                context.setenv("A2UI_TENSORBOARD_ROOT", environment["A2UI_TENSORBOARD_ROOT"])
                _script_module("prepare_review_training").main()
        elif "launch_review_training.py" in command[1]:
            _script_module("launch_review_training").verify_launch_binding(options.output_dir / "fit/training_config.yaml")
            if "--preflight-only" not in command:
                if fail_training:
                    raise RuntimeError("fixture training failure")
                for role in ("best_golden_checkpoint", "final_adapter" if options.profile == "e2b" else "final_model"):
                    directory = options.output_dir / "fit/training" / role
                    directory.mkdir(parents=True)
                    (directory / "model.safetensors").write_bytes(b"mock trained fixture, not weights")
        else:
            assert "--require-prepared-contract" in command
            from ir_training.common.config import load_yaml
            from ir_training.eval.prepared_contract import verify_evaluation_prepared_contract
            config_path = Path(command[command.index("--config") + 1])
            assert environment["CUDA_VISIBLE_DEVICES"] == load_yaml(config_path)["runtime"]["cuda_visible_devices"]
            assert environment["A2UI_SKIP_CUDA_DEVICE_NORMALIZE"] == "1"
            assert "A2UI_CUDA_VISIBLE_DEVICES" not in environment
            assert "A2UI_EXCLUDE_CUDA_DEVICES" not in environment
            verify_evaluation_prepared_contract(
                config_path, Path(command[command.index("--split") + 1]),
                required_rows=int(command[command.index("--required-rows") + 1]),
                max_input_tokens=int(command[command.index("--max-input-tokens") + 1]),
            )
            name = command[command.index("--evaluation-name") + 1]
            if name == fail_evaluation:
                raise RuntimeError("fixture evaluation failure")
            output = Path(command[command.index("--output-dir") + 1])
            count = int(command[command.index("--required-rows") + 1])
            output.mkdir(parents=True)
            for file, value in (("evaluation_result.json", {"row_count": count, "mock": True}),
                                ("aggregate_metrics.json", {"count": count, "mock": True})):
                (output / file).write_text(json.dumps(value), encoding="utf-8")
            for file in ("predictions.jsonl", "scored_predictions.jsonl"):
                write_jsonl(output / file, [{"mock": True}] * count)
    return run


@pytest.mark.parametrize("profile,qat,devices", [("e2b", False, "auto"), ("270m", False, "1"), ("270m", True, "auto")])
def test_complete_pipeline_runs_preflight_training_and_four_bound_evaluations(options, monkeypatch, profile, qat, devices):
    options = replace(options, profile=profile, qat=qat, devices=devices)
    calls = []
    state = workflow.run_pipeline(options, execute=True, tokenizer_loader=lambda *_: FixtureTokenizer(),
                                  command_runner=fake_runner(options, monkeypatch, calls))
    assert state["status"] == "complete"
    assert list(state["completed"]) == state["plan"]["stages"]
    evaluations = [call for call in calls if "evaluate_checkpoint_on_golden.py" in call[1]]
    assert len(evaluations) == 4
    assert [call[call.index("--required-rows") + 1] for call in evaluations] == ["32", "35", "32", "35"]
    assert all(call[call.index("--run-id") + 1] == options.output_dir.name for call in evaluations)
    assert len(json.loads((options.output_dir / "evaluation_scorecard.json").read_text())["evaluations"]) == 4


def test_continue_after_preparation_and_retry_evaluation_never_retrains(options, monkeypatch):
    prepare(options)
    calls = []
    with pytest.raises(RuntimeError, match="evaluation failure"):
        workflow.run_pipeline(options, execute=True, continue_run=True,
                              command_runner=fake_runner(options, monkeypatch, calls, fail_evaluation="best_golden35"))
    before = len([call for call in calls if "launch_review_training.py" in call[1]])
    result = workflow.run_pipeline(options, execute=True, continue_run=True,
                                   command_runner=fake_runner(options, monkeypatch, calls))
    assert result["status"] == "complete"
    assert len([call for call in calls if "launch_review_training.py" in call[1]]) == before
    assert (options.output_dir / "evaluations/best_golden35/attempt_002/evaluation_result.json").is_file()


def test_failed_training_cannot_be_silently_restarted(options, monkeypatch):
    with pytest.raises(RuntimeError, match="training failure"):
        workflow.run_pipeline(options, execute=True, tokenizer_loader=lambda *_: FixtureTokenizer(),
                              command_runner=fake_runner(options, monkeypatch, [], fail_training=True))
    with pytest.raises(ValueError, match="optimizer silently"):
        workflow.run_pipeline(options, execute=True, continue_run=True,
                              command_runner=lambda *args: pytest.fail("must not relaunch"))


def test_invalid_cadence_and_e2b_official_qat_mislabel_fail_in_plan(options):
    with pytest.raises(ValueError, match="multiple"):
        workflow.build_plan(replace(options, eval_steps=500, golden_every_steps=750))
    with pytest.raises(ValueError, match="Official E2B"):
        workflow.build_plan(replace(options, qat=True))


def test_verified_cache_owns_copy_and_rejects_changed_cache_artifact(options, monkeypatch):
    monkeypatch.setattr(workflow, "_load_tokenizer", lambda *_: FixtureTokenizer())
    first = workflow.run_pipeline(options, prepare_only=True)
    assert first["status"] == "prepared"
    assert (options.output_dir / "preparation_receipt.json").is_file()
    second_options = replace(options, output_dir=options.output_dir.with_name("second-run"), steps=40, epochs=2)
    original = workflow._prepare_uncached
    monkeypatch.setattr(workflow, "_prepare_uncached", lambda *args: pytest.fail("Verified cache should avoid preparing rows again"))
    second = workflow.run_pipeline(second_options, prepare_only=True)
    assert second["status"] == "prepared"
    assert (second_options.output_dir / "cache_reuse.json").is_file()
    assert (options.output_dir / "prepared/train.jsonl").read_bytes() == (second_options.output_dir / "prepared/train.jsonl").read_bytes()
    # Reuse must be a separate copy, not a mutable hard link to the old run.
    (second_options.output_dir / "prepared/train.jsonl").write_text("{}\n", encoding="utf-8")
    assert (options.output_dir / "prepared/train.jsonl").read_text(encoding="utf-8") != "{}\n"
    calls = []
    def rebuild(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(workflow, "_prepare_uncached", rebuild)
    third_options = replace(options, output_dir=options.output_dir.with_name("third-run"))
    workflow.run_pipeline(third_options, prepare_only=True)
    assert calls == []
    assert (third_options.output_dir / "cache_reuse.json").is_file()
    # Mutating an output cannot poison the store. Only corruption in the owned
    # cache entry forces a rebuild, never reuse of unchecked artifacts.
    entries = list((options.output_dir.parent / ".golden-preparation-cache/prepared-v2").glob("*/prepared/train.jsonl"))
    assert len(entries) == 1
    entries[0].write_text("{}\n", encoding="utf-8")
    fourth_options = replace(options, output_dir=options.output_dir.with_name("fourth-run"))
    workflow.run_pipeline(fourth_options, prepare_only=True)
    assert calls == [1]
    assert not (fourth_options.output_dir / "cache_reuse.json").exists()


def test_changed_tokenizer_assets_invalidate_cache(options, monkeypatch):
    monkeypatch.setattr(workflow, "_load_tokenizer", lambda *_: FixtureTokenizer())
    workflow.run_pipeline(options, prepare_only=True)
    (options.model_dir / "tokenizer_config.json").write_text('{"changed": true}', encoding="utf-8")
    def stop(*args):
        raise RuntimeError("expected cache miss")
    monkeypatch.setattr(workflow, "_prepare_uncached", stop)
    with pytest.raises(RuntimeError, match="expected cache miss"):
        workflow.run_pipeline(replace(options, output_dir=options.output_dir.with_name("changed-tokenizer")), prepare_only=True)


def test_input_json_errors_are_fatal_not_silently_skipped(options):
    with (options.input_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("not JSON\n")
    with pytest.raises(ValueError, match="no rows may be silently skipped"):
        prepare(options)
    assert not (options.output_dir / "prepared").exists()


def test_prepare_progress_is_both_on_console_and_in_log(options, capsys):
    prepare(options)
    console = capsys.readouterr().out
    saved = (options.output_dir / "logs/prepare.log").read_text(encoding="utf-8")
    for message in ("Strict filter train", "Prepare/tokenize train", "Verify prepared splits"):
        assert message in console and message in saved
    assert "rows/s" in saved and "ETA" in saved


def test_continuation_rejects_changed_tokenizer_before_loading(options, monkeypatch):
    monkeypatch.setattr(workflow, "_load_tokenizer", lambda *_: FixtureTokenizer())
    workflow.run_pipeline(options, prepare_only=True)
    (options.model_dir / "tokenizer_config.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer changed"):
        workflow.run_pipeline(options, prepare_only=True, continue_run=True)


def test_unwritable_optional_cache_does_not_lose_verified_preparation(options, monkeypatch):
    from ir_training.pipeline import preparation_cache
    monkeypatch.setattr(workflow, "_load_tokenizer", lambda *_: FixtureTokenizer())
    def unavailable(*args, **kwargs):
        raise PermissionError("fixture shared index is read-only")
    monkeypatch.setattr(preparation_cache, "publish", unavailable)
    result = workflow.run_pipeline(options, prepare_only=True)
    assert result["status"] == "prepared"
    assert (options.output_dir / "prepared/manifest.json").is_file()


def test_invalid_startup_controls_fail_before_creating_outputs(options):
    for bad in (replace(options, prepare_workers=-1), replace(options, progress_seconds=float("nan")),
                replace(options, preparation_cache_dir=options.input_dir / "cache")):
        with pytest.raises(ValueError):
            workflow.build_plan(bad)
    assert not options.output_dir.exists()


def test_tuning_controls_forwarded_and_golden35_reserved_but_deferred(options, monkeypatch):
    options = replace(options, learning_rate=1e-5, weight_decay=0.05, warmup_ratio=0.05,
                      seed=123, logging_steps=5, gradient_checkpointing=False, evaluate_golden35=False)
    calls = []
    state = workflow.run_pipeline(options, execute=True, tokenizer_loader=lambda *_: FixtureTokenizer(),
                                  command_runner=fake_runner(options, monkeypatch, calls))
    command = workflow.configure_command(state["plan"])
    for flag, value in (("--learning-rate", "1e-05"), ("--weight-decay", "0.05"), ("--warmup-ratio", "0.05"),
                        ("--seed", "123"), ("--logging-steps", "5")):
        assert command[command.index(flag) + 1] == value
    assert "--no-gradient-checkpointing" in command
    assert "--golden35-file" in command  # still bind and reserve the entire holdout
    evaluations = [call for call in calls if "evaluate_checkpoint_on_golden.py" in call[1]]
    assert len(evaluations) == 2
    assert all(call[call.index("--required-rows") + 1] == "32" for call in evaluations)
    assert list(state["completed"]) == state["plan"]["stages"]
    scorecard = json.loads((options.output_dir / "evaluation_scorecard.json").read_text())
    assert scorecard["golden35_evaluated"] is False


@pytest.mark.parametrize("overrides", [
    {"learning_rate": 0}, {"learning_rate": float("nan")}, {"weight_decay": -1},
    {"warmup_ratio": 1}, {"warmup_ratio": -0.01}, {"logging_steps": 0}, {"seed": -1},
    {"epochs": float("nan")}, {"augmentation": "invent_ir"},
    {"augmentation_max_extra_fraction": 0.8}, {"augmentation_max_family_repeats": 10},
    {"attn_implementation": "unknown"},
])
def test_invalid_experiment_controls_fail_before_output(options, overrides):
    with pytest.raises(ValueError):
        workflow.build_plan(replace(options, **overrides))
    assert not options.output_dir.exists()


def test_optional_augmentation_routes_a_separate_copy_into_training(options, monkeypatch):
    options = replace(options, augmentation="rare_components")
    calls = []
    state = workflow.run_pipeline(options, execute=True, tokenizer_loader=lambda *_: FixtureTokenizer(),
                                  command_runner=fake_runner(options, monkeypatch, calls))
    assert list(state["completed"]) == state["plan"]["stages"]
    command = workflow.configure_command(state["plan"])
    assert Path(command[command.index("--dataset-dir") + 1]) == options.output_dir / "augmented"
    report = json.loads((options.output_dir / "augmented/augmentation.json").read_text())
    assert report["original_rows"] == 2 and report["added_rows"] == 0
    for name in ("val", "golden32", "golden35"):
        assert (options.output_dir / f"prepared/{name}.jsonl").read_bytes() == (options.output_dir / f"augmented/{name}.jsonl").read_bytes()
