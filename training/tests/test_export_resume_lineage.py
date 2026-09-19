"""Export provenance with real hashes and synthetic weights; no GPU/model execution."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest
import yaml
from ir_training.pipeline import checkpoint_export as ce
from ir_training.pipeline import deployment_export as de
from ir_training.train import resume_contract as rc
from ir_training.train.callbacks import _write_checkpoint_provenance
from ir_training.train.resume_contract import (
    build_resume_contract,
    file_sha256,
    verify_resume_contract,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def checkpoint(path, config_path, *, step, source=None, profile="e2b"):
    path.mkdir(parents=True)
    config = yaml.safe_load(config_path.read_text())
    if profile == "e2b":
        (path / "adapter_model.safetensors").write_bytes(b"fixture adapter")
        dump(path / "adapter_config.json", {"r": 16})
    else:
        (path / "model.safetensors").write_bytes(b"fixture full model")
        dump(path / "config.json", {"model_type": "gemma3"})
    dump(path / "tokenizer.json", {"fixture": "tokenizer"})
    world = config["runtime"]["world_size"]
    for name in (
        "optimizer.pt",
        "scheduler.pt",
        *(f"rng_state_{rank}.pth" for rank in range(world)),
    ):
        (path / name).write_bytes(b"fixture optimizer or RNG")
    dump(path / "trainer_state.json", {"global_step": step})
    contract = build_resume_contract(
        config, Path(config["run"]["dataset_dir"]), effective_batch=2 * world
    )
    metadata = {
        "config_path": str(config_path),
        "training_config_sha256": file_sha256(config_path),
        "checkpoint_step": step,
        "model": config["model"],
        "training": config["training"],
        "lora": config.get("lora", {}),
        "dataset_dir": config["run"]["dataset_dir"],
        "effective_batch_size": 2 * world,
        "resume_contract": contract,
        "resume_from_checkpoint": str(source) if source else None,
        "resume_state": verify_resume_contract(source, contract) if source else None,
    }
    _write_checkpoint_provenance(path, role="trainer_intermediate", payload=metadata)
    shutil.copy2(config_path, path / "training_config.yaml")


def fixture(tmp_path, profile="e2b", world=2):
    fit, base, data = tmp_path / "fit", tmp_path / "model", tmp_path / "data"
    dump(
        base / "config.json", {"model_type": "gemma4" if profile == "e2b" else "gemma3"}
    )
    (base / "model.safetensors").write_bytes(b"fixture original model")
    dump(data / "manifest.json", {})
    for name in ("train", "val"):
        (data / f"{name}.jsonl").write_text("{}\n", encoding="utf-8")
    original = fit / "training_config.yaml"
    config = {
        "model": {
            "model_id": "local-test-model",
            "model_source": str(base),
            "dtype": "bfloat16",
        },
        "run": {
            "id": "test",
            "dataset_dir": str(data),
            "output_dir": str(fit / "training"),
        },
        "runtime": {"world_size": world},
        "training": {
            "method": "lora_sft" if profile == "e2b" else "full_finetune_sft",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "expected_effective_batch_size": 2 * world,
            "epochs": 3,
            "learning_rate": 0.00002,
        },
        "lora": {"r": 16} if profile == "e2b" else {},
        "golden_eval": {"max_input_tokens": 4096, "max_new_tokens": 2048},
    }
    dump(original, config)
    dump(
        fit / "preparation_report.json",
        {
            "training_config_sha256": file_sha256(original),
            "model_files": {path.name: file_sha256(path) for path in base.iterdir()},
        },
    )
    source = fit / "training/checkpoint-3500"
    checkpoint(source, original, step=3500, profile=profile)
    resumed = fit / "training_config_resume_3500.yaml"
    config["training"]["resume_from_checkpoint"] = str(source)
    dump(resumed, config)
    selected = fit / "training/best_golden_checkpoint"
    checkpoint(selected, resumed, step=5000, source=source, profile=profile)
    return types.SimpleNamespace(
        fit=fit,
        original=original,
        resumed=resumed,
        source=source,
        selected=selected,
        base=base,
        data=data,
        profile=profile,
    )


def update_metadata(path, update):
    record = path / "training_metadata.json"
    value = json.loads(record.read_text())
    update(value)
    dump(record, value)


def options(fx, tmp_path):
    return ce.CheckpointExportOptions(
        profile=fx.profile,
        fit_dir=fx.fit,
        output_dir=tmp_path / "exports",
        exporter_python=Path(sys.executable),
        allow_experimental_formats=True,
    )


def remove_retained_source(fx):
    source = fx.source.resolve()
    assert source == (fx.fit / "training/checkpoint-3500").resolve()
    shutil.rmtree(source)


@pytest.mark.parametrize("profile", ["e2b", "270m"])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_uninterrupted_and_resumed_exports_preserve_all_source_bytes(
    tmp_path, profile, world
):
    fx = fixture(tmp_path, profile, world)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    baseline = de.verify_checkpoint_source(fx.original, fx.source, profile)
    assert baseline["training_config_sha256"] == file_sha256(fx.original)
    assert baseline["resume_lineage"] == {
        "verified": True,
        "resumed": False,
        "hops": [],
    }
    resumed = de.verify_checkpoint_source(fx.original, fx.selected, profile)
    assert resumed["training_config"] == str(fx.resumed)
    assert resumed["training_config_sha256"] == file_sha256(fx.resumed)
    assert resumed["preparation_config_sha256"] == file_sha256(fx.original)
    assert resumed["config"]["training"]["resume_from_checkpoint"] == str(fx.source)
    assert len(resumed["resume_lineage"]["hops"]) == 1
    plan = ce.build_checkpoint_export_plan(options(fx, tmp_path))
    cmd = plan["prepare_command"]
    assert cmd[cmd.index("--training-config") + 1] == str(fx.resumed)
    assert cmd[cmd.index("--preparation-config") + 1] == str(fx.original)
    assert plan["resume_lineage"] == resumed["resume_lineage"]
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (tmp_path / "exports").exists()


def test_multiple_resumes_terminate_at_original_preparation(tmp_path):
    fx = fixture(tmp_path)
    again = fx.fit / "training_config_resume_5000.yaml"
    config = yaml.safe_load(fx.resumed.read_text())
    config["training"]["resume_from_checkpoint"] = str(fx.selected)
    dump(again, config)
    final = fx.fit / "training/final_adapter"
    checkpoint(final, again, step=6000, source=fx.selected)
    report = de.verify_checkpoint_source(fx.original, final, "e2b")
    assert report["training_config"] == str(again)
    assert len(report["resume_lineage"]["hops"]) == 2


def test_selected_best_step_can_predate_resume_source(tmp_path):
    # The existing callback can rebind a retained best checkpoint to the active
    # resume config even when no later step has beaten its Golden score.
    fx = fixture(tmp_path)
    update_metadata(fx.selected, lambda m: m.update(checkpoint_step=3000))
    report = de.verify_checkpoint_source(fx.original, fx.selected, "e2b")
    assert report["metadata"]["checkpoint_step"] == 3000
    assert report["resume_lineage"]["hops"][0]["resume_state"]["global_step"] == 3500


def test_original_anchor_remains_selected_when_old_recorded_path_is_missing(tmp_path):
    fx = fixture(tmp_path)
    update_metadata(
        fx.source, lambda m: m.update(config_path=str(tmp_path / "missing.yaml"))
    )
    report = de.verify_checkpoint_source(fx.original, fx.source, "e2b")
    assert report["training_config"] == str(fx.original)
    assert report["resume_lineage"]["resumed"] is False


def test_extra_unbound_resume_source_tokenizer_is_rejected(tmp_path):
    fx = fixture(tmp_path)
    dump(fx.source / "tokenizer_config.json", {"extra": True})
    with pytest.raises(ValueError, match="inventory changed"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


def test_explicitly_named_original_config_still_supported(tmp_path):
    fx = fixture(tmp_path)
    named = fx.original.rename(fx.fit / "original_named_config.yaml")
    report = de.verify_checkpoint_source(named, fx.source, "e2b")
    assert report["training_config"] == str(named)
    assert report["preparation_config"] == str(named)
    assert report["resume_lineage"]["resumed"] is False


def test_missing_resume_config_and_snapshot_are_rejected(tmp_path):
    fx = fixture(tmp_path)
    fx.resumed.unlink()
    (fx.selected / "training_config.yaml").unlink()
    with pytest.raises(ValueError, match="bound resume config is missing"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


@pytest.mark.parametrize(
    "field,value",
    [("metadata_sha256", "0" * 64), ("global_step", 3499), ("verified", False)],
)
def test_tampered_recorded_resume_state_is_rejected(tmp_path, field, value):
    fx = fixture(tmp_path)
    update_metadata(fx.selected, lambda m: m["resume_state"].update({field: value}))
    with pytest.raises(ValueError, match="[Rr]esume"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


@pytest.mark.parametrize(
    "key,value",
    [
        ("learning_rate", 0.0001),
        ("epochs", 4),
        ("per_device_train_batch_size", 2),
        ("gradient_accumulation_steps", 1),
    ],
)
@pytest.mark.parametrize("source_deleted", [False, True])
def test_rehashed_but_changed_training_recipe_is_rejected(
    tmp_path, key, value, source_deleted
):
    fx = fixture(tmp_path)
    if source_deleted:
        remove_retained_source(fx)
    config = yaml.safe_load(fx.resumed.read_text())
    config["training"][key] = value
    dump(fx.resumed, config)
    shutil.copy2(fx.resumed, fx.selected / "training_config.yaml")
    update_metadata(
        fx.selected,
        lambda m: m.update(
            training_config_sha256=file_sha256(fx.resumed), training=config["training"]
        ),
    )
    with pytest.raises(ValueError, match="Unrelated config changes"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "model_id", "different"),
        ("lora", "r", 32),
        ("runtime", "world_size", 1),
        ("run", "dataset_dir", "/different/data"),
        ("golden_eval", "max_new_tokens", 1000),
    ],
)
@pytest.mark.parametrize("source_deleted", [False, True])
def test_other_rehashed_config_changes_are_rejected(
    tmp_path, section, key, value, source_deleted
):
    fx = fixture(tmp_path)
    if source_deleted:
        remove_retained_source(fx)
    config = yaml.safe_load(fx.resumed.read_text())
    config[section][key] = value
    dump(fx.resumed, config)
    shutil.copy2(fx.resumed, fx.selected / "training_config.yaml")
    update_metadata(
        fx.selected, lambda m: m.update(training_config_sha256=file_sha256(fx.resumed))
    )
    with pytest.raises(ValueError, match="Unrelated config changes"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


@pytest.mark.parametrize(
    "field,value",
    [
        ("effective_batch_size", 8),
        ("lora", {"r": 32}),
        ("model", {"model_id": "different"}),
        ("dataset_dir", "/different/data"),
        ("training", {}),
        ("resume_contract", {}),
        ("resume_state", None),
        ("resume_from_checkpoint", "/wrong/checkpoint"),
    ],
)
@pytest.mark.parametrize("source_deleted", [False, True])
def test_changed_resume_metadata_is_rejected(tmp_path, field, value, source_deleted):
    fx = fixture(tmp_path)
    if source_deleted:
        remove_retained_source(fx)
    update_metadata(fx.selected, lambda m: m.update({field: value}))
    with pytest.raises(ValueError, match="[Rr]esume"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


@pytest.mark.parametrize(
    "target",
    [
        "train",
        "val",
        "source_weight",
        "source_tokenizer",
        "source_metadata",
        "selected_weight",
        "base_weight",
        "resume_config",
        "resume_snapshot",
        "original_config",
    ],
)
def test_mutated_provenance_or_weights_stay_rejected(tmp_path, target):
    fx = fixture(tmp_path)
    path = {
        "train": fx.data / "train.jsonl",
        "val": fx.data / "val.jsonl",
        "source_weight": fx.source / "adapter_model.safetensors",
        "source_tokenizer": fx.source / "tokenizer.json",
        "source_metadata": fx.source / "training_metadata.json",
        "selected_weight": fx.selected / "adapter_model.safetensors",
        "base_weight": fx.base / "model.safetensors",
        "resume_config": fx.resumed,
        "resume_snapshot": fx.selected / "training_config.yaml",
        "original_config": fx.original,
    }[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


@pytest.mark.parametrize(
    "missing",
    ["optimizer.pt", "scheduler.pt", "rng_state_1.pth", "training_metadata.json"],
)
def test_missing_resume_source_evidence_is_not_bypassed(tmp_path, missing):
    fx = fixture(tmp_path)
    (fx.source / missing).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


def test_source_step_and_recorded_resume_report_are_bound(tmp_path):
    fx = fixture(tmp_path)
    dump(fx.source / "trainer_state.json", {"global_step": 3400})
    with pytest.raises(ValueError, match="Recorded resume state"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


def test_resume_cycle_rejected(tmp_path):
    fx = fixture(tmp_path)
    config = yaml.safe_load(fx.resumed.read_text())
    config["training"]["resume_from_checkpoint"] = str(fx.selected)
    dump(fx.resumed, config)
    shutil.copy2(fx.resumed, fx.selected / "training_config.yaml")
    update_metadata(
        fx.selected,
        lambda m: m.update(
            training_config_sha256=file_sha256(fx.resumed),
            training=config["training"],
            resume_from_checkpoint=str(fx.selected),
        ),
    )
    with pytest.raises(ValueError, match="Cycle"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


def test_changed_merge_lineage_is_not_accepted_by_export_launcher(tmp_path):
    fx = fixture(tmp_path)
    plan = ce.build_checkpoint_export_plan(options(fx, tmp_path))
    proof = {
        key: plan[key]
        for key in (
            "profile",
            "source_checkpoint",
            "merged_model_dir",
            "training_config_sha256",
            "preparation_config",
            "preparation_config_sha256",
            "resume_lineage",
        )
    }
    proof.update(merged_files={"test": "hash"}, official_retained_scale_export=False)
    proof["resume_lineage"] = {"verified": True, "resumed": False, "hops": []}
    dump(Path(plan["merged_model_dir"]) / "deployment_source.json", proof)
    with pytest.raises(ValueError, match="resume provenance changed"):
        ce._validate_stage(plan, "merge")


@pytest.mark.parametrize(
    "profile,remove_recorded_config", [("e2b", False), ("e2b", True), ("270m", False)]
)
@pytest.mark.parametrize("source_deleted", [False, True])
def test_merge_uses_actual_resume_config_and_preserves_original_anchor(
    tmp_path, monkeypatch, profile, remove_recorded_config, source_deleted
):
    fx = fixture(tmp_path, profile)
    if source_deleted:
        remove_retained_source(fx)
    import ir_training.eval.prepared_contract as contracts
    import ir_training.export.merge_lora as merge
    from ir_training.models import registry

    class Tokenizer:
        def save_pretrained(self, target):
            dump(Path(target) / "tokenizer.json", {"fixture": "tokenizer"})

    merged_configs = []

    def fake_merge(**kwargs):
        merged_configs.append(kwargs["training_config_path"])
        folder = kwargs["output_dir"]
        folder.mkdir()
        (folder / "model.safetensors").write_bytes(b"fixture merged model")
        dump(folder / "config.json", {"model_type": "gemma4"})

    monkeypatch.setattr(merge, "merge_lora_adapter", fake_merge)
    monkeypatch.setattr(
        registry,
        "create_adapter",
        lambda cfg: types.SimpleNamespace(load_tokenizer=Tokenizer),
    )
    monkeypatch.setattr(
        contracts, "checked_preparation_manifest", lambda path: {"tokenizer": {}}
    )
    monkeypatch.setattr(
        contracts, "verify_loaded_evaluation_tokenizer", lambda *args: None
    )
    template = (
        Path(de.__file__).resolve().parents[3]
        / "configs/export/gemma4_e2b_training_minijinja.jinja"
    )
    monkeypatch.setattr(
        de,
        "verify_deployment_template",
        lambda *a, **k: {"passed": True, "sha256": file_sha256(template)},
    )
    config_bytes = fx.resumed.read_bytes()
    if remove_recorded_config:
        fx.resumed.unlink()
    expected = (
        fx.selected / "training_config.yaml" if remove_recorded_config else fx.resumed
    )
    plan = ce.build_checkpoint_export_plan(options(fx, tmp_path))
    assert plan["training_config"] == str(expected)
    output = tmp_path / "merged"
    result = de.prepare_deployment_checkpoint(
        profile=profile,
        training_config=Path(plan["training_config"]),
        preparation_config=fx.original,
        checkpoint=fx.selected,
        output_dir=output,
    )
    assert result["training_config"] == str(expected)
    assert (output / "training_config.yaml").read_bytes() == config_bytes
    assert result["preparation_config_sha256"] == file_sha256(fx.original)
    assert result["resume_lineage"] == plan["resume_lineage"]
    assert yaml.safe_load((output / "training_config.yaml").read_text())["training"][
        "resume_from_checkpoint"
    ] == str(fx.source)
    if profile == "e2b":
        assert merged_configs == [expected]


@pytest.mark.parametrize("profile", ["e2b", "270m"])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_deleted_source_uses_saved_evidence_without_changing_any_inputs(
    tmp_path, capsys, profile, world
):
    fx = fixture(tmp_path, profile, world)
    update_metadata(fx.selected, lambda m: m.update(checkpoint_step=7000))
    saved_state = json.loads((fx.selected / "training_metadata.json").read_text())[
        "resume_state"
    ]
    remove_retained_source(fx)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = de.verify_checkpoint_source(fx.original, fx.selected, profile)
    lineage = result["resume_lineage"]
    assert lineage["verification_mode"] == "saved_resume_evidence"
    assert lineage["physical_chain_complete"] is False
    assert lineage["missing_sources"] == [str(fx.source)]
    hop = lineage["hops"][0]
    assert hop["resume_state"] == saved_state
    assert hop["source_metadata_rehashed"] is False
    assert hop["source_files_rechecked"] is False
    assert hop["surviving_checkpoint_files_rechecked"] is True
    assert hop["surviving_metadata_sha256"] == file_sha256(
        fx.selected / "training_metadata.json"
    )
    assert hop["prepared_split_sha256"] == {
        name: file_sha256(fx.data / f"{name}.jsonl") for name in ("train", "val")
    }
    assert (
        "source_config_sha256" not in hop
    )  # Never invent a deleted config's identity.
    plan = ce.build_checkpoint_export_plan(options(fx, tmp_path))
    assert plan["resume_lineage"] == lineage
    assert plan["training_config"] == str(fx.resumed)
    assert not fx.source.exists()
    assert not (tmp_path / "exports").exists()
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert "deleted source bytes cannot be rechecked" in capsys.readouterr().out


@pytest.mark.parametrize(
    "field,value",
    [
        ("verified", False),
        ("verified", "true"),
        ("verified", 1),
        ("global_step", None),
        ("global_step", True),
        ("global_step", 3500.0),
        ("global_step", 0),
        ("global_step", -1),
        ("global_step", 3501),
        ("metadata_sha256", None),
        ("metadata_sha256", ""),
        ("metadata_sha256", "bad"),
        ("metadata_sha256", "x" * 64),
        ("metadata_sha256", "0" * 64),
        ("checkpoint", "/wrong/checkpoint-3500"),
    ],
)
def test_deleted_source_rejects_missing_or_inconsistent_resume_state(
    tmp_path, field, value
):
    fx = fixture(tmp_path)
    remove_retained_source(fx)
    update_metadata(fx.selected, lambda m: m["resume_state"].update({field: value}))
    with pytest.raises(ValueError, match="[Rr]esume|retained"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


@pytest.mark.parametrize(
    "target",
    ["train", "val", "weights", "tokenizer", "base", "original", "resume", "snapshot"],
)
def test_deleted_source_does_not_bypass_surviving_file_hashes(tmp_path, target):
    fx = fixture(tmp_path)
    remove_retained_source(fx)
    path = {
        "train": fx.data / "train.jsonl",
        "val": fx.data / "val.jsonl",
        "weights": fx.selected / "adapter_model.safetensors",
        "tokenizer": fx.selected / "tokenizer.json",
        "base": fx.base / "model.safetensors",
        "original": fx.original,
        "resume": fx.resumed,
        "snapshot": fx.selected / "training_config.yaml",
    }[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_inventory",
        "missing_tokenizer",
        "extra_weight",
        "before_source",
        "missing_val",
    ],
)
def test_deleted_source_requires_sufficient_surviving_evidence(tmp_path, mutation):
    fx = fixture(tmp_path)
    remove_retained_source(fx)
    if mutation == "missing_inventory":
        update_metadata(fx.selected, lambda m: m.pop("checkpoint_adapter_files"))
    elif mutation == "missing_tokenizer":
        (fx.selected / "tokenizer.json").unlink()
        update_metadata(
            fx.selected,
            lambda m: m.update(
                checkpoint_adapter_files=[
                    item
                    for item in m["checkpoint_adapter_files"]
                    if item["path"] != "tokenizer.json"
                ]
            ),
        )
    elif mutation == "extra_weight":
        (fx.selected / "adapter_extra.safetensors").write_bytes(
            b"unbound extra weights"
        )
    elif mutation == "before_source":
        update_metadata(fx.selected, lambda m: m.update(checkpoint_step=3000))
    else:
        (fx.data / "val.jsonl").unlink()
        update_metadata(
            fx.selected, lambda m: m["resume_contract"]["splits"].pop("val")
        )
    with pytest.raises(ValueError):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


@pytest.mark.parametrize(
    "path", ["checkpoint-3499", "not-a-trainer-checkpoint", "other-run/checkpoint-3500"]
)
def test_deleted_source_path_must_match_run_and_recorded_step(tmp_path, path):
    fx = fixture(tmp_path)
    remove_retained_source(fx)
    source = fx.source.parent / path
    source.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(fx.resumed.read_text())
    config["training"]["resume_from_checkpoint"] = str(source)
    dump(fx.resumed, config)
    shutil.copy2(fx.resumed, fx.selected / "training_config.yaml")
    update_metadata(
        fx.selected,
        lambda m: m.update(
            training=config["training"],
            training_config_sha256=file_sha256(fx.resumed),
            resume_from_checkpoint=str(source),
        ),
    )
    with pytest.raises(ValueError, match="path/step"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


def test_existing_empty_source_directory_cannot_trigger_retention_fallback(tmp_path):
    fx = fixture(tmp_path)
    remove_retained_source(fx)
    fx.source.mkdir()
    with pytest.raises(FileNotFoundError):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


def test_source_reappearing_during_verification_requires_physical_checks(
    tmp_path, monkeypatch
):
    fx = fixture(tmp_path)
    remove_retained_source(fx)
    verify = rc._verify_source_inventory

    def reappearing(checkpoint_path, metadata):
        verify(checkpoint_path, metadata)
        fx.source.mkdir()

    monkeypatch.setattr(rc, "_verify_source_inventory", reappearing)
    with pytest.raises(ValueError, match="reappeared"):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


def test_source_io_error_cannot_trigger_retention_fallback(tmp_path, monkeypatch):
    fx = fixture(tmp_path)
    lstat = Path.lstat

    def guarded(path, *args, **kwargs):
        if path == fx.source:
            raise PermissionError("fixture permission error")
        return lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", guarded)
    with pytest.raises(PermissionError):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


def test_dangling_link_is_not_treated_as_deleted_source(tmp_path, monkeypatch):
    fx = fixture(tmp_path)
    remove_retained_source(fx)
    lstat = Path.lstat
    # A dangling link has an lstat entry even though its target cannot be read.
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path, *a, **k: object() if path == fx.source else lstat(path, *a, **k),
    )
    assert rc._resume_source_directory_absent(str(fx.source), fx.source) is False
    with pytest.raises(FileNotFoundError):
        de.verify_checkpoint_source(fx.original, fx.selected, "e2b")


def test_missing_ancestor_does_not_hide_existing_resolved_source(tmp_path):
    fx = fixture(tmp_path)
    alias = fx.source.parent / "absent" / ".." / fx.source.name
    assert rc._resume_source_directory_absent(str(alias), fx.source) is False


def test_physical_then_deleted_source_chain_records_both_verification_modes(tmp_path):
    fx = fixture(tmp_path)
    intermediate = fx.fit / "training/checkpoint-5000"
    checkpoint(intermediate, fx.resumed, step=5000, source=fx.source)
    config = yaml.safe_load(fx.resumed.read_text())
    config["training"]["resume_from_checkpoint"] = str(intermediate)
    again = fx.fit / "training_config_resume_5000.yaml"
    dump(again, config)
    final = fx.fit / "training/final_adapter"
    checkpoint(final, again, step=7000, source=intermediate)
    remove_retained_source(fx)
    result = de.verify_checkpoint_source(fx.original, final, "e2b")
    assert [hop["verification_mode"] for hop in result["resume_lineage"]["hops"]] == [
        "physical_source",
        "saved_resume_evidence",
    ]
    assert result["training_config"] == str(again)


def test_real_training_resume_still_requires_physical_source(tmp_path):
    fx = fixture(tmp_path)
    contract = json.loads((fx.selected / "training_metadata.json").read_text())[
        "resume_contract"
    ]
    remove_retained_source(fx)
    with pytest.raises(ValueError, match="Cannot resume"):
        verify_resume_contract(fx.source, contract)


def test_saved_evidence_changed_between_plan_and_merge_is_rejected(tmp_path):
    fx = fixture(tmp_path)
    remove_retained_source(fx)
    plan = ce.build_checkpoint_export_plan(options(fx, tmp_path))
    update_metadata(
        fx.selected,
        lambda m: m["resume_state"].update(metadata_sha256="0123456789abcdef" * 4),
    )
    # The missing digest cannot be recomputed, but any change after planning is
    # bound by the surviving metadata hash and must invalidate the merge proof.
    changed = de.verify_checkpoint_source(fx.original, fx.selected, "e2b")
    proof = {
        key: plan[key]
        for key in (
            "profile",
            "source_checkpoint",
            "merged_model_dir",
            "training_config_sha256",
            "preparation_config",
            "preparation_config_sha256",
        )
    }
    proof.update(
        merged_files={"test": "hash"},
        official_retained_scale_export=False,
        resume_lineage=changed["resume_lineage"],
    )
    dump(Path(plan["merged_model_dir"]) / "deployment_source.json", proof)
    with pytest.raises(ValueError, match="resume provenance changed"):
        ce._validate_stage(plan, "merge")
