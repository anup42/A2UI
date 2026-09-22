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

from ir_training.qat.numeric_preflight import OFFICIAL_MOBILE_WORKFLOW
from ir_training.train import mobile_resume
from ir_training.train.callbacks import _write_checkpoint_provenance
from ir_training.train.resume_contract import build_resume_contract, file_sha256


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _base_config(tmp_path: Path, *, epochs: float = 2, max_steps=None) -> dict:
    dataset = tmp_path / "prepared"
    dataset.mkdir(parents=True)
    (dataset / "train.jsonl").write_text("".join(f'{{"id":{i}}}\n' for i in range(8)))
    (dataset / "val.jsonl").write_text('{"id":"v"}\n')
    run_root = tmp_path / "original"
    return {
        "run": {
            "id": "original",
            "purpose": OFFICIAL_MOBILE_WORKFLOW,
            "dataset_dir": str(dataset),
            "output_dir": str(run_root / "trainer"),
            "launch_plan_path": str(run_root / "launch/launch_plan.json"),
            "preflight_report_path": str(run_root / "launch/preflight_report.json"),
        },
        "model": {"model_id": "google/gemma-4-e2b-it", "dtype": "bfloat16"},
        "training": {
            "method": "qat_lora_sft",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "optim": "adamw_torch_fused",
            "seed": 7,
            "max_seq_length": 512,
            "epochs": epochs,
            "max_steps": max_steps,
            "warmup_ratio": 0.03,
            "warmup_steps": None,
            "max_grad_norm": 1.0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "expected_effective_batch_size": 2,
            "refuse_resume": True,
            "logging_dir": str(run_root / "logs"),
        },
        "lora": {"r": 8, "alpha": 16, "dropout": 0.0},
        "qat": {"enabled": True, "scale_mode": "retained_mobile", "bits": 4},
        "golden_eval": {
            "enabled": True,
            "output_dir": str(run_root / "golden_eval"),
            "best_checkpoint_dir": str(run_root / "best_golden_checkpoint"),
        },
        "runtime": {"world_size": 1, "cuda_visible_devices": "0"},
    }


def _checkpoint(
    path: Path,
    config_path: Path,
    config: dict,
    *,
    step: int,
    trainer_max_steps: int,
    resume_state: dict | None = None,
) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(f"adapter-{step}".encode())
    (path / "tokenizer_config.json").write_text("{}")
    (path / "optimizer.pt").write_bytes(f"optimizer-{step}".encode())
    (path / "scheduler.pt").write_bytes(f"scheduler-{step}".encode())
    (path / "rng_state.pth").write_bytes(f"rng-{step}".encode())
    (path / "golden_callback_state.json").write_text(
        json.dumps({"best": {"step": step, "metric_value": 1.0}})
    )
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": trainer_max_steps})
    )
    dataset = Path(config["run"]["dataset_dir"])
    contract = build_resume_contract(config, dataset, effective_batch=2)
    payload = {
        "training_metadata_version": 4,
        "config_path": str(config_path),
        "training_config_sha256": file_sha256(config_path),
        "checkpoint_step": step,
        "resume_contract": contract,
        "effective_batch_size": 2,
        "training": config["training"],
        "model": config["model"],
        "lora": config["lora"],
        "qat": config["qat"],
        "dataset_dir": config["run"]["dataset_dir"],
        "resume_from_checkpoint": config["training"].get("resume_from_checkpoint"),
        "resume_state": resume_state,
    }
    _write_checkpoint_provenance(path, role="trainer_intermediate", payload=payload)
    (path / "training_config.yaml").write_bytes(config_path.read_bytes())


def _original(tmp_path: Path, *, epochs=2, max_steps=None, step=3):
    config = _base_config(tmp_path, epochs=epochs, max_steps=max_steps)
    config_path = tmp_path / "original/config.yaml"
    _write_yaml(config_path, config)
    (config_path.parent / "preparation_report.json").write_text(
        json.dumps(
            {
                "training_config_sha256": file_sha256(config_path),
                "model_files": [{"path": "seed.safetensors", "sha256": "a" * 64}],
            }
        )
    )
    total = max_steps if max_steps is not None else 8
    checkpoint = tmp_path / f"original/trainer/checkpoint-{step}"
    _checkpoint(checkpoint, config_path, config, step=step, trainer_max_steps=total)
    return config_path, config, checkpoint


def _continuation(
    source: Path,
    source_config: dict,
    tmp_path: Path,
    name: str,
    *,
    epochs=None,
    max_steps="keep",
):
    config = copy.deepcopy(source_config)
    root = tmp_path / name
    config["run"].update(
        id=name,
        output_dir=str(root / "trainer"),
        launch_plan_path=str(root / "launch/launch_plan.json"),
        preflight_report_path=str(root / "launch/preflight_report.json"),
    )
    config["training"].update(
        refuse_resume=False,
        resume_policy=mobile_resume.POLICY,
        resume_from_checkpoint=str(source.resolve()),
        logging_dir=str(root / "logs"),
    )
    if epochs is not None:
        config["training"]["epochs"] = epochs
    if max_steps != "keep":
        config["training"]["max_steps"] = max_steps
    config["golden_eval"].update(
        output_dir=str(root / "golden_eval"),
        best_checkpoint_dir=str(root / "best_golden_checkpoint"),
    )
    config["training"]["resume_horizon"] = mobile_resume.horizon_record(source, config)
    path = root / "config.yaml"
    _write_yaml(path, config)
    return path, config


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [("epochs", {"epochs": 3}), ("max_steps", {"max_steps": 12})],
)
def test_horizon_extension_verifies_without_mutating_source(tmp_path, mode, kwargs):
    original_path, original, checkpoint = _original(
        tmp_path, max_steps=8 if mode == "max_steps" else None
    )
    before = _tree_hashes(checkpoint)
    _, config = _continuation(checkpoint, original, tmp_path, "continued", **kwargs)
    report = mobile_resume.verify_continuation(checkpoint, config)
    assert report["verified"] is True
    assert report["continuation"]["original"]["total_optimizer_steps"] == 8
    assert report["continuation"]["requested"]["total_optimizer_steps"] == 12
    assert set(report["state_files_sha256"]) == {
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "golden_callback_state.json",
        "rng_state.pth",
    }
    assert _tree_hashes(checkpoint) == before
    assert original_path.is_file()


def test_unchanged_horizon_is_allowed_only_below_target(tmp_path):
    _, original, checkpoint = _original(tmp_path, step=3)
    _, config = _continuation(checkpoint, original, tmp_path, "continued")
    assert mobile_resume.verify_continuation(checkpoint, config)["global_step"] == 3

    _, original2, completed = _original(tmp_path / "done", step=8)
    with pytest.raises(ValueError, match="extend beyond the completed global step"):
        _continuation(completed, original2, tmp_path / "done", "continued")


@pytest.mark.parametrize(
    "mutation",
    ["decrease", "switch", "inactive_epochs"],
)
def test_invalid_horizon_changes_fail(tmp_path, mutation):
    max_steps = 8 if mutation in {"switch", "inactive_epochs"} else None
    _, original, checkpoint = _original(tmp_path, max_steps=max_steps)
    kwargs = {
        "decrease": {"epochs": 1},
        "switch": {"max_steps": None, "epochs": 3},
        "inactive_epochs": {"max_steps": 12, "epochs": 3},
    }[mutation]
    with pytest.raises(ValueError, match="horizon|switch"):
        _continuation(checkpoint, original, tmp_path, "continued", **kwargs)


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("learning rate", lambda c: c["training"].update(learning_rate=9e-4)),
        ("optimizer", lambda c: c["training"].update(optim="sgd")),
        ("LoRA", lambda c: c["lora"].update(r=16)),
        ("QAT", lambda c: c["qat"].update(bits=8)),
        (
            "data",
            lambda c: c["run"].update(dataset_dir=c["run"]["dataset_dir"] + "-other"),
        ),
        (
            "batch",
            lambda c: c["training"].update(
                gradient_accumulation_steps=4, expected_effective_batch_size=4
            ),
        ),
        ("model", lambda c: c["model"].update(model_id="other/model")),
    ],
)
def test_recipe_changes_fail(tmp_path, label, mutate):
    _, original, checkpoint = _original(tmp_path)
    _, config = _continuation(checkpoint, original, tmp_path, "continued", epochs=3)
    mutate(config)
    with pytest.raises((ValueError, FileNotFoundError)):
        mobile_resume.verify_continuation(checkpoint, config)


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("purpose", lambda c: c["run"].update(purpose="ordinary_training")),
        ("refusal", lambda c: c["training"].update(refuse_resume=True)),
        ("method", lambda c: c["training"].update(method="lora_sft")),
        ("scale", lambda c: c["qat"].update(scale_mode="dynamic")),
    ],
)
def test_continuation_is_restricted_to_explicit_official_mobile_policy(
    tmp_path, label, mutate
):
    _, original, checkpoint = _original(tmp_path)
    _, config = _continuation(checkpoint, original, tmp_path, "continued", epochs=3)
    mutate(config)
    with pytest.raises(ValueError, match="official retained-mobile LoRA resume"):
        mobile_resume.verify_continuation(checkpoint, config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("run", "id", "wrong"),
        ("run", "output_dir", "wrong/trainer"),
        ("run", "launch_plan_path", "wrong/launch.json"),
        ("run", "preflight_report_path", "wrong/preflight.json"),
        ("golden_eval", "best_checkpoint_dir", "wrong/best"),
        ("golden_eval", "output_dir", "C:/outside-golden"),
    ],
)
def test_new_run_relocation_paths_are_strict(tmp_path, section, key, value):
    _, original, checkpoint = _original(tmp_path)
    _, config = _continuation(checkpoint, original, tmp_path, "continued", epochs=3)
    config[section][key] = value
    with pytest.raises(ValueError, match="relocation|new run"):
        mobile_resume.verify_continuation(checkpoint, config)


@pytest.mark.parametrize(
    "missing",
    ["optimizer.pt", "scheduler.pt", "rng_state.pth", "golden_callback_state.json"],
)
def test_required_resume_state_files_are_enforced(tmp_path, missing):
    _, original, checkpoint = _original(tmp_path)
    _, config = _continuation(checkpoint, original, tmp_path, "continued", epochs=3)
    (checkpoint / missing).unlink()
    with pytest.raises(ValueError, match="Cannot resume|golden_callback_state"):
        mobile_resume.verify_continuation(checkpoint, config)


def test_config_hash_and_saved_state_hash_tampering_fail(tmp_path):
    original_path, original, checkpoint = _original(tmp_path)
    continuation_path, continuation = _continuation(
        checkpoint, original, tmp_path, "continued", epochs=3
    )
    report = mobile_resume.verify_continuation(checkpoint, continuation)
    next_checkpoint = tmp_path / "continued/trainer/checkpoint-9"
    _checkpoint(
        next_checkpoint,
        continuation_path,
        continuation,
        step=9,
        trainer_max_steps=12,
        resume_state=report,
    )
    assert mobile_resume.verify_export_lineage(continuation_path, next_checkpoint)[
        "verified"
    ]

    (checkpoint / "optimizer.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Saved continuation state"):
        mobile_resume.verify_export_lineage(continuation_path, next_checkpoint)

    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-3")
    original_path.write_text(original_path.read_text() + "\n")
    with pytest.raises(ValueError, match="config|changed"):
        mobile_resume.verify_export_lineage(continuation_path, next_checkpoint)


@pytest.mark.parametrize("damage", ["missing", "wrong_hash", "no_models"])
def test_export_requires_original_bound_preparation_report(tmp_path, damage):
    original_path, _, checkpoint = _original(tmp_path)
    report_path = original_path.parent / "preparation_report.json"
    if damage == "missing":
        report_path.unlink()
    else:
        report = json.loads(report_path.read_text())
        if damage == "wrong_hash":
            report["training_config_sha256"] = "0" * 64
        else:
            report["model_files"] = []
        report_path.write_text(json.dumps(report))
    with pytest.raises(
        (FileNotFoundError, ValueError), match="prepar|identity|No such"
    ):
        mobile_resume.verify_export_lineage(original_path, checkpoint)


def test_multihop_lineage_preserves_original_previous_and_requested_horizons(tmp_path):
    original_path, original, first = _original(tmp_path)
    first_path, first_config = _continuation(
        first, original, tmp_path, "continued-1", epochs=3
    )
    first_report = mobile_resume.verify_continuation(first, first_config)
    second = tmp_path / "continued-1/trainer/checkpoint-9"
    _checkpoint(
        second,
        first_path,
        first_config,
        step=9,
        trainer_max_steps=12,
        resume_state=first_report,
    )

    second_path, second_config = _continuation(
        second, first_config, tmp_path, "continued-2", epochs=4
    )
    second_report = mobile_resume.verify_continuation(second, second_config)
    final = tmp_path / "continued-2/trainer/checkpoint-13"
    _checkpoint(
        final,
        second_path,
        second_config,
        step=13,
        trainer_max_steps=16,
        resume_state=second_report,
    )

    lineage = mobile_resume.verify_export_lineage(second_path, final)
    assert lineage["verified"] is True and lineage["resumed"] is True
    assert len(lineage["hops"]) == 2
    assert lineage["original_config"] == str(original_path.resolve())
    assert lineage["original_horizon"]["total_optimizer_steps"] == 8
    assert lineage["extended_horizon"]["total_optimizer_steps"] == 16
    newest = lineage["hops"][0]["resume_state"]["continuation"]
    assert newest["original"]["total_optimizer_steps"] == 8
    assert newest["previous"]["total_optimizer_steps"] == 12
    assert newest["requested"]["total_optimizer_steps"] == 16


def test_export_config_requires_byte_identity_not_only_equivalent_yaml(tmp_path):
    original_path, _, checkpoint = _original(tmp_path)
    alternate = tmp_path / "equivalent.yaml"
    alternate.write_bytes(original_path.read_bytes() + b"\n# unbound copy\n")
    with pytest.raises(ValueError, match="Supplied export config.*SHA256"):
        mobile_resume.verify_export_lineage(alternate, checkpoint)
