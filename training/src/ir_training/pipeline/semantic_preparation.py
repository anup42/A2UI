"""Generate a reusable Muse augmentation folder independently of training.

Dataset code owns all teacher generation. This orchestration loads only the
student tokenizer and reuses the existing admission/isolation gates. It never
starts or stops a teacher server or launches student training.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ir_training.pipeline import golden_training
from ir_training.pipeline.golden_training import GoldenTrainingOptions, _write, sha256


def _seal_evidence(output: Path) -> None:
    """Keep provenance portable without following absolute manifest paths."""
    import json

    bundle = output / "augmented"
    report = json.loads((bundle / "augmentation.json").read_text(encoding="utf-8"))
    generated = output / "semantic_augmentation/generated"
    generation = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    evidence = {
        "augmentation_source_manifest.json": (output / "prepared/manifest.json", report["source_manifest_sha256"]),
        "augmentation_generation_manifest.json": (generated / "manifest.json", report["generation_manifest_sha256"]),
        "augmentation_donors.jsonl": (output / "semantic_augmentation/donors.jsonl", report["donors_sha256"]),
        "augmentation_accepted_genui.jsonl": (generated / "accepted_genui.jsonl", generation["accepted_genui_sha256"]),
    }
    for name, (source, expected) in evidence.items():
        destination = bundle / name
        if destination.exists() or source.is_symlink() or not source.is_file():
            raise ValueError(f"Augmentation evidence must be a new regular copy: {name}")
        if sha256(source) != expected:
            raise ValueError(f"Augmentation evidence changed before publication: {name}")
        shutil.copyfile(source, destination)
        if sha256(destination) != expected or sha256(source) != expected:
            raise ValueError(f"Augmentation evidence changed during publication: {name}")


def prepare_semantic_dataset(
    options: GoldenTrainingOptions, *, execute: bool = False,
    tokenizer_loader: Callable | None = None, command_runner: Callable | None = None,
) -> dict:
    """Explicit execution only; a completed run is data, never a training run."""
    if options.augmentation != "semantic":
        raise ValueError("Standalone semantic preparation requires augmentation=semantic")
    if options.prepared_input_dir is not None:
        raise ValueError("Standalone augmentation takes raw input, not an already prepared bundle")
    if options.augmentation_dir is not None:
        raise ValueError("Standalone generation creates an augmentation folder; it does not consume --augmentation-dir")
    if options.input_dir is None and options.source_run_dir is None:
        raise ValueError("Choose an explicit --input-dir or --source-run-dir for standalone augmentation")
    plan = golden_training.build_plan(options, preparation_only=True, tokenizer_only=True)
    output = Path(plan["options"]["output_dir"])
    for protected in (options.model_dir, options.input_dir or options.source_run_dir):
        if output.is_relative_to(Path(protected).expanduser().resolve()):
            raise ValueError("Standalone output must be outside model and source directories")
    plan.update(workflow="standalone_muse_semantic_preparation_v1", stages=["prepare", "augment", "seal", "verify"])
    state = {"schema_version": 1, "status": "plan_only", "plan": plan,
             "training_executed": False, "student_weights_loaded": False,
             "teacher_server_managed": False, "augmentation_dir": str(output / "augmented"),
             "prepared_input_dir": str(output / "augmented")}
    if not execute:
        return state
    # Deliberately no continuation: partial teacher runs require fresh output.
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "augmentation_preparation_manifest.json"
    state.update(status="running", active_stage="prepare")
    _write(receipt, state)
    try:
        with golden_training._console_log(output / "logs/prepare.log"):
            golden_training.prepare_data(plan, tokenizer_loader=tokenizer_loader)
        state["active_stage"] = "augment"
        _write(receipt, state)
        from ir_training.data.semantic_augmentation import augment_training_at_startup

        with golden_training._console_log(output / "logs/augmentation.log"):
            augment_training_at_startup(plan, tokenizer_loader=tokenizer_loader, command_runner=command_runner)
        state["active_stage"] = "seal"
        _write(receipt, state)
        _seal_evidence(output)
        state["active_stage"] = "verify"
        _write(receipt, state)
        from ir_training.data.prepared_input import (
            prepared_input_files,
            validate_prepared_input,
        )

        # Verify exactly the contract the later training-only consumer uses.
        import_options = replace(options, input_dir=None, source_run_dir=None,
                                 prepared_input_dir=output / "augmented", augmentation="none")
        validation_plan = {**plan, "options": golden_training._options(import_options)}
        with golden_training._console_log(output / "logs/verify.log"):
            state["validation"] = validate_prepared_input(validation_plan, tokenizer_loader=tokenizer_loader)
        state.update(status="prepared", active_stage=None,
                     bundle_files={path.name: sha256(path) for path in prepared_input_files(output / "augmented")})
        _write(receipt, state)
        return state
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write(receipt, state)
        raise
