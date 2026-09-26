"""Generate a reusable Muse augmentation folder independently of training.

Dataset code owns all teacher generation. This orchestration loads only the
student tokenizer and reuses the existing admission/isolation gates. It never
starts or stops a teacher server or launches student training.
"""
from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ir_training.pipeline import golden_training
from ir_training.pipeline.golden_training import GoldenTrainingOptions, sha256
from ir_training.pipeline.preparation_cache import _atomic_json as _write


def _read(path: Path) -> dict:
    from ir_training.pipeline.preparation_cache import _assert_no_links

    _assert_no_links(path)
    if not path.is_file():
        raise ValueError(f"Resume requires an existing regular artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")  # noqa: TRY004 - invalid external artifact
    return value


def _producer_contract(plan: dict, raw_files: dict) -> dict:
    from ir_training.pipeline.preparation_cache import identity

    root = golden_training.repo_root()
    paths = [
        "training/src/ir_training/pipeline/semantic_preparation.py",
        "training/scripts/prepare_semantic_augmentation.py",
        "training/src/ir_training/data/semantic_augmentation.py",
        "training/src/ir_training/data/semantic_reference_normalization.py",
        "training/src/ir_training/data/prepared_input.py",
        "dataset/scripts/generate_training_augmentations.py",
        "dataset/configs/models.yaml",
        "dataset/prompts/training_augmentation_v1.md",
        "dataset/prompts/genui_gen_mobile_a2ui_express_v1.md",
        "dataset/prompts/muse_stage3_quality_v1.md",
        "dataset/schema/canonical_ui_graph_v1.schema.json",
    ]
    # Include transitive generation/reference gates, not just its CLI wrapper.
    paths.extend(path.relative_to(root).as_posix() for directory in ("pipeline", "llm", "utils")
                 for path in sorted((root / "dataset/src" / directory).rglob("*.py")))
    return {"preparation": identity(plan, raw_files),
            "producer_files": {name: sha256(root / name) for name in sorted(set(paths))}}


def _validation_plan(plan: dict, options: GoldenTrainingOptions, directory: Path) -> dict:
    imported = replace(options, input_dir=None, source_run_dir=None,
                       prepared_input_dir=directory, augmentation="none")
    return {**plan, "options": golden_training._options(imported)}


def _base_files(output: Path) -> dict:
    from ir_training.data.prepared_input import prepared_input_files

    paths = [*prepared_input_files(output / "prepared"), output / "data_audit.json"]
    if (output / "preparation_receipt.json").exists():
        paths.append(output / "preparation_receipt.json")
    return {path.relative_to(output).as_posix(): sha256(path) for path in paths}


def _resume_state(plan: dict, options: GoldenTrainingOptions, *, tokenizer_loader=None) -> dict:
    """Validate every reuse decision before changing the old run or its receipt."""
    from ir_training.data.prepared_input import (
        prepared_input_files,
        validate_prepared_input,
    )
    from ir_training.pipeline.preparation_cache import _assert_no_links

    output = Path(plan["options"]["output_dir"])
    state = _read(output / "augmentation_preparation_manifest.json")
    if state.get("plan") != plan or state.get("schema_version") != 1:
        raise ValueError("Resume options, source paths or prompt contract changed; use a fresh output")
    if state.get("status") not in {"failed", "running", "prepared"}:
        raise ValueError("Only an executed standalone preparation can be resumed")
    if state.get("active_stage") not in {"augment", "seal", "verify", None}:
        raise ValueError("Incomplete original preparation has no verified resume boundary; use a fresh output directory")
    audit = _read(output / "data_audit.json")
    for path in plan["source_files"]:
        _assert_no_links(Path(path))
    raw = {path: sha256(Path(path)) for path in plan["source_files"]}
    if audit.get("source_files") != raw or state.get("raw_files", raw) != raw:
        raise ValueError("Original raw input changed since preparation")
    contract = _producer_contract(plan, raw)
    legacy = "producer_contract" not in state
    if not legacy and state["producer_contract"] != contract:
        raise ValueError("Resume producer code, configuration or preparation contract changed")
    current_base = _base_files(output)
    if not legacy and state.get("base_files") != current_base:
        raise ValueError("Original prepared base or preparation receipt changed")
    preparation_receipt = output / "preparation_receipt.json"
    if preparation_receipt.exists():
        saved = _read(preparation_receipt)
        if saved.get("identity") != contract["preparation"]:
            raise ValueError("Original preparation implementation, tokenizer assets or source binding changed")
        for name, expected in saved.get("files", {}).items():
            path = output / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file() or sha256(path) != expected:
                raise ValueError("Original preparation receipt artifact changed")
    validate_prepared_input(_validation_plan(plan, options, output / "prepared"), tokenizer_loader=tokenizer_loader)
    if legacy:
        from ir_training.data.express_preparation import _api

        _api()
        from pipeline.training_augmentation import validate_generation_resume

        migration = validate_generation_resume(
            output / "semantic_augmentation/donors.jsonl", output / "semantic_augmentation/generated",
            teacher_model=options.augmentation_teacher_model, seed=options.seed,
            max_new_samples=len((output / "semantic_augmentation/donors.jsonl").read_text(encoding="utf-8").splitlines()),
        )
        if migration.get("legacy_upgrade") is not True:
            raise ValueError("Missing standalone producer contract is supported only for the known legacy producer")
        state["legacy_migration"] = migration
    if state.get("status") == "prepared":
        bundle = output / "augmented"
        current = {path.name: sha256(path) for path in prepared_input_files(bundle)}
        if state.get("bundle_files") != current:
            raise ValueError("Completed augmentation bundle changed")
        validate_prepared_input(_validation_plan(plan, options, bundle), tokenizer_loader=tokenizer_loader)
    state.update(raw_files=raw, base_files=current_base, producer_contract=contract)
    return state


def _seal_evidence(output: Path) -> None:
    """Keep provenance portable without following absolute manifest paths."""
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
    if "reference_bindings_sha256" in report:
        evidence["augmentation_reference_bindings.json"] = (
            output / "semantic_augmentation/reference_bindings.json", report["reference_bindings_sha256"],
        )
    for name, (source, expected) in evidence.items():
        destination = bundle / name
        if destination.is_symlink() or source.is_symlink() or not source.is_file():
            raise ValueError(f"Augmentation evidence must be a new regular copy: {name}")
        if sha256(source) != expected:
            raise ValueError(f"Augmentation evidence changed before publication: {name}")
        if destination.exists():
            if not destination.is_file() or sha256(destination) != expected:
                raise ValueError(f"Previously sealed augmentation evidence changed: {name}")
        else:
            shutil.copyfile(source, destination)
        if sha256(destination) != expected or sha256(source) != expected:
            raise ValueError(f"Augmentation evidence changed during publication: {name}")


def prepare_semantic_dataset(
    options: GoldenTrainingOptions, *, execute: bool = False, resume: bool = False,
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
    from ir_training.common.cache_store import cache_lock, digest

    # Keep this lock outside the new output: fresh creation must remain exclusive.
    # Its lifetime includes resume validation, receipt history, sealing and verify.
    key = "standalone-augmentation-" + digest(os.path.normcase(str(output)))
    with cache_lock(output.parent, key, timeout=1, interval=1):
        return _execute_preparation(plan, options, state, resume=resume,
                                    tokenizer_loader=tokenizer_loader, command_runner=command_runner)


def _execute_preparation(plan: dict, options: GoldenTrainingOptions, state: dict, *, resume: bool,
                         tokenizer_loader: Callable | None, command_runner: Callable | None) -> dict:
    output = Path(plan["options"]["output_dir"])
    receipt = output / "augmentation_preparation_manifest.json"
    if resume:
        state = _resume_state(plan, options, tokenizer_loader=tokenizer_loader)
        if state["status"] == "prepared":
            return state
        # Keep the exact old receipt, including failure diagnostics, before retry.
        from ir_training.pipeline.preparation_cache import _assert_no_links

        history = output / "resume_history"
        _assert_no_links(history)
        history.mkdir(exist_ok=True)
        index = 1
        while (history / f"attempt-{index:04d}.json").exists():
            index += 1
        shutil.copyfile(receipt, history / f"attempt-{index:04d}.json")
        state.pop("error", None)
        state.update(status="running", active_stage="augment")
    else:
        output.mkdir(parents=True, exist_ok=False)
        raw = {path: sha256(Path(path)) for path in plan["source_files"]}
        state.update(status="running", active_stage="prepare", raw_files=raw,
                     producer_contract=_producer_contract(plan, raw))
    _write(receipt, state)
    try:
        if not resume:
            with golden_training._console_log(output / "logs/prepare.log"):
                golden_training.prepare_data(plan, tokenizer_loader=tokenizer_loader)
            if _read(output / "data_audit.json").get("source_files") != state["raw_files"]:
                raise ValueError("Original raw inputs changed during preparation")
            state["base_files"] = _base_files(output)
        state["active_stage"] = "augment"
        _write(receipt, state)
        from ir_training.data.semantic_augmentation import augment_training_at_startup

        with golden_training._console_log(output / "logs/augmentation.log"):
            augment_training_at_startup(plan, tokenizer_loader=tokenizer_loader, command_runner=command_runner, resume=resume)
        current_raw = {path: sha256(Path(path)) for path in plan["source_files"]}
        if (current_raw != state["raw_files"] or _base_files(output) != state["base_files"]
                or _producer_contract(plan, current_raw) != state["producer_contract"]):
            raise ValueError("Original inputs, prepared base or producer contract changed during augmentation")
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
        validation_plan = _validation_plan(plan, options, output / "augmented")
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
