"""Separate Muse preprocessing publishes a reusable, training-free bundle."""
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))

from ir_training.common.jsonl import write_jsonl
from ir_training.pipeline import golden_training, semantic_preparation
from ir_training.pipeline.golden_training import GoldenTrainingOptions, sha256
from test_semantic_training_augmentation import Teacher, Tokenizer, source_row


@pytest.fixture
def options(tmp_path):
    model, source = tmp_path / "tokenizer", tmp_path / "source"
    model.mkdir()
    source.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}", encoding="utf-8")
    write_jsonl(source / "train.jsonl", [source_row(f"train-{i}") for i in range(20)])
    write_jsonl(source / "val.jsonl", [source_row("val-a"), source_row("val-b")])
    return GoldenTrainingOptions(model, tmp_path / "preprocess", profile="e2b", input_dir=source,
                                 max_seq_length=4096, max_input_tokens=5120, augmentation="semantic",
                                 prepare_workers=1, preparation_cache=False)


def test_plan_only_needs_no_student_weights_or_teacher(options, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Plan-only must not load tokenizer, start training, or contact teacher")
    monkeypatch.setattr(golden_training, "prepare_data", forbidden)
    state = semantic_preparation.prepare_semantic_dataset(options, tokenizer_loader=forbidden, command_runner=forbidden)
    assert state["status"] == "plan_only"
    assert state["plan"]["stages"] == ["prepare", "augment", "seal", "verify"]
    assert state["training_executed"] is False
    assert state["student_weights_loaded"] is False
    assert not options.output_dir.exists()


@pytest.mark.parametrize("profile,max_input_tokens", [("e2b", 5120), ("270m", 4096)])
def test_standalone_then_training_adopts_same_bytes_without_muse(options, profile, max_input_tokens, monkeypatch):
    options = replace(options, profile=profile, max_input_tokens=max_input_tokens)
    teacher = Teacher()
    state = semantic_preparation.prepare_semantic_dataset(
        options, execute=True, tokenizer_loader=lambda *_: Tokenizer(), command_runner=teacher,
    )
    assert state["status"] == "prepared"
    assert state["training_executed"] is False and state["student_weights_loaded"] is False
    assert len(teacher.calls) == 1
    assert not (options.output_dir / "fit").exists()
    bundle = Path(state["prepared_input_dir"])
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert 20 < manifest["splits"]["train"]["accepted_rows"] <= 22
    assert manifest["splits"]["val"]["accepted_rows"] == 2
    for name in ("val", "golden32", "golden35", "bixby50"):
        assert (bundle / f"{name}.jsonl").read_bytes() == (options.output_dir / "prepared" / f"{name}.jsonl").read_bytes()
    assert {path.name: sha256(path) for path in bundle.iterdir()} == state["bundle_files"]
    for name in ("source_manifest.json", "generation_manifest.json", "donors.jsonl", "accepted_genui.jsonl"):
        assert (bundle / f"augmentation_{name}").is_file()
    # Portable: copy just the bundle. No dependency on original absolute paths.
    portable = options.output_dir.parent / "copied-bundle"
    shutil.copytree(bundle, portable)
    options.output_dir.rename(options.output_dir.with_name("archived-preprocessing"))
    (options.model_dir / "model.safetensors").write_bytes(b"fixture, never loaded")
    consumer = replace(options, input_dir=None, prepared_input_dir=portable,
                       output_dir=options.output_dir.parent / "training-run", augmentation="none")
    def forbidden(*args, **kwargs):
        pytest.fail("Frozen consumption must not re-filter, re-tokenize, or generate data")
    monkeypatch.setattr(golden_training, "_prepare_uncached", forbidden)
    import ir_training.data.semantic_augmentation as semantic
    monkeypatch.setattr(semantic, "augment_training_at_startup", forbidden)
    plan = golden_training.build_plan(consumer, preparation_only=True)
    consumer.output_dir.mkdir()
    golden_training.prepare_data(plan, tokenizer_loader=lambda *_: Tokenizer())
    assert {path.name: sha256(path) for path in (consumer.output_dir / "prepared").iterdir()} == state["bundle_files"]
    assert len(teacher.calls) == 1


def test_failed_teacher_does_not_start_training_or_report_completion(options):
    def failed(*args, **kwargs):
        raise RuntimeError("fixture Muse endpoint unavailable")
    with pytest.raises(RuntimeError, match="Muse endpoint unavailable"):
        semantic_preparation.prepare_semantic_dataset(options, execute=True, tokenizer_loader=lambda *_: Tokenizer(), command_runner=failed)
    receipt = json.loads((options.output_dir / "augmentation_preparation_manifest.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed" and receipt["active_stage"] == "augment"
    assert receipt["training_executed"] is False
    assert not (options.output_dir / "augmented").exists()
    assert not (options.output_dir / "fit").exists()
    with pytest.raises(FileExistsError):
        semantic_preparation.prepare_semantic_dataset(options, execute=True)


def test_evidence_sealing_rejects_changed_donor_file(options, monkeypatch):
    original = semantic_preparation._seal_evidence
    def tamper(output):
        with (output / "semantic_augmentation/donors.jsonl").open("ab") as stream:
            stream.write(b"\n")
        original(output)
    monkeypatch.setattr(semantic_preparation, "_seal_evidence", tamper)
    with pytest.raises(ValueError, match="evidence changed"):
        semantic_preparation.prepare_semantic_dataset(options, execute=True, tokenizer_loader=lambda *_: Tokenizer(), command_runner=Teacher())
    receipt = json.loads((options.output_dir / "augmentation_preparation_manifest.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed" and receipt["active_stage"] == "seal"


@pytest.fixture
def interrupted_augmentation(options, monkeypatch):
    from ir_training.data import semantic_augmentation

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted before generation")

    with monkeypatch.context() as patch:
        patch.setattr(semantic_augmentation, "augment_training_at_startup", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            semantic_preparation.prepare_semantic_dataset(options, execute=True, tokenizer_loader=lambda *_: Tokenizer())
    return options


def test_resume_reuses_original_preparation_and_preserves_failed_receipt(interrupted_augmentation, monkeypatch):
    from ir_training.data import semantic_augmentation

    options = interrupted_augmentation
    receipt = options.output_dir / "augmentation_preparation_manifest.json"
    original_receipt = receipt.read_bytes()
    original_base = {p.name: p.read_bytes() for p in (options.output_dir / "prepared").iterdir()}
    original_helper = semantic_augmentation.augment_training_at_startup
    calls = []

    def resumed(plan, **kwargs):
        calls.append(kwargs.pop("resume"))
        # This interruption occurred before a dataset work directory existed.
        return original_helper(plan, **kwargs)

    monkeypatch.setattr(semantic_augmentation, "augment_training_at_startup", resumed)
    monkeypatch.setattr(golden_training, "prepare_data", lambda *a, **k: pytest.fail("Original preparation must not rerun"))
    result = semantic_preparation.prepare_semantic_dataset(options, execute=True, resume=True,
        tokenizer_loader=lambda *_: Tokenizer(), command_runner=Teacher())
    assert result["status"] == "prepared" and calls == [True]
    assert {p.name: p.read_bytes() for p in (options.output_dir / "prepared").iterdir()} == original_base
    assert (options.output_dir / "resume_history/attempt-0001.json").read_bytes() == original_receipt
    assert result["base_files"] and result["raw_files"] and result["producer_contract"]


@pytest.mark.parametrize("change", ["raw", "base", "contract", "options"])
def test_resume_rejects_changed_bindings_without_writes(interrupted_augmentation, monkeypatch, change):
    from ir_training.data import semantic_augmentation

    options = interrupted_augmentation
    receipt = options.output_dir / "augmentation_preparation_manifest.json"
    if change == "raw":
        with (options.input_dir / "train.jsonl").open("ab") as stream:
            stream.write(b"\n")
    elif change == "base":
        with (options.output_dir / "prepared/train.jsonl").open("ab") as stream:
            stream.write(b"\n")
    elif change == "contract":
        state = json.loads(receipt.read_text(encoding="utf-8"))
        state["producer_contract"]["producer_files"]["dataset/configs/models.yaml"] = "0" * 64
        receipt.write_text(json.dumps(state), encoding="utf-8")
    else:
        options = replace(options, seed=999)
    before = receipt.read_bytes()
    monkeypatch.setattr(semantic_augmentation, "augment_training_at_startup", lambda *a, **k: pytest.fail("Must fail before generation"))
    with pytest.raises(ValueError, match="changed"):
        semantic_preparation.prepare_semantic_dataset(options, execute=True, resume=True, tokenizer_loader=lambda *_: Tokenizer())
    assert receipt.read_bytes() == before
    assert not (options.output_dir / "resume_history").exists()


def test_resume_incomplete_prepare_fails_without_recovery_guess(options, monkeypatch):
    def interrupted(*args, **kwargs):
        raise RuntimeError("preparation interrupted")

    monkeypatch.setattr(golden_training, "prepare_data", interrupted)
    with pytest.raises(RuntimeError):
        semantic_preparation.prepare_semantic_dataset(options, execute=True)
    receipt = options.output_dir / "augmentation_preparation_manifest.json"
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="Incomplete original preparation"):
        semantic_preparation.prepare_semantic_dataset(options, execute=True, resume=True)
    assert receipt.read_bytes() == before


def test_resume_completed_is_verified_noop(options, monkeypatch):
    from ir_training.data import semantic_augmentation

    first = semantic_preparation.prepare_semantic_dataset(options, execute=True,
        tokenizer_loader=lambda *_: Tokenizer(), command_runner=Teacher())
    receipt = options.output_dir / "augmentation_preparation_manifest.json"
    before = receipt.read_bytes()
    monkeypatch.setattr(semantic_augmentation, "augment_training_at_startup", lambda *a, **k: pytest.fail("Completed data needs no teacher"))
    second = semantic_preparation.prepare_semantic_dataset(options, execute=True, resume=True, tokenizer_loader=lambda *_: Tokenizer())
    assert second == first and receipt.read_bytes() == before
    assert not (options.output_dir / "resume_history").exists()
    with (options.output_dir / "augmented/train.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="bundle changed"):
        semantic_preparation.prepare_semantic_dataset(options, execute=True, resume=True, tokenizer_loader=lambda *_: Tokenizer())


def test_resume_after_partial_sealing_reuses_generation(options, monkeypatch):
    original_seal = semantic_preparation._seal_evidence
    teacher = Teacher()

    def interrupted(output):
        original_seal(output)
        raise RuntimeError("interrupted after copying evidence")

    with monkeypatch.context() as patch:
        patch.setattr(semantic_preparation, "_seal_evidence", interrupted)
        with pytest.raises(RuntimeError, match="copying evidence"):
            semantic_preparation.prepare_semantic_dataset(options, execute=True,
                tokenizer_loader=lambda *_: Tokenizer(), command_runner=teacher)
    result = semantic_preparation.prepare_semantic_dataset(options, execute=True, resume=True,
        tokenizer_loader=lambda *_: Tokenizer(), command_runner=lambda *a, **k: pytest.fail("Do not regenerate a completed bundle"))
    assert result["status"] == "prepared" and len(teacher.calls) == 1
    original_seal(options.output_dir)  # Already sealed exact bytes are idempotent.
    (options.output_dir / "augmented/augmentation_donors.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Previously sealed"):
        original_seal(options.output_dir)


def test_legacy_resume_without_generation_manifest_fails_before_writes(interrupted_augmentation):
    options = interrupted_augmentation
    receipt = options.output_dir / "augmentation_preparation_manifest.json"
    state = json.loads(receipt.read_text(encoding="utf-8"))
    for key in ("producer_contract", "base_files", "raw_files"):
        state.pop(key)
    receipt.write_text(json.dumps(state), encoding="utf-8")
    before = receipt.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError), match="manifest|donors|resume|Resume"):
        semantic_preparation.prepare_semantic_dataset(options, execute=True, resume=True, tokenizer_loader=lambda *_: Tokenizer())
    assert receipt.read_bytes() == before and not (options.output_dir / "resume_history").exists()


@pytest.mark.parametrize("profile,max_input_tokens", [("e2b", 5120), ("270m", 4096)])
def test_independent_training_includes_saved_augmentation_only_when_enabled(options, profile, max_input_tokens, monkeypatch):
    options = replace(options, profile=profile, max_input_tokens=max_input_tokens)
    teacher = Teacher()
    saved = semantic_preparation.prepare_semantic_dataset(
        options, execute=True, tokenizer_loader=lambda *_: Tokenizer(), command_runner=teacher,
    )
    bundle = Path(saved["augmentation_dir"])
    original_bytes = (options.output_dir / "prepared/train.jsonl").read_bytes()
    combined_bytes = (bundle / "train.jsonl").read_bytes()
    assert original_bytes != combined_bytes and combined_bytes.startswith(original_bytes)
    (options.model_dir / "model.safetensors").write_bytes(b"fixture, never loaded")
    import ir_training.data.semantic_augmentation as semantic
    from ir_training.data.prepared_input import adopt_prepared_augmentation

    def forbidden(*args, **kwargs):
        pytest.fail("Training must not invoke Muse or the standalone generator")
    monkeypatch.setattr(semantic, "augment_training_at_startup", forbidden)
    consumer = replace(options, augmentation_dir=bundle, output_dir=options.output_dir.parent / "with-augmentation")
    if profile == "270m":
        state = golden_training.run_pipeline(consumer, prepare_only=True,
            tokenizer_loader=lambda *_: Tokenizer(), command_runner=forbidden)
        assert list(state["completed"]) == ["prepare", "augment"]
    else:
        # Official mobile uses this shared CPU preparation with separate 4096/5120 limits.
        plan = golden_training.build_plan(consumer, preparation_only=True)
        consumer.output_dir.mkdir()
        golden_training.prepare_data(plan, tokenizer_loader=lambda *_: Tokenizer())
        adopt_prepared_augmentation(plan, tokenizer_loader=lambda *_: Tokenizer())
    assert (consumer.output_dir / "prepared/train.jsonl").read_bytes() == original_bytes
    assert (consumer.output_dir / "augmented/train.jsonl").read_bytes() == combined_bytes
    for name in ("val", "golden32", "golden35", "bixby50"):
        assert (consumer.output_dir / f"prepared/{name}.jsonl").read_bytes() == (consumer.output_dir / f"augmented/{name}.jsonl").read_bytes()
    assert len(teacher.calls) == 1

    plain = replace(consumer, augmentation="none", augmentation_dir=None,
                    output_dir=options.output_dir.parent / "without-augmentation")
    plan = golden_training.build_plan(plain, preparation_only=True)
    plain.output_dir.mkdir()
    golden_training.prepare_data(plan, tokenizer_loader=lambda *_: Tokenizer())
    assert (plain.output_dir / "prepared/train.jsonl").read_bytes() == original_bytes
    assert not (plain.output_dir / "augmented").exists()
    assert len(teacher.calls) == 1


@pytest.mark.parametrize("change,match", [
    ({"augmentation": "none"}, "augmentation=semantic"),
    ({"input_dir": None}, "explicit"),
])
def test_standalone_rejects_ambiguous_source_or_mode(options, change, match):
    with pytest.raises(ValueError, match=match):
        semantic_preparation.prepare_semantic_dataset(replace(options, **change))


def test_standalone_rejects_source_nested_output(options):
    with pytest.raises(ValueError, match="outside"):
        semantic_preparation.prepare_semantic_dataset(replace(options, output_dir=options.input_dir / "derived"))


@pytest.mark.parametrize("profile,expected_limit", [("e2b", 5120), ("270m", 4096)])
def test_cli_defaults_plan_only_and_profile_limits(options, capsys, profile, expected_limit):
    spec = importlib.util.spec_from_file_location("prepare_semantic_script", ROOT / "training/scripts/prepare_semantic_augmentation.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--profile", profile, "--model-dir", str(options.model_dir),
                        "--input-dir", str(options.input_dir), "--output-dir", str(options.output_dir)]) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["status"] == "plan_only"
    assert state["plan"]["options"]["max_input_tokens"] == expected_limit
    assert state["plan"]["options"]["augmentation"] == "semantic"
    assert not options.output_dir.exists()


def test_cli_resume_is_forwarded_only_to_standalone(options, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("resume_semantic_script", ROOT / "training/scripts/prepare_semantic_augmentation.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    calls = []

    def capture(parsed, **kwargs):
        calls.append(kwargs)
        return {"status": "plan_only"}

    monkeypatch.setattr(script, "prepare_semantic_dataset", capture)
    assert script.main(["--profile", "270m", "--model-dir", str(options.model_dir),
                        "--input-dir", str(options.input_dir), "--output-dir", str(options.output_dir),
                        "--resume", "--execute"]) == 0
    assert calls == [{"execute": True, "resume": True}]
    assert not options.output_dir.exists()


def test_sealing_optional_reference_bindings_is_hash_checked(tmp_path):
    prepared, generated, bundle = tmp_path / "prepared", tmp_path / "semantic_augmentation/generated", tmp_path / "augmented"
    for directory in (prepared, generated, bundle):
        directory.mkdir(parents=True)
    sources = {
        prepared / "manifest.json": "{}",
        generated.parent / "donors.jsonl": "{}\n",
        generated.parent / "reference_bindings.json": '{"version":1}',
        generated / "accepted_genui.jsonl": "{}\n",
    }
    for path, content in sources.items():
        path.write_text(content, encoding="utf-8")
    (generated / "manifest.json").write_text(json.dumps({
        "accepted_genui_sha256": sha256(generated / "accepted_genui.jsonl"),
    }), encoding="utf-8")
    report = {"source_manifest_sha256": sha256(prepared / "manifest.json"),
              "generation_manifest_sha256": sha256(generated / "manifest.json"),
              "donors_sha256": sha256(generated.parent / "donors.jsonl"),
              "reference_bindings_sha256": sha256(generated.parent / "reference_bindings.json")}
    (bundle / "augmentation.json").write_text(json.dumps(report), encoding="utf-8")
    semantic_preparation._seal_evidence(tmp_path)
    destination = bundle / "augmentation_reference_bindings.json"
    assert sha256(destination) == report["reference_bindings_sha256"]
    semantic_preparation._seal_evidence(tmp_path)
    destination.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Previously sealed"):
        semantic_preparation._seal_evidence(tmp_path)


def test_interrupted_atomic_receipt_write_keeps_previous_json(tmp_path, monkeypatch):
    from ir_training.pipeline import preparation_cache

    receipt = tmp_path / "augmentation_preparation_manifest.json"
    semantic_preparation._write(receipt, {"status": "failed", "active_stage": "augment"})
    before = receipt.read_bytes()

    def interrupted(*args, **kwargs):
        raise OSError("interrupted before atomic replacement")

    monkeypatch.setattr(preparation_cache.os, "replace", interrupted)
    with pytest.raises(OSError, match="atomic replacement"):
        semantic_preparation._write(receipt, {"status": "running"})
    assert receipt.read_bytes() == before
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "failed"
    assert not list(tmp_path.glob(".receipt-*"))


def test_parallel_standalone_execution_is_locked_before_output_write(options, monkeypatch):
    from ir_training.common.cache_store import cache_lock, digest

    output = options.output_dir.resolve()
    key = "standalone-augmentation-" + digest(os.path.normcase(str(output)))
    monkeypatch.setattr(golden_training, "prepare_data", lambda *a, **k: pytest.fail("Competing run must not start"))
    with cache_lock(output.parent, key), pytest.raises(TimeoutError, match="timed out"):
        semantic_preparation.prepare_semantic_dataset(options, execute=True)
    assert not output.exists()
