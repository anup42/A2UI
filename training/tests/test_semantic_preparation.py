"""Separate Muse preprocessing publishes a reusable, training-free bundle."""
import importlib.util
import json
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
