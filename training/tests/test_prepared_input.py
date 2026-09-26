"""Frozen imports use actual published holdouts, a CPU tokenizer and stub teacher."""

from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))

from ir_training.common.jsonl import write_jsonl
from ir_training.data.express_preparation import prepare_splits
from ir_training.data.prepared_input import (
    adopt_prepared_input,
    prepared_input_files,
    validate_prepared_input,
)
from ir_training.data.semantic_augmentation import augment_training_at_startup
from ir_training.data.shared_prompt import create_shared_prompt_contract
from ir_training.eval.prepared_contract import file_sha256
from ir_training.pipeline.golden_training import GOLDEN_MANIFEST_SHA256, GOLDENS
from test_semantic_training_augmentation import Teacher, Tokenizer, source_row


class SmallTokenizer(Tokenizer):
    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": list(range(max(1, len(text.split()) // 10)))}


@pytest.fixture(scope="module")
def base_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("prepared-import-source")
    sources = {name: ROOT / item[0] for name, item in GOLDENS.items()}
    for name, count in (("train", 20), ("val", 1)):
        sources[name] = root / f"{name}.jsonl"
        write_jsonl(sources[name], [source_row(f"portable-{name}-{index}") for index in range(count)])
    shared = create_shared_prompt_contract(ordering="root-first")
    prepare_splits(sources, root / "prepared", tokenizer=SmallTokenizer(), ordering="root-first",
                   max_seq_length=4096, max_input_tokens=5120, shared_prompt=shared,
                   chat_template_kwargs={"enable_thinking": False}, evaluation_splits=set(GOLDENS))
    return root / "prepared"


@pytest.fixture
def plan(tmp_path, base_bundle):
    source = tmp_path / "portable"
    shutil.copytree(base_bundle, source)
    return {
        "options": {"prepared_input_dir": str(source), "output_dir": str(tmp_path / "run"),
                    "model_dir": str(tmp_path / "tokenizer-only"), "profile": "e2b",
                    "max_seq_length": 4096, "max_input_tokens": 5120},
        "shared_prompt": create_shared_prompt_contract(ordering="root-first"),
        "goldens": {name: {"path": str(ROOT / item[0]), "sha256": item[2], "rows": item[1],
                           "benchmark_manifest_path": str((ROOT / item[0]).parent / "benchmark_manifest.json"),
                           "benchmark_manifest_sha256": GOLDEN_MANIFEST_SHA256[name]}
                    for name, item in GOLDENS.items()},
    }


def load_tokenizer(*_):
    return SmallTokenizer()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_portable_plain_bundle_copies_every_byte_and_never_loads_weights(plan):
    source = Path(plan["options"]["prepared_input_dir"])
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["splits"].values():
        entry["source_path"] = "/producer-machine-no-longer-exists/source.jsonl"
    save(manifest_path, manifest)
    before = {path.name: path.read_bytes() for path in prepared_input_files(source)}
    checked = validate_prepared_input(plan, tokenizer_loader=load_tokenizer)
    assert checked["generation_executed"] is False
    assert checked["student_weights_loaded"] is False
    assert not Path(plan["options"]["output_dir"]).exists()
    report = adopt_prepared_input(plan, tokenizer_loader=load_tokenizer)
    destination = Path(plan["options"]["output_dir"]) / "prepared"
    assert {path.name: path.read_bytes() for path in prepared_input_files(destination)} == before
    assert {path.name: path.read_bytes() for path in prepared_input_files(source)} == before
    assert report["source_files"] == report["destination_files"]
    assert set(report["goldens"]) == set(GOLDENS)
    with pytest.raises(FileExistsError):
        adopt_prepared_input(plan, tokenizer_loader=load_tokenizer)


@pytest.mark.parametrize("change,reason", [
    ("missing", "missing required"), ("subdirectory", "regular files only"),
    ("train", "train split changed"), ("prompt", "shared prompt differs"),
    ("limit", "limit mismatch"), ("tokenizer", "tokenizer binding mismatch"),
    ("benchmark", "source/benchmark hash mismatch"),
])
def test_invalid_bundles_fail_before_publication(plan, change, reason):
    source = Path(plan["options"]["prepared_input_dir"])
    loader = load_tokenizer
    if change == "missing":
        (source / "val.jsonl").unlink()
    elif change == "subdirectory":
        (source / "unexpected").mkdir()
    elif change == "train":
        with (source / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif change == "prompt":
        plan["shared_prompt"] = {"stale": True}
    elif change == "limit":
        plan["options"]["max_input_tokens"] = 4096
    elif change == "tokenizer":
        tokenizer = SmallTokenizer()
        tokenizer.chat_template = "different"
        loader = lambda *_: tokenizer
    elif change == "benchmark":
        plan["goldens"]["golden32"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=reason):
        adopt_prepared_input(plan, tokenizer_loader=loader)
    assert not (Path(plan["options"]["output_dir"]) / "prepared").exists()


def test_rebound_benchmark_lineage_is_not_accepted(plan):
    source = Path(plan["options"]["prepared_input_dir"])
    path = source / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # Membership can still pass its own contract; pinned provenance cannot.
    manifest["splits"]["golden32"]["benchmark"]["source_run"] = "different-origin"
    save(path, manifest)
    with pytest.raises(ValueError, match="benchmark lineage"):
        validate_prepared_input(plan, tokenizer_loader=load_tokenizer)


def test_links_are_rejected_before_reading(plan, monkeypatch):
    source = Path(plan["options"]["prepared_input_dir"])
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == source / "train.jsonl" or original(path))
    with pytest.raises(ValueError, match="must not traverse links"):
        prepared_input_files(source)


def test_source_mutation_during_copy_never_publishes(plan, monkeypatch):
    import ir_training.data.prepared_input as module

    source = Path(plan["options"]["prepared_input_dir"])
    original = shutil.copyfile

    def copy_then_mutate(src, dst):
        result = original(src, dst)
        if Path(src).name == "train.jsonl":
            with (source / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n")
        return result

    monkeypatch.setattr(module.shutil, "copyfile", copy_then_mutate)
    with pytest.raises(ValueError, match="changed while importing"):
        adopt_prepared_input(plan, tokenizer_loader=load_tokenizer)
    assert not (Path(plan["options"]["output_dir"]) / "prepared").exists()


@pytest.fixture
def semantic_plan(plan, tmp_path):
    source = Path(plan["options"]["prepared_input_dir"])
    work = tmp_path / "producer"
    work.mkdir()
    shutil.copytree(source, work / "prepared")
    producer = deepcopy(plan)
    producer["options"].update(output_dir=str(work), augmentation="semantic", seed=42,
        augmentation_max_samples=2, augmentation_max_extra_fraction=.1,
        augmentation_max_family_repeats=2)
    augment_training_at_startup(producer, tokenizer_loader=load_tokenizer, command_runner=Teacher())
    bundle = work / "augmented"
    for original, name in (
        (work / "prepared/manifest.json", "augmentation_source_manifest.json"),
        (work / "semantic_augmentation/generated/manifest.json", "augmentation_generation_manifest.json"),
        (work / "semantic_augmentation/donors.jsonl", "augmentation_donors.jsonl"),
        (work / "semantic_augmentation/generated/accepted_genui.jsonl", "augmentation_accepted_genui.jsonl"),
    ):
        shutil.copyfile(original, bundle / name)
    plan["options"]["prepared_input_dir"] = str(bundle)
    return plan


def test_sealed_semantic_bundle_keeps_generation_lineage(semantic_plan):
    report = adopt_prepared_input(semantic_plan, tokenizer_loader=load_tokenizer)
    assert report["augmentation"]["added_rows"] > 0
    assert report["source_files"] == report["destination_files"]
    assert "augmentation_accepted_genui.jsonl" in report["source_files"]


@pytest.mark.parametrize("name", [
    "augmentation_source_manifest.json", "augmentation_generation_manifest.json",
    "augmentation_donors.jsonl", "augmentation_accepted_genui.jsonl", "augmentation.json",
])
def test_semantic_evidence_is_required_and_hash_bound(semantic_plan, name):
    source = Path(semantic_plan["options"]["prepared_input_dir"])
    with (source / name).open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="augmentation|generation"):
        validate_prepared_input(semantic_plan, tokenizer_loader=load_tokenizer)


def test_rebound_prepared_candidate_cannot_forge_accepted_stage3(semantic_plan):
    source = Path(semantic_plan["options"]["prepared_input_dir"])
    path = source / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["metadata"]["augmentation"]["stage3_record_sha256"] = "0" * 64
    write_jsonl(path, rows)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((source / "augmentation.json").read_text(encoding="utf-8"))
    report["output_train_sha256"] = file_sha256(path)
    save(source / "augmentation.json", report)
    manifest["augmentation"] = report
    manifest["augmentation_sha256"] = file_sha256(source / "augmentation.json")
    manifest["splits"]["train"]["output_sha256"] = file_sha256(path)
    save(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="unique Stage 3 evidence"):
        validate_prepared_input(semantic_plan, tokenizer_loader=load_tokenizer)
