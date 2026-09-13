"""Exact train resampling keeps source/target/token bindings and holdouts intact."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.data.augmentation import augment_prepared_training, validate_augmentation_options
from ir_training.data.chat_templates import build_messages, build_prompt
from ir_training.data.express_preparation import prepare_splits
from ir_training.eval.prepared_contract import checked_preparation_manifest, file_sha256


class Tokenizer:
    name_or_path, chat_template = "augmentation-fixture", "role-content"
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def get_vocab(self):
        return {"fixture": 0}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        return "".join(f"{item['role']}:\n{item['content']}\n" for item in messages) + ("assistant:\n" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": list(range(len(text.split())))}


def row(identity, *, component="Text", family=None, source=None):
    source = source or f"Independent training response {identity}"
    completion = f'<a2ui>\nroot={component}("https://example.com/{identity}")\n</a2ui>' if component == "Image" else f'<a2ui>\nroot=Text("{identity}")\n</a2ui>'
    return {"id": identity, "row_id": identity, "source_id": family or identity,
            "response_id": f"response-{identity}", "response_text": source,
            "messages": build_messages("Fixture system", source, completion, target_format="a2ui_express_v1"),
            "prompt": build_prompt("Fixture system", source, target_format="a2ui_express_v1"),
            "completion": completion, "metadata": {}}


def prepare(tmp_path, train_rows=None):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = train_rows if train_rows is not None else [row(f"common-{i}") for i in range(28)] + [row(f"image-{i}", component="Image") for i in range(2)]
    write_jsonl(raw / "train.jsonl", rows)
    for name in ("val", "golden32", "golden35"):
        write_jsonl(raw / f"{name}.jsonl", [row(f"{name}-{i}", component="Image") for i in range(2)])
    source = tmp_path / "prepared"
    prepare_splits({name: raw / f"{name}.jsonl" for name in ("train", "val", "golden32", "golden35")}, source,
                   ordering="root-first", tokenizer=Tokenizer(), max_seq_length=4096, max_input_tokens=4096,
                   evaluation_splits={"golden32", "golden35"})
    return source


def augment(source, output, **kwargs):
    return augment_prepared_training(source, output, rare_frequency=0.10, min_component_families=2, **kwargs)


def test_resampling_preserves_inputs_and_all_nontraining_files(tmp_path):
    source = prepare(tmp_path)
    hashes = {path.name: file_sha256(path) for path in source.iterdir()}
    output = tmp_path / "augmented"
    report = augment(source, output)
    assert report["original_rows"] == 30 and report["added_rows"] == 2 and report["output_rows"] == 32
    assert report["source_families"] == 30 and report["resampled_families"] == 2
    assert report["eligible_components"] == ["Image"]
    assert report["synthetic_targets_created"] == report["source_texts_changed"] == report["independent_examples_added"] == 0
    assert report["component_exposure"]["Image"] == {"original_rows": 2, "output_rows": 4, "source_families": 2, "eligible": True}
    for path in source.iterdir():
        assert file_sha256(path) == hashes[path.name]
        if path.name not in {"train.jsonl", "manifest.json"}:
            assert file_sha256(output / path.name) == hashes[path.name]
    original = {item["id"]: item for item in read_jsonl(source / "train.jsonl")}
    rows = list(read_jsonl(output / "train.jsonl"))
    assert len({item["id"] for item in rows}) == len(rows)
    assert max(Counter(item["source_id"] for item in rows).values()) == 2
    for item in rows:
        if "augmentation" not in item["metadata"]:
            assert item == original[item["id"]]
            continue
        restored = deepcopy(item)
        before = restored["metadata"].pop("augmentation")["original_occurrence_id"]
        restored["id"] = restored["row_id"] = before
        assert restored == original[before]
    manifest = checked_preparation_manifest(output)
    assert manifest["splits"]["train"]["accepted_rows"] == 32
    assert manifest["splits"]["train"]["output_sha256"] == file_sha256(output / "train.jsonl")
    assert manifest["augmentation_sha256"] == file_sha256(output / "augmentation.json")
    for name in ("val", "golden32", "golden35"):
        assert manifest["splits"][name] == json.loads((source / "manifest.json").read_text())["splits"][name]


def test_repeat_cap_is_absolute_and_family_links_are_transitive(tmp_path):
    rows = [row(f"common-{i}") for i in range(60)]
    rows += [row("image-a", component="Image", family="family-a", source="Shared first response"),
             row("image-b", component="Image", family="family-a", source="Shared second response"),
             row("image-c", component="Image", family="family-c", source="Shared second response"),
             row("image-d", component="Image")]
    source = prepare(tmp_path, rows)
    report = augment(source, tmp_path / "augmented")
    assert report["source_families"] == 62
    assert report["added_rows"] == 1
    assert report["selections"][0]["original_occurrence_id"] == "image-d"
    assert report["component_exposure"]["Image"]["source_families"] == 2


def test_default_support_floor_excludes_tiny_component_cohorts(tmp_path):
    source = prepare(tmp_path)
    report = augment_prepared_training(source, tmp_path / "augmented")
    assert report["added_rows"] == 0
    assert not report["eligible_components"]


def test_budget_and_seed_are_reproducible(tmp_path):
    source = prepare(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    first = augment(source, a, max_extra_fraction=0.04, seed=101)
    second = augment(source, b, max_extra_fraction=0.04, seed=101)
    assert first == second and first["added_rows"] == 1
    assert (a / "train.jsonl").read_bytes() == (b / "train.jsonl").read_bytes()
    assert (a / "manifest.json").read_bytes() == (b / "manifest.json").read_bytes()


def test_zero_budget_and_cap_one_add_nothing(tmp_path):
    source = prepare(tmp_path)
    assert augment(source, tmp_path / "zero", max_extra_fraction=0)["added_rows"] == 0
    assert augment(source, tmp_path / "one", max_family_copies=1)["added_rows"] == 0


def test_tampered_training_fails_before_any_output(tmp_path):
    source = prepare(tmp_path)
    with (source / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="hash differs"):
        augment(source, tmp_path / "augmented")
    assert not (tmp_path / "augmented").exists()


def test_existing_nested_and_recursive_outputs_rejected(tmp_path):
    source = prepare(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        augment(source, source / "augmented")
    output = tmp_path / "augmented"
    augment(source, output)
    with pytest.raises(FileExistsError):
        augment(source, output)
    with pytest.raises(ValueError, match="recursively"):
        augment(output, tmp_path / "second")


def test_real_golden_preparation_remains_valid_with_actual_training_copies(tmp_path):
    prepare(tmp_path)
    from ir_training.data.shared_prompt import create_shared_prompt_contract
    from ir_training.pipeline.golden_training import GOLDENS
    source = tmp_path / "checked"
    prepare_splits({**{name: tmp_path / "raw" / f"{name}.jsonl" for name in ("train", "val")},
                    **{name: ROOT / values[0] for name, values in GOLDENS.items()}}, source,
                   ordering="root-first", tokenizer=Tokenizer(), max_seq_length=4096, max_input_tokens=4096,
                   evaluation_splits={"golden32", "golden35"}, shared_prompt=create_shared_prompt_contract())
    output = tmp_path / "augmented"
    report = augment(source, output)
    assert report["added_rows"] == 2
    spec = importlib.util.spec_from_file_location("augmentation_fixture_preparer", ROOT / "training/scripts/prepare_review_training.py")
    preparer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparer)
    verified = preparer.verify_prepared(output, source / "golden32.jsonl", max_sequence=4096,
                                       max_prompt=4096, golden35=source / "golden35.jsonl")
    assert verified["split_rows"]["train"] == 32
    assert verified["golden_unique_sources"] == 31
    assert verified["final_evaluation_datasets"]["golden35"]["selection_role"] == "final_only_holdout"


@pytest.mark.parametrize("kwargs", [
    {"max_extra_fraction": -0.1}, {"max_extra_fraction": 0.51}, {"max_extra_fraction": float("nan")},
    {"max_extra_fraction": True}, {"max_family_copies": 0}, {"max_family_copies": 6},
    {"max_family_copies": True}, {"rare_frequency": 0}, {"rare_frequency": 0.11},
    {"min_component_families": 1}, {"min_component_families": True},
])
def test_unsafe_augmentation_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        validate_augmentation_options(**kwargs)
