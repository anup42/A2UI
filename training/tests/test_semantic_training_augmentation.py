"""Startup augmentation keeps teacher output behind provenance and split gates."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.data.chat_templates import build_messages, build_prompt
from ir_training.data.express_preparation import prepare_splits
from ir_training.data.shared_prompt import create_shared_prompt_contract
from ir_training.eval.prepared_contract import checked_preparation_manifest, file_sha256


class Tokenizer:
    name_or_path, chat_template = "semantic-augmentation-fixture", "role-content"
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def get_vocab(self):
        return {"fixture": 0}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        return "".join(f"{item['role']}:\n{item['content']}\n" for item in messages) + (
            "assistant:\n" if add_generation_prompt else ""
        )

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": list(range(len(text.split())))}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_row(identity, *, family=None, source=None):
    source = source or f"Warehouse {identity} has 12 sealed boxes ready for collection."
    completion = f"<a2ui>\nroot=Text({json.dumps(source)})\n</a2ui>"
    return {
        "id": identity, "row_id": identity, "source_id": family or identity,
        "response_id": f"response-{identity}", "response_text": source,
        "messages": build_messages("Fixture system", source, completion, target_format="a2ui_express_v1"),
        "prompt": build_prompt("Fixture system", source, target_format="a2ui_express_v1"),
        "completion": completion, "metadata": {},
    }


def make_plan(tmp_path, *, rows=None):
    raw = tmp_path / "raw"
    raw.mkdir()
    train = rows if rows is not None else [source_row(f"train-{i}") for i in range(20)]
    write_jsonl(raw / "train.jsonl", train)
    splits = ("train", "val", "golden32", "golden35", "bixby50")
    for name in splits[1:]:
        write_jsonl(raw / f"{name}.jsonl", [source_row(f"{name}-unique")])
    shared = create_shared_prompt_contract(ordering="root-first")
    prepare_splits(
        {name: raw / f"{name}.jsonl" for name in splits}, tmp_path / "prepared",
        ordering="root-first", tokenizer=Tokenizer(), max_seq_length=4096,
        max_input_tokens=4096, shared_prompt=shared, evaluation_splits=set(splits[2:]),
    )
    return {
        "options": {
            "output_dir": str(tmp_path), "model_dir": str(tmp_path / "model"), "profile": "270m",
            "seed": 42, "max_seq_length": 4096, "max_input_tokens": 4096,
            "prepare_workers": 1, "progress_seconds": 10, "augmentation": "semantic",
            "augmentation_max_extra_fraction": 0.10, "augmentation_max_family_repeats": 2,
            "augmentation_teacher_model": "muse_glimmer_30b_sglang_reasoning_dflash",
            "augmentation_python": None, "augmentation_max_samples": 500,
            "augmentation_timeout_seconds": 7200,
        },
        "shared_prompt": shared,
        "goldens": {name: {"path": str(raw / f"{name}.jsonl")} for name in splits[2:]},
    }


def candidate(donor, index=0, *, source=None):
    source = source or f"Warehouse variant {donor['donor_id']} item {index} has 24 sealed boxes ready for collection."
    identity = f"semantic-test-{donor['donor_id']}-{index}"
    completion = f"<a2ui>\nroot=Text({json.dumps(source)})\n</a2ui>"
    return {
        "ui_id": identity, "response_id": identity + "-response", "query_id": identity + "-query",
        "response_text": source, "a2ui_express": completion, "completion": completion,
        "source_format": "a2ui_express_v1",
        "record_status": "accepted", "validation": {"schema_valid_strict": True, "standard_a2ui_valid": True},
        "metrics": {"standard_a2ui_valid": True},
        "training_acceptance": {"eligible": True, "blocking_reasons": [], "review_reasons": []},
        "gen": {"model": "meta-models/Muse-Glimmer-30B", "error": None, "completion_complete": True},
        "augmentation": {
            "version": "training-augmentation-v1", "category": "coherent_numeric_entities",
            "donor_id": donor["donor_id"], "source_group_id": donor["source_group_id"],
            "seed": 42, "candidate_seed": 42 + index,
            "source_sha256": sha(source), "donor_source_sha256": sha(donor["response_text"]),
            "teacher_model": "muse_glimmer_30b_sglang_reasoning_dflash", "teacher_prompt_sha256": "1" * 64,
        },
    }


class Teacher:
    def __init__(self, transform=None, manifest_transform=None):
        self.transform = transform
        self.manifest_transform = manifest_transform
        self.calls = []
        self.donors = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        donors_path = Path(command[command.index("--donors") + 1])
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        self.donors = list(read_jsonl(donors_path))
        records = [candidate(donor) for donor in self.donors]
        if self.transform is not None:
            records = self.transform(records, self.donors)
        accepted = output / "accepted_genui.jsonl"
        write_jsonl(accepted, records)
        manifest = {
            "version": "training-augmentation-v1", "status": "completed", "seed": 42,
            "teacher_model": "muse_glimmer_30b_sglang_reasoning_dflash", "teacher_prompt_sha256": "1" * 64,
            "donors_sha256": file_sha256(donors_path), "accepted_genui_sha256": file_sha256(accepted),
            "accepted_rows": len(records), "attempted_rows": len(self.donors),
        }
        if self.manifest_transform is not None:
            self.manifest_transform(manifest)
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return 0


def run(plan, teacher):
    from ir_training.data.semantic_augmentation import augment_training_at_startup
    return augment_training_at_startup(plan, tokenizer_loader=lambda *args: Tokenizer(), command_runner=teacher)


def test_semantic_startup_preserves_originals_and_evaluation_bytes(tmp_path):
    plan = make_plan(tmp_path)
    before = {path.name: path.read_bytes() for path in (tmp_path / "prepared").iterdir()}
    teacher = Teacher()
    run(plan, teacher)
    assert len(teacher.calls) == 1
    assert teacher.donors and all(row["split"] == "train" for row in teacher.donors)
    original = list(read_jsonl(tmp_path / "prepared/train.jsonl"))
    augmented = list(read_jsonl(tmp_path / "augmented/train.jsonl"))
    assert 20 < len(augmented) <= 22
    by_id = {row["id"]: row for row in augmented}
    for row in original:
        assert by_id[row["id"]] == row
    original_by_id = {row["id"]: row for row in original}
    for row in augmented:
        provenance = row["metadata"].get("augmentation")
        if provenance is not None:
            donor = original_by_id[provenance["donor_id"]]
            assert by_id[donor["id"]] == donor
            assert row["source_id"] == donor["source_id"]
    for name, contents in before.items():
        assert (tmp_path / "prepared" / name).read_bytes() == contents
        if name not in {"train.jsonl", "manifest.json"}:
            assert (tmp_path / "augmented" / name).read_bytes() == contents
    manifest = checked_preparation_manifest(tmp_path / "augmented")
    assert manifest["splits"]["train"]["accepted_rows"] == len(augmented)
    assert manifest["splits"]["train"]["output_sha256"] == file_sha256(tmp_path / "augmented/train.jsonl")


def test_zero_teacher_candidates_fails_without_publishing(tmp_path):
    plan = make_plan(tmp_path)
    with pytest.raises(ValueError, match="No accepted semantic"):
        run(plan, Teacher(lambda records, donors: []))
    assert not (tmp_path / "augmented").exists()


@pytest.mark.parametrize("field,value", [
    ("donor_id", "unknown-training-donor"), ("source_group_id", "foreign-source-family"),
    ("source_sha256", "0" * 64), ("donor_source_sha256", "0" * 64),
    ("teacher_model", "unapproved-teacher"), ("teacher_prompt_sha256", "missing"),
])
def test_tampered_candidate_provenance_cannot_enter_training(tmp_path, field, value):
    plan = make_plan(tmp_path)

    def tamper(records, donors):
        for row in records:
            row["augmentation"][field] = value
        return records

    reason = "Unknown or repeated" if field == "donor_id" else (
        "teacher-prompt hash" if field == "teacher_prompt_sha256" else "lineage/category/teacher binding"
    )
    with pytest.raises(ValueError, match=reason):
        run(plan, Teacher(tamper))
    assert not (tmp_path / "augmented").exists()


@pytest.mark.parametrize("split", ["val", "golden32", "golden35", "bixby50"])
def test_new_candidate_matching_evaluation_text_is_rejected(tmp_path, split):
    plan = make_plan(tmp_path)
    reserved = next(read_jsonl(tmp_path / "prepared" / f"{split}.jsonl"))["response_text"]

    def leak(records, donors):
        return [candidate(donors[0], source=reserved)]

    with pytest.raises(ValueError, match="reserved-source/strict validation"):
        run(plan, Teacher(leak))
    assert not (tmp_path / "augmented").exists()
    quarantine = list(read_jsonl(tmp_path / "semantic_augmentation/candidate_quarantine.jsonl"))
    assert [row["reason"] for row in quarantine] == ["reserved_evaluation_source"]


def test_token_overflow_rejects_whole_candidate_without_clipping(tmp_path):
    plan = make_plan(tmp_path)
    long_source = "A complete warehouse record. " * 5000
    teacher = Teacher(lambda records, donors: [candidate(donors[0], source=long_source)])
    with pytest.raises(ValueError, match="Split train has no accepted rows"):
        run(plan, teacher)
    assert not (tmp_path / "augmented").exists()


def test_already_full_source_family_never_becomes_teacher_donor(tmp_path):
    rows = [source_row(f"ordinary-{i}") for i in range(18)]
    rows += [source_row("full-a", family="already-full"), source_row("full-b", family="already-full")]
    plan = make_plan(tmp_path, rows=rows)
    teacher = Teacher()
    run(plan, teacher)
    assert not {"full-a", "full-b"} & {row["donor_id"] for row in teacher.donors}
    augmented = list(read_jsonl(tmp_path / "augmented/train.jsonl"))
    assert max(Counter(row["source_id"] for row in augmented).values()) <= 2


def test_prepared_train_tampering_stops_before_teacher(tmp_path):
    plan = make_plan(tmp_path)
    with (tmp_path / "prepared/train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    teacher = Teacher()
    with pytest.raises(ValueError, match="hash differs"):
        run(plan, teacher)
    assert teacher.calls == []
    assert not (tmp_path / "augmented").exists()


def test_donor_source_mutated_during_teacher_run_is_not_published(tmp_path):
    plan = make_plan(tmp_path)

    def mutate(records, donors):
        with (tmp_path / "prepared/val.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        return records

    with pytest.raises(ValueError, match="Prepared source changed"):
        run(plan, Teacher(mutate))
    assert not (tmp_path / "augmented").exists()


def test_no_eligible_families_does_not_call_teacher(tmp_path):
    rows = [source_row(f"family-{family}-{item}", family=f"family-{family}")
            for family in range(10) for item in range(2)]
    plan = make_plan(tmp_path, rows=rows)
    teacher = Teacher()
    with pytest.raises(ValueError, match="No eligible train-only donor"):
        run(plan, teacher)
    assert not teacher.calls
    assert not (tmp_path / "augmented").exists()


def test_tokenizer_drift_rejects_candidate_publication(tmp_path):
    from ir_training.data.semantic_augmentation import augment_training_at_startup

    class ChangedTokenizer(Tokenizer):
        def get_vocab(self):
            return {"different-token": 0}

    plan = make_plan(tmp_path)
    with pytest.raises(ValueError, match="tokenizer differs.*vocabulary_sha256"):
        augment_training_at_startup(plan, tokenizer_loader=lambda *args: ChangedTokenizer(), command_runner=Teacher())
    assert not (tmp_path / "augmented").exists()


@pytest.mark.parametrize("key,value", [
    ("augmentation_max_extra_fraction", float("nan")),
    ("augmentation_timeout_seconds", float("nan")),
    ("augmentation_timeout_seconds", float("inf")),
    ("augmentation_max_samples", True),
    ("augmentation_max_family_repeats", True),
])
def test_invalid_semantic_limits_fail_before_reading_or_generation(tmp_path, key, value):
    from ir_training.data.semantic_augmentation import augment_training_at_startup

    options = {"augmentation": "semantic", "output_dir": str(tmp_path), key: value}
    teacher = Teacher()
    with pytest.raises(ValueError, match="augmentation|max_extra_fraction|max_family_copies"):
        augment_training_at_startup({"options": options}, command_runner=teacher)
    assert not teacher.calls
    assert not (tmp_path / "semantic_augmentation").exists()


def test_repeated_candidate_donor_cannot_bypass_family_cap(tmp_path):
    plan = make_plan(tmp_path)
    teacher = Teacher(lambda records, donors: [candidate(donors[0], index=i) for i in range(2)])
    with pytest.raises(ValueError, match="Unknown or repeated augmentation donor"):
        run(plan, teacher)
    assert not (tmp_path / "augmented").exists()


def test_transitive_source_aliases_share_one_family_exposure_cap(tmp_path):
    shared = "The connected warehouse contains 12 sealed boxes."
    rows = [
        source_row("chain-a", family="first-family"),
        source_row("chain-b", family="first-family", source=shared),
        source_row("chain-c", family="second-family", source=shared),
    ]
    rows += [source_row(f"full-{family}-{item}", family=f"full-{family}")
             for family in range(8) for item in range(2)]
    rows.append(source_row("only-eligible"))
    plan = make_plan(tmp_path, rows=rows)
    teacher = Teacher()
    report = run(plan, teacher)
    assert report["source_families"] == 10
    assert [row["donor_id"] for row in teacher.donors] == ["only-eligible"]
    augmented = list(read_jsonl(tmp_path / "augmented/train.jsonl"))
    originals = {row["id"]: row for row in read_jsonl(tmp_path / "prepared/train.jsonl")}
    assert {row["id"]: row for row in augmented if row["id"] in originals} == originals


def test_total_added_token_budget_caps_individually_admissible_candidates(tmp_path):
    plan = make_plan(tmp_path)

    def longer(records, donors):
        return [candidate(donor, source=f"Synthetic warehouse {donor['donor_id']}. " + "Complete record detail. " * 130)
                for donor in donors]

    report = run(plan, Teacher(longer))
    assert report["stage3_accepted_rows"] == 2
    assert report["candidate_tokenization"]["accepted_rows"] == 2
    assert report["added_rows"] == report["token_budget_rejected_rows"] == 1
    assert 0 < report["added_tokens"] <= report["extra_token_budget"]
    assert report["extra_token_budget"] == int(report["original_tokens"] * 0.10)
    budget_rejections = list(read_jsonl(tmp_path / "semantic_augmentation/token_budget_quarantine.jsonl"))
    assert budget_rejections[0]["reason"] == "augmentation_token_budget_exceeded"
    assert report["added_tokens"] + budget_rejections[0]["tokens"] > report["extra_token_budget"]


@pytest.mark.parametrize("field", ["query_id", "response_id", "ui_id"])
def test_raw_stage3_holdout_identity_rejected_before_new_id_assignment(tmp_path, field):
    plan = make_plan(tmp_path)
    reserved = next(read_jsonl(tmp_path / "prepared/golden35.jsonl"))["response_id"]

    def tamper(records, donors):
        records[0][field] = reserved
        return records

    with pytest.raises(ValueError, match="reserved|holdout|evaluation"):
        run(plan, Teacher(tamper))
    assert not (tmp_path / "augmented").exists()


@pytest.mark.parametrize("key,value", [
    ("version", "unknown-producer-v999"), ("status", "running"), ("status", "failed"),
    ("teacher_model", "unexpected-teacher"), ("seed", 999),
    ("teacher_prompt_sha256", "2" * 64),
    ("accepted_genui_sha256", "0" * 64), ("donors_sha256", "0" * 64),
])
def test_generator_manifest_binding_is_required(tmp_path, key, value):
    plan = make_plan(tmp_path)
    teacher = Teacher(manifest_transform=lambda manifest: manifest.update({key: value}))
    with pytest.raises(ValueError, match="manifest|binding|hash|teacher|version|completed"):
        run(plan, teacher)
    assert not (tmp_path / "augmented").exists()


@pytest.mark.parametrize("field,value", [
    ("version", "unknown-candidate-v999"), ("teacher_prompt_sha256", "2" * 64),
])
def test_candidate_provenance_must_match_generator_manifest(tmp_path, field, value):
    plan = make_plan(tmp_path)

    def tamper(records, donors):
        records[0]["augmentation"][field] = value
        return records

    with pytest.raises(ValueError, match="version|binding|hash|lineage"):
        run(plan, Teacher(tamper))
    assert not (tmp_path / "augmented").exists()


def test_real_dataset_generator_outputs_feed_training_startup(tmp_path):
    """Exercise the actual producer/consumer boundary with only model calls faked."""
    plan = make_plan(tmp_path)
    from llm.base import LLMResult
    from llm.factory import load_model_specs
    from pipeline.training_augmentation import (
        DEFAULT_TEACHER,
        generate_training_augmentations,
    )
    from utils.config import load_yaml

    class FakeAdapter:
        def __init__(self):
            self.spec = next(item for item in load_model_specs(load_yaml(ROOT / "dataset/configs/models.yaml"))
                             if item.name == DEFAULT_TEACHER)
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            context = json.loads(kwargs["prompt"].split("\n")[1])
            category = context["category"]
            if kwargs["prompt"].startswith("TASK: GENERATE_SOURCE"):
                result = {"category": category, "synthetic": True,
                          "response_text": f"Synthetic depot {kwargs['seed']} contains 24 sealed boxes."}
            else:
                result = {"category": category, "approved": True, "coherent": True,
                          "category_satisfied": True, "synthetic_provenance_clear": True, "issues": []}
            return LLMResult(text=json.dumps(result), raw={}, latency_ms=1, input_tokens=20, output_tokens=20,
                             cost_usd=None, model=self.spec.model, provider=self.spec.provider,
                             finish_reason="stop", completion_complete=True)

    adapter = FakeAdapter()

    def stage3(**kwargs):
        records = []
        for response in read_jsonl(kwargs["responses_path"]):
            # Only Stage 3 inference is stubbed: producer creates the real lineage.
            source = response["response_text"]
            completion = f"<a2ui>\nroot=Text({json.dumps(source)})\n</a2ui>"
            records.append({
                **response, "ui_id": "ui-" + response["response_id"], "source_format": "a2ui_express_v1",
                "completion": completion, "a2ui_express": completion, "record_status": "accepted",
                "training_acceptance": {"eligible": True, "blocking_reasons": [], "review_reasons": []},
                "validation": {"schema_valid_strict": True, "standard_a2ui_valid": True},
                "gen": {"completion_complete": True, "error": None},
            })
        write_jsonl(kwargs["genui_path"], records)

    def command_runner(command, **kwargs):
        return generate_training_augmentations(
            Path(command[command.index("--donors") + 1]), Path(command[command.index("--output-dir") + 1]),
            teacher_model=command[command.index("--teacher-model") + 1],
            max_new_samples=int(command[command.index("--max-new-samples") + 1]),
            seed=int(command[command.index("--seed") + 1]), adapter=adapter, stage3_runner=stage3,
        )

    report = run(plan, command_runner)
    assert report["stage3_accepted_rows"] == report["added_rows"] == 2
    generated = list(read_jsonl(tmp_path / "semantic_augmentation/generated/accepted_genui.jsonl"))
    assert {row["augmentation"]["seed"] for row in generated} == {42}
    assert {row["augmentation"]["candidate_seed"] for row in generated} == {42, 43}
    assert len(adapter.calls) == 4


@pytest.mark.parametrize("mode", ["none", "rare_components"])
def test_legacy_nonsemantic_plan_continues_with_only_new_defaults(tmp_path, mode):
    from dataclasses import replace

    from ir_training.pipeline import golden_training as workflow
    from ir_training.pipeline.golden_training import GoldenTrainingOptions

    model, inputs = tmp_path / "model", tmp_path / "inputs"
    model.mkdir()
    inputs.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture, never loaded")
    for name in ("train", "val"):
        write_jsonl(inputs / f"{name}.jsonl", [source_row(name)])
    options = GoldenTrainingOptions(model, tmp_path / "run", input_dir=inputs,
                                    augmentation=mode, steps=20, prepare_workers=1)
    plan = workflow.build_plan(options)
    for name in ("augmentation_teacher_model", "augmentation_python", "augmentation_max_samples", "augmentation_timeout_seconds"):
        del plan["options"][name]
    options.output_dir.mkdir()
    completed = {name: {"files": {}} for name in ("prepare", "augment") if name in plan["stages"]}
    workflow._write(options.output_dir / "pipeline_manifest.json", {
        "plan": plan, "completed": completed, "attempts": {}, "active_stage": None,
    })
    assert workflow.run_pipeline(options, prepare_only=True, continue_run=True)["status"] == "prepared"
    for override in ({"augmentation_max_samples": 42}, {"augmentation_teacher_model": "other-teacher"},
                     {"augmentation_timeout_seconds": 60}, {"augmentation_python": model / "python"}, {"epochs": 2}):
        with pytest.raises(ValueError, match="Workflow options or prompt contract changed"):
            workflow.run_pipeline(replace(options, **override), prepare_only=True, continue_run=True)
