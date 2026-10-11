"""CPU regression tests. Repository boundaries are mocked explicitly where noted."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
sys.path.insert(0, str(ROOT / "dataset/src"))

from ir_training.data.student_feedback import (
    FeedbackOptions, build_feedback, checked_score, exposure_schedule, fingerprint,
    select_probe_indices, text_hash,
)
from ir_training.pipeline import student_feedback as pipeline
from ir_training.train.feedback_objective import (
    FeedbackCollator, completion_logps, encode_completion, feedback_loss,
)
from ir_training.train.student_feedback import FeedbackDataset, preserve_rng


def row(i, *, tokens=100):
    return {"id": f"r{i}", "source_id": f"s{i}", "response_text": f"source {i}",
            "completion": f"good:{i}", "intent_bucket": "table" if i % 2 else "text",
            "messages": [{"role": "user", "content": f"source {i}"}, {"role": "assistant", "content": f"good:{i}"}],
            "metadata": {"express_preparation": {"tokenization": {"sequence_tokens": tokens}}}}


def score(source, text):
    assert text.split(":")[-1] == source["id"][1:], "scorer must receive exactly the same source"
    kind = text.split(":")[0]
    return {"eligible": kind in {"good", "alternative"}, "strict_valid": kind != "invalid",
            "quality": {"good": 1., "alternative": .8, "invalid": .1, "ambiguous": .98, "omission": .7}[kind],
            "blocking_reasons": ["raw_express_invalid"] if kind == "invalid" else [],
            "review_reasons": ["content_unit_fidelity"] if kind in {"ambiguous", "omission"} else [],
            "metric_fingerprint": "frozen-test-metric"}


def obs(rows, i, kind="invalid", stop="closing_sentinel"):
    return {"train_index": i, "row_sha256": fingerprint(rows[i]), "prompt_sha256": text_hash(f"prompt:{i}"),
            "completion": f"{kind}:{i}", "stop_reason": stop}


def build(rows, observations, families=None, **options):
    return build_feedback(rows, families or [str(i) for i in range(len(rows))], observations,
                          score=score, options=FeedbackOptions(**options))


@pytest.mark.parametrize("change", [
    {"seed": True}, {"max_extra_token_fraction": 0}, {"max_extra_token_fraction": .51},
    {"max_extra_token_fraction": float("nan")}, {"max_extra_token_fraction": True},
    {"max_replays_per_family": 0}, {"max_replays_per_family": True},
    {"min_quality_gap": 0}, {"min_quality_gap": float("inf")},
])
def test_policy_rejects_invalid_controls(change):
    with pytest.raises(ValueError):
        FeedbackOptions(**change)


def test_valid_alternative_is_not_negative_and_inputs_are_unchanged():
    rows = [row(i) for i in range(20)]
    original = deepcopy(rows)
    result = build(rows, [obs(rows, 0, "alternative"), obs(rows, 1)], max_extra_token_fraction=.5)
    assert result["replay_indices"] == [1]
    assert result["preferences"][0]["chosen"] == "good:1"
    assert result["preferences"][0]["rejected"] == "invalid:1"
    assert rows == original
    assert result["report"]["targets_modified"] is False


@pytest.mark.parametrize("stop", ["max_new_tokens", "unknown", None])
def test_truncated_or_unexplained_stops_are_replay_only(stop):
    rows = [row(i) for i in range(20)]
    result = build(rows, [obs(rows, 1, stop=stop)])
    assert result["replay_indices"] == [1]
    assert result["preferences"] == []


def test_semantic_margin_filters_ambiguous_pairs_but_keeps_replay():
    rows = [row(i) for i in range(20)]
    result = build(rows, [obs(rows, 0, "ambiguous"), obs(rows, 1, "omission")], max_extra_token_fraction=.5)
    assert set(result["replay_indices"]) == {0, 1}
    assert [p["train_index"] for p in result["preferences"]] == [1]


def test_ineligible_reference_never_becomes_replay_or_preferred():
    rows = [row(i) for i in range(20)]
    rows[1]["completion"] = "invalid:1"
    result = build(rows, [obs(rows, 1)])
    assert result["replay_indices"] == result["preferences"] == []
    assert result["diagnostics"][0]["status"] == "reference_not_eligible"


@pytest.mark.parametrize("mutation", [
    {"train_index": -1}, {"train_index": 20}, {"train_index": True},
    {"row_sha256": "wrong"}, {"prompt_sha256": "x" * 64}, {"completion": None},
])
def test_source_and_prompt_binding_fail_closed(mutation):
    rows = [row(i) for i in range(20)]
    with pytest.raises(ValueError):
        build(rows, [{**obs(rows, 0), **mutation}])


def test_duplicate_probe_rows_rejected():
    rows = [row(i) for i in range(20)]
    with pytest.raises(ValueError, match="unique"):
        build(rows, [obs(rows, 0), obs(rows, 0)])


def test_scorer_failure_is_not_converted_to_a_training_negative():
    rows = [row(i) for i in range(20)]
    def broken(*args):
        raise RuntimeError("validator unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        build_feedback(rows, list(map(str, range(20))), [obs(rows, 0)], score=broken, options=FeedbackOptions())


@pytest.mark.parametrize("change", [{"quality": float("nan")}, {"quality": 2.}, {"quality": True},
                                     {"metric_fingerprint": ""}, {"blocking_reasons": "bad"}, {"strict_valid": False}])
def test_invalid_scorer_contract_is_rejected(change):
    with pytest.raises(ValueError):
        checked_score({**score(row(0), "good:0"), **change})


def test_probe_selection_is_deterministic_and_family_deduplicated():
    rows = [row(i, tokens=100 + 1000*(i % 3)) for i in range(20)]
    families = [str(i//2) for i in range(20)]
    a = select_probe_indices(rows, families, max_rows=9, seed=42)
    assert a == select_probe_indices(rows, families, max_rows=9, seed=42)
    assert len({families[i] for i in a}) == len(a) == 9
    assert {rows[i]["metadata"]["express_preparation"]["tokenization"]["sequence_tokens"]//1024 for i in a} == {0, 1, 2}


def test_replay_token_and_family_caps_are_real():
    rows = [row(i, tokens=100 if i else 2000) for i in range(20)]
    families = ["same" if i < 3 else str(i) for i in range(20)]
    result = build(rows, [obs(rows, i) for i in range(20)], families=families)
    assert result["report"]["replay_sequence_tokens"] <= result["report"]["extra_sequence_token_budget"]
    assert len([i for i in result["replay_indices"] if families[i] == "same"]) <= 1
    assert 0 not in result["replay_indices"]  # Too long for this replay budget, never truncated.


def test_actual_training_schedule_exposes_hard_examples_without_more_steps():
    rows = [row(i) for i in range(100)]
    args = dict(total_examples=200, seed=42, max_hard_token_fraction=.1, max_hard_exposures_per_family=3)
    a = exposure_schedule(rows, list(map(str, range(100))), [0, 1, 2], **args)
    assert a == exposure_schedule(rows, list(map(str, range(100))), [0, 1, 2], **args)
    assert len(a["baseline_indices"]) == len(a["replay_indices"]) == 200
    assert sum(a["hard_mask"]) == 9
    assert a["report"]["hard_sequence_tokens"] <= a["report"]["hard_token_budget"]
    for ordinary, replay, hard in zip(a["baseline_indices"], a["replay_indices"], a["hard_mask"]):
        assert replay in {0, 1, 2} if hard else ordinary == replay


class TinyTokenizer:
    bos_token_id = 1
    pad_token_id = 2
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize
        return "^" + "".join("<" + m["role"] + ">" + m["content"] + "~" for m in messages) + ("<assistant>" if add_generation_prompt else "")
    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(x) + 3 for x in text]}


def test_tokenization_preserves_chat_boundary_and_native_template_ending():
    source = row(0)
    before = deepcopy(source)
    encoded = encode_completion(source, TinyTokenizer(), max_length=1000)
    assert encoded["labels"][:len(encoded["prompt"])] == [-100] * len(encoded["prompt"])
    assert encoded["labels"][-1] == ord("~") + 3
    rejected = encode_completion(source, TinyTokenizer(), max_length=1000, completion="invalid:0")
    assert rejected["prompt"] == encoded["prompt"]
    assert source == before
    with pytest.raises(ValueError, match="truncation"):
        encode_completion(source, TinyTokenizer(), max_length=3)
    with pytest.raises(ValueError, match="boundaries"):
        encode_completion(source, TinyTokenizer(), max_length=1000, template_kwargs={"tokenize": True})


def test_mismatched_saved_assistant_is_rejected():
    source = row(0)
    source["messages"][-1]["content"] = "other"
    with pytest.raises(ValueError, match="exact"):
        encode_completion(source, TinyTokenizer(), max_length=1000)


def test_collator_never_masks_real_eos_when_pad_equals_eos():
    pytest.importorskip("torch")
    features = [{"chosen": {"input_ids": [1, 3, 2], "labels": [-100, 3, 2]}},
                {"chosen": {"input_ids": [1, 4, 4, 2], "labels": [-100, 4, 4, 2]}}]
    batch = FeedbackCollator(2)(features)
    assert batch["labels"][0].tolist() == [-100, 3, 2, -100]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0]
    assert batch["reference"].shape == (0, 2)


def test_pair_collation_requires_finite_frozen_references():
    pytest.importorskip("torch")
    encoded = {"input_ids": [1, 3], "labels": [-100, 3]}
    for ref in (None, (0., float("nan")), (True, 0.), (0.,)):
        with pytest.raises(ValueError, match="reference"):
            FeedbackCollator(2)([{"chosen": encoded, "rejected": encoded, "reference": ref}])


def test_chunked_logps_match_cross_entropy_and_mask_prompt_and_padding_gradients():
    torch = pytest.importorskip("torch")
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, -100], [-100, 1, 2, 3, 4]])
    sums, counts = completion_logps(logits, labels, chunk_tokens=1)
    expected = -torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1),
                                                   ignore_index=-100, reduction="none").reshape(2, 4).sum(1)
    assert torch.allclose(sums, expected, atol=1e-6)
    assert counts.tolist() == [2., 4.]
    sums.sum().backward()
    assert not bool(logits.grad[0, 0].any())
    assert not bool(logits.grad[0, 3:].any())
    assert bool(logits.grad[0, 1:3].any())


def test_dpo_is_sequence_sum_not_length_normalized_and_reference_is_frozen():
    torch = pytest.importorskip("torch")
    values = torch.tensor([-6., -8.], requires_grad=True)
    refs = torch.tensor([[-6., -8.]], requires_grad=True)
    loss, metrics = feedback_loss(values, torch.tensor([2., 4.]), 1, torch.tensor([0]), refs, beta=.1)
    assert float(loss.detach()) == pytest.approx(3. + math.log(2.))
    loss.backward()
    assert float(values.grad[0]) == pytest.approx(-.55)
    assert float(values.grad[1]) == pytest.approx(.05)
    assert refs.grad is None
    assert float(metrics["rejected_tokens"]) == 4.


def test_mixed_sft_pair_loss_matches_two_accumulated_microbatches():
    torch = pytest.importorskip("torch")
    values = torch.tensor([-4., -6., -9.], requires_grad=True)
    counts = torch.tensor([2., 3., 5.])
    ref = torch.tensor([[-7., -8.]])
    joint, _ = feedback_loss(values, counts, 2, torch.tensor([1]), ref)
    a, _ = feedback_loss(values[:1], counts[:1], 1, torch.empty(0, dtype=torch.long), torch.empty((0, 2)))
    b, _ = feedback_loss(values[1:], counts[1:], 1, torch.tensor([0]), ref)
    assert torch.allclose(joint, (a + b)/2)
    assert torch.allclose(torch.autograd.grad(joint, values)[0], torch.autograd.grad((a+b)/2, values)[0])


@pytest.mark.parametrize("bad", ["logps", "counts", "reference"])
def test_nonfinite_loss_inputs_are_rejected(bad):
    torch = pytest.importorskip("torch")
    values, counts, refs = torch.tensor([-4., -5.]), torch.tensor([2., 2.]), torch.tensor([[-4., -5.]])
    {"logps": values, "counts": counts, "reference": refs}[bad].flatten()[0] = float("inf")
    with pytest.raises(ValueError):
        feedback_loss(values, counts, 1, torch.tensor([0]), refs)


def test_dataset_applies_preferences_only_to_replay_occurrences():
    rows = [row(i) for i in range(2)]
    def encode(source, **kwargs):
        return encode_completion(source, TinyTokenizer(), max_length=1000, **kwargs)
    pair = {"row_sha256": fingerprint(rows[0]), "chosen": rows[0]["completion"], "rejected": "invalid:0",
            "prompt_sha256": text_hash(encode(rows[0])["prompt"])}
    ds = FeedbackDataset(rows, [0, 0], [False, True], encode, pairs={0: pair}, references={0: (-4., -5.)})
    assert "rejected" not in ds[0]
    assert ds[1]["reference"] == (-4., -5.)
    ds.references.clear()
    with pytest.raises(ValueError, match="not been computed"):
        ds[1]


def test_rng_scope_restores_state_even_on_failure():
    torch = pytest.importorskip("torch")
    import numpy as np
    import random
    py, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with preserve_rng():
            random.random(); np.random.rand(); torch.rand(1)
            raise RuntimeError("fixture")
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_indexed_rows_does_not_materialize_or_lose_unicode(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('\ufeff{"source":"₹ हिन्दी"}\n\n{"source":"second"}\n', encoding="utf-8")
    rows = pipeline.IndexedRows(path)
    assert len(rows) == 2
    assert rows[0]["source"] == "₹ हिन्दी"
    assert rows[-1]["source"] == "second"


@pytest.fixture
def repository_boundaries(monkeypatch, tmp_path):
    """Mock only repository IO/admission dependencies, not feedback decisions."""
    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
    def no_links(path):
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError("symlink")
    module("ir_training.common.cache_store", assert_no_links=no_links)
    def identity(r):
        return {"row_id": r["id"], "source_id": r.get("source_id"), "query_id": r.get("query_id")}
    def keys(r):
        return {str(v) for v in r.values() if v}
    def hashes(r):
        return {text_hash(" ".join(r["response_text"].split()))}
    def reserved(paths):
        ids, responses, evidence = set(), set(), []
        for path in paths:
            for r in pipeline.IndexedRows(path):
                ids |= keys(identity(r)); responses |= hashes(r)
            evidence.append({"path": str(path), "sha256": pipeline.sha_file(path)})
        return {"identities": ids, "responses": responses, "evidence": evidence}
    module("ir_training.data.audit_filter", _identity_keys=keys, _source_hashes=hashes, load_reserved_cohorts=reserved)
    module("ir_training.data.golden_replacement", source_identity_record=identity)
    module("ir_training.eval.prepared_contract", checked_preparation_manifest=lambda p: json.loads((p / "manifest.json").read_text()))
    module("ir_training.common.config", repo_root=lambda: tmp_path, load_yaml=lambda p: json.loads(Path(p).read_text()))
    module("ir_training.pipeline.golden_training", GOLDENS={"golden": ("golden.jsonl", 1, "fixture")})
    pipeline.write_rows(tmp_path / "golden.jsonl", [row(999)])
    source = tmp_path / "prepared"
    source.mkdir()
    rows = [row(i) for i in range(20)]
    pipeline.write_rows(source / "train.jsonl", rows)
    pipeline.write_rows(source / "val.jsonl", [row(888)])
    manifest = {"splits": {name: {"accepted_rows": count, "input_rows": count,
                    "output_sha256": pipeline.sha_file(source / f"{name}.jsonl")}
                    for name, count in (("train", 20), ("val", 1))}}
    pipeline.write_json(source / "manifest.json", manifest)
    return source, rows


def test_prepared_hash_and_source_family_leakage_guards(repository_boundaries):
    source, rows = repository_boundaries
    assert len(pipeline.checked_prepared(source)[0]) == 20
    rows[0]["metadata"]["augmentation"] = {"donor_id": "s999"}
    pipeline.write_rows(source / "train.jsonl", rows)
    with pytest.raises(ValueError, match="hash"):
        pipeline.checked_prepared(source)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["splits"]["train"]["output_sha256"] = pipeline.sha_file(source / "train.jsonl")
    pipeline.write_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="reserved"):
        pipeline.checked_prepared(source)


def test_coverage_guard_rejects_old_five_percent_subset(repository_boundaries):
    source, _ = repository_boundaries
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["splits"]["train"]["input_rows"] = 400
    pipeline.write_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="50%"):
        pipeline.checked_prepared(source)


def test_immutable_artifact_rejects_extra_missing_and_changed_bytes(repository_boundaries, tmp_path):
    directory = tmp_path / "artifact"
    directory.mkdir()
    for name in ("binding.json", "replay.json", "preferences.jsonl", "diagnostics.jsonl", "report.json"):
        (directory / name).write_text("{}\n")
    receipt = {"version": "student_feedback_v1", "complete": True, "files": pipeline.inventory(directory)}
    pipeline.write_json(directory / "manifest.json", receipt)
    pipeline.checked_artifact(directory, "manifest.json")
    (directory / "report.json").write_text('{"modified":true}\n')
    with pytest.raises(ValueError, match="changed"):
        pipeline.checked_artifact(directory, "manifest.json")
    (directory / "report.json").write_text("{}\n")
    (directory / "extra.json").write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        pipeline.checked_artifact(directory, "manifest.json")
    (directory / "extra.json").unlink()
    (directory / "report.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        pipeline.checked_artifact(directory, "manifest.json")


def test_output_cannot_overwrite_or_nest_under_inputs(repository_boundaries, tmp_path):
    source, _ = repository_boundaries
    with pytest.raises(ValueError, match="disjoint"):
        pipeline.new_output(source / "trial", source)
    output = tmp_path / "output"
    pipeline.new_output(output, source)
    with pytest.raises(FileExistsError):
        pipeline.new_output(output, source)
    (tmp_path / "link").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        pipeline.inventory(tmp_path / "link")


def test_loader_reuses_original_contract_without_editing_it(repository_boundaries, tmp_path, monkeypatch):
    source, _ = repository_boundaries
    config = tmp_path / "sft.json"
    original = {"model": {"model_id": "google/gemma-4-E2B-it-qat-mobile-transformers"},
                "training": {"method": "qat_lora_sft", "resume_from_checkpoint": "old", "resume_policy": "old"},
                "lora": {"r": 8}, "qat": {"scale_mode": "retained_mobile"}}
    pipeline.write_json(config, original)
    before = config.read_bytes()
    monkeypatch.setenv("WORLD_SIZE", "4")
    result = pipeline.loading_config(config, tmp_path / "adapter", source, tmp_path / "trial", accumulation=8)
    assert result["qat"] == original["qat"] and result["lora"] == original["lora"]
    assert result["training"]["expected_effective_batch_size"] == 32
    assert "resume_policy" not in result["training"]
    assert config.read_bytes() == before


def test_real_v54_scorer_uses_existing_admission_when_full_repo_is_available():
    pytest.importorskip("pipeline.genui_quality", reason="full repository scorer is not mounted in the partial review checkout")
    source = {"id": "fixture", "source_id": "fixture", "response_text": "Hello.",
              "completion": '<a2ui>root=Text("Hello.")</a2ui>'}
    actual = pipeline.make_scorer()
    assert actual(source, source["completion"])["strict_valid"] is True
    assert actual(source, "not an Express completion")["eligible"] is False


@pytest.fixture
def sealed_probe(repository_boundaries, tmp_path, monkeypatch):
    source, rows = repository_boundaries
    # Explicit mocks for source-context construction and serving-stop policy.
    module = ModuleType("ir_training.eval.generate")
    module.build_prediction_record = lambda r, text: {"source_context_sha256": fingerprint(r)}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module = ModuleType("ir_training.generation_policy")
    module.stop_express_completion = lambda text: text
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(pipeline, "make_scorer", lambda: score)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.txt").write_text("never loaded")
    config = tmp_path / "sft.json"
    pipeline.write_json(config, {"training": {"method": "qat_lora_sft"}})
    _, _, _, files, reserved = pipeline.checked_prepared(source)
    directory = tmp_path / "probe"
    directory.mkdir()
    indices = [0, 1]
    observations = [obs(rows, 0), obs(rows, 1, "alternative")]
    requests = [{key: o[key] for key in ("train_index", "row_sha256", "prompt_sha256")} for o in observations]
    plan = {"version": "student_feedback_v1", "source_dir": str(source), "source_files": files,
            "source_config": str(config), "source_config_sha256": pipeline.sha_file(config),
            "checkpoint": str(checkpoint), "checkpoint_files": pipeline.inventory(checkpoint),
            "reserved": reserved, "indices": indices, "max_new_tokens": 2048,
            "producer_sha256": {"pipeline": pipeline.sha_file(Path(pipeline.__file__)),
                "decisions": pipeline.sha_file(ROOT / "training/src/ir_training/data/student_feedback.py")}}
    pipeline.write_json(directory / "plan.json", plan)
    pipeline.write_rows(directory / "observations.jsonl", observations)
    pipeline.write_rows(directory / "inputs.jsonl", [rows[i] for i in indices])
    pipeline.write_rows(directory / "requests.jsonl", requests)
    predictions = [{"raw_generated_text": o["completion"], "generated_text": o["completion"],
                    "source_context_sha256": fingerprint(rows[i]),
                    "runtime": {"prompt_sha256": o["prompt_sha256"], "stop_reason": o["stop_reason"]}}
                   for i, o in zip(indices, observations)]
    pipeline.write_rows(directory / "predictions.jsonl", predictions)
    pipeline.write_json(directory / "receipt.json", {"version": "student_feedback_v1", "complete": True,
                                                     "files": pipeline.inventory(directory)})
    return directory, rows, plan


def test_complete_build_and_reload_use_actual_feedback_decisions(sealed_probe, tmp_path):
    directory, rows, _ = sealed_probe
    output = tmp_path / "feedback"
    report = pipeline.build(directory, output, FeedbackOptions())
    assert report["replay_rows"] == report["preference_pairs"] == 1
    actual_rows, _, replay, pairs, _, _ = pipeline.load_feedback(output)
    assert list(actual_rows) == rows
    assert replay == [0]
    assert pairs[0]["chosen"] == rows[0]["completion"]
    assert pairs[0]["rejected"] == "invalid:0"


def test_fabricated_observation_disagrees_with_raw_prediction(sealed_probe):
    directory, rows, plan = sealed_probe
    observations = list(pipeline.IndexedRows(directory / "observations.jsonl"))
    observations[0]["completion"] = "omission:0"
    pipeline.write_rows(directory / "observations.jsonl", observations)
    with pytest.raises(ValueError, match="binding"):
        pipeline.checked_observations(directory, plan, rows)


def test_feedback_build_does_not_reinterpret_an_old_producer(sealed_probe, tmp_path):
    directory, _, plan = sealed_probe
    plan["producer_sha256"]["pipeline"] = "changed"
    pipeline.write_json(directory / "plan.json", plan)
    (directory / "receipt.json").unlink()
    pipeline.write_json(directory / "receipt.json", {"version": "student_feedback_v1", "complete": True,
                                                     "files": pipeline.inventory(directory)})
    with pytest.raises(ValueError, match="producer"):
        pipeline.build(directory, tmp_path / "feedback", FeedbackOptions())


def test_trial_plan_only_does_not_import_or_load_an_ml_model(sealed_probe, tmp_path, monkeypatch):
    from ir_training.train.student_feedback import train
    directory, _, _ = sealed_probe
    feedback = tmp_path / "feedback"
    pipeline.build(directory, feedback, FeedbackOptions())
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setenv("WORLD_SIZE", "4")
    for arm in ("sft", "replay", "dpo_sft"):
        destination = tmp_path / arm
        plan = train(feedback, destination, arm=arm, max_steps=10, accumulation=2)
        assert plan["status"] == "plan_only"
        assert plan["effective_batch_size"] == 8
        assert plan["schedule"]["total_examples"] == 80
        assert plan["production_promotion_allowed"] is False
        assert not destination.exists()


def test_scorer_change_is_detected_on_feedback_reload(sealed_probe, tmp_path, monkeypatch):
    directory, _, _ = sealed_probe
    output = tmp_path / "feedback"
    pipeline.build(directory, output, FeedbackOptions())
    monkeypatch.setattr(pipeline, "make_scorer", lambda: lambda source, text: {**score(source, text), "metric_fingerprint": "changed"})
    with pytest.raises(ValueError, match="admission"):
        pipeline.load_feedback(output)


def test_stale_checkpoint_blocks_trial_plan(sealed_probe, tmp_path):
    from ir_training.train.student_feedback import train
    directory, _, plan = sealed_probe
    output = tmp_path / "feedback"
    pipeline.build(directory, output, FeedbackOptions())
    (Path(plan["checkpoint"]) / "weights.txt").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        train(output, tmp_path / "trial", arm="sft")


@pytest.mark.parametrize("fails", [False, True])
def test_offline_scope_restores_callers_environment(monkeypatch, fails):
    import os
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    try:
        with pipeline.offline_hf():
            assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"
            if fails:
                raise RuntimeError("fixture")
    except RuntimeError:
        pass
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
