"""Offline admission and exact-token supervision; no model or GPU required."""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
sys.path.insert(0, str(ROOT / "dataset" / "src"))

from ir_training.train.qat_posttrain_data import (
    REHEARSAL_TOKEN_CACHE_SIZE,
    _digest,
    build_rehearsal_bank,
    prepare_dpo_pairs,
    tokenize_supervised_target,
    verify_admission_manifest,
)

GOOD = '<a2ui>root=Text("Hello world.")</a2ui>'
BAD = '<a2ui>root=Text("Hello.")</a2ui>'
LIMITS = {"max_prompt_length": 1000, "max_completion_length": 2000, "max_sequence_length": 3000}


class CharacterTokenizer:
    eos_token_id = 1

    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return {"input_ids": [ord(char) + 10 for char in text]}


def row(**updates):
    source = "Hello world."
    return {"source_id": "s1", "response_text": source, "completion": GOOD,
            "prompt": "<bos>User: Hello world.\nAssistant:\n", "url_map": {},
            "response_text_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source_context_sha256": "a" * 64, **updates}


def pair_file(tmp_path, rows=None):
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in
                              (rows if rows is not None else [{"source_id": "s1", "chosen": GOOD, "rejected": BAD}])),
                    encoding="utf-8")
    return path


def prepare(path, **kwargs):
    return prepare_dpo_pairs(path, [row()], CharacterTokenizer(), **LIMITS, **kwargs)


def test_rehearsal_preserves_prompt_masks_and_exact_eos():
    bank, manifest = build_rehearsal_bank([row()], CharacterTokenizer(), **LIMITS)
    tokens = bank[row()["prompt"]]
    n = len(tokens["prompt_input_ids"])
    assert tokens["labels"][:n] == [-100] * n
    assert tokens["labels"][n:] == tokens["input_ids"][n:]
    assert tokens["input_ids"][-1] == 1
    assert tokens["completion_mask"] == [0] * n + [1] * (len(GOOD) + 1)
    assert manifest["rows"] == 1
    assert manifest["admitted_rows_sha256"]
    assert manifest == build_rehearsal_bank([row()], CharacterTokenizer(), **LIMITS)[1]


def test_rehearsal_bank_is_lazy_bounded_and_stream_hash_matches_exact_records():
    rows = [row(source_id=f"s{i}", prompt=f"User {i}: Hello world.\nAssistant:\n")
            for i in range(REHEARSAL_TOKEN_CACHE_SIZE + 3)]
    bank, manifest = build_rehearsal_bank(rows, CharacterTokenizer(), **LIMITS)
    assert not bank._cache
    assert all("input_ids" not in entry and "labels" not in entry for entry in bank._entries)
    assert bank.prompt_to_index[rows[-1]["prompt"]] == len(rows) - 1
    assert manifest["admitted_rows_sha256"] == _digest(list(bank.values()))
    assert len(bank._cache) == REHEARSAL_TOKEN_CACHE_SIZE
    assert 0 not in bank._cache
    assert bank.tokenized(0) == bank[rows[0]["prompt"]]
    assert len(bank._cache) == REHEARSAL_TOKEN_CACHE_SIZE
    assert 0 in bank._cache
    with pytest.raises(IndexError):
        bank.tokenized(-1)


def test_lazy_tokenization_refuses_drift_from_full_admission():
    class MutableTokenizer(CharacterTokenizer):
        offset = 0

        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            return {"input_ids": [value + self.offset for value in result["input_ids"]]}

    tokenizer = MutableTokenizer()
    bank, _ = build_rehearsal_bank([row()], tokenizer, **LIMITS)
    tokenizer.offset = 1
    with pytest.raises(ValueError, match="tokenization changed"):
        bank.tokenized(0)


def test_lazy_bank_still_rejects_unfaithful_late_rows_at_admission():
    with pytest.raises(ValueError, match="s2.*fidelity"):
        build_rehearsal_bank([row(), row(source_id="s2", prompt="other prompt", completion=BAD)],
                             CharacterTokenizer(), **LIMITS)


def test_dpo_positive_fidelity_gap_and_immutable_pair_hash(tmp_path):
    path = pair_file(tmp_path)
    pairs, manifest = prepare(path)
    pair = pairs[0]
    assert pair["reward_gap"] > 0
    assert pair["chosen_input_ids"][-1] == pair["rejected_input_ids"][-1] == 1
    assert pair["chosen_completion_mask"][-1] == 1
    assert pair["chosen_labels"][:len(pair["prompt_input_ids"])] == [-100] * len(pair["prompt_input_ids"])
    assert manifest["pairs_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["unique_train_sources"] == 1
    assert manifest["metric_fingerprint"]
    assert manifest == prepare(path)[1]


def test_invalid_rejected_is_allowed_but_never_supervised(tmp_path):
    pairs, _ = prepare(pair_file(tmp_path, [{"source_id": "s1", "chosen": GOOD, "rejected": "not valid Express"}]))
    assert pairs[0]["rejected_reward"] == -1


@pytest.mark.parametrize("target", [BAD, "not valid Express", GOOD + " explanation", GOOD.replace("world", "Mars")])
def test_rehearsal_fails_instead_of_dropping_bad_supervision(target):
    with pytest.raises(ValueError, match="s1"):
        build_rehearsal_bank([row(completion=target)], CharacterTokenizer(), **LIMITS)


@pytest.mark.parametrize("field", ["source_id", "response_text", "prompt"])
def test_train_heldout_overlap_rejected(field):
    with pytest.raises(ValueError, match="held-out overlap"):
        build_rehearsal_bank([row()], CharacterTokenizer(), forbidden_rows=[{field: row()[field]}], **LIMITS)


@pytest.mark.parametrize("change", [{"source_id": "golden35"}, {"chosen": BAD}, {"chosen": GOOD, "rejected": GOOD},
                                    {"response_text_sha256": "b" * 64}, {"source_context_sha256": "b" * 64},
                                    {"prompt_sha256": "b" * 64}, {"prompt": "override"}, {"chosen": ""}])
def test_bad_dpo_pairs_fail_closed(tmp_path, change):
    pair = {"source_id": "s1", "chosen": GOOD, "rejected": BAD, **change}
    with pytest.raises(ValueError):
        prepare(pair_file(tmp_path, [pair]))


def test_matching_optional_source_bindings_accepted(tmp_path):
    pair = {"source_id": "s1", "chosen": GOOD, "rejected": BAD,
            "response_text_sha256": row()["response_text_sha256"],
            "source_context_sha256": row()["source_context_sha256"],
            "prompt_sha256": hashlib.sha256(row()["prompt"].encode()).hexdigest()}
    assert len(prepare(pair_file(tmp_path, [pair]))[0]) == 1


@pytest.mark.parametrize("variant", ["duplicate", "reverse", "id_collision"])
def test_duplicate_and_reversed_pairs_fail(tmp_path, variant):
    pair = {"source_id": "s1", "chosen": GOOD, "rejected": BAD, "pair_id": "p1"}
    second = dict(pair)
    if variant == "reverse":
        second.update(chosen=BAD, rejected=GOOD)
    if variant == "id_collision":
        second["rejected"] = "not valid Express"
    with pytest.raises(ValueError, match="duplicate|reversed"):
        prepare(pair_file(tmp_path, [pair, second]))


def test_tied_semantic_equivalents_not_preferences(tmp_path):
    # Different formatting, identical visible meaning: not a preference.
    pair = {"source_id": "s1", "chosen": GOOD, "rejected": GOOD.replace("<a2ui>", "<a2ui>\n", 1)}
    with pytest.raises(ValueError, match="positive preference gap"):
        prepare(pair_file(tmp_path, [pair]))


@pytest.mark.parametrize("content", ["", "{broken", "[]", "null", '{"source_id":"a","source_id":"b"}'])
def test_malformed_or_empty_input_never_silently_dropped(tmp_path, content):
    path = tmp_path / "pairs.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        prepare(path)


@pytest.mark.parametrize("field", ["max_prompt_length", "max_completion_length", "max_sequence_length"])
def test_overbudget_supervision_is_rejected_not_truncated(field):
    limits = {**LIMITS, field: 1}
    with pytest.raises(ValueError, match="budget.*no truncation"):
        tokenize_supervised_target(CharacterTokenizer(), row()["prompt"], GOOD, **limits)


def test_prompt_completion_boundary_merge_fails():
    class MergingTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text.endswith(GOOD):
                result["input_ids"][len(row()["prompt"]) - 1] += 1
            return result
    with pytest.raises(ValueError, match="boundary"):
        tokenize_supervised_target(MergingTokenizer(), row()["prompt"], GOOD, **LIMITS)


def test_missing_and_embedded_eos_fail():
    tokenizer = CharacterTokenizer()
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="EOS token id"):
        tokenize_supervised_target(tokenizer, "prompt", GOOD, **LIMITS)
    tokenizer.eos_token_id = ord("H") + 10
    with pytest.raises(ValueError, match="embedded EOS"):
        tokenize_supervised_target(tokenizer, "prompt", GOOD, **LIMITS)


@pytest.mark.parametrize("rows", [[row(), row()], [row(response_text_sha256="b" * 64)], []])
def test_ambiguous_or_tampered_train_identity_fails(rows):
    with pytest.raises(ValueError):
        build_rehearsal_bank(rows, CharacterTokenizer(), **LIMITS)


def action_row():
    source = "- Hotel Aurora: Goa; Price: INR 5000; Rating: 4.5; [Book now](https://example.test/details)"
    target = ('<a2ui>root=Card([city,price,rating,book],title="Hotel Aurora")\n'
              'city=Text("Goa")\nprice=Text("Price: INR 5000")\nrating=Text("Rating: 4.5")\n'
              'book=Button("Book now",onPress=openUrl("[ACTION_URL_1]"))</a2ui>')
    return row(response_text=source, completion=target, response_text_sha256=None,
               prompt=source.replace("https://example.test/details", "[ACTION_URL_1]") + "\nAssistant:\n",
               url_map={"[ACTION_URL_1]": {"url": "https://example.test/details", "role": "action"}})


def test_existing_url_bindings_validate_without_rewriting_targets():
    sample = action_row()
    bank, _ = build_rehearsal_bank([sample], CharacterTokenizer(), **LIMITS)
    ids = bank[sample["prompt"]]["input_ids"]
    assert "[ACTION_URL_1]" in "".join(chr(value - 10) for value in ids[:-1])


@pytest.mark.parametrize("escaped", [False, True])
def test_masked_source_does_not_supervise_hidden_literal_urls(escaped):
    sample = action_row()
    url = "https://example.test/details"
    if escaped:
        url = url.replace("/", "\\/")
    sample["completion"] = sample["completion"].replace("[ACTION_URL_1]", url)
    with pytest.raises(ValueError, match="literal reference is hidden"):
        build_rehearsal_bank([sample], CharacterTokenizer(), **LIMITS)


def test_literal_reference_is_allowed_when_model_can_see_it():
    sample = action_row()
    sample["completion"] = sample["completion"].replace("[ACTION_URL_1]", "https://example.test/details")
    sample["prompt"] = sample["response_text"] + "\nAssistant:\n"
    assert len(build_rehearsal_bank([sample], CharacterTokenizer(), **LIMITS)[0]) == 1


def test_supervised_placeholder_must_be_prompt_visible():
    sample = action_row()
    sample["prompt"] = sample["response_text"] + "\nAssistant:\n"
    with pytest.raises(ValueError, match="bound and prompt-visible"):
        build_rehearsal_bank([sample], CharacterTokenizer(), **LIMITS)


def test_rejected_bad_reference_is_still_an_admissible_negative(tmp_path):
    sample = action_row()
    pair = {"source_id": sample["source_id"], "chosen": sample["completion"],
            "rejected": sample["completion"].replace("[ACTION_URL_1]", "https://evil.test/wrong")}
    pairs, _ = prepare_dpo_pairs(pair_file(tmp_path, [pair]), [sample], CharacterTokenizer(), **LIMITS)
    assert len(pairs) == 1 and pairs[0]["reward_gap"] > 0


@pytest.fixture
def rehearsal_manifest():
    return build_rehearsal_bank([row()], CharacterTokenizer(), **LIMITS)[1]


def test_current_admission_manifest_verifies_without_model_dependencies(rehearsal_manifest):
    assert verify_admission_manifest(rehearsal_manifest, kind="grpo_sft_rehearsal",
                                     expected_limits=LIMITS) == rehearsal_manifest


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("admission_policy", "relaxed"), ("kind", "qat_dpo"),
    ("rows", 0), ("rows", True), ("rows", 1.5), ("admitted_rows_sha256", "bad"),
    ("train_split_binding_sha256", "b" * 63), ("admission_implementation_sha256", "b" * 64),
    ("reward_mapping_sha256", "b" * 64), ("metric_fingerprint", "b" * 64),
    ("reward_pipeline_fingerprint", "b" * 64), ("token_limits", {}),
    ("token_limits", {**LIMITS, "max_prompt_length": False}),
])
def test_manifest_rejects_stale_malformed_or_relabeled_evidence(rehearsal_manifest, field, value):
    with pytest.raises(ValueError):
        verify_admission_manifest({**rehearsal_manifest, field: value}, kind="grpo_sft_rehearsal")


def test_manifest_checks_configured_limits_and_actual_reward_config(rehearsal_manifest):
    from pipeline.genui_quality.config_v5_4 import load_v5_4_reward_config
    with pytest.raises(ValueError, match="token limits"):
        verify_admission_manifest(rehearsal_manifest, kind="grpo_sft_rehearsal",
                                  expected_limits={**LIMITS, "max_sequence_length": 1})
    changed = replace(load_v5_4_reward_config(), accessibility_penalty_alpha=0.02)
    with pytest.raises(ValueError, match="scorer fingerprints"):
        verify_admission_manifest(rehearsal_manifest, kind="grpo_sft_rehearsal", reward_config=changed)


def test_dpo_manifest_checks_pairs_and_admission_gap(tmp_path):
    _, manifest = prepare(pair_file(tmp_path))
    assert verify_admission_manifest(manifest, kind="qat_dpo", expected_limits=LIMITS) == manifest
    for change in ({"pairs_sha256": "bad"}, {"unique_train_sources": 2}, {"unique_train_sources": False},
                   {"min_reward_gap": 0}, {"min_reward_gap": float("nan")}, {"min_reward_gap": True}):
        with pytest.raises(ValueError):
            verify_admission_manifest({**manifest, **change}, kind="qat_dpo")


def test_missing_manifest_or_unknown_method_fails():
    with pytest.raises(ValueError, match="lacks an admission"):
        verify_admission_manifest(None, kind="qat_dpo")
    with pytest.raises(ValueError, match="Unsupported"):
        verify_admission_manifest({}, kind="plain_sft")
