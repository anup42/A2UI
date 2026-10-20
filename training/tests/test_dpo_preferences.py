"""Offline, train-only GRPO preference mining; no model or GPU required."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
sys.path.insert(0, str(ROOT / "dataset" / "src"))

from ir_training.train import dpo_preferences
from ir_training.train.dpo_preferences import build_preferences

GOOD = '<a2ui>root=Text("Hello world.")</a2ui>'
BAD = '<a2ui>root=Text("Hello.")</a2ui>'
LIMITS = {"max_prompt_length": 1000, "max_completion_length": 2000, "max_sequence_length": 3000}


class CharacterTokenizer:
    eos_token_id = 1

    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return {"input_ids": [ord(char) + 10 for char in text]}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source(**updates):
    return {"source_id": "s1", "response_text": "Hello world.", "completion": GOOD,
            "prompt": "<bos>User: Hello world.\nAssistant:\n", "url_map": {},
            "response_text_sha256": digest("Hello world."),
            "source_context_sha256": "a" * 64, **updates}


def audit(raw_completion, sample=None, **updates):
    sample = source() if sample is None else sample
    return {"step": 10, "mode": "train", "rank": 0, "local_index": 0,
            "source_id": sample["source_id"], "prompt_sha256": digest(sample["prompt"]),
            "raw_completion": raw_completion, "termination_reason": "closing_sentinel",
            "reward": -99, "breakdown": {}, **updates}


def audit_file(tmp_path, rows, name="grpo_periodic_rewards.rank0.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
    return path


def build(tmp_path, records=None, **kwargs):
    path = audit_file(tmp_path, records if records is not None else [audit(GOOD), audit(BAD)])
    settings = {"limits": LIMITS, "min_pairs": 1, **kwargs}
    return build_preferences([path], [source()], CharacterTokenizer(), tmp_path / "out", **settings)


def preferences(tmp_path, directory="out"):
    return [json.loads(line) for line in (tmp_path / directory / "preferences.jsonl").read_text(
        encoding="utf-8").splitlines()]


def test_rescores_raw_completions_and_ignores_old_reward_and_restored_completion(tmp_path):
    records = [audit(GOOD, reward=-1, completion=BAD), audit(BAD, reward=1, completion=GOOD)]
    report = build(tmp_path, records)
    pairs = preferences(tmp_path)
    assert report["pairs"] == 1
    assert len(pairs) == 1 and pairs[0]["chosen"] == GOOD and pairs[0]["rejected"] == BAD
    assert pairs[0]["source_id"] == "s1"
    assert pairs[0]["prompt_sha256"] == digest(source()["prompt"])
    assert pairs[0]["response_text_sha256"] == digest(source()["response_text"])
    for name in ("preference_build_report.json", "preference_admission.json", "preference_provenance.jsonl"):
        assert (tmp_path / "out" / name).is_file()


def test_one_pair_per_source_uses_highest_scoring_negative_and_deduplicates(tmp_path):
    records = [audit(GOOD), audit(BAD), audit("not valid Express"),
               audit(GOOD, step=20), audit(BAD, step=30)]
    build(tmp_path, records)
    pairs = preferences(tmp_path)
    assert len(pairs) == 1 and pairs[0]["rejected"] == BAD


def test_selection_is_independent_of_audit_order(tmp_path):
    records = [audit(BAD), audit(GOOD), audit("not valid Express"), audit(GOOD)]
    first = audit_file(tmp_path, records)
    second = audit_file(tmp_path, [{**record, "rank": 1} for record in reversed(records)],
                        "grpo_periodic_rewards.rank1.jsonl")
    for directory, files in (("a", [first, second]), ("b", [second, first])):
        build_preferences(files, [source()], CharacterTokenizer(), tmp_path / directory,
                          limits=LIMITS, min_pairs=1)
    assert preferences(tmp_path, "a") == preferences(tmp_path, "b")


@pytest.mark.parametrize("reason", ["eos_token", "closing_sentinel"])
def test_verified_termination_reasons_are_supported(tmp_path, reason):
    build(tmp_path, [audit(GOOD, termination_reason=reason), audit(BAD, termination_reason=reason)])
    assert preferences(tmp_path)[0]["chosen"] == GOOD


@pytest.mark.parametrize("reason", ["max_new_tokens", "unknown"])
def test_unterminated_candidates_are_not_supervised(tmp_path, reason):
    with pytest.raises(ValueError):
        build(tmp_path, [audit(GOOD, termination_reason=reason), audit(BAD)])
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


@pytest.mark.parametrize("reason", [None, "length", "arbitrary"])
def test_unrecognized_termination_schema_fails(tmp_path, reason):
    with pytest.raises(ValueError):
        build(tmp_path, [audit(GOOD, termination_reason=reason), audit(BAD)])
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


def test_eval_rows_are_never_candidates_even_with_a_matching_train_identity(tmp_path):
    with pytest.raises(ValueError):
        build(tmp_path, [audit(GOOD, mode="eval"), audit(BAD)])
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


def test_known_eval_rows_are_skipped_without_scoring_or_binding_to_train(tmp_path):
    build(tmp_path, [audit(GOOD), audit(BAD),
                     audit(GOOD, mode="eval", source_id="golden35", prompt_sha256="b" * 64)])
    assert len(preferences(tmp_path)) == 1


@pytest.mark.parametrize("change", [
    {"mode": None}, {"mode": "test"}, {"source_id": "unknown"},
    {"source_id": ""}, {"prompt_sha256": "0" * 64}, {"prompt_sha256": None},
])
def test_missing_train_membership_or_prompt_binding_fails_closed(tmp_path, change):
    with pytest.raises(ValueError):
        build(tmp_path, [audit(GOOD, **change), audit(BAD)])
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


@pytest.mark.parametrize("heldout", [
    {"source_id": "s1"}, {"response_text": "Hello world."},
    {"prompt": "<bos>User: Hello world.\nAssistant:\n"},
])
def test_train_overlap_with_reserved_or_validation_sources_fails(tmp_path, heldout):
    with pytest.raises(ValueError, match="held-out overlap"):
        build(tmp_path, forbidden_rows=[heldout])
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


def test_better_but_unfaithful_candidate_cannot_be_chosen(tmp_path):
    # BAD outranks unparsable text but still loses source information.
    with pytest.raises(ValueError):
        build(tmp_path, [audit(BAD), audit("not valid Express")])
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


def test_duplicate_only_and_semantic_ties_do_not_create_preferences(tmp_path):
    equivalent = GOOD.replace("<a2ui>", "<a2ui>\n", 1)
    with pytest.raises(ValueError):
        build(tmp_path, [audit(GOOD), audit(GOOD), audit(equivalent)])
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


def test_token_identical_alternatives_do_not_create_preferences(tmp_path):
    class CollapsingTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            return super().__call__(text.replace(BAD, GOOD), **kwargs)

    path = audit_file(tmp_path, [audit(GOOD), audit(BAD)])
    with pytest.raises(ValueError):
        build_preferences([path], [source()], CollapsingTokenizer(), tmp_path / "out",
                          limits=LIMITS, min_pairs=1)
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


def test_changed_audit_cannot_overwrite_the_initial_evidence_hash(tmp_path, monkeypatch):
    path = audit_file(tmp_path, [audit(GOOD), audit(BAD)]).resolve()
    original = dpo_preferences.file_sha256
    initial_digest = original(path)
    calls = 0

    def changed_after_verification(candidate):
        nonlocal calls
        if Path(candidate).resolve() == path:
            calls += 1
            if calls > 1:
                return "b" * 64
        return original(candidate)

    monkeypatch.setattr(dpo_preferences, "file_sha256", changed_after_verification)
    with pytest.raises(ValueError, match="changed"):
        build_preferences([path], [source()], CharacterTokenizer(), tmp_path / "out",
                          limits=LIMITS, min_pairs=1,
                          evidence={"files": {str(path): initial_digest}})
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


def test_audit_rank_must_match_filename(tmp_path):
    with pytest.raises(ValueError, match="rank"):
        build(tmp_path, [audit(GOOD, rank=1), audit(BAD)])
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


@pytest.mark.parametrize("limit", list(LIMITS))
def test_token_budgets_are_enforced_without_truncating_candidates(tmp_path, limit):
    with pytest.raises(ValueError):
        build(tmp_path, limits={**LIMITS, limit: 1})
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


def test_positive_reward_gap_is_enforced(tmp_path):
    with pytest.raises(ValueError):
        build(tmp_path, min_reward_gap=1.9)
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


def test_default_gap_retains_faithful_list_to_card_extraction_preferences(tmp_path):
    text = ("- Hotel Aurora: Goa; Price: INR 5000; Rating: 4.5\n"
            "- Hotel Cedar: Pune; Price: INR 4000; Rating: 4.2")
    card_target = ('<a2ui>root=Column([c0,c1])\n'
                   'c0=Card([c0f0,c0f1,c0f2],title="Hotel Aurora")\n'
                   'c0f0=Text("Goa")\nc0f1=Text("Price: INR 5000")\nc0f2=Text("Rating: 4.5")\n'
                   'c1=Card([c1f0,c1f1,c1f2],title="Hotel Cedar")\n'
                   'c1f0=Text("Pune")\nc1f1=Text("Price: INR 4000")\nc1f2=Text("Rating: 4.2")</a2ui>')
    list_target = "<a2ui>root=List(items=" + json.dumps([line[2:] for line in text.splitlines()]) + ")</a2ui>"
    sample = source(response_text=text, response_text_sha256=digest(text),
                    prompt=text + "\nAssistant:\n", completion=card_target)
    path = audit_file(tmp_path, [audit(list_target, sample), audit(card_target, sample)])
    build_preferences([path], [sample], CharacterTokenizer(), tmp_path / "out",
                      limits=LIMITS, min_pairs=1)
    assert preferences(tmp_path)[0]["chosen"] == card_target
    assert preferences(tmp_path)[0]["rejected"] == list_target
    evidence = json.loads((tmp_path / "out" / "preference_provenance.jsonl").read_text(encoding="utf-8"))
    assert 0.01 <= evidence["reward_gap"] < 0.05
    assert not evidence["chosen"]["reasons"] and not evidence["rejected"]["reasons"]


def test_minimum_pairs_is_not_silently_reduced(tmp_path):
    with pytest.raises(ValueError):
        build(tmp_path, min_pairs=2)
    assert not (tmp_path / "out" / "preferences.jsonl").exists()


@pytest.mark.parametrize("option,value", [
    ("min_pairs", 0), ("min_pairs", True), ("min_pairs", 1.5),
    ("min_reward_gap", 0), ("min_reward_gap", float("nan")), ("min_reward_gap", True),
])
def test_invalid_selection_settings_fail(tmp_path, option, value):
    with pytest.raises(ValueError):
        build(tmp_path, **{option: value})


def test_existing_output_is_not_overwritten(tmp_path):
    (tmp_path / "out").mkdir()
    sentinel = tmp_path / "out" / "preferences.jsonl"
    sentinel.write_text("keep me", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        build(tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("contents", [
    "{broken\n", "[]\n", '{"mode":"train","mode":"eval"}\n',
])
def test_malformed_or_duplicate_key_audit_is_not_silently_dropped(tmp_path, contents):
    path = tmp_path / "grpo_periodic_rewards.rank0.jsonl"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        build_preferences([path], [source()], CharacterTokenizer(), tmp_path / "out",
                          limits=LIMITS, min_pairs=1)
    assert not (tmp_path / "out" / "preferences.jsonl").exists()
