"""Synthetic training-reward adversaries; no held-out/Golden records are read."""
from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
sys.path.insert(0, str(ROOT / "dataset" / "src"))

from ir_training.train.grpo_runtime import audited_reward
from ir_training.train.qat_grpo_reward import make_qat_grpo_reward, map_qat_reward
from pipeline.genui_quality import generation_reward_v5_4, load_v5_4_reward_config


def express(body: str) -> str:
    return f"<a2ui>{body}</a2ui>"


def text(value: str) -> str:
    return express(f"root=Column([a])\na=Text({json.dumps(value)})")


@pytest.fixture(scope="module")
def reward():
    return make_qat_grpo_reward()


def capture(reward, candidates, source, **kwargs):
    evidence = {}
    values = reward(candidates, source, log_extra=lambda key, value: evidence.update({key: value}), **kwargs)
    return values, evidence["qat_reward"]


@pytest.mark.parametrize("candidate", [
    "", "{", '{"root":"text"}',
    'Here is the UI: <a2ui>root=Text("Hello")</a2ui>',
    '```xml\n<a2ui>root=Text("Hello")</a2ui>\n```',
    '<a2ui>root=Text("Hello")</a2ui><a2ui>root=Text("Hello")</a2ui>',
    '<a2ui>root=Unsupported("Hello")</a2ui>',
    '<a2ui>root=Column([missing])</a2ui>',
    '<a2ui>root=Column([a])\na=Column([root])</a2ui>',
    '<a2ui>root=Text("Hello",sourceText="Hello")</a2ui>',
])
def test_strict_raw_express_validity_is_a_hard_gate(reward, candidate):
    values, rows = capture(reward, [candidate], "Hello")
    assert values == [-1.0]
    assert not rows[0]["strict_valid"]
    assert any(reason.startswith("invalid:") for reason in rows[0]["reasons"])


def test_visible_content_and_exact_numbers_drive_real_variation(reward):
    source = "Paris is 21 C. Rome is 25 C. Berlin is 18 C."
    candidates = [
        text(source), text("Paris is 21 C. Rome is 25 C."),
        text("Paris is 21 C."), text("Paris is 99 C."), express("root=Column([])"),
    ]
    values, rows = capture(reward, candidates, source)
    assert values[0] > values[1] > values[2] > values[3] > values[4]
    assert values[0] > 0.9
    assert rows[3]["components"]["critical_fidelity"]["exact_numbers_dates_units_fbeta"] < 1
    assert all(math.isfinite(value) and -1 <= value <= 1 for value in values)


def test_hidden_state_and_unreachable_source_do_not_count_as_visible_fidelity(reward):
    source = "Paris is 21 C."
    hidden = express('$/={sourceText:"Paris is 21 C."}\nroot=Text("Hello")')
    unreachable = express('root=Text("Hello")\nunused=Text("Paris is 21 C.")')
    values, rows = capture(reward, [text(source), hidden, unreachable], source)
    assert values[0] > 0.9
    assert values[1:] == [-1.0, -1.0]
    assert rows[1]["strict_valid"]  # Failure is evidence visibility, not parser rejection.
    assert rows[1]["components"]["source_fidelity"] == 0


def test_repetition_extra_prose_and_wrapper_padding_cannot_improve_reward(reward):
    source = "A quiet garden with green trees."
    baseline = text(source)
    repeated_node = express(
        f'root=Column([a,b])\na=Text({json.dumps(source)})\nb=Text({json.dumps(source)})'
    )
    wrappers = baseline.replace("Column([a])", "Column([b])\nb=Column([a])")
    values, rows = capture(reward, [baseline, repeated_node, text(source + " " + source),
                                  text(source + " An invented red cottage."), wrappers], source)
    assert all(value < values[0] for value in values[1:])
    assert rows[1]["components"]["precision"]["semantic_non_duplication"] < 1


def test_action_reference_mismatch_and_unrequested_action_are_penalized(reward):
    source = "[Book now](https://example.test/book)"
    correct = express('root=Button("Book now",onPress=openUrl("https://example.test/book"))')
    wrong = correct.replace("https://example.test/book", "https://invented.test/book")
    values, rows = capture(reward, [correct, wrong], source)
    assert values[0] > values[1]
    assert rows[0]["components"]["critical_fidelity"]["action_and_source_link_fidelity"] == 1
    assert rows[1]["components"]["precision"]["unsupported_external_addition_precision"] == 0

    source = "Paris is 21 C."
    added = express('root=Column([a,b])\na=Text("Paris is 21 C.")\n'
                    'b=Button("Buy",onPress=openUrl("https://invented.test"))')
    values, rows = capture(reward, [text(source), added], source)
    assert values[0] > values[1] == -1
    assert rows[1]["strict_valid"]
    assert "precision:unsupported_external_addition_precision" in rows[1]["reasons"]


def test_table_value_and_semantic_role_fidelity_are_required(reward):
    source = "| City | Temperature |\n| --- | --- |\n| Paris | 21 C |\n| Rome | 25 C |"
    table = express('root=Table(columns=[{key:"city",label:"City"},'
                    '{key:"temperature",label:"Temperature"}],'
                    'rows=[{city:"Paris",temperature:"21 C"},{city:"Rome",temperature:"25 C"}])')
    values, rows = capture(reward, [table, table.replace("25 C", "99 C"),
                                  text("Paris 21 C Rome 25 C")], source)
    assert values[0] > values[1] > values[2]
    assert rows[1]["components"]["critical_fidelity"]["markdown_table_fidelity"] < 1
    assert "cap:missing_table" in rows[2]["reasons"]


def test_declared_media_reference_fidelity_is_required(reward):
    # Use the repository's Stage 2 media declaration contract.
    source = "Media: Image = https://example.test/garden.png Alt = Garden"
    correct = express('root=Image("https://example.test/garden.png",alt="Garden")')
    values, rows = capture(reward, [correct, correct.replace("garden.png", "invented.png")], source)
    assert values[0] > 0.9
    assert values[1] == -1
    assert rows[0]["components"]["critical_fidelity"]["media_fidelity"] == 1
    assert rows[1]["components"]["critical_fidelity"]["media_fidelity"] == 0


def test_no_novelty_or_reference_completion_bonus_and_group_order_independence(reward):
    source = "Paris is 21 C."
    good, partial = text(source), text("Paris is 22 C.")
    first = reward([good, partial, good], [source] * 3, reference_completion=[partial] * 3)
    second = reward([partial, good], [source] * 2, reference_completion=[good] * 2)
    assert first[0] == first[2] == second[1]
    assert first[1] == second[0]
    assert reward([good], source) == [first[0]]


def test_grouped_sources_keep_evidence_aligned(reward):
    a, b = "Paris is 21 C.", "Rome is 25 C."
    grouped = reward([text(a), text(b), text(a)], [a, b, a])
    assert grouped == [reward([text(a)], a)[0], reward([text(b)], b)[0], reward([text(a)], a)[0]]
    with pytest.raises(ValueError, match="response_text"):
        reward([text(a), text(b)], [a])
    with pytest.raises(ValueError, match="nonempty"):
        reward([text(a)], "")


def test_canonical_caps_preserved_without_flattening_fidelity_variation():
    baseline = generation_reward_v5_4(text("Paris is 21 C."), "Paris is 21 C.")
    capped = replace(baseline, cap_0_1=0.5,
                     active_caps=[{"name": "matching_uncertified", "cap": 0.5}])
    lower_atomics = deepcopy(capped.atomics)
    lower_atomics["fidelity"]["content_unit_fidelity"] = 0.5
    lower = replace(capped, atomics=lower_atomics)
    mapped, partial = map_qat_reward(capped), map_qat_reward(lower)
    assert 0 <= partial.quality < mapped.quality <= 0.5
    assert "cap:matching_uncertified" in mapped.reasons
    assert baseline.cap_0_1 == 1  # Mapping never mutates the canonical result.


def test_corrupt_metric_component_fails_instead_of_producing_nan_reward():
    result = generation_reward_v5_4(text("Hello"), "Hello")
    atomics = deepcopy(result.atomics)
    atomics["fidelity"]["content_unit_fidelity"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        map_qat_reward(replace(result, atomics=atomics))
    with pytest.raises(ValueError, match="Out-of-range"):
        map_qat_reward(replace(result, cap_0_1=2.0))


def test_required_unsupported_reference_penalty_cannot_be_disabled():
    config = replace(load_v5_4_reward_config(), unsupported_external_additions_policy="disabled")
    with pytest.raises(ValueError, match="penalize"):
        make_qat_grpo_reward(config)


def test_shared_runtime_restoration_stop_boundary_and_audit_are_used(reward, tmp_path):
    restored_source = "[Book now](https://example.test/book)"
    candidate = express('root=Button("Book now",onPress=openUrl("[URL_1]"))')
    restored = candidate.replace("[URL_1]", "https://example.test/book")
    wrapped = audited_reward(reward, str(tmp_path), rank=0, audit_limit=2)
    actual = wrapped(completions=[candidate + " ignored serving overshoot"],
                     response_text=[restored_source], source_id=["synthetic-training-row"],
                     url_map=[{"[URL_1]": {"url": "https://example.test/book"}}])
    assert actual == reward([restored], restored_source)
    record = json.loads((tmp_path / "grpo_rewards.rank0.jsonl").read_text(encoding="utf-8"))
    assert record["completion"] == restored
    assert record["raw_completion"].endswith("ignored serving overshoot")
    assert record["breakdown"]["qat_reward"]["strict_valid"]
    assert record["breakdown"]["qat_reward"]["components"]["critical_guard"] == 1
    assert len(record["breakdown"]["qat_reward_policy_sha256"]) == 64


def test_metric_logging_distinguishes_canonical_and_optimized_rewards(reward):
    extras, metrics = {}, {}
    values = reward([text("Paris is 99 C.")], "Paris is 21 C.",
                    log_extra=lambda key, value: extras.update({key: value}),
                    log_metric=lambda key, value: metrics.update({key: value}))
    assert extras["genui_reward"][0] != values[0]
    assert extras["qat_reward"][0]["reward"] == values[0]
    assert metrics["qat_reward/mean"] == values[0]
    assert metrics["qat_reward/sd"] == 0
    assert metrics["qat_reward/strict_valid_fraction"] == 1
