"""DPO-only scheduler compatibility; no CUDA/model/Trainer imports required."""
import math

import pytest
from ir_training.train.qat_dpo import dpo_warmup_arguments


@pytest.mark.parametrize("max_steps,expected", [(100, 3), (200, 6), (99, 3)])
def test_new_trainer_preserves_inherited_ratio(max_steps, expected):
    training = {"warmup_ratio": .03}
    assert dpo_warmup_arguments(training, {"warmup_steps"}, max_steps=max_steps) == {
        "warmup_steps": expected,
    }
    assert training == {"warmup_ratio": .03}


def test_old_trainer_retains_supported_ratio():
    assert dpo_warmup_arguments({"warmup_ratio": .03},
        {"warmup_steps", "warmup_ratio"}, max_steps=100) == {"warmup_ratio": .03}
    assert dpo_warmup_arguments({"warmup_ratio": .03},
        {"warmup_steps", "warmup_ratio"}, max_steps=None) == {"warmup_ratio": .03}


@pytest.mark.parametrize("steps", [0, 3, 6])
@pytest.mark.parametrize("parameters", [{"warmup_steps"}, {"warmup_steps", "warmup_ratio"}])
def test_explicit_steps_take_precedence_even_over_stale_ratio(steps, parameters):
    assert dpo_warmup_arguments({"warmup_steps": steps, "warmup_ratio": .9},
        parameters, max_steps=100) == {"warmup_steps": steps}


@pytest.mark.parametrize("parameters", [{"warmup_steps"}, {"warmup_steps", "warmup_ratio"}])
def test_two_update_probe_keeps_production_warmup_horizon(parameters):
    arguments = dpo_warmup_arguments({"warmup_ratio": .03}, parameters,
        max_steps=200, runtime_preflight_only=True)
    # train_qat_dpo changes max_steps to 2 only after resolving these arguments.
    arguments.update(max_steps=2)
    assert arguments == {"warmup_steps": 6, "max_steps": 2}


@pytest.mark.parametrize("ratio", [True, -0.1, 1, math.inf, math.nan, "0.03", None])
@pytest.mark.parametrize("parameters", [{"warmup_steps"}, {"warmup_steps", "warmup_ratio"}])
def test_invalid_ratio_fails_on_both_apis(ratio, parameters):
    with pytest.raises(ValueError, match="warmup_ratio"):
        dpo_warmup_arguments({"warmup_ratio": ratio}, parameters, max_steps=100)


def test_default_ratio_is_not_silently_dropped():
    assert dpo_warmup_arguments({}, {"warmup_steps"}, max_steps=100) == {"warmup_steps": 3}


def test_zero_ratio_needs_no_horizon():
    assert dpo_warmup_arguments({"warmup_ratio": 0}, {"warmup_steps"}, max_steps=None) == {
        "warmup_steps": 0,
    }


@pytest.mark.parametrize("max_steps", [None, 0, -1, True])
def test_unknown_production_horizon_fails_instead_of_guessing(max_steps):
    with pytest.raises(ValueError, match="production max_steps"):
        dpo_warmup_arguments({"warmup_ratio": .03}, {"warmup_steps"}, max_steps=max_steps)


def test_unsupported_warmup_fails_instead_of_silently_dropping():
    with pytest.raises(ValueError, match="cannot apply"):
        dpo_warmup_arguments({"warmup_ratio": .03}, set(), max_steps=100)
    with pytest.raises(ValueError, match="cannot apply"):
        dpo_warmup_arguments({"warmup_steps": 3}, {"warmup_ratio"}, max_steps=100)


def test_rounded_ratio_cannot_consume_entire_production_horizon():
    with pytest.raises(ValueError, match="after warmup"):
        dpo_warmup_arguments({"warmup_ratio": .999}, {"warmup_steps"}, max_steps=20)
