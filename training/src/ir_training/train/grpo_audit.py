"""Small, lazy-import GRPO correctness guards shared by both training runners.

The trainer extension is scoped to the source-reviewed TRL 0.29.1 runtime.
It observes already-gathered rewards; it never changes rewards, advantages,
loss normalization, or the number/order of distributed collectives.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from numbers import Real
from typing import Any, Sequence


def validate_reward_values(values: Any, count: int) -> None:
    """Do not let TRL broadcast a short vector or turn a broken scorer into NaNs."""
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError("GRPO reward must return exactly one value per completion")
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"GRPO reward[{index}] must be a finite real number, got {value!r}")


def group_reward_statistics(rewards: Any, weights: Any, generations: int) -> dict[str, float]:
    """Measure *within-prompt* variance on TRL's globally gathered reward rows.

    TRL 0.29.1's frac_reward_zero_std uses the batch standard deviation when
    scale_rewards='batch'. Different constant-reward groups therefore look
    diverse even though every advantage is zero. Keep that normalization, but
    compute health evidence from the contiguous groups used by the objective.
    """
    import torch

    if type(generations) is not int or generations < 1:
        raise ValueError("GRPO generation group size must be a positive integer")
    if rewards.ndim != 2 or not rewards.shape[0] or rewards.shape[0] % generations:
        raise ValueError("Gathered GRPO rewards must contain complete generation groups")
    weights = torch.as_tensor(weights, dtype=rewards.dtype, device=rewards.device)
    if weights.ndim != 1 or weights.numel() != rewards.shape[1]:
        raise ValueError("GRPO reward weights do not match reward columns")
    if not bool(torch.isfinite(rewards).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("Gathered GRPO rewards and weights must be finite")
    # Match TRL's sum_then_normalize reduction and floating-point precision.
    weighted = (rewards * weights.unsqueeze(0)).sum(dim=1)
    if not bool(torch.isfinite(weighted).all()):
        raise ValueError("Weighted GRPO rewards must be finite")
    groups = weighted.reshape(-1, generations)
    std = groups.std(dim=1) if generations > 1 else torch.zeros_like(groups[:, 0])
    if not bool(torch.isfinite(std).all()):
        raise ValueError("GRPO group standard deviations must be finite")
    return {
        "frac_reward_zero_std": torch.isclose(std, torch.zeros_like(std)).float().mean().item(),
        "group_reward_std": std.mean().item(),
    }


def build_audited_grpo_trainer(base_trainer: Any) -> Any:
    """Correct group-health telemetry after TRL has applied its own objective."""
    class AuditedGRPOTrainer(base_trainer):
        def _calculate_rewards(self, *args: Any, **kwargs: Any) -> Any:
            rewards = super()._calculate_rewards(*args, **kwargs)
            if self.multi_objective_aggregation != "sum_then_normalize":
                raise RuntimeError("Reviewed GRPO health supports sum_then_normalize only")
            mode = "train" if self.model.training else "eval"
            generations = self.num_generations if mode == "train" else self.num_generations_eval
            self._a2ui_group_statistics = group_reward_statistics(rewards, self.reward_weights, generations)
            return rewards

        def _generate_and_score_completions(self, *args: Any, **kwargs: Any) -> Any:
            mode = "train" if self.model.training else "eval"
            self._a2ui_group_statistics = None
            result = super()._generate_and_score_completions(*args, **kwargs)
            statistics = self._a2ui_group_statistics
            series = self._metrics[mode].get("frac_reward_zero_std")
            if statistics is None or not series:
                raise RuntimeError("TRL did not produce the expected GRPO group-health evidence")
            # Keep the upstream statistic for diagnosis, not for the health gate.
            self._metrics[mode]["trl/frac_reward_zero_std"].append(series[-1])
            series[-1] = statistics["frac_reward_zero_std"]
            self._metrics[mode]["grpo/group_reward_std"].append(statistics["group_reward_std"])
            return result

    return AuditedGRPOTrainer


def rollout_generation_config(trainer: Any, eos_token_ids: Sequence[int]) -> Any:
    """Build a complete policy config, never inheriting checkpoint sampling rules.

    Fresh, on-policy rollouts have no correction for filtering, forced tokens,
    or repetition penalties. Allow only the controls used by these runners.
    The tokenizer/model still supplies its native BOS/PAD/EOS identities.
    """
    from transformers import GenerationConfig

    supplied = dict(trainer.generation_kwargs)
    allowed = {
        "max_new_tokens", "do_sample", "pad_token_id", "bos_token_id", "eos_token_id",
        "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "cache_implementation",
    }
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError("Unreviewed GRPO generation controls: " + ", ".join(unknown))
    for key, expected in (("do_sample", True), ("top_p", 1.0), ("top_k", 0), ("repetition_penalty", 1.0)):
        if supplied.get(key, expected) != expected:
            raise ValueError(f"On-policy GRPO requires {key}={expected}")
    if supplied.get("min_p") not in (None, 0, 0.0):
        raise ValueError("On-policy GRPO requires min_p=None or 0")
    temperature = supplied.get("temperature", getattr(trainer, "temperature", 1.0))
    if isinstance(temperature, bool) or not isinstance(temperature, Real) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("GRPO rollout temperature must be finite and positive")
    if temperature != getattr(trainer, "temperature", temperature):
        raise ValueError("GRPO sampling temperature differs from likelihood temperature")
    tokenizer = trainer.processing_class
    return GenerationConfig(
        max_new_tokens=trainer.max_completion_length,
        do_sample=True, temperature=float(temperature), top_p=1.0, top_k=0, min_p=None,
        repetition_penalty=1.0, num_beams=1, num_return_sequences=1,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        pad_token_id=tokenizer.pad_token_id, eos_token_id=list(eos_token_ids),
        stop_strings=None, use_cache=True,
        cache_implementation=supplied.get("cache_implementation"),
        return_dict_in_generate=False,
    )


@contextmanager
def isolated_generation_config(model: Any, generation_config: Any):
    """Prevent HF 4/5 from filling unset fields from the saved checkpoint.

    Transformers 5.3 fills None-valued fields even for an explicitly supplied
    GenerationConfig. PEFT may delegate generate to its base model, so isolate
    both owners. Restore original objects, including on generation failure.
    No model weights, token IDs, or persistent inference defaults are changed.
    """
    targets = [model]
    if callable(getattr(model, "get_base_model", None)):
        base = model.get_base_model()
        if base is not model:
            targets.append(base)
    missing = object()
    originals = []
    try:
        for target in targets:
            previous = getattr(target, "generation_config", missing)
            # PEFT can expose this attribute through __getattr__. Do not leave
            # a shadowing attribute behind after restoring the base config.
            local = "generation_config" in getattr(target, "__dict__", {}) or any(
                "generation_config" in cls.__dict__ for cls in type(target).__mro__
            )
            originals.append((target, previous, local))
            target.generation_config = generation_config
        yield
    finally:
        for target, previous, local in reversed(originals):
            if previous is missing or not local:
                delattr(target, "generation_config")
            else:
                target.generation_config = previous
