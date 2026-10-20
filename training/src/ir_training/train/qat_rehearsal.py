"""Completion-only supervised rehearsal, opt-in on the reviewed GRPO trainer.

One trusted target is paired with each sampled completion. Repetition within a
GRPO group therefore keeps the per-prompt relative weight unchanged. Indices are
carried in TRL's shuffled/sliced tensors, never guessed from a local step counter.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any


def attach_rehearsal_indices(result: dict, inputs: list[dict], bank: Mapping[str, dict]) -> dict:
    import torch

    prompt_ids, prompt_mask = result["prompt_ids"], result["prompt_mask"]
    if len(inputs) != len(prompt_ids):
        raise ValueError("Rehearsal/GRPO generation row alignment failed")
    indices_by_prompt = getattr(bank, "prompt_to_index", None)
    if indices_by_prompt is None:
        indices_by_prompt = {prompt: i for i, prompt in enumerate(bank)}
    indices = []
    for row, ids, mask in zip(inputs, prompt_ids, prompt_mask, strict=True):
        prompt = row["prompt"]
        if prompt not in bank:
            raise ValueError("Rehearsal requires an admitted training source; evaluation rows are forbidden")
        if ids[mask.bool()].tolist() != bank[prompt]["prompt_input_ids"]:
            raise ValueError("Rehearsal target prompt differs from sampled policy prompt")
        indices.append(indices_by_prompt[prompt])
    result["a2ui_rehearsal_indices"] = torch.tensor(indices, device=prompt_ids.device, dtype=torch.long)
    return result


def rehearsal_loss(model: Any, inputs: dict, bank: Mapping[str, dict], *, device: Any) -> Any:
    """Mean per-example completion CE, no temperature, no prompt/pad supervision."""
    import torch

    from ir_training.train.qat_dpo import completion_log_probability

    indices = inputs.get("a2ui_rehearsal_indices")
    if indices is None or indices.ndim != 1 or not indices.numel():
        raise ValueError("Missing aligned rehearsal target indices")
    # Production uses an admitted, bounded-cache token bank. Do not materialize
    # all tokenized targets again on every microbatch (large datasets are common).
    tokenized = getattr(bank, "tokenized", None)
    keys = None if tokenized is not None else tuple(bank)
    counts = Counter(indices.tolist())
    total = None
    for index, count in counts.items():
        if type(index) is not int or not 0 <= index < len(bank):
            raise ValueError("Invalid rehearsal target index")
        row = tokenized(index) if tokenized is not None else bank[keys[index]]
        candidate = {f"chosen_{name}": row[name] for name in
                     ("input_ids", "attention_mask", "labels", "completion_mask")}
        value = -completion_log_probability(model, candidate, "chosen", device=device) / row["completion_tokens"]
        if not bool(torch.isfinite(value)):
            raise ValueError("Nonfinite supervised rehearsal loss")
        weighted = value * (count / indices.numel())
        total = weighted if total is None else total + weighted
    return total


def add_rehearsal_loss(trainer: Any, model: Any, inputs: dict, rl_loss: Any,
                       bank: Mapping[str, dict], weight: float) -> Any:
    """Match Dr. GRPO's accumulation normalization before parent loss handling."""
    steps = trainer.current_gradient_accumulation_steps
    if type(steps) is not int or steps < 1:
        raise ValueError("Rehearsal requires the actual positive gradient accumulation length")
    with trainer.accelerator.autocast():
        ce = rehearsal_loss(model, inputs, bank, device=trainer.accelerator.device)
    trainer._metrics["train"]["rehearsal/completion_ce"].append(float(ce.detach()))
    trainer._metrics["train"]["rehearsal/weight"].append(weight)
    return rl_loss + weight * ce / steps
