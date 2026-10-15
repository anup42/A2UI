"""Explicit completion-only SFT and DPO+SFT math; lazy torch imports.

DPO uses sequence SUM log probabilities. SFT is mean per-example completion NLL.
No prompt/padding tokens are supervised, no truncated sequence is repaired, and
reference values are supplied by a frozen pre-optimizer snapshot of the QAT SFT
policy (not by disabling its adapter).
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


def encode_completion(row, tokenizer, *, template_kwargs=None, max_length: int,
                      completion: str | None = None):
    kwargs = dict(template_kwargs or {})
    if {"tokenize", "add_generation_prompt"} & kwargs.keys():
        raise ValueError("Template options cannot change training boundaries")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be a positive sequence budget")
    messages = deepcopy(row.get("messages"))
    if (not isinstance(messages, list) or len(messages) < 2 or messages[-1].get("role") != "assistant"
            or messages[-2].get("role") != "user" or messages[-1].get("content") != row.get("completion")):
        raise ValueError("Saved SFT messages must end with the exact validated completion")
    if completion is not None:
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError("A nonempty rejected completion is required")
        messages[-1]["content"] = completion
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kwargs)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if not prompt_ids or ids[:len(prompt_ids)] != prompt_ids or len(ids) <= len(prompt_ids):
        raise ValueError("Chat-template/tokenizer prompt boundary differs from the full completion")
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and len(ids) > 1 and ids[:2] == [bos, bos]:
        raise ValueError("Duplicated BOS at the training boundary")
    if len(ids) > max_length:
        raise ValueError("Complete preference/SFT sequence exceeds budget; truncation is forbidden")
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + ids[len(prompt_ids):], "prompt": prompt}


class FeedbackCollator:
    def __init__(self, pad_token_id: int):
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("An explicit padding token is required")
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        if not features:
            raise ValueError("Cannot collate an empty batch")
        encoded = [f["chosen"] for f in features]
        pairs = [i for i, f in enumerate(features) if f.get("rejected") is not None]
        encoded += [features[i]["rejected"] for i in pairs]
        width = max(len(f["input_ids"]) for f in encoded)
        ids, labels, masks = [], [], []
        for row in encoded:
            count = len(row["input_ids"])
            if count != len(row["labels"]) or not any(x != -100 for x in row["labels"][1:]):
                raise ValueError("Aligned labels with at least one completion token are required")
            ids.append(row["input_ids"] + [self.pad_token_id] * (width - count))
            labels.append(row["labels"] + [-100] * (width - count))
            masks.append([1] * count + [0] * (width - count))
        refs = [features[i].get("reference") for i in pairs]
        if any(not isinstance(ref, (tuple, list)) or len(ref) != 2
               or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in ref)
               for ref in refs):
            raise ValueError("Every preference needs two finite frozen reference log probabilities")
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
                "labels": torch.tensor(labels), "num_chosen": len(features),
                "pair_positions": torch.tensor(pairs, dtype=torch.long),
                "reference": torch.tensor(refs, dtype=torch.float32).reshape(-1, 2)}


def completion_logps(logits, labels, *, chunk_tokens=64):
    """Stable FP32 reductions in small chunks; never allocate full FP32 logits."""
    import torch
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("Invalid logits, labels or chunk size")
    values, counts = [], []
    for row in range(labels.shape[0]):
        positions = torch.where(labels[row, 1:] != -100)[0]
        if positions.numel() == 0:
            raise ValueError("Completion has no supervised tokens")
        chunks = []
        for part in positions.split(chunk_tokens):
            selected = logits[row, part].float()
            targets = labels[row, part + 1]
            if bool(((targets < 0) | (targets >= logits.shape[-1])).any()):
                raise ValueError("Completion token lies outside the model vocabulary")
            chunks.append((selected.gather(-1, targets[:, None]).squeeze(-1) - selected.logsumexp(-1)).sum())
        values.append(torch.stack(chunks).sum())
        counts.append(positions.numel())
    sums = torch.stack(values)
    if not bool(torch.isfinite(sums).all()):
        raise FloatingPointError("Nonfinite completion log probabilities")
    return sums, torch.tensor(counts, dtype=sums.dtype, device=sums.device)


def feedback_loss(logps, counts, num_chosen, pair_positions, reference, *, beta=0.1, sft_weight=1.0):
    import torch
    import torch.nn.functional as F
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
           for x in (beta, sft_weight)):
        raise ValueError("beta and sft_weight must be finite and positive")
    n = pair_positions.numel()
    if (type(num_chosen) is not int or num_chosen < 1 or logps.ndim != 1 or counts.shape != logps.shape
            or logps.numel() != num_chosen + n or reference.shape != (n, 2)
            or not bool((counts > 0).all()) or not bool(torch.isfinite(counts).all()) or not bool(torch.isfinite(logps).all())
            or not bool(torch.isfinite(reference).all())):
        raise ValueError("Invalid preference loss dimensions or finite values")
    if n and (bool(((pair_positions < 0) | (pair_positions >= num_chosen)).any())
              or pair_positions.unique().numel() != n):
        raise ValueError("Invalid paired chosen indices")
    reference = reference.detach()
    nll = -(logps[:num_chosen] / counts[:num_chosen]).mean()
    margin = (logps[pair_positions] - logps[num_chosen:]) - (reference[:, 0] - reference[:, 1])
    # Average across *all* examples, not just paired rows. This leaves ordinary
    # rehearsal present and does not give tiny paired subsets arbitrary weight.
    dpo = -F.logsigmoid(beta * margin).sum() / num_chosen
    loss = sft_weight * nll + dpo
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite DPO+SFT loss")
    metrics = {"sft_nll": nll.detach(), "dpo_loss": dpo.detach(),
               "paired_fraction": logps.new_tensor(n / num_chosen),
               "chosen_tokens": counts[:num_chosen].sum().detach(),
               "rejected_tokens": counts[num_chosen:].sum().detach()}
    return loss, metrics
