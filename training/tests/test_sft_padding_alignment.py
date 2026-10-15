"""Completion-only targets follow the tokenizer's actual padding side."""
from __future__ import annotations

import pytest
from ir_training.train.sft import _CausalLMDataCollator


class Tokenizer:
    pad_token_id = 2  # EOS is also padding; only attention may mask padding.

    def __init__(self, padding_side):
        self.padding_side = padding_side

    def pad(self, features, **kwargs):
        import torch

        width = max(len(row["input_ids"]) for row in features)
        values, masks = [], []
        for row in features:
            ids = list(row["input_ids"])
            padding = [self.pad_token_id] * (width - len(ids))
            mask = list(row.get("attention_mask", [1] * len(ids)))
            zeros = [0] * len(padding)
            values.append(padding + ids if self.padding_side == "left" else ids + padding)
            masks.append(zeros + mask if self.padding_side == "left" else mask + zeros)
        return {"input_ids": torch.tensor(values), "attention_mask": torch.tensor(masks)}


@pytest.mark.parametrize("side,expected", [
    ("left", [[-100, -100, -100, 2], [-100, -100, 4, 2]]),
    ("right", [[-100, 2, -100, -100], [-100, -100, 4, 2]]),
])
def test_completion_labels_align_with_unequal_padded_sequences(side, expected):
    pytest.importorskip("torch")
    collator = _CausalLMDataCollator(Tokenizer(side), vocab_size=8)
    batch = collator([
        {"input_ids": [1, 2], "labels": [-100, 2]},
        {"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, 2]},
    ])
    assert batch["labels"].tolist() == expected
    assert batch["labels"][batch["attention_mask"] == 0].tolist() == [-100, -100]


@pytest.mark.parametrize("bad_labels", [None, [-100], [-100, 2, 2]])
def test_mixed_or_misaligned_labels_fail_closed(bad_labels):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="labels must"):
        _CausalLMDataCollator(Tokenizer("left"), vocab_size=8)([
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [1, 2], "labels": bad_labels},
        ])
