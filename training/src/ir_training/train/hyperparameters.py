"""Validated overrides for the portable Golden training recipes (no model load)."""
from __future__ import annotations

import math


def review_overrides(*, learning_rate=None, weight_decay=None, warmup_ratio=None,
                     logging_steps=10, seed=42) -> dict:
    values = {}
    for name, value in (("learning_rate", learning_rate), ("weight_decay", weight_decay),
                        ("warmup_ratio", warmup_ratio)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if (name == "learning_rate" and value <= 0) or (name == "weight_decay" and value < 0) or (name == "warmup_ratio" and not 0 <= value < 1):
            raise ValueError(f"Invalid {name}: {value}")
        values[name] = float(value)
    if type(logging_steps) is not int or logging_steps <= 0:
        raise ValueError("logging_steps must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return {**values, "logging_steps": logging_steps, "seed": seed}
