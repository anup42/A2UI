from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def stratified_split(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify_key: str,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        key = metadata.get(stratify_key) or row.get(stratify_key) or "unknown"
        grouped[str(key)].append(row)

    splits = {"train": [], "val": [], "test": []}
    for group_rows in grouped.values():
        rng.shuffle(group_rows)
        n = len(group_rows)
        if n == 1:
            splits["train"].extend(group_rows)
            continue
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        if n >= 10:
            if val_ratio > 0:
                n_val = max(1, n_val)
            if test_ratio > 0:
                n_test = max(1, n_test)
        if n_val + n_test >= n:
            n_val = min(n_val, max(0, n - 1))
            n_test = min(n_test, max(0, n - 1 - n_val))
        n_train = n - n_val - n_test
        splits["train"].extend(group_rows[:n_train])
        splits["val"].extend(group_rows[n_train:n_train + n_val])
        splits["test"].extend(group_rows[n_train + n_val:])

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits
