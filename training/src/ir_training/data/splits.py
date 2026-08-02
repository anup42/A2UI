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
    group_key: str = "source_id",
) -> dict[str, list[dict[str, Any]]]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    rng = random.Random(seed)
    # Group before assigning a split. A source/query may materialize multiple
    # candidates or target views; none of those rows may straddle train/val/test.
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_id = row.get(group_key) or row.get("response_id") or row.get("id") or "unknown"
        source_groups[str(source_id)].append(row)

    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group_rows in source_groups.values():
        representative = group_rows[0]
        metadata = representative.get("metadata") if isinstance(representative.get("metadata"), dict) else {}
        key = metadata.get(stratify_key) or representative.get(stratify_key) or "unknown"
        grouped[str(key)].append(group_rows)

    splits = {"train": [], "val": [], "test": []}
    for source_group_rows in grouped.values():
        rng.shuffle(source_group_rows)
        n = len(source_group_rows)
        if n == 1:
            splits["train"].extend(source_group_rows[0])
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
        for group in source_group_rows[:n_train]:
            splits["train"].extend(group)
        for group in source_group_rows[n_train:n_train + n_val]:
            splits["val"].extend(group)
        for group in source_group_rows[n_train + n_val:]:
            splits["test"].extend(group)

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits
