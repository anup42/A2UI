from __future__ import annotations

from ir_training.data.splits import stratified_split


def test_source_id_group_never_crosses_split() -> None:
    rows = [
        {"id": "q1-a", "source_id": "q1", "intent_bucket": "travel"},
        {"id": "q1-b", "source_id": "q1", "intent_bucket": "travel"},
        {"id": "q2-a", "source_id": "q2", "intent_bucket": "travel"},
        {"id": "q2-b", "source_id": "q2", "intent_bucket": "travel"},
        {"id": "q3-a", "source_id": "q3", "intent_bucket": "status"},
        {"id": "q3-b", "source_id": "q3", "intent_bucket": "status"},
        {"id": "q4-a", "source_id": "q4", "intent_bucket": "status"},
        {"id": "q4-b", "source_id": "q4", "intent_bucket": "status"},
    ]
    splits = stratified_split(rows, 0.5, 0.25, 0.25, "intent_bucket", seed=42)
    memberships: dict[str, set[str]] = {}
    for split_name, split_rows in splits.items():
        for row in split_rows:
            memberships.setdefault(str(row["source_id"]), set()).add(split_name)
    assert memberships
    assert all(len(split_names) == 1 for split_names in memberships.values())
