"""Prepare a hash-pinned evaluation-only cohort without reselecting membership."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ir_training.common.config import resolve_path, training_root
from ir_training.data.express_preparation import prepare_splits
from ir_training.eval.golden_set import benchmark_contract_for_split, load_fixed_golden_rows


def prepare_frozen_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    run = config.get("run") or {}
    filters = config.get("filters") or {}
    source = resolve_path(run["frozen_eval_source"], training_root())
    expected = str(run.get("frozen_eval_sha256") or "")
    if len(expected) != 64 or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Frozen evaluation source hash differs from its independent config pin")
    count = filters.get("required_accepted_rows")
    if type(count) is not int or count <= 0 or filters.get("require_exact_accepted_rows") is not True or filters.get("require_unique_source_ids") is not True:
        raise ValueError("Frozen evaluation requires exact positive row count and unique source IDs")
    rows = load_fixed_golden_rows(source, required_rows=count, require_unique_rows=True)
    contract = benchmark_contract_for_split(source, rows)
    if not contract or contract.get("benchmark_id") != run.get("frozen_eval_benchmark_id"):
        raise ValueError("Frozen evaluation benchmark identity differs from config")
    output = resolve_path(run["output_dir"], training_root())
    # prepare_splits refuses existing destinations and carries the approved
    # source/target membership into the strict preparation manifest.
    return prepare_splits({"all": Path(source)}, output, ordering="root-first")
