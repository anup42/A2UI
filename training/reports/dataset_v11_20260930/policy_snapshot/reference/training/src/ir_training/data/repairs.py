"""Training-facing import surface for the shared IR repair layer."""
from __future__ import annotations

import sys

from ir_training.common.config import repo_root

_DATASET_SRC = repo_root() / "dataset" / "src"
if str(_DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(_DATASET_SRC))

from pipeline.ir_formats.repair import (  # noqa: E402,F401
    GAP_TOKENS,
    RepairChange,
    RepairResult,
    normalize_gap,
    repair_graph,
)

__all__ = ["GAP_TOKENS", "RepairChange", "RepairResult", "normalize_gap", "repair_graph"]
