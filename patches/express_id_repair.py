"""Deprecated diagnostic API: ambiguous component-ID repair is disabled.

Two definitions with the same ID do not encode which definition a reference
intended. The old nearest-definition heuristic was not lossless. Termination
is a separate generation policy; this API does not trim output either.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "src"))
from ir_training.data.express_preparation import _api

OPEN, CLOSE = "<a2ui>", "</a2ui>"


@dataclass(frozen=True)
class RepairResult:
    text: str
    changed: bool
    status: str
    renamed_defs: int = 0
    redirected_refs: int = 0
    dropped_sentinels: int = 0
    detail: str | None = None


def repair(text: str) -> RepairResult:
    active, _, _, _ = _api()
    try:
        active.decode_express_completion(text)
    except ValueError as exc:
        detail = str(exc)
        status = "ambiguous_duplicate_ids_repair_disabled" if "Duplicate A2UI Express component id" in detail else "invalid_no_repair"
        return RepairResult(text, False, status, detail=detail)
    return RepairResult(text, False, "nothing_to_repair")


def content_signature(text: str):
    """Full canonical semantic hash, or None for invalid input.

    Diagnostic compatibility only: invalid duplicate IDs have no signature
    from which the intended attachments can be recovered.
    """
    active, _, semantic_hash, _ = _api()
    try:
        return semantic_hash(active.decode_express_completion(text))
    except ValueError:
        return None
