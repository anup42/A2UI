"""Compatibility wrapper for checked children-before-parent serialization.

Uses the production catalog/reference inventory. Invalid or disconnected input
is returned unchanged with a reason; IDs and literals are never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "src"))
from ir_training.data.express_preparation import PreparationError, serialize_checked


@dataclass(frozen=True)
class Result:
    text: str
    changed: bool
    status: str
    detail: str | None = None


def to_bottom_up(text: str) -> Result:
    if not isinstance(text, str):
        return Result(text, False, "express_invalid", "A2UI Express payload must be text")
    try:
        converted = serialize_checked(text, "bottom-up")
    except PreparationError as exc:
        return Result(text, False, exc.reason, str(exc))
    return Result(converted.text, converted.text != text, "reordered")
