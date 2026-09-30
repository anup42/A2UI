"""Explicit one-time legacy import boundary for training data.

This module is never used to construct an active completion target.  It exists
only so historical FlatSpec rows can be decoded once into the shared
canonical graph before :mod:`ir_targets` emits A2UI Express text. Compact IR
must be converted by the isolated dataset migration command first; keeping
its decoder out of normal training prevents a second active dependency.
"""
from __future__ import annotations

from typing import Any

from ir_training.common.config import repo_root

FLAT_SPEC_V1 = "flat_spec_v1"
_LEGACY_FORMATS = {FLAT_SPEC_V1}


def canonical_graph_from_legacy_source(
    value: Any,
    source_format: str = FLAT_SPEC_V1,
) -> dict[str, Any]:
    token = str(source_format or FLAT_SPEC_V1).strip().lower()
    if token in {"compact_ir_v2", "compact_ir", "gci2"}:
        raise ValueError(
            "Compact IR is migration-only; run dataset/scripts/"
            "migrate_legacy_dataset_to_a2ui_express.py before training"
        )
    if token not in _LEGACY_FORMATS:
        raise ValueError(f"Unsupported legacy source format {source_format!r}")
    dataset_src = repo_root() / "dataset" / "src"
    import sys

    if str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))
    from pipeline.ir_formats import decode_to_flat_spec

    return decode_to_flat_spec(value, format_hint=token).flat_spec


def canonical_flat_spec(value: Any, source_format: str | None = None) -> dict[str, Any]:
    """Compatibility name retained only for migration/offline callers."""

    if isinstance(value, str) and value.strip().startswith("<a2ui>"):
        dataset_src = repo_root() / "dataset" / "src"
        import sys

        if str(dataset_src) not in sys.path:
            sys.path.insert(0, str(dataset_src))
        from pipeline.ir_formats import decode_express_completion

        return decode_express_completion(value)
    return canonical_graph_from_legacy_source(value, source_format or FLAT_SPEC_V1)


__all__ = ["FLAT_SPEC_V1", "canonical_flat_spec", "canonical_graph_from_legacy_source"]
