"""Portable identity helpers for the retained compiled-parity report."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping


OFFICIAL_RETAINED_COMPILED_REPORT_SHA256 = (
    "6744af4144688b93ef7009a67218700fcaa2b78b4186872496a8e17cd72e4008"
)


def canonical_retained_parity_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the path-independent report payload used for provenance hashing.

    Only the two machine-local path fields are removed.  All semantic fields,
    including artifact/source hashes and every retained mapping, remain bound.
    """

    canonical = copy.deepcopy(dict(report))
    for section in ("artifact", "source"):
        value = canonical.get(section)
        if isinstance(value, dict):
            value.pop("path", None)
    return canonical


def canonical_retained_parity_report_sha256(report: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_retained_parity_report(report),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "OFFICIAL_RETAINED_COMPILED_REPORT_SHA256",
    "canonical_retained_parity_report",
    "canonical_retained_parity_report_sha256",
]
