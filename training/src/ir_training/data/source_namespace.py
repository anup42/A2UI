"""Versioned identities for inferred IDs that are local to a Stage 3 source."""
from __future__ import annotations

import hashlib
import re

_PREFIX = "run-local-v1:"


def scoped_source_id(source_key: str, local_id: str) -> str:
    if not source_key or not local_id:
        raise ValueError("Run-local source identity requires provenance and a local ID")
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
    return f"{_PREFIX}{digest}:{local_id}"


def source_namespace(source_id: str) -> str | None:
    match = re.match(r"^(run-local-v1:[0-9a-f]{64}):.+$", source_id)
    return match.group(1) if match else None
