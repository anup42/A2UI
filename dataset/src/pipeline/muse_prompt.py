"""Muse-only Stage 3 guidance layered over the generated Express contract."""
from __future__ import annotations

import json
from pathlib import Path

from llm.base import ModelSpec

MUSE_MODEL = "meta-models/Muse-Glimmer-30B"
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "muse_stage3_quality_v1.md"


def uses_muse_stage3_prompt(spec: ModelSpec) -> bool:
    return spec.provider.lower() == "local" and spec.model == MUSE_MODEL


def compose_stage3_prompt(template: str, spec: ModelSpec) -> str:
    """Retain the canonical grammar/catalog verbatim and version the overlay.

    The composed text participates in the existing per-request prompt hash and
    attempt ledger; the phase manifest also hashes every prompt Markdown file.
    """
    if not uses_muse_stage3_prompt(spec):
        return template
    guidance = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not guidance or "{response_text}" in guidance or template.count("{response_text}") != 1:
        raise ValueError("Muse Stage 3 requires nonempty guidance and exactly one source placeholder")
    return f"{guidance}\n\n{template}"


def format_muse_source(source: str, asset_policy: str, asset_mapping: str = "") -> str:
    """Separate visible source data from pipeline metadata without delimiter collisions."""
    return json.dumps(
        {"source_response": source,
         "reference_metadata": {"asset_policy": asset_policy, "asset_mapping": asset_mapping}},
        ensure_ascii=False,
    )
