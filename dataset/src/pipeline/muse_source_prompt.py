"""Select a non-conflicting Stage 2 source prompt for the exact local Muse model."""

from __future__ import annotations

from pathlib import Path

from llm.base import ModelSpec

MUSE_MODEL = "meta-models/Muse-Glimmer-30B"
PROMPT_VERSION = "muse_stage2_quality_v1"
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / f"{PROMPT_VERSION}.md"


def uses_muse_source_prompt(spec: ModelSpec) -> bool:
    return spec.provider.lower() == "local" and spec.model == MUSE_MODEL


def select_stage2_prompt(template: str, spec: ModelSpec) -> str:
    """Replace, rather than prepend to, the generic live-data/style instructions.

    The caller still appends the per-query source contract and batch instructions.
    The resulting prompt participates in Stage 2's existing cache key.
    """
    if not uses_muse_source_prompt(spec):
        return template
    selected = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not selected.startswith(f"# {PROMPT_VERSION}") or any(
        selected.count("{" + name + "}") != 1 for name in ("query_text", "intent", "tags")
    ):
        raise ValueError("Muse Stage 2 prompt requires its version and one of each task placeholder")
    return selected
