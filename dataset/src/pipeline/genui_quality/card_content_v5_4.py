"""Conservative, occurrence-aware matching of source units to card fields.

Only the assignment view is grouped. Raw visible blocks remain authoritative
for word/value counts, duplication, links, media, and all validation gates.
This is not a Card-count bonus or permission to bag-match unrelated fields.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .matching_v5_1 import PreparedTextBlock, prepare_text_block

CARD_CONTENT_POLICY_VERSION = "1.0.0"


def card_content_assignment_v5_4(
    source: Sequence[PreparedTextBlock],
    ownership: Sequence[Mapping[str, Any]],
    generic_owners: set[str],
) -> tuple[tuple[PreparedTextBlock, ...], dict[str, Any]]:
    """Collapse a card's fields only after certifying a whole source unit.

    Ordered token equality preserves entity/field/value association, while
    allowing harmless punctuation and component boundaries. Partial, invented,
    duplicated, reordered, or cross-entity fields receive no grouping credit.
    Each visible occurrence is consumed once; repeated templates never share
    an occurrence ID. Nested card boundaries cannot be collapsed together.
    The downstream certified assignment still enforces source multiplicity.
    """
    rows = [item for item in ownership if item.get("owner") in generic_owners
            and str(item.get("text") or "").strip()]
    blocks = tuple(prepare_text_block(str(item["text"])) for item in rows)
    source_tokens = {block.tokens for block in source if block.tokens}
    positions: dict[str, list[int]] = defaultdict(list)
    nested_parents: set[str] = set()
    for index, item in enumerate(rows):
        instance = item.get("card_instance")
        if isinstance(instance, str) and instance:
            positions[instance].append(index)
            nested_parents.update(str(value) for value in item.get("card_ancestors", ())
                                  if value != instance)
    replacements: dict[int, PreparedTextBlock] = {}
    consumed: set[int] = set()
    groups: list[dict[str, Any]] = []
    for instance, indices in positions.items():
        if len(indices) < 2 or instance in nested_parents:
            continue
        if any(rows[index].get("component_type") not in {"card", "text"} for index in indices):
            continue
        # Do not move evidence across another visible record or component.
        if indices != list(range(indices[0], indices[-1] + 1)):
            continue
        joined = prepare_text_block(" ".join(blocks[index].raw for index in indices))
        if joined.tokens not in source_tokens:
            continue
        replacements[indices[0]] = joined
        consumed.update(indices[1:])
        groups.append({"card_instance": instance, "field_count": len(indices),
                       "output_indices": indices, "ordered_tokens_exact": True})
    result = tuple(replacements.get(index, block) for index, block in enumerate(blocks)
                   if index not in consumed)
    return result, {"policy_version": CARD_CONTENT_POLICY_VERSION,
                    "raw_block_count": len(blocks), "assignment_block_count": len(result),
                    "certified_groups": groups}
