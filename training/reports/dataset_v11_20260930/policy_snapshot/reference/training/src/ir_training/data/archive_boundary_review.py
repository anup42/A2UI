"""Source-prove the exact boundary of historical 180-character Text joins."""

import re
from copy import deepcopy

from ir_training.data.archive_recovery import _join_midword_text_chunks

POLICY = "messages-archive-boundary-review-v9-20260913"


def _plain(value):
    value = value.replace("**", "").replace("__", "")
    value = re.sub(r"(?m)^\s*[-*•]\s+", "", value)
    return " ".join(value.split())


def repair_join_boundaries(source, graph, original_graph):
    """Recheck only values actually formed by the historical join helper.

    A length of 180 and two alphanumeric endpoints are not enough to prove a
    word was split. Require unique source context across the boundary. Preserve
    a real split word; otherwise insert only the separator present in source.
    """
    joined, joins = _join_midword_text_chunks(original_graph)
    old_text = {
        e.get("props", {}).get("text")
        for e in original_graph["elements"].values()
        if e.get("type") == "Text"
    }
    joined_values = {
        e.get("props", {}).get("text")
        for e in joined["elements"].values()
        if e.get("type") == "Text" and e.get("props", {}).get("text") not in old_text
    }
    fixed, proofs, issues = deepcopy(graph), [], []
    if len(joined_values) != joins:
        issues.append("historical_join_values_not_uniquely_reconstructed")
    normalized = _plain(source)
    for value in sorted(joined_values):
        candidates = [
            e
            for e in fixed["elements"].values()
            if e.get("type") == "Text" and e.get("props", {}).get("text") == value
        ]
        # Idempotence: an already source-separated value is also recognized.
        left, right = value[:180], value[180:]
        if len(left) != 180 or not right:
            issues.append("join_context_too_short")
            continue
        pattern = (
            re.escape(_plain(left[-64:]))
            + r"(?P<gap>[\s.,;:!?—–-]{0,8})"
            + re.escape(_plain(right[:64]))
        )
        matches = list(re.finditer(pattern, normalized))
        if len(matches) != 1:
            issues.append("join_boundary_not_uniquely_proven_in_source")
            continue
        separator = matches[0]["gap"]
        corrected = left + separator + right
        if not candidates:
            already = [
                e
                for e in fixed["elements"].values()
                if e.get("type") == "Text"
                and e.get("props", {}).get("text") == corrected
            ]
            if len(already) != 1:
                issues.append("historical_join_not_found_in_current_target")
            continue
        if len(candidates) != 1:
            issues.append("historical_join_ambiguous_in_current_target")
            continue
        if separator:
            candidates[0]["props"]["text"] = corrected
            proofs.append(
                {
                    "before_boundary": left[-24:] + right[:24],
                    "after_boundary": left[-24:] + separator + right[:24],
                    "separator_from_source": separator,
                }
            )
    return fixed, proofs, issues
