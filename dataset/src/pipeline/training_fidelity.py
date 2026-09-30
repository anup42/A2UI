"""Dataset-only guidance and exact literal diagnostics; never rewrites a target.

This audit supplements, but does not override, v5.4 training acceptance. Exact
matching can flag valid paraphrases and regrouping; a passed audit is not a
semantic, source-factuality or device-rendering certificate.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
import unicodedata

from pipeline.genui_quality.config_v5_4 import coerce_reward_config_v5_4
from pipeline.genui_quality.evidence_v5_4 import collect_output_evidence_v5_4
from pipeline.genui_quality.graph import audit_renderer_graph
from pipeline.genui_quality.ownership_v5_4 import build_source_ownership
from pipeline.genui_quality.source_contract_v5_4 import extract_expected_ui_contract_v5_4

POLICY_VERSION = "stage3-training-fidelity-20261001-v1"
GUIDANCE_PATH = Path(__file__).resolve().parents[2] / "prompts/stage3_training_fidelity_v1.md"
_REFERENCES = re.compile(r"\[(?:[A-Z_]+URL|URL|[A-Z_]+ASSET)_?\d+\]|(?:https?|ftp)://[^\s<>\"']+")
_NUMBER = re.compile(r"(?<![\w.])[+\-−]?\d+(?:[.,:/\-]\d+)*(?:[%℃°]+)?")
_PREFIX = re.compile(r"(?m)^\s*(?:#{1,6}\s+|(?:[-*+•]|\d+[.)])\s+)")
_GROUPED_NUMBER = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d|,\d)")


def compose_training_fidelity_prompt(template: str) -> str:
    """Append to the actual Stage 3 template without changing its version/header."""
    guidance = GUIDANCE_PATH.read_text(encoding="utf-8").strip()
    if not guidance or "{response_text}" in guidance or template.count("{response_text}") != 1:
        raise ValueError("Stage 3 training fidelity requires one complete source placeholder")
    return template.rstrip() + "\n\n" + guidance + "\n"


def guidance_identity() -> dict[str, str]:
    return {"policy_version": POLICY_VERSION,
            "prompt_file": GUIDANCE_PATH.name,
            "prompt_sha256": hashlib.sha256(GUIDANCE_PATH.read_bytes()).hexdigest()}


def _plain(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace("−", "-")
    text = _PREFIX.sub("", text.replace("**", "").replace("__", ""))
    text = _REFERENCES.sub(" ", text)
    return _GROUPED_NUMBER.sub(lambda match: match[0].replace(",", ""), text)


def _tokens(value: Any) -> str:
    text = _plain(value).casefold().replace(">=", "≥").replace("<=", "≤").replace("!=", "≠")
    return " ".join(re.findall(r"[+\-]?\d+(?:[.,:/\-]\d+)*|\w+|[<>≤≥≠=+%$€£¥₹°]", text))


def _numbers(value: Any) -> set[str]:
    return set(_NUMBER.findall(_plain(value)))


@lru_cache(maxsize=1)
def _config():
    return coerce_reward_config_v5_4(None)


def bounded_fidelity_config(source: str, graph: Mapping[str, Any], config=None):
    """Keep full concrete literals under the unchanged total evidence budgets."""
    config = _config() if config is None else config
    graph_size = len(json.dumps(graph, ensure_ascii=False))
    bound = min(config.max_dynamic_evidence_bytes,
                max(config.max_string_expansion_length, len(source), graph_size))
    return replace(config, max_string_expansion_length=bound)


def _static_card_labels(graph: Mapping[str, Any], reachable: set[str]) -> list[str]:
    """Only literal normal-route Card labels, with special/dynamic routes held out."""
    def dynamic(value: Any) -> bool:
        if isinstance(value, Mapping):
            if any(str(key).startswith("$") or str(key).casefold() in
                   {"visible", "visibility", "hidden", "condition", "repeat", "template", "if", "when"}
                   for key in value):
                return True
            return any(dynamic(item) for item in value.values())
        return isinstance(value, list) and any(dynamic(item) for item in value)

    if dynamic(graph):
        return []
    elements = graph.get("elements", {})
    labels = []
    for element_id in sorted(reachable):
        element = elements.get(element_id, {})
        if element.get("type") != "Card":
            continue
        pending, seen, subtree = [element_id], set(), []
        while pending:
            child_id = pending.pop()
            if child_id in seen:
                continue
            seen.add(child_id)
            child = elements.get(child_id, {})
            subtree.append(child)
            pending.extend(child.get("children") or [])
        serialized = json.dumps(subtree, ensure_ascii=False)
        if _REFERENCES.search(serialized) or any(child.get("type") == "EmailPreview" for child in subtree):
            continue
        child_text = "\n".join(str((child.get("props") or {}).get("text", "")) for child in subtree)
        if re.search(r"(?im)^\s*(?:subject|from|to|sender|recipient|body)\s*:", child_text):
            continue
        labels.extend(value for key in ("title", "subtitle")
                      if isinstance(value := (element.get("props") or {}).get(key), str) and value.strip())
    return labels


def _join_diagnostics(source: str, visible: list[str]) -> list[dict[str, Any]]:
    """Report target words that join two source words across whitespace; never edit."""
    source = _plain(source)
    source_words = set(re.findall(r"\w+", source.casefold()))
    # Lookahead includes overlapping adjacent pairs (A B and B C).
    adjacent = {match[1] + match[2]: (match[1], match[2])
                for match in re.finditer(r"(?=\b(\w+)\s+(\w+)\b)", source.casefold())}
    found = []
    seen = set()
    for index, block in enumerate(visible):
        for match in re.finditer(r"\b\w{2,}\b", block):
            word = match[0].casefold()
            if word not in source_words and word in adjacent and word not in seen:
                seen.add(word)
                found.append({"visible_block": index, "target_word": match[0],
                              "source_words": list(adjacent[word])})
    return found


def audit_training_fidelity(source: str, graph: Mapping[str, Any]) -> dict[str, Any]:
    """Check all literal prose/heading units using renderer-visible evidence.

    Bound evidence expansion by the concrete source/graph and retain the existing
    aggregate work/byte limits. Hitting a limit is explicitly incomplete, never
    a proof of missing source or permission to clip/modify the training pair.
    """
    graph_text = json.dumps(graph, ensure_ascii=False, sort_keys=True)
    bounded_config = bounded_fidelity_config(source, graph)
    bound = bounded_config.max_string_expansion_length
    output = collect_output_evidence_v5_4(graph, audit_renderer_graph(graph), bounded_config)
    visible = list(output.output.visible_blocks)
    labels = _static_card_labels(graph, output.output.reachable_ids)
    visible.extend(labels)
    # Numeric field associations are represented through their own visible row,
    # not by scanning arbitrary state or props. Table.title is intentionally absent.
    contexts = [" ".join(str(header) + " " + str(row.get(key, ""))
                          for header, key in zip(table.headers, table.keys))
                for table in output.output_tables for row in table.rows]
    for role in output.role_instances.get("email", ()):
        for field, label in (("subject", "Subject"), ("from", "From"), ("to", "To")):
            if role.get(field):
                contexts.append(label + ": " + str(role[field]))
    represented = " " + _tokens("\n".join(visible)) + " "
    context_tokens = [" " + _tokens(value) + " " for value in contexts]
    contract = extract_expected_ui_contract_v5_4(source)
    units = [unit for unit in build_source_ownership(source, contract)
             if unit.owner in {"generic_prose", "heading"}]
    missing = []
    for unit in units:
        literal = _tokens(unit.text)
        present = not literal or " " + literal + " " in represented or any(
            " " + literal + " " in context for context in context_tokens)
        if not present:
            missing.append({"unit_id": unit.unit_id, "owner": unit.owner,
                            "source_span": list(unit.source_span) if unit.source_span else None,
                            "text": unit.text})
    unsupported = sorted(_numbers("\n".join(visible)) - _numbers(source))
    joins = _join_diagnostics(source, visible)
    unsupported_charts = [item for item in output.effective_components
                          if item.get("kind") == "chart" and
                          "unsupported_chart_subtype" in item.get("diagnostics", ())]
    reasons = []
    if not output.dynamic_evidence_complete:
        reasons.append("renderer_evidence_incomplete")
    if missing:
        reasons.append("source_literal_not_exactly_represented")
    if unsupported:
        reasons.append("unsupported_visible_numeric_anchor")
    if joins:
        reasons.append("suspected_source_word_join")
    if unsupported_charts:
        reasons.append("unsupported_chart_subtype")
    return {"policy_version": POLICY_VERSION, "status": "needs_review" if reasons else "checks_passed",
            "diagnostic_only": True, "reasons": reasons,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "graph_sha256": hashlib.sha256(graph_text.encode("utf-8")).hexdigest(),
            "source_unit_count": len(units), "checked_source_unit_count": len(units),
            "missing_unit_count": len(missing), "missing_units": missing[:16],
            "missing_unit_examples_truncated": len(missing) > 16,
            "unsupported_numeric_anchors": unsupported,
            "suspected_word_join_count": len(joins), "suspected_word_joins": joins[:16],
            "unsupported_chart_count": len(unsupported_charts),
            "unsupported_charts": [{"component_id": item["component_id"], "chart_type": item.get("chart_type")}
                                   for item in unsupported_charts],
            "visible_block_count": len(visible), "normal_card_label_count": len(labels),
            "table_row_binding_context_count": len(contexts), "evidence_string_bound": bound,
            "renderer_evidence_complete": output.dynamic_evidence_complete,
            "renderer_diagnostics": list(output.unknown_diagnostics),
            "limitations": "Exact literal diagnostic only; v5.4 acceptance remains authoritative. "
                           "Paraphrases may need review; passing does not prove entity association, "
                           "all added nonnumeric claims, source factuality or device rendering."}
