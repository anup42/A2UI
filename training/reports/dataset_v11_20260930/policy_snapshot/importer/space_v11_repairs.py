"""Deterministic, audited reuse of the v10 source-grounded repair chain.

The caller must initialize the explicitly selected v10 policy repository first.
The source archive is never changed. Every emitted edit carries exact JSON
pointer before/after evidence, source hashes and the originating repair stage.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import inspect
import json
import re
import sys

sys.dont_write_bytecode = True
POLICY = "space-v11-v10-source-proven-repairs-20260930"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=1)
def _semantic_function():
    """Derive one guarded expression without changing the reference checkout.

    Element IDs and arbitrary state keys can be literally 'type'; their values
    need not be hashable component type strings. The membership test is the
    only changed expression. All source repair rules remain byte-identical.
    """
    from ir_training.data import archive_semantic_review as policy
    original = inspect.getsource(policy.repair_graph)
    before = 'protected |= value.get("type") in {"CodeBlock", "ConsoleLog", "Formula"}'
    after = 'protected |= isinstance(value.get("type"), str) and value.get("type") in {"CodeBlock", "ConsoleLog", "Formula"}'
    if original.count(before) != 1:
        raise ValueError("v10 semantic repair function changed; re-audit guarded derivation")
    derived = original.replace(before, after, 1)
    namespace = dict(vars(policy))
    exec(compile(derived, "<space_v11_guarded_v10_repair_graph>", "exec"), namespace)
    return namespace["repair_graph"], {
        "function": "ir_training.data.archive_semantic_review.repair_graph",
        "original_sha256": _sha(original), "derived_sha256": _sha(derived),
        "exact_original_expression": before, "exact_derived_expression": after,
    }


def implementation_fingerprint():
    return deepcopy(_semantic_function()[1])


def _diff(before, after, path=""):
    """Exact deterministic RFC6901 locations; retain explicit absence flags."""
    if type(before) is type(after) and before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            location = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                result.append({"location": location, "before_exists": key in before,
                               "after_exists": key in after,
                               "before": deepcopy(before.get(key)), "after": deepcopy(after.get(key))})
            else:
                result.extend(_diff(before[key], after[key], location))
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [item for index, (left, right) in enumerate(zip(before, after))
                for item in _diff(left, right, f"{path}/{index}")]
    return [{"location": path, "before_exists": True, "after_exists": True,
             "before": deepcopy(before), "after": deepcopy(after)}]


def normalize_for_repair(source, graph, mapping):
    """Mask source first with exact inverse, retaining known reference identity.

    Unresolved target tokens survive normalization for the v10 rebinder/pruner;
    closure is mandatory after the repair chain. Known symbolic identities are
    restored only through the supplied map, never guessed from token suffixes.
    """
    from ir_training.data.reference_binding import reference_tokens
    from ir_training.data.url_preprocess import (
        SOURCE_IDENTITY_BINDING, UrlPreprocessResult,
        preprocess_training_urls, restore_url_placeholders,
    )
    mapping = deepcopy(mapping or {})
    for token, entry in mapping.items():
        destination = entry.get("url") if isinstance(entry, dict) else entry
        if not isinstance(token, str) or not isinstance(destination, str) or not destination:
            raise ValueError("invalid_reference_identity_map")
    extras = reference_tokens(graph) - reference_tokens(source)
    mixed = bool(extras & mapping.keys())
    input_source = restore_url_placeholders(source, mapping) if mixed else source
    input_graph = restore_url_placeholders(graph, mapping) if mixed else deepcopy(graph)
    processed = preprocess_training_urls(input_source, input_graph, binding_policy=SOURCE_IDENTITY_BINDING)
    if restore_url_placeholders(processed.response_text, processed.url_map) != input_source or restore_url_placeholders(processed.canonical_graph, processed.url_map) != input_graph:
        raise ValueError("url_mask_not_lossless")
    used = reference_tokens(processed.response_text) | reference_tokens(processed.canonical_graph)
    retained = {token: value for token, value in mapping.items() if token in used and token not in processed.url_map}
    combined = {**retained, **processed.url_map}
    if restore_url_placeholders(processed.response_text, combined) != restore_url_placeholders(source, mapping) or restore_url_placeholders(processed.canonical_graph, combined) != restore_url_placeholders(graph, mapping):
        raise ValueError("reference_identity_not_lossless")
    return UrlPreprocessResult(processed.response_text, processed.canonical_graph, combined, processed.metrics), mixed


def _safe_rebind(source, graph, mapping):
    from ir_training.data.archive_recovery import _rebind_placeholders, placeholder_tokens
    rebound, metrics = _rebind_placeholders(source, graph)
    blocked = 0

    def destination(token):
        entry = mapping.get(token)
        return entry.get("url") if isinstance(entry, dict) else entry

    def guard(before, after):
        nonlocal blocked
        if isinstance(before, dict) and isinstance(after, dict):
            return {key: guard(value, after[key]) for key, value in before.items()}
        if isinstance(before, list) and isinstance(after, list):
            return [guard(left, right) for left, right in zip(before, after)]
        if isinstance(before, str) and isinstance(after, str) and before != after:
            removed, added = placeholder_tokens(before)-placeholder_tokens(after), placeholder_tokens(after)-placeholder_tokens(before)
            # A recorded target URL must never be rebound to a different URL or
            # an unknown destination merely because its label/index resembles it.
            if any(destination(old) is not None and any(destination(old) != destination(new) for new in added) for old in removed):
                blocked += 1
                return before
        return after

    rebound = guard(graph, rebound)
    if blocked:
        metrics = {"source_identity_guarded_placeholder_rebindings": len(_diff(graph, rebound))} if rebound != graph else {}
    return rebound, metrics


def _source_lines(source):
    """Literal prose lines only; leave code, tables and reference declarations alone."""
    fenced = False
    result = []
    for number, raw in enumerate(source.splitlines(), 1):
        line = raw.strip()
        if re.match(r"^(?:```|~~~)", line):
            fenced = not fenced
            continue
        if fenced or not line or "|" in line or "`" in line or re.search(r"\[(?:[A-Z_]+URL|URL|[A-Z_]+ASSET)_\d+\]", line):
            continue
        if re.match(r"^(?:Media|Action|Source|Sources|Formula|Chart(?: Title)?|[XY]-Axis)\s*:", line, re.I):
            continue
        literal = re.sub(r"^#{1,6}\s+", "", line).replace("**", "").replace("__", "")
        literal = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", literal)
        result.append((number, literal))
    return result


def _restore_small_literal_expansions(source, graph):
    """Restore <=3 omitted source words in one uniquely matching existing item.

    Existing characters, punctuation and numbers must appear contiguously in
    the source, ignoring case only. No paraphrase or value replacement is used.
    """
    from ir_training.data.archive_semantic_review import norm, visible_strings
    fixed = deepcopy(graph)
    units = [(number, literal, norm(literal)) for number, literal in _source_lines(source)]
    proofs = []
    visible_blocks = [" " + norm(value) + " " for value in visible_strings(fixed)]
    neighbors = {}
    for parent in fixed["elements"].values():
        if parent.get("type") not in {"Stack", "Card"} or any(key in parent for key in ("visible", "repeat", "watch")):
            continue
        children = parent.get("children", [])
        for index, child in enumerate(children):
            adjacent = []
            for step in (-1, 1):
                position = index + step
                # One decorative icon between a label and its body does not
                # invalidate their same-parent content association.
                if 0 <= position < len(children) and fixed["elements"][children[position]].get("type") == "Icon":
                    position += step
                element = fixed["elements"].get(children[position]) if 0 <= position < len(children) else None
                if element and element.get("type") in {"Text", "Button"} and not any(k in element for k in ("visible", "repeat", "watch")):
                    key = "label" if element.get("type") == "Button" else "text"
                    adjacent.append(norm(str(element.get("props", {}).get(key, ""))))
                else:
                    adjacent.append("")
            if child in neighbors:
                neighbors[child] = ("", "")  # Ambiguous shared node.
            else:
                neighbors[child] = tuple(adjacent)
    for eid, element in fixed["elements"].items():
        props = element.get("props", {})
        targets = []
        if element.get("type") == "Text" and isinstance(props.get("text"), str):
            targets = [(props, "text", f"/elements/{eid}/props/text")]
        elif element.get("type") == "List" and isinstance(props.get("items"), list):
            targets = [(props["items"], i, f"/elements/{eid}/props/items/{i}") for i, value in enumerate(props["items"]) if isinstance(value, str)]
        for container, key, location in targets:
            old = container[key]
            old_words = norm(old).split()
            if "\n" in old or not 4 <= len(old_words) <= 40 or len(old) > 220 or "`" in old:
                continue
            candidates = {}
            for line, literal, normalized in units:
                new_words = normalized.split()
                added = len(new_words) - len(old_words)
                if not 1 <= added <= 3 or len(literal) > 240 or len(old_words)/len(new_words) < .75:
                    continue
                match = re.search(r"(?<!\w)" + re.escape(old) + r"(?!\w)", literal, re.I)
                if not match or any((" " + normalized + " ") in block for block in visible_blocks):
                    continue
                added_before, added_after = norm(literal[:match.start()]), norm(literal[match.end():])
                # A heading and body can be separated by an icon, badge or
                # other renderer prop. Do not repeat a label already visible
                # in the same adjacent child group merely to make their
                # concatenation contiguous. Unrelated labels do not qualify.
                prior_text, following_text = neighbors.get(eid, ("", ""))
                if element.get("type") in {"Text", "List"} and (not added_before or added_before == prior_text) and (not added_after or added_after == following_text):
                    continue
                candidates[literal] = line
            if len(candidates) == 1:
                literal, line = next(iter(candidates.items()))
                container[key] = literal
                proofs.append({"kind": "unique_short_source_literal_expansion", "location": location,
                               "before": old, "after": literal, "source_line": line,
                               "source_literal": literal, "evidence": "One source line contains every original character contiguously (case-insensitive), with <=3 added words"})
                visible_blocks.append(" " + norm(literal) + " ")
    return fixed, proofs


def _unique_anchor(source, element):
    """Find a unique literal source span for one adjacent renderer child."""
    if any(key in element for key in ("visible", "repeat", "watch", "on")):
        return None
    kind, props = element.get("type"), element.get("props", {})
    if kind == "Text":
        values = [props.get("text")]
    elif kind == "CodeBlock":
        values = [props.get("code")]
    elif kind == "List" and not element.get("children"):
        values = props.get("items")
    else:
        return None
    if not isinstance(values, list) or not values or not all(isinstance(v, str) and len(v.strip()) >= 8 for v in values):
        return None
    spans = []
    for value in values:
        value = value.strip()
        hits = list(re.finditer(re.escape(value), source))
        if len(hits) != 1:
            return None
        spans.append((hits[0].start(), hits[0].end()))
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        return None
    start, end = spans[0][0], spans[-1][1]
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", end)
    if line_end < 0:
        line_end = len(source)
    before, after = source[line_start:start].strip(), source[end:line_end].strip()
    if kind == "CodeBlock":
        if before or after:
            return None
        prior_start = source.rfind("\n", 0, max(0, line_start-1)) + 1
        prior = source[prior_start:line_start].strip()
        next_end = source.find("\n", min(len(source), line_end+1))
        if next_end < 0:
            next_end = len(source)
        following = source[line_end:next_end].strip()
        if not re.fullmatch(r"(?:```|~~~)[\w+-]*", prior) or following not in {"```", "~~~"}:
            return None
        start, end = prior_start, next_end
    else:
        if before and not re.fullmatch(r"(?:#{1,6}\s*|[-*•]\s*|\d+[.)]\s*)?(?:\*\*|__)?", before):
            return None
        if after not in {"", "**", "__"}:
            return None
        start, end = line_start, line_end
    return start, end


def _restore_anchored_prose_gaps(source, graph):
    """Restore one short literal prose line between two unique adjacent anchors.

    Only unconditional direct children of Stack/Card qualify. Existing body
    Text is extended when possible; otherwise one plain Text is inserted in
    the proven source order. Up to six gaps and 1,200 characters per pass;
    repair_chain permits at most three passes and requires clean replay.
    """
    from ir_training.data.archive_semantic_review import norm, visible_strings
    from ir_training.data.express_preparation import _api
    fixed = deepcopy(graph)
    _, _, _, references = _api()
    incoming = Counter(edge.target_id for e in fixed["elements"].values() for edge in references(e))
    visible = " " + norm("\n".join(visible_strings(fixed))) + " "
    anchors = {eid: _unique_anchor(source, element) for eid, element in fixed["elements"].items()}
    proofs, added_chars = [], 0
    for pid, parent in list(fixed["elements"].items()):
        if parent.get("type") not in {"Stack", "Card"} or any(key in parent for key in ("visible", "repeat", "watch")):
            continue
        children = parent.get("children", [])
        output = []
        for left, right in zip(children, children[1:]):
            output.append(left)
            a, b = anchors.get(left), anchors.get(right)
            if not a or not b or a[1] >= b[0] or incoming[left] != 1 or incoming[right] != 1 or len(proofs) >= 6:
                continue
            literal = source[a[1]:b[0]].strip()
            if not 8 <= len(literal) <= 240 or added_chars+len(literal) > 1200 or "\n" in literal or not 2 <= len(norm(literal).split()) <= 40:
                continue
            if re.match(r"^(?:#|[-*•]\s|\d+[.)]\s|\*\*|__|Media\s*:|Action\s*:|Sources?\s*:|Formula\s*:|Chart(?: Title)?\s*:|[XY]-Axis\s*:)", literal, re.I):
                continue
            if any(char in literal for char in ("|", "`", "=", "<", ">")) or re.search(r"\[(?:[A-Z_]+URL|URL|[A-Z_]+ASSET)_\d+\]", literal):
                continue
            if (" " + norm(literal) + " ") in visible:
                continue
            prior = fixed["elements"][left]
            if prior.get("type") == "Text" and prior.get("props", {}).get("variant", "body") in {"body", "body1", "body2"}:
                prior["props"]["text"] += "\n\n" + literal
                inserted = None
            else:
                inserted = "v11_repair_text_" + _sha(pid+"\0"+left+"\0"+right+"\0"+literal)[:16]
                if inserted in fixed["elements"]:
                    continue
                fixed["elements"][inserted] = {"type": "Text", "props": {"text": literal}, "children": []}
                output.append(inserted)
            proofs.append({"kind": "adjacent_unique_source_prose_restoration", "parent": pid,
                           "left": left, "right": right, "left_source_span": list(a), "right_source_span": list(b),
                           "source_gap_span": [a[1], b[0]], "source_literal": literal,
                           "left_anchor_literal": source[a[0]:a[1]],
                           "right_anchor_literal": source[b[0]:b[1]],
                           "left_anchor_sha256": _sha(source[a[0]:a[1]]),
                           "right_anchor_sha256": _sha(source[b[0]:b[1]]), "inserted_element": inserted})
            added_chars += len(literal)
            visible += " " + norm(literal) + " "
        if children:
            output.append(children[-1])
            parent["children"] = output
    return fixed, proofs


def _round(source, graph, mapping, original_sha, iteration):
    from ir_training.data.archive_recovery import _prune_ungrounded_references, _join_midword_text_chunks
    from ir_training.data.archive_final_review import repair_final_text
    proofs, changes = [], Counter()

    def record(stage, prior_source, prior_graph, prior_map, new_source, new_graph, new_map, metrics, policy_proofs=()):
        differences = _diff({"source": prior_source, "graph": prior_graph, "url_map": prior_map},
                            {"source": new_source, "graph": new_graph, "url_map": new_map})
        if differences:
            positive = {name: count for name, count in metrics.items() if count}
            if not positive:
                positive = {stage: 1}
            changes.update(positive)
            for difference in differences:
                proofs.append({"kind": next(iter(positive)) if len(positive) == 1 else stage,
                               "stage": stage, "iteration": iteration, **difference,
                               "repair_kinds": sorted(positive),
                               "original_source_sha256": original_sha,
                               "source_sha256": _sha(prior_source),
                               "result_source_sha256": _sha(new_source),
                               "policy_proofs": deepcopy(list(policy_proofs))})

    fixed, metrics = _safe_rebind(source, graph, mapping)
    record("v10_placeholder_rebinding", source, graph, mapping, source, fixed, mapping, metrics)
    graph = fixed
    fixed, metrics, _ = _prune_ungrounded_references(source, graph)
    record("v10_ungrounded_reference_pruning", source, graph, mapping, source, fixed, mapping, metrics)
    graph = fixed
    fixed, count = _join_midword_text_chunks(graph)
    record("v10_midword_chunk_join", source, graph, mapping, source, fixed, mapping,
           {"joined_midword_text_chunks": count})
    graph = fixed
    fixed, metrics = repair_final_text(source, graph)
    record("v10_final_text_repair", source, graph, mapping, source, fixed, mapping, metrics)
    graph = fixed
    fixed, literal_proofs = _restore_small_literal_expansions(source, graph)
    record("unique_short_source_literal_expansion", source, graph, mapping, source, fixed, mapping,
           Counter(item["kind"] for item in literal_proofs), literal_proofs)
    graph = fixed
    fixed, gap_proofs = _restore_anchored_prose_gaps(source, graph)
    record("adjacent_unique_source_prose_restoration", source, graph, mapping, source, fixed, mapping,
           Counter(item["kind"] for item in gap_proofs), gap_proofs)
    graph = fixed
    new_source, fixed, new_map, semantic_proofs = _semantic_function()[0](source, graph, mapping)
    metrics = Counter(item["kind"] for item in semantic_proofs)
    record("v10_semantic_repair", source, graph, mapping, new_source, fixed, new_map, metrics, semantic_proofs)
    return new_source, fixed, new_map, proofs, changes


def _literal_lists_supported(source, graph):
    from ir_training.data.archive_semantic_review import norm
    source_plain = norm(source)
    found = False
    for element in graph["elements"].values():
        if element.get("type") != "List":
            continue
        items = element.get("props", {}).get("items")
        if not isinstance(items, list) or not items:
            continue
        if not all(isinstance(item, str) and norm(item) and f" {norm(item)} " in f" {source_plain} " for item in items):
            return False
        found = True
    return found


def _issues(source, target, checked, mapping, original_sha):
    from ir_training.data.archive_semantic_review import review_graph
    from ir_training.data.archive_refinement import review_warnings
    from ir_training.data.archive_final_review import paragraph_gaps
    from ir_training.data.archive_letter_review import letter_gaps
    from ir_training.data.archive_recovery import placeholder_tokens
    issues = review_graph(source, checked.graph, original_source_sha256=original_sha)
    warnings, resolutions, _ = review_warnings(source, target, checked.graph)
    if "layout_only" in warnings and _literal_lists_supported(source, checked.graph):
        warnings.remove("layout_only")
        resolutions.append("source_supported_literal_list_content")
    for code, evidence in (("legacy_paragraph_gap", paragraph_gaps(source, checked.graph)),
                           ("legacy_letter_gap", letter_gaps(source, checked.graph))):
        if evidence:
            issues.append({"code": code, "detail": evidence})
    if not placeholder_tokens(target) <= placeholder_tokens(source):
        warnings.append("target_reference_absent_from_source")
    issues.extend({"code": code, "detail": "Existing v10 content/closure gate"} for code in sorted(set(warnings)))
    return issues, sorted(set(resolutions))


def repair_chain(source, graph, url_map, original_source_sha256=""):
    """Return source, graph, map, exact proofs, counts, issues and strict check.

    Up to three deterministic passes may converge; a final independent replay
    must make zero edits. Residual source/content/reference defects quarantine.
    """
    from ir_training.data.express_preparation import _api, serialize_checked
    _, express, _, _ = _api()
    original_sha = original_source_sha256 or _sha(source)
    source, graph, mapping = source, deepcopy(graph), deepcopy(url_map or {})
    proofs, changes, checked = [], Counter(), None
    for iteration in range(1, 4):
        source, graph, mapping, evidence, metrics = _round(source, graph, mapping, original_sha, iteration)
        proofs.extend(evidence)
        changes.update(metrics)
        checked = serialize_checked(express.encode(graph, shorten_ids=False), "root-first")
        if checked.graph != graph:
            raise ValueError("repair_graph_changed_during_canonical_serialization")
        if not evidence:
            break
    replay_source, replay_graph, replay_map, replay_proofs, _ = _round(source, checked.graph, mapping, original_sha, 4)
    idempotent = not replay_proofs and (source, graph, mapping) == (replay_source, replay_graph, replay_map)
    issues, resolutions = _issues(source, checked.text, checked, mapping, original_sha)
    if not idempotent:
        issues.append({"code": "repair_chain_did_not_converge", "detail": replay_proofs})
    return {"source": source, "graph": graph, "url_map": mapping, "proofs": proofs,
            "changes": dict(changes), "issues": issues, "checked": checked,
            "idempotent": idempotent, "resolutions": resolutions,
            "implementation": implementation_fingerprint(), "policy": POLICY}


def residual_gates(source, target, checked, url_map=None, original_sha=""):
    """Recheck a finished row, including a clean repair replay, without mutation."""
    issues, _ = _issues(source, target, checked, url_map or {}, original_sha)
    replay_source, replay_graph, replay_map, proofs, changes = _round(source, checked.graph, url_map or {}, original_sha or _sha(source), 1)
    reasons = {issue["code"] for issue in issues}
    if proofs or (source, checked.graph, url_map or {}) != (replay_source, replay_graph, replay_map):
        reasons.add("residual_repair_required")
        reasons.update("residual_repair:" + kind for kind in changes)
    return sorted(reasons)
