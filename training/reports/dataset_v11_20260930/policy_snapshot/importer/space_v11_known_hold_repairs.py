"""Reproducible, source-proven recovery of reviewed v10 hold families.

Call prepare_space_v11.init(policy_repo) first. repair_known_hold(row, case)
returns (processed_copy, proof_records, unresolved_reasons). The input row is
never changed. Only the audited exact source hashes below are eligible; graph
matching uses source evidence, not sample-specific element IDs. This module
does not fix medical, legal or product facts, or call a generative model.

Case 84 stays held: the requested pie chart is outside the current renderer's
bar/column contract, although the more permissive Express wire schema allows
it. The four arithmetic/unit fixes apply coherent changes to both sides.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import re

from space_v11_repairs import _diff

POLICY = "v11-known-v10-holds-source-proven-20260930"
SOURCE_HASHES = {
    2: "4f5fb1a6be08d71415600497810540e2dfc782accba372745af1abfada06da0a",
    28: "7b0afad90b2b1ccd4451766f30ad5f6c2b18ccc81e3696447b798e947ae5ad8e",
    33: "0f479e9bec9660d9f66fb9894f5b9989674b135f3010de083228431fe8340941",
    46: "c80b85db2bafe4b2ed1e2f5c640b3efe1763626165671ce49966fd8dacd641ac",
    51: "a3ed71a976893c04c55ac862d179f45651d62e790c378a079ba528f608c64592",
    73: "e767023efa939c25adc3bb698a5f1f123eb159a2fbe8bc484558f0506df243d6",
    82: "ffcda2b9e9b573ac487e4532c8237cbab509a1d4538cbb0c83fc1f4db6d8747a",
    84: "0cbb35218933c44b07d0df9cc910feb9920e54cec8c36f75ffac19351768d8a1",
    85: "06f09e307c328686e5c8271df9314a36531a0fb29213ec94af4d60734fbce32e",
    88: "0c57704b2d5dab7608299876f1e9e513d539d7e9610f70e7c5397830d0fcb644",
    100: "b84cb4a8d86e59e6fb39809fa415b5f2105967d7f4423da7b57b223e5613fa51",
}


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def plain(value):
    # Remove only Markdown emphasis that the reviewed source uses.
    return value.replace("**", "").strip()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def unique(values, reason):
    require(len(values) == 1, reason)
    return values[0]


def source_tables(source):
    lines = source.splitlines()
    tables = []
    for index, line in enumerate(lines[:-1]):
        if "|" not in line or not re.fullmatch(r"[\s|:–-]+", lines[index + 1]):
            continue
        headers = [plain(x) for x in line.strip().strip("|").split("|")]
        rows = []
        for raw in lines[index + 2:]:
            if "|" not in raw or not raw.strip():
                break
            cells = [plain(x) for x in raw.strip().strip("|").split("|")]
            if len(cells) != len(headers):
                break
            rows.append(cells)
        if rows:
            tables.append((headers, rows))
    return tables


def graph_tables(graph):
    from ir_training.data.archive_semantic_review import table_rows
    return [(key, element, table_rows(element.get("props", {}), graph.get("state", {})))
            for key, element in graph["elements"].items() if element["type"] == "Table"]


def text_elements(graph):
    return [(key, e["props"]) for key, e in graph["elements"].items()
            if e["type"] == "Text" and isinstance(e.get("props", {}).get("text"), str)]


def fresh_id(graph, prefix):
    index = 1
    while f"{prefix}{index}" in graph["elements"]:
        index += 1
    return f"{prefix}{index}"


def root_children(graph):
    root = graph["elements"][graph["root"]]
    require(root["type"] == "Stack" and isinstance(root.get("children"), list),
            "root_must_be_static_stack")
    return root["children"]


def replace_source_table(graph, element, headers, rows):
    """Project exact source cells into compact rows; preserve unrelated state."""
    props = element["props"]
    # Reusing a state key is safe only when exactly this table consumes it.
    old_path = props.get("statePath")
    require(isinstance(old_path, str) and old_path.startswith("/")
            and old_path.count("/") == 1, "table_pointer_not_simple")
    refs = [e for e in graph["elements"].values() if old_path in
            [e.get("props", {}).get(k) for k in ("statePath", "rowsPath", "dataPath")]]
    require(len(refs) == 1, "table_state_has_other_consumers")
    key = old_path[1:].replace("~1", "/").replace("~0", "~")
    require(key in graph["state"], "table_state_missing")
    columns = [{"key": f"column_{i+1}", "label": h} for i, h in enumerate(headers)]
    projected = [{c["key"]: value for c, value in zip(columns, row)} for row in rows]
    graph["state"][key] = projected
    for prop in ("rows", "rowsPath", "dataPath", "primaryColumn", "highlightColumns", "numericColumns"):
        props.pop(prop, None)
    props.update(columns=columns, preferredPresentation="table")
    return {"headers": headers, "rows": rows, "state_path": old_path}


def restore_transposed_table(source, graph):
    headers, cells = unique(source_tables(source), "source_table_not_unique")
    require(len(headers) == len(cells) == 3, "unexpected_matrix_shape")
    candidates = []
    for key, e, rows in graph_tables(graph):
        if not rows or len(rows) != len(headers):
            continue
        if [r.get("device") for r in rows] != headers:
            continue
        if [[r.get(k) for r in rows] for k in ("accuracy", "battery", "focus")] == cells:
            candidates.append((key, e))
    key, e = unique(candidates, "transposed_table_correspondence_not_unique")
    evidence = replace_source_table(graph, e, headers, cells)
    return {"kind": "source_matrix_orientation_restoration", "element": key, **evidence}


def restore_split_entity_table(source, graph):
    headers, cells = unique(source_tables(source), "source_table_not_unique")
    require(headers == ["Liability", "Taxation", "Setup Complexity"], "source_headers_changed")
    candidates = []
    for key, e, rows in graph_tables(graph):
        if not rows or len(rows) != len(cells):
            continue
        projected = [[str(r.get("entity")) + ": " + str(r.get("liability")),
                      r.get("taxation"), r.get("complexity")] for r in rows]
        if projected == cells:
            candidates.append((key, e))
    key, e = unique(candidates, "split_entity_correspondence_not_unique")
    evidence = replace_source_table(graph, e, headers, cells)
    return {"kind": "source_table_entity_column_restoration", "element": key, **evidence}


def restore_nested_placements(source, graph):
    heading = "### 3. Place Predefined Signing Tags"
    require(source.count(heading) == 1, "source_signing_section_not_unique")
    section = source.split(heading, 1)[1].split("\n### ", 1)[0]
    bullet_lines = [plain(re.sub(r"^\s*-\s*", "", line))
                    for line in section.splitlines() if re.match(r"^\s*-\s+", line)]
    expected = ["Signature: In the main signature block.",
                "Date Signed: Adjacent to the signature.",
                "Initials: At the bottom of each page."]
    require(all(line in bullet_lines for line in expected), "placement_instructions_missing")
    key, props = unique([(k, p) for k, p in text_elements(graph)
                         if "Click Next to enter the document editor." in p["text"]
                         and "Select the Client Signer role" in p["text"]
                         and "repeat the placement for the firm's signatory." in p["text"]],
                        "signing_paragraph_not_unique")
    require(all(x not in props["text"] for x in expected), "placements_already_present")
    props["text"] = "\n".join(bullet_lines)
    return {"kind": "source_nested_bullet_restoration", "element": key,
            "source_section": heading, "exact_source_bullets": bullet_lines}


def restore_option_suffixes(source, graph):
    options = [plain(line) for line in source.splitlines() if re.match(r"^Option \d+: ", line)]
    require(len(options) == 3, "option_count_changed")
    restored = []
    for option in options:
        fields = option.split(" | ")
        require(len(fields) == 3, "option_fields_not_three")
        prefix = " | ".join(fields[:2])
        key, props = unique([(k, p) for k, p in text_elements(graph) if p["text"] == prefix],
                            "truncated_option_not_unique")
        props["text"] = option
        restored.append({"element": key, "exact_source_option": option})
    return {"kind": "complete_source_option_restoration", "options": restored}


def restore_source_attribution(source, graph):
    require(source.count("## Sources\n") == 1, "sources_section_not_unique")
    section = source.split("## Sources\n", 1)[1].split("\n## ", 1)[0]
    entries = re.findall(r"^- ([^\n:]+): <(\[[A-Z_]+_\d+\])>$", section, re.M)
    require(len(entries) == 3, "source_citations_not_complete")
    visible = "\n".join(p["text"] for _, p in text_elements(graph))
    require(all(label not in visible for label, _ in entries), "attribution_already_present")
    from ir_training.data.archive_semantic_review import event_urls
    existing = {a["params"].get("url") for e in graph["elements"].values() for a in event_urls(e)}
    require({url for _, url in entries} <= existing, "source_urls_not_existing_actions")
    children = root_children(graph)
    heading = fresh_id(graph, "restored_sources_heading_")
    graph["elements"][heading] = {"type": "Text", "props": {"text": "Sources", "variant": "h3"}, "children": []}
    item_ids = [heading]
    for label, url in entries:
        key = fresh_id(graph, "restored_source_link_")
        graph["elements"][key] = {"type": "Button", "props": {"label": label, "variant": "borderless"}, "children": [],
                                   "on": {"press": {"action": "openUrl", "params": {"url": url}}}}
        item_ids.append(key)
    container = fresh_id(graph, "restored_sources_")
    graph["elements"][container] = {"type": "Stack", "props": {"direction": "vertical", "gap": "sm"},
                                     "children": item_ids}
    children.append(container)
    return {"kind": "source_attribution_restoration", "citations": entries,
            "note": "Existing source URL tokens retain their exact destinations."}


def remove_unsupported_row_action(source, graph):
    from ir_training.data.archive_semantic_review import event_urls
    actions = dict(re.findall(r"Action: \[Button: ([^\]]+)\] <?(\[[A-Z_]+_\d+\])>?", source))
    require("Book Hotel A" in actions and "Book Hotel B" in actions and "Book Hotel C" not in actions,
            "hotel_source_action_scope_changed")
    candidates = []
    for key, e, rows in graph_tables(graph):
        if rows and [r.get("hotel") for r in rows] == ["Hotel A", "Hotel B", "Hotel C"]:
            candidates.append((key, e, rows))
    key, element, rows = unique(candidates, "hotel_table_not_unique")
    require(not any(c.get("key") in {"bookingUrl", "actionLabel"}
                    for c in element["props"]["columns"]), "action_metadata_is_projected")
    require(all(r.get("bookingUrl") == actions.get(r.get("actionLabel")) for r in rows[:2]),
            "supported_hotel_actions_do_not_match")
    require(rows[2].get("actionLabel") == "Book Hotel C" and
            rows[2].get("bookingUrl") == actions["Book Hotel A"], "unsupported_row_action_shape_changed")
    require(not any(e.get("props", {}).get("label") == "Book Hotel C" and event_urls(e)
                    for e in graph["elements"].values()), "visible_unsupported_hotel_action_also_exists")
    removed = {k: rows[2].pop(k) for k in ("bookingUrl", "actionLabel")}
    return {"kind": "unsupported_latent_row_action_removal", "element": key,
            "row": 2, "source_actions": actions, "removed": removed}


def map_strings(value, function, *, keys=False):
    if isinstance(value, str):
        return function(value)
    if isinstance(value, list):
        return [map_strings(x, function, keys=keys) for x in value]
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            new_key = function(k) if keys else k
            require(new_key not in result, "replacement_key_collision")
            result[new_key] = map_strings(v, function, keys=keys)
        return result
    return value


def replace_both_once(source, graph, before, after):
    require(source.count(before) == 1, "source_exact_replacement_not_unique")
    count = 0
    def replace(value):
        nonlocal count
        count += value.count(before)
        return value.replace(before, after)
    fixed = map_strings(graph, replace)
    require(count == 1, "target_exact_replacement_not_unique")
    return source.replace(before, after), fixed


def repair_arithmetic(case, source, graph):
    if case == 2:
        m = re.search(r"reduce a 5k time from (\d+):(\d+) to (\d+):(\d+)", source)
        require(m is not None, "pace_times_missing")
        a, b, c, d = map(int, m.groups())
        current, target = a * 60 + b, c * 60 + d
        change = (Fraction(current, target) - 1) * 100
        after = f"{float(change):.1f}%"
        require(current == 1500 and target == 1350 and after == "11.1%", "pace_arithmetic_changed")
        source, graph = replace_both_once(source, graph, "10.4%", after)
        return source, graph, {"kind": "exact_speed_percentage_correction",
                               "formula": "(current_seconds / target_seconds - 1) * 100",
                               "inputs": {"current_seconds": current, "target_seconds": target},
                               "exact_fraction": str(change), "rounded_one_decimal": after}
    if case == 73:
        data = re.findall(r"^(Offer [A-Z]) \| \$([\d,]+) \| (\d+)% \| \$([\d,]+) \| \$([\d,]+)$",
                          source, re.M)
        require(len(data) == 3, "compensation_rows_missing")
        totals = {}
        for name, base, pct, bonus, total in data:
            base, pct, bonus, total = [Decimal(x.replace(",", "")) for x in (base, pct, bonus, total)]
            require(base * pct / 100 == bonus and base + bonus == total, "compensation_arithmetic_mismatch")
            totals[name] = int(total)
        winner = max(totals, key=totals.get)
        require(winner == "Offer B" and list(totals.values()).count(totals[winner]) == 1,
                "compensation_winner_not_unique")
        before = "but Offer C has the highest potential total cash when the bonus is included."
        after = f"and {winner} has the highest potential total cash when the bonus is included."
        source, graph = replace_both_once(source, graph, before, after)
        return source, graph, {"kind": "source_proven_total_cash_winner_correction", "totals": totals,
                               "winner": winner, "formula": "base + base * bonus_percent / 100"}
    if case == 100:
        rows = re.findall(r"^Step \d+ \| ([^|]+) \| (\d{4}-\d{2}-\d{2} \d{2}:\d{2})$", source, re.M)
        require(len(rows) == 3, "shipping_timeline_missing")
        dates = [datetime.strptime(t, "%Y-%m-%d %H:%M") for _, t in rows]
        seconds = [(b-a).total_seconds() for a, b in zip(dates, dates[1:])]
        require(seconds == [109800.0, 67500.0] and sum(seconds) == 177300.0,
                "shipping_intervals_changed")
        before = "The longest leg was between the sorting facility and the local hub (approx. 18.75 hours)."
        after = f"The longest interval was from {rows[0][0]} to {rows[1][0]} (30.5 hours)."
        source, graph = replace_both_once(source, graph, before, after)
        return source, graph, {"kind": "exact_timestamp_interval_comparison_correction",
                               "events": rows, "interval_seconds": seconds,
                               "interval_hours": [s / 3600 for s in seconds], "total_hours": sum(seconds) / 3600,
                               "convention": "Same local timestamp convention as the source; no timezone invented."}
    if case == 51:
        data = re.findall(r"^(Employee [A-Z]) \| \$([\d,]+) \| (\d+) \| \$([\d,]+)$", source, re.M)
        require(len(data) == 3 and "Sales | Total revenue generated | USD" in source
                and "Rate | Revenue generated per lead | USD/Lead" in source, "revenue_unit_evidence_missing")
        calculations = []
        for name, sales, leads, rate in data:
            sales, leads, rate = [Decimal(x.replace(",", "")) for x in (sales, leads, rate)]
            require(sales / leads == rate, "revenue_per_lead_arithmetic_mismatch")
            calculations.append({"employee": name, "sales_usd": int(sales), "leads": int(leads),
                                 "revenue_per_lead": int(rate)})
        replacements = [("lead conversion rates", "revenue per lead"),
                        ("Conversion Rate", "Revenue per Lead"),
                        ("conversion rate", "revenue per lead")]
        def rename(text):
            for before, after in replacements:
                text = text.replace(before, after)
            return text
        require(all(before in source for before, _ in replacements), "metric_label_patterns_missing")
        source = rename(source)
        graph = map_strings(graph, rename, keys=True)
        require("conversion" not in (source + json.dumps(graph)).lower(), "old_conversion_metric_remains")
        return source, graph, {"kind": "dimensionally_proven_metric_name_correction",
                               "units": "USD / lead count = USD per lead", "calculations": calculations,
                               "exact_replacements": replacements,
                               "note": "Includes state keys and all Table column/numeric references coherently."}
    raise ValueError("unsupported_arithmetic_case")


TARGET_REPAIRS = {28: restore_transposed_table, 33: restore_nested_placements,
                  46: restore_option_suffixes, 82: restore_split_entity_table,
                  85: restore_source_attribution, 88: remove_unsupported_row_action}


def repair_known_hold(row, case):
    """Return a strictly validated processed copy, exact proofs and open reasons.

    Recovered rows keep id/source_id/assigned_split. Source verification status
    stays needs_review; this certifies only the enumerated conversion/arithmetic
    defects. Unknown or changed evidence fails closed with the original copy.
    """
    from ir_training.data.express_preparation import serialize_checked, _api, TASK_PREFIX, _render_prompt
    from ir_training.data.reference_binding import reference_tokens
    result = deepcopy(row)
    case = int(case)
    previous = result.get("metadata", {}).get("v11_known_hold_repair")
    if previous and previous.get("case") == case and previous.get("policy") == POLICY:
        if previous.get("effective_source_sha256") == sha(result["response_text"]) and previous.get("effective_target_sha256") == sha(result["completion"]):
            return result, [], []
        return result, [], ["known_hold_previous_repair_identity_mismatch"]
    if case not in SOURCE_HASHES:
        return result, [], ["known_hold_requires_source_review"]
    source, target = row["response_text"], row["completion"]
    if sha(source) != SOURCE_HASHES[case]:
        return result, [], ["known_hold_source_hash_not_audited"]
    if case == 84:
        return result, [], ["known_hold_requested_pie_chart_outside_bar_column_renderer_contract"]
    try:
        checked = serialize_checked(target, "root-first")
        require(row["messages"][-2] == {"role": "user", "content": TASK_PREFIX + source}
                and row["messages"][-1] == {"role": "assistant", "content": target},
                "original_source_target_messages_not_bound")
        graph = deepcopy(checked.graph)
        original = {"source": source, "graph": deepcopy(graph)}
        if case in TARGET_REPAIRS:
            evidence = TARGET_REPAIRS[case](source, graph)
        else:
            source, graph, evidence = repair_arithmetic(case, source, graph)
        _, express, _, _ = _api()
        fixed = serialize_checked(express.encode(graph, shorten_ids=False), "root-first")
        require(fixed.graph == graph, "repaired_graph_serializer_changed_semantics")
        require(reference_tokens(fixed.graph) <= reference_tokens(source), "repaired_reference_not_source_closed")
        require(reference_tokens(fixed.graph) == reference_tokens(checked.graph), "repair_changed_reference_inventory")
        changes = _diff(original, {"source": source, "graph": fixed.graph})
        require(bool(changes), "repair_did_not_resolve_defect")
        proof = {"policy": POLICY, "case": case, "kind": evidence["kind"], "source_grounded": True,
                 "source_corrected": source != row["response_text"], "evidence": evidence,
                 "original_source_sha256": sha(row["response_text"]), "original_target_sha256": sha(target),
                 "original_semantic_sha256": checked.semantic_sha256,
                 "effective_source_sha256": sha(source), "effective_target_sha256": sha(fixed.text),
                 "effective_semantic_sha256": fixed.semantic_sha256, "changes": changes,
                 "source_factual_status": "needs_review", "validation": "strict_wire_roundtrip_and_source_reference_closure",
                 "stage3_run": False, "scope": "Deterministic reviewed-data repair; no independent factual certification."}
        result["response_text"] = source
        result["completion"] = fixed.text
        result["messages"][-2]["content"] = TASK_PREFIX + source
        result["messages"][-1]["content"] = fixed.text
        if "prompt" in result:
            result["prompt"] = _render_prompt(result["messages"][:-1])
        if "a2ui_express" in result:
            result["a2ui_express"] = fixed.text
        if "completion_targets" in result:
            result["completion_targets"] = {"a2ui_express_v1": fixed.text}
        if "canonical_graph" in result:
            result["canonical_graph"] = deepcopy(fixed.graph)
        if "semantic_hash" in result:
            result["semantic_hash"] = fixed.semantic_sha256
        meta = result.setdefault("metadata", {})
        meta["v11_known_hold_repair"] = deepcopy(proof)
        # Archive original identities remain untouched; effective identities
        # describe the new processed pair, as in the original v10 repair chain.
        recovery = meta.get("archive_recovery", {})
        recovery.update(effective_source_sha256=sha(source), effective_target_sha256=sha(fixed.text),
                        effective_semantic_sha256=fixed.semantic_sha256)
        recovery.setdefault("transformations", []).append(evidence["kind"])
        repair = result.setdefault("repair", {})
        repair["applied"] = True
        repair.setdefault("changes", []).append({"kind": evidence["kind"], "lossless": False, "source_grounded": True})
        return result, [proof], []
    except ValueError as exc:
        return deepcopy(row), [], ["known_hold_repair_unproven:" + str(exc)]
