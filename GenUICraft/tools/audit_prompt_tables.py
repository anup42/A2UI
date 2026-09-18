"""Audit saved Bixby50 table content and presentation choices; never calls a model/device.

This is an output-selection audit, not an A2UI schema validator or a visual-quality proof.
Missing outputs remain failures, including for source cases with no tables.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTATIONS = REPO / "experiments/gemma_prompt_study_20260918/table_expectations.json"
LITERAL_PREFIX = "\x1eGenUICraftLiteral:v1:"
OUTPUT_NAMES = ("revalidated.output.a2ui.json", "output.a2ui.json", "source.output.a2ui.json")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def presentation_text(value: str) -> str:
    """Ignore paired inline emphasis/code markers and whitespace, never facts or citations."""
    for pattern in (r"\*\*([^*\n]+)\*\*", r"(?<!\w)__([^_\n]+)__(?!\w)",
                    r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"(?<!\w)_([^_\n]+)_(?!\w)",
                    r"(?<!`)`([^`\n]+)`(?!`)"):
        value = re.sub(pattern, r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def table_fingerprint(columns: list[str], rows: list[list[str]], *, presentation=False) -> str:
    transform = presentation_text if presentation else lambda value: value
    payload = {"columns": [transform(value) for value in columns],
               "rows": [[transform(value) for value in row] for row in rows]}
    return sha_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8"))


def markdown_cells(line: str) -> list[str] | None:
    text = line.strip()
    cells, cell = [], []
    ticks = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            cell.append(char)
            if index + 1 < len(text):
                index += 1
                cell.append(text[index])
        elif char == "`":
            count = len(text[index:]) - len(text[index:].lstrip("`"))
            ticks = 0 if ticks == count else count if ticks == 0 else ticks
            cell.extend("`" * count)
            index += count - 1
        elif char == "|" and ticks == 0:
            cells.append("".join(cell).strip())
            cell = []
        else:
            cell.append(char)
        index += 1
    if ticks:
        return None
    cells.append("".join(cell).strip())
    if text.startswith("|"):
        cells.pop(0)
    if text.endswith("|") and cells[-1] == "":
        cells.pop()
    return cells


def source_tables(markdown: str) -> list[dict]:
    """Fixture extraction follows SourceBindings' conservative compact-table boundaries."""
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    result = []
    index = 0
    while index < len(lines):
        fence = re.fullmatch(r" {0,3}(`{3,}|~{3,})(.*)", lines[index])
        if fence:
            marker = fence.group(1)
            closing = next((n for n in range(index + 1, len(lines))
                            if len(lines[n]) - len(lines[n].lstrip(" ")) <= 3
                            and len(lines[n].strip()) >= len(marker)
                            and set(lines[n].strip()) == {marker[0]}), None)
            if closing is None:
                break
            index = closing + 1
            continue
        if index + 1 >= len(lines) or "|" not in lines[index]:
            index += 1
            continue
        columns, separators = markdown_cells(lines[index]), markdown_cells(lines[index + 1])
        if not columns or len(columns) < 2 or not separators or len(columns) != len(separators) \
                or any(not re.fullmatch(r":?-{3,}:?", value) for value in separators):
            index += 1
            continue
        rows, end, valid = [], index + 2, True
        while end < len(lines) and lines[end].strip() and "|" in lines[end]:
            row = markdown_cells(lines[end])
            if row is None or len(row) != len(columns):
                valid = False
                break
            rows.append(row)
            end += 1
        if valid and rows:
            result.append({"columns": columns, "rows": rows, "source_line": index + 1})
            index = end
        else:
            index += 1
    return result


def visible_scalar(value) -> str:
    if isinstance(value, str):
        # Decode once; a collision intentionally leaves one literal prefix in the result.
        return value.removeprefix(LITERAL_PREFIX)
    if value is None or isinstance(value, (dict, list)):
        raise ValueError("Table audit requires literal scalar cell values, not paths/expressions/null.")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def output_tables(value) -> tuple[list[dict], list[str]]:
    messages = value if isinstance(value, list) else [value]
    surfaces: dict[str, dict] = {}
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Expected A2UI message objects.")
        update = message.get("updateComponents")
        if update is None:
            continue
        if not isinstance(update, dict) or not isinstance(update.get("components"), list):
            raise ValueError("Malformed updateComponents message.")
        surface = str(update.get("surfaceId", "default_surface"))
        elements = surfaces.setdefault(surface, {})
        for component in update["components"]:
            if not isinstance(component, dict) or not isinstance(component.get("id"), str):
                raise ValueError("Malformed A2UI component.")
            elements[component["id"]] = component
    if not surfaces:
        raise ValueError("No A2UI updateComponents graph found.")
    tables, graph_issues = [], []
    for surface, elements in surfaces.items():
        visited = set()

        def visit(component_id, depth=0):
            if depth > 128:
                graph_issues.append(f"{surface}: graph depth exceeds 128")
                return
            if component_id in visited:
                graph_issues.append(f"{surface}: repeated/cyclic reference {component_id}")
                return
            visited.add(component_id)
            component = elements.get(component_id)
            if component is None:
                graph_issues.append(f"{surface}: missing component {component_id}")
                return
            children = component.get("children", [])
            if not isinstance(children, list) or any(not isinstance(child, str) for child in children):
                graph_issues.append(f"{surface}: unsupported dynamic children on {component_id}")
                return
            for child in children:
                visit(child, depth + 1)

        visit("root")
        for component_id, component in elements.items():
            if component.get("component") != "Table":
                continue
            definitions = component.get("columns")
            raw_rows = component.get("rows")
            if not isinstance(definitions, list) or not definitions or not isinstance(raw_rows, list):
                raise ValueError(f"{component_id}: table must have literal columns and rows.")
            columns, keys = [], []
            for column in definitions:
                if isinstance(column, dict):
                    key = column.get("key", column.get("id"))
                    label = column.get("label", key)
                    if key is None or label is None:
                        raise ValueError(f"{component_id}: column lacks key/label.")
                    keys.append(str(key))
                    columns.append(visible_scalar(label))
                else:
                    keys.append(visible_scalar(column))
                    columns.append(visible_scalar(column))
            rows = []
            for raw_row in raw_rows:
                if isinstance(raw_row, dict):
                    if set(raw_row) != set(keys):
                        raise ValueError(f"{component_id}: row keys differ from column keys.")
                    raw_row = [raw_row[key] for key in keys]
                if not isinstance(raw_row, list) or len(raw_row) != len(columns):
                    raise ValueError(f"{component_id}: non-rectangular or dynamic table row.")
                rows.append([visible_scalar(cell) for cell in raw_row])
            tables.append({"component_id": component_id, "surface_id": surface,
                           "reachable": component_id in visited, "columns": columns, "rows": rows,
                           "domain": component.get("domain"),
                           "presentation": component.get("preferredPresentation"),
                           "fingerprint": table_fingerprint(columns, rows),
                           "content_fingerprint": table_fingerprint(columns, rows, presentation=True)})
    return tables, graph_issues


def choice_review(expected: dict, actual: dict) -> dict:
    domain = actual.get("domain")
    presentation = actual.get("presentation")
    domain = domain.strip().lower() if isinstance(domain, str) else domain
    presentation = presentation.strip().lower() if isinstance(presentation, str) else presentation
    domain_status = ("implicit" if domain is None else "preferred" if domain in expected["preferred_domains"]
                     else "acceptable" if domain in expected["acceptable_domains"] else "unrelated")
    presentation_status = ("implicit" if presentation in (None, "auto") else
                           "preferred" if presentation == expected["preferred_presentation"] else
                           "acceptable_alternative" if presentation in expected["acceptable_presentations"] else "review")
    review = domain_status in ("implicit", "unrelated") or presentation_status in ("implicit", "review")
    baseline = expected.get("baseline_observation")
    baseline_choice = {key: baseline.get(key) for key in ("domain", "presentation")} if baseline else None
    return {"domain": domain, "domain_status": domain_status,
            "presentation": presentation, "presentation_status": presentation_status,
            "preferred_presentation": expected["preferred_presentation"],
            "acceptable_presentations": expected["acceptable_presentations"],
            "preferred_domains": expected["preferred_domains"], "acceptable_domains": expected["acceptable_domains"],
            "baseline_choice": baseline_choice,
            "selection_changed_from_baseline": (baseline_choice != {"domain": domain, "presentation": presentation}) if baseline else None,
            "needs_choice_review": review, "rationale": expected["rationale"]}


def changed_cells(expected: dict, actual: dict) -> dict:
    changes = []
    for row_index, (left, right) in enumerate(zip(expected["rows"], actual["rows"])):
        for column, (before, after) in enumerate(zip(left, right)):
            if presentation_text(before) != presentation_text(after) and len(changes) < 12:
                changes.append({"row": row_index, "column": column, "expected": before, "actual": after})
    return {"expected_row_count": len(expected["rows"]), "actual_row_count": len(actual["rows"]),
            "expected_columns": expected["columns"], "actual_columns": actual["columns"],
            "changed_cells_sample": changes}


def audit_case(case_id: str, expected: dict, case_dir: Path, source_report: dict | None) -> dict:
    report = {"id": case_id, "source_status": (source_report or {}).get("status", "unknown"),
              "expected_table_count": len(expected["tables"]), "tables": [], "hard_issues": []}
    path = next((case_dir / name for name in OUTPUT_NAMES if (case_dir / name).is_file()), None)
    if path is None:
        report.update(output_status="missing_output", content_status="not_evaluable")
        report["hard_issues"].append("Missing saved compiled output; case is not skipped.")
        return report
    report.update(output_status="available", output_file=str(path), output_sha256=sha_bytes(path.read_bytes()))
    try:
        actual, graph_issues = output_tables(read_json(path))
    except (ValueError, TypeError, KeyError) as error:
        report.update(output_status="invalid_output", content_status="not_evaluable")
        report["hard_issues"].append(str(error))
        return report
    report["hard_issues"].extend(graph_issues)
    for table in actual:
        if not table["reachable"]:
            report["hard_issues"].append(f"Unreachable table: {table['component_id']}")
    available = {index for index, table in enumerate(actual) if table["reachable"]}
    for table in expected["tables"]:
        matched = next((i for i in sorted(available) if actual[i]["fingerprint"] == table["fingerprint"]), None)
        match_kind = "exact_content"
        if matched is None:
            matched = next((i for i in sorted(available)
                            if actual[i]["content_fingerprint"] == table["content_fingerprint"]), None)
            match_kind = "formatting_equivalent"
        if matched is not None:
            available.remove(matched)
            candidate = actual[matched]
            report["tables"].append({"expectation_id": table["id"], "component_id": candidate["component_id"],
                                     "shape": table["shape"], "content_status": match_kind,
                                     **choice_review(table, candidate)})
            continue
        same_headers = [i for i in sorted(available)
                        if list(map(presentation_text, actual[i]["columns"])) == list(map(presentation_text, table["columns"]))]
        same_rows = [i for i in sorted(available)
                     if [[presentation_text(v) for v in r] for r in actual[i]["rows"]] ==
                     [[presentation_text(v) for v in r] for r in table["rows"]]]
        candidates = same_headers or same_rows
        if len(candidates) == 1:
            matched = candidates[0]
            available.remove(matched)
            report["tables"].append({"expectation_id": table["id"], "component_id": actual[matched]["component_id"],
                                     "shape": table["shape"], "content_status": "altered_table",
                                     **changed_cells(table, actual[matched])})
            report["hard_issues"].append(f"Altered table content: {table['id']}")
        else:
            report["tables"].append({"expectation_id": table["id"], "shape": table["shape"],
                                     "content_status": "missing_table", "columns": table["columns"]})
            report["hard_issues"].append(f"Missing source table: {table['id']}")
    report["unexpected_tables"] = [{"component_id": actual[i]["component_id"], "columns": actual[i]["columns"],
                                     "fingerprint": actual[i]["fingerprint"]} for i in sorted(available)]
    if available:
        report["hard_issues"].append("Unexpected or duplicated table content.")
    report["content_status"] = "content_issue" if report["hard_issues"] else "preserved"
    report["needs_choice_review"] = any(t.get("needs_choice_review", False) for t in report["tables"])
    return report


def audit_run(run: Path, expectations: dict, selected: list[str] | None = None) -> dict:
    expected = expectations["cases"]
    config = {}
    for name in ("experiment.json", "run_config.json", "replay_config.json"):
        path = run / name
        if path.is_file():
            config.update(read_json(path))
    planned = selected if selected is not None else config.get("cases", list(expected))
    if not isinstance(planned, list) or not planned or len(planned) != len(set(planned)) or any(i not in expected for i in planned):
        raise ValueError(f"{run}: unknown, duplicated, or empty planned case IDs.")
    records = []
    for name in ("results.json", "replay_results.json"):
        if (run / name).is_file():
            records = read_json(run / name)
            break
    if not isinstance(records, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in records):
        raise ValueError(f"{run}: result records must be objects with string case IDs.")
    observed_ids = [row.get("id") for row in records]
    duplicate_ids = sorted(key for key, count in Counter(observed_ids).items() if count > 1)
    by_id = {row.get("id"): row for row in records}
    cases = [audit_case(case, expected[case], run / case, by_id.get(case)) for case in planned]
    statuses = Counter(table["content_status"] for case in cases for table in case["tables"])
    presentations = Counter(table.get("presentation_status", "not_evaluable") for case in cases for table in case["tables"])
    domains = Counter(table.get("domain_status", "not_evaluable") for case in cases for table in case["tables"])
    missing_results = [case for case in planned if case not in by_id]
    unexpected_results = sorted(set(observed_ids) - (set(planned) if selected is None else set(expected)))
    non_success = [{"id": case, "status": by_id[case].get("status"), "message": by_id[case].get("message")}
                   for case in planned if case in by_id and by_id[case].get("status") not in ("success", "rendered")]
    return {"run": str(run), "audit_kind": "saved_table_selection_audit", "model_calls": 0,
            "planned_cases": planned, "case_count": len(cases), "missing_result_ids": missing_results,
            "duplicate_result_ids": duplicate_ids, "unexpected_result_ids": unexpected_results,
            "source_non_successes": non_success,
            "summary": {"content_issue_cases": sum(bool(case["hard_issues"]) for case in cases),
                        "expected_tables": sum(len(expected[case]["tables"]) for case in planned),
                        "cases_without_source_tables": sum(not expected[case]["tables"] for case in planned),
                        "missing_output_cases": sum(case["output_status"] == "missing_output" for case in cases),
                        "choice_review_cases": sum(case.get("needs_choice_review", False) for case in cases),
                        "table_content": dict(statuses), "presentation_choices": dict(presentations),
                        "domain_choices": dict(domains)},
            "cases": cases}


def self_test():
    columns, rows = ["Model", "Value"], [["**One**", "31°C"]]
    expected_table = {"id": "T1", "columns": columns, "rows": rows, "shape": "entity_rows",
                      "fingerprint": table_fingerprint(columns, rows),
                      "content_fingerprint": table_fingerprint(columns, rows, presentation=True),
                      "preferred_domains": ["comparison"], "acceptable_domains": ["comparison", "generic"],
                      "preferred_presentation": "cards", "acceptable_presentations": ["cards", "table"],
                      "rationale": "test"}
    expected = {"tables": [expected_table]}
    messages = [{"updateComponents": {"surfaceId": "s", "components": [
        {"id": "root", "component": "Stack", "children": ["different_id"]},
        {"id": "different_id", "component": "Table", "columns": columns, "rows": rows,
         "domain": "generic", "preferredPresentation": "table"}]}}]
    with tempfile.TemporaryDirectory(prefix="genuicraft_table_audit_") as directory:
        case = Path(directory)
        def check_output(value):
            (case / "output.a2ui.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            return audit_case("BXP-001", expected, case, {"status": "success"})
        result = check_output(messages)
        assert result["content_status"] == "preserved" and not result["needs_choice_review"]
        messages[0]["updateComponents"]["components"][1]["rows"] = [["One", "31°C"]]
        assert check_output(messages)["tables"][0]["content_status"] == "formatting_equivalent"
        messages[0]["updateComponents"]["components"][1]["rows"] = [["One", "32°C"]]
        assert check_output(messages)["tables"][0]["content_status"] == "altered_table"
        messages[0]["updateComponents"]["components"][1]["rows"] = rows
        messages[0]["updateComponents"]["components"][1]["domain"] = "flight"
        assert check_output(messages)["needs_choice_review"]
        messages[0]["updateComponents"]["components"][1]["domain"] = "comparison"
        messages[0]["updateComponents"]["components"][1]["preferredPresentation"] = "cards"
        expected_table["acceptable_presentations"] = ["table"]
        expected_table["preferred_presentation"] = "table"
        assert check_output(messages)["tables"][0]["presentation_status"] == "review"
        messages[0]["updateComponents"]["components"][0]["children"] = []
        assert check_output(messages)["content_status"] == "content_issue"
        messages[0]["updateComponents"]["components"] = [{"id": "root", "component": "Text", "text": "No table"}]
        assert check_output(messages)["tables"][0]["content_status"] == "missing_table"
        (case / "output.a2ui.json").unlink()
        assert audit_case("BXP-002", {"tables": []}, case, None)["output_status"] == "missing_output"
    assert markdown_cells(r"| a\|b | `x|y` |") == [r"a\|b", "`x|y`"]
    assert len(source_tables("| A | B |\n|---|---|\n| 1 | 2 |")) == 1
    assert source_tables("```\n| A | B |\n|---|---|\n| 1 | 2 |\n```") == []
    assert visible_scalar(LITERAL_PREFIX + LITERAL_PREFIX + "x") == LITERAL_PREFIX + "x"
    print("Self-test passed: identity-independent matching, formatting, altered facts, domains, unreachable/missing outputs, and parsing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--expectations", type=Path, default=DEFAULT_EXPECTATIONS)
    parser.add_argument("--cases", help="Comma-separated planned case IDs; otherwise use run config, then all 50.")
    parser.add_argument("--output", type=Path, help="Write complete JSON report; stdout is used when omitted.")
    parser.add_argument("--strict-choices", action="store_true", help="Also exit nonzero for choices requiring review.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if not args.runs:
            return 0
    if not args.runs:
        parser.error("Supply at least one saved run directory, or --self-test.")
    expectations = read_json(args.expectations)
    for case in expectations["cases"].values():
        for table in case["tables"]:
            if table_fingerprint(table["columns"], table["rows"]) != table["fingerprint"] or \
                    table_fingerprint(table["columns"], table["rows"], presentation=True) != table["content_fingerprint"]:
                raise ValueError(f"Expectation content hash mismatch: {table['id']}")
    selected = [value.strip() for value in args.cases.split(",") if value.strip()] if args.cases else None
    runs = []
    for run in args.runs:
        try:
            runs.append(audit_run(run, expectations, selected))
        except (OSError, ValueError, TypeError, KeyError) as error:
            runs.append({"run": str(run), "audit_kind": "saved_table_selection_audit", "model_calls": 0,
                         "audit_error": str(error), "summary": {"audit_error": str(error)}})
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "expectations_sha256": sha_bytes(args.expectations.read_bytes()), "runs": runs,
              "limitations": "Audits saved content and selection properties, not actual pixel layout or inference quality."}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({run["run"]: run["summary"] for run in runs}, ensure_ascii=False, indent=2))
    else:
        print(rendered)
    return int(any(run.get("audit_error") or run["summary"]["content_issue_cases"] or run["missing_result_ids"]
                   or run["duplicate_result_ids"] or run["unexpected_result_ids"] or run["source_non_successes"]
                   or (args.strict_choices and run["summary"]["choice_review_cases"])
                   for run in runs))


if __name__ == "__main__":
    sys.exit(main())
