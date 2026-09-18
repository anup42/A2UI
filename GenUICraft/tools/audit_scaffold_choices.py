"""Check source-bound scaffold copies, allowing only table presentation selections.

This deliberately supports the simple six-component scaffold dialect, not arbitrary
A2UI Express. It parses syntax with ast; it never evaluates generated code.
"""
import argparse
import ast
import json
from pathlib import Path

MARKER = "Express scaffold (preserve all lines; replace table SELECT_DOMAIN and SELECT_PRESENTATION):\n"
POSITIONAL = {"Column": ["children"], "Text": ["text"], "List": ["items"],
              "Table": ["columns", "rows"], "CodeBlock": ["code", "title"], "Divider": []}


def value(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return node.value
    if isinstance(node, ast.Name):
        return {"ref": node.id}
    if isinstance(node, ast.List):
        return [value(n) for n in node.elts]
    raise ValueError("Unsupported scaffold argument syntax")


def graph(program, *, expected_scaffold=False):
    program = program.strip()
    if not (program.startswith("<a2ui>") and program.endswith("</a2ui>")):
        raise ValueError("Missing complete Express delimiters")
    tree = ast.parse(program[len("<a2ui>"):-len("</a2ui>")].strip())
    result = {}
    for line in tree.body:
        if not isinstance(line, ast.Assign) or len(line.targets) != 1 or not isinstance(line.targets[0], ast.Name):
            raise ValueError("Expected a simple assignment")
        name = line.targets[0].id
        call = line.value
        if name in result or not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id not in POSITIONAL:
            raise ValueError("Duplicate component or unsupported component syntax")
        kind = call.func.id
        positional = POSITIONAL[kind]
        if len(call.args) > len(positional):
            raise ValueError("Too many positional arguments")
        props = {key: value(arg) for key, arg in zip(positional, call.args)}
        for kw in call.keywords:
            if kw.arg is None or kw.arg in props:
                raise ValueError("Repeated property or expanded keyword arguments")
            props[kw.arg] = value(kw.value)
        if kind == "Table":
            if not expected_scaffold:
                if props.get("domain") not in {"weather", "flight", "booking", "schedule", "status", "comparison", "generic"}:
                    raise ValueError("Missing or invalid selected table domain")
                if props.get("preferredPresentation") not in {"cards", "table"}:
                    raise ValueError("Missing or invalid selected table presentation")
            props.pop("domain", None)
            props.pop("preferredPresentation", None)
        if kind == "Column" and isinstance(props.get("children"), list):
            props["children"] = [{"ref": child} if isinstance(child, str) else child for child in props["children"]]
        result[name] = {"type": kind, "props": props}
    return result


def audit(run):
    rows = json.loads((run / "results.json").read_text(encoding="utf-8-sig"))
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8-sig"))
    planned = config["cases"]
    observed = [row["id"] for row in rows]
    missing = sorted(set(planned) - set(observed))
    unexpected = sorted(set(observed) - set(planned))
    duplicates = sorted({case_id for case_id in observed if observed.count(case_id) > 1})
    complete = bool(planned) and len(planned) == len(set(planned)) and not missing and not unexpected and not duplicates and (run / "summary.json").is_file()
    output = []
    for row in rows:
        result = {"id": row["id"], "conversion_status": row["status"]}
        if row["status"] != "success":
            result["scope_status"] = "conversion_failed"
        else:
            case = run / row["id"]
            attempt = row["attempts"]
            try:
                source = (case / f"attempt_{attempt}_input.txt").read_text(encoding="utf-8")
                if MARKER not in source:
                    raise ValueError("Input has no recorded scaffold")
                expected = graph(source.rsplit(MARKER, 1)[1], expected_scaffold=True)
                actual = graph((case / f"attempt_{attempt}.txt").read_text(encoding="utf-8"))
                differences = [key for key in sorted(set(expected) | set(actual)) if expected.get(key) != actual.get(key)]
                result["scope_status"] = "preserved" if not differences else "changed_non_table_selection_fields"
                result["assignment_order_preserved"] = list(expected) == list(actual)
                result["differing_components"] = differences
                if differences:
                    result["details"] = {key: {"expected": expected.get(key), "actual": actual.get(key)} for key in differences}
            except (OSError, ValueError, SyntaxError) as error:
                result["scope_status"] = "unavailable_or_unsupported"
                result["message"] = str(error)
        output.append(result)
    return {"run": run.name, "model_calls": 0, "kind": "saved_scaffold_scope_audit", "cases": output,
            "scope": "semantic component graph and source order; declaration order recorded separately",
            "planned": len(planned), "complete": complete, "missing_ids": missing,
            "duplicate_ids": duplicates, "unexpected_ids": unexpected,
            "preserved": sum(r["scope_status"] == "preserved" for r in output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["complete"] and result["preserved"] == result["planned"] else 1)
