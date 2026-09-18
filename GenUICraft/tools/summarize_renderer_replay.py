"""Verify a complete Bixby50 JSON renderer replay using saved artifacts only.

Usage: python tools/summarize_renderer_replay.py REPLAY_RUN --source-run SOURCE_RUN

This is a bounded accessibility/capture audit, not OCR, full-cell visibility proof,
or a new inference result. Only byte-preserving replayMode=json is accepted.
SOURCE_RUN may be an original successful generation run or an explicit
staged_revalidated_documents source. Requires the adjacent audit_prompt_tables.py
for conservative source-table parsing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib

from audit_prompt_tables import output_tables, source_tables, table_fingerprint


REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "genuicraft/src/test/resources/genuicraft_bixby50.jsonl"
CORPUS_SHA = "fc46aa381957bed206f9e0f53e28edbb21b5096ca093985ad184fda73faaea0a"
IDS = [f"BXP-{index:03}" for index in range(1, 51)]
PACKAGE = "com.samsung.genuicraft"
LITERAL_PREFIX = "\x1eGenUICraftLiteral:v1:"
BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")
STAGED_SOURCE_KIND = "staged_revalidated_documents"
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(value):
    return json.loads(value, object_pairs_hook=unique_object)


def read_json(path):
    return parse_json(path.read_text(encoding="utf-8-sig"))


def integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, f"Invalid {label}: {value!r}")
    return value


def expected_ids(values, label):
    require(isinstance(values, list) and all(isinstance(v, str) for v in values), f"Invalid {label} IDs")
    counts = Counter(values)
    missing, extra = sorted(set(IDS) - counts.keys()), sorted(counts.keys() - set(IDS))
    duplicate = sorted(key for key, count in counts.items() if count != 1)
    require(not missing and not extra and not duplicate and len(values) == 50,
            f"{label}: missing={missing}, unexpected={extra}, duplicate={duplicate}; expected all 50 exactly once")


def indexed_reports(value, label):
    require(isinstance(value, list) and all(isinstance(row, dict) for row in value), f"Invalid {label}")
    expected_ids([row.get("id") for row in value], label)
    return {row["id"]: row for row in value}


def relative_member(root, value, label):
    require(isinstance(value, str) and value and not Path(value).is_absolute(), f"Invalid {label}")
    root = root.resolve()
    resolved = (root / value).resolve()
    require(resolved == root or root in resolved.parents, f"{label} escapes its declared root")
    return resolved


def validate_staged_report(case_id, report):
    require(isinstance(report, dict) and report.get("id") == case_id,
            f"Invalid staged source report for {case_id}")
    require(report.get("kind") == "captured_model_revalidation",
            f"{case_id} source is not a captured-model revalidation")
    require(report.get("replayMode") == "raw_model", f"{case_id} source is not raw-model revalidation")
    require(report.get("status") == "rendered" and report.get("issues") == [],
            f"{case_id} source did not previously render without errors")
    integer(report.get("capturedProviderCalls"), 1, 1, f"{case_id}.capturedProviderCalls")
    integer(report.get("modelCalls"), 0, 0, f"{case_id}.modelCalls")
    require(report.get("inferenceEvaluated") is False,
            f"{case_id} source must explicitly exclude inference evaluation")
    require(isinstance(report.get("sourceJsonSha256"), str)
            and SHA256.fullmatch(report["sourceJsonSha256"]),
            f"{case_id}.sourceJsonSha256 is invalid")


def staged_source(source):
    manifest_path = source / "source_manifest.json"
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict), "Invalid staged source manifest")
    integer(manifest.get("schemaVersion"), 1, 1, "source manifest schemaVersion")
    require(manifest.get("kind") == STAGED_SOURCE_KIND,
            f"source_manifest.json.kind must be {STAGED_SOURCE_KIND!r}")
    require(manifest.get("stagedRunId") == source.name, "Staged source runId does not match directory")
    require(isinstance(manifest.get("sourceRunId"), str) and manifest["sourceRunId"],
            "Missing staged source provenance runId")
    integer(manifest.get("caseCount"), 50, 50, "source manifest caseCount")
    integer(manifest.get("modelCalls"), 0, 0, "source manifest modelCalls")
    require(manifest.get("inferenceEvaluated") is False and manifest.get("llmInvoked") is False,
            "Staging must explicitly record no LLM invocation or inference evaluation")

    original_relative = manifest.get("originalRunRelativePath")
    require(isinstance(original_relative, str) and original_relative and not Path(original_relative).is_absolute(),
            "Invalid originalRunRelativePath")
    original = (source / original_relative).resolve()
    require(original.is_dir() and original != source, "Missing or invalid original revalidation run")
    require(manifest["sourceRunId"] == original.name, "Original source runId does not match directory")

    require(manifest.get("originalReplayResultsRelativePath") == "replay_results.json"
            and manifest.get("stagedReplayResultsRelativePath") == "replay_results.json",
            "Unexpected replay-results paths in source manifest")
    expected_top_hash = manifest.get("replayResultsSha256")
    require(isinstance(expected_top_hash, str) and SHA256.fullmatch(expected_top_hash),
            "Invalid replayResultsSha256 in source manifest")
    staged_top = (source / "replay_results.json").read_bytes()
    original_top = (original / "replay_results.json").read_bytes()
    require(staged_top == original_top, "Staged replay_results.json is not byte-identical to its original")
    require(sha(staged_top) == expected_top_hash, "Staged replay_results.json SHA-256 mismatch")
    reports = indexed_reports(read_json(source / "replay_results.json"), "staged source replay results")
    require(reports == indexed_reports(read_json(original / "replay_results.json"),
                                       "original source replay results"),
            "Original and staged replay results differ")
    entries = indexed_reports(manifest.get("cases"), "source manifest cases")
    for directory, label in ((source, "staged source"), (original, "original source")):
        expected_ids(sorted(path.name for path in directory.iterdir()
                            if path.is_dir() and path.name.startswith("BXP-")), f"{label} directories")

    for case_id in IDS:
        entry = entries[case_id]
        expected_paths = {
            "originalDocumentRelativePath": f"{case_id}/revalidated.output.a2ui.json",
            "stagedDocumentRelativePath": f"{case_id}/output.a2ui.json",
            "originalReportRelativePath": f"{case_id}/replay_result.json",
            "stagedReportRelativePath": f"{case_id}/replay_result.json",
        }
        for field, expected in expected_paths.items():
            require(entry.get(field) == expected, f"Unexpected {case_id}.{field}")
        original_document = relative_member(original, entry["originalDocumentRelativePath"],
                                            f"{case_id} original document path").read_bytes()
        staged_document = relative_member(source, entry["stagedDocumentRelativePath"],
                                          f"{case_id} staged document path").read_bytes()
        require(original_document == staged_document,
                f"{case_id} staged document is not byte-identical to its original")
        document_hash = sha(staged_document)
        require(entry.get("documentSha256") == document_hash,
                f"{case_id} staged document SHA-256 mismatch")
        original_report_path = relative_member(original, entry["originalReportRelativePath"],
                                               f"{case_id} original report path")
        staged_report_path = relative_member(source, entry["stagedReportRelativePath"],
                                             f"{case_id} staged report path")
        original_report, staged_report = original_report_path.read_bytes(), staged_report_path.read_bytes()
        require(original_report == staged_report,
                f"{case_id} staged report is not byte-identical to its original")
        report_hash = sha(staged_report)
        require(entry.get("sourceReportSha256") == report_hash,
                f"{case_id} staged source-report SHA-256 mismatch")
        report = read_json(staged_report_path)
        require(report == reports[case_id],
                f"{case_id} replay_result.json differs from staged replay_results.json")
        validate_staged_report(case_id, report)
        if "revalidatedJsonSha256" in report:
            require(report["revalidatedJsonSha256"] == document_hash,
                    f"{case_id}.revalidatedJsonSha256 does not match the staged document")
    return reports, manifest, original


def replay_identity(record, source_id, label):
    require(isinstance(record, dict), f"Invalid {label}")
    for key, expected in (("kind", "renderer_replay"), ("replayMode", "json"), ("sourceRunId", source_id)):
        require(record.get(key) == expected, f"{label}.{key} must be {expected!r}")
    integer(record.get("modelCalls"), 0, 0, f"{label}.modelCalls")
    require(record.get("inferenceEvaluated") is False, f"{label} must explicitly exclude inference evaluation")


def comparable(value):
    # Same intentionally limited comparison as the instrumentation. Never read content-desc.
    value = value.removeprefix(LITERAL_PREFIX)
    value = re.sub(r"\[\d+(?:\s*[,–-]\s*\d+)*]", "", value)
    value = re.sub(r"[*_`]", "", value)
    return " ".join(value.split()).lower()


def rect(value):
    match = BOUNDS.fullmatch(value or "")
    return tuple(map(int, match.groups())) if match else None


def intersection(left, right):
    return (max(left[0], right[0]), max(left[1], right[1]),
            min(left[2], right[2]), min(left[3], right[3]))


def png_dimensions(data):
    require(data.startswith(b"\x89PNG\r\n\x1a\n"), "Invalid PNG signature")
    position, dimensions, ended, image_data = 8, None, False, False
    while position < len(data):
        require(position + 12 <= len(data), "Truncated PNG chunk")
        size = struct.unpack_from(">I", data, position)[0]
        require(size <= 64_000_000 and position + size + 12 <= len(data), "Truncated/oversized PNG payload")
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + size]
        checksum = struct.unpack_from(">I", data, position + 8 + size)[0]
        require(zlib.crc32(kind + payload) & 0xffffffff == checksum, "PNG CRC mismatch")
        if position == 8:
            require(kind == b"IHDR" and size == 13, "PNG lacks initial IHDR")
            dimensions = struct.unpack_from(">II", payload)
            require(all(100 < n <= 20_000 for n in dimensions), "Invalid screenshot dimensions")
        if kind == b"IDAT" and size > 0:
            image_data = True
        position += size + 12
        if kind == b"IEND":
            require(size == 0 and position == len(data), "Invalid PNG ending")
            ended = True
            break
    require(ended and dimensions is not None and image_data, "Incomplete PNG")
    return dimensions


def capture_nodes(xml_data, dimensions):
    require(len(xml_data) <= 8_000_000 and b"<!DOCTYPE" not in xml_data.upper(), "Unsafe/oversized hierarchy XML")
    root = ET.fromstring(xml_data)
    require(root.tag == "hierarchy", "Expected UIAutomator hierarchy")
    nodes = list(root.iter("node"))
    require(len(nodes) <= 10_000, "Hierarchy node limit exceeded")
    # GenUiView's outer ScrollView bounds expose its padded content viewport in the dump.
    # Do not guess density/insets from screen resolution or accept a whole-screen rectangle.
    outer_scrolls = []

    def walk(node, in_app_scroll=False):
        is_scroll = node.get("package") == PACKAGE and node.get("class") == "android.widget.ScrollView"
        if is_scroll and not in_app_scroll:
            outer_scrolls.append(node)
        for child in node:
            walk(child, in_app_scroll or is_scroll)

    walk(root)
    require(len(outer_scrolls) == 1, "Cannot establish one outer GenUiView ScrollView viewport")
    viewport = rect(outer_scrolls[0].get("bounds"))
    require(viewport is not None, "Viewport bounds missing")
    require(viewport == intersection(viewport, (0, 0, *dimensions)), "Viewport exceeds paired screenshot bounds")
    require(viewport[2] - viewport[0] > 100 and viewport[3] - viewport[1] > 100, "Invalid replay viewport")
    compose = [node for node in outer_scrolls[0].iter("node")
               if node.get("package") == PACKAGE and node.get("class") == "androidx.compose.ui.platform.ComposeView"
               and rect(node.get("bounds")) == viewport]
    require(len(compose) == 1, "Replay viewport lacks one matching ComposeView")
    visible = []
    for node in nodes:
        if node.get("package") != PACKAGE or node.get("visible-to-user") == "false":
            continue
        bounds = rect(node.get("bounds"))
        if bounds is None:
            continue
        clipped = intersection(bounds, viewport)
        if clipped[2] - clipped[0] > 8 and clipped[3] - clipped[1] > 8:
            text = node.get("text", "")
            visible.append({"text": text, "normalized": comparable(text), "bounds": list(clipped)})
    return list(viewport), visible


def node_fingerprint(nodes):
    return [(node["text"], node["bounds"]) for node in nodes if node["text"].strip()]


def audit_case(case_id, report, source_report, replay, source, config, expected,
               source_kind="generation"):
    result = {"id": case_id, "issues": [], "tables": [], "captures": []}
    try:
        require(report == read_json(replay / case_id / "replay_result.json"), "Per-case result differs from replay_results.json")
        replay_identity(report, source.name, case_id)
        integer(report.get("capturedProviderCalls"), 0, 0, "capturedProviderCalls")
        require(report.get("status") == "rendered" and report.get("issues") == [], "Replay did not render cleanly")
        if source_kind == STAGED_SOURCE_KIND:
            validate_staged_report(case_id, source_report)
            require(source_report == read_json(source / case_id / "replay_result.json"),
                    "Staged per-case source report differs from replay_results.json")
        else:
            require(source_report.get("status") == "success" and source_report.get("usedFallback") is False,
                    "Original generation must be successful with usedFallback=false")
            require(source_report == read_json(source / case_id / "result.json"),
                    "Original per-case result differs from results.json")
        require(report.get("sourceArtifact") == f"{source.name}/{case_id}/output.a2ui.json", "Unexpected sourceArtifact")
        original = (source / case_id / "output.a2ui.json").read_bytes()
        copied = (replay / case_id / "source.output.a2ui.json").read_bytes()
        if source_kind == STAGED_SOURCE_KIND:
            require(original == copied, "Replay source JSON is not byte-identical to its staged source document")
        else:
            require(original == copied, "Replay source JSON is not byte-identical to original generation")
        digest = sha(original)
        require(report.get("sourceJsonSha256") == digest, "Reported source JSON SHA-256 mismatch")
        result["source_json_sha256"] = digest
        actual_tables, graph_issues = output_tables(parse_json(original))
        require(not graph_issues and all(t["reachable"] for t in actual_tables), f"Invalid table graph: {graph_issues}")
        expected_fingerprints = Counter(table_fingerprint(t["columns"], t["rows"], presentation=True) for t in expected)
        require(Counter(t["content_fingerprint"] for t in actual_tables) == expected_fingerprints,
                "Generated tables differ from the pinned corpus (or are missing/duplicated)")
        tables = report.get("tables")
        require(isinstance(tables, list) and len(tables) == len(actual_tables), "Replay table count differs from source JSON")
        actual_by_id = {t["component_id"]: t for t in actual_tables}
        require(len(actual_by_id) == len(actual_tables), "Ambiguous table IDs across surfaces")
        require(all(isinstance(t, dict) for t in tables), "Malformed replay tables")
        require(Counter(t.get("id") for t in tables) == Counter(actual_by_id.keys()), "Missing/duplicate/unexpected replay table IDs")
        captures = report.get("captures")
        require(isinstance(captures, list) and captures and all(isinstance(c, dict) for c in captures), "Missing capture records")
        names = [c.get("name") for c in captures]
        require(all(isinstance(n, str) for n in names) and len(set(names)) == len(names), "Duplicate/invalid capture names")
        require(names[0] == "initial", "First capture must be initial")
        vertical = integer(report.get("verticalSwipes"), 0, config["maxVerticalSwipes"], "verticalSwipes")
        require([n for n in names if n.startswith("vertical_")] == [f"vertical_{i}" for i in range(1, vertical + 1)],
                "Vertical capture sequence differs from swipe count")
        known_names = {"initial"} | {f"vertical_{i}" for i in range(1, vertical + 1)}
        for index, table in enumerate(tables, 1):
            prefix = f"table_{index}_horizontal_"
            horizontal = [n for n in names if n.startswith(prefix)]
            require(len(horizontal) <= config["maxHorizontalSwipesPerTable"], "Horizontal capture limit exceeded")
            require(horizontal == [f"{prefix}{i}" for i in range(1, len(horizontal) + 1)], "Broken horizontal capture sequence")
            reset = f"table_{index}_reset"
            require((reset in names) == bool(horizontal), "Horizontal sequence must have exactly one reset capture")
            require(table.get("horizontalScrollObserved") is bool(horizontal), "Horizontal observation/captures disagree")
            known_names.update(horizontal)
            if horizontal:
                known_names.add(reset)
        require(set(names) == known_names, "Unknown or unbounded capture labels")
        for suffix in ("png", "xml"):
            require({path.stem for path in (replay / case_id).glob(f"*.{suffix}")} == set(names),
                    f"Missing or unreported {suffix} capture artifacts")
        all_nodes, fingerprints = {}, []
        for capture in captures:
            name = capture["name"]
            require(capture.get("screenshot") is True, f"Screenshot failed: {name}")
            png = (replay / case_id / f"{name}.png").read_bytes()
            xml = (replay / case_id / f"{name}.xml").read_bytes()
            viewport, nodes = capture_nodes(xml, png_dimensions(png))
            count = sum(bool(node["normalized"]) for node in nodes)
            integer(capture.get("visibleTextNodes"), 1, 10_000, f"{name}.visibleTextNodes")
            require(count == capture.get("visibleTextNodes"), f"Visible text count mismatch: {name}")
            require(count > 0, f"Empty viewport: {name}")
            require(not any("Unable to render GenUI" in node["text"] for node in nodes), f"Renderer error: {name}")
            all_nodes[name] = nodes
            fingerprints.append(node_fingerprint(nodes))
            result["captures"].append({"name": name, "viewport": viewport, "visible_text_nodes": count,
                                       "png_sha256": sha(png), "xml_sha256": sha(xml)})
        end = report.get("verticalEndObserved")
        limit = report.get("verticalLimitReached")
        require(type(end) is bool and type(limit) is bool, "Missing vertical coverage flags")
        require(limit == (not end and vertical == config["maxVerticalSwipes"]), "Inconsistent vertical coverage flags")
        if end:
            require(vertical > 0 and names[-1] == f"vertical_{vertical}" and len(fingerprints) >= 2
                    and fingerprints[-1] == fingerprints[-2], "End-of-scroll claim lacks consecutive unchanged viewport evidence")
        result.update(vertical_swipes=vertical, vertical_end_observed=end, vertical_limit_reached=limit)
        for table in tables:
            actual = actual_by_id[table["id"]]
            columns = actual["columns"]
            probes = [next((row[i] for row in actual["rows"] if row[i].strip()), "") for i in range(len(columns))]
            require(table.get("columns") == columns, "Reported columns differ from source JSON")
            require(table.get("missingColumns") == [], "Replay reports unobserved columns")
            observed_headers, observed_cells = [], []
            evidence = []
            for index, (column, probe) in enumerate(zip(columns, probes)):
                matches = {}
                for kind, target in (("header", column), ("representative_cell", probe)):
                    normalized = comparable(target)
                    if not normalized:
                        continue
                    match = next((dict(capture=name, text=node["text"], bounds=node["bounds"])
                                  for name, nodes in all_nodes.items() for node in nodes
                                  if normalized in node["normalized"]), None)
                    if match:
                        matches[kind] = match
                        (observed_headers if kind == "header" else observed_cells).append(target)
                require(matches, f"No viewport text evidence for column {index}: {column}")
                evidence.append({"column": column, "evidence": matches})
            require(table.get("observedHeaders") == observed_headers, "Claimed observed headers lack matching saved evidence")
            require(table.get("observedRepresentativeCells") == observed_cells, "Claimed representative cells lack matching saved evidence")
            result["tables"].append({"id": table["id"], "columns": evidence,
                                      "horizontal_scroll_observed": table["horizontalScrollObserved"]})
        require(end and not limit, "Vertical exploration stopped at its bound before an unchanged end viewport was observed")
    except (OSError, ValueError, TypeError, KeyError, ET.ParseError, RecursionError) as error:
        result["issues"].append(str(error))
    result["status"] = "verified" if not result["issues"] else "failed"
    return result


def summarize(replay: Path, source: Path):
    replay, source = replay.resolve(), source.resolve()
    report = {"schema_version": 1, "kind": "host_renderer_replay_verification", "model_calls": 0,
              "inference_evaluated": False, "replay_run": str(replay), "source_run": str(source),
              "expected_cases": 50, "expected_tables": 23, "expected_columns": 100,
              "issues": [], "cases": [], "created_utc": datetime.now(timezone.utc).isoformat(),
              "limitations": [
                  "Visibility means an intersecting accessibility text node (>8px each dimension), not OCR or full text visibility.",
                  "Each column needs a visible header or first nonblank representative cell; this does not prove every table cell is visible.",
                  "Matching follows the instrumentation's case/whitespace/Markdown/citation normalization; repeated values or prose can be ambiguous.",
                  "Vertical end is the instrumentation's unchanged text/bounds heuristic, not an independent proof of all prose coverage.",
                  "Viewport is derived from the saved outer GenUiView ScrollView; screenshot/XML consistency is structural, not pixel-semantic verification.",
              ]}
    try:
        require(replay != source, "Replay and generation directories must differ")
        corpus_bytes = CORPUS.read_bytes()
        require(sha(corpus_bytes) == CORPUS_SHA, "Pinned Bixby50 corpus SHA mismatch")
        corpus = indexed_reports([parse_json(line) for line in corpus_bytes.decode().splitlines() if line.strip()], "corpus")
        expected = {key: source_tables(value["text"]) for key, value in corpus.items()}
        require(sum(map(len, expected.values())) == 23 and sum(len(t["columns"]) for ts in expected.values() for t in ts) == 100,
                "Pinned corpus must contain 23 tables and 100 columns")
        config = read_json(replay / "replay_config.json")
        replay_identity(config, source.name, "replay config")
        require(config.get("runId") == replay.name, "Replay runId does not match directory")
        require(config.get("corpusSha256") == CORPUS_SHA, "Replay corpus SHA mismatch")
        expected_ids(config.get("cases"), "replay config")
        integer(config.get("maxVerticalSwipes"), 0, 30, "maxVerticalSwipes")
        integer(config.get("maxHorizontalSwipesPerTable"), 1, 16, "maxHorizontalSwipesPerTable")
        summary = read_json(replay / "replay_summary.json")
        replay_identity(summary, source.name, "replay summary")
        require(summary.get("runId") == replay.name, "Replay summary runId mismatch")
        for field, value in (("total", 50), ("rendered", 50), ("failed", 0)):
            integer(summary.get(field), value, value, f"summary.{field}")
        reports = indexed_reports(read_json(replay / "replay_results.json"), "replay results")
        manifest_path = source / "source_manifest.json"
        if manifest_path.is_file():
            source_kind = STAGED_SOURCE_KIND
            source_directory_label = "staged source"
            source_reports, manifest, original = staged_source(source)
            report["provenance"] = {
                "corpus_sha256": CORPUS_SHA,
                "source_kind": source_kind,
                "source_manifest_sha256": sha(manifest_path.read_bytes()),
                "source_replay_results_sha256": sha((source / "replay_results.json").read_bytes()),
                "original_revalidation_run": str(original),
                "original_replay_results_sha256": manifest["replayResultsSha256"],
                "replay_config_sha256": sha((replay / "replay_config.json").read_bytes()),
                "replay_results_sha256": sha((replay / "replay_results.json").read_bytes()),
            }
        else:
            source_kind = "generation"
            source_directory_label = "generation"
            source_reports = indexed_reports(read_json(source / "results.json"), "generation results")
            source_config = read_json(source / "run_config.json")
            expected_ids(source_config.get("cases"), "generation config")
            if "corpusSha256" in source_config:
                require(source_config["corpusSha256"] == CORPUS_SHA, "Generation corpus SHA mismatch")
            report["provenance"] = {
                "corpus_sha256": CORPUS_SHA,
                "generation_corpus_hash_recorded": "corpusSha256" in source_config,
                "replay_config_sha256": sha((replay / "replay_config.json").read_bytes()),
                "replay_results_sha256": sha((replay / "replay_results.json").read_bytes()),
                "generation_results_sha256": sha((source / "results.json").read_bytes()),
            }
        for directory, label in ((replay, "replay"), (source, source_directory_label)):
            expected_ids(sorted(p.name for p in directory.iterdir()
                                if p.is_dir() and p.name.startswith("BXP-")), f"{label} directories")
        report["cases"] = [
            audit_case(case_id, reports[case_id], source_reports[case_id], replay, source,
                       config, expected[case_id], source_kind=source_kind)
            for case_id in IDS
        ]
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["issues"].append(str(error))
    cases = report["cases"]
    report["counts"] = {
        "cases_verified": sum(case["status"] == "verified" for case in cases),
        "cases_failed": sum(case["status"] == "failed" for case in cases),
        "source_hashes_verified": sum("source_json_sha256" in case for case in cases),
        "tables_evidenced": sum(len(case["tables"]) for case in cases),
        "columns_evidenced": sum(len(table["columns"]) for case in cases for table in case["tables"]),
        "columns_with_header_evidence": sum("header" in column["evidence"] for case in cases for table in case["tables"] for column in table["columns"]),
        "columns_with_representative_cell_evidence": sum("representative_cell" in column["evidence"] for case in cases for table in case["tables"] for column in table["columns"]),
        "captures_verified": sum(len(case["captures"]) for case in cases),
        "vertical_swipes": sum(case.get("vertical_swipes", 0) for case in cases),
        "vertical_end_observed": sum(case.get("vertical_end_observed", False) for case in cases),
        "vertical_limit_reached": sum(case.get("vertical_limit_reached", False) for case in cases),
        "horizontal_captures": sum("_horizontal_" in capture["name"] for case in cases for capture in case["captures"]),
    }
    counts = report["counts"]
    report["passed"] = (not report["issues"] and counts["cases_verified"] == 50 and counts["source_hashes_verified"] == 50
                        and counts["tables_evidenced"] == 23 and counts["columns_evidenced"] == 100)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_run", type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Write full evidence JSON; otherwise print it to stdout")
    args = parser.parse_args(argv)
    report = summarize(args.replay_run, args.source_run)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        require(not args.output.exists(), "Refusing to overwrite an existing report; choose a new --output path")
        args.output.write_text(serialized, encoding="utf-8")
        print(json.dumps({"passed": report["passed"], "counts": report["counts"], "issues": report["issues"],
                          "failed_cases": [{"id": case["id"], "issues": case["issues"]} for case in report["cases"] if case["issues"]]}, indent=2))
    else:
        print(serialized)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
