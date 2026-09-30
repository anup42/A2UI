"""Read-only v11s audit; requires explicit inputs and a new output report."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset/src"))
sys.path.insert(0, str(ROOT / "training/src"))
from pipeline.ir_formats.active import decode_express_completion
from pipeline.ir_formats.canonical import semantic_hash
from ir_training.data.express_preparation import prepare_row


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_stream(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-dir", required=True, type=Path, help="Restored v11s folder with train.jsonl and val.jsonl")
parser.add_argument("--decision-ledger", type=Path,
    default=ROOT / "training/reports/dataset_v11_20260930/new_decisions.jsonl.gz",
    help="Original new-row decisions JSONL or gzip ledger")
parser.add_argument("--output-report", required=True, type=Path, help="New report path; existing paths are refused")
args = parser.parse_args()
args.dataset_dir = args.dataset_dir.resolve()
args.output_report = args.output_report.resolve()
if args.output_report.exists():
    parser.error("--output-report must name a new file; existing reports are never overwritten")
if args.output_report.is_relative_to(args.dataset_dir):
    parser.error("--output-report must be outside the immutable dataset directory")
if not args.decision_ledger.is_file() or any(not (args.dataset_dir / f"{split}.jsonl").is_file() for split in ("train", "val")):
    parser.error("Expected train.jsonl, val.jsonl, and the supplied decision ledger")

capability_path = ROOT / "dataset/schema/renderer_capabilities.json"
subtypes = json.loads(capability_path.read_text(encoding="utf-8"))["chart_subtypes"]
supported = set(subtypes["canonical"])
aliases = subtypes.get("aliases", {})


def display(text):
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", text)
    return text


joins = {}
for line in text_stream(args.decision_ledger):
    decision = json.loads(line)
    if decision["outcome"] != "REPAIR" or "joined_midword_text_chunks" not in decision.get("repair_changes", {}):
        continue
    joins[(decision["run"], decision["ui_id"])] = [proof for proof in decision["repair_proofs"]
        if proof.get("location", "").endswith("/props/text")
        and "joined_midword_text_chunks" in proof.get("repair_kinds", [])
        and isinstance(proof.get("before"), str) and isinstance(proof.get("after"), str)
        and proof["after"].startswith(proof["before"])]

join_rows = []
chart_rows = []
types = Counter()
states = Counter()
prep_samples = []
rows_checked = 0
for split in ("train", "val"):
    with (args.dataset_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows_checked += 1
            imported = row["metadata"]["space_import"]
            key = (imported["run"], row["id"].split(":")[1])
            is_join = key in joins
            chart_candidate = "Chart(" in row["completion"]
            if not is_join and not chart_candidate:
                continue
            graph = decode_express_completion(row["completion"])
            identity = {"id": row["id"], "source_id": row["source_id"], "split": split,
                "source_sha256": sha(row["response_text"]), "target_sha256": sha(row["completion"]), "semantic_sha256": semantic_hash(graph),
                "lineage": {"run": imported["run"], "ui_id": key[1], "archive_record": imported["archive_record"], "original_source_sha256": imported["original_source_sha256"], "original_target_sha256": imported["original_target_sha256"]}}
            if is_join:
                boundaries = []
                for proof in joins[key]:
                    left, after = proof["before"], proof["after"]
                    right = after[len(left):]
                    raw_edge = left[-40:] + right[:40]
                    raw_matches = list(re.finditer(re.escape(left[-40:]) + r"(?P<separator>\s+)" + re.escape(right[:40]), row["response_text"]))
                    plain_left, plain_right, plain_source = display(left), display(right), display(row["response_text"])
                    plain_matches = list(re.finditer(re.escape(plain_left[-30:]) + r"(?P<separator>\s+)" + re.escape(plain_right[:30]), plain_source))
                    match = raw_matches[0] if len(raw_matches) == 1 else plain_matches[0] if len(plain_matches) == 1 else None
                    if raw_edge in row["response_text"]:
                        status = "source_has_exact_continuation"
                    elif len(raw_matches) == 1:
                        status = "confirmed_raw_source_separator_deleted"
                    elif len(plain_matches) == 1:
                        status = "confirmed_display_syntax_source_separator_deleted"
                    else:
                        status = "no_exact_source_boundary_match"
                    states[status] += 1
                    element_id = proof["location"].split("/")[3]
                    final_text = graph["elements"].get(element_id, {}).get("props", {}).get("text", "")
                    boundaries.append({"location": proof["location"], "left_tail": left[-80:], "right_head": right[:80],
                        "joined_edge": raw_edge, "source_separator": match.group("separator") if match else None,
                        "source_matching_scope": "raw" if len(raw_matches) == 1 else "display_syntax_only" if len(plain_matches) == 1 else "unproven",
                        "joined_boundary_present_in_final_target": raw_edge in final_text, "status": status})
                unsafe = any(boundary["status"].startswith("confirmed_") and boundary["joined_boundary_present_in_final_target"] for boundary in boundaries)
                join_rows.append({**identity, "confirmed_unsafe": unsafe, "boundaries": boundaries})
                if len(prep_samples) < 2 and unsafe:
                    try:
                        prepare_row(row, "root-first")
                        prep = "accepted"
                    except Exception as exc:
                        prep = f"{type(exc).__name__}: {exc}"
                    prep_samples.append({"id": row["id"], "preparation_result": prep})
            if chart_candidate:
                bad = []
                for element_id, element in graph["elements"].items():
                    if element.get("type") != "Chart":
                        continue
                    subtype = str(element.get("props", {}).get("chartType") or "bar").strip().casefold()
                    types[subtype] += 1
                    if aliases.get(subtype, subtype) not in supported:
                        bad.append({"element_id": element_id, "chart_type": subtype})
                if bad:
                    chart_rows.append({**identity, "unsupported_charts": bad})

result = {"scope": "Read-only current package scan; original data was not modified; display syntax normalization removes Markdown emphasis/code wrappers and line prefixes only.",
    "inputs_sha256": {split: file_sha(args.dataset_dir / f"{split}.jsonl") for split in ("train", "val")},
    "decision_ledger_sha256": file_sha(args.decision_ledger),
    "renderer_capabilities_sha256": file_sha(capability_path), "supported_chart_subtypes": sorted(supported), "chart_subtype_aliases": aliases,
    "v11s_rows_checked": rows_checked, "join_repaired_rows": len(join_rows), "join_boundary_status_counts": dict(states),
    "confirmed_unsafe_join_rows": sum(row["confirmed_unsafe"] for row in join_rows),
    "join_rows": join_rows, "strict_preparation_samples": prep_samples,
    "chart_subtype_component_counts": dict(types), "unsupported_chart_rows": len(chart_rows), "chart_rows": chart_rows}
path = args.output_report
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as report:
    report.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key not in {"join_rows", "chart_rows"}}, ensure_ascii=True, indent=2))
print(str(path))
