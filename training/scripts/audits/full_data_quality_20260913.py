"""Deterministic qualitative sample audit; never rewrites source/target data.

This is not a benchmark scorer. Lexical/placeholder warnings are review hints,
not claims that paraphrases are wrong or that a model will achieve a score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.data.express_preparation import TASK_PREFIX, PreparationError, serialize_checked
from ir_training.data.shared_prompt import create_shared_prompt_contract

PLACEHOLDER = re.compile(r"\[(?:IMAGE_URL|ICON_URL|ACTION_URL|SOURCE_URL|MEDIA_URL|URL|IMAGE_ASSET|ICON_ASSET|MEDIA_ASSET)_\d+\]")
WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
NUMBERS = re.compile(r"(?<!\w)\d+(?:[.,:/-]\d+)*%?")
STOP = set("a an and are as at be by for from has have in is it its of on or that the their this to was were will with you your".split())
MOJIBAKE = ("ΓÇ", "â€", "Ã", "\ufffd")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def sample_indices(total: int, count: int) -> set[int]:
    if total < 1 or count < 2:
        raise ValueError("Positive row totals and at least two sample positions are required")
    return {1 + round(i * (total - 1) / (min(total, count) - 1)) for i in range(min(total, count))} if total > 1 else {1}


def scalar_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in scalar_text(item)]
    if isinstance(value, list):
        return [text for item in value for text in scalar_text(item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [str(value)]
    return []


def words(text: str) -> set[str]:
    return {word.casefold() for word in WORDS.findall(PLACEHOLDER.sub("", text)) if len(word) > 2} - STOP


def inspect_row(row: dict, split: str, line: int, offset: int, shared: dict) -> tuple[dict, dict]:
    messages = row.get("messages") or []
    task = messages[-2].get("content", "") if len(messages) > 1 else ""
    response = task[len(TASK_PREFIX):] if task.startswith(TASK_PREFIX) else task
    target = messages[-1].get("content", "") if messages else ""
    scaffold = messages[:-2]
    graph, strict_error = None, None
    try:
        parsed = serialize_checked(target, "root-first")
        graph = parsed.graph
    except (PreparationError, ValueError, TypeError) as exc:
        strict_error = f"{getattr(exc, 'reason', type(exc).__name__)}: {exc}"
    literals = scalar_text(graph) if graph is not None else []
    text = "\n".join(literals) if graph is not None else target
    source_words, target_words = words(response), words(text)
    source_numbers = set(NUMBERS.findall(PLACEHOLDER.sub("", response)))
    target_numbers = set(NUMBERS.findall(PLACEHOLDER.sub("", text)))
    source_placeholders, target_placeholders = set(PLACEHOLDER.findall(response)), set(PLACEHOLDER.findall(target))
    literal_normalized = " ".join(text.casefold().split())
    headings = re.findall(r"(?m)^#{1,6}\s+(.+)$", response)
    action_labels = re.findall(r"\[Button:\s*([^\]]+)\]", response)
    info = {
        "split": split, "line_1based": line, "byte_offset": offset,
        "row_sha256": digest(row), "scaffold_sha256": digest(scaffold),
        "production_scaffold_exact_match": scaffold == shared["scaffold"]["messages"],
        "top_level_fields": sorted(row), "roles": [m.get("role") for m in messages],
        "task_prefix_valid": task.startswith(TASK_PREFIX),
        "response_chars": len(response), "target_chars": len(target),
        "strict_valid": strict_error is None, "strict_error": strict_error,
        "component_counts": dict(Counter(n.get("type") for n in (graph or {}).get("elements", {}).values())),
        "source_mojibake_markers": {m: response.count(m) for m in MOJIBAKE if m in response},
        "target_mojibake_markers": {m: target.count(m) for m in MOJIBAKE if m in target},
        "distinct_word_recall_hint": len(source_words & target_words) / len(source_words) if source_words else None,
        "missing_number_anchors_hint": sorted(source_numbers - target_numbers),
        "source_only_placeholders_hint": sorted(source_placeholders - target_placeholders),
        "target_only_placeholders_hint": sorted(target_placeholders - source_placeholders),
        "source_headings": headings,
        "source_headings_not_literal_in_target_hint": [x for x in headings if " ".join(x.casefold().split()) not in literal_normalized],
        "source_action_labels": action_labels,
        "source_action_labels_not_literal_in_target_hint": [x for x in action_labels if " ".join(x.casefold().split()) not in literal_normalized],
        "generic_open_source_button": bool(re.search(r'Button\("Open Source"\)', target)),
        "response_excerpt": response[:250],
    }
    return info, {"split": split, "line_1based": line, "byte_offset": offset, "response": response, "target": target, "literal_text": text}


def exhaustive_hints(row: dict) -> dict:
    messages = row.get("messages") or []
    task = messages[-2].get("content", "") if len(messages) > 1 else ""
    response = task[len(TASK_PREFIX):] if task.startswith(TASK_PREFIX) else task
    target = messages[-1].get("content", "") if messages else ""
    sw, tw = words(response), words(target)
    sn, tn = set(NUMBERS.findall(PLACEHOLDER.sub("", response))), set(NUMBERS.findall(PLACEHOLDER.sub("", target)))
    sp, tp = set(PLACEHOLDER.findall(response)), set(PLACEHOLDER.findall(target))
    ratio = len(target) / max(1, len(response))
    recall = len(sw & tw) / len(sw) if sw else 1.0
    labels = re.findall(r"\[Button:\s*([^\]]+)\]", response)
    folded = " ".join(target.casefold().split())
    return {
        "source_chars": len(response), "target_chars": len(target), "lexical_recall_hint": recall,
        "source_mojibake": any(m in response for m in MOJIBAKE), "target_mojibake": any(m in target for m in MOJIBAKE),
        "source_replacement_character": "\ufffd" in response, "target_replacement_character": "\ufffd" in target,
        "target_todo_placeholder": bool(re.search(r"\b(?:TODO|TBD|Lorem ipsum)\b", target, re.I)),
        "target_only_placeholder": bool(tp-sp), "source_only_placeholder": bool(sp-tp),
        "source_has_named_actions": bool(labels), "source_action_label_missing": any(" ".join(s.casefold().split()) not in folded for s in labels),
        "generic_open_source_button": bool(re.search(r'Button\("Open Source"\)', target)),
        "lexical_recall_under_half": recall < .5, "lexical_recall_under_three_quarters": recall < .75,
        "numeric_source_with_three_anchors": len(sn) >= 3,
        "numeric_half_missing_with_three_anchors": len(sn) >= 3 and len(sn-tn) / len(sn) >= .5,
        "target_source_char_ratio_bucket": next(label for bound, label in ((.25,"under_0.25"),(.5,"0.25_to_0.5"),(.75,"0.5_to_0.75"),(1,"0.75_to_1"),(2,"1_to_2"),(float("inf"),"2_or_more")) if ratio < bound),
    }


def audit_file(path: Path, split: str, total: int, count: int, shared: dict, evidence_stream, signals_stream) -> tuple[list[dict], dict]:
    wanted, result, observed = sample_indices(total, count), [], 0
    counts, ratios, risky = Counter(), Counter(), []
    writer = None
    with path.open("rb") as probe:
        prefix = probe.read(4)
    encoding = "utf-16" if prefix.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    with path.open("r", encoding=encoding, newline="") as stream:
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            observed += 1
            row = json.loads(raw)
            hints = exhaustive_hints(row)
            if writer is None:
                writer = csv.DictWriter(signals_stream, fieldnames=["split", "line_1based", "byte_offset", *hints])
                if signals_stream.tell() == 0:
                    writer.writeheader()
            writer.writerow({"split": split, "line_1based": observed, "byte_offset": offset, **hints})
            counts.update({key: int(value) for key, value in hints.items() if isinstance(value, bool)})
            counts["source_chars_total"] += hints["source_chars"]
            counts["target_chars_total"] += hints["target_chars"]
            ratios[hints["target_source_char_ratio_bucket"]] += 1
            if observed in wanted:
                info, evidence = inspect_row(row, split, observed, offset, shared)
                info["selection"] = "equally_spaced"
                result.append(info)
                evidence_stream.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            elif hints["source_chars"] >= 500 and hints["lexical_recall_hint"] < .35:
                risky.append((hints["lexical_recall_hint"], observed, offset, row))
                risky.sort(key=lambda item: (item[0], item[1]))
                del risky[6:]
            if observed % 25000 == 0:
                print(json.dumps({"split": split, "rows_scanned": observed}), flush=True)
    if observed != total:
        raise ValueError(f"{split}: expected {total} physical lines; observed {observed}. Recompute deterministic sampling positions.")
    for _, line, offset, row in risky:
        info, evidence = inspect_row(row, split, line, offset, shared)
        info["selection"] = "six_lowest_lexical_recall_nonregular_samples_with_source_at_least_500_chars"
        result.append(info)
        evidence_stream.write(json.dumps(evidence, ensure_ascii=False) + "\n")
    return result, {"encoding": encoding, "physical_rows_scanned": observed, "signals": dict(counts), "target_source_char_ratio_histogram": dict(ratios)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=150292)
    parser.add_argument("--val-rows", type=int, default=910)
    parser.add_argument("--train-samples", type=int, default=40)
    parser.add_argument("--val-samples", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    shared = create_shared_prompt_contract(ordering="root-first")
    reports, exhaustive = [], {}
    evidence = args.evidence_dir / "qualitative_samples.jsonl"
    signals_path = args.evidence_dir / "row_text_signals.csv"
    with evidence.open("w", encoding="utf-8", newline="\n") as stream, signals_path.open("w", encoding="utf-8", newline="") as signals_stream:
        for split, total, count in (("train", args.train_rows, args.train_samples), ("val", args.val_rows, args.val_samples)):
            selected, exhaustive[split] = audit_file(args.source_dir / f"{split}.jsonl", split, total, count, shared, stream, signals_stream)
            reports.extend(selected)
    summary = {
        "scope": "Exhaustive cheap text signals plus deterministic equally spaced samples and six low-lexical-recall cases per split. Heuristics are not a factual quality or benchmark score; sample defect prevalence is not extrapolated.",
        "source_dir": str(args.source_dir.resolve()), "production_prompt_contract_sha256": shared["contract_sha256"],
        "sample_evidence": str(evidence.resolve()),
        "row_text_signals": str(signals_path.resolve()),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["Lexical recall allows neither paraphrase equivalence nor faithful numeric formatting; warnings require review.", "Placeholder role remapping can be legitimate with a restoration map; messages-only rows lack that evidence.", "Strict validation runs only on sampled rows here; use the separate exhaustive IR audit for corpus totals."],
        "splits": {}, "exhaustive_text_signals": exhaustive, "samples": reports,
    }
    for split in ("train", "val"):
        samples = [r for r in reports if r["split"] == split]
        summary["splits"][split] = {
            "sampled_rows": len(samples), "strict_valid_rows": sum(r["strict_valid"] for r in samples),
            "production_scaffold_exact_matches": sum(r["production_scaffold_exact_match"] for r in samples),
            "distinct_scaffolds": len({r["scaffold_sha256"] for r in samples}),
            "target_mojibake_rows": sum(bool(r["target_mojibake_markers"]) for r in samples),
            "source_mojibake_rows": sum(bool(r["source_mojibake_markers"]) for r in samples),
            "missing_source_action_label_rows_hint": sum(bool(r["source_action_labels_not_literal_in_target_hint"]) for r in samples),
            "generic_open_source_button_rows": sum(r["generic_open_source_button"] for r in samples),
            "target_only_placeholder_rows_hint": sum(bool(r["target_only_placeholders_hint"]) for r in samples),
            "word_recall_below_half_rows_hint": sum((r["distinct_word_recall_hint"] or 0) < .5 for r in samples),
        }
    destination = args.output_dir / "sample_quality_summary.json"
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination), "splits": summary["splits"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
