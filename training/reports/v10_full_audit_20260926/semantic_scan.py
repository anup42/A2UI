"""Read-only, source/target candidate scan for the recovered v10 archive.

The flags are retrieval aids, not defect verdicts. Run from the repository root:
    python training/reports/v10_full_audit_20260926/semantic_scan.py
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "training/outputs/datasets/full_data_archive_recovered_v10"
OUT = Path(__file__).resolve().parent / "semantic_candidates.json"

TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)+\s*\|?\s*$")
CITATION = re.compile(r"^\s*[-*]?\s*([^\n<>]+?):\s*<?(\[(?:SOURCE_URL|ACTION_URL|URL)_\d+\])>?\s*$", re.M)
ACTION = re.compile(r"Action:\s*\[Button:\s*([^\]]+)\]\s*<?\[(?:ACTION_URL|SOURCE_URL|URL)_\d+\]>?", re.I)
CHART = re.compile(r"\b(pie|bar|line|scatter|area)\s+chart\b", re.I)
EXPLICIT_CHART = re.compile(r"(?im)(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?(?:chart\s+(?:type|title|configuration|data)\b|(?:stacked\s+)?(?:pie|bar|line|scatter|area)\s+chart\s*:|\(?render\s+as\s+[^\n]*?\bchart\b)")
NUMBER = re.compile(r"(?<![\w])(?:\d{2,}(?:[.,]\d+)*%?|\d+\.\d+%?)(?![\w])")
STOP = set("the a an and or to for in of on with by from your you this that its is are be was were as at per each every into than then can will have has their these those".split())


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("**", "").replace("__", "")
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    return " ".join(re.findall(r"\w+", value))


def table_cells(source: str) -> list[str]:
    lines = source.splitlines()
    result: list[str] = []
    in_table = False
    for i, raw in enumerate(lines):
        if TABLE_DIVIDER.fullmatch(raw):
            in_table = True
            continue
        if in_table and raw.strip().startswith("|") and raw.count("|") >= 2:
            result.extend(part.strip().strip("`*") for part in raw.strip().strip("|").split("|"))
        else:
            in_table = False
    return [cell for cell in result if len(norm(cell)) >= 5 and not re.fullmatch(r"[-\s:]+", cell)]


def source_table_width_mismatches(source: str) -> list[dict]:
    """Source markdown rows with more/fewer cells than their declared header."""
    lines = source.splitlines()
    mismatches = []
    expected = 0
    for index, raw in enumerate(lines):
        if TABLE_DIVIDER.fullmatch(raw) and index > 0:
            expected = len(lines[index - 1].strip().strip("|").split("|"))
            continue
        if not raw.strip() or not raw.strip().startswith("|"):
            expected = 0
            continue
        if expected:
            actual = len(raw.strip().strip("|").split("|"))
            if actual != expected:
                mismatches.append({"line": index + 1, "expected": expected, "actual": actual, "row": raw.strip()[:300]})
    return mismatches


def option_fields(source: str) -> list[str]:
    result = []
    for line in source.splitlines():
        if line.count("|") < 1 or line.lstrip().startswith("|") or line.startswith("Action:"):
            continue
        if len(line) > 400:
            continue
        result.extend(part.strip().strip("`*") for part in line.split("|")[1:])
    return [part for part in result if len(norm(part)) >= 9]


def short_lines(source: str) -> list[str]:
    result = []
    for raw in source.splitlines():
        line = raw.strip()
        if not re.match(r"^(?:[-*]|\d+[.)])\s+", line) or len(line) > 125:
            continue
        if "[URL_" in line or "_URL_" in line or line.startswith("- ["):
            continue
        words = [word for word in norm(line).split() if word not in STOP]
        if 4 <= len(words) <= 18:
            result.append(line)
    return result


def flags(source: str, target: str) -> dict:
    target_norm = norm(target)
    target_words = set(target_norm.split())
    result: dict = {}
    cells = list(dict.fromkeys(table_cells(source)))
    missing_cells = [cell for cell in cells if norm(cell) not in target_norm]
    if cells and missing_cells:
        result["table_cell_exact_gap"] = {"total": len(cells), "missing": missing_cells[:12], "count": len(missing_cells)}

    fields = list(dict.fromkeys(option_fields(source)))
    missing_fields = [field for field in fields if norm(field) not in target_norm]
    if fields and missing_fields:
        result["option_field_exact_gap"] = {"total": len(fields), "missing": missing_fields[:12], "count": len(missing_fields)}

    bullets = list(dict.fromkeys(short_lines(source)))
    missing_bullets = []
    for bullet in bullets:
        words = {w for w in norm(bullet).split() if w not in STOP}
        absent = words - target_words
        if len(absent) >= 2 and len(absent) / len(words) >= 0.4:
            missing_bullets.append({"source": bullet, "absent": sorted(absent)})
    if missing_bullets:
        result["short_bullet_word_gap"] = {"total": len(bullets), "missing": missing_bullets[:8], "count": len(missing_bullets)}

    labels = [m.group(1).strip() for m in CITATION.finditer(source) if "button:" not in m.group(1).lower()]
    missing_labels = [label for label in labels if norm(label) not in target_norm]
    if missing_labels:
        result["citation_label_exact_gap"] = {"total": len(labels), "missing": missing_labels[:12], "count": len(missing_labels)}

    actions = [m.group(1).strip() for m in ACTION.finditer(source)]
    missing_actions = [label for label in actions if norm(label) not in target_norm]
    if missing_actions:
        result["action_label_exact_gap"] = {"total": len(actions), "missing": missing_actions[:12], "count": len(missing_actions)}

    charts = sorted({m.group(1).lower() for m in CHART.finditer(source)})
    if charts and "Chart(" not in target:
        result["chart_component_absent"] = {"kinds": charts}
        if EXPLICIT_CHART.search(source):
            result["explicit_chart_component_absent"] = {"kinds": charts}

    width_mismatches = source_table_width_mismatches(source)
    if width_mismatches:
        result["source_table_width_mismatch"] = {"total": len(width_mismatches), "examples": width_mismatches[:8], "count": len(width_mismatches)}

    source_numbers = set(n.replace(",", "") for n in NUMBER.findall(re.sub(r"\[(?:[A-Z_]+)_\d+\]", "", source)))
    # Exclude numbers inside URL-like literals or icon paths and single-digit day indexes.
    target_numbers = set(n.replace(",", "") for n in NUMBER.findall(target))
    missing_numbers = sorted(source_numbers - target_numbers)
    if missing_numbers:
        result["number_exact_gap"] = {"total": len(source_numbers), "missing": missing_numbers[:20], "count": len(missing_numbers)}
    return result


def main() -> None:
    counts = Counter()
    rows = Counter()
    samples: dict[str, list[dict]] = {}
    for split in ("train", "val"):
        with (DATA / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                record = json.loads(line)
                source = record["messages"][-2]["content"].split("\n\n", 1)[-1]
                target = record["completion"]
                found = flags(source, target)
                rows[split] += 1
                for category, detail in found.items():
                    counts[f"{split}:{category}"] += 1
                    counts[f"all:{category}"] += 1
                    bucket = samples.setdefault(category, [])
                    # Stable bounded evidence; highest missing fraction first, then source row.
                    score = detail.get("count", 1) / max(detail.get("total", 1), 1)
                    item = {"split": split, "line": line_no, "id": record["id"], "score": round(score, 4), "detail": detail}
                    bucket.append(item)
                    if len(bucket) > 160:
                        bucket.sort(key=lambda x: (-x["score"], x["split"], x["line"]))
                        del bucket[100:]
    for bucket in samples.values():
        bucket.sort(key=lambda x: (-x["score"], x["split"], x["line"]))
        del bucket[100:]
    OUT.write_text(json.dumps({"rows": rows, "flagged_rows": counts, "candidate_examples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "flagged_rows": counts}, indent=2))


if __name__ == "__main__":
    main()
