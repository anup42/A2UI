"""Source-only, opt-in coverage sampling; these heuristics are NOT fidelity gates."""
from __future__ import annotations

import heapq
import re
from collections import Counter

from pipeline.genui_quality.grpo_entity_content import extract_grpo_entity_records

PROFILE = "failure_mix_v1"
QUOTAS = {"entity_lists": 300, "comparisons": 200, "long_sections": 150,
          "procedures": 100, "numerical": 100, "references": 100, "controls": 50}
CITATION_TARGET = 60
PILOT_SIZES = (100, 1000)
_BULLET = re.compile(r"^\s*(?:[-*+] |\d+[.)] )(.+)$", re.MULTILINE)
_CODE = re.compile(r"```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)|`[^`\n]*`", re.DOTALL)
_MARKER = re.compile(r"(?<=[\s.!?])((?:\[[1-9]\d?\](?!\()\s*)+)")
_URL = re.compile(r"https?://[^\s<>\]\)]+")


def quotas_for_samples(samples: int) -> dict[str, int]:
    """Keep the original 1,000-source mix or scale it exactly for a 100-source trial."""
    if type(samples) is not int or samples not in PILOT_SIZES:
        raise ValueError("failure_mix_v1 requires 100 or 1000 samples")
    return {key: count * samples // sum(QUOTAS.values()) for key, count in QUOTAS.items()}


def features(text: str) -> dict:
    """Conservative English/Markdown structure hints; never rewrite source text."""
    prose = _CODE.sub("", text)
    lines = prose.splitlines()
    citations = []
    for line in lines:
        if "|" in line or re.match(r"^\s*\[\d+\]", line):
            continue  # Table/form row labels are not claim citations.
        for match in _MARKER.finditer(line):
            before, after = line[:match.start()], line[match.end():]
            if re.search(r":\s*\d+(?:\s+(?:pairs?|sets?|per person|per couple))?\s*$", before, re.IGNORECASE):
                continue  # Packing quantities such as "3 pairs [6]" are not citations.
            if (len(re.findall(r"\b[a-zA-Z]{2,}\b", before)) >= 5
                    and re.fullmatch(r"(?:\s*\[\d+\])*[.,;:]?\s*", after)):
                citations.extend(re.findall(r"\d+", match.group(1)))
    table = sum("|" in line for line in lines) >= 3 or any(len(cells := line.strip().strip("|").split("|")) >= 2
                and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)
                for line in lines)
    bullets = _BULLET.findall(prose)
    # Reuse the bounded source-record parser used by the extraction policy.
    # Counting colons alone incorrectly treats HH:MM or nested template labels
    # as separate entity fields. This reads only source text, never old targets.
    records = extract_grpo_entity_records(
        [line.strip() for line in lines if line.strip()], _URL.findall(prose), source_text=prose)
    entity_lines = len(records)
    headings = sum(bool(re.match(r"^\s*#{1,6}\s+\S", line)) for line in lines)
    procedure = bool(re.search(
        r"\b(?:checklist|ingredients|instructions|procedure|recipe|step[- ]by[- ]step|"
        r"packing|run of show|weekly plan|daily routine|week\s+[1-9]|day\s+[1-9])\b", prose, re.IGNORECASE))
    numeric = len(re.findall(r"(?:[$€£₹]\s*\d|\d\s*(?:%|(?:kg|km|GB|MB|hours?|minutes?|days?)\b)|"
                             r"\b\d{1,2}:\d{2}\b|\b\d{4}-\d{2}-\d{2}\b)", prose, re.IGNORECASE))
    tags = []
    if entity_lines >= 2 and not procedure:
        tags.append("entity_lists")
    if table:
        tags.append("comparisons")
    if headings >= 3 and len(prose.split()) >= 250:
        tags.append("long_sections")
    if procedure:
        tags.append("procedures")
    if numeric >= 3 or (numeric and re.search(r"\b(?:from|up to|approximately|claimed|estimated)\b", prose, re.IGNORECASE)):
        tags.append("numerical")
    # Called on the canonical RESTORED source. Placeholder strings alone cannot
    # satisfy the URL quota; source-quality/binding checks still run afterwards.
    if _URL.search(prose):
        tags.append("references")
    if citations:
        tags.append("numbered_citations")
    if not table and not procedure and numeric < 3 and entity_lines < 2 and len(bullets) <= 2 and len(prose.split()) < 250:
        tags.append("controls")
    return {"tags": tags, "numbered_citation_markers": len(citations),
            "numeric_bracket_patterns": len(re.findall(r"\[\d+\]", text)),
            "words": len(text.split()), "characters": len(text)}


class MixSampler:
    """Bounded deterministic pools; final rows have exactly one primary bucket."""

    def __init__(self, quotas=None, citation_target=CITATION_TARGET):
        self.quotas = dict(QUOTAS if quotas is None else quotas)
        self.size = sum(self.quotas.values())
        self.citation_target = min(citation_target, self.quotas["references"])
        self.pools = {key: [] for key in (*self.quotas, "numbered_citations", "all")}
        self.detected = Counter()

    def observe(self, evidence):
        self.detected.update(evidence["tags"])
        if evidence["numeric_bracket_patterns"]:
            self.detected["numeric_bracket_pattern_rows"] += 1

    def wants(self, priority, evidence):
        return any(len(self.pools[key]) < self.size or priority < -self.pools[key][0][0]
                   for key in (*evidence["tags"], "all"))

    def offer(self, priority, entry, evidence):
        entry = dict(entry, selection_features=evidence)
        for key in (*evidence["tags"], "all"):
            pool = self.pools[key]
            item = (-priority, entry["source_id"], entry)
            if len(pool) < self.size:
                heapq.heappush(pool, item)
            elif priority < -pool[0][0]:
                heapq.heapreplace(pool, item)

    def finish(self):
        selected, used = [], set()
        achieved, replacements = Counter(), Counter()
        pools = {key: [item[2] for item in sorted(pool, reverse=True)]
                 for key, pool in self.pools.items()}
        assigned = {}
        entries = {entry["source_id"]: entry for pool in pools.values() for entry in pool}

        def augment(bucket, visited):
            # A deterministic augmenting path avoids stealing a multi-tag source
            # from a scarce category when another valid assignment exists.
            visited = visited | {bucket}
            for entry in pools[bucket]:
                identifier = entry["source_id"]
                if identifier not in assigned:
                    assigned[identifier] = bucket
                    return True
            for entry in pools[bucket]:
                identifier = entry["source_id"]
                previous = assigned[identifier]
                if previous not in visited and augment(previous, visited):
                    assigned[identifier] = bucket
                    return True
            return False

        cited = sum(augment("numbered_citations", set()) for _ in range(self.citation_target))
        url_count = sum(augment("references", set()) for _ in range(self.quotas["references"] - cited))
        achieved["references"] = cited + url_count
        for bucket in ("controls", "procedures", "entity_lists", "comparisons", "numerical", "long_sections"):
            achieved[bucket] = sum(augment(bucket, set()) for _ in range(self.quotas[bucket]))
        for identifier, bucket in assigned.items():
            selected.append(dict(entries[identifier], selection_bucket=(
                "references" if bucket == "numbered_citations" else bucket), selection_replacement_for=None))
            used.add(identifier)

        def take(pool_name, count, bucket, replacement=None):
            taken = 0
            for entry in pools[pool_name]:
                identifier = entry["source_id"]
                if identifier in used:
                    continue
                if taken >= count:
                    break
                selected.append(dict(entry, selection_bucket=bucket,
                                     selection_replacement_for=replacement))
                used.add(identifier)
                taken += 1
            return taken

        shortfalls = {key: self.quotas[key] - achieved[key] for key in self.quotas}
        # Never pretend a replacement met the missing category. Keep the requested
        # sources, reporting every shortage and the actual replacement features.
        for bucket, count in shortfalls.items():
            replacements[bucket] = take("all", count, "coverage_backfill", bucket)
        if len(selected) != self.size:
            raise ValueError(f"Only {len(selected)} eligible distinct training sources; requested {self.size}")
        selected_cited = sum("numbered_citations" in item["selection_features"]["tags"] for item in selected)
        report = {"profile": PROFILE, "requested": self.quotas, "achieved": dict(achieved),
                  "category_shortfalls": shortfalls, "coverage_backfill": dict(replacements),
                  "selected": len(selected), "detected_before_quality_checks": dict(self.detected),
                  "numbered_citations": {"requested": self.citation_target,
                      "selected_heuristic_candidates": selected_cited,
                      "shortfall": max(0, self.citation_target - selected_cited),
                      "reference_slots_replaced_by_url_sources": max(0, url_count - (self.quotas["references"] - self.citation_target)),
                      "independently_verified": False},
                  "length_diagnostics": {"under_250_words": sum(
                      item["selection_features"]["words"] < 250 for item in selected),
                      "at_least_250_words": sum(item["selection_features"]["words"] >= 250 for item in selected)},
                  "limits": "Source-only heuristics, not factuality or fidelity certification. Detected counts "
                  "follow holdout/dedup exclusions but precede sampled quality gates. Lengths are word-count "
                  "diagnostics, not token budgets or an enforced 50/50 split. No source text or references "
                  "were invented, rewritten or truncated. Review citation candidates manually."}
        return selected, report
