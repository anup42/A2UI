"""Validate hand-selected exact anchors against current extracted v10 pairs."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "semantic_confirmed_evidence.json"

# Classification and interpretation are manual; this only proves exact anchors.
ANCHORS = [
    ("train", 60428, "target_instruction_loss", 'Click the "Bell" or "Alert" icon', 'check:"Ensure Price trigger is selected"', "target", 'Click the "Bell" or "Alert" icon'),
    ("train", 64773, "source_arithmetic_and_chart", 'Total quarterly revenue reached **$73,500**', 'category_contribution:[{category:"Electronics",revenue:"$38,000",percentage:"51.7%"}', "target", "Chart("),
    ("train", 69525, "target_chart_omission", "Chart Title: Employee Sales vs. Leads", 'l=Table(columns=[{key:"name",label:"Employee Name"}', "target", "Chart("),
    ("train", 73806, "source_budget_contradiction", 'After meeting all 50/30/20 targets', 'After meeting all 50/30/20 targets', "", ""),
    ("train", 1702, "target_schedule_omission", 'Swimming: 2 sessions | 35 mins | Sustained aerobic laps', 'n=Text("Weekly Activity Schedule","h3")', "target", "Sustained aerobic laps"),
    ("train", 37244, "target_passenger_detail_omission", 'Passenger Name**: Kenji Tanaka', 'r=Text("Passenger Status: Validated"', "target", "Kenji Tanaka"),
    ("train", 14450, "target_listing_detail_omission", 'No pets allowed; strictly non-smoking.', 'Modern compact living near the tech hub | £1,150/mo', "target", "strictly non-smoking"),
    ("train", 4964, "target_itinerary_detail_omission", 'Travel time to ensure on-time arrival.', 'k=Text("Daily Itinerary Breakdown","h2")', "target", "Travel time to ensure on-time arrival"),
    ("train", 40233, "target_chart_and_highlight_omission", 'Mapo: 1400 (Highlight: Green)', 'Lowest Rent: Mapo, Seoul ($1,400)', "target", "Highlight: Green"),
    ("train", 1584, "target_source_attribution_omission", 'Bauhaus Germany: <[ACTION_URL_1]>', 'Button("Shop at Bauhaus"', "target", "Bauhaus Germany"),
    ("train", 2228, "target_source_attribution_omission", 'California Privacy Protection Agency: <[ACTION_URL_2]>', 'Button("CCPA Official Site"', "target", "California Privacy Protection Agency"),
    ("train", 30894, "target_table_field_misbinding", 'Basic Activity Tracking', 'focus:"Basic Activity Tracking"', "", ""),
    ("train", 87790, "unsupported_latent_action", 'Option 2: Hotel B | Budget-friendly alternative', 'actionLabel:"Book Hotel C"', "source", "Book Hotel C"),
]


def main() -> None:
    pairs = {}
    for filename in ("semantic_fresh20.jsonl", "semantic_selected_pairs.jsonl"):
        for line in (HERE / filename).open(encoding="utf-8"):
            row = json.loads(line)
            pairs[(row["split"], row["line"])] = row
    findings = []
    for split, line, kind, source_quote, target_quote, absence_side, absent_quote in ANCHORS:
        row = pairs[(split, line)]
        assert source_quote in row["source"], (split, line, "source", source_quote)
        assert target_quote in row["target"], (split, line, "target", target_quote)
        if absence_side:
            assert absent_quote not in row[absence_side], (split, line, absence_side, absent_quote)
        findings.append({"split": split, "line": line, "id": row["id"], "line_sha256": row["line_sha256"],
                         "kind": kind, "source_quote": source_quote, "target_quote": target_quote,
                         "absent_from": absence_side or None, "absent_quote": absent_quote or None})
    OUT.write_text(json.dumps({"evidence_kind": "Exact anchor assertions on current pairs; semantic verdicts remain manual.",
                               "findings": findings}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validated {len(findings)} anchors: {OUT}")


if __name__ == "__main__":
    main()
