"""Independently validate the complete report's population and arithmetic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from statistics import mean


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cases = 0
        self.modes = 0
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article" and attrs.get("class") == "case":
            self.cases += 1
        if tag == "section" and attrs.get("class") == "mode":
            self.modes += 1
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    report = run / "full_native_repaired_report"
    summary = read_json(report / "summary.json")
    sources = rows(run / "model_inputs.jsonl")
    repairs = {(r["model"], r["id"]): r for r in rows(run / "repair_outcomes.jsonl")}
    expected_ids = {f"BXP-{i:03d}" for i in range(1, 51)}
    assert len(sources) == len(repairs) == 100
    assert Counter(r["generation_origin"] for r in sources) == {"prior_verified": 27, "new_run": 73}
    with (report / "per_case_scores.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 100
    csv_index = {(r["model"], r["id"]): r for r in csv_rows}
    assert set(csv_index) == set(repairs)
    verification = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "modes": {}}
    modes = {}
    for mode in ("litert_mtp_off", "litert_mtp_on"):
        data = rows(report / "scored" / f"{mode}.jsonl")
        assert len(data) == 50 and {r["id"] for r in data} == expected_ids
        modes[mode] = {r["id"]: r for r in data}
        for row in data:
            key = mode, row["id"]
            repair = repairs[key]
            assert row["repair_success"] == repair["repair_success"]
            assert row["raw_output"] is not None  # Actual collection produced all 100 raw artifacts.
            assert hashlib.sha256(row["raw_output"].encode()).hexdigest() == row["raw_output_sha256"]
            assert row["raw_output"] == Path(row["raw_input_file"]).read_text(encoding="utf-8")
            if row["repair_success"]:
                assert row["repaired_score"] is not None and row["repaired_strict_valid"] is True
                assert row["source_contract_equal"] is True
                assert row["repaired_express"] == repair["repaired_express"]
                assert hashlib.sha256(row["repaired_express"].encode()).hexdigest() == row["repaired_express_sha256"]
            else:
                assert row["repaired_score"] is None and row["repaired_express"] is None
                assert row["repaired_score_for_aggregate"] == 0
            assert float(csv_index[key]["repaired_score_for_aggregate"]) == row["repaired_score_for_aggregate"]
        result = {
            "n": len(data),
            "raw_mean": mean(r["raw_score_for_aggregate"] for r in data),
            "repaired_mean": mean(r["repaired_score_for_aggregate"] for r in data),
            "repair_accepted": sum(r["repair_success"] for r in data),
            "failures": [r["id"] for r in data if not r["repair_success"]],
        }
        assert math.isclose(result["raw_mean"], summary["modes"][mode]["raw_mean_failures_zero"], abs_tol=1e-12)
        assert math.isclose(result["repaired_mean"], summary["modes"][mode]["repaired_mean_failures_zero"], abs_tol=1e-12)
        assert result["repair_accepted"] == summary["modes"][mode]["repair_accepted"]
        verification["modes"][mode] = result
    for case_id in expected_ids:
        off, on = (modes[m][case_id] for m in modes)
        assert off["serial"] == on["serial"] and off["response_text"] == on["response_text"]
    parity = read_json(run / "repair_device_parity.json")
    assert parity and all(r["matches"] for r in parity)
    page = Page()
    page.feed((report / "index.html").read_text(encoding="utf-8"))
    assert page.cases == 50 and page.modes == 100
    assert all((report / link).is_file() for link in page.links if ":" not in link and not link.startswith("#"))
    for path, expected_hash in read_json(report / "scoring_input_hashes.json").items():
        assert digest(Path(path)) == expected_hash, f"Changed report input: {path}"
    verification.update({"status": "PASS", "html_cases": page.cases, "html_mode_panels": page.modes,
                         "csv_rows": len(csv_rows), "device_repair_parity_passed": len(parity),
                         "source_files_unchanged": True})
    (report / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
