#!/usr/bin/env python3
"""Write a per-case comparison of the two matched Flip8 benchmark runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "device" / "old_mtp_on"
NEW = ROOT / "device" / "r64_mtp_on"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(path: Path) -> tuple[dict, dict[str, dict]]:
    config = read_json(path / "run_config.json")
    results = read_json(path / "results.json")
    assert read_json(path / "status.json")["state"] == "complete"
    assert [row["id"] for row in results] == config["cases"]
    assert len({row["id"] for row in results}) == len(results)
    return config, {row["id"]: row for row in results}


def main() -> None:
    old_config, old = run(OLD)
    new_config, new = run(NEW)
    assert old_config["cases"] == new_config["cases"]
    assert old_config["corpus"]["sha256"] == new_config["corpus"]["sha256"]
    assert old_config["prompt"] == new_config["prompt"]
    assert old_config["runtime"] == new_config["runtime"]
    assert old_config["device"]["fingerprint"] == new_config["device"]["fingerprint"]
    assert old_config["runtime"]["accelerator"] == "GPU"
    assert old_config["runtime"]["mtpEnabled"] is True
    assert old_config["runtime"]["sourceFallbackEnabled"] is False

    rows = []
    for case_id in old_config["cases"]:
        a, b = old[case_id], new[case_id]
        assert a["renderedPromptSha256"] == b["renderedPromptSha256"], case_id
        assert read_json(OLD / case_id / "source.json") == read_json(NEW / case_id / "source.json")
        rows.append({
            "id": case_id,
            "old_status": a["status"],
            "r64_status": b["status"],
            "old_raw_strict": a.get("rawStrictValid", False),
            "r64_raw_strict": b.get("rawStrictValid", False),
            "old_rendered": a.get("renderValid", False),
            "r64_rendered": b.get("renderValid", False),
            "old_repair": a.get("repairKind", ""),
            "r64_repair": b.get("repairKind", ""),
            "old_source_fidelity_warnings": a.get("sourceFidelityWarnings", ""),
            "r64_source_fidelity_warnings": b.get("sourceFidelityWarnings", ""),
            "old_seconds": round(a["elapsedMs"] / 1000, 3),
            "r64_seconds": round(b["elapsedMs"] / 1000, 3),
            "old_output_tokens": a["metrics"].get("outputTokens"),
            "r64_output_tokens": b["metrics"].get("outputTokens"),
            "old_decode_tokens_per_second": a["metrics"].get("decodeTokensPerSecond"),
            "r64_decode_tokens_per_second": b["metrics"].get("decodeTokensPerSecond"),
            "prompt_sha256": a["renderedPromptSha256"],
        })
    with (ROOT / "device_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} matched cases, same prompt, corpus, source, runtime, and device build")


if __name__ == "__main__":
    main()
