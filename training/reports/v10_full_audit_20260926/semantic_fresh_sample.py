"""Deterministic fresh 18+2 source/target sample, excluding v10 manual100."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "training/outputs/datasets/full_data_archive_recovered_v10"
PREVIOUS = ROOT / "training/reports/v10_manual100_20260916/samples.jsonl"
OUT = Path(__file__).resolve().parent / "semantic_fresh20.jsonl"
SEED = 2026092601


def main() -> None:
    previous = {json.loads(line)["id"] for line in PREVIOUS.open(encoding="utf-8")}
    selected = []
    counts = {}
    for split, wanted, split_seed in (("train", 18, SEED), ("val", 2, SEED + 1)):
        rng = random.Random(split_seed)
        reservoir = []
        eligible = 0
        total = 0
        with (DATA / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                total += 1
                row = json.loads(line)
                if row["id"] in previous:
                    continue
                eligible += 1
                if len(reservoir) < wanted:
                    slot = len(reservoir)
                else:
                    slot = rng.randrange(eligible)
                if slot >= wanted:
                    continue
                reservoir_item = {
                    "split": split, "line": line_no, "id": row["id"], "source_id": row["source_id"],
                    "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                    "source": row["messages"][-2]["content"].split("\n\n", 1)[-1],
                    "target": row["completion"],
                }
                if len(reservoir) < wanted:
                    reservoir.append(reservoir_item)
                else:
                    reservoir[slot] = reservoir_item
        counts[split] = {"total": total, "excluded_previous100": total - eligible, "eligible": eligible, "selected": wanted, "seed": split_seed}
        selected.extend(reservoir)
    selected.sort(key=lambda row: (row["split"], row["line"]))
    with OUT.open("w", encoding="utf-8") as output:
        for row in selected:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(counts, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
