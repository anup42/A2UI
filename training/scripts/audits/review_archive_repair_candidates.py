"""Print a deterministic, read-only sample of v5 quarantine repair evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "training/src"))
sys.path.insert(0, str(ROOT / "training/scripts"))
from recover_full_data_archive import _init_worker
from ir_training.data.archive_recovery import recover_pair
from ir_training.data.express_preparation import TASK_PREFIX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--per-reason", type=int, default=3)
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    db = sqlite3.connect(args.inventory.resolve().as_uri() + "?mode=ro", uri=True)
    groups = defaultdict(list)
    with (args.candidate_dir / "quarantine.csv").open(encoding="utf-8", newline="") as stream:
        for item in csv.DictReader(stream):
            key = item["warnings"] or item["reason"]
            if len(groups[key]) < args.per_reason:
                groups[key].append(item)
    _init_worker()
    for key, items in sorted(groups.items()):
        if args.reason and key != args.reason:
            continue
        for item in items:
            offset, length = db.execute("SELECT byte_offset,byte_length FROM rows WHERE split=? AND line=?", (item["split"], int(item["line"]))).fetchone()
            with (args.source_dir / f"{item['split']}.jsonl").open("rb") as stream:
                stream.seek(offset)
                row = json.loads(stream.read(length).decode("utf-16-le").removeprefix("\ufeff"))
            source = row["messages"][-2]["content"][len(TASK_PREFIX):]
            result = recover_pair(source, row["messages"][-1]["content"])
            print(json.dumps({"coordinate": f"{item['split']}:{item['line']}", "prior_reason": key, "source": result.source_text, "target": result.target_text, "warnings": result.warnings}, ensure_ascii=True))


if __name__ == "__main__":
    main()
