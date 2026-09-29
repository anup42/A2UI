#!/usr/bin/env python3
"""Score saved FP16/FP32 app outputs unchanged, before and after SDK repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
sys.path.insert(0, str(ROOT / "dataset/src"))
from ir_training.eval.metrics import score_prediction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    rows = []
    cache = {}
    for arm in protocol["plan"]:
        for case in protocol["cases"]:
            directory = root / "runs" / arm["label"] / case
            result_file = directory / "result.json"
            if not result_file.is_file():
                continue
            result = json.loads(result_file.read_text(encoding="utf-8"))
            source = json.loads((directory / "source.json").read_text(encoding="utf-8"))["text"]
            row = {"label": arm["label"], "precision": arm["precision"], "mtp": arm["mtp"],
                   "case": case, "runtimeStatus": result["status"], "usedFallback": result["usedFallback"]}
            for kind, name in (("raw", "attempt_1_raw.express"), ("repaired", "repaired.express")):
                path = directory / name
                if not path.is_file():
                    row[kind] = {"available": False, "scoreForAggregate": 0}
                    continue
                raw = path.read_text(encoding="utf-8")
                key = (source, raw)
                if key not in cache:
                    cache[key] = score_prediction(source, None, raw, metric_version="v5_4")
                metrics = cache[key]
                row[kind] = {"available": True, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "scoreForAggregate": metrics["generation_reward_v5_4"], "metrics": metrics}
            rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["precision"], row["mtp"])].append(row)
    summary = [{"precision": precision, "mtp": mtp, "n": len(group),
                "rawMean": statistics.mean(r["raw"]["scoreForAggregate"] for r in group),
                "repairedMean": statistics.mean(r["repaired"]["scoreForAggregate"] for r in group),
                "fallbackCount": sum(r["usedFallback"] is True for r in group)}
               for (precision, mtp), group in sorted(grouped.items())]
    (root / "quality_scores.json").write_text(json.dumps({"summary": summary, "rows": rows},
                                                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# FP32 / corrected FP16 sample quality", "",
             "V5.4 is a source-to-UI representation score (0–100), not factual accuracy or human visual quality. "
             "Scores use the saved raw output and the exact SDK-repaired Express without edits. "
             "Unavailable pipeline artifacts count as zero. These three selected cases do not establish Bixby50 accuracy.", "",
             "| Precision | MTP | Measured outputs | Raw mean | After repair mean | Fallbacks |",
             "|---|---|---:|---:|---:|---:|"]
    lines += [f'| {s["precision"]} | {"On" if s["mtp"] else "Off"} | {s["n"]} | '
              f'{s["rawMean"]:.2f} | {s["repairedMean"]:.2f} | {s["fallbackCount"]} |' for s in summary]
    lines += ["", "| Run | Case | Raw score | After repair score |", "|---|---|---:|---:|"]
    lines += [f'| {r["label"]} | {r["case"]} | {r["raw"]["scoreForAggregate"]:.2f} | '
              f'{r["repaired"]["scoreForAggregate"]:.2f} |' for r in rows]
    (root / "QUALITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
