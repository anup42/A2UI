#!/usr/bin/env python3
"""Plot saved inference measurements; never runs inference or changes devices."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = list(csv.DictReader((root / "per_request.csv").open(encoding="utf-8")))
    devices = [("R37L20059SY", "Fold8"), ("R3GL203AKSF", "Flip8")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    colors = ["#7986cb", "#42a5f5", "#1565c0", "#90a4ae", "#ffb74d"]
    labels = ["Engine initialization", "Prefill (derived)", "Decode (derived)",
              "Remaining conversion", "Metrics/close (separate interval)"]
    for ax, (serial, name) in zip(axes, devices):
        selected = [r for r in rows if r["serial"] == serial and r["case_id"] == "BXP-003"
                    and r["arm"] in ("baseline_cold_mtp_on", "controls_warm_mtp_on")]
        if len(selected) != 4:
            raise ValueError(f"Expected four cold/warm train records for {name}")
        parts = []
        for r in selected:
            init, prefill, decode = [float(r[k]) / 1000 for k in
                                    ("init_ms", "prefill_ms_derived", "decode_ms_derived")]
            remaining = float(r["conversion_ms"]) / 1000 - init - prefill - decode
            teardown = 0.0
            if r["arm"] == "baseline_cold_mtp_on":
                folder = root / serial / "runs" / r["arm"] / "device"
                records = json.loads((folder / "teardown.json").read_text(encoding="utf-8"))
                data = next(t for t in records if t["label"] == f"{int(r['case_ordinal']):03d}_BXP-003")
                teardown = (data["finishMetricsMs"] + data["closeAndAwaitMs"]) / 1000
            parts.append([init, prefill, decode, remaining, teardown])
        values = np.array(parts)
        bottom = np.zeros(len(parts))
        x = np.arange(len(parts))
        for i, (color, label) in enumerate(zip(colors, labels)):
            ax.bar(x, values[:, i], bottom=bottom, color=color, label=label, width=.63)
            bottom += values[:, i]
        for i, r in enumerate(selected):
            ax.text(i, bottom[i] + .4, f"{bottom[i]:.2f}s", ha="center", weight="bold")
            ax.text(i, -.14, f"Thermal {r['thermal_start']}→{r['thermal_end']}",
                    transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.set_xticks(x, ["Process first", "Reinitialize", "Warm #1", "Warm #2"])
        ax.set_title(f"{name} · BXP-003 train · MTP on", loc="left", weight="bold")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Seconds")
    fig.suptitle("Engine reuse removes a substantial repeated cost", x=.07, ha="left", fontsize=19, weight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .045), fontsize=10)
    fig.text(.07, .015, "Cold totals sum separately measured conversion and teardown intervals; not tap-to-ready timings. Conditions differ; 428 output tokens in every bar.", fontsize=9)
    fig.tight_layout(rect=(.04, .18, 1, .93))
    fig.savefig(root / "latency_breakdown.png", dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cases = ["BXP-001", "BXP-003", "BXP-031"]
    for ax, (serial, name) in zip(axes, devices):
        for arm, offset, color, label in [("controls_warm_mtp_off", -.16, "#90a4ae", "MTP off"),
                                          ("controls_warm_mtp_on", .16, "#1565c0", "MTP on")]:
            for i, case in enumerate(cases):
                rr = [r for r in rows if r["serial"] == serial and r["arm"] == arm and r["case_id"] == case]
                if len(rr) != 2:
                    raise ValueError(f"Expected two records for {name}/{arm}/{case}")
                rate = sum(float(r["output_tokens"]) for r in rr) / sum(float(r["decode_ms_derived"]) / 1000 for r in rr)
                ax.bar(i + offset, rate, width=.28, color=color, label=label if i == 0 else None)
                ax.scatter([i + offset] * 2, [float(r["decode_tok_s"]) for r in rr],
                           s=24, c="#263238", edgecolors="white", zorder=3)
                ax.text(i + offset, max(float(r["decode_tok_s"]) for r in rr) + 3,
                        f"{rate:.1f}", ha="center", fontsize=10)
        ax.set_title(name, loc="left", weight="bold")
        ax.set_xticks(range(3), ["Weather\nBXP-001", "Train\nBXP-003", "Checklist\nBXP-031"])
        ax.set_ylim(0, 95)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Native decode tokens/second")
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, loc="upper right",
               bbox_to_anchor=(.96, .91), ncol=2)
    fig.suptitle("MTP improves observed warm decode throughput", x=.07, ha="left", fontsize=19, weight="bold")
    fig.text(.07, .035, "Bars: total tokens / total derived decode time; dots: two individual requests. Thermal states and generated outputs differ; not cooled A/B estimates.", fontsize=9)
    fig.tight_layout(rect=(.04, .09, 1, .93))
    fig.savefig(root / "mtp_comparison.png", dpi=170, facecolor="white")
    plt.close(fig)
    print(f"Wrote latency_breakdown.png and mtp_comparison.png to {root}")


if __name__ == "__main__":
    main()
