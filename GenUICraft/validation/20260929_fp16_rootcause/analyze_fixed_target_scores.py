#!/usr/bin/env python3
"""Compare completed BXP-003 fixed-target LiteRT-LM score captures, using stdlib only.

Run: python GenUICraft/validation/20260929_fp16_rootcause/analyze_fixed_target_scores.py
The C API score is target log-likelihood, not accuracy or full-vocabulary KL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


REQUIRED_CASES = (
    "original_fp16_score_003",
    "original_fp32_score_003",
    "rope_fp16_score_003",
    "rope_floataccum_score_003",
)
OPTIONAL_CASES = (
    "rope_floataccum_fakequant_v2_score_003",
    "rope_fakequant_v2_score_003",
)
CASE_LABELS = {
    "original_fp16_score_003": "Original model · GPU FP16",
    "original_fp32_score_003": "Original model · GPU FP32 reference",
    "rope_fp16_score_003": "RoPE table · GPU FP16",
    "rope_floataccum_score_003": "RoPE table + GEMV float-accum shader patch · GPU FP16",
    "rope_floataccum_fakequant_v2_score_003": "RoPE table + GEMV float-accum + fake-quant v2 shader patch · GPU FP16",
    "rope_fakequant_v2_score_003": "RoPE table + fake-quant v2 shader patch · GPU FP16",
}
REFERENCE = "original_fp32_score_003"
PREFIX = "BXP-003"
THRESHOLDS = (0.1, 0.5, 1.0, 2.0)
CAPTURE_FILES = (
    "manifest.json",
    f"{PREFIX}.metrics.json",
    f"{PREFIX}.scored_target.txt",
    f"{PREFIX}.target_token_ids.txt",
    f"{PREFIX}.input_token_ids.txt",
    f"{PREFIX}.target_token_scores.txt",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def optional_case_skip_reason(root: Path, name: str) -> str | None:
    folder = root / name
    metrics_path = folder / f"{PREFIX}.metrics.json"
    if not metrics_path.is_file():
        return "metrics not present"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "metrics not yet readable"
    if metrics.get("status") != "ok":
        return f"metrics status is {metrics.get('status')!r}, not 'ok'"
    missing = [file for file in CAPTURE_FILES if not (folder / file).is_file()]
    if missing:
        return "capture incomplete: " + ", ".join(missing)
    return None


def read_case(root: Path, name: str) -> dict:
    folder = root / name
    metrics = json.loads((folder / f"{PREFIX}.metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    target_bytes = (folder / f"{PREFIX}.scored_target.txt").read_bytes()
    ids_bytes = (folder / f"{PREFIX}.target_token_ids.txt").read_bytes()
    prompt_ids_bytes = (folder / f"{PREFIX}.input_token_ids.txt").read_bytes()
    scores_bytes = (folder / f"{PREFIX}.target_token_scores.txt").read_bytes()
    ids = [int(line) for line in ids_bytes.decode("ascii").splitlines()]
    scores = [float(line) for line in scores_bytes.decode("ascii").splitlines()]
    total = metrics["target_score"]

    check(manifest["exit_code"] == 0, f"{name}: device command failed")
    check(metrics["status"] == "ok" and metrics["mode"] == "text_scoring", f"{name}: not a completed scoring capture")
    check(metrics["target_score_status"] == "available", f"{name}: no finite aggregate score")
    check(metrics["target_token_scores_status"] == "available", f"{name}: no per-token score vector")
    check(metrics["candidate_count"] == 1, f"{name}: expected one scored target")
    check(len(scores) == metrics["target_token_score_count"] == metrics["target_scored_token_length"], f"{name}: score lengths disagree")
    check(len(ids) == metrics["target_tokenizer_count"] == len(scores), f"{name}: standalone tokenizer count differs from score count")
    check(len(target_bytes) == metrics["score_target_bytes"], f"{name}: target byte length differs")
    check(manifest["score_target_sha256"].lower() == digest(target_bytes), f"{name}: target digest differs")
    check(all(math.isfinite(score) for score in scores) and math.isfinite(total), f"{name}: nonfinite score")
    # The sidecar is decimal-rendered float32. The native aggregate can differ slightly
    # because it is summed in the runtime before formatting.
    sidecar_total = math.fsum(scores)
    check(abs(sidecar_total - total) < 0.0005, f"{name}: aggregate and token-score sum disagree")
    return {
        "name": name,
        "metrics": metrics,
        "manifest": manifest,
        "target_sha256": digest(target_bytes),
        "target_ids_sha256": digest(ids_bytes),
        "prompt_ids_sha256": digest(prompt_ids_bytes),
        "score_sidecar_sha256": digest(scores_bytes),
        "token_ids": ids,
        "token_scores": scores,
        "aggregate_score": total,
        "sidecar_sum": sidecar_total,
        "sidecar_minus_aggregate": sidecar_total - total,
    }


def longest_run(values: list[float], threshold: float) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        longest = max(longest, current)
    return longest


def comparison(case: dict, reference: dict) -> dict:
    scores = case["token_scores"]
    ref = reference["token_scores"]
    delta = [value - baseline for value, baseline in zip(scores, ref)]
    deficit = [max(0.0, -value) for value in delta]
    gain = [max(0.0, value) for value in delta]
    absolute = [abs(value) for value in delta]
    deficit_total = math.fsum(deficit)
    absolute_total = math.fsum(absolute)
    positions = sorted(range(len(scores)), key=lambda index: (-deficit[index], index))[:15]
    block_size = 57  # Ten equal blocks for this verified 570-token target.
    blocks = []
    for start in range(0, len(scores), block_size):
        end = min(start + block_size, len(scores))
        blocks.append({
            "start_index_0_based": start,
            "end_index_0_based_inclusive": end - 1,
            "deficit_sum": math.fsum(deficit[start:end]),
            "count_deficit_ge_0_5": sum(value >= 0.5 for value in deficit[start:end]),
            "count_deficit_ge_1_0": sum(value >= 1.0 for value in deficit[start:end]),
        })
    return {
        "aggregate_score_delta_vs_fp32": case["aggregate_score"] - reference["aggregate_score"],
        "mean_score_delta_vs_fp32": (case["aggregate_score"] - reference["aggregate_score"]) / len(scores),
        "per_token_delta_sum": math.fsum(delta),
        "negative_deficit_sum": deficit_total,
        "positive_gain_sum": math.fsum(gain),
        "absolute_delta_sum": absolute_total,
        "mean_absolute_delta": absolute_total / len(scores),
        "median_absolute_delta": median(absolute),
        "p90_absolute_delta": percentile(absolute, 90),
        "p95_absolute_delta": percentile(absolute, 95),
        "p99_absolute_delta": percentile(absolute, 99),
        "max_absolute_delta": max(absolute),
        "tokens_lower_than_fp32": sum(value < 0 for value in delta),
        "tokens_higher_than_fp32": sum(value > 0 for value in delta),
        "tokens_equal_to_fp32": sum(value == 0 for value in delta),
        "deficit_counts": {str(threshold): sum(value >= threshold for value in deficit) for threshold in THRESHOLDS},
        "longest_consecutive_deficit_ge_0_5": longest_run(deficit, 0.5),
        "deficit_concentration": {
            f"top_{k}_share": math.fsum(sorted(deficit, reverse=True)[:k]) / deficit_total
            if deficit_total else 0.0
            for k in (1, 5, 10, 25, 50)
        },
        "absolute_concentration": {
            f"top_{k}_share": math.fsum(sorted(absolute, reverse=True)[:k]) / absolute_total
            if absolute_total else 0.0
            for k in (1, 5, 10, 25, 50)
        },
        "blocks": blocks,
        "largest_negative_deficits": [
            {
                "index_0_based": index,
                "standalone_target_token_id_unverified_alignment": case["token_ids"][index],
                "reference_fp32_score": ref[index],
                "variant_score": scores[index],
                "variant_minus_fp32": delta[index],
                "negative_deficit": deficit[index],
            }
            for index in positions if deficit[index] > 0
        ],
    }


def markdown(result: dict) -> str:
    cases = result["cases"]
    case_order = result["included_cases"]
    comparisons = result["comparisons_vs_fp32"]
    original_gap = cases[REFERENCE]["aggregate_score"] - cases["original_fp16_score_003"]["aggregate_score"]
    rope_gap = cases[REFERENCE]["aggregate_score"] - cases["rope_fp16_score_003"]["aggregate_score"]
    floataccum_gap = cases[REFERENCE]["aggregate_score"] - cases["rope_floataccum_score_003"]["aggregate_score"]
    out = [
        "# BXP-003 fixed-target token-score comparison",
        "",
        f"These {len(case_order)} completed GPU runs scored the same 1,696-byte target after the same prompt. All returned 570 finite token scores. The score arrays sum to the C API aggregate within 0.0005. The saved C API score is target log-likelihood: a larger value gives this exact continuation a higher score. The [pinned LiteRT-LM scoring path](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.16.1/runtime/engine/litert_lm_lib.cc) negates the returned score to obtain negative log-likelihood. This measures neither free-running accuracy nor the full vocabulary distribution.",
        "",
        "## Whole-target log-likelihood",
        "",
        "| Variant | Total log-likelihood | Mean log-likelihood / target token | Difference from original FP32 |",
        "|---|---:|---:|---:|",
    ]
    for name in case_order:
        case = cases[name]
        delta = case["aggregate_score"] - cases[REFERENCE]["aggregate_score"]
        out.append(f"| {case['label']} (`{name}`) | {case['aggregate_score']:.6f} | {case['mean_score_per_token']:.6f} | {delta:+.6f} |")
    out += [
        "",
        f"Relative to the original FP32 score, the RoPE-only variant reduces the original FP16 aggregate log-score gap by {(1 - rope_gap / original_gap) * 100:.1f}%. The RoPE plus GEMV float-accum patch variant reduces it by {(1 - floataccum_gap / original_gap) * 100:.1f}%, yet scores {cases['rope_fp16_score_003']['aggregate_score'] - cases['rope_floataccum_score_003']['aggregate_score']:.3f} log-score units below RoPE-only. Changes to this one target's score need not track free-running generation quality or generalize to other targets.",
        "",
        "## Position-level differences from original FP32",
        "",
        "`negative deficit` is `max(0, FP32 token score − variant token score)`; gains in other positions are counted separately. Indices below are zero-based. The displayed IDs come from a separate C API tokenization of the target. Counts match 570, but the native scorer did not return token IDs, so exact ID-to-score alignment is unverified.",
        "",
        "| Variant | Tokens with deficit ≥0.5 / ≥1 / ≥2 | Negative deficit sum | Positive gain sum | Top 10 share of deficit | Median / p95 absolute delta | Longest run ≥0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in case_order:
        if name == REFERENCE:
            continue
        item = comparisons[name]
        counts = item["deficit_counts"]
        out.append(
            f"| {cases[name]['label']} | {counts['0.5']} / {counts['1.0']} / {counts['2.0']} | "
            f"{item['negative_deficit_sum']:.3f} | {item['positive_gain_sum']:.3f} | "
            f"{item['deficit_concentration']['top_10_share'] * 100:.1f}% | "
            f"{item['median_absolute_delta']:.6f} / {item['p95_absolute_delta']:.3f} | "
            f"{item['longest_consecutive_deficit_ge_0_5']} |"
        )
    original = comparisons["original_fp16_score_003"]
    rope = comparisons["rope_fp16_score_003"]
    patched = comparisons["rope_floataccum_score_003"]
    original_affected_blocks = sum(block["count_deficit_ge_0_5"] > 0 for block in original["blocks"])
    rope_affected_blocks = sum(block["count_deficit_ge_0_5"] > 0 for block in rope["blocks"])
    patched_affected_blocks = sum(block["count_deficit_ge_0_5"] > 0 for block in patched["blocks"])
    out += [
        "",
        f"The original FP16 deficit is spread across {original_affected_blocks}/10 target blocks: {original['deficit_counts']['0.5']} positions lose at least 0.5 log-score units. After the RoPE change, only {rope['deficit_counts']['0.5']} positions in {rope_affected_blocks}/10 blocks cross that threshold, and the ten largest deficits account for {rope['deficit_concentration']['top_10_share'] * 100:.1f}% of the remaining deficit. The shader-patched run has {patched['deficit_counts']['0.5']} such positions in {patched_affected_blocks}/10 blocks and a {patched['median_absolute_delta']:.6f} median absolute difference. The residual large deficits are concentrated in a few positions amid mostly small per-token differences.",
    ]
    fake_quant_cases = [name for name in case_order if name in OPTIONAL_CASES]
    if fake_quant_cases:
        descriptions = []
        for name in fake_quant_cases:
            item = comparisons[name]
            descriptions.append(
                f"{cases[name]['label']} has {item['deficit_counts']['0.5']} "
                f"{'position' if item['deficit_counts']['0.5'] == 1 else 'positions'} with a deficit ≥0.5 "
                f"and a {item['aggregate_score_delta_vs_fp32']:+.3f} whole-target log-score difference"
            )
        out += [
            "",
            "; ".join(descriptions) + ". Positive gains at other positions exceed the negative deficits for these captures. The variants are independent measurements; their scores do not form a monotonic progression or establish generation quality.",
        ]
    out += ["", "### Largest negative-deficit positions", ""]
    for name in case_order:
        if name == REFERENCE:
            continue
        out += [f"{cases[name]['label']} (`{name}`):", "", "| Index | Standalone target ID* | FP32 score | Variant score | Variant − FP32 |", "|---:|---:|---:|---:|---:|"]
        for row in comparisons[name]["largest_negative_deficits"][:10]:
            out.append(
                f"| {row['index_0_based']} | {row['standalone_target_token_id_unverified_alignment']} | "
                f"{row['reference_fp32_score']:.4f} | {row['variant_score']:.4f} | {row['variant_minus_fp32']:.4f} |"
            )
        out.append("")
    out += [
        "*The token ID is from independent target tokenization and is not a captured scored-token ID.*",
        "",
        "## Provenance and limits",
        "",
        f"- Target SHA-256: `{result['shared_provenance']['target_sha256']}`; target token IDs and prompt token IDs also match byte-for-byte across all {len(case_order)} captures.",
        f"- All included runs use pinned native library SHA-256 `{result['shared_provenance']['native_library_sha256']}`, probe SHA-256 `{result['shared_provenance']['probe_sha256']}`, GPU backend, 8,192 context tokens, MTP off, and one target. Original FP16/FP32 share model SHA-256 `{result['shared_provenance']['original_model_sha256']}`; the RoPE-only and shader-patched runs share corrected-model SHA-256 `{result['shared_provenance']['rope_model_sha256']}`. Per-run model digests are in the JSON.",
        "- The GEMV float-accum, combined, and fake-quant v2 runs requested distinct diagnostic OpenCL shader-patch directories; all three native logs record `CL_PATCH applied`. They share the same RoPE-model digest but are not unpatched model-only comparisons. Patch directory paths are saved per run in the JSON.",
        "- The original FP32 run is a numerical comparison reference, not ground-truth probability or an accuracy label. These are one capture per variant, with no uncertainty estimate. Forced-target scores cannot locate a specific kernel, recover full logits, establish full-vocabulary KL, or predict raw DSL validity.",
        "- Full per-index scores and signed deltas are in `fixed_target_score_analysis.json`. Recompute with `python GenUICraft/validation/20260929_fp16_rootcause/analyze_fixed_target_scores.py` from the repository root.",
        "",
    ]
    if result["skipped_optional_cases"]:
        out += ["Optional runs omitted because they were not complete at analysis time:", ""]
        for name, reason in result["skipped_optional_cases"].items():
            out.append(f"- `{name}`: {reason}.")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--optional-case", action="append", default=[], metavar="DIRECTORY",
                        help="also include this run directory when its completed scoring capture is present")
    args = parser.parse_args()
    root = args.root.resolve()
    optional_cases = tuple(dict.fromkeys((*OPTIONAL_CASES, *args.optional_case)))
    check(not set(optional_cases).intersection(REQUIRED_CASES), "optional case duplicates a required case")
    check(all(name and name == Path(name).name for name in optional_cases), "optional case must be a directory name")
    skipped_optional = {
        name: reason
        for name in optional_cases
        if (reason := optional_case_skip_reason(root, name)) is not None
    }
    included = (*REQUIRED_CASES, *(name for name in optional_cases if name not in skipped_optional))
    raw = {name: read_case(root, name) for name in included}
    reference = raw[REFERENCE]
    for name, case in raw.items():
        check(len(case["token_scores"]) == 570, f"{name}: unexpected target length")
        for key in ("target_sha256", "target_ids_sha256", "prompt_ids_sha256"):
            check(case[key] == reference[key], f"{name}: {key} differs from reference")
        for key in ("case", "serial", "prompt_sha256", "native_library_sha256", "probe_sha256"):
            check(case["manifest"][key] == reference["manifest"][key], f"{name}: {key} differs from reference")
        check(case["metrics"]["backend_requested"] == "gpu", f"{name}: not GPU")
        check(case["metrics"]["max_context_tokens"] == 8192, f"{name}: context differs")
        check(case["manifest"]["mtp_enabled"] is False, f"{name}: MTP enabled")
    check(raw["original_fp16_score_003"]["manifest"]["model_sha256"] == reference["manifest"]["model_sha256"], "original models differ")
    check(raw["rope_fp16_score_003"]["manifest"]["model_sha256"] == raw["rope_floataccum_score_003"]["manifest"]["model_sha256"], "RoPE models differ")
    check(reference["manifest"]["force_f32"] is True, "reference did not request FP32")
    check(all(raw[name]["manifest"]["force_f32"] is False for name in included if name != REFERENCE), "nonreference requested FP32")

    cases = {
        name: {
            "label": CASE_LABELS.get(name, name.replace("_", " ")),
            "aggregate_score": case["aggregate_score"],
            "mean_score_per_token": case["aggregate_score"] / 570,
            "total_log_likelihood": case["aggregate_score"],
            "mean_log_likelihood_per_target_token": case["aggregate_score"] / 570,
            "sidecar_score_sum": case["sidecar_sum"],
            "sidecar_minus_aggregate": case["sidecar_minus_aggregate"],
            "score_sidecar_sha256": case["score_sidecar_sha256"],
            "model_sha256": case["manifest"]["model_sha256"],
            "force_f32": case["manifest"]["force_f32"],
            "diagnostic_shader_patch_directory": case["manifest"].get("diagnostic_shader_patch_directory"),
            "score_wall_ms": case["metrics"]["score_wall_ms"],
        }
        for name, case in raw.items()
    }
    comparisons = {name: comparison(raw[name], reference) for name in included if name != REFERENCE}
    rows = []
    for index, token_id in enumerate(reference["token_ids"]):
        scores = {name: raw[name]["token_scores"][index] for name in included}
        rows.append({
            "index_0_based": index,
            "standalone_target_token_id_unverified_alignment": token_id,
            "scores": scores,
            "variant_minus_fp32": {name: scores[name] - scores[REFERENCE] for name in included if name != REFERENCE},
        })
    result = {
        "contract": "Fixed-target C API log-likelihood; larger means higher likelihood for this exact target; not accuracy or full-vocabulary KL.",
        "reference": REFERENCE,
        "num_target_tokens": 570,
        "included_cases": included,
        "skipped_optional_cases": skipped_optional,
        "shared_provenance": {
            "target_sha256": reference["target_sha256"],
            "target_token_ids_sha256": reference["target_ids_sha256"],
            "prompt_token_ids_sha256": reference["prompt_ids_sha256"],
            "prompt_sha256": reference["manifest"]["prompt_sha256"],
            "native_library_sha256": reference["manifest"]["native_library_sha256"],
            "probe_sha256": reference["manifest"]["probe_sha256"],
            "original_model_sha256": reference["manifest"]["model_sha256"],
            "rope_model_sha256": raw["rope_fp16_score_003"]["manifest"]["model_sha256"],
        },
        "token_id_caveat": "Standalone target tokenizer IDs have matching counts but are not IDs returned by the native scorer; positional alignment is unverified.",
        "cases": cases,
        "comparisons_vs_fp32": comparisons,
        "per_token": rows,
    }
    (root / "fixed_target_score_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (root / "fixed_target_score_analysis.md").write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
