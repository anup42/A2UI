"""Read-only local SentencePiece proxy for current v10 preparation token boundaries.

This intentionally does not claim parity with the unavailable remote HF tokenizer.
Run with: python training/reports/v10_full_audit_20260926/pipeline_token_probe.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "training/src"), str(ROOT / "dataset/src"), str(ROOT / "training/scripts/audits")]

import sentencepiece as spm

from full_data_ir_20260913 import wire_setup
from ir_training.data.express_preparation import prepare_row
from ir_training.data.shared_prompt import load_shared_prompt_contract, normalize_row

wire_setup(False)
SP = spm.SentencePieceProcessor(model_file=str(ROOT / "tmp/e2b_mobile_20260921/tokenizer.model"))
CONTRACT = load_shared_prompt_contract(
    ROOT / "GenUICraft/genuicraft/src/main/assets/genuicraft/prompts/e2b_v10_shared_prompt.json",
    require_current=True,
)


def render_prefix(messages: list[dict[str, str]], *, trim: bool) -> str:
    body = "".join(
        "<|turn>" + ("model" if item["role"] == "assistant" else item["role"])
        + "\n" + (item["content"].strip() if trim else item["content"]) + "<turn|>\n"
        for item in messages
    )
    return body + "<|turn>model\n"


def token_count(value: str) -> int:
    return len(SP.encode(value, out_type=int))


def summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {}
    return {
        "min": min(values), "median": statistics.median(values),
        "max": max(values), "nonzero": sum(value != 0 for value in values),
        "negative": sum(value < 0 for value in values), "positive": sum(value > 0 for value in values),
    }


def main() -> None:
    records: list[dict] = []
    failed: list[dict] = []
    selections = {
        "train": set(range(1, 129)) | set(range(700, 91116, 700)) | {6416},
        "val": set(range(1, 129)) | set(range(15, 1863, 15)),
    }
    for split, selected in selections.items():
        source = ROOT / "training/outputs/datasets/full_data_archive_recovered_v10" / f"{split}.jsonl"
        with source.open(encoding="utf-8") as stream:
            for line, raw in enumerate(stream, 1):
                if line > max(selected):
                    break
                if line not in selected:
                    continue
                row = json.loads(raw)
                try:
                    normalized = normalize_row(row, CONTRACT)
                    prepared, target, _ = prepare_row(normalized, "root-first")
                except Exception as exc:
                    failed.append({"split": split, "line": line, "id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"})
                    continue
                response = row["response_text"]
                root_prefix = render_prefix(
                    CONTRACT["scaffold"]["messages"]
                    + [{"role": "user", "content": CONTRACT["scaffold"]["task_prefix"] + response}],
                    trim=False,
                )
                actual_prefix = render_prefix(prepared["messages"][:-1], trim=True)
                suffix = target.text + "<turn|>\n"
                root_sequence = 1 + token_count(root_prefix + row["completion"] + "<turn|>\n")
                canonical_concat = 1 + token_count(actual_prefix + suffix)
                canonical_separate = 1 + token_count(actual_prefix) + token_count(suffix)
                records.append({
                    "split": split, "line": line, "id": row.get("id"),
                    "root_prompt": 1 + token_count(root_prefix),
                    "actual_prompt": 1 + token_count(actual_prefix),
                    "root_sequence": root_sequence,
                    "canonical_concat": canonical_concat,
                    "canonical_separate": canonical_separate,
                    "raw_target": token_count(row["completion"]),
                    "canonical_target": token_count(target.text),
                })
    comparisons = {
        "prompt_trim_delta": [r["actual_prompt"] - r["root_prompt"] for r in records],
        "serialization_and_trim_delta": [r["canonical_concat"] - r["root_sequence"] for r in records],
        "boundary_delta": [r["canonical_separate"] - r["canonical_concat"] for r in records],
        "true_proxy_minus_root": [r["canonical_separate"] - r["root_sequence"] for r in records],
        "canonical_target_minus_raw": [r["canonical_target"] - r["raw_target"] for r in records],
    }
    report = {
        "sample": "first 128 of each split, spaced rows across each split, and train 6416; deterministic; accepted normalization only",
        "rows_examined": sum(len(selected) for selected in selections.values()),
        "normalized": len(records), "failed": len(failed), "failure_examples": failed[:10],
        "tokenizer_sha256": "e594c8a90eb08d8bda498ff4747977dc827ae0c3c56b5c0d41a605a22d02ef03",
        "comparison": {key: summary(values) for key, values in comparisons.items()},
        "threshold_disagreements": {
            str(limit): sum((r["root_sequence"] > limit) != (r["canonical_separate"] > limit) for r in records)
            for limit in (4096, 6144, 8192)
        },
        "target_2048_disagreements": sum((r["raw_target"] > 2048) != (r["canonical_target"] > 2048) for r in records),
        "largest_boundary_differences": sorted(records, key=lambda r: abs(r["canonical_separate"] - r["canonical_concat"]), reverse=True)[:8],
        "largest_total_differences": sorted(records, key=lambda r: abs(r["canonical_separate"] - r["root_sequence"]), reverse=True)[:8],
    }
    out = Path(__file__).with_name("pipeline_token_probe.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("normalized", "failed", "comparison", "threshold_disagreements", "target_2048_disagreements")}, indent=2))


if __name__ == "__main__":
    main()
