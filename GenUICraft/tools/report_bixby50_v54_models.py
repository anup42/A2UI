"""Re-score saved checkpoint and native Bixby50 outputs without running inference.

Raw completions are scored unchanged. Missing native cases remain unmeasured;
post-SDK repair is a separate population. Existing inputs are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import html
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
sys.path.insert(0, str(ROOT / "dataset/src"))
from ir_training.eval.metrics import score_prediction  # noqa: E402
from pipeline.genui_quality import generation_reward_v5_4  # noqa: E402
from pipeline.genui_quality.identity_v5_4 import (  # noqa: E402
    android_renderer_manifest,
    score_source_manifest_v5_4,
)

LABELS = {
    "checkpoint_r32": "Checkpoint LoRA R32",
    "checkpoint_r64": "Checkpoint LoRA R64",
    "litert_mtp_off": "LiteRT W4 / GPU FP32 / MTP off",
    "litert_mtp_on": "LiteRT W4 / GPU FP32 / MTP on",
    "sdk_mtp_off": "MTP-off output after SDK 0.5.6 repair",
    "sdk_mtp_on": "MTP-on output after SDK 0.5.6 repair",
}
RAW_MODELS = tuple(LABELS)[:4]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def path_label(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def stats(rows):
    scores = [r["score"] for r in rows]
    assert scores and all(math.isfinite(x) and 0 <= x <= 100 for x in scores)
    caps = Counter(c["name"] if "name" in c else c.get("code", str(c))
                   for r in rows for c in r["breakdown"]["active_caps"])
    dims = {}
    for dim in ("integrity", "fidelity", "semantic_mapping", "hierarchy", "economy", "accessibility"):
        observed = [r["breakdown"]["dimensions"].get(dim) for r in rows]
        observed = [x for x in observed if x is not None]
        dims[dim] = {"observed_count": len(observed),
                     "mean_0_100": statistics.mean(observed) * 100 if observed else None}
    return {
        "n": len(rows), "ids": sorted(r["id"] for r in rows),
        "mean": statistics.mean(scores), "median": statistics.median(scores),
        "min": min(scores), "max": max(scores),
        "strict_valid": sum(r["strict_valid"] for r in rows),
        "wire_valid": sum(r["metrics"]["raw_standard_a2ui_valid"] for r in rows),
        "fully_reachable": sum(r["metrics"]["fully_root_reachable_v5_4"] for r in rows),
        "zero_score": sum(x == 0 for x in scores),
        "at_least_80": sum(x >= 80 for x in scores),
        "active_cap_case_counts": dict(caps), "dimensions_observed_only": dims,
        "finish_reason_counts": dict(Counter(str(r["finish_reason"]) for r in rows)),
    }


def table(headers, rows):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(cell, r)) + " |" for r in rows])


def fmt(value):
    return "N/A" if value is None else f"{value:.2f}"


def failure_category(row):
    if row["strict_valid"]:
        return None
    error = row["metrics"].get("schema_error") or ""
    raw = row["raw_output"]
    if "references missing" in error:
        return "Missing component reference"
    if "Duplicate A2UI" in error:
        return "Duplicate component ID"
    if "exactly one" in error:
        if raw.strip().startswith("Hungary"):
            return "Extraneous Hungary prefix outside the envelope"
        if row["finish_reason"] == "max_new_tokens":
            return "Output-token limit and incomplete envelope"
        return "Incomplete envelope"
    return "Expression syntax or delimiter"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(r"C:\Users\anupk\Downloads\trained_model_data\e2b_runD_predictions"))
    parser.add_argument("--native-dir", type=Path, default=ROOT / "GenUICraft/validation/20260928_fp32_bixby50_mtp")
    parser.add_argument("--repair-dir", type=Path, default=ROOT / "GenUICraft/validation/20260929_bixby50_presentation/device/after")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit("Choose a new output directory; reports are immutable.")
    out.mkdir(parents=True)
    (out / "evidence").mkdir()
    (out / "scored").mkdir()
    manifest_files = {}

    def track(path):
        key = path_label(path)
        manifest_files[key] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
        return key

    corpus_path = ROOT / "training/data/eval/bixby50_v1/bixby50.jsonl"
    android_path = ROOT / "android/app/src/main/assets/genuicraft_bixby50.jsonl"
    corpus = {r["id"]: r for r in read_jsonl(corpus_path)}
    android = {r["id"]: r for r in read_jsonl(android_path)}
    all_ids = {f"BXP-{i:03}" for i in range(1, 51)}
    assert set(corpus) == set(android) == all_ids
    assert all(corpus[i]["response_text"] == android[i]["text"] for i in all_ids)
    for p in (corpus_path, android_path, Path(__file__),
              ROOT / "dataset/configs/genui_metric_v5_4.yaml"):
        track(p)
    provenance = read_json(args.native_dir / "provenance.json")
    protocol = read_json(args.native_dir / "protocol.json")
    track(args.native_dir / "provenance.json")
    track(args.native_dir / "protocol.json")
    save_json(out / "evidence/native_provenance.json", provenance)
    save_json(out / "evidence/native_protocol.json", protocol)
    sets = {}
    checkpoint_rows = {}
    validations = {"source_corpus_cases": 50, "android_training_sources_equal": True,
                   "raw_scores_use_repair": False, "reference_ir_available": False,
                   "new_inference_performed": False, "prompt_hash_checks": [],
                   "native_excluded_incomplete_directories": []}

    def evaluate(model, case, raw, origin, *, runtime, archived=None, sdk=None):
        source = corpus[case]["response_text"]
        # Deliberately no domain/intent hints: checkpoint rows have none, and
        # native rows must be measured with the identical source-derived contract.
        metrics = score_prediction(source, None, raw, metric_version="v5_4")
        breakdown = dataclasses.asdict(generation_reward_v5_4(raw, source))
        assert abs(metrics["generation_reward_v5_4"] - breakdown["quality_0_100"]) < 1e-9
        assert metrics["metric_identity_v5_4"]["metric_fingerprint"] == breakdown["metric_fingerprint"]
        assert "exact_match" not in metrics and "semantic_match" not in metrics
        finish = runtime.get("finishReason", runtime.get("stop_reason", "not recorded"))
        return {
            "model": model, "id": case, "query": android[case]["query"],
            "response_text": source, "raw_output": raw,
            "raw_output_sha256": sha_text(raw), "source_sha256": sha_text(source),
            "input_file": origin, "score": metrics["generation_reward_v5_4"],
            "strict_valid": metrics["schema_valid_strict"],
            "archived_score": archived, "score_delta_from_archive":
                metrics["generation_reward_v5_4"] - archived if archived is not None else None,
            "finish_reason": finish, "runtime": runtime, "sdk": sdk,
            "metrics": metrics, "breakdown": breakdown,
        }

    for rank in (32, 64):
        key = f"checkpoint_r{rank}"
        path = args.checkpoint_dir / f"r{rank}_bixby50_scored.jsonl"
        origin = track(path)
        supplied = read_jsonl(path)
        assert len(supplied) == 50 and {r["id"] for r in supplied} == all_ids
        checkpoint_rows[key] = {r["id"]: r for r in supplied}
        sets[key] = []
        for r in sorted(supplied, key=lambda x: x["id"]):
            assert r["response_text"] == corpus[r["id"]]["response_text"]
            assert r.get("expected") is None and r.get("reference_available") is False
            assert r.get("intent") is None and not r.get("assets")
            assert r.get("expected_ui_contract_v5_4") is None
            assert r["raw_generated_text"] == r["generated_text"]
            sets[key].append(evaluate(key, r["id"], r["raw_generated_text"], origin,
                runtime=r["runtime"], archived=r["metrics"]["generation_reward_v5_4"]))
            if len(sets[key]) % 10 == 0:
                print(f"Scored {key}: {len(sets[key])}/50", flush=True)
        save_json(out / f"evidence/{key}_supplied_score_identity.json",
                  supplied[0]["metrics"]["metric_identity_v5_4"])

    for mode in ("off", "on"):
        key = f"litert_mtp_{mode}"
        repair_key = f"sdk_mtp_{mode}"
        sets[key], sets[repair_key] = [], []
        prior_path = args.native_dir / f"report/mtp_{mode}/scored_predictions.jsonl"
        prior = {r["id"]: r for r in read_jsonl(prior_path)}
        track(prior_path)
        for batch in sorted((args.native_dir / "batches").glob(f"*_mtp_{mode}")):
            if not (batch / "run_config.json").is_file():
                continue
            config = read_json(batch / "run_config.json")
            track(batch / "run_config.json")
            rt = config["runtime"]
            assert rt["mtpEnabled"] == (mode == "on")
            assert rt["gpuPrecision"] == "FP32" and rt["accelerator"] == "GPU"
            assert rt["maxOutputTokens"] == 2048 and rt["temperature"] == 0.0
            assert config["model"]["basename"] == Path(protocol["modelPath"]).name
            for folder in sorted(batch.glob("BXP-*")):
                if not (folder / "result.json").is_file():
                    validations["native_excluded_incomplete_directories"].append(path_label(folder))
                    continue
                result = read_json(folder / "result.json")
                source = read_json(folder / "source.json")
                case = result["id"]
                assert source["id"] == folder.name == case
                assert source["text"] == corpus[case]["response_text"]
                assert result["providerCalls"] == 1 and result["usedFallback"] is False
                prompt = result["renderedPromptSha256"]
                for ck in checkpoint_rows:
                    assert prompt == checkpoint_rows[ck][case]["runtime"]["prompt_sha256"]
                validations["prompt_hash_checks"].append({"model": key, "id": case, "sha256": prompt, "matches_both_checkpoints": True})
                metrics = read_json(folder / "metrics.json")
                for p in (folder / "source.json", folder / "metrics.json", folder / "result.json"):
                    track(p)
                raw_path = folder / "output.express"
                raw = raw_path.read_text(encoding="utf-8")
                origin = track(raw_path)
                old_score = prior[case]["metrics"]["generation_reward_v5_4"]
                sets[key].append(evaluate(key, case, raw, origin, runtime=metrics, archived=old_score))
                assert sets[key][-1]["strict_valid"] == result["rawStrictValid"]
                rp = args.repair_dir / f"r64_fp32_{mode}" / case
                if rp.is_dir():
                    assert (rp / "source.output.express").read_bytes() == raw_path.read_bytes()
                    replay = read_json(rp / "replay_result.json")
                    assert replay["modelCalls"] == 0 and replay["sourceExpressSha256"] == sha_file(raw_path)
                    recovered_path = rp / "recovered.output.express"
                    assert replay["recoveredExpressSha256"] == sha_file(recovered_path)
                    track(rp / "source.output.express")
                    track(rp / "replay_result.json")
                    recovered_origin = track(recovered_path)
                    sets[repair_key].append(evaluate(repair_key, case,
                        recovered_path.read_text(encoding="utf-8"), recovered_origin,
                        runtime=metrics, sdk={"version": "0.5.6", "repair_kind": replay["repairKind"],
                                             "model_calls": 0, "source_fallback": False}))
        assert len({r["id"] for r in sets[key]}) == len(sets[key])
        assert {r["id"] for r in sets[key]} == set(prior)
        print(f"Scored {key}: {len(sets[key])}/50; repaired: {len(sets[repair_key])}", flush=True)

    paired = sorted(set.intersection(*({r["id"] for r in sets[k]} for k in RAW_MODELS)))
    assert paired
    identity_fields = ("metric_version", "metric_name", "metric_fingerprint", "reward_pipeline_fingerprint")
    identities = {tuple(r["metrics"]["metric_identity_v5_4"][k] for k in identity_fields)
                  for rows in sets.values() for r in rows}
    assert len(identities) == 1
    identity = dict(zip(identity_fields, identities.pop()))
    by_id = {k: {r["id"]: r for r in rows} for k, rows in sets.items()}
    # Source contracts must also match for each scored model/case, including repairs.
    for case in all_ids:
        contract_hashes = {rows[case]["metrics"]["metric_identity_v5_4"]["expected_contract_hash"]
                           for rows in by_id.values() if case in rows}
        assert len(contract_hashes) == 1
    aggregate = {k: stats(rows) for k, rows in sets.items()}
    paired_stats = {k: stats([r for r in rows if r["id"] in paired]) for k, rows in sets.items()}
    pair_diffs = {}
    for first, second in (("checkpoint_r32", "checkpoint_r64"),
                          ("checkpoint_r64", "litert_mtp_off"),
                          ("checkpoint_r64", "litert_mtp_on"),
                          ("litert_mtp_off", "litert_mtp_on")):
        diffs = [by_id[second][i]["score"] - by_id[first][i]["score"] for i in paired]
        pair_diffs[f"{second}_minus_{first}"] = {
            "mean_points": statistics.mean(diffs),
            "second_higher": sum(d > 1e-9 for d in diffs),
            "first_higher": sum(d < -1e-9 for d in diffs),
            "ties": sum(abs(d) <= 1e-9 for d in diffs), "n": len(paired),
        }
    archived = {k: {"archived_mean": statistics.mean(r["archived_score"] for r in sets[k]),
                    "current_mean": aggregate[k]["mean"],
                    "changed_cases": sum(abs(r["score_delta_from_archive"]) > 1e-9 for r in sets[k]),
                    "max_absolute_delta": max(abs(r["score_delta_from_archive"]) for r in sets[k])}
                for k in RAW_MODELS}
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "metric_identity": identity,
               "score_scope": "Raw source-grounded generation quality; not factual accuracy",
               "all_available": aggregate, "paired_ids": paired, "paired": paired_stats,
               "paired_differences": pair_diffs, "archived_score_reproduction": archived}
    save_json(out / "summary.json", summary)
    validations.update({"common_metric_identity": True, "same_source_contract_for_each_case": True,
                        "all_raw_archived_scores_reproduced": all(x["changed_cases"] == 0 for x in archived.values()),
                        "native_full_bixby50_complete": all(aggregate[k]["n"] == 50 for k in RAW_MODELS[2:]),
                        "missing_native_ids": {k: sorted(all_ids - set(by_id[k])) for k in RAW_MODELS[2:]}})
    save_json(out / "validation.json", validations)
    save_json(out / "evidence/scorer_source_manifest.json", score_source_manifest_v5_4())
    save_json(out / "evidence/android_renderer_manifest.json", android_renderer_manifest())
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        git_head = None
    save_json(out / "input_manifest.json", {"files": manifest_files, "git_head": git_head,
              "python": sys.version, "platform": platform.platform(), "native_provenance": provenance,
              "metric_identity": identity, "supplied_checkpoint_weight_hash": None,
              "checkpoint_identity_limit": "Supplied prediction filenames and CUDA runtime metadata identify ranks; exact checkpoint weight/step hashes were not provided in these JSONL records."})
    for key, rows in sets.items():
        with (out / f"scored/{key}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    fields = ["id", "model", "score", "strict_valid", "finish_reason", "archived_score", "score_delta_from_archive", "source_sha256", "raw_output_sha256", "input_file"]
    with (out / "per_case_scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rows in sets.values():
            writer.writerows({k: r[k] for k in fields} for r in rows)
    with (out / "bixby50_matrix.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "query", *RAW_MODELS])
        for case in sorted(all_ids):
            writer.writerow([case, android[case]["query"], *[by_id[k][case]["score"] if case in by_id[k] else "" for k in RAW_MODELS]])
    write_reports(out, summary, by_id, android, validations)
    print(json.dumps({"report": str(out / "REPORT.md"), "raw_all_available": {k: aggregate[k] for k in RAW_MODELS},
                      "paired_means": {k: paired_stats[k]["mean"] for k in RAW_MODELS},
                      "archived_score_reproduction": archived}, ensure_ascii=False, indent=2), flush=True)


def write_reports(out, summary, by_id, android, validation):
    available, paired = summary["all_available"], summary["paired"]
    identity = summary["metric_identity"]
    n = len(summary["paired_ids"])
    coverage_table = table(["Model / runtime", "Completed / 50", "Raw v5.4 mean /100", "Median", "Strict valid", "Zero-score cases"],
        [[LABELS[k], f'{available[k]["n"]}/50', fmt(available[k]["mean"]), fmt(available[k]["median"]),
          f'{available[k]["strict_valid"]}/{available[k]["n"]}', available[k]["zero_score"]] for k in RAW_MODELS])
    paired_table = table(["Model / runtime", "Same cases", "Raw v5.4 mean /100", "Strict valid", "Zero-score cases"],
        [[LABELS[k], n, fmt(paired[k]["mean"]), f'{paired[k]["strict_valid"]}/{n}', paired[k]["zero_score"]] for k in RAW_MODELS])
    case_table = table(["Case", "R32 checkpoint", "R64 checkpoint", "LiteRT MTP off", "LiteRT MTP on"],
        [[i, *[fmt(by_id[k][i]["score"]) if i in by_id[k] else "N/A" for k in RAW_MODELS]] for i in sorted(android)])
    repairs = table(["SDK output population", "Cases", "Raw mean", "After SDK repair mean", "Strict valid after repair"],
        [[LABELS[f"sdk_mtp_{m}"], available[f"sdk_mtp_{m}"]["n"], fmt(available[f"litert_mtp_{m}"]["mean"]),
          fmt(available[f"sdk_mtp_{m}"]["mean"]), f'{available[f"sdk_mtp_{m}"]["strict_valid"]}/{available[f"sdk_mtp_{m}"]["n"]}'] for m in ("off", "on")])
    delta = summary["paired_differences"]["litert_mtp_on_minus_litert_mtp_off"]
    lines = ["# Bixby50: checkpoint and LiteRT-LM v5.4 score report", "", "Generated: " + summary["created_at_utc"], "",
        "**The two supplied checkpoints have complete 50-case scores. The latest saved corrected LiteRT-LM GPU FP32 run is partial: 15 cases with MTP off and 12 with MTP on. A full 50-case LiteRT score is not yet measured.**", "",
        "V5.4 is **GenUI Representation Quality**, a source-grounded engineering score from 0 to 100. It measures how the generated UI represents the captured Bixby answer; it is **not factual-answer accuracy, exact-match accuracy, or a human visual-quality rating**. The scorer configuration marks it `uncalibrated_engineering_score`.", "",
        "## 1. All available raw outputs", "", coverage_table, "",
        "The 50-case checkpoint means and partial native means have different denominators. **Do not compare those means as an export accuracy gain/loss.** Every completed zero-scoring output stays in its denominator. Missing native cases remain N/A, not zero.", "",
        "## 2. Fair comparison on the same completed cases", "",
        "Paired cases: " + ", ".join(summary["paired_ids"]) + ".", "", paired_table, "",
        f"On these {n} cases, MTP-on minus MTP-off is **{delta['mean_points']:+.2f} score points**. MTP on scores higher in {delta['second_higher']} cases, lower in {delta['first_higher']}, and ties in {delta['ties']}. This is one deterministic-generation run per mode, not repeated trials or a full-cohort result.", "",
        "These numbers measure observed output quality, not checkpoint-to-native numerical parity. Different arithmetic, stopping/repetition guards, the speculative drafter, and export can all affect output. They do not isolate quantization as the cause. Bixby50 is a final-only holdout and should not be used to choose a checkpoint or tune hyperparameters.", "",
        "## 3. Method and provenance", "",
        "- Inputs: `r32_bixby50_scored.jsonl` and `r64_bixby50_scored.jsonl` in the supplied `e2b_runD_predictions` folder. Both contain 50 unique cases. Raw and serving-output fields are identical in every supplied row.",
        "- Native inputs: completed `output.express` files from `20260928_fp32_bixby50_mtp/batches`. No new inference was performed for this report. Incomplete case directories are excluded and listed in `validation.json`.",
        "- Native runtime: corrected R64 QAT-compatible LiteRT-LM, **W4 weights, GPU FP32 arithmetic**, temperature 0, 8,192 context tokens, 2,048 output-token limit, thinking disabled, MTP switched by batch. Device: Samsung SM-F776U / R3GL203AKSF / Android 17. FP32 here does not mean FP32 weights.",
        "- Native model SHA-256: `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62` from the saved device provenance. The supplied checkpoint JSONL records identify CUDA inference and rank by filename but do not supply an exact checkpoint weight/step hash; this report cannot independently certify the export lineage.",
        "- Every source text is byte-equivalent after UTF-8 text decoding to the frozen Bixby50 source in training and Android. All completed native rendered-prompt hashes match both supplied checkpoint prompt hashes for the same case.",
        "- All raw outputs are scored unchanged by `score_prediction(..., expected=None, metric_version='v5_4')`. Results are cross-checked against the underlying `generation_reward_v5_4` breakdown. The same source-derived expected contract is used for every model. There is no reference IR and no invented reference/exact-match score.",
        "- No repair, fallback, source reconstruction, Android render result, or repaired graph is passed into the headline raw score. Strict validity is the official Python Express syntax/catalog result; it is separate from v5.4 quality and Android screen success.",
        "- Default dimension weights: integrity 25%, fidelity 40%, semantic mapping 15%, hierarchy 10%, economy 7%, accessibility 3%. Applicability reweighting and quality caps also apply, so the result is not a simple correctness percentage.", "",
        "## 4. Archived-score reproducibility", "",
        table(["Population", "Supplied/archived mean", "Current mean", "Changed cases", "Maximum absolute delta"],
              [[LABELS[k], fmt(v["archived_mean"]), fmt(v["current_mean"]), v["changed_cases"], f'{v["max_absolute_delta"]:.12g}'] for k,v in summary["archived_score_reproduction"].items()]), "",
        "All populations were re-scored using one current identity. The archived fingerprint differs from the current one; the identity also hashes renderer sources and other contract artifacts. Numeric equivalence is checked case by case above, rather than assumed from the version number.", "",
        f"Current metric version: `{identity['metric_version']}`  \nMetric fingerprint: `{identity['metric_fingerprint']}`  \nReward pipeline fingerprint: `{identity['reward_pipeline_fingerprint']}`", "",
        "## 5. Why scores are lost", ""]
    for k in RAW_MODELS:
        rows = list(by_id[k].values())
        invalid = [r["id"] for r in rows if not r["strict_valid"]]
        zeros = [r["id"] for r in rows if r["score"] == 0]
        lines += [f"### {LABELS[k]}", "",
                  f"Strict-invalid cases ({len(invalid)}): " + (", ".join(invalid) or "none") + ".", "",
                  f"Zero-score cases ({len(zeros)}): " + (", ".join(zeros) or "none") + ".", "",
                  "Recorded termination reasons: " + "; ".join(f"{name}: {count}" for name, count in available[k]["finish_reason_counts"].items()) + ".", "",
                  "Active cap counts (a case can trigger several; the final binding cap is in its JSONL record):", "",
                  table(["Cap", "Cases"], list(available[k]["active_cap_case_counts"].items())) if available[k]["active_cap_case_counts"] else "No active caps.", "",
                  "Lowest-scoring examples:", "",
                  table(["Case", "Raw v5.4", "Strict valid", "First reported issue"],
                        [[r["id"], fmt(r["score"]), r["strict_valid"], r["metrics"].get("schema_error") or "; ".join(r["breakdown"]["errors"][:2]) or "See source fidelity, graph reachability and cap details"]
                         for r in sorted(rows, key=lambda r:(r["score"],r["id"]))[:5]]), ""]
    lines += ["Do not interpret the generic exact-number channel alone as table accuracy: native table data has its own table-fidelity channel. Legacy lexical coverage and Android source-integrity warning counts are not used as headline accuracy. Dimension means in `summary.json` are observed-only and carry their own denominators; absent diagnostics on parse failures are not treated as perfect scores. See [QUALITY_FINDINGS.md](QUALITY_FINDINGS.md) for concrete failures and structural-loss counts.", "",
        "## 6. SDK repair: separate application-output result", "", repairs, "",
        "This appendix re-scores the already captured SDK 0.5.6 repaired Express outputs from the same native generations. It performs no new generation and uses no source-text fallback. It is **post-processing quality**, not raw model accuracy. Successful compilation/rendering does not establish that all source content survived. Checkpoint outputs were not replayed through this SDK here, so repair scores must not be compared with unrepaired checkpoint scores as model accuracy.", "",
        "## 7. Complete 50-case score matrix", "", "All entries are raw v5.4 /100; N/A means no completed native output in this saved run.", "", case_table, "",
        "## 8. Files and reproduction", "",
        "- `index.html`: all 50 queries, source answers, raw outputs, errors, caps and dimension breakdowns side by side; SDK output in a separate appendix per case.",
        "- `CHECKPOINT_REPORT.md`: full 50-case R32/R64 checkpoint results.",
        "- `LITERT_MTP_REPORT.md`: partial native results and paired MTP comparison.",
        "- `QUALITY_FINDINGS.md`: invalid-output categories, disconnected graphs and representative raw-output examples.",
        "- `per_case_scores.csv`: all raw and separately labeled repaired records, hashes, runtime termination and score deltas.",
        "- `bixby50_matrix.csv`: raw 50-case matrix; missing native values are empty.",
        "- `scored/*.jsonl`: unmodified raw strings, source answers, metrics and complete official v5.4 breakdowns.",
        "- `summary.json`, `validation.json`, `input_manifest.json`, `evidence/`: aggregates, coverage, input hashes, runtime/scorer provenance.", "",
        "```powershell", "python GenUICraft/tools/report_bixby50_v54_models.py `", '  --checkpoint-dir "C:\\Users\\anupk\\Downloads\\trained_model_data\\e2b_runD_predictions" `',
        '  --output-dir "GenUICraft/validation/bixby50_v54_rescore_NEW"', "```", "",
        "Use a new output directory. Reproducing the same identity requires the same scorer and renderer contract sources. No checkpoint weight loading or native inference is performed by this report command.", ""]
    report = "\n".join(lines)
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    (out / "CHECKPOINT_REPORT.md").write_text("# Supplied checkpoint Bixby50 v5.4 report\n\n" +
        table(["Checkpoint", "Cases", "Raw v5.4 /100", "Strict valid", "Zero score"],
              [[LABELS[k], 50, fmt(available[k]["mean"]), f'{available[k]["strict_valid"]}/50',available[k]["zero_score"]] for k in RAW_MODELS[:2]]) +
        "\n\nThese are full 50-case raw-output representation-quality scores, not factual-answer accuracy. Source and prompt checks and the unchanged archived-score comparison are documented in [REPORT.md](REPORT.md). No repair is applied.\n\n" +
        table(["Case", "Query", "R32 score", "R32 strict", "R64 score", "R64 strict"],
              [[i,android[i]["query"],fmt(by_id["checkpoint_r32"][i]["score"]),by_id["checkpoint_r32"][i]["strict_valid"],fmt(by_id["checkpoint_r64"][i]["score"]),by_id["checkpoint_r64"][i]["strict_valid"]] for i in sorted(android)]) + "\n", encoding="utf-8")
    (out / "LITERT_MTP_REPORT.md").write_text("# LiteRT-LM Bixby50 v5.4 / MTP report\n\n" +
        "**Partial coverage: MTP off 15/50; MTP on 12/50. Full-cohort native scores are not measured.** This report uses the saved corrected R64 W4 / GPU FP32 runs on SM-F776U, temperature 0 and 2,048 output-token limit. No new inference was performed.\n\n" +
        "## Identical 12-case comparison\n\n" + paired_table +
        f"\n\nMTP-on minus off: {delta['mean_points']:+.2f} points. MTP on wins {delta['second_higher']}, off wins {delta['first_higher']}, ties {delta['ties']}.\n\n" +
        "The score measures source-to-UI representation quality, not factual correctness. Missing cases are N/A. Repairs are excluded from the primary score. These results do not isolate export or quantization loss.\n\n" +
        table(["Case", "MTP-off raw", "Off strict", "MTP-on raw", "On strict"],
              [[i, *[v for k in RAW_MODELS[2:] for v in ([fmt(by_id[k][i]["score"]),by_id[k][i]["strict_valid"]] if i in by_id[k] else ["N/A","N/A"])]] for i in sorted(set(by_id[RAW_MODELS[2]])|set(by_id[RAW_MODELS[3]]))]) +
        "\n\n## Separate SDK repair result\n\n" + repairs +
        "\n\nFull methods, provenance, missing-case lists and error details: [REPORT.md](REPORT.md), [validation.json](validation.json), [interactive report](index.html).\n", encoding="utf-8")
    category_counts = {k: Counter(failure_category(r) for r in rows.values() if not r["strict_valid"])
                       for k, rows in by_id.items() if k in RAW_MODELS}
    categories = sorted({c for counts in category_counts.values() for c in counts})
    findings = ["# Bixby50 v5.4 quality findings", "",
        f"R32 scores {available['checkpoint_r32']['mean']-available['checkpoint_r64']['mean']:.2f} points above R64 on the full supplied 50-case cohort. Both have large structural losses. This is an evaluation of the supplied predictions, not a new CUDA inference run or a checkpoint-selection recommendation.", "",
        "## Raw structural quality", "",
        table(["Population", "Cases", "Strict invalid / zero", "Strict valid but disconnected", "Strict valid and fully connected"],
              [[LABELS[k], available[k]["n"], available[k]["zero_score"],
                available[k]["strict_valid"]-available[k]["fully_reachable"], available[k]["fully_reachable"]] for k in RAW_MODELS]), "",
        "A defined component is only usable if the root can reach it through child references. The scorer penalizes a program that contains correct words in unattached components: reachability below 90% caps quality at 40/100; partial reachability otherwise caps it at 70/100. Invalid raw programs get zero under this scorer, including dangling child references. These caps explain many repeated 0, 40 and 70 scores; they are not arbitrary rounding.", "",
        "## Primary strict-validation failures", "",
        "Each invalid output is assigned one category from the first official validation error. Counts are diagnostic, not separate accuracy scores.", "",
        table(["Category", "R32 /50", "R64 /50", "MTP off /15", "MTP on /12"],
              [[c, *[category_counts[k].get(c,0) for k in RAW_MODELS]] for c in categories]), "",
        "R32 has 13 outputs that reach the recorded 2,048-token limit; R64 has 7. All of those outputs are invalid. Raising the budget would not directly fix the other malformed-reference, duplicate-ID or disconnected-graph cases. Among outputs that report stopping at the closing sentinel, 7 R32 and 14 R64 outputs are still strict-invalid.", "",
        "## Concrete examples", "",
        "- **BXP-002, AQI:** R32 scores 92.08; R64 and both native modes score 40.00. R64 defines 14 nodes but only 7 are reachable; the PM2.5 and outdoor-activity guidance nodes are disconnected. Both native modes define 10 nodes but only 5 are reachable, omitting the pollutant and guidance from the root's rendered content. Inspect the raw programs in `index.html#BXP-002`.",
        "- **BXP-008, R32:** raw output begins with the unrelated text ` Hungary` before `<a2ui>`, violating the strict completion envelope. R64 returns only `</a2ui>` for this case. R32 BXP-012 and R64 BXP-012 also contain the stray prefix. The report does not strip it to inflate the raw score.",
        "- **BXP-003, MTP comparison:** MTP off scores 99.01; MTP on scores 0 because element `j` references missing child `l`. This is a malformed graph despite the provider reporting `COMPLETED`.",
        "- **BXP-006:** MTP off scores 0 because element `m` references missing `q`; MTP on scores 89.26. MTP does not worsen every case.",
        "- **BXP-012, MTP on:** output contains an opening `<a2ui>` but no closing sentinel and scores 0. The saved provider label is `COMPLETED`, so that label cannot be used as proof of a complete DSL envelope or a correct EOS stop.", "",
        "## Matched native repair result", "",
        table(["Same BXP-001–012", "Raw mean /100", "After SDK 0.5.6 /100", "Strict valid after repair"],
              [[f"MTP {mode}", fmt(paired[f"litert_mtp_{mode}"]["mean"]),
                fmt(paired[f"sdk_mtp_{mode}"]["mean"]), f'{paired[f"sdk_mtp_{mode}"]["strict_valid"]}/{n}'] for mode in ("off","on")]), "",
        "Post-repair quality can reverse the raw ordering because repair reconnects or recovers different generated content. These repaired values are application-pipeline scores, not raw checkpoint/native model accuracy. They do not establish human visual quality or guarantee every source fact is preserved.", "",
        "## What can be reported", "",
        "Report the complete checkpoint means with n=50. For the quantized native comparison, report the matched raw 12-case means and explicitly label the run partial. The latest corrected native MTP-off sample scores above the corresponding R64 checkpoint subset, so these scores do not support a blanket claim that quantization reduced v5.4 quality. They also cannot certify export parity or extrapolate native quality to all 50 cases. The remaining native cases require a separately completed benchmark under the same recorded model/prompt/runtime conditions.", ""]
    # Keep the example score derived from the evaluated artifact, not hand-maintained.
    findings = [line.replace("R32 scores 92.08;", f'R32 scores {by_id["checkpoint_r32"]["BXP-002"]["score"]:.2f};') for line in findings]
    (out / "QUALITY_FINDINGS.md").write_text("\n".join(findings), encoding="utf-8")
    make_html(out, summary, by_id, android)


def make_html(out, summary, by_id, android):
    esc = html.escape
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bixby50 v5.4 scores</title>',
        '<style>body{font:15px/1.55 system-ui,sans-serif;margin:0;color:#17243d;background:#f5f7fb}main{max-width:1700px;margin:auto;padding:30px}h1{font-size:30px;margin:0 0 10px}h2{margin-top:28px}a{color:#195bc7}header,.case,.summary{background:white;border:1px solid #dce3ee;border-radius:14px;padding:22px;margin-bottom:18px}.note{background:#fff6df;padding:12px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{border:1px solid #dce3ee;border-radius:10px;padding:14px;min-width:0}.score{font-size:27px;font-weight:700}.small{font-size:12px;color:#526079}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.55 Consolas,monospace;background:#f5f7fb;padding:12px;border-radius:8px;max-height:560px;overflow:auto}summary{cursor:pointer;font-weight:600;padding:6px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;border-bottom:1px solid #e0e5ed;padding:8px}input{width:min(650px,90%);padding:12px;border:1px solid #9caac0;border-radius:8px}nav{display:flex;gap:15px;flex-wrap:wrap}.tag{border-radius:5px;background:#eef4ff;padding:4px 7px;display:inline-block}.bad{background:#ffeded}.na{color:#64748b}.repairs .grid{grid-template-columns:repeat(2,minmax(0,1fr))}@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.grid,.repairs .grid{grid-template-columns:1fr}main{padding:12px}}</style><main>',
        '<header><div class="small">BIXBY50 · SAVED OUTPUT EVALUATION · v5.4.0</div><h1>Checkpoint and LiteRT-LM quality</h1><p>Raw generation scores use the same source answers and current official scorer. Expand each case to inspect the actual output before repair.</p><p class="note"><b>Coverage:</b> R32 and R64 checkpoints: 50/50 each. Latest LiteRT MTP off: 15/50; on: 12/50. Only BXP-001–012 form the shared comparison. <b>V5.4 /100 is representation quality, not factual accuracy.</b></p><nav><a href="REPORT.md">Complete Markdown report</a><a href="CHECKPOINT_REPORT.md">Checkpoint report</a><a href="LITERT_MTP_REPORT.md">MTP report</a><a href="per_case_scores.csv">Per-case CSV</a><a href="input_manifest.json">Provenance</a></nav></header>',
        '<section class="summary"><h2>Same 12 cases · raw output</h2><div class="grid">']
    for k in RAW_MODELS:
        s = summary["paired"][k]
        parts.append(f'<div class="card"><b>{esc(LABELS[k])}</b><div class="score">{s["mean"]:.2f}<span class="small"> /100</span></div><div>{s["strict_valid"]}/{s["n"]} strict valid · {s["zero_score"]} zero scores</div></div>')
    parts += ['</div><h2>All available cases</h2><table><tr><th>Model</th><th>Cases</th><th>Mean /100</th><th>Strict valid</th></tr>']
    for k in RAW_MODELS:
        s = summary["all_available"][k]
        parts.append(f'<tr><td>{esc(LABELS[k])}</td><td>{s["n"]}/50</td><td>{s["mean"]:.2f}</td><td>{s["strict_valid"]}/{s["n"]}</td></tr>')
    parts += ['</table><p class="small">Different cohort sizes above: do not use these means as a direct export-gap comparison. All archived raw scores were individually checked against the new calculation; see the Markdown report.</p></section><p><label>Find a case or query<br><input id="filter" placeholder="BXP-001, trains, weather…"></label></p>']
    for case in sorted(android):
        q = android[case]["query"]
        parts.append(f'<article class="case" data-search="{esc((case+" "+q).lower(), quote=True)}" id="{case}"><h2>{case} · {esc(q)}</h2><details><summary>Captured Bixby source answer</summary><pre>{esc(android[case]["text"])}</pre></details><div class="grid">')
        for k in RAW_MODELS:
            if case not in by_id[k]:
                parts.append(f'<div class="card na"><b>{esc(LABELS[k])}</b><p>Not measured in this saved run</p></div>')
                continue
            r = by_id[k][case]
            status = "Strict valid" if r["strict_valid"] else "Strict invalid"
            klass = "tag" if r["strict_valid"] else "tag bad"
            diagnostics = {"parse_stage":r["breakdown"]["parse_stage"], "schema_error":r["metrics"].get("schema_error"),
                           "active_caps":r["breakdown"]["active_caps"], "errors":r["breakdown"]["errors"],
                           "dimensions":r["breakdown"]["dimensions"], "fidelity":r["breakdown"]["atomics"].get("fidelity"),
                           "graph":r["metrics"].get("graph_evidence_v5_4"), "finish_reason":r["finish_reason"]}
            parts.append(f'<div class="card"><b>{esc(LABELS[k])}</b><div class="score">{r["score"]:.2f}<span class="small"> /100</span></div><span class="{klass}">{status}</span><details><summary>Actual raw output</summary><pre>{esc(r["raw_output"])}</pre></details><details><summary>Why this score</summary><pre>{esc(json.dumps(diagnostics,ensure_ascii=False,indent=2))}</pre></details><p class="small">Output SHA-256: {r["raw_output_sha256"]}</p></div>')
        parts.append('</div>')
        if any(case in by_id[k] for k in ("sdk_mtp_off","sdk_mtp_on")):
            parts.append('<details class="repairs"><summary>SDK 0.5.6 post-processing appendix · not raw model accuracy</summary><div class="grid">')
            for mode in ("off","on"):
                k = f"sdk_mtp_{mode}"
                if case in by_id[k]:
                    r = by_id[k][case]
                    parts.append(f'<div class="card"><b>{esc(LABELS[k])}</b><p>Post-repair score: {r["score"]:.2f} /100 · strict valid: {r["strict_valid"]}</p><pre>{esc(r["raw_output"])}</pre></div>')
            parts.append('</div></details>')
        parts.append('</article>')
    parts += ['<footer><p>No new inference or human visual evaluation was performed. Source and prompt hashes, complete scorer breakdowns and missing-case lists accompany this report.</p></footer></main><script>document.getElementById("filter").addEventListener("input",e=>{let q=e.target.value.trim().toLowerCase();document.querySelectorAll(".case").forEach(c=>c.hidden=!c.dataset.search.includes(q))})</script></html>']
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
