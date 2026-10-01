"""Score actual SDK repair outcomes beside immutable Bixby50 raw predictions.

Run GenUiRepairReplay against the released AAR first. This command never repairs
text itself, does not call a model, and scores only the SDK's returned Express.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import html
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import report_bixby50_v54_models as base

MODELS = base.RAW_MODELS


def summarize(rows):
    successful = [r for r in rows if r["repair_success"]]
    return {
        "n": len(rows), "ids": [r["id"] for r in rows],
        "raw_mean": statistics.mean(r["raw_score"] for r in rows),
        "repaired_mean_failures_zero": statistics.mean(r["repaired_score_for_aggregate"] for r in rows),
        "repaired_median_failures_zero": statistics.median(r["repaired_score_for_aggregate"] for r in rows),
        "repaired_success_only_mean": statistics.mean(r["repaired_score"] for r in successful) if successful else None,
        "raw_strict_valid": sum(r["raw_strict_valid"] for r in rows),
        "repair_successes": len(successful), "repair_failures": len(rows) - len(successful),
        "repaired_strict_valid": sum(r["repaired_strict_valid"] is True for r in rows),
        "fully_reachable_after_repair": sum(r["repaired_metrics"]["fully_root_reachable_v5_4"] for r in successful),
        "repair_kinds": dict(Counter(r["repair_kind"] for r in rows)),
        "improved": sum(r["score_delta"] > 1e-9 for r in rows),
        "unchanged": sum(abs(r["score_delta"]) <= 1e-9 for r in rows),
        "worsened": sum(r["score_delta"] < -1e-9 for r in rows),
    }


def write_html(out, populations, summary):
    e = html.escape
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bixby50 scores after SDK repair</title>',
        '<style>body{font:15px/1.55 system-ui,sans-serif;background:#f5f7fb;color:#14243e;margin:0}main{max-width:1680px;margin:auto;padding:26px}header,section,article{background:white;border:1px solid #dce4ee;border-radius:14px;padding:20px;margin-bottom:18px}h1{font-size:30px;margin:5px 0}h2{font-size:20px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{border:1px solid #dce4ee;border-radius:9px;padding:13px;min-width:0}.score{font-size:24px;font-weight:700}.small{font-size:12px;color:#52637a;overflow-wrap:anywhere}.note{padding:12px;background:#fff5da;border-radius:8px}pre{font:12px/1.5 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:550px;overflow:auto;background:#f4f7fb;padding:12px}summary{cursor:pointer;padding:6px 0;font-weight:600}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #dce4ee}a{color:#165bc4}nav{display:flex;gap:16px;flex-wrap:wrap}.bad{color:#a01f2d}input{padding:11px;width:min(630px,90%);border:1px solid #9aa9bb;border-radius:8px}@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:640px){.grid{grid-template-columns:1fr}main{padding:10px}}</style><main>',
        '<header><div class="small">BIXBY50 · GenUICraft SDK 0.5.6 · v5.4.0</div><h1>Quality after the same SDK repair</h1><p>All 127 available raw outputs were processed by the released AAR. The repair receives generated text only. No source-text fallback, new inference or model-based repair.</p><p class="note"><b>Checkpoint coverage:</b> 50 cases each. <b>LiteRT coverage:</b> MTP off 15/50; MTP on 12/50. Native full-50 scores remain unmeasured. Repair failures contribute zero to their completed-case averages. V5.4 measures source-to-UI representation quality, not factual-answer accuracy.</p><nav><a href="REPORT.md">Full Markdown report</a><a href="per_case_scores.csv">Per-case CSV</a><a href="summary.json">Summary JSON</a><a href="repair_failures.md">Repair failures</a><a href="repair_provenance.json">AAR provenance</a></nav></header>',
        '<section><h2>Identical BXP-001–012 cases</h2><div class="grid">']
    for model in MODELS:
        s = summary["paired"][model]
        parts.append(f'<div class="card"><b>{e(base.LABELS[model])}</b><p class="score">{s["raw_mean"]:.2f} → {s["repaired_mean_failures_zero"]:.2f}</p><p class="small">Raw → after repair /100<br>{s["repair_successes"]}/{s["n"]} SDK repair accepted · failures stay in the mean</p></div>')
    parts += ['</div><h2>All available cases</h2><table><tr><th>Population</th><th>Cases</th><th>Raw</th><th>After repair</th><th>SDK accepted</th></tr>']
    for model in MODELS:
        s = summary["all_available"][model]
        parts.append(f'<tr><td>{e(base.LABELS[model])}</td><td>{s["n"]}/50</td><td>{s["raw_mean"]:.2f}</td><td>{s["repaired_mean_failures_zero"]:.2f}</td><td>{s["repair_successes"]}/{s["n"]}</td></tr>')
    parts += ['</table><p class="small">The rows above have different cohort sizes. Use the identical 12-case table to compare checkpoint and LiteRT or MTP modes.</p></section><p><label>Find a case or query<br><input id="filter" placeholder="BXP-001, trains, weather…"></label></p>']
    for case in sorted(populations[MODELS[0]]):
        ref = populations[MODELS[0]][case]
        parts.append(f'<article id="{case}" data-search="{e((case+" "+ref["query"]).lower(),quote=True)}"><h2>{case} · {e(ref["query"])}</h2><details><summary>Source answer used for scoring only</summary><pre>{e(ref["response_text"])}</pre></details><div class="grid">')
        for model in MODELS:
            if case not in populations[model]:
                parts.append(f'<div class="card"><b>{e(base.LABELS[model])}</b><p>Native output not measured.</p></div>')
                continue
            r = populations[model][case]
            tag = r["repair_kind"] if r["repair_success"] else "SDK repair rejected · aggregate score 0"
            parts.append(f'<div class="card"><b>{e(base.LABELS[model])}</b><p class="score">{r["raw_score"]:.2f} → {r["repaired_score_for_aggregate"]:.2f}</p><p class="small">{e(tag)}</p><details><summary>Actual raw output</summary><pre>{e(r["raw_output"])}</pre></details>')
            if r["repair_success"]:
                parts.append(f'<details><summary>SDK repaired output</summary><pre>{e(r["repaired_express"])}</pre></details>')
            else:
                parts.append(f'<p class="bad">No repaired artifact was returned. No source fallback was substituted.</p>')
            diagnostics = {"repair_success":r["repair_success"],"repair_kind":r["repair_kind"],
                           "repair_error":r.get("repair_error"),"diagnostics":r.get("diagnostics"),
                           "strict_valid_after_repair":r["repaired_strict_valid"],
                           "score_caps":r["repaired_breakdown"]["active_caps"] if r["repair_success"] else None,
                           "fidelity_atomics":r["repaired_metrics"]["fidelity_atomics_v5_4"] if r["repair_success"] else None}
            parts.append(f'<details><summary>Repair and score diagnostics</summary><pre>{e(json.dumps(diagnostics,ensure_ascii=False,indent=2))}</pre></details></div>')
        parts.append('</div></article>')
    parts += ['<p class="small">This run verifies SDK compilation and offline v5.4 scores. No new device rendering or screenshot-based quality review was performed.</p></main><script>document.getElementById("filter").addEventListener("input",e=>{let q=e.target.value.trim().toLowerCase();document.querySelectorAll("article").forEach(c=>c.hidden=!c.dataset.search.includes(q))})</script></html>']
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def write_reports(out, populations, summary):
    all_stats, paired = summary["all_available"], summary["paired"]
    full_table = base.table(["Population", "Cases", "Raw v5.4", "After SDK repair", "SDK accepted", "Python strict valid"],
        [[base.LABELS[k],f'{s["n"]}/50',base.fmt(s["raw_mean"]),base.fmt(s["repaired_mean_failures_zero"]),f'{s["repair_successes"]}/{s["n"]}',f'{s["repaired_strict_valid"]}/{s["n"]}'] for k,s in all_stats.items()])
    paired_table = base.table(["Population", "Same cases", "Raw v5.4", "After SDK repair", "Repair failures"],
        [[base.LABELS[k],s["n"],base.fmt(s["raw_mean"]),base.fmt(s["repaired_mean_failures_zero"]),s["repair_failures"]] for k,s in paired.items()])
    kinds = base.table(["Population", "NONE", "STRUCTURAL", "GENERATED_DSL_REPAIR", "REJECTED", "Score improved / same / worse"],
        [[base.LABELS[k],*[s["repair_kinds"].get(t,0) for t in ("NONE","STRUCTURAL","GENERATED_DSL_REPAIR","REJECTED")],f'{s["improved"]} / {s["unchanged"]} / {s["worsened"]}'] for k,s in all_stats.items()])
    failure_rows = [r for rows in populations.values() for r in rows.values() if not r["repair_success"]]
    failure_table = base.table(["Population", "Case", "Raw score", "After repair", "Recorded generation stop", "Raw characters"],
        [[base.LABELS[r["model"]],r["id"],base.fmt(r["raw_score"]),"0 (no artifact)",r["raw_finish_reason"],len(r["raw_output"])] for r in failure_rows])
    identity = summary["metric_identity"]
    lines = ["# Bixby50 v5.4 scores after GenUICraft SDK repair", "", "Generated: " + summary["created_at_utc"], "",
        "Applied the same **published GenUICraft 0.5.6 AAR** to every available raw prediction: 50 R32 checkpoints, 50 R64 checkpoints, 15 LiteRT MTP-off outputs, and 12 LiteRT MTP-on outputs. **119/127** returned repaired/compiled artifacts. All **27/27** native results exactly match the earlier on-device repaired Express bytes.", "",
        "These are **post-processing pipeline scores**, not raw model accuracy. V5.4 is an uncalibrated 0–100 source-to-UI representation-quality score; it does not measure factual-answer accuracy or human visual quality.", "",
        "## All available outputs", "", full_table, "",
        "Repair failures remain in the denominator and contribute zero because the pipeline returned no usable artifact. Missing native cases are **unmeasured**, not zero. The 15-case and 12-case native means cannot be compared directly with full 50-case checkpoint means.", "",
        "## Fair comparison on the same 12 cases", "", "Cases: BXP-001 through BXP-012.", "", paired_table, "",
        f'On these same cases, after-repair MTP-on minus MTP-off is **{paired["litert_mtp_on"]["repaired_mean_failures_zero"]-paired["litert_mtp_off"]["repaired_mean_failures_zero"]:+.2f} points**. The repair stage can change the ordering seen in raw scores. This is a single saved generation per case/mode, and 38 native paired cases remain unmeasured.', "",
        "## Exact repair policy", "", "```java", "GenUiCompiler.compileWithRepair(rawOutput, null, false, true)", "```", "",
        "- The repair receives only generated output. `sourceText=null`, source fallback is disabled, and generated-DSL recovery is enabled.",
        "- The source answer is supplied only to the official scorer, after repair. No model, external API, source reconstruction, hand-edited IR or model-specific fix is used.",
        "- The library runs strict compilation first, then its existing syntax and generated-content/graph recovery. `NONE` can still return canonicalized Express; therefore string changes alone do not imply a content repair.",
        "- All outputs, including previously valid outputs, go through the same call so disconnected generated graphs can be recovered consistently.",
        "- Returned Express is recompiled with the SDK and scored unchanged using `score_prediction(source, None, repairedExpress, metric_version='v5_4')`. Its score is cross-checked against `generation_reward_v5_4`.",
        "- For a failed repair there is no invented empty candidate: `repaired_score=null` in JSON, with `repaired_score_for_aggregate=0` under the explicit pipeline-failure policy.",
        "- The original raw report and predictions are unchanged. Full generated and repaired text, errors, cap evidence and content-fidelity diagnostics are available in `scored/*.jsonl` and `index.html`.", "",
        "## Repair outcomes", "", kinds, "",
        "`NONE` means the SDK accepted the generated document without recovery; `STRUCTURAL` and `GENERATED_DSL_REPAIR` identify its repair routes. The scores are computed for the final returned Express, regardless of route.", "",
        "## Failed repairs", "", failure_table, "",
        "R64 BXP-008 contains only `</a2ui>`, with no generated UI content. The other seven failed inputs reached their recorded 2,048-token output limit and were incomplete. The current SDK returned no valid generated-only candidate for these cases. This does not prove that every fragment is unrecoverable; it records the behavior of this exact library version without inventing missing content. See [repair_failures.md](repair_failures.md) for every error and raw ending.", "",
        "## Remaining limits after successful repair", "",
        "Compiling a repaired UI does not restore facts that were never generated, prove complete source coverage, or certify a good screenshot. A high repaired score must therefore remain labeled as a source-grounded engineering score. This run did not perform new inference, device rendering or visual review. The JVM repair path was verified against all 27 earlier Android repairs by exact Express equality; checkpoint outputs have SDK compile validation and official Python scoring here, not fresh on-device screenshots.", "",
        base.table(["Population", "Fully reachable after repair", "Repair accepted", "Successful-output-only mean (diagnostic)"],
            [[base.LABELS[k],f'{s["fully_reachable_after_repair"]}/{s["n"]}',f'{s["repair_successes"]}/{s["n"]}',base.fmt(s["repaired_success_only_mean"])] for k,s in all_stats.items()]), "",
        "The successful-output-only mean excludes failures and is supplied only as a diagnostic. Use the all-attempt mean above for reporting pipeline performance.", "",
        "## Provenance and validation", "",
        "AAR SHA-256: `c59eeae4debf4e8428a806ac7b7d7cb1b13f2d3559a12a655a56ef2f0be723b1`.", "",
        "LiteRT native generations are the saved corrected R64 QAT-compatible W4 / GPU FP32 runs, with MTP selected by batch. Native model SHA-256: `de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`. Repair on the host does not alter those generation/runtime labels.", "",
        f'Metric version: `{identity["metric_version"]}`.  \nMetric fingerprint: `{identity["metric_fingerprint"]}`.  \nReward pipeline fingerprint: `{identity["reward_pipeline_fingerprint"]}`.', "",
        "The raw and repaired scores use the same scorer identity and per-case source contract. Source hashes are checked against the prior report. Java runs only the published AAR and its Gson/Kotlin dependencies; the driver catches candidate validation errors while letting missing classes or runtime failures abort the run.", "",
        "Files: [interactive report](index.html), [per-case CSV](per_case_scores.csv), [50-case matrix](bixby50_matrix.csv), [summary](summary.json), [repair outcomes](repair_outcomes.jsonl), [AAR/runtime provenance](repair_provenance.json), [device parity checks](device_repair_parity.json), and `scored/*.jsonl`.", "",
        "## Per-case score matrix", "", "Each cell is **raw → after repair**. N/A means the native output is unavailable. Failed repair has a reported pipeline score of 0.", "",
        base.table(["Case", "R32 checkpoint", "R64 checkpoint", "LiteRT MTP off", "LiteRT MTP on"],
            [[case,*[f'{rows[case]["raw_score"]:.2f} → {rows[case]["repaired_score_for_aggregate"]:.2f}' if case in rows else "N/A" for rows in populations.values()]] for case in sorted(populations[MODELS[0]])]), ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    fail = ["# SDK repair failures", "", "These remain in the reporting denominator with pipeline score zero. No source text was passed to the repair API.", "", failure_table, ""]
    for r in failure_rows:
        fail += [f'## {base.LABELS[r["model"]]} — {r["id"]}', "", r["query"], "", "SDK diagnostic:", "", "```text", r["repair_error"], "```", "", "Raw beginning:", "", "```text",r["raw_output"][:600],"```", "", "Raw ending:", "", "```text",r["raw_output"][-900:],"```", ""]
    (out / "repair_failures.md").write_text("\n".join(fail), encoding="utf-8")
    write_html(out, populations, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", type=Path, default=base.ROOT / "GenUICraft/validation/20260929_bixby50_v54_model_scores")
    parser.add_argument("--repair-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.repair_dir.resolve()
    if (out / "summary.json").exists():
        raise SystemExit("Existing scored report found; preserve it and choose a new run directory.")
    outcome_list = base.read_jsonl(out / "repair_outcomes.jsonl")
    outcomes = {(r["model"],r["id"]):r for r in outcome_list}
    assert len(outcomes) == len(outcome_list) == 127
    prior_summary = base.read_json(args.raw_report / "summary.json")
    parity = base.read_json(out / "device_repair_parity.json")
    assert len(parity) == 27 and all(r["exact_express_match"] for r in parity)
    (out / "scored").mkdir()
    populations = {}
    input_hashes = {}
    source_files = [args.raw_report / "summary.json", out / "repair_outcomes.jsonl",out / "repair_provenance.json",out / "device_repair_parity.json",Path(__file__),base.ROOT / "GenUICraft/tools/GenUiRepairReplay.java"]
    for model in MODELS:
        raw_path = args.raw_report / f"scored/{model}.jsonl"
        source_files.append(raw_path)
        rows = base.read_jsonl(raw_path)
        populations[model] = {}
        for raw in rows:
            assert base.sha_text(raw["raw_output"]) == raw["raw_output_sha256"]
            assert base.sha_text(raw["response_text"]) == raw["source_sha256"]
            result = dict(outcomes[(model,raw["id"])])
            assert result["model_calls"] == 0 and not result["source_text_supplied_to_repair"] and not result["source_fallback_enabled"]
            result.update({"query":raw["query"],"response_text":raw["response_text"],"source_sha256":raw["source_sha256"],
                "raw_output":raw["raw_output"],"raw_output_sha256":raw["raw_output_sha256"],
                "raw_score":raw["score"],"raw_strict_valid":raw["strict_valid"],"raw_finish_reason":raw["finish_reason"]})
            if result["repair_success"]:
                express = result["repaired_express"]
                metrics = base.score_prediction(raw["response_text"],None,express,metric_version="v5_4")
                breakdown = dataclasses.asdict(base.generation_reward_v5_4(express,raw["response_text"]))
                assert metrics["generation_reward_v5_4"] == breakdown["quality_0_100"]
                for key in ("metric_version","metric_fingerprint","reward_pipeline_fingerprint","expected_contract_hash"):
                    assert metrics["metric_identity_v5_4"][key] == raw["metrics"]["metric_identity_v5_4"][key]
                result.update({"repaired_score":metrics["generation_reward_v5_4"],"repaired_metrics":metrics,
                    "repaired_breakdown":breakdown,"repaired_strict_valid":metrics["schema_valid_strict"],
                    "repaired_express_sha256":base.sha_text(express),"repaired_a2ui_sha256":base.sha_text(result["repaired_a2ui_json"])})
                if model.startswith("litert_"):
                    mode = model.removeprefix("litert_mtp_")
                    previous = next(r for r in base.read_jsonl(args.raw_report / f"scored/sdk_mtp_{mode}.jsonl") if r["id"] == raw["id"])
                    assert express == previous["raw_output"] and result["repaired_score"] == previous["score"]
            else:
                result.update({"repaired_score":None,"repaired_metrics":None,"repaired_breakdown":None,"repaired_strict_valid":None})
            result["repaired_score_for_aggregate"] = result["repaired_score"] if result["repair_success"] else 0.0
            result["score_delta"] = result["repaired_score_for_aggregate"] - result["raw_score"]
            populations[model][raw["id"]] = result
            if len(populations[model]) % 10 == 0:
                print(f'Scored repaired {model}: {len(populations[model])}/{len(rows)}',flush=True)
        with (out / f"scored/{model}.jsonl").open("w",encoding="utf-8",newline="\n") as f:
            for r in populations[model].values():
                f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
    assert sum(map(len,populations.values())) == 127
    paired_ids = sorted(set.intersection(*(set(rows) for rows in populations.values())))
    assert paired_ids == prior_summary["paired_ids"]
    summary = {"created_at_utc":datetime.now(timezone.utc).isoformat(),"metric_identity":prior_summary["metric_identity"],
        "repair_library":"GenUICraft 0.5.6","repair_api":"GenUiCompiler.compileWithRepair(raw, null, false, true)",
        "scope":"SDK post-processing quality; no model inference or source fallback", "failed_repair_aggregate_policy":"zero; retained in denominator",
        "all_available":{k:summarize(list(rows.values())) for k,rows in populations.items()},"paired_ids":paired_ids,
        "paired":{k:summarize([rows[i] for i in paired_ids]) for k,rows in populations.items()}}
    base.save_json(out / "summary.json",summary)
    for p in source_files:
        input_hashes[str(p.resolve())] = base.sha_file(p)
    base.save_json(out / "scoring_input_hashes.json",input_hashes)
    cols = ["model","id","raw_score","repaired_score","repaired_score_for_aggregate","score_delta","raw_strict_valid","repaired_strict_valid","repair_success","repair_kind","raw_finish_reason","raw_output_sha256","repaired_express_sha256","repair_error"]
    with (out / "per_case_scores.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for rows in populations.values():
            w.writerows({k:r.get(k) for k in cols} for r in rows.values())
    with (out / "bixby50_matrix.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["id",*[f"{k}_{stage}" for k in MODELS for stage in ("raw","repaired")]])
        for case in sorted(populations[MODELS[0]]):
            w.writerow([case,*[v for rows in populations.values() for v in ([rows[case]["raw_score"],rows[case]["repaired_score_for_aggregate"]] if case in rows else ["",""])]])
    write_reports(out,populations,summary)
    print(json.dumps({"all_available":summary["all_available"],"paired_means":{k:s["repaired_mean_failures_zero"] for k,s in summary["paired"].items()}},ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
