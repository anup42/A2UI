"""Refresh the fixed-50 progress report from saved evidence only; no ADB/build calls."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, os, re

REVIEW = Path(__file__).resolve().parent
WORKSPACE = REVIEW.parents[2]
CORPUS = WORKSPACE / "android/app/src/main/assets/genuicraft_bixby50.jsonl"
_spec = importlib.util.spec_from_file_location("fold7_receipt_checks", WORKSPACE / "GenUICraft/tools/review_fold7_case.py")
_receipt_checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_receipt_checks)

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def rel(path):
    return Path(os.path.relpath(path.resolve(), REVIEW)).as_posix()

def repo(path):
    return path.resolve().relative_to(WORKSPACE).as_posix()

def saved(path):
    return REVIEW / path

def latest(selected, phase):
    paths = list((REVIEW / selected / phase).glob("*/receipt.json"))
    return max(((read(path).get("startedAtUtc", ""), path, read(path)) for path in paths), default=None, key=lambda value: value[0])

def select_saved_source(receipt, prepared, repair_raw):
    source = (receipt or {}).get("source", {})
    if source.get("replayMode") == _receipt_checks.EXPRESS_REPAIR_ONLY:
        # An explicit repair request must never fall back to an unrelated final.
        return next((item for item in repair_raw if item["sha256"] == source.get("sha256")), None)
    selected = next((item for item in prepared if item["sha256"] == source.get("sha256")), None)
    choices = [item for item in prepared if item["format"] == "json"]
    return selected or (choices[0] if len(choices) == 1 else None)


def verified_repaired_output(receipt):
    if not receipt or receipt.get("source", {}).get("replayMode") != _receipt_checks.EXPRESS_REPAIR_ONLY or \
            not _receipt_checks.receipt_express_repair_complete(REVIEW, receipt):
        return None
    artifact = receipt["repairArtifacts"]["recoveredJson"]
    return {"path": artifact["path"], "sha256": artifact["sha256"], "format": "json",
            "repairKind": receipt["result"].get("repairKind"), "rawSha256": receipt["source"]["sha256"],
            "scope": "SDK-produced repaired canonical document. Table-cell coverage refers to this output; raw-text completeness and source fidelity are not certified."}


def update(output):
    manifest = read(REVIEW / "native_generation/manifest.json")
    findings = read(REVIEW / "review_findings.json").get("cases", {})
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 50 and len({row["id"] for row in rows}) == 50
    inputs = {row["id"]: [] for row in rows}
    generation = {row["id"]: [] for row in rows}
    repair_raw = {row["id"]: [] for row in rows}
    issues = []
    def add(selected, path, digest, origin, format):
        if not path.is_file() or sha(path) != digest:
            issues.append({"case": selected, "problem": "Prepared source file missing or hash changed", "path": rel(path)})
            return
        inputs[selected].append({"path": rel(path), "sha256": digest, "format": format, "origin": origin})
    def add_repair_raw(item):
        if not item.get("raw"): return
        expected = Path(item["runId"]) / item["id"] / "output.express"
        path = REVIEW / "native_generation" / item["raw"]
        digest = item.get("rawSha256")
        if item["raw"] != expected.as_posix() or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or not path.is_file() or sha(path) != digest:
            issues.append({"case": item["id"], "problem": "Manifest-pinned raw Express path/hash is invalid", "path": item["raw"]})
            return
        repair_raw[item["id"]].append({"path": rel(path), "sha256": digest, "format": _receipt_checks.EXPRESS_REPAIR_ONLY,
            "origin": item["runId"], "scope": "Pinned raw generation artifact; not a prepared or renderable canonical final."})
    for item in manifest["cases"]:
        add_repair_raw(item)
        selected = item["id"]
        generation[selected].append({"runId": item["runId"], "status": item.get("recordedGenerationStatus"), "cause": item.get("cause"), "result": "native_generation/" + item["result"] if item.get("result") else None})
        if item.get("final"):
            final = REVIEW / "native_generation" / item["final"]
            add(selected, final, item["finalSha256"], item["runId"], "json")
            result = read(REVIEW / "native_generation" / item["result"])
            if result.get("rawStrictValid") is True and result.get("repairKind") == "NONE" and item.get("raw"):
                add(selected, REVIEW / "native_generation" / item["raw"], item["rawSha256"], item["runId"], "express")
    for item in manifest["priorValidationReferences"]["cases"]:
        final = (REVIEW / "native_generation" / item["final"]).resolve()
        add(item["id"], final, item["finalSha256"], "October 8 selected streaming-on document", "json")
        expression = final.with_name("final.express")
        if expression.is_file():
            add(item["id"], expression, sha(expression), "October 8 selected streaming-on document", "express")
    records = []
    counts = Counter()
    for row in rows:
        selected = row["id"]
        review_path = REVIEW / selected / "review.json"
        review = read(review_path) if review_path.is_file() else {"status": "pending", "notes": []}
        before, after = latest(selected, "before"), latest(selected, "after")
        checks = {}
        declared = review.get("status", "pending")
        notes = [str(note).strip() for note in review.get("notes", []) if str(note).strip()]
        active = after or before
        active_receipt = active[2] if active else None
        repair_only = bool(active_receipt and active_receipt.get("source", {}).get("replayMode") == _receipt_checks.EXPRESS_REPAIR_ONLY)
        selected_input = select_saved_source(active_receipt, inputs[selected], repair_raw[selected])
        repaired_output = verified_repaired_output(active_receipt) if repair_only and selected_input else None
        if repair_only and selected_input is None:
            issues.append({"case": selected, "problem": "Repair-only receipt has no matching manifest-pinned raw source"})
        baseline_valid = bool(before and after and _receipt_checks.pinned_before_evidence(REVIEW, after[2], before[1], before[2]))
        baseline_override = after[2].get("source", {}).get("beforeRenderFailureOverride") if after and baseline_valid else None
        if declared == "accepted":
            checks["nonemptyExplicitNotes"] = bool(notes)
            checks["latestAfterReceiptReferenced"] = bool(after and review.get("reviewedAfterReceipt") == rel(after[1]))
            checks["reviewReceiptHashMatches"] = bool(after and review.get("reviewedAfterReceiptSha256") == sha(after[1]))
            checks["afterCollectionComplete"] = bool(after and after[2].get("status") == "collected" and after[2].get("automatedChecksSatisfied") is True)
            source = after[2].get("source", {}) if after else {}
            checks["reviewSourceHashMatches"] = bool(source.get("sha256") and review.get("sourceSha256") == source.get("sha256"))
            checks["sameBeforeAfterInput"] = baseline_valid
            archived = saved(source["archivedPath"]) if source.get("archivedPath") else None
            checks["archivedInputHashMatches"] = bool(archived and archived.is_file() and sha(archived) == source.get("sha256"))
            if repair_only:
                checks["sourceMatchesManifestPinnedRaw"] = bool(selected_input and selected_input["sha256"] == source.get("sha256"))
                checks["recoveredDocumentHashesMatch"] = bool(repaired_output)
            else:
                checks["sourceMatchesPreparedCurrentModelArtifact"] = bool(selected_input and selected_input["sha256"] == source.get("sha256"))
            files = after[2].get("capturedFiles", []) if after else []
            checks["captureFilesAndHashesMatch"] = bool(files) and all(saved(item["path"]).is_file() and sha(saved(item["path"])) == item["sha256"] for item in files)
        accepted = declared == "accepted" and all(checks.values()) and bool(checks)
        available = bool(inputs[selected])
        visual = "accepted" if accepted else "needs_work" if declared == "needs_work" else "pending" if available else "not_reviewable"
        if accepted: bucket = "visually_accepted"
        elif available: bucket = "renderable_needs_work" if visual == "needs_work" else "renderable_unreviewed"
        elif generation[selected]: bucket = "no_renderable_output"
        else: bucket = "generation_pending"
        counts[bucket] += 1
        if declared == "accepted" and not accepted:
            issues.append({"case": selected, "problem": "Accepted annotation lacks current evidence", "failedChecks": [key for key, value in checks.items() if value is not True]})
        candidate = selected_input if repair_only else (selected_input or next((item for item in inputs[selected] if item["format"] == "json"), None))
        if not active and len([item for item in inputs[selected] if item["format"] == "json"]) > 1:
            candidate = None
            issues.append({"case": selected, "problem": "Multiple prepared documents need an explicit selected source"})
        records.append({"id": selected, "domain": row["domain"], "bucket": bucket,
            "generationAvailability": "prepared_final" if available else "no_renderable_output" if generation[selected] else "generation_pending",
            "visualStatus": visual, "declaredVisualStatus": declared, "review": rel(review_path), "reviewFileExists": review_path.is_file(),
            "reviewNotes": notes, "acceptanceChecks": checks, "beforeReceipt": rel(before[1]) if before else None,
            "afterReceipt": rel(after[1]) if after else None, "selectedSource": candidate,
            "preparedInputs": inputs[selected], "generationAttempts": generation[selected],
            "rawRepairInputs": repair_raw[selected], "repairedOutput": repaired_output,
            "repairComparison": active_receipt.get("repairComparison") if repair_only else None,
            "findings": findings.get(selected, {}), "sourceFidelityCertified": False,
            "beforeRenderFailureBaseline": baseline_override})
    names = ("visually_accepted", "renderable_needs_work", "renderable_unreviewed", "no_renderable_output", "generation_pending")
    totals = {key: counts[key] for key in names}
    assert sum(totals.values()) == 50
    snapshot = {"schemaVersion": 1, "generatedAtUtc": datetime.now(timezone.utc).isoformat(), "denominator": 50,
        "counts": totals, "availableCurrentModelDocuments": sum(bool(inputs[row["id"]]) for row in rows),
        "casesWithVerifiedRepairOutputs": sum(bool(record["repairedOutput"]) for record in records),
        "reviewedCasesWithExplicitModelFindings": sum(record["visualStatus"] == "accepted" and bool(record["findings"].get("modelFindings")) for record in records),
        "noRenderableOutputCaseIds": [record["id"] for record in records if record["bucket"] == "no_renderable_output"],
        "evidenceIssues": issues, "cases": records,
        "expectedBeforeRenderFailureCases": [record["id"] for record in records if record["beforeRenderFailureBaseline"]],
        "scope": "Fixed-50 rendering progress. Explicit notes and matching capture evidence are required for acceptance. Prepared/native-render-valid outputs and fidelity warning counts do not certify visual quality or model/source correctness.",
        "refresh": repo(Path(__file__)), "inputs": {"generationManifest": "native_generation/manifest.json", "explicitFindings": "review_findings.json", "reviewPaths": "BXP-xxx/review.json"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "progress.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Fold7 visual review progress", "", f"Snapshot: {snapshot['generatedAtUtc']}. Denominator: **50**.", "", "| Status | Cases |", "|---|---:|"]
    lines += [f"| {name.replace('_', ' ')} | {totals[name]} |" for name in names]
    lines += ["", f"Current-model renderable documents: **{snapshot['availableCurrentModelDocuments']}/50**. This is preparation coverage, not visual acceptance.", "", "Current cases without renderer input: " + (", ".join(snapshot["noRenderableOutputCaseIds"]) or "none") + ". Guard-cutoff records are not labelled model hallucinations.", "", "Acceptance covers rendering of the selected document. Model/source gaps remain separate. Fidelity warning counts are diagnostic, especially for state-bound tables.", "", "Confirmed model findings are recorded only from explicit reviewer observations in [review_findings.json](review_findings.json); all original notes remain linked below. The historical train-fixture hash failure is a separate validation finding.", "", "| Case | Domain | Visual status | Current evidence |", "|---|---|---|",]
    for record in records:
        links = f"[Review]({record['review']})" if record["reviewFileExists"] else "Not reviewed"
        if record["selectedSource"]:
            label = "Pinned raw Express" if record["selectedSource"]["format"] == _receipt_checks.EXPRESS_REPAIR_ONLY else "Prepared document"
            links += f" · [{label}]({record['selectedSource']['path']})"
        elif record["generationAttempts"] and record["generationAttempts"][0]["result"]: links += f" · [Generation result]({record['generationAttempts'][0]['result']})"
        if record["repairedOutput"]: links += f" · [Repaired JSON]({record['repairedOutput']['path']})"
        if record["beforeRenderFailureBaseline"]:
            links += " · Preserved failed before baseline: " + ", ".join(record["beforeRenderFailureBaseline"]["overriddenChecks"])
        lines.append(f"| {record['id']} | {record['domain']} | {record['visualStatus'].replace('_', ' ')} | {links} |")
    lines += ["", "Explicit before-render waivers preserve the failed baseline receipt and source hashes. Only the two named render/column checks may be waived for comparison; every after check must pass. These baseline diagnostics are not marked as passing checks or stale accepted evidence."]
    lines += ["", "Refresh after later reviews with:", "", "```powershell", "python GenUICraft/validation/20261009_fold7_visual_review/refresh_progress.py", "```", "", "The script reads existing reviews/receipts/native-generation inventory and writes only these progress reports and the source listing. It performs no model inference, device action, SDK build, or edits to review annotations/generated UI.", ""]
    lines += ["", "Prepared input counts above retain the canonical/strict-Express inventory. Manifest-pinned damaged raw is separate and selectable only by an explicit express_repair_only receipt. Verified replay-repaired outputs are linked separately; complete cell coverage covers the repaired graph, not all raw text or source facts."]
    (output / "PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")
    listing = ["# Prepared source mapping: BXP-009 onward", "", "Paths are relative to this review folder. Use the selected source for before/after; five cutoffs require new SDK generation when their turn arrives. Historical R32 outputs are never substituted.", "", "| Case | Prepared input |", "|---|---|"]
    for record in records[8:]:
        source = record["selectedSource"]
        value = f"[{source['path']}]({source['path']})" if source else "**No renderable output — app guard cutoff**" if record["bucket"] == "no_renderable_output" else "Explicit source selection required"
        if source and source["format"] == _receipt_checks.EXPRESS_REPAIR_ONLY:
            value = f"Pinned raw Express: [{source['path']}]({source['path']})"
            repaired = record["repairedOutput"]
            value += f" · SDK-repaired JSON: [{repaired['path']}]({repaired['path']})" if repaired else " · No verified repaired output"
        listing.append(f"| {record['id']} | {value} |")
    (output / "SOURCE_MAP.md").write_text("\n".join(listing) + "\n", encoding="utf-8")
    return snapshot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    result = update(args.output_dir.resolve())
    print(json.dumps({"denominator": 50, "counts": result["counts"], "evidenceIssues": len(result["evidenceIssues"])}))
