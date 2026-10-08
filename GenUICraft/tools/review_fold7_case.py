#!/usr/bin/env python3
"""Collect one Fold7 case at a time; preserve every generation and replay attempt.

The script never builds, installs, deletes device files, or edits model/UI input.
Replay uses a final JSON/Express artifact selected by the caller. An after replay
defaults to the archived before bytes and rejects a different document.

Examples (run separately after inspecting each result):
  python GenUICraft/tools/review_fold7_case.py generate --case BXP-002 --serial R3CY30QFWLP
  python GenUICraft/tools/review_fold7_case.py replay --case BXP-001 --phase before --source FINAL_JSON
  python GenUICraft/tools/review_fold7_case.py replay --case BXP-001 --phase after
  python GenUICraft/tools/review_fold7_case.py annotate --case BXP-001 --status accepted --note "Reviewed all captures."
  python GenUICraft/tools/review_fold7_case.py report

Use --attempt r02 for an explicitly requested new attempt; existing evidence is
never skipped or overwritten. A device-wait timeout keeps a local operation lock
because instrumentation may still be running. Inspect the device before manually
removing that lock. No conversion or screenshot is automatically a visual pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import quote
import uuid

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "GenUICraft/validation/20261009_fold7_visual_review"
CORPUS = ROOT / "android/app/src/main/assets/genuicraft_bixby50.jsonl"
PACKAGE = "com.samsung.genuicraft"
RUNNER = f"{PACKAGE}.test/androidx.test.runner.AndroidJUnitRunner"
BENCHMARK = f"/sdcard/Android/data/{PACKAGE}/files/sdk_benchmark"
MODEL = f"/sdcard/Android/data/{PACKAGE}/files/sdk_models/model-fp16-corrected.litertlm"
MODEL_SHA256 = "7f01bdf1c6ba9bdf658e57c75001fc35a42edad88238bc5dab0a5f7dcf3de373"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def case_id(value):
    require(re.fullmatch(r"BXP-\d{3}", value) and 1 <= int(value[-3:]) <= 50,
            "--case must select exactly one BXP-001 through BXP-050")
    return value


def name(value):
    require(re.fullmatch(r"[A-Za-z0-9_-]{1,32}", value), "--attempt must be a simple name (1..32 characters)")
    return value


def remote_path(value):
    require(re.fullmatch(r"/[A-Za-z0-9_./~+=-]+", value), "Remote path contains unsupported shell characters")
    return value


def corpus_row(selected):
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == 50 and len({row["id"] for row in rows}) == 50, "Expected the unique frozen Bixby50 corpus")
    return next(row for row in rows if row["id"] == selected)


class DeviceMayStillRun(RuntimeError):
    pass


@contextmanager
def operation_lock(root, selected, action):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".active_device_operation.json"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"case": selected, "action": action, "startedAtUtc": utc()}, stream)
    except FileExistsError:
        raise ValueError(f"Another operation or an unresolved timeout is recorded in {lock}") from None
    retain = False
    try:
        yield
    except DeviceMayStillRun:
        retain = True
        raise
    finally:
        if not retain:
            lock.unlink()


class Adb:
    def __init__(self, executable, serial):
        require(re.fullmatch(r"[A-Za-z0-9._:-]+", serial), "Invalid explicit ADB serial")
        self.prefix = [executable, "-s", serial]

    def result(self, *parts, timeout=45):
        return subprocess.run(self.prefix + list(parts), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)

    def command(self, *parts, timeout=45):
        result = self.result(*parts, timeout=timeout)
        require(result.returncode == 0, f"ADB command failed ({result.returncode}): {result.stderr[-800:]}")
        return result.stdout.strip()

    def file_hash(self, path):
        output = self.command("shell", "sha256sum", remote_path(path), timeout=240)
        value = output.split()[0] if output else ""
        require(re.fullmatch(r"[0-9a-fA-F]{64}", value), f"Invalid device hash for {path}")
        return value.lower()

    def absent(self, path):
        result = self.result("shell", "test", "-e", remote_path(path))
        require(result.returncode == 1, f"Remote output must be new and accessible: {path}")


def provenance(adb, args, generation):
    model = adb.command("shell", "getprop", "ro.product.model")
    require(re.fullmatch(r"SM-F966[A-Za-z0-9]*", model), f"Selected device is not the expected Fold7: {model}")
    value = {"serial": args.serial, "deviceModel": model,
             "fingerprint": adb.command("shell", "getprop", "ro.build.fingerprint"),
             "corpusSha256": digest(CORPUS), "apks": {}}
    for package in (PACKAGE, PACKAGE + ".test"):
        paths = adb.command("shell", "pm", "path", package).splitlines()
        require(paths and all(line.startswith("package:") for line in paths), f"Missing installed package: {package}")
        value["apks"][package] = [{"path": line[8:], "sha256": adb.file_hash(line[8:])} for line in paths]
    renderer = ROOT / "GenUICraft/genuicraft/src/main/java/com/samsung/genuicraft/sdk/internal/renderer"
    files = sorted(renderer.rglob("*.kt"))
    fingerprint = "\n".join(f"{path.relative_to(renderer).as_posix()}:{digest(path)}" for path in files)
    value["hostRendererSource"] = {"files": len(files), "sha256": hashlib.sha256(fingerprint.encode()).hexdigest(),
                                    "note": "Host source fingerprint; installed APK hashes are recorded separately."}
    if generation:
        actual = adb.file_hash(args.model_path)
        value["model"] = {"path": args.model_path, "sha256": actual, "expectedSha256": args.expected_model_sha256,
                          "gpuPrecision": "FP16_CORRECTED", "mtp": True, "lineage": "pinned R64 corrected FP16"}
        require(actual == args.expected_model_sha256, "Selected model SHA-256 differs from the requested corrected package")
    return value


def instrument(adb, invocation, remote, target, timeout, receipt):
    log = target / "instrumentation.txt"
    started = time.monotonic()
    receipt.update({"command": adb.prefix + invocation, "remoteOutput": remote})
    with log.open("wb") as stream:
        process = subprocess.Popen(adb.prefix + invocation, stdout=stream, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            receipt["deviceInstrumentationMayStillBeRunning"] = True
            try:
                process.terminate()  # End only the local adb wait; do not stop the app/device.
                process.wait(timeout=15)
            except Exception as wait_error:
                receipt["localAdbTerminationError"] = str(wait_error)
            try:
                adb.command("pull", remote, str(target / "artifacts"), timeout=120)
            except Exception as pull_error:
                receipt["partialPullError"] = str(pull_error)
            raise DeviceMayStillRun("Local ADB wait ended; device instrumentation may still run. Evidence and operation lock retained.") from error
    receipt.update({"adbExitCode": process.returncode, "hostElapsedSeconds": round(time.monotonic() - started, 3)})
    adb.command("pull", remote, str(target / "artifacts"), timeout=120)
    text = log.read_text(encoding="utf-8", errors="replace")
    receipt["instrumentationPassed"] = (process.returncode == 0 and bool(re.search(r"OK \(1 test\)", text))
                                        and not re.search(r"FAILURES!!!|INSTRUMENTATION_FAILED|Process crashed", text))


def invocation(test, pairs):
    command = ["shell", "am", "instrument", "-w", "-r", "-e", "class", f"{PACKAGE}.{test}"]
    for key, value in pairs.items():
        command += ["-e", key, str(value)]
    return command + [RUNNER]


def latest_receipt(root, selected, phase):
    entries = []
    for path in (root / selected / phase).glob("*/receipt.json"):
        value = read(path)
        entries.append((value.get("startedAtUtc", ""), path, value))
    require(entries, f"No {phase} attempt for {selected}")
    _, path, value = max(entries, key=lambda entry: entry[0])
    return path, value


def choose_source(root, args):
    before = None
    if args.phase == "after":
        before_path, before = latest_receipt(root, args.case, "before")
        require(before.get("automatedChecksSatisfied") is True and before.get("status") == "collected",
                "The newest before attempt did not satisfy every check; inspect it before continuing")
    source = args.source.resolve() if args.source else (root / before["source"]["archivedPath"] if before else None)
    require(source is not None and source.is_file(), "Before replay requires --source pointing to an existing final artifact")
    require(source.suffix.lower() in (".json", ".express"), "--source must be a final .json or .express artifact")
    value = {"originalPath": str(source), "sha256": digest(source),
             "replayMode": "json" if source.suffix.lower() == ".json" else "express"}
    if before:
        require(value["sha256"] == before["source"]["sha256"] and value["replayMode"] == before["source"]["replayMode"],
                "After replay must use exactly the same accepted document bytes and format as before")
        value["pinnedBeforeReceipt"] = before_path.relative_to(root).as_posix()
        value["pinnedBeforeReceiptSha256"] = digest(before_path)
    adjacent = source.parent / "source.json"
    if adjacent.is_file():
        original = read(adjacent)
        row = corpus_row(args.case)
        require(original.get("id") == args.case and original.get("text") == row["text"],
                "Adjacent source.json does not match the selected frozen corpus case")
        value["sourceMetadata"] = {"path": str(adjacent), "sha256": digest(adjacent), "matchesFrozenCase": True}
    if args.source_info:
        value["callerSuppliedProvenance"] = read(args.source_info)
        value["callerSuppliedProvenanceSha256"] = digest(args.source_info)
    value["documentOrigin"] = "Caller-selected saved final output; model provenance is not inferred from its filename."
    return source, value


def collect(args):
    root = args.output.resolve()
    phase = "generation" if args.action == "generate" else args.phase
    target = root / args.case / phase / args.attempt
    require(not target.exists(), f"Preserve existing attempt: {target}; select a new --attempt")
    corpus_row(args.case)
    source_pair = choose_source(root, args) if args.action == "replay" else None
    adb = Adb(args.adb, args.serial)
    with operation_lock(root, args.case, args.action):
        target.mkdir(parents=True)
        run_id = f"visual_{args.case}_{phase}_{args.attempt}_{uuid.uuid4().hex[:8]}"
        remote = f"{BENCHMARK}/{run_id}"
        receipt = {"schemaVersion": 1, "case": args.case, "phase": phase, "attempt": args.attempt,
                   "startedAtUtc": utc(), "runId": run_id, "status": "running",
                   "automatedChecksSatisfied": False, "visualReviewStatus": "pending",
                   "inferenceAttempted": args.action == "generate"}
        save(target / "receipt.json", receipt)
        try:
            receipt["provenance"] = provenance(adb, args, args.action == "generate")
            adb.absent(remote)
            if args.action == "generate":
                command = invocation("GenUiTrainedBixby50Test#convertAndRenderTrainedBixby50", {
                    "runId": run_id, "cases": args.case, "backend": "GPU", "gpuPrecision": "FP16_CORRECTED",
                    "mtp": "true", "modelPath": args.model_path, "allowSourceTextFallback": "false",
                    "allowGeneratedDslRepair": "true", "requireSourceIntegrity": "false",
                    "caseTimeoutMs": args.case_timeout_ms,
                })
            else:
                source, receipt["source"] = source_pair
                filename = "output.a2ui.json" if receipt["source"]["replayMode"] == "json" else "output.express"
                staged = target / "input" / args.case / filename
                staged.parent.mkdir(parents=True)
                shutil.copyfile(source, staged)
                receipt["source"]["archivedPath"] = staged.relative_to(root).as_posix()
                require(digest(staged) == receipt["source"]["sha256"], "Staging changed source bytes")
                source_run = run_id + "_input"
                source_remote = f"{BENCHMARK}/{source_run}"
                adb.absent(source_remote)
                adb.command("shell", "mkdir", "-p", BENCHMARK)
                adb.command("push", str(target / "input"), source_remote, timeout=120)
                receipt["source"]["remotePath"] = f"{source_remote}/{args.case}/{filename}"
                receipt["source"]["remoteSha256"] = adb.file_hash(receipt["source"]["remotePath"])
                require(receipt["source"]["remoteSha256"] == receipt["source"]["sha256"], "Device staging changed source bytes")
                command = invocation("GenUiSdkBixby50Test#replaySavedBixbyCorpus", {
                    "sourceRunId": source_run, "runId": run_id, "cases": args.case,
                    "replayMode": receipt["source"]["replayMode"], "maxVerticalSwipes": args.max_vertical_swipes,
                    "maxHorizontalSwipes": args.max_horizontal_swipes, "renderFontScale": "1.0", "renderDark": "false",
                })
            instrument(adb, command, remote, target, args.timeout, receipt)
            artifacts = target / "artifacts"
            reports = read(artifacts / ("results.json" if args.action == "generate" else "replay_results.json"))
            require(len(reports) == 1 and reports[0].get("id") == args.case, "Instrumentation did not collect exactly the selected case")
            report = reports[0]
            config = read(artifacts / ("run_config.json" if args.action == "generate" else "replay_config.json"))
            summary = read(artifacts / ("summary.json" if args.action == "generate" else "replay_summary.json"))
            receipt.update({"result": report, "summary": summary})
            checks = {"instrumentationPassed": receipt.get("instrumentationPassed") is True,
                      "singleSelectedCase": config.get("cases") == [args.case],
                      "frozenCorpus": (config.get("corpus", {}).get("sha256") if args.action == "generate" else config.get("corpusSha256")) == digest(CORPUS)}
            if args.action == "generate":
                runtime = config.get("runtime", {})
                checks.update({"runComplete": summary.get("runComplete") is True and summary.get("completed") == 1,
                               "validGeneratedResult": report.get("status") == "valid" and report.get("strictValid") is True and report.get("renderValid") is True,
                               "oneProviderCall": report.get("providerCalls") == 1,
                               "correctedGpuMtpRuntime": "GPU+FP16_CORRECTED+MTP" in report.get("runtime", "") and
                                   runtime.get("accelerator") == "GPU" and runtime.get("gpuPrecision") == "FP16_CORRECTED" and runtime.get("mtpEnabled") is True,
                               "noSourceFallback": runtime.get("sourceFallbackEnabled") is False and report.get("usedFallback") is False,
                               "sourceIntegrityDiagnosticMode": runtime.get("requireSourceIntegrity") is False})
                receipt["sourceFidelityAudit"] = {"enforced": False, "status": "not_established",
                    "warnings": [warning for warning in report.get("warnings", []) if "fidelity" in warning.lower()],
                    "reportedWarningCount": report.get("sourceFidelityWarnings")}
                final = artifacts / args.case / "a2ui.json"
                if final.is_file():
                    receipt["finalArtifact"] = {"path": final.relative_to(root).as_posix(), "sha256": digest(final)}
                raw = artifacts / args.case / "output.express"
                if raw.is_file():
                    receipt["rawArtifact"] = {"path": raw.relative_to(root).as_posix(), "sha256": digest(raw)}
            else:
                key = "sourceJsonSha256" if receipt["source"]["replayMode"] == "json" else "sourceExpressSha256"
                captures = report.get("captures", [])
                checks.update({"noInference": summary.get("modelCalls") == 0 and config.get("inferenceEvaluated") is False,
                               "sameAcceptedInput": report.get(key) == receipt["source"]["sha256"],
                               "renderedWithoutIssues": report.get("status") == "rendered" and report.get("issues") == [],
                               "capturesComplete": bool(captures) and all(capture.get("screenshot") is True and
                                   (artifacts / args.case / (capture["name"] + ".png")).is_file() and
                                   (artifacts / args.case / (capture["name"] + ".xml")).is_file() for capture in captures),
                               "verticalEndObserved": report.get("verticalEndObserved") is True and report.get("verticalLimitReached") is False,
                               "allTableColumnsObserved": all(table.get("missingColumns") == [] for table in report.get("tables", []))})
            receipt["checks"] = checks
            receipt["documentHashes"] = {path.relative_to(root).as_posix(): digest(path)
                                         for path in sorted(artifacts.rglob("*")) if path.is_file() and path.suffix in (".json", ".express")}
            receipt["capturedFiles"] = [{"path": path.relative_to(root).as_posix(), "sha256": digest(path)}
                                        for path in sorted((artifacts / args.case).glob("*")) if path.suffix in (".png", ".xml")]
            receipt["automatedChecksSatisfied"] = all(value is True for value in checks.values())
            require(receipt["automatedChecksSatisfied"], "Checks did not all pass; inspect the receipt. The attempt is preserved.")
            receipt["status"] = "collected"
        except BaseException as error:
            receipt.update({"status": "failed", "error": str(error)})
            raise
        finally:
            receipt["finishedAtUtc"] = utc()
            save(target / "receipt.json", receipt)
            build_report(root)
    print(f"Collected {args.case} {phase}/{args.attempt}; visual review pending. Receipt: {target / 'receipt.json'}")


def build_report(root):
    root.mkdir(parents=True, exist_ok=True)

    def link(path, label):
        return f'<a href="{quote(path.relative_to(root).as_posix())}">{html.escape(label)}</a>'

    def passed(receipt):
        return receipt.get("automatedChecksSatisfied") is True and receipt.get("status") == "collected"

    def warnings(receipt):
        audit = receipt.get("sourceFidelityAudit") or {}
        result = receipt.get("result") or {}
        values = list(audit.get("warnings") or [])
        values += [value for value in result.get("warnings", []) if "fidelity" in str(value).lower()]
        if result.get("sourceIntegrityFailure"):
            values.append(result["sourceIntegrityFailure"])
        return list(dict.fromkeys(str(value) for value in values if value))

    def figure(image):
        image_url = quote(image.relative_to(root).as_posix())
        xml = image.with_suffix(".xml")
        return (f'<figure><a href="{image_url}" target="_blank" rel="noopener"><img loading="lazy" '
                f'src="{image_url}" alt="{html.escape(image.parent.name + " · " + image.stem)}"></a>'
                f'<figcaption>{link(image, image.name)} · {link(xml, "XML") if xml.exists() else "XML unavailable"}</figcaption></figure>')

    def attempt(path, receipt, featured=False):
        pictures = sorted((path.parent / "artifacts" / receipt["case"]).glob("*.png"))
        first = next((image for stem in ("initial", "screen", "page_0", "page_00", "page_01")
                      for image in pictures if image.stem == stem), pictures[0] if pictures else None)
        badge = "Checks passed" if passed(receipt) else ("Checks pending" if receipt["status"] == "running" else "Checks need attention")
        failed_checks = [key for key, value in receipt.get("checks", {}).items() if value is not True]
        issues = list((receipt.get("result") or {}).get("issues", []))
        if receipt.get("error"):
            issues.insert(0, receipt["error"])
        if failed_checks:
            issues.append("Checks requiring attention: " + ", ".join(failed_checks))
        issue_html = ('<ul class="issues">' + "".join(f'<li>{html.escape(str(value))}</li>' for value in issues) + '</ul>') if issues else ""
        log = path.parent / "instrumentation.txt"
        metadata = (f'<p class="attempt-meta">{link(path, "Receipt, checks and provenance")} · '
                    f'{link(log, "Instrumentation log") if log.exists() else "Instrumentation did not start"}</p>')
        orphan_xml = [link(xml, xml.name) for xml in sorted((path.parent / "artifacts" / receipt["case"]).glob("*.xml"))
                      if not xml.with_suffix(".png").is_file()]
        fidelity = warnings(receipt)
        fidelity_html = ('<div class="fidelity"><strong>Recorded fidelity warnings</strong><ul>' +
                        "".join(f'<li>{html.escape(value)}</li>' for value in fidelity) + '</ul></div>') if fidelity else ""
        all_captures = ('<div class="captures">' + "".join(figure(image) for image in pictures) + '</div>') if pictures else '<p class="empty">No captures collected.</p>'
        body = metadata + issue_html + fidelity_html
        if featured:
            body += figure(first) if first else '<p class="empty">No captures collected.</p>'
            if len(pictures) > 1:
                body += f'<details class="capture-pages"><summary>Review all {len(pictures)} capture pages</summary>{all_captures}</details>'
        else:
            body += all_captures
        if orphan_xml:
            body += '<p class="attempt-meta">XML without a screenshot: ' + " · ".join(orphan_xml) + '</p>'
        title = html.escape(receipt["phase"].capitalize() + " · " + receipt["attempt"])
        state = f'<span class="badge {"check-pass" if passed(receipt) else "check-pending"}">{badge}</span>'
        if featured:
            return f'<article class="attempt"><header><h3>{title}</h3>{state}</header>{body}</article>'
        return f'<details class="past-attempt"><summary>{title} · {html.escape(receipt["status"])} · {badge}</summary>{body}</details>'

    cards = []
    index = []
    navigation = []
    domains = set()
    visual_counts = {"accepted": 0, "pending": 0, "needs_work": 0}
    check_counts = {phase: {"passed": 0, "attention": 0, "pending": 0} for phase in ("before", "after")}
    for directory in sorted(root.glob("BXP-???")):
        if not directory.is_dir():
            continue
        selected = case_id(directory.name)
        row = corpus_row(selected)
        review = read(directory / "review.json") if (directory / "review.json").is_file() else {"status": "pending", "notes": []}
        if review.get("status") == "accepted":
            after_path, after = latest_receipt(root, selected, "after")
            current = (review.get("reviewedAfterReceipt") == after_path.relative_to(root).as_posix() and
                       review.get("reviewedAfterReceiptSha256") == digest(after_path) and
                       after.get("automatedChecksSatisfied") is True and after.get("status") == "collected")
            if not current:
                review = {**review, "status": "pending", "previousVisualStatus": "accepted",
                          "pendingReason": "The newest after attempt has not received the referenced visual review."}
        runs = []
        entries = []
        order = {"before": 0, "after": 1, "generation": 2}
        for path in sorted(directory.glob("*/*/receipt.json"), key=lambda item: (order.get(item.parent.parent.name, 3), item.parent.name)):
            receipt = read(path)
            relative = path.relative_to(root).as_posix()
            runs.append({"receipt": relative, "phase": receipt["phase"], "attempt": receipt["attempt"],
                         "status": receipt["status"], "automatedChecksSatisfied": receipt["automatedChecksSatisfied"]})
            entries.append((path, receipt))
        latest = {}
        for phase in ("before", "after", "generation"):
            phase_entries = [entry for entry in entries if entry[1]["phase"] == phase]
            if phase_entries:
                latest[phase] = max(phase_entries, key=lambda entry: entry[1].get("startedAtUtc", ""))
        for phase in check_counts:
            receipt = latest.get(phase, (None, None))[1]
            check_counts[phase]["pending" if receipt is None or receipt.get("status") == "running" else
                                "passed" if passed(receipt) else "attention"] += 1
        status = review["status"]
        visual_counts[status] = visual_counts.get(status, 0) + 1
        domain = str(row.get("domain", "Unspecified"))
        domains.add(domain)
        notes = "".join(f'<p>{html.escape(str(note))}</p>' for note in review.get("notes", [])) or '<p>No visual review notes recorded.</p>'
        pending = f'<p class="issues">{html.escape(review["pendingReason"])}</p>' if review.get("pendingReason") else ""
        featured = []
        for phase in ("before", "after"):
            featured.append(attempt(*latest[phase], featured=True) if phase in latest else
                            f'<article class="attempt empty"><h3>{phase.capitalize()}</h3><p>No {phase} attempt collected.</p></article>')
        shown = {entry[0] for phase, entry in latest.items() if phase in ("before", "after")}
        history = [(path, receipt) for path, receipt in entries if path not in shown]
        history_html = (f'<details class="history"><summary>Previous attempts and generation evidence · {len(history)}</summary>' +
                        "".join(attempt(path, receipt) for path, receipt in history) + '</details>') if history else ""
        current_warnings = {}
        for phase, (_, receipt) in latest.items():
            for warning in warnings(receipt):
                current_warnings.setdefault(warning, []).append(phase + "/" + receipt["attempt"])
        fidelity_html = ('<div class="fidelity"><strong>Latest recorded fidelity warnings</strong><ul>' +
                         "".join(f'<li>{html.escape(warning)} <span class="attempt-meta">({html.escape(", ".join(origins))})</span></li>'
                                 for warning, origins in current_warnings.items()) + '</ul></div>') if current_warnings else ""
        document_note = ""
        if "before" in latest and "after" in latest:
            before_hash = latest["before"][1].get("source", {}).get("sha256")
            after_hash = latest["after"][1].get("source", {}).get("sha256")
            if before_hash and after_hash:
                document_note = ('<p class="document-note">Same saved document in both replays.</p>' if before_hash == after_hash else
                                 '<p class="issues">Before and after document hashes differ; inspect the receipts.</p>')
        badge_text = {"accepted": "Visually accepted", "pending": "Visual review pending", "needs_work": "Needs work"}.get(status, status)
        search_text = " ".join((selected, domain, row["query"], " ".join(str(value) for value in review.get("notes", [])))).lower()
        attrs = f'data-case="{selected}" data-domain="{html.escape(domain, quote=True)}" data-status="{html.escape(status, quote=True)}" data-search="{html.escape(search_text, quote=True)}"'
        navigation.append(f'<li><a class="case-link" href="#case-{selected}" data-case="{selected}">{selected}<span>{html.escape(domain)}</span></a></li>')
        cards.append(f'<section class="case-card" id="case-{selected}" {attrs}><header class="case-heading">'
                     f'<div><p class="eyebrow">{html.escape(domain)}</p><h2>{selected}</h2></div><span class="badge {html.escape(status, quote=True)}">{html.escape(badge_text)}</span></header>'
                     f'<div class="review-notes"><h3>Review notes</h3>{notes}{pending}</div>'
                     '<p class="scope">Visual acceptance and automated checks do not certify source fidelity. Recorded fidelity warnings appear with their attempts.</p>'
                     f'{fidelity_html}{document_note}<div class="comparison">{"".join(featured)}</div>{history_html}'
                     f'<details class="frozen-source"><summary>Frozen source query and answer</summary><p>{html.escape(row["query"])}</p><pre>{html.escape(row["text"])}</pre></details></section>')
        index.append({"case": selected, "visualReview": review, "runs": runs})
    save(root / "review_index.json", {"schemaVersion": 1, "generatedAtUtc": utc(), "cases": index,
         "note": "Automated collection/coverage checks do not establish visual acceptance or source fidelity."})
    metrics = (f'<div><strong>{len(index)}</strong><span>Recorded cases</span></div>'
               f'<div><strong>{visual_counts["accepted"]}</strong><span>Visually accepted</span></div>'
               f'<div><strong>{visual_counts["pending"]}</strong><span>Visual review pending</span></div>'
               f'<div><strong>{visual_counts["needs_work"]}</strong><span>Need work</span></div>')
    for phase, counts in check_counts.items():
        metrics += (f'<div><strong>{counts["passed"]}</strong><span>{phase.capitalize()} checks passed</span>'
                    f'<small>{counts["attention"]} need attention · {counts["pending"]} pending or absent</small></div>')
    domain_options = "".join(f'<option value="{html.escape(domain, quote=True)}">{html.escape(domain)}</option>' for domain in sorted(domains))
    css = """
      :root{color-scheme:light;--ink:#172334;--muted:#5a6879;--line:#dfe5ec;--blue:#205a9d}
      *{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
      main{max-width:1400px;margin:auto;padding:28px 24px 56px}h1,h2,h3,p{margin:0}h1{font-size:clamp(25px,3vw,36px);letter-spacing:-.6px}
      h2{font-size:25px}h3{font-size:15px}a{color:var(--blue);text-underline-offset:3px}a:focus-visible,input:focus-visible,select:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #79a8df;outline-offset:3px}
      .intro>p{color:var(--muted);max-width:900px;margin-top:8px}.intro .eyebrow{margin-bottom:4px}.eyebrow{text-transform:uppercase;letter-spacing:1px;font-size:11px;font-weight:700;color:var(--muted)}
      .metrics{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:22px 0}.metrics>div{background:white;border:1px solid var(--line);border-radius:12px;padding:13px 15px;display:flex;flex-direction:column;gap:1px}
      .metrics strong{font-size:27px;line-height:1.2}.metrics span{font-size:12px;color:var(--muted)}.metrics small{font-size:10px;color:var(--muted);margin-top:4px}
      .toolbar{display:flex;gap:12px;align-items:end;flex-wrap:wrap;background:white;border:1px solid var(--line);padding:16px;border-radius:14px}.toolbar label{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:600}.search{flex:1;min-width:200px}
      input,select,button{font:inherit;border:1px solid #cbd5e1;border-radius:8px;background:white;color:var(--ink);padding:8px 10px;min-height:40px}button{cursor:pointer}.result-count{font-size:12px;color:var(--muted);margin:10px 0}
      .case-nav{list-style:none;display:flex;gap:8px;flex-wrap:wrap;padding:0;margin:0 0 24px}.case-link{text-decoration:none;display:flex;align-items:center;gap:7px;background:white;border:1px solid var(--line);padding:6px 10px;border-radius:8px;font-weight:600;font-size:12px}.case-link span{font-weight:400;color:var(--muted);font-size:11px}.case-link[aria-current]{border-color:var(--blue);background:#eef4fc}
      .case-card{background:white;border:1px solid var(--line);border-radius:18px;padding:20px;margin:0 0 24px;scroll-margin-top:16px}.case-heading,.attempt>header{display:flex;justify-content:space-between;gap:12px;align-items:center}.case-heading{margin-bottom:16px}.badge{font-size:11px;font-weight:600;padding:5px 9px;border:1px solid var(--line);border-radius:999px;white-space:nowrap;color:var(--muted);background:#f6f8fa}
      .accepted,.check-pass{color:#245e42;background:#edf7f0;border-color:#cee5d6}.needs_work{color:#855a22;background:#fff6e8;border-color:#ead7b7}.review-notes{background:#f6f8fb;border-radius:10px;padding:13px 15px}.review-notes p{margin-top:6px;white-space:pre-wrap;overflow-wrap:anywhere}.scope{font-size:12px;color:var(--muted);margin:12px 0}.document-note{font-size:12px;color:var(--muted);margin:0 0 10px}
      .comparison{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.attempt{min-width:0;background:#fafbfd;border:1px solid var(--line);border-radius:12px;padding:14px}.attempt-meta{font-size:11px;color:var(--muted);margin:9px 0;overflow-wrap:anywhere}
      figure{margin:0 auto;min-width:0;max-width:360px}figure>a{display:block}img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:7px;background:white}figcaption{font-size:11px;text-align:center;padding:6px 0;color:var(--muted)}
      .captures{display:flex;gap:12px;overflow-x:auto;align-items:flex-start;padding:4px 0 9px;scroll-snap-type:x proximity}.captures figure{flex:0 0 min(280px,100%);scroll-snap-align:start;margin:0}.captures img{border-radius:5px}
      details{margin-top:12px}summary{cursor:pointer;font-size:12px;font-weight:600;padding:6px 0}.history,.frozen-source{border-top:1px solid var(--line);padding-top:5px}.past-attempt{padding:8px 12px;background:#f8fafc;border:1px solid var(--line);border-radius:9px}.frozen-source p{margin:8px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:12px;font:13px/1.6 inherit}
      .issues,.fidelity{font-size:12px;background:#fff7ed;color:#704e22;border-radius:8px;padding:10px 14px;margin:10px 0;overflow-wrap:anywhere}.issues li,.fidelity li{margin:3px 0}.empty{color:var(--muted);padding:22px 14px}.no-results{padding:28px;text-align:center;color:var(--muted)}[hidden]{display:none!important}
      @media(max-width:1000px){.metrics{grid-template-columns:repeat(3,minmax(0,1fr))}}
      @media(max-width:680px){main{padding:20px 12px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.metrics>div{padding:10px 12px}.comparison{grid-template-columns:1fr}.case-card{padding:14px}.case-heading{align-items:flex-start}.badge{white-space:normal}.toolbar{gap:10px;padding:12px}.toolbar label{flex:1;min-width:130px}.toolbar .search{flex-basis:100%}.case-link span{display:none}figure{max-width:320px}}
    """
    script = """
      const search = document.getElementById('case-search');
      const domain = document.getElementById('domain-filter');
      const status = document.getElementById('status-filter');
      const cards = Array.from(document.querySelectorAll('.case-card'));
      const links = Array.from(document.querySelectorAll('.case-link'));
      function filterCases() {
        const term = search.value.trim().toLowerCase();
        let visible = 0;
        cards.forEach(card => {
          const show = (!term || card.dataset.search.includes(term)) &&
            (!domain.value || card.dataset.domain === domain.value) &&
            (!status.value || card.dataset.status === status.value);
          card.hidden = !show;
          if (show) visible++;
          const link = links.find(link => link.dataset.case === card.dataset.case);
          if (link) link.parentElement.hidden = !show;
        });
        document.getElementById('result-count').textContent = visible + ' of ' + cards.length + ' recorded cases shown';
        document.getElementById('no-results').hidden = visible !== 0;
      }
      function markCurrentCase() {
        links.forEach(link => {
          if (link.hash === location.hash) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      }
      search.addEventListener('input', filterCases);
      domain.addEventListener('change', filterCases);
      status.addEventListener('change', filterCases);
      document.getElementById('clear-filters').addEventListener('click', () => {
        search.value = ''; domain.value = ''; status.value = ''; filterCases();
      });
      window.addEventListener('hashchange', markCurrentCase);
      filterCases(); markCurrentCase();
    """
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Fold7 sequential visual review</title><style>' + css + '</style></head><body><main>'
            '<header class="intro"><p class="eyebrow">GenUICraft · Fold7</p><h1>Sequential visual review</h1>'
            '<p>Latest saved before and after captures for each recorded Bixby50 case. Visual acceptance requires an explicit review of the referenced captures.</p>'
            '<p>Before/after counts below are collection and coverage checks, not quality scores. <a href="review_index.json">Review index JSON</a></p></header>'
            f'<div class="metrics" aria-label="Recorded review summary">{metrics}</div>'
            '<div class="toolbar"><label class="search" for="case-search">Search cases<input id="case-search" type="search" placeholder="Case ID, domain, query or review notes"></label>'
            f'<label for="domain-filter">Domain<select id="domain-filter"><option value="">All domains</option>{domain_options}</select></label>'
            '<label for="status-filter">Visual review<select id="status-filter"><option value="">All statuses</option><option value="accepted">Accepted</option><option value="pending">Pending</option><option value="needs_work">Needs work</option></select></label>'
            '<button id="clear-filters" type="button">Clear filters</button></div>'
            f'<p id="result-count" class="result-count" role="status">{len(index)} recorded cases shown</p><nav aria-label="Recorded cases"><ol class="case-nav">{"".join(navigation)}</ol></nav>'
            '<p id="no-results" class="no-results" hidden>No recorded cases match these filters.</p>' +
            ("".join(cards) or '<p class="empty">No case evidence collected.</p>') + '<script>' + script + '</script></main></body></html>')
    (root / "index.html").write_text(page, encoding="utf-8")


def annotate(args):
    root = args.output.resolve()
    directory = root / args.case
    require(directory.is_dir(), "Collect evidence before adding a review")
    review = {"status": args.status, "notes": args.note or [], "reviewer": args.reviewer, "updatedAtUtc": utc()}
    if args.notes_file:
        review["notes"].append(args.notes_file.read_text(encoding="utf-8-sig"))
    if args.status == "accepted":
        path, receipt = latest_receipt(root, args.case, "after")
        require(receipt.get("automatedChecksSatisfied") is True and receipt.get("status") == "collected",
                "The newest after attempt did not satisfy every check; it cannot receive an accepted visual review")
        require(bool(review["notes"]), "An explicit accepted visual review requires notes describing what was checked")
        review["reviewedAfterReceipt"] = path.relative_to(root).as_posix()
        review["reviewedAfterReceiptSha256"] = digest(path)
        review["sourceSha256"] = receipt["source"]["sha256"]
        review["scope"] = "Explicit visual review of the referenced after captures; source fidelity is not certified."
    save(directory / "review.json", review)
    build_report(root)
    print(f"Recorded visual review {args.status} for {args.case}")


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest="action", required=True)
    for action in ("generate", "replay", "annotate", "report"):
        command = commands.add_parser(action)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        if action == "report":
            continue
        command.add_argument("--case", required=True, type=case_id)
        if action == "annotate":
            command.add_argument("--status", required=True, choices=("pending", "needs_work", "accepted"))
            command.add_argument("--note", action="append")
            command.add_argument("--notes-file", type=Path)
            command.add_argument("--reviewer", default="codex")
            continue
        command.add_argument("--serial", default="R3CY30QFWLP")
        command.add_argument("--adb", default=shutil.which("adb") or "adb")
        command.add_argument("--attempt", type=name, default="r01")
        command.add_argument("--timeout", type=int, default=480)
        if action == "generate":
            command.add_argument("--model-path", type=remote_path, default=MODEL)
            command.add_argument("--expected-model-sha256", default=MODEL_SHA256)
            command.add_argument("--case-timeout-ms", type=int, default=360_000)
        else:
            command.add_argument("--phase", required=True, choices=("before", "after"))
            command.add_argument("--source", type=Path)
            command.add_argument("--source-info", type=Path)
            command.add_argument("--max-vertical-swipes", type=int, default=30)
            command.add_argument("--max-horizontal-swipes", type=int, default=8)
    return result


def main():
    args = parser().parse_args()
    try:
        if args.action == "report":
            build_report(args.output.resolve())
            print(args.output.resolve() / "index.html")
        elif args.action == "annotate":
            annotate(args)
        else:
            require(30 <= args.timeout <= 7200, "--timeout must be between 30 and 7200 seconds")
            if args.action == "generate":
                require(re.fullmatch(r"[0-9a-f]{64}", args.expected_model_sha256), "Expected model hash must be lowercase SHA-256")
                require(1000 <= args.case_timeout_ms <= 3_600_000, "Invalid --case-timeout-ms")
                require(args.timeout >= args.case_timeout_ms / 1000 + 30, "Host timeout must exceed case timeout by at least 30 seconds")
            else:
                require(1 <= args.max_vertical_swipes <= 30 and 1 <= args.max_horizontal_swipes <= 16, "Invalid replay swipe limit")
            collect(args)
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
