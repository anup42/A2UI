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
    cards = []
    index = []
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
        blocks = []
        order = {"before": 0, "after": 1, "generation": 2}
        for path in sorted(directory.glob("*/*/receipt.json"), key=lambda item: (order.get(item.parent.parent.name, 3), item.parent.name)):
            receipt = read(path)
            relative = path.relative_to(root).as_posix()
            runs.append({"receipt": relative, "phase": receipt["phase"], "attempt": receipt["attempt"],
                         "status": receipt["status"], "automatedChecksSatisfied": receipt["automatedChecksSatisfied"]})
            pictures = []
            for image in sorted((path.parent / "artifacts" / selected).glob("*.png")):
                image_url = quote(image.relative_to(root).as_posix())
                xml = image.with_suffix(".xml")
                pictures.append(f'<figure><a href="{image_url}"><img loading="lazy" src="{image_url}" alt="{html.escape(image.stem)}"></a>'
                                f'<figcaption>{link(image, image.name)} {link(xml, "XML") if xml.exists() else ""}</figcaption></figure>')
            orphan_xml = [link(xml, xml.name) for xml in sorted((path.parent / "artifacts" / selected).glob("*.xml"))
                          if not xml.with_suffix(".png").is_file()]
            issue = html.escape(receipt.get("error", ""))
            log = path.parent / "instrumentation.txt"
            audit = receipt.get("sourceFidelityAudit")
            fidelity = (f'<details><summary>Source fidelity: not established; reported warnings {audit.get("reportedWarningCount")}</summary>'
                        f'<pre>{html.escape(chr(10).join(audit["warnings"]))}</pre></details>') if audit else ""
            blocks.append(f'<details open><summary>{html.escape(receipt["phase"] + "/" + receipt["attempt"])} — '
                          f'{html.escape(receipt["status"])}; automated checks {receipt["automatedChecksSatisfied"]}</summary>'
                          f'<p>{link(path, "Receipt, checks and provenance")} · {link(log, "Instrumentation log") if log.exists() else "Instrumentation did not start."}</p>'
                          f'<p>{issue}</p>{fidelity}<div class="captures">{"".join(pictures) or "No captures collected."}</div><p>{" · ".join(orphan_xml)}</p></details>')
        notes = "\n".join(review.get("notes", []))
        cards.append(f'<section><h2>{selected} · {html.escape(row.get("domain", ""))}</h2>'
                     f'<p>Visual review: <strong>{html.escape(review["status"])}</strong></p><pre>{html.escape(notes)}</pre>'
                     f'<details><summary>Frozen source query and answer</summary><p>{html.escape(row["query"])}</p><pre>{html.escape(row["text"])}</pre></details>'
                     f'{"".join(blocks)}</section>')
        index.append({"case": selected, "visualReview": review, "runs": runs})
    save(root / "review_index.json", {"schemaVersion": 1, "generatedAtUtc": utc(), "cases": index,
         "note": "Automated collection/coverage checks do not establish visual acceptance or source fidelity."})
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Fold7 sequential visual review</title>'
            '<style>body{font:15px system-ui;max-width:1500px;margin:24px auto;padding:0 20px;color:#172033}section{border-top:1px solid #ccc;padding:20px 0}'
            '.captures{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0;max-width:280px}img{width:260px;height:auto;border:1px solid #ddd}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:12px}summary{cursor:pointer;margin:12px 0}a{color:#1759ad}</style>'
            '<h1>Fold7 sequential visual review</h1><p>Each operation selects one frozen Bixby50 case. '
            'Saved captures and automated checks remain pending until explicitly reviewed. Source fidelity is reported separately.</p>'
            '<p><a href="review_index.json">Review index JSON</a></p>' + ("".join(cards) or '<p>No case evidence collected.</p>') + '</html>')
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
