#!/usr/bin/env python3
"""Read host files only; record visual-renderer source/artifact provenance against accepted v10.

Run --source-only while compiling. Run without it only after final SDK and consumer builds
complete. This collector does not build, install, or access a device. Optional installed-hash
comparison consumes only an existing device-validation record supplied by the run owner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


SDK = Path(__file__).resolve().parents[1]
MAIN = "genuicraft/src/main/"
JAVA = MAIN + "java/com/samsung/genuicraft/sdk/"
EXPECTED_CHANGED = {
    JAVA + "GenUiContent.kt",
    JAVA + "GenUiView.kt",
    JAVA + "internal/theme/RendererTheme.kt",
    JAVA + "internal/renderer/FlatSpecRenderer.kt",
    JAVA + "internal/renderer/FlatTableLayout.kt",
    JAVA + "internal/renderer/flat/compose/FlatPrimitiveComposables.kt",
    JAVA + "internal/renderer/flat/domain/FlatItineraryDomain.kt",
    JAVA + "internal/renderer/native/intents/weather/NativeWeatherUiRenderer.kt",
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest(path),
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_report(baseline_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    current = {path.relative_to(SDK).as_posix(): digest(path)
               for path in sorted((SDK / MAIN).rglob("*")) if path.is_file()}
    common = baseline.keys() & current.keys()
    changed = sorted(path for path in common if baseline[path].lower() != current[path])
    added, removed = sorted(current.keys() - baseline.keys()), sorted(baseline.keys() - current.keys())
    valid = (len(baseline) == len(current) == 106 and not added and not removed
             and set(changed) == EXPECTED_CHANGED)
    return {
        "schema": "genuicraft_visual_source_delta_v1", "baseline": file_record(baseline_path),
        "baseline_count": len(baseline), "current_count": len(current),
        "unchanged_count": len(common) - len(changed), "changed_count": len(changed),
        "changed": [{"path": path, "v10_sha256": baseline[path], "current_sha256": current[path]}
                    for path in changed],
        "added": added, "removed": removed,
        "unexpected_changes": sorted(set(changed) - EXPECTED_CHANGED),
        "expected_changes_not_present": sorted(EXPECTED_CHANGED - set(changed)),
        "expected_visual_only_delta_verified": valid,
        "provider_compiler_converter_bindings_prompts_unchanged": valid,
        "current_source_sha256": current,
    }


def archive_members(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP members: {path}")
        selected = [name for name in names if name.endswith(".so") or
                    name.startswith("assets/genuicraft/prompts/")]
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(selected)}


def jvm_report(directory: Path, expected_count: int) -> dict:
    files = sorted(directory.glob("TEST-*.xml"))
    totals = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
    suites = []
    for path in files:
        suite = ET.parse(path).getroot()
        if suite.tag != "testsuite":
            raise ValueError(f"Expected testsuite XML: {path}")
        counts = {key: int(suite.attrib.get(key, 0)) for key in totals}
        if counts["tests"] != len(suite.findall("testcase")):
            raise ValueError(f"Test count does not match testcase nodes: {path}")
        for key, value in counts.items():
            totals[key] += value
        suites.append({"name": suite.attrib["name"], **counts, **file_record(path)})
    return {**totals, "suite_count": len(suites), "suites": suites,
            "expected_test_count": expected_count,
            "verified": bool(files) and totals == {"tests": expected_count, "failures": 0,
                                                     "errors": 0, "skipped": 0}}


def publication_report(directory: Path, aar_sha: str) -> dict:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    records, checksum_errors = [], []
    for path in files:
        records.append({"relative_path": path.relative_to(directory).as_posix(), **file_record(path)})
        algorithm = path.suffix.lstrip(".")
        if algorithm in {"md5", "sha1", "sha256", "sha512"}:
            target = path.with_suffix("")
            if not target.is_file() or path.read_text().strip().lower() != digest(target, algorithm):
                checksum_errors.append(path.relative_to(directory).as_posix())
    published_aars = [path for path in files if path.suffix == ".aar"]
    same_aar = len(published_aars) == 1 and digest(published_aars[0]) == aar_sha
    return {"root": str(directory.resolve()), "file_count": len(files), "files": records,
            "checksum_errors": checksum_errors, "published_aar_matches_build": same_aar,
            "verified": bool(files) and not checksum_errors and same_aar}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--output", type=Path, default=SDK / "validation/20260918_visual")
    parser.add_argument("--aar", type=Path, default=SDK / "genuicraft/build/outputs/aar/genuicraft-release.aar")
    parser.add_argument("--apk", type=Path, default=SDK.parent / "android/app/build/outputs/apk/debug/app-debug.apk")
    parser.add_argument("--test-apk", type=Path,
                        default=SDK.parent / "android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk")
    parser.add_argument("--jvm-results", type=Path, default=SDK / "genuicraft/build/test-results/testDebugUnitTest")
    parser.add_argument("--expected-tests", type=int, default=293)
    parser.add_argument("--maven-root", type=Path, default=SDK / "build/repo")
    parser.add_argument("--installed-apk-evidence", type=Path,
                        help="Optional existing adb sha256sum output supplied by the device-validation owner")
    args = parser.parse_args()
    source = source_report(SDK / "validation/20260918/v10_sdk_source_hashes.json")
    source["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "source_delta.json", source)
    if not source["expected_visual_only_delta_verified"]:
        raise SystemExit("Source delta differs from the eight expected visual files; see source_delta.json")
    if args.source_only:
        print(f"Verified 106 production files: {source['changed_count']} visual changes, "
              f"{source['unchanged_count']} unchanged; {args.output / 'source_delta.json'}")
        return
    baseline_path = SDK / "validation/20260918/v10_build_manifest.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    artifacts = {name: file_record(path) for name, path in
                 (("aar", args.aar), ("apk", args.apk), ("test_apk", args.test_apk))}
    aar_members, apk_members = archive_members(args.aar), archive_members(args.apk)
    aar_native = {name: sha for name, sha in aar_members.items() if name.endswith(".so")}
    apk_native = {name: sha for name, sha in apk_members.items() if name.endswith(".so")}
    expected_prompts = {path.removeprefix("genuicraft/src/main/"): sha
                        for path, sha in source["current_source_sha256"].items()
                        if path.startswith(MAIN + "assets/genuicraft/prompts/")}
    prompts = {"expected_source_sha256": expected_prompts,
               "aar": {name: sha for name, sha in aar_members.items()
                       if name.startswith("assets/genuicraft/prompts/")},
               "apk": {name: sha for name, sha in apk_members.items()
                       if name.startswith("assets/genuicraft/prompts/")}}
    prompts["verified"] = bool(expected_prompts) and prompts["aar"] == prompts["apk"] == expected_prompts
    native = {"aar_sha256": aar_native, "apk_all_abi_sha256": apk_native,
              "aar_matches_v10": aar_native == baseline["aar_native_hashes"],
              "apk_matches_v10": apk_native == baseline["all_abi_native_hashes"]}
    jvm = jvm_report(args.jvm_results, args.expected_tests)
    publication = publication_report(args.maven_root, artifacts["aar"]["sha256"])
    installation = None
    if args.installed_apk_evidence:
        evidence = args.installed_apk_evidence.read_text(encoding="utf-8-sig").strip()
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+(/data/app/[^\r\n]+/base\.apk)", evidence)
        if not match:
            raise ValueError("Expected one sha256sum line for the installed /data/app/.../base.apk")
        copied_evidence = args.output / "final_installed_apk_sha256.txt"
        if args.installed_apk_evidence.resolve() != copied_evidence.resolve():
            shutil.copyfile(args.installed_apk_evidence, copied_evidence)
        installation = {
            "scope": "Parent agent supplied device sha256sum output; collector read the saved host file only",
            "evidence": file_record(copied_evidence), "installed_base_apk_path": match[2],
            "installed_main_apk_sha256": match[1].lower(),
            "matches_local_main_apk": match[1].lower() == artifacts["apk"]["sha256"],
        }
    report = {
        "schema": "genuicraft_visual_host_build_evidence_v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "Host source files, built archives, JVM XML, and local Maven publication only",
        "baseline_manifest": file_record(baseline_path), "source_delta": file_record(args.output / "source_delta.json"),
        "artifacts": artifacts, "native_closure": native, "packaged_prompts": prompts,
        "jvm_tests": jvm, "maven_publication": publication,
        "installation_verification": installation, "inference_executed_by_collector": False,
        "verified": native["aar_matches_v10"] and native["apk_matches_v10"] and
                    prompts["verified"] and jvm["verified"] and publication["verified"] and
                    (installation is None or installation["matches_local_main_apk"]),
    }
    write_json(args.output / "host_build_evidence.json", report)
    print(json.dumps({"verified": report["verified"], "jvm_tests": jvm["tests"],
                      "maven_files": publication["file_count"], "artifacts": artifacts}, indent=2))
    if not report["verified"]:
        raise SystemExit("Host validation failed; inspect host_build_evidence.json")


if __name__ == "__main__":
    main()
