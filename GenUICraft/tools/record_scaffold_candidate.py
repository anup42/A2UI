"""Record the exact built SDK consumer and verify the unchanged GPU native closure."""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True,
                        help="Directory holding v10_build_manifest.json and v10_sdk_source_hashes.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sdk = Path(__file__).resolve().parents[1]
    root = sdk.parent
    aar = sdk / "genuicraft/build/outputs/aar/genuicraft-release.aar"
    apk = root / "android/app/build/outputs/apk/debug/app-debug.apk"
    test_apk = root / "android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
    prompt = sdk / "genuicraft/src/main/assets/genuicraft/prompts/gemma.txt"
    previous = json.loads((args.previous / "v10_build_manifest.json").read_text())
    old = json.loads((args.previous / "v10_sdk_source_hashes.json").read_text())
    current = {p.relative_to(sdk).as_posix(): sha(p)
               for p in sorted((sdk / "genuicraft/src/main").rglob("*")) if p.is_file()}
    delta = [p for p in sorted(set(old) | set(current)) if old.get(p) != current.get(p)]
    counts = dict(tests=0, failures=0, errors=0, skipped=0, suites=0)
    for path in (sdk / "genuicraft/build/test-results/testDebugUnitTest").glob("TEST-*.xml"):
        xml = ET.parse(path).getroot()
        counts["suites"] += 1
        for key in ("tests", "failures", "errors", "skipped"):
            counts[key] += int(xml.get(key, "0"))
    assert counts["tests"] > 0 and counts["failures"] == counts["errors"] == counts["skipped"] == 0, counts
    with zipfile.ZipFile(apk) as archive:
        native = {name: hashlib.sha256(archive.read(name)).hexdigest()
                  for name in archive.namelist() if name.startswith("lib/") and name.endswith(".so")}
        assert archive.read("assets/genuicraft/prompts/gemma.txt") == prompt.read_bytes()
    with zipfile.ZipFile(aar) as archive:
        aar_native = {name: hashlib.sha256(archive.read(name)).hexdigest()
                      for name in archive.namelist() if name.startswith("jni/") and name.endswith(".so")}
        assert archive.read("assets/genuicraft/prompts/gemma.txt") == prompt.read_bytes()
    for name, digest in aar_native.items():
        assert native[name.replace("jni/", "lib/", 1)] == digest
    assert native == previous["all_abi_native_hashes"], "Native closure changed"
    unchanged = not any("/provider/" in p or "/jniLibs/" in p for p in delta)
    assert unchanged, "Runtime/provider changed during prompt study"
    result = dict(artifact_role="candidate_v11_gemma_source_layout_scaffold",
                  recorded_utc=datetime.now(timezone.utc).isoformat(),
                  aar_sha256=sha(aar), aar_bytes=aar.stat().st_size,
                  apk_sha256=sha(apk), test_apk_sha256=sha(test_apk),
                  gemma_prompt_sha256=sha(prompt), sdk_only_native=True,
                  all_abi_native_hashes=native, aar_native_hashes=aar_native,
                  jvm_tests=counts, main_source_delta_from_v10=delta,
                  provider_native_source_unchanged_from_v10=unchanged,
                  provider_defaults=previous["provider_defaults"],
                  query_sent_to_model=False, gemma_source_bindings=True,
                  bundled_gemma_layout_scaffold=True,
                  custom_prompt_layout_scaffold_default=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "v11_build_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output / "v11_sdk_source_hashes.json").write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
