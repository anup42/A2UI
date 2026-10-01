"""Compile unchanged Android repair/compiler sources into a reusable JVM bridge.

Provide a Kotlin compiler classpath (compiler-embeddable + dependencies), stdlib,
Gson and Java, or --download-dependencies to fetch pinned Maven artifacts. Only
the legacy FlatSpec detector is extracted from its Android source because the
rest of that migration-only file imports the Compose renderer. No repair or
production parser implementation is copied or replaced.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/src"))
from ir_training.eval.android_repair import PROFILE, SDK, source_digest, source_files

DEPENDENCIES = {
    "compiler": "org/jetbrains/kotlin/kotlin-compiler-embeddable/2.2.21/kotlin-compiler-embeddable-2.2.21.jar",
    "stdlib": "org/jetbrains/kotlin/kotlin-stdlib/2.2.21/kotlin-stdlib-2.2.21.jar",
    "reflect": "org/jetbrains/kotlin/kotlin-reflect/1.6.10/kotlin-reflect-1.6.10.jar",
    "daemon": "org/jetbrains/kotlin/kotlin-daemon-embeddable/2.2.21/kotlin-daemon-embeddable-2.2.21.jar",
    "coroutines": "org/jetbrains/kotlinx/kotlinx-coroutines-core-jvm/1.8.0/kotlinx-coroutines-core-jvm-1.8.0.jar",
    "annotations": "org/jetbrains/annotations/13.0/annotations-13.0.jar",
    "gson": "com/google/code/gson/gson/2.13.2/gson-2.13.2.jar",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--java", default=shutil.which("java"))
    parser.add_argument("--compiler-classpath")
    parser.add_argument("--stdlib")
    parser.add_argument("--gson")
    parser.add_argument("--download-dependencies", action="store_true")
    args = parser.parse_args()
    if not args.java:
        parser.error("Java is required; supply --java pointing to a JRE/JDK")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.download_dependencies:
        deps = output / "deps"
        deps.mkdir(exist_ok=True)
        paths = {}
        for key, artifact in DEPENDENCIES.items():
            target = deps / Path(artifact).name
            if not target.is_file():
                partial = target.with_suffix(".part")
                urllib.request.urlretrieve("https://repo.maven.apache.org/maven2/" + artifact, partial)
                partial.replace(target)
            paths[key] = str(target)
        args.compiler_classpath = os.pathsep.join(paths.values())
        args.stdlib, args.gson = paths["stdlib"], paths["gson"]
    if not all((args.compiler_classpath, args.stdlib, args.gson)):
        parser.error("Provide --compiler-classpath, --stdlib and --gson, or --download-dependencies")
    contract = (ROOT / SDK / "internal/pipeline/FlatSpecContract.kt").read_text(encoding="utf-8")
    detector = re.search(r"    fun looksLikeFlatSpec\(json: JsonElement\?\): Boolean \{.*?\n    \}", contract, re.S)
    if not detector:
        raise ValueError("Cannot extract exact Android legacy format detector")
    shim = output / "FlatSpecDetector.kt"
    shim.write_text("package com.samsung.genuicraft.sdk.internal.pipeline\nimport com.google.gson.JsonElement\n"
        "internal object FlatSpecContract {\n" + detector[0] + "\n}\n", encoding="utf-8")
    jar = output / "android-repair.jar"
    command = [args.java, "-Xmx2g", "-cp", args.compiler_classpath,
        "org.jetbrains.kotlin.cli.jvm.K2JVMCompiler", "-no-stdlib", "-no-reflect",
        "-jvm-target", "17", "-classpath", os.pathsep.join([args.stdlib, args.gson]),
        "-d", str(jar), *map(str, source_files()), str(shim)]
    subprocess.run(command, check=True)
    config = {
        "profile": PROFILE, "source_sha256": source_digest(),
        "command": [str(Path(args.java).resolve()), "-Xmx512m", "-cp",
            os.pathsep.join([str(jar), str(Path(args.stdlib).resolve()), str(Path(args.gson).resolve())]),
            "com.samsung.genuicraft.sdk.AndroidRepairMainKt"],
        "source_files": [str(path.relative_to(ROOT)) for path in source_files()],
        "legacy_detector": "exact method extracted from Android FlatSpecContract.kt",
    }
    manifest = output / "android_repair_runtime.json"
    manifest.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(str(manifest))


if __name__ == "__main__":
    main()
