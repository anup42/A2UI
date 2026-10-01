"""Run the actual Android trained-model compiler/repair in one JVM per batch.

Build once with training/scripts/setup_android_repair.py. The source digest is
checked at every use so stale Android code cannot silently produce new scores.
The bridge receives only model output, never source responses or reference IR.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

PROFILE = "TRAINED_E2B_V10_W4"
ROOT = Path(__file__).resolve().parents[4]
SDK = Path("GenUICraft/genuicraft/src/main/java/com/samsung/genuicraft/sdk")
MAIN = Path("training/tools/android_repair/AndroidRepairMain.kt")
SDK_FILES = ("Api.kt", "GenUiCompiler.kt", "ContentIntegrity.kt", "SourceTextFallback.kt", "SourceBindings.kt")
PIPELINE_FILES = (
    "A2uiCanonicalGraph.kt", "A2uiExpressCodec.kt", "A2uiExpressGeneralRepair.kt",
    "A2uiExpressOutputRepair.kt", "A2uiWireCodec.kt", "FlatSpecIdRewriter.kt",
    "FlatSpecReferenceSemantics.kt", "GeneratedStateRecovery.kt", "GenUiA2uiCatalog.kt",
    "GenUiIrCodec.kt", "GenUiIrFormat.kt", "LiteralTextCodec.kt", "RendererReferenceSemantics.kt",
)


def source_files(root: Path = ROOT) -> list[Path]:
    return [root / SDK / name for name in SDK_FILES] + [
        root / SDK / "internal/pipeline" / name for name in PIPELINE_FILES
    ] + [root / MAIN]


def source_digest(root: Path = ROOT) -> str:
    # Include the deployed profile policy and the exact legacy-format detector,
    # even though the Android UI/renderer itself is not a JVM build dependency.
    paths = source_files(root) + [root / SDK / "GenUiSession.kt", root / SDK / "GenUiTrainedConverter.kt",
        root / SDK / "internal/pipeline/FlatSpecContract.kt"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_runtime(config: str | Path | None = None) -> Path | None:
    value = config or os.environ.get("A2UI_ANDROID_REPAIR_RUNTIME")
    return Path(value).expanduser().resolve() if value else None


def repair_batch(texts: list[str], runtime_path: str | Path, *, timeout_seconds: float = 300) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = Path(runtime_path).resolve()
    config = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = source_digest()
    if config.get("source_sha256") != expected:
        raise ValueError("Android repair runtime is stale; rebuild it from the current Android sources")
    if config.get("profile") != PROFILE:
        raise ValueError("Android repair runtime profile differs from the deployed trained profile")
    command = config.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise ValueError("Android repair runtime requires a command argument list")
    request = "".join(json.dumps({"index": index, "generated_text": text}, ensure_ascii=False) + "\n"
                      for index, text in enumerate(texts))
    start = time.perf_counter()
    completed = subprocess.run(command, input=request, text=True, encoding="utf-8",
        capture_output=True, timeout=timeout_seconds, check=False, cwd=manifest_path.parent)
    elapsed = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(f"Android repair bridge exited {completed.returncode}: {completed.stderr[-4000:]}")
    results = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if (len(results) != len(texts) or any(not isinstance(row, dict) or type(row.get("index")) is not int for row in results)
            or [row.get("index") for row in results] != list(range(len(texts)))):
        raise ValueError("Android repair bridge returned missing, duplicated, or reordered rows")
    for row in results:
        seconds = row.get("repair_seconds")
        if (type(row.get("success")) is not bool or type(seconds) not in (float, int)
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Android repair bridge returned malformed evidence")
        if row["success"] and (not isinstance(row.get("express"), str) or not row["express"].strip() or row.get("repair_kind") not in
                {"NONE", "STRUCTURAL", "GENERATED_DSL_REPAIR"}):
            raise ValueError("Android repair bridge returned unsupported repair/fallback")
        if not row["success"] and row.get("repair_kind") != "REJECTED":
            raise ValueError("Android repair bridge returned unsupported rejection evidence")
    durations = [row["repair_seconds"] for row in results]
    ordered = sorted(durations)
    def percentile(fraction: float) -> float:
        if not ordered:
            return 0.0
        index = (len(ordered) - 1) * fraction
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return results, {
        "profile": PROFILE, "source_sha256": expected, "runtime_manifest": str(manifest_path),
        "implementation": "actual Android SDK Kotlin compiler and repair sources",
        "allow_generated_dsl_repair": True, "allow_source_text_fallback": False,
        "source_or_reference_supplied_to_repair": False,
        "batch_wall_seconds": elapsed, "repair_compute_seconds": sum(durations),
        "mean_repair_seconds": sum(durations) / len(durations) if durations else 0.0,
        "max_repair_seconds": max(durations, default=0.0),
        "p50_repair_seconds": percentile(.5), "p95_repair_seconds": percentile(.95),
        "repair_kind_counts": dict(Counter(row["repair_kind"] for row in results)),
        "accepted_count": sum(row["success"] for row in results),
        "rejected_count": sum(not row["success"] for row in results),
    }
