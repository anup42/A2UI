"""Validate completed audit artifacts and write a small reproducibility manifest."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

TRAINING = Path(__file__).resolve().parents[2]
REPO = TRAINING.parent
REPORT = TRAINING / "reports/full_data_audit_20260913"
OUTPUT = TRAINING / "outputs/audits/full_data_20260913"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    source_manifest = json.loads((REPORT / "inventory/files.json").read_text(encoding="utf-8"))
    sources = {}
    for split, item in source_manifest.items():
        path = Path(item["path"])
        current = sha(path)
        if current != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Original source changed: {split}")
        sources[split] = {"path": str(path), "bytes": path.stat().st_size, "sha256": current, "rows": item["physical_lines"]}
    counts = {}
    for filename, table in (("inventory.sqlite", "rows"), ("ir.sqlite", "rows"), ("synthesis.sqlite", "triage")):
        connection = sqlite3.connect((OUTPUT / filename).resolve().as_uri() + "?mode=ro", uri=True)
        counts[filename] = dict(connection.execute(f"SELECT split,COUNT(*) FROM {table} GROUP BY split"))
        connection.close()
        if counts[filename] != {s:v["rows"] for s, v in sources.items()}:
            raise ValueError(f"Audit row count mismatch: {filename}")
    parity = [json.loads((REPORT / "ir" / name).read_text(encoding="utf-8")) for name in ("reference_parity_initial.json", "reference_parity.json")]
    if any(p["mismatch_count"] for p in parity):
        raise ValueError("Production validator parity mismatch")
    broken = []
    for match in re.finditer(r"\]\(([^)]+)\)", (REPORT / "REPORT.md").read_text(encoding="utf-8")):
        target = match.group(1)
        resolved = REPORT / target.split("#")[0]
        if not target.startswith(("https:", "http:", "#")) and resolved != REPORT / "audit_manifest.json" and not resolved.exists():
            broken.append(target)
    if broken:
        raise ValueError(f"Broken main report links: {broken}")
    artifacts = {}
    manifest_path = REPORT / "audit_manifest.json"
    for path in sorted(REPORT.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        artifacts[str(path.relative_to(REPO)).replace("\\", "/")] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    source_paths = list((TRAINING / "scripts/audits").glob("*.py")) + list((TRAINING / "tests").glob("test_full_data*py"))
    source_paths += [TRAINING / "src/ir_training/data/express_preparation.py",
                    REPO / "dataset/schema/genuicraft_a2ui_v1_wire.schema.json",
                    REPO / "dataset/src/pipeline/renderer_semantics.py",
                    REPO / "dataset/src/pipeline/flat_spec_contract.py"]
    source_paths += list((REPO / "dataset/src/pipeline/ir_formats").glob("*.py"))
    code = {str(p.relative_to(REPO)).replace("\\", "/"): sha(p) for p in sorted(set(source_paths))}
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "checkout_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "sources": sources, "row_reconciliation": counts,
              "production_reference_parity_cases": sum(p["sample_count"] for p in parity),
              "production_reference_parity_mismatches": 0,
              "main_report_local_links_valid": True,
              "runtime": {"python": sys.version, **{n:importlib.metadata.version(n) for n in ("jsonschema", "fastjsonschema", "tokenizers", "transformers")}},
              "report_files": artifacts, "source_code_sha256": code,
              "source_data_modified": False, "trained_or_generated_model_outputs": False,
              "note": "Large row-level indices/queues are local ignored diagnostics, not approved training artifacts. Report hashes exclude this manifest itself."}
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "manifest": str(manifest_path), "sources": sources, "report_files": len(artifacts)}))


if __name__ == "__main__":
    main()
