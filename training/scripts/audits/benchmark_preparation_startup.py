"""Bounded CPU filtering benchmark; no model/tokenizer load or input mutation."""
from __future__ import annotations

import argparse
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ir_training.common.progress import log
from ir_training.data.audit_filter import audit_and_filter_rows
from ir_training.data.express_preparation import serialize_checked, _wire_validator
from ir_training.pipeline.golden_training import _strict_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if not 1 <= args.rows <= 1024 or any(worker < 1 for worker in args.workers):
        parser.error("Use 1..1024 rows and positive worker counts")
    rows = list(islice(_strict_rows(args.input), args.rows))
    if not rows:
        parser.error("Input is empty")
    _wire_validator()
    expected, reports = None, []
    for workers in args.workers:
        serialize_checked.cache_clear()
        log(f"Bounded CPU benchmark: {len(rows)} rows, {workers} worker(s), including worker startup")
        started = time.perf_counter()
        accepted, quarantine, report = audit_and_filter_rows(rows, require_source_identities=True, workers=workers)
        elapsed = time.perf_counter() - started
        digest = hashlib.sha256(json.dumps([accepted, quarantine, report], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("Serial and parallel decisions differ")
        expected = digest
        reports.append({"workers": workers, "rows": len(rows), "accepted": len(accepted), "seconds": round(elapsed, 3),
                        "rows_per_second": round(len(rows) / elapsed, 3), "decisions_sha256": digest})
    print(json.dumps({"local_cpu_only": True, "model_loaded": False, "tokenizer_loaded": False,
                      "inputs_modified": False, "worker_startup_included": True, "results": reports}, indent=2), flush=True)


if __name__ == "__main__":
    main()
