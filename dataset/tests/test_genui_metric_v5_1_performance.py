from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_reports_required_observability(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(DATASET_ROOT / "scripts" / "benchmark_genui_metric_v5_1.py"),
            "--iterations",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["metric_version"] == "5.1.0"
    assert result["network_calls"] == 0
    assert result["p50_scalar_ms"] >= 0.0
    assert result["p95_scalar_ms"] >= 0.0
    assert result["p50_group_of_8_ms"] >= 0.0
    assert result["p95_group_of_8_ms"] >= 0.0
    assert result["peak_memory_bytes"] > 0
    assert result["evaluated_edge_counts"]
    assert isinstance(result["matching_complete"], bool)
