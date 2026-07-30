from __future__ import annotations

from pathlib import Path
import sys
import time


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_2_support import simple_spec  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    prepare_source_context_v5_2,
    score_completion_group_v5_2,
)
from pipeline.genui_quality.metrics_v5_2 import (  # noqa: E402
    content_fidelity_v5_2,
)


def test_exact_thousand_unit_matching_is_linear_preallocation() -> None:
    values = [f"Unique exact unit {index}" for index in range(1000)]
    started = time.perf_counter()
    atomics, diagnostics = content_fidelity_v5_2(
        values,
        values,
        exact_dense_limit=64,
        max_edges=65536,
        top_k=16,
    )
    elapsed = time.perf_counter() - started
    assert atomics["content_unit_fidelity"] == 1.0
    assert diagnostics["matching"]["exact_preallocated_count"] == 1000
    assert diagnostics["matching"]["evaluated_edge_count"] == 1000
    assert diagnostics["matching"]["certification"][
        "optimality_certified"
    ]
    # Generous guard catches accidental million-edge dense regression without
    # pretending to be a hardware-independent GRPO SLA.
    assert elapsed < 10.0


def test_eight_completion_group_reuses_one_prepared_source() -> None:
    source = "Hello world"
    prepared = prepare_source_context_v5_2(source)
    results = score_completion_group_v5_2(
        [simple_spec(source) for _ in range(8)], prepared
    )
    assert len(results) == 8
    assert len({result.metric_fingerprint for result in results}) == 1
    assert all(
        result.matching_certification["optimality_certified"]
        for result in results
    )
