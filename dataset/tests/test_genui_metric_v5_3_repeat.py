from __future__ import annotations

from pathlib import Path
import sys

import pytest

DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_3_support import repeat_spec, score, static_spec  # noqa: E402


@pytest.mark.parametrize("count", [32, 33, 100, 1000])
def test_repeat_has_same_semantic_evidence_as_static(count: int) -> None:
    values = [f"Item {index}" for index in range(count)]
    source = "\n".join(values)
    repeat = score(repeat_spec(values), source)
    static = score(static_spec(values), source)
    assert repeat.dynamic_evidence_certification["complete"]
    assert repeat.atomics["fidelity"]["content_unit_fidelity"] == pytest.approx(
        static.atomics["fidelity"]["content_unit_fidelity"], abs=1e-12
    )
    assert repeat.atomics["fidelity"]["output_block_precision"] == pytest.approx(
        static.atomics["fidelity"]["output_block_precision"], abs=1e-12
    )
    assert repeat.evidence["output"]["expanded_evidence_nodes"] == count + 2
