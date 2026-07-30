from __future__ import annotations

import json
from pathlib import Path
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_ROOT.parent
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_quality import load_v5_3_reward_config  # noqa: E402
from pipeline.genui_quality.evidence_v5_3 import (  # noqa: E402
    DynamicEvidenceResolverV53,
    deep_equal_v5_3,
    kotlin_string_v5_3,
)

FIXTURE = DATASET_ROOT / "tests" / "fixtures" / "flat_expr_parity_vectors.json"
ANDROID = REPO_ROOT / "android" / "app" / "src" / "test" / "resources" / "flat_expr_parity_vectors.json"


def test_every_shared_android_vector_passes_python() -> None:
    corpus = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert corpus["version"] == "2.0.0"
    for vector in corpus["vectors"]:
        resolver = DynamicEvidenceResolverV53(
            vector["state"], load_v5_3_reward_config()
        )
        actual = resolver.resolve(
            vector["expression"],
            item=vector["item"],
            index=vector["index"],
            base_path=vector["base_path"],
        )
        if "expected_value" in vector:
            assert actual == vector["expected_value"], vector["name"]
        assert resolver.unknown == vector["expected_unknown_codes"]


def test_shared_corpora_identical_and_typed_semantics() -> None:
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == json.loads(
        ANDROID.read_text(encoding="utf-8")
    )
    assert not deep_equal_v5_3(3.0, 3)
    assert not deep_equal_v5_3({"x": [3.0]}, {"x": [3]})
    assert kotlin_string_v5_3({"x": [True, None]}) == "{x=[true, null]}"
