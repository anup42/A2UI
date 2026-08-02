from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.genui_quality.candidate_normalization_v5_4 import (  # noqa: E402
    normalize_and_validate_express_candidate_v5_4,
)
from pipeline.genui_quality import generation_reward_a2ui_express_v1  # noqa: E402
from pipeline.ir_formats import encode_express_completion  # noqa: E402


def _spec() -> dict:
    return {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Text", "props": {"text": "Hello"}, "children": []}
        },
    }


def test_active_boundary_rejects_json_even_when_legacy_normalizer_can_recover() -> None:
    legacy = '{"root":"root","state":{},"elements":{"root":{"type":"Text","props":{"text":"Hello"},"children":[]}}}'
    result = normalize_and_validate_express_candidate_v5_4(legacy)
    assert not result.raw_parse_ok
    assert not result.production_valid
    assert result.canonical_spec is None


def test_repaired_completion_is_not_reported_as_raw_valid() -> None:
    express = encode_express_completion(_spec())
    malformed = express.replace("</a2ui>", "")
    result = normalize_and_validate_express_candidate_v5_4(malformed)
    assert not result.raw_parse_ok
    assert not result.raw_valid
    assert not result.repaired_valid
    reward = generation_reward_a2ui_express_v1(malformed, "Hello")
    assert reward.quality_0_1 == pytest.approx(0.0)
