from __future__ import annotations

from pathlib import Path
import sys

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_judge.protocol import PASS_DIMENSIONS  # noqa: E402
from pipeline.genui_judge.reliability_rescore import (  # noqa: E402
    EXPECTED_MODEL_IDENTIFIER,
    rescore_protocol_fingerprint,
    select_rescore_positions,
    validate_rescore_pass,
)


def _pass(packet_id: str, pass_type: str, fingerprint: str) -> dict:
    return {
        "packet_id": packet_id,
        "pass_type": pass_type,
        "protocol_fingerprint": fingerprint,
        "dimensions": {
            name: 75 for name in PASS_DIMENSIONS[pass_type]
        },
        "confidence_0_1": 0.9,
        "evidence_observations": ["Visible evidence."],
        "visible_defects": [],
        "cannot_assess": [],
        "rationale": "Good with noticeable defects.",
        "judge_task_id": "task-rescore",
        "judge_model_identifier": EXPECTED_MODEL_IDENTIFIER,
        "judged_at": "2026-07-30T00:00:00+00:00",
    }


def test_rescore_protocol_fingerprint_is_target_sensitive() -> None:
    assert rescore_protocol_fingerprint((0, 3)) == (
        rescore_protocol_fingerprint((3, 0))
    )
    assert rescore_protocol_fingerprint((0, 3)) != (
        rescore_protocol_fingerprint((0, 3, 7))
    )


def test_milestone_selection_includes_all_repeat_counterparts() -> None:
    positions = list(range(400))
    groups, anchors = select_rescore_positions(
        positions,
        [
            {
                "original_schedule_position": 5,
                "repeat_schedule_position": 205,
            },
            {
                "original_schedule_position": 100,
                "repeat_schedule_position": 250,
            },
            {
                "original_schedule_position": 10,
                "repeat_schedule_position": 20,
            },
        ],
        (0, 3),
    )
    assert groups == [list(range(80)), list(range(240, 320))]
    assert anchors == [100, 205]


def test_rescore_pass_enforces_fingerprint_model_and_arrays() -> None:
    fingerprint = rescore_protocol_fingerprint((0, 3))
    value = _pass("r_packet", "screenshot_only", fingerprint)
    normalized = validate_rescore_pass(
        value,
        fingerprint=fingerprint,
        packet_id="r_packet",
        pass_type="screenshot_only",
        expected_task_id="task-rescore",
    )
    assert normalized["cannot_assess"] == []
    drifted = dict(value)
    drifted["judge_model_identifier"] = "different-model"
    with pytest.raises(ValueError, match="model identifier drift"):
        validate_rescore_pass(
            drifted,
            fingerprint=fingerprint,
            packet_id="r_packet",
            pass_type="screenshot_only",
        )
    invalid_arrays = dict(value)
    invalid_arrays["cannot_assess"] = False
    with pytest.raises(ValueError, match="must be an array"):
        validate_rescore_pass(
            invalid_arrays,
            fingerprint=fingerprint,
            packet_id="r_packet",
            pass_type="screenshot_only",
        )
