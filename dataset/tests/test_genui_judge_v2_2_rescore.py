from __future__ import annotations

from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_judge import reliability_rescore as v2_1  # noqa: E402
from pipeline.genui_judge import reliability_rescore_v2_2 as v2_2  # noqa: E402
from pipeline.genui_judge.protocol import PASS_DIMENSIONS  # noqa: E402


def test_v2_2_profile_is_versioned_and_does_not_mutate_v2_1() -> None:
    target = (0, 3, 7)
    before = v2_1.rescore_protocol_fingerprint(target)
    mapping = v2_2.rescore_protocol_mapping(target)
    after = v2_1.rescore_protocol_fingerprint(target)

    assert before == after
    assert mapping["schema_version"].endswith(".v2.2")
    assert mapping["protocol_version"].endswith(".v2.2.0")
    assert mapping["target_milestones"] == [0, 3, 7]
    assert v2_2.rescore_protocol_fingerprint(target) != before


def test_v2_2_judgment_validation_uses_its_fingerprint() -> None:
    fingerprint = v2_2.rescore_protocol_fingerprint()
    value = {
        "packet_id": "r_v2_2_packet",
        "pass_type": "screenshot_only",
        "protocol_fingerprint": fingerprint,
        "dimensions": {
            name: 75
            for name in PASS_DIMENSIONS["screenshot_only"]
        },
        "confidence_0_1": 0.9,
        "evidence_observations": ["Visible evidence."],
        "visible_defects": ["[material] one important region clips."],
        "cannot_assess": [],
        "rationale": "One material defect yields 75.",
        "judge_task_id": "task-v2-2",
        "judge_model_identifier": v2_2.EXPECTED_MODEL_IDENTIFIER,
        "judged_at": "2026-07-30T00:00:00+00:00",
    }

    normalized = v2_2.validate_rescore_pass(
        value,
        fingerprint=fingerprint,
        packet_id="r_v2_2_packet",
        pass_type="screenshot_only",
        expected_task_id="task-v2-2",
    )
    assert normalized["dimensions"] == value["dimensions"]
    assert normalized["cannot_assess"] == []
