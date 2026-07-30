from __future__ import annotations

from pathlib import Path
import sys

import pytest

DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_3_support import compact, simple_spec  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    aggregate_v5_3_records,
    genui_grpo_reward,
    score_genui_completion_v5_3,
    score_record_generation_v5_3,
)


def test_offline_aggregate_and_grpo_generation_paths_match() -> None:
    raw = compact(simple_spec())
    record = {
        "ui_id": "one",
        "response_text": "Hello world",
        "genui_raw_completion": raw,
        "genui_json": simple_spec(),
    }
    offline = score_genui_completion_v5_3(raw, "Hello world")
    aggregate_row = score_record_generation_v5_3(record)
    grpo = genui_grpo_reward([raw], "Hello world")[0]
    assert aggregate_row.reward == pytest.approx(offline.reward, abs=1e-12)
    assert grpo == pytest.approx(offline.reward, abs=1e-12)
    summary = aggregate_v5_3_records([record])
    assert summary["metric_version"] == "5.3.0"


def test_historical_v5_2_breakdown_is_not_read_as_v5_3() -> None:
    from pipeline.genui_quality import breakdown_from_mapping

    value = {
        "reward": 0,
        "quality_0_100": 50,
        "quality_0_1": 0.5,
        "cap_0_1": 1,
        "parse_stage": "json",
        "dimensions": {},
        "atomics": {},
        "evidence": {},
        "metric_version": "5.2.0",
    }
    assert breakdown_from_mapping(value).metric_version == "5.2.0"
    assert not hasattr(
        breakdown_from_mapping(value), "reward_pipeline_fingerprint"
    )
