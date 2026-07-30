from __future__ import annotations

from pathlib import Path
import sys

import pytest

DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_3_support import compact, simple_spec  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    genui_grpo_reward,
    score_genui_completion_v5_3,
)


def test_grpo_is_same_current_v5_3_scalar_path() -> None:
    raw = compact(simple_spec())
    offline = score_genui_completion_v5_3(raw, "Hello world")
    reward = genui_grpo_reward([raw], "Hello world")[0]
    assert reward == pytest.approx(offline.reward, abs=1e-12)
    assert offline.reward_pipeline_fingerprint


def test_group_is_deterministic() -> None:
    raw = compact(simple_spec())
    first = genui_grpo_reward([raw] * 8, "Hello world")
    second = genui_grpo_reward([raw] * 8, "Hello world")
    assert first == second
