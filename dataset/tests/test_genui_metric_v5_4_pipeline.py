from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_4_support import root_spec, text  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    RewardInflationMonitor,
    breakdown_to_mapping,
    generation_reward_v5_4,
    load_v5_4_reward_config,
    make_genui_grpo_reward_v5_4,
    render_artifact_quality_v5_4,
    score_record_v5_4,
)
from pipeline.genui_quality.evidence_v5_4 import (  # noqa: E402
    ComputedFunctionRegistryV54,
)
from pipeline.genui_quality.identity_v5_4 import (  # noqa: E402
    metric_fingerprint_v5_4,
)


GLOBAL_MULTIPLIER_V54 = 2


def _global_dependent(args: dict) -> float:
    return float(args.get("value", 0)) * GLOBAL_MULTIPLIER_V54


def test_computed_global_dependency_changes_metric_fingerprint() -> None:
    global GLOBAL_MULTIPLIER_V54
    config = load_v5_4_reward_config()
    registry = ComputedFunctionRegistryV54(
        registry_id="test",
        registry_version="1",
        functions={"scale": _global_dependent},
        manifest_hash="declared-test-manifest",
    )
    original = GLOBAL_MULTIPLIER_V54
    try:
        first = metric_fingerprint_v5_4(
            config, computed_registry=registry
        )
        GLOBAL_MULTIPLIER_V54 = 3
        second = metric_fingerprint_v5_4(
            config, computed_registry=registry
        )
    finally:
        GLOBAL_MULTIPLIER_V54 = original
    assert first != second


def test_stored_score_reuse_requires_exact_payload_identity() -> None:
    first_candidate = root_spec({"text": text("Hello")}, ["text"])
    first = render_artifact_quality_v5_4(
        first_candidate, "Hello", legacy_comparison=True
    )
    record = {
        "ui_id": "sample",
        "response_text": "Hello",
        "genui_json": first_candidate,
        "target_format": "flat_spec_v1",
        "render_artifact_quality_v5_4": breakdown_to_mapping(first),
    }
    reused = score_record_v5_4(record)
    assert reused.evidence["score_reuse"]["reused"]

    record["genui_json"] = root_spec({"text": text("Different")}, ["text"])
    recomputed = score_record_v5_4(record)
    assert not recomputed.evidence["score_reuse"]["reused"]
    assert "raw_candidate_hash_mismatch" in recomputed.evidence[
        "score_reuse"
    ]["stale_reasons"]


def test_stage3_artifact_and_grpo_share_semantic_normalization() -> None:
    candidate = root_spec({"text": text("Hello")}, ["text"])
    raw = json.dumps(candidate)
    aliased = raw.replace('"Column"', '"column"').replace(
        '"Text"', '"text"'
    )
    artifact = render_artifact_quality_v5_4(aliased, "Hello")
    generation = generation_reward_v5_4(aliased, "Hello")
    assert generation.artifact_quality_0_1 == pytest.approx(
        artifact.quality_0_1, abs=1e-12
    )
    assert generation.identity["canonical_candidate_hash"] == (
        artifact.identity["canonical_candidate_hash"]
    )
    assert generation.quality_0_1 <= artifact.quality_0_1


def test_config_change_changes_fingerprint() -> None:
    config = load_v5_4_reward_config()
    changed = replace(
        config,
        accessibility_penalty_alpha=(
            config.accessibility_penalty_alpha / 2.0
        ),
    )
    assert metric_fingerprint_v5_4(config) != metric_fingerprint_v5_4(
        changed
    )


def test_stage3_aggregate_dashboard_and_training_dispatch_v5_4() -> None:
    repo_root = DATASET_ROOT.parent
    files = {
        "stage3": (
            DATASET_ROOT / "src" / "pipeline" / "stage3_genui.py"
        ).read_text(encoding="utf-8"),
        "aggregate": (
            DATASET_ROOT / "src" / "pipeline" / "metrics.py"
        ).read_text(encoding="utf-8"),
        "dashboard": (
            DATASET_ROOT / "scripts" / "dataset_dashboard.py"
        ).read_text(encoding="utf-8"),
        "training": (
            repo_root / "training" / "scripts" / "train_grpo.py"
        ).read_text(encoding="utf-8"),
    }
    assert "generation_reward_v5_4" in files["stage3"]
    assert "render_artifact_quality_v5_4" in files["stage3"]
    assert "metric_identity_v5_4" in files["stage3"]
    assert "aggregate_v5_4_records" in files["aggregate"]
    assert '"v5_4"' in files["dashboard"]
    assert "make_genui_grpo_reward_v5_4" in files["training"]
    assert "ensure_v5_4_validation_ready" in files["training"]


def test_training_reward_factory_accepts_monitor_and_logs() -> None:
    metrics: dict[str, float] = {}
    reward = make_genui_grpo_reward_v5_4(
        load_v5_4_reward_config(),
        model_checkpoint="checkpoint",
        inflation_monitor=RewardInflationMonitor(),
    )
    values = reward(
        [root_spec({"text": text("Hello")}, ["text"])],
        ["Hello"],
        log_metric=lambda name, value: metrics.__setitem__(name, value),
    )
    assert len(values) == 1
    assert -1.0 <= values[0] <= 1.0
    assert "genui/reward_mean" in metrics
    assert "reward/group_p95_ms" in metrics
