from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_3_support import simple_spec  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    breakdown_to_mapping,
    load_v5_3_reward_config,
    score_record_v5_3,
)
from pipeline.genui_quality.identity_v5_3 import (  # noqa: E402
    metric_fingerprint_v5_3,
    reward_pipeline_fingerprint_v5_3,
)
from pipeline.genui_quality.source_contract_v5_3 import (  # noqa: E402
    resolve_expected_ui_contract_v5_3,
    source_contract_cache_key_v5_3,
)


def test_metric_and_pipeline_fingerprints_cover_distinct_manifests() -> None:
    config = load_v5_3_reward_config()
    base = metric_fingerprint_v5_3(config)
    changed = metric_fingerprint_v5_3(
        config,
        source_hash_overrides={"metrics_v5_3.py": "changed"},
    )
    assert base != changed
    pipeline = reward_pipeline_fingerprint_v5_3(base)
    changed_pipeline = reward_pipeline_fingerprint_v5_3(
        base,
        source_hash_overrides={"grpo_reward.py": "changed"},
    )
    assert pipeline != changed_pipeline


def test_fingerprints_are_stable_across_processes() -> None:
    config = load_v5_3_reward_config()
    expected = metric_fingerprint_v5_3(config)
    code = (
        "from pipeline.genui_quality import "
        "load_v5_3_reward_config, metric_fingerprint;"
        "print(metric_fingerprint(load_v5_3_reward_config()))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(DATASET_ROOT / "src")
    actual = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=DATASET_ROOT.parent,
        env=environment,
        text=True,
    ).strip()
    assert actual == expected


def test_exact_identity_reuses_and_stale_pipeline_recomputes() -> None:
    record = {
        "response_text": "Hello world",
        "genui_json": simple_spec(),
    }
    first = score_record_v5_3(record)
    record["genui_quality_v5_3"] = breakdown_to_mapping(first)
    reused = score_record_v5_3(record)
    assert reused.evidence["score_reuse"]["reused"]
    record["genui_quality_v5_3"][
        "reward_pipeline_fingerprint"
    ] = "stale"
    rescored = score_record_v5_3(record)
    assert not rescored.evidence["score_reuse"]["reused"]
    assert "reward_pipeline_fingerprint_mismatch" in rescored.evidence[
        "score_reuse"
    ]["stale_reasons"]


def test_contract_cache_is_intent_safe_and_stale_hash_is_rejected() -> None:
    assert source_contract_cache_key_v5_3(
        "Hello", intent="summary"
    ) != source_contract_cache_key_v5_3("Hello", intent="comparison")
    valid = resolve_expected_ui_contract_v5_3(
        "Hello", intent="summary"
    ).contract
    stale = dict(valid)
    stale["source_hash"] = "stale"
    resolved = resolve_expected_ui_contract_v5_3(
        "Hello",
        intent="summary",
        persisted=stale,
        persisted_source="human benchmark",
    )
    assert resolved.source == "deterministic fallback"
    assert any("source_hash mismatch" in error for error in resolved.errors)
