from __future__ import annotations

from pathlib import Path
import subprocess
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_2_support import contract, simple_spec  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    breakdown_to_mapping,
    load_default_reward_config,
    score_record,
)
from pipeline.genui_quality.evidence_v5_2 import (  # noqa: E402
    ComputedFunctionRegistryV52,
)
from pipeline.genui_quality.identity_v5_2 import (  # noqa: E402
    metric_fingerprint_v5_2,
    score_source_manifest_v5_2,
)


def test_transitive_manifest_covers_score_defining_modules() -> None:
    manifest = score_source_manifest_v5_2()
    required = {
        "_core.py",
        "_v5.py",
        "_v5_2.py",
        "metrics_v5_2.py",
        "evidence_v5_2.py",
        "matching_v5_2.py",
        "structure_v5_1.py",
        "validation_v5_2.py",
        "candidate_normalization.py",
        "source_contract.py",
        "graph.py",
        "config.py",
        "aggregate.py",
        "../flat_spec_contract.py",
        "../flat_spec_semantics.py",
    }
    assert required <= set(manifest)
    assert all(len(value) == 64 for value in manifest.values())


def test_core_and_v5_changes_alter_fingerprint() -> None:
    config = load_default_reward_config()
    before = metric_fingerprint_v5_2(config)
    for module in ("_core.py", "_v5.py"):
        changed = metric_fingerprint_v5_2(
            config,
            source_hash_overrides={module: f"changed-{module}"},
        )
        assert changed != before


def test_fingerprint_is_stable_across_python_processes() -> None:
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(DATASET_ROOT / 'src')!r});"
        "from pipeline.genui_quality import load_default_reward_config;"
        "from pipeline.genui_quality.identity_v5_2 import "
        "metric_fingerprint_v5_2;"
        "print(metric_fingerprint_v5_2(load_default_reward_config()))"
    )
    values = [
        subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=DATASET_ROOT.parent,
            text=True,
        ).strip()
        for _ in range(2)
    ]
    assert values[0] == values[1] == metric_fingerprint_v5_2(
        load_default_reward_config()
    )


def test_config_parity_vector_and_registry_change_identity() -> None:
    config = load_default_reward_config()
    before = metric_fingerprint_v5_2(config)
    config.good_json_to_source_ratio += 0.01
    assert metric_fingerprint_v5_2(config) != before
    assert (
        metric_fingerprint_v5_2(
            config, parity_vector_hash_override="different"
        )
        != metric_fingerprint_v5_2(config)
    )
    good = ComputedFunctionRegistryV52(
        registry_id="registry",
        registry_version="1",
        functions={"value": lambda args: "good"},
        manifest_hash="good",
    )
    bad = ComputedFunctionRegistryV52(
        registry_id="registry",
        registry_version="1",
        functions={"value": lambda args: "bad"},
        manifest_hash="bad",
    )
    assert metric_fingerprint_v5_2(
        config, computed_registry=good
    ) != metric_fingerprint_v5_2(config, computed_registry=bad)


def test_stored_score_reuse_requires_exact_fingerprint_and_payload() -> None:
    source = "Hello world"
    expected = contract(source)
    record = {
        "ui_id": "reuse",
        "response_text": source,
        "genui_json": simple_spec(source),
        "expected_ui_contract": expected,
    }
    first = score_record(record)
    record["render_artifact_quality_v5_2"] = breakdown_to_mapping(first)
    reused = score_record(record)
    assert reused.evidence["score_reuse"]["reused"]

    changed = load_default_reward_config()
    changed.good_json_to_source_ratio += 0.01
    rescored = score_record(record, config=changed)
    assert not rescored.evidence["score_reuse"]["reused"]
    assert "metric_fingerprint_mismatch" in rescored.evidence[
        "score_reuse"
    ]["stale_reasons"]

    record["genui_json"] = simple_spec("changed")
    payload_changed = score_record(record)
    assert not payload_changed.evidence["score_reuse"]["reused"]
    assert any(
        reason.endswith("candidate_hash_mismatch")
        for reason in payload_changed.evidence["score_reuse"][
            "stale_reasons"
        ]
    )
