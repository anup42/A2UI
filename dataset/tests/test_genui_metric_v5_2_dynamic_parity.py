from __future__ import annotations

import json
from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_ROOT.parent
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_2_support import simple_spec  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    load_default_reward_config,
)
from pipeline.genui_quality._v5_2 import (  # noqa: E402
    score_genui_completion_v5_2,
)
from pipeline.genui_quality.evidence_v5_2 import (  # noqa: E402
    ComputedFunctionRegistryV52,
    DynamicEvidenceResolverV52,
)


FIXTURE = DATASET_ROOT / "tests" / "fixtures" / (
    "flat_expr_parity_vectors.json"
)
ANDROID_FIXTURE = REPO_ROOT / "android" / "app" / "src" / "test" / (
    "resources"
) / "flat_expr_parity_vectors.json"


def test_python_runs_every_shared_android_parity_vector() -> None:
    corpus = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # V5.2 remains frozen on the pre-typed-equality subset. The shared corpus
    # is now v2.0.0 for v5.3/Android and retains every historical vector.
    assert corpus["version"] == "2.0.0"
    assert len(corpus["vectors"]) >= 25
    for vector in corpus["vectors"]:
        if vector["name"] in {
            "sum double is not equal to integral literal",
            "sum double equals double literal",
            "nested numeric collection equality preserves type",
            "collection concat uses Kotlin stringification",
            "inline collection stringification",
            "lexical numeric string is not equal to number",
            "lexical numeric string participates in numeric comparison",
        }:
            continue
        resolver = DynamicEvidenceResolverV52(
            vector["state"], load_default_reward_config()
        )
        actual = resolver.resolve(
            vector["expression"],
            item=vector["item"],
            index=vector["index"],
            base_path=vector["base_path"],
        )
        if vector.get("expected_unknown"):
            assert resolver.unknown, vector["name"]
        else:
            assert actual == vector["expected_value"], vector["name"]
        assert resolver.unknown == vector["expected_unknown_codes"], (
            vector["name"]
        )


def test_android_and_python_corpora_are_semantically_identical() -> None:
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == json.loads(
        ANDROID_FIXTURE.read_text(encoding="utf-8")
    )


def test_state_paths_blank_dotted_item_and_boolean_numeric_match_android() -> None:
    resolver = DynamicEvidenceResolverV52(
        {
            "user": {"name": "Ada"},
            "user.name": "literal-state",
            "flag": True,
        },
        load_default_reward_config(),
    )
    assert resolver.resolve({"$state": "user/name"}) == "Ada"
    assert resolver.resolve({"$state": "/user/name"}) == "Ada"
    assert resolver.resolve({"$state": ""})["flag"] is True
    assert resolver.resolve(
        {"$item": "user.name"},
        item={"user.name": "literal", "user": {"name": "nested"}},
    ) == "literal"
    assert resolver.resolve(
        {
            "$cond": {"$state": "flag", "gt": 0},
            "$then": "wrong",
            "$else": "android",
        }
    ) == "android"


def test_unknown_dollar_map_is_ordinary_and_builtins_are_default() -> None:
    resolver = DynamicEvidenceResolverV52(
        {"name": "Ada"}, load_default_reward_config()
    )
    assert resolver.resolve(
        {"$futureKey": {"$state": "name"}, "ok": True}
    ) == {"$futureKey": "Ada", "ok": True}
    assert resolver.unknown == []
    assert resolver.resolve(
        {
            "$computed": "concat",
            "args": {"a": "A", "b": 2, "c": None, "d": True},
        }
    ) == "A2true"
    assert resolver.resolve(
        {"$computed": "uppercase", "args": {"value": "Ada"}}
    ) == "ADA"
    assert resolver.resolve(
        {
            "$computed": "coalesce",
            "args": {"values": [None, "", "Ada"]},
        }
    ) == "Ada"
    assert resolver.resolve(
        {
            "$computed": "sum",
            "args": {"values": [1, 2.5, True, "4"]},
        }
    ) == 3.5


def test_custom_registry_behavior_changes_metric_identity() -> None:
    good = ComputedFunctionRegistryV52(
        registry_id="test",
        registry_version="1",
        functions={"constant": lambda args: "good"},
        manifest_hash="good-manifest",
    )
    bad = ComputedFunctionRegistryV52(
        registry_id="test",
        registry_version="1",
        functions={"constant": lambda args: "bad"},
        manifest_hash="bad-manifest",
    )
    first = score_genui_completion_v5_2(
        simple_spec(), "Hello world", computed_registry=good
    )
    second = score_genui_completion_v5_2(
        simple_spec(), "Hello world", computed_registry=bad
    )
    assert first.metric_fingerprint != second.metric_fingerprint
    assert (
        first.identity["computed_registry_manifest_hash"]
        != second.identity["computed_registry_manifest_hash"]
    )


def test_depth_and_string_expansion_limits_are_explicit() -> None:
    config = load_default_reward_config()
    config.max_expression_depth = 1
    config.max_string_expansion_length = 3
    resolver = DynamicEvidenceResolverV52({"name": "Ada Lovelace"}, config)
    resolver.resolve({"ordinary": {"nested": {"$state": "name"}}})
    assert any(code.startswith("expression_depth:") for code in resolver.unknown)
    assert resolver.resolve("${name}") == "Ada"
    assert any(code.startswith("string_truncated:") for code in resolver.unknown)
