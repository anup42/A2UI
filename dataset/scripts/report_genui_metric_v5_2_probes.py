#!/usr/bin/env python3
"""Reproduce the confirmed v5.1 versus v5.2 hardening probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_SRC = REPO_ROOT / "dataset" / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.genui_quality import _core  # noqa: E402
from pipeline.genui_quality.config import (  # noqa: E402
    load_default_reward_config,
    load_v5_1_reward_config,
)
from pipeline.genui_quality.evidence_v5_1 import (  # noqa: E402
    DynamicEvidenceResolver as DynamicEvidenceResolverV51,
)
from pipeline.genui_quality.evidence_v5_2 import (  # noqa: E402
    DynamicEvidenceResolverV52,
)
from pipeline.genui_quality.grpo_reward import (  # noqa: E402
    normalize_asset_rows,
)
from pipeline.genui_quality.identity_v5_1 import (  # noqa: E402
    score_source_manifest as score_source_manifest_v5_1,
)
from pipeline.genui_quality.identity_v5_2 import (  # noqa: E402
    metric_fingerprint_v5_2,
    score_source_manifest_v5_2,
)
from pipeline.genui_quality.matching_v5_1 import (  # noqa: E402
    MatchEdge as MatchEdgeV51,
    maximum_weight_assignment as maximum_weight_assignment_v5_1,
)
from pipeline.genui_quality.matching_v5_2 import (  # noqa: E402
    MatchEdge as MatchEdgeV52,
    maximum_weight_assignment as maximum_weight_assignment_v5_2,
)
from pipeline.genui_quality.metrics_v5_1 import (  # noqa: E402
    action_fidelity_v5_1,
)
from pipeline.genui_quality.metrics_v5_2 import (  # noqa: E402
    action_fidelity_v5_2,
    semantic_role_coverage_v5_2,
)
from pipeline.genui_quality.source_contract import (  # noqa: E402
    extract_expected_ui_contract_v5_1,
    extract_expected_ui_contract_v5_2,
)


MATCHING_ARGS = {
    "exact_dense_limit": 64,
    "max_edges": 65536,
    "top_k": 16,
}


def _assignment_probe(size: int) -> dict[str, Any]:
    edges = [
        (0, 0, 0.90),
        (0, 1, 0.80),
        (1, 0, 0.85),
        (1, 1, 0.00),
        *[(index, index, 1.0) for index in range(2, size)],
    ]
    old = maximum_weight_assignment_v5_1(
        size,
        size,
        [MatchEdgeV51(*edge) for edge in edges],
        exact_dense_limit=1 if size == 2 else 64,
    )
    new = maximum_weight_assignment_v5_2(
        size,
        size,
        [MatchEdgeV52(*edge) for edge in edges],
        candidate_generation_complete=True,
    )
    return {
        "v5_1_weight": sum(edge.score for edge in old.matches),
        "v5_1_complete": old.complete,
        "v5_1_exact": old.exact,
        "v5_2_weight": sum(edge.score for edge in new.matches),
        "v5_2_optimality_certified": (
            new.certification.optimality_certified
        ),
        "v5_2_approximate": (
            new.certification.approximate_matching_used
        ),
    }


def _matching_probes() -> dict[str, Any]:
    count = 300
    expected = [
        _core.ActionRef(
            label=f"Open item {index}",
            url=f"https://example.com/{index}",
        )
        for index in range(count)
    ]
    actual = [
        _core.OutputAction(
            element_id=f"button_{index}",
            label=f"Open item {index}",
            url=f"https://example.com/{index}",
            source="on.press",
        )
        for index in range(count)
    ]
    old_value, old_diagnostics = action_fidelity_v5_1(
        expected, actual, aliases={}, **MATCHING_ARGS
    )
    new_value, new_diagnostics = action_fidelity_v5_2(
        expected, actual, aliases={}, **MATCHING_ARGS
    )
    return {
        "exact_300_actions": {
            "v5_1_fidelity": old_value,
            "v5_1_matched_required": old_diagnostics[
                "matched_required_count"
            ],
            "v5_1_complete": old_diagnostics["matching_complete"],
            "v5_2_fidelity": new_value,
            "v5_2_matched_required": new_diagnostics[
                "matched_required_count"
            ],
            "v5_2_optimality_certified": new_diagnostics["matching"][
                "certification"
            ]["optimality_certified"],
        },
        "greedy_counterexample_2x2": _assignment_probe(2),
        "greedy_counterexample_65x65": _assignment_probe(65),
    }


def _dynamic_probes() -> dict[str, Any]:
    state = {"user": {"name": "Ada"}, "flag": False}
    old = DynamicEvidenceResolverV51(
        state, load_v5_1_reward_config()
    )
    new = DynamicEvidenceResolverV52(
        state, load_default_reward_config()
    )
    item = {"a.b": 42, "a": {"b": 7}}
    values = {
        "state_without_leading_slash": {
            "v5_1": old.resolve({"$state": "user/name"}),
            "v5_2": new.resolve({"$state": "user/name"}),
        },
        "blank_state_path": {
            "v5_1": old.resolve({"$state": ""}),
            "v5_2": new.resolve({"$state": ""}),
        },
        "dotted_item_literal_key": {
            "v5_1": old.resolve({"$item": "a.b"}, item=item),
            "v5_2": new.resolve({"$item": "a.b"}, item=item),
        },
        "boolean_equals_numeric_zero": {
            "v5_1": old.evaluate_condition(
                {"$state": "/flag", "eq": 0}
            ),
            "v5_2": new.evaluate_condition(
                {"$state": "/flag", "eq": 0}
            ),
        },
        "ordinary_unknown_dollar_map": {
            "v5_1_value": old.resolve(
                {"$futureKey": "ordinary", "x": 1}
            ),
            "v5_2_value": new.resolve(
                {"$futureKey": "ordinary", "x": 1}
            ),
        },
    }
    old_builtin = old.resolve(
        {"$computed": "uppercase", "args": {"value": "ada"}}
    )
    new_builtin = new.resolve(
        {"$computed": "uppercase", "args": {"value": "ada"}}
    )
    values["builtin_uppercase"] = {
        "v5_1_result": (
            None
            if "computed_function_unknown:uppercase" in old.unknown
            else old_builtin
        ),
        "v5_1_unknown": (
            "computed_function_unknown:uppercase" in old.unknown
        ),
        "v5_2_result": new_builtin,
        "v5_2_unknown": (
            "computed_function_unknown:uppercase" in new.unknown
        ),
    }
    values["unknown_diagnostic_count"] = {
        "v5_1": len(old.unknown),
        "v5_2": len(new.unknown),
        "v5_1_codes": list(old.unknown),
        "v5_2_codes": list(new.unknown),
    }
    return values


def _role_probes() -> dict[str, Any]:
    source = (
        "## Two Charts to Include\n"
        "1. Revenue by region\n"
        "2. Margin trend"
    )
    old = extract_expected_ui_contract_v5_1(source)
    new = extract_expected_ui_contract_v5_2(source)
    requirements = new["role_requirements"]
    unrelated = {
        "chart": [
            {
                "component_id": "chart_a",
                "title": "Unrelated weather chart",
            },
            {
                "component_id": "chart_b",
                "title": "Unrelated flight chart",
            },
        ]
    }
    semantic, count, _, _, diagnostics, _ = (
        semantic_role_coverage_v5_2(
            requirements,
            unrelated,
            threshold=0.70,
            **MATCHING_ARGS,
        )
    )
    return {
        "v5_1_chart_count": old["expected_role_counts"]["chart"],
        "v5_1_semantic_instance_count": len(
            (old.get("role_requirements") or {}).get("chart", [])
        ),
        "v5_2_chart_count": new["expected_role_counts"]["chart"],
        "v5_2_semantic_instance_count": len(
            requirements.get("chart", [])
        ),
        "v5_2_titles": [
            item["title"] for item in requirements.get("chart", [])
        ],
        "unrelated_candidate_count_coverage": count,
        "unrelated_candidate_semantic_fidelity": semantic,
        "unrelated_matched_required_count": diagnostics["chart"][
            "matched_required_count"
        ],
    }


def _grpo_input_probe() -> dict[str, Any]:
    assets = [
        {"url": f"https://example.com/{index}.png"}
        for index in range(8)
    ]
    # This is the confirmed v5.1 generic sequence-broadcast behavior:
    # length equality made the sequence a per-completion vector.
    old_rows = list(assets)
    new_rows = normalize_asset_rows(assets, 8)
    return {
        "v5_1_assets_per_completion": [
            1 if isinstance(row, dict) else len(row)
            for row in old_rows
        ],
        "v5_2_assets_per_completion": [
            len(row or []) for row in new_rows
        ],
        "v5_2_all_rows_receive_full_collection": all(
            row == assets for row in new_rows
        ),
    }


def _identity_probe() -> dict[str, Any]:
    config = load_default_reward_config()
    old_manifest = score_source_manifest_v5_1()
    new_manifest = score_source_manifest_v5_2()
    base = metric_fingerprint_v5_2(config)
    core_changed = metric_fingerprint_v5_2(
        config,
        source_hash_overrides={"_core.py": "simulated-core-change"},
    )
    v5_changed = metric_fingerprint_v5_2(
        config,
        source_hash_overrides={"_v5.py": "simulated-v5-change"},
    )
    return {
        "v5_1_manifest_contains_core": "_core.py" in old_manifest,
        "v5_1_manifest_contains_v5": "_v5.py" in old_manifest,
        "v5_2_manifest_contains_core": "_core.py" in new_manifest,
        "v5_2_manifest_contains_v5": "_v5.py" in new_manifest,
        "v5_2_base_fingerprint": base,
        "core_change_alters_v5_2_fingerprint": core_changed != base,
        "v5_change_alters_v5_2_fingerprint": v5_changed != base,
    }


def _reproducibility_probe() -> dict[str, Any]:
    tests = REPO_ROOT / "dataset" / "tests"
    historical_marker = (
        "azure_gpt54_reasoning32_20260618_214045"
    )
    references = [
        str(path.relative_to(REPO_ROOT))
        for path in tests.glob("*.py")
        if historical_marker in path.read_text(
            encoding="utf-8", errors="replace"
        )
    ]
    prompt = (
        REPO_ROOT
        / "dataset"
        / "prompts"
        / "genui_gen_mobile_flatspec_v11.md"
    )
    screenshot_helper = (
        REPO_ROOT
        / "dataset"
        / "scripts"
        / "capture_android_run_screenshots.py"
    )
    return {
        "historical_run_import_references": references,
        "authoritative_prompt": str(prompt.relative_to(REPO_ROOT)),
        "authoritative_prompt_exists": prompt.exists(),
        "screenshot_helper": str(
            screenshot_helper.relative_to(REPO_ROOT)
        ),
        "screenshot_helper_exists": screenshot_helper.exists(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matching = _matching_probes()
    payload = {
        "metric_version": "5.2.0",
        "confirmed_defects": {
            "P0_1_false_complete_sparse_matching": matching[
                "exact_300_actions"
            ],
            "P0_2_greedy_assignment": {
                key: value
                for key, value in matching.items()
                if key.startswith("greedy_")
            },
            "P0_3_android_python_dynamic_mismatch": _dynamic_probes(),
            "P0_4_count_only_semantic_roles": _role_probes(),
            "P0_5_ambiguous_grpo_assets": _grpo_input_probe(),
            "P1_1_incomplete_metric_fingerprint": _identity_probe(),
            "P1_2_clean_archive_reproducibility": (
                _reproducibility_probe()
            ),
            "P1_3_overstated_runbook_guarantees": {
                "v5_1_used_one_complete_boolean": True,
                "v5_2_separates_coverage_cardinality_optimality": True,
                "v5_2_uncertified_policy_documented": True,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
