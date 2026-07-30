#!/usr/bin/env python3
"""Reproduce the six v5.2-to-v5.3 correction probes."""

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

from pipeline.genui_quality import (  # noqa: E402
    extract_expected_ui_contract_v5_2,
    extract_expected_ui_contract_v5_3,
    load_default_reward_config,
    load_v5_3_reward_config,
    prepare_source_context_v5_3,
    score_genui_completion_v5_2,
    score_genui_completion_v5_3,
)
from pipeline.genui_quality._core import ActionRef, OutputAction  # noqa: E402
from pipeline.genui_quality.evidence_v5_2 import (  # noqa: E402
    DynamicEvidenceResolverV52,
)
from pipeline.genui_quality.evidence_v5_3 import (  # noqa: E402
    DynamicEvidenceResolverV53,
)
from pipeline.genui_quality.metrics_v5_2 import (  # noqa: E402
    action_fidelity_v5_2,
)
from pipeline.genui_quality.metrics_v5_3 import (  # noqa: E402
    action_exact_key_expected_v5_3,
    action_fidelity_v5_3,
)


MATCHING = {
    "exact_dense_limit": 64,
    "max_edges": 65_536,
    "top_k": 16,
}


def simple(text: str) -> dict[str, Any]:
    return {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {},
                "children": ["body"],
            },
            "body": {
                "type": "Text",
                "props": {"text": text},
                "children": [],
            },
        },
    }


def repeat(count: int) -> tuple[str, dict[str, Any]]:
    values = [f"Item {index}" for index in range(count)]
    return "\n".join(values), {
        "root": "root",
        "state": {"rows": [{"name": value} for value in values]},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {},
                "children": ["list"],
            },
            "list": {
                "type": "List",
                "props": {},
                "children": [],
                "repeat": {
                    "statePath": "/rows",
                    "itemTemplate": "template",
                },
            },
            "template": {
                "type": "Text",
                "props": {"text": "{{$item/name}}"},
                "children": [],
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = "A short greeting."
    candidate = simple(source)
    candidate["elements"]["table"] = {
        "type": "Table",
        "props": {
            "columns": [{"key": "x", "label": "X"}],
            "rows": [{"x": "1"}],
        },
        "children": [],
    }
    candidate["elements"]["root"]["children"].append("table")
    before_applicability = score_genui_completion_v5_2(candidate, source)
    after_applicability = score_genui_completion_v5_3(candidate, source)

    role_source = "Create a bar chart showing quarterly revenue."
    role_before = extract_expected_ui_contract_v5_2(role_source)
    role_after = extract_expected_ui_contract_v5_3(role_source)

    count = 300
    expected = [
        ActionRef(
            label=f"Open {index}",
            url=f"https://example.com/{index}",
            id=f"requirement-{index}",
        )
        for index in range(count)
    ]
    actual = [
        OutputAction(
            element_id=f"button-{index}",
            label=f"Open {index}",
            url=f"https://example.com/{index}",
            source="on.press",
        )
        for index in range(count)
    ]
    old_action, old_action_diag = action_fidelity_v5_2(
        expected, actual, aliases={}, **MATCHING
    )
    prepared = prepare_source_context_v5_3(
        "\n".join(
            f"Action: [Button: Open {index}] https://example.com/{index}"
            for index in range(count)
        )
    )
    new_action, new_action_diag = action_fidelity_v5_3(
        expected,
        actual,
        aliases={},
        expected_exact_index={
            key: tuple(values)
            for key, values in prepared.action_exact_index.items()
        },
        **MATCHING,
    )

    equality_expression = {
        "$cond": {
            "value": {
                "$computed": "sum",
                "args": {"values": [1, 2]},
            },
            "eq": 3,
        },
        "$then": "equal",
        "$else": "typed-not-equal",
    }
    old_resolver = DynamicEvidenceResolverV52(
        {}, load_default_reward_config()
    )
    new_resolver = DynamicEvidenceResolverV53(
        {}, load_v5_3_reward_config()
    )

    repeat_rows: dict[str, Any] = {}
    for size in (32, 33):
        repeat_source, repeat_candidate = repeat(size)
        before = score_genui_completion_v5_2(
            repeat_candidate, repeat_source
        )
        after = score_genui_completion_v5_3(
            repeat_candidate, repeat_source
        )
        repeat_rows[str(size)] = {
            "v5_2": {
                "quality_0_100": before.quality_0_100,
                "dynamic": before.dynamic_semantics,
                "active_caps": before.active_caps,
            },
            "v5_3": {
                "quality_0_100": after.quality_0_100,
                "dynamic_certification": (
                    after.dynamic_evidence_certification
                ),
                "active_caps": after.active_caps,
            },
        }

    score = score_genui_completion_v5_3(simple("Hello"), "Hello")
    payload = {
        "metric_versions": {"before": "5.2.0", "after": "5.3.0"},
        "candidate_driven_applicability": {
            "before_table_fidelity": before_applicability.atomics[
                "fidelity"
            ]["markdown_table_fidelity"],
            "after_table_fidelity": after_applicability.atomics[
                "fidelity"
            ]["markdown_table_fidelity"],
            "after_applicable": after_applicability.atomic_applicability[
                "fidelity.markdown_table_fidelity"
            ],
        },
        "ordinary_role_extraction": {
            "before": role_before.get("role_requirements"),
            "after": role_after.get("role_requirements"),
            "after_specificity": role_after.get(
                "contract_semantic_specificity"
            ),
        },
        "generic_requirement_ids": {
            "before_fidelity": old_action,
            "before_exact_preallocated_count": old_action_diag[
                "matching"
            ].get("exact_preallocated_count"),
            "after_fidelity": new_action,
            "after_exact_preallocated_count": new_action_diag[
                "matching"
            ].get("exact_preallocated_count"),
            "after_certified": new_action_diag["matching"][
                "certification"
            ]["optimality_certified"],
            "example_v5_3_key": repr(
                action_exact_key_expected_v5_3(expected[0], {})
            ),
        },
        "android_typed_equality": {
            "before": old_resolver.resolve(equality_expression),
            "after": new_resolver.resolve(equality_expression),
        },
        "repeat_cliff": repeat_rows,
        "prepared_indices_and_grpo_identity": {
            "prepared_content_key_count": len(
                prepare_source_context_v5_3("Hello").content_exact_index
            ),
            "metric_fingerprint": score.metric_fingerprint,
            "reward_pipeline_fingerprint": (
                score.reward_pipeline_fingerprint
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
