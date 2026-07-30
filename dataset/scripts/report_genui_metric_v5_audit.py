#!/usr/bin/env python3
"""Write reproducible v4/v5 run comparisons and confirmed-defect probes."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping


DATASET_ROOT = Path(__file__).resolve().parents[1]
DATASET_SRC = DATASET_ROOT / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.genui_quality import (  # noqa: E402
    extract_expected_ui_contract,
    breakdown_to_mapping,
    genui_grpo_reward,
    load_default_reward_config,
    load_v4_reward_config,
    metric_fingerprint,
    resolve_expected_ui_contract,
    score_genui_completion,
    score_genui_completion_v4,
    score_record,
    source_contract_cache_key,
)
from pipeline.genui_quality._v5 import allocate_capped_atomic_weights  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def simple_spec(text: str = "Hello world") -> dict[str, Any]:
    return {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {"direction": "vertical"},
                "children": ["body"],
            },
            "body": {
                "type": "Text",
                "props": {"text": text},
                "children": [],
            },
        },
    }


def stamped_contract(source: str, **updates: Any) -> dict[str, Any]:
    result = extract_expected_ui_contract(source)
    result.update(copy.deepcopy(updates))
    return result


def old_score(
    candidate: Any,
    source: str,
    contract: Mapping[str, Any] | None = None,
):
    return score_genui_completion_v4(
        candidate,
        source,
        expected_ui_contract=contract,
        config=load_v4_reward_config(),
    )


def new_score(
    candidate: Any,
    source: str,
    contract: Mapping[str, Any] | None = None,
):
    return score_genui_completion(
        candidate,
        source,
        expected_ui_contract=contract,
    )


def cap_names(result: Any) -> list[str]:
    return sorted(
        str(item.get("name"))
        for item in result.active_caps
        if isinstance(item, Mapping) and item.get("name")
    )


def confirmed_probes() -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []

    padded = simple_spec()
    for index in range(100):
        element_id = f"valid_{index}"
        padded["elements"][element_id] = {
            "type": "Card",
            "props": {},
            "children": [],
        }
        padded["elements"]["root"]["children"].append(element_id)
    padded["elements"]["unsupported"] = {
        "type": "ProductionUnsupported",
        "props": {},
        "children": [],
    }
    padded["elements"]["root"]["children"].append("unsupported")
    old = old_score(padded, "Hello world")
    new = new_score(padded, "Hello world")
    probes.append(
        {
            "probe": "unsupported_type_with_100_valid_nodes",
            "v4_quality_0_100": old.quality_0_100,
            "v5_quality_0_100": new.quality_0_100,
            "v5_cap": new.cap_0_1,
            "v5_production_valid": new.normalization.get("production_valid"),
            "v5_active_caps": cap_names(new),
        }
    )

    clean = simple_spec()
    forbidden = copy.deepcopy(clean)
    forbidden["debug"] = True
    old_clean = old_score(clean, "Hello world")
    old_forbidden = old_score(forbidden, "Hello world")
    new_clean = new_score(clean, "Hello world")
    new_forbidden = new_score(forbidden, "Hello world")
    probes.append(
        {
            "probe": "forbidden_top_level_property",
            "v4_clean": old_clean.quality_0_100,
            "v4_mutated": old_forbidden.quality_0_100,
            "v4_delta": old_forbidden.quality_0_100 - old_clean.quality_0_100,
            "v5_clean": new_clean.quality_0_100,
            "v5_mutated": new_forbidden.quality_0_100,
            "v5_delta": new_forbidden.quality_0_100 - new_clean.quality_0_100,
            "v5_strict_schema_valid": new_forbidden.normalization.get(
                "strict_schema_valid"
            ),
            "v5_unknown_top_level_properties": new_forbidden.normalization.get(
                "unknown_top_level_properties"
            ),
            "v5_active_caps": cap_names(new_forbidden),
        }
    )

    tabs = {
        "root": "tabs",
        "state": {},
        "elements": {
            "tabs": {
                "type": "Tabs",
                "props": {"tabs": [{"child": "panel"}]},
                "children": [],
            },
            "panel": {
                "type": "Modal",
                "props": {"trigger": "trigger", "content": "content"},
                "children": [],
            },
            "trigger": {
                "type": "Button",
                "props": {"label": "Open"},
                "children": [],
            },
            "content": {
                "type": "Text",
                "props": {"text": "Panel content"},
                "children": [],
            },
        },
    }
    old = old_score(tabs, "Panel content")
    new = new_score(tabs, "Panel content")
    probes.append(
        {
            "probe": "tabs_and_modal_nested_references",
            "v4_reachable_elements": old.evidence["output"]["reachable_elements"],
            "v5_reachable_ids": new.evidence["output"]["reachable_ids"],
            "v5_missing_references": new.evidence["output"]["missing_references"],
        }
    )

    table_source = "Item | Price\n--- | ---\nA | 10\nB | 20"
    table_contract = stamped_contract(table_source)
    table = {
        "root": "root",
        "state": {
            "rows": [
                {"item": "A", "price": "10"},
                {"item": "B", "price": "20"},
            ]
        },
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["table"]},
            "table": {
                "type": "Table",
                "props": {
                    "columns": [
                        {"key": "item", "label": "Item"},
                        {"key": "price", "label": "Price"},
                    ],
                    "statePath": "/rows",
                },
                "children": [],
            },
        },
    }
    swapped = copy.deepcopy(table)
    swapped["state"]["rows"][0]["price"], swapped["state"]["rows"][1]["price"] = (
        swapped["state"]["rows"][1]["price"],
        swapped["state"]["rows"][0]["price"],
    )
    old_good = old_score(table, table_source, table_contract)
    old_bad = old_score(swapped, table_source, table_contract)
    new_good = new_score(table, table_source, table_contract)
    new_bad = new_score(swapped, table_source, table_contract)
    probes.append(
        {
            "probe": "table_row_value_swap",
            "v4_table_atomic_before": old_good.atomics["fidelity"][
                "markdown_table_fidelity"
            ],
            "v4_table_atomic_after": old_bad.atomics["fidelity"][
                "markdown_table_fidelity"
            ],
            "v5_table_atomic_before": new_good.atomics["fidelity"][
                "markdown_table_fidelity"
            ],
            "v5_table_atomic_after": new_bad.atomics["fidelity"][
                "markdown_table_fidelity"
            ],
            "v5_quality_delta": new_bad.quality_0_100 - new_good.quality_0_100,
        }
    )

    ordered_source = "First alpha statement.\nSecond beta statement."
    ordered_contract = stamped_contract(ordered_source)
    ordered = simple_spec("")
    ordered["elements"].pop("body")
    ordered["elements"]["root"]["children"] = ["first", "second"]
    ordered["elements"]["first"] = {
        "type": "Text",
        "props": {"text": "First alpha statement."},
        "children": [],
    }
    ordered["elements"]["second"] = {
        "type": "Text",
        "props": {"text": "Second beta statement."},
        "children": [],
    }
    reversed_output = copy.deepcopy(ordered)
    reversed_output["elements"]["root"]["children"].reverse()
    old_good = old_score(ordered, ordered_source, ordered_contract)
    old_bad = old_score(reversed_output, ordered_source, ordered_contract)
    new_good = new_score(ordered, ordered_source, ordered_contract)
    new_bad = new_score(reversed_output, ordered_source, ordered_contract)
    probes.append(
        {
            "probe": "ordered_content_reversal",
            "v4_quality_before": old_good.quality_0_100,
            "v4_quality_after": old_bad.quality_0_100,
            "v5_order_atomic_before": new_good.atomics["fidelity"][
                "content_order_preservation"
            ],
            "v5_order_atomic_after": new_bad.atomics["fidelity"][
                "content_order_preservation"
            ],
            "v5_quality_delta": new_bad.quality_0_100 - new_good.quality_0_100,
        }
    )

    action_source = "Action: [Button: Open account] https://example.com/open"
    action_contract = stamped_contract(action_source)
    wrong_action = simple_spec("Open account")
    wrong_action["elements"]["button"] = {
        "type": "Button",
        "props": {"label": "Open account"},
        "children": [],
        "on": {
            "press": {
                "action": "openUrl",
                "params": {"url": "https://wrong.invalid/"},
            }
        },
    }
    wrong_action["elements"]["root"]["children"].append("button")
    action = new_score(wrong_action, action_source, action_contract)
    probes.append(
        {
            "probe": "wrong_action_destination",
            "v5_matched_required_count": action.evidence["action_matching"][
                "matched_required_count"
            ],
            "v5_action_role_coverage": action.evidence["role_values"]["action"],
            "v5_active_caps": cap_names(action),
        }
    )

    chart_contract = stamped_contract(
        "Quarterly charts",
        required_roles={"chart": 2},
        expected_role_counts={"chart": 2},
    )
    charts = simple_spec("Quarterly charts")
    charts["state"]["series"] = [{"quarter": "Q1", "value": 10}]
    for index in range(3):
        charts["elements"][f"chart_{index}"] = {
            "type": "Chart",
            "props": {
                "title": "Same chart",
                "columns": [
                    {"key": "quarter", "label": "Quarter"},
                    {"key": "value", "label": "Value"},
                ],
                "statePath": "/series",
                "xKey": "quarter",
                "yKey": "value",
            },
            "children": [],
        }
        charts["elements"]["root"]["children"].append(f"chart_{index}")
    chart = new_score(charts, "Quarterly charts", chart_contract)
    probes.append(
        {
            "probe": "duplicate_identical_charts",
            "candidate_chart_count": 3,
            "required_distinct_count": 2,
            "v5_distinct_role_coverage": chart.evidence["role_values"]["chart"],
            "v5_active_caps": cap_names(chart),
        }
    )

    media_source = (
        "Media: Image = https://example.com/a.png Alt = A\n"
        "Media: Image = https://example.com/b.png Alt = B"
    )
    media_contract = stamped_contract(media_source)
    media = simple_spec("A and B")
    media["elements"]["image"] = {
        "type": "Image",
        "props": {"url": "https://example.com/a.png", "alt": "A"},
        "children": [],
    }
    media["elements"]["root"]["children"].append("image")
    media_result = new_score(media, media_source, media_contract)
    probes.append(
        {
            "probe": "one_image_for_two_required_images",
            "v5_required_count": media_result.evidence["media_matching"][
                "required_count"
            ],
            "v5_matched_required_count": media_result.evidence["media_matching"][
                "matched_required_count"
            ],
            "v5_image_role_coverage": media_result.evidence["role_values"]["image"],
            "v5_active_caps": cap_names(media_result),
        }
    )

    optional_contract = stamped_contract(
        "Optional information",
        actions=[
            {
                "id": "optional-action",
                "label": "Open",
                "target": "https://example.com",
                "required": False,
            }
        ],
        tables=[
            {
                "id": "optional-table",
                "headers": ["A"],
                "rows": [["1"]],
                "required": False,
            }
        ],
        media=[
            {
                "id": "optional-media",
                "kind": "Image",
                "url": "https://example.com/a.png",
                "required": False,
                "media_policy": "exact",
            }
        ],
        required_roles={"action": 0, "table": 0, "image": 0},
        expected_role_counts={"action": 0, "table": 0, "image": 0},
    )
    optional = new_score(
        simple_spec("Optional information"),
        "Optional information",
        optional_contract,
    )
    probes.append(
        {
            "probe": "optional_requirements_omitted",
            "v5_action_atomic": optional.atomics["fidelity"][
                "action_and_source_link_fidelity"
            ],
            "v5_table_atomic": optional.atomics["fidelity"][
                "markdown_table_fidelity"
            ],
            "v5_media_atomic": optional.atomics["fidelity"]["media_fidelity"],
            "v5_active_caps": cap_names(optional),
        }
    )

    repeat_source = "Alpha 1\nBeta 2"
    repeat_contract = stamped_contract(repeat_source)
    repeated = {
        "root": "root",
        "state": {
            "items": [
                {"name": "Alpha", "value": 1},
                {"name": "Beta", "value": 2},
            ]
        },
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["repeat"]},
            "repeat": {
                "type": "Column",
                "props": {},
                "children": ["row"],
                "repeat": {"statePath": "/items"},
            },
            "row": {
                "type": "Text",
                "props": {
                    "text": {"$template": "${$item.name} ${$item.value}"}
                },
                "children": [],
            },
        },
    }
    static = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["a", "b"]},
            "a": {"type": "Text", "props": {"text": "Alpha 1"}, "children": []},
            "b": {"type": "Text", "props": {"text": "Beta 2"}, "children": []},
        },
    }
    repeated_result = new_score(repeated, repeat_source, repeat_contract)
    static_result = new_score(static, repeat_source, repeat_contract)
    probes.append(
        {
            "probe": "repeat_vs_static_equivalent",
            "v5_repeat_quality": repeated_result.quality_0_100,
            "v5_static_quality": static_result.quality_0_100,
            "absolute_delta": abs(
                repeated_result.quality_0_100 - static_result.quality_0_100
            ),
            "v5_dynamic_unknowns": repeated_result.evidence["output"][
                "dynamic_unknowns"
            ],
        }
    )

    config = load_default_reward_config()
    atomics = {
        dimension: {name: 0.5 for name in values}
        for dimension, values in config.atomic_weights.items()
    }
    weights, _, feasible = allocate_capped_atomic_weights(atomics, config)
    probes.append(
        {
            "probe": "global_atomic_budget",
            "feasible": feasible,
            "applicable_atomic_count": len(weights),
            "effective_weight_sum": sum(weights.values()),
            "maximum_effective_atomic_weight": max(weights.values()),
            "configured_maximum": config.max_atomic_global_weight,
        }
    )

    cache_source = "Intent-sensitive contract"
    probes.append(
        {
            "probe": "intent_sensitive_source_contract_cache",
            "booking_cache_key": source_contract_cache_key(
                cache_source, intent="booking"
            ),
            "technical_support_cache_key": source_contract_cache_key(
                cache_source, intent="technical_support"
            ),
            "keys_differ": source_contract_cache_key(
                cache_source, intent="booking"
            )
            != source_contract_cache_key(
                cache_source, intent="technical_support"
            ),
        }
    )

    stale_contract = stamped_contract("Fresh source")
    stale_contract["source_hash"] = "0" * 64
    stale_resolution = resolve_expected_ui_contract(
        "Fresh source",
        persisted=stale_contract,
        persisted_source="human benchmark",
    )
    probes.append(
        {
            "probe": "stale_persisted_source_hash",
            "resolved_source": stale_resolution.source,
            "accepted_as_human_benchmark": (
                stale_resolution.source == "human benchmark"
            ),
            "resolution_errors": list(stale_resolution.errors),
        }
    )

    reuse_source = "Reuse identity"
    reuse_candidate = simple_spec(reuse_source)
    reuse_contract = stamped_contract(reuse_source)
    reuse_result = new_score(reuse_candidate, reuse_source, reuse_contract)
    reuse_record = {
        "response_text": reuse_source,
        "genui_raw_completion": reuse_candidate,
        "genui_json": reuse_candidate,
        "expected_ui_contract": reuse_contract,
        "genui_quality_v5": breakdown_to_mapping(reuse_result),
    }
    default_config = load_default_reward_config()
    changed_config = replace(
        default_config,
        content_beta=default_config.content_beta + 0.25,
    )
    reused = score_record(reuse_record, config=default_config)
    recomputed = score_record(reuse_record, config=changed_config)
    probes.append(
        {
            "probe": "stored_score_fingerprint_reuse",
            "default_fingerprint": metric_fingerprint(default_config),
            "changed_fingerprint": metric_fingerprint(changed_config),
            "matching_identity_reused": reused.evidence["score_reuse"]["reused"],
            "changed_config_reused": recomputed.evidence["score_reuse"]["reused"],
            "changed_config_stale_reasons": recomputed.evidence["score_reuse"][
                "stale_reasons"
            ],
        }
    )

    raw_alias = copy.deepcopy(reuse_candidate)
    raw_alias["elements"]["root"]["type"] = "column"
    raw_alias_text = json.dumps(raw_alias, separators=(",", ":"))
    offline = score_genui_completion(raw_alias_text, reuse_source)
    grpo_reward = genui_grpo_reward([raw_alias_text], [reuse_source])[0]
    probes.append(
        {
            "probe": "shared_raw_candidate_boundary",
            "offline_reward": offline.reward,
            "grpo_reward": grpo_reward,
            "rewards_identical": offline.reward == grpo_reward,
            "production_valid": offline.normalization["production_valid"],
            "strict_schema_valid": offline.normalization["strict_schema_valid"],
            "raw_format_utility": offline.normalization["raw_format_utility"],
            "metric_version": offline.metric_version,
        }
    )
    return probes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecar_dir")
    parser.add_argument("runs", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sidecar = Path(args.sidecar_dir).resolve()
    scores_path = sidecar / "scores.jsonl"
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    new_by_key = {
        (str(row.get("source_run")), str(row.get("ui_id"))): row
        for row in iter_jsonl(scores_path)
    }
    rows: list[dict[str, Any]] = []
    for supplied in args.runs:
        run = Path(supplied).resolve()
        for record in iter_jsonl(run / "genui.jsonl"):
            key = (str(run), str(record.get("ui_id")))
            sidecar_row = new_by_key[key]
            old = record.get("genui_quality_v4")
            if not isinstance(old, Mapping):
                continue
            new = sidecar_row["genui_quality_v5"]
            rows.append(
                {
                    "source_run": run.name,
                    "ui_id": record.get("ui_id"),
                    "intent_bucket": record.get("intent_bucket"),
                    "model": (
                        record.get("gen", {}).get("model")
                        if isinstance(record.get("gen"), Mapping)
                        else None
                    ),
                    "v4_quality_0_100": float(old["quality_0_100"]),
                    "v5_quality_0_100": float(new["quality_0_100"]),
                    "delta_v5_minus_v4": (
                        float(new["quality_0_100"])
                        - float(old["quality_0_100"])
                    ),
                    "v5_active_caps": cap_names(
                        type("_Result", (), {"active_caps": new.get("active_caps", [])})
                    ),
                }
            )

    summaries: list[dict[str, Any]] = []
    for run_name in sorted({row["source_run"] for row in rows}):
        selected = [row for row in rows if row["source_run"] == run_name]
        old_values = [row["v4_quality_0_100"] for row in selected]
        new_values = [row["v5_quality_0_100"] for row in selected]
        summaries.append(
            {
                "source_run": run_name,
                "count": len(selected),
                "v4_mean": statistics.fmean(old_values),
                "v5_mean": statistics.fmean(new_values),
                "mean_delta_v5_minus_v4": (
                    statistics.fmean(new_values) - statistics.fmean(old_values)
                ),
                "v4_median": statistics.median(old_values),
                "v5_median": statistics.median(new_values),
            }
        )
    old_all = [row["v4_quality_0_100"] for row in rows]
    new_all = [row["v5_quality_0_100"] for row in rows]
    comparison = {
        "metric_versions": {"before": "4.0.0", "after": "5.0.0"},
        "count": len(rows),
        "combined": {
            "v4_mean": statistics.fmean(old_all),
            "v5_mean": statistics.fmean(new_all),
            "mean_delta_v5_minus_v4": (
                statistics.fmean(new_all) - statistics.fmean(old_all)
            ),
            "v4_median": statistics.median(old_all),
            "v5_median": statistics.median(new_all),
        },
        "by_run": summaries,
        "rows": rows,
        "calibration_note": (
            "Descriptive non-gating audit only. No weights or thresholds were "
            "fit to these 20 rows."
        ),
    }
    (sidecar / "v4_vs_v5.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    with (sidecar / "v4_vs_v5.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_run",
                "ui_id",
                "intent_bucket",
                "model",
                "v4_quality_0_100",
                "v5_quality_0_100",
                "delta_v5_minus_v4",
                "v5_active_caps",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "v5_active_caps": "|".join(row["v5_active_caps"]),
                }
            )

    probes = {
        "metric_versions": {"before": "4.0.0", "after": "5.0.0"},
        "probes": confirmed_probes(),
    }
    (sidecar / "confirmed_defect_probes.json").write_text(
        json.dumps(probes, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sidecar_dir": str(sidecar),
                "comparison_rows": len(rows),
                "probe_count": len(probes["probes"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
