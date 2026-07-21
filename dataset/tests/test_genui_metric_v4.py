from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_quality import (  # noqa: E402
    RewardConfig,
    genui_grpo_reward,
    load_reward_config,
    score_genui_completion,
)

ROWS = [
    json.loads(line)
    for line in (
        DATASET_ROOT
        / "data"
        / "runs"
        / "azure_gpt54_reasoning32_20260618_214045"
        / "gpt54_reasoning_medium"
        / "genui.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
BY_INTENT = {row["intent_bucket"]: row for row in ROWS}


def dump(spec: dict) -> str:
    return json.dumps(spec, ensure_ascii=False, separators=(",", ":"))


def score(row: dict, spec: dict, *, contract=None, render_ok=None):
    return score_genui_completion(
        dump(spec),
        row["response_text"],
        intent=row.get("intent_bucket"),
        assets=row.get("assets"),
        expected_ui_contract=contract,
        render_ok=render_ok,
    )


def simple_spec(text: str = "Hello world") -> dict:
    return {
        "root": "root",
        "state": {},
        "elements": {
            "root": {
                "type": "Stack",
                "props": {"direction": "vertical", "gap": "md"},
                "children": ["body"],
            },
            "body": {
                "type": "Text",
                "props": {"text": text, "variant": "body"},
                "children": [],
            },
        },
    }


class RewardBehaviorTests(unittest.TestCase):
    def test_batch_wrapper_returns_one_scalar_per_completion(self):
        specs = [simple_spec("Alpha"), simple_spec("Beta")]
        rewards = genui_grpo_reward(
            [dump(s) for s in specs],
            ["Alpha", "Beta"],
            intent_bucket=["documentation", "documentation"],
        )
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(isinstance(v, float) and -1.0 <= v <= 1.0 for v in rewards))

    def test_card_stack_renderer_pattern_is_not_counted_as_redundant(self):
        row = BY_INTENT["technical_support"]
        result = score(row, row["genui_json"])
        self.assertEqual(
            result.atomics["economy"]["renderer_aware_wrapper_economy"], 1.0
        )

    def test_chat_style_completion_is_supported(self):
        row = BY_INTENT["creative_writing"]
        completion = [{"role": "assistant", "content": dump(row["genui_json"])}]
        result = score_genui_completion(
            completion,
            row["response_text"],
            intent=row["intent_bucket"],
            assets=row.get("assets"),
        )
        self.assertGreater(result.quality_0_1, 0.8)

    def test_config_file_loads_and_normalizes(self):
        cfg = load_reward_config(DATASET_ROOT / "configs" / "genui_metric_v4.yaml")
        self.assertAlmostEqual(sum(cfg.dimension_weights.values()), 1.0)
        for group in cfg.atomic_weights.values():
            self.assertAlmostEqual(sum(group.values()), 1.0)
        maximum = max(
            cfg.dimension_weights[d] * w
            for d, group in cfg.atomic_weights.items()
            for w in group.values()
        )
        self.assertLessEqual(maximum, cfg.max_atomic_global_weight + 1e-9)

    def test_contract_explicit_empty_actions_removes_false_action_requirement(self):
        source = "Read more at https://example.com/reference"
        spec = simple_spec("Read more at the reference")
        fallback = score_genui_completion(dump(spec), source)
        explicit = score_genui_completion(
            dump(spec),
            source,
            expected_ui_contract={"actions": [], "required_roles": {"action": 0}},
        )
        self.assertIsNone(
            explicit.atomics["fidelity"]["action_and_source_link_fidelity"]
        )
        self.assertGreaterEqual(explicit.cap_0_1, fallback.cap_0_1)

    def test_cycle_is_non_compensatory(self):
        spec = simple_spec()
        spec["elements"]["body"]["children"] = ["root"]
        result = score_genui_completion(dump(spec), "Hello world")
        self.assertLessEqual(result.quality_0_1, 0.25)
        self.assertTrue(any(c["name"] == "reference_or_cycle" for c in result.active_caps))

    def test_disconnected_padding_never_increases_reward(self):
        row = BY_INTENT["booking"]
        original = score(row, row["genui_json"])
        mutated = copy.deepcopy(row["genui_json"])
        for i in range(55):
            mutated["elements"][f"unreachable_padding_{i}"] = {
                "type": "Card",
                "props": {},
                "children": [],
            }
        padded = score(row, mutated)
        self.assertLessEqual(padded.quality_0_1, original.quality_0_1)
        self.assertLessEqual(padded.cap_0_1, 0.40)

    def test_dropping_table_column_decreases_reward(self):
        row = BY_INTENT["booking"]
        original = score(row, row["genui_json"])
        mutated = copy.deepcopy(row["genui_json"])
        table = next(e for e in mutated["elements"].values() if e.get("type") == "Table")
        columns = table["props"].get("columns", [])
        removed_key = columns[-1]["key"]
        table["props"]["columns"] = columns[:-1]
        path = table["props"].get("statePath", "").lstrip("/")
        if path in mutated.get("state", {}):
            for item in mutated["state"][path]:
                if isinstance(item, dict):
                    item.pop(removed_key, None)
        dropped = score(row, mutated)
        self.assertLess(dropped.quality_0_1, original.quality_0_1)

    def test_dropping_table_row_decreases_reward(self):
        row = BY_INTENT["booking"]
        original = score(row, row["genui_json"])
        mutated = copy.deepcopy(row["genui_json"])
        table = next(e for e in mutated["elements"].values() if e.get("type") == "Table")
        path = table["props"].get("statePath", "").lstrip("/")
        mutated["state"][path] = mutated["state"][path][:-1]
        dropped = score(row, mutated)
        self.assertLess(dropped.quality_0_1, original.quality_0_1)

    def test_duplicate_visible_content_decreases_reward(self):
        row = BY_INTENT["booking"]
        original = score(row, row["genui_json"])
        mutated = copy.deepcopy(row["genui_json"])
        mutated["elements"]["duplicate_title"] = {
            "type": "Text",
            "props": {"text": "Best Hotel for Your Rome Family Stay", "variant": "h2"},
            "children": [],
        }
        mutated["elements"][mutated["root"]]["children"].append("duplicate_title")
        duplicated = score(row, mutated)
        self.assertLess(duplicated.quality_0_1, original.quality_0_1)

    def test_element_map_order_is_invariant(self):
        row = BY_INTENT["recipe"]
        spec = row["genui_json"]
        reordered = copy.deepcopy(spec)
        reordered["elements"] = OrderedDict(reversed(list(reordered["elements"].items())))
        self.assertAlmostEqual(score(row, spec).quality_0_1, score(row, reordered).quality_0_1, places=12)

    def test_equivalent_root_layout_type_is_nearly_invariant(self):
        source = "Hello world"
        a = simple_spec()
        b = copy.deepcopy(a)
        b["elements"]["root"]["type"] = "Column"
        b["elements"]["root"]["props"].pop("direction", None)
        qa = score_genui_completion(dump(a), source).quality_0_1
        qb = score_genui_completion(dump(b), source).quality_0_1
        self.assertLess(abs(qa - qb), 0.03)

    def test_expected_contract_can_require_role_without_heuristic_phrase(self):
        spec = simple_spec("Quarterly results")
        result = score_genui_completion(
            dump(spec),
            "Quarterly results",
            expected_ui_contract={"required_roles": {"chart": 1}},
        )
        self.assertLessEqual(result.cap_0_1, 0.78)

    def test_image_without_alt_decreases_accessibility(self):
        source = "A product image"
        contract = {
            "media": [{"kind": "Image", "url": "https://example.com/a.png", "alt": "Product"}],
            "required_roles": {"image": 1},
        }
        good = simple_spec("Product")
        good["elements"]["image"] = {
            "type": "Image",
            "props": {"url": "https://example.com/a.png", "alt": "Product"},
            "children": [],
        }
        good["elements"]["root"]["children"].append("image")
        bad = copy.deepcopy(good)
        bad["elements"]["image"]["props"].pop("alt")
        q_good = score_genui_completion(dump(good), source, expected_ui_contract=contract)
        q_bad = score_genui_completion(dump(bad), source, expected_ui_contract=contract)
        self.assertGreater(q_good.dimensions["accessibility"], q_bad.dimensions["accessibility"])
        self.assertGreater(q_good.quality_0_1, q_bad.quality_0_1)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            RewardConfig(dimension_weights={"integrity": -1.0, "fidelity": 2.0})
        with self.assertRaises(ValueError):
            RewardConfig(good_json_to_source_ratio=7.0, bad_json_to_source_ratio=3.0)

    def test_invalid_json_gets_dense_low_reward(self):
        result = score_genui_completion('{"root":', "Hello")
        self.assertGreaterEqual(result.reward, -1.0)
        self.assertLess(result.quality_0_1, 0.10)
        self.assertTrue(math.isfinite(result.reward))

    def test_missing_reference_is_non_compensatory(self):
        spec = simple_spec()
        spec["elements"]["root"]["children"].append("missing")
        result = score_genui_completion(dump(spec), "Hello world")
        self.assertLessEqual(result.quality_0_1, 0.25)

    def test_missing_required_action_activates_cap(self):
        result = score_genui_completion(
            dump(simple_spec("Open account")),
            "Open account",
            expected_ui_contract={
                "actions": [{"label": "Open account", "target": "https://example.com/open"}],
                "required_roles": {"action": 1},
            },
        )
        self.assertLessEqual(result.cap_0_1, 0.75)

    def test_missing_required_chart_activates_cap(self):
        result = score_genui_completion(
            dump(simple_spec("Quarterly results")),
            "Quarterly results",
            expected_ui_contract={"required_roles": {"chart": 1}},
        )
        self.assertLessEqual(result.cap_0_1, 0.78)

    def test_missing_required_formula_activates_cap(self):
        result = score_genui_completion(
            dump(simple_spec("Calculation result")),
            "Calculation result",
            expected_ui_contract={"required_roles": {"formula": 1}},
        )
        self.assertLessEqual(result.cap_0_1, 0.78)

    def test_missing_required_image_is_continuous_penalty(self):
        contract = {
            "media": [{"kind": "Image", "url": "https://example.com/a.png", "alt": "A"}],
            "required_roles": {"image": 1},
        }
        missing = score_genui_completion(dump(simple_spec("A")), "A", expected_ui_contract=contract)
        present_spec = simple_spec("A")
        present_spec["elements"]["image"] = {
            "type": "Image", "props": {"url": "https://example.com/a.png", "alt": "A"}, "children": []
        }
        present_spec["elements"]["root"]["children"].append("image")
        present = score_genui_completion(dump(present_spec), "A", expected_ui_contract=contract)
        self.assertEqual(missing.cap_0_1, 1.0)
        self.assertGreater(present.quality_0_1, missing.quality_0_1)

    def test_missing_root_is_non_compensatory(self):
        spec = simple_spec()
        spec["root"] = "does_not_exist"
        result = score_genui_completion(dump(spec), "Hello world")
        self.assertLessEqual(result.quality_0_1, 0.10)

    def test_non_applicable_accessibility_is_not_free_credit(self):
        result = score_genui_completion(dump(simple_spec()), "Hello world")
        self.assertIsNone(result.dimensions["accessibility"])
        self.assertFalse(
            result.evidence["atomic_applicability"]["accessibility"]["applicable_accessibility_contracts"]
        )

    def test_over_fragmenting_text_does_not_improve_reward(self):
        source = "A clear paragraph with enough words to represent one useful content block."
        original = simple_spec(source)
        fragmented = copy.deepcopy(original)
        fragmented["elements"].pop("body")
        children = []
        for i, word in enumerate(source.split()):
            eid = f"w{i}"
            fragmented["elements"][eid] = {
                "type": "Text", "props": {"text": word, "variant": "body"}, "children": []
            }
            children.append(eid)
        fragmented["elements"]["root"]["children"] = children
        q_original = score_genui_completion(dump(original), source)
        q_fragmented = score_genui_completion(dump(fragmented), source)
        self.assertLessEqual(q_fragmented.quality_0_1, q_original.quality_0_1)

    def test_reachable_empty_component_never_increases_reward(self):
        row = BY_INTENT["booking"]
        original = score(row, row["genui_json"])
        mutated = copy.deepcopy(row["genui_json"])
        mutated["elements"]["empty_card"] = {"type": "Card", "props": {}, "children": []}
        mutated["elements"][mutated["root"]]["children"].append("empty_card")
        changed = score(row, mutated)
        self.assertLessEqual(changed.quality_0_1, original.quality_0_1)

    def test_semantically_equivalent_id_renaming_is_invariant(self):
        row = BY_INTENT["recipe"]
        spec = row["genui_json"]
        mapping = {old: f"renamed_{i}" for i, old in enumerate(spec["elements"])}
        renamed = copy.deepcopy(spec)
        renamed["root"] = mapping[spec["root"]]
        renamed_elements = {}
        for old_id, element in spec["elements"].items():
            new_element = copy.deepcopy(element)
            new_element["children"] = [mapping.get(c, c) for c in new_element.get("children", [])]
            renamed_elements[mapping[old_id]] = new_element
        renamed["elements"] = renamed_elements
        self.assertAlmostEqual(score(row, spec).quality_0_1, score(row, renamed).quality_0_1, places=12)

    def test_unrelated_extra_action_decreases_reward(self):
        row = BY_INTENT["booking"]
        original = score(row, row["genui_json"])
        mutated = copy.deepcopy(row["genui_json"])
        mutated["elements"]["unrelated_action"] = {
            "type": "Button",
            "props": {"label": "Unrelated"},
            "children": [],
            "on": {"press": {"action": "openUrl", "params": {"url": "https://unrelated.invalid/"}}},
        }
        mutated["elements"][mutated["root"]]["children"].append("unrelated_action")
        changed = score(row, mutated)
        self.assertLess(changed.quality_0_1, original.quality_0_1)

    def test_valid_reward_is_bounded_and_deterministic(self):
        row = BY_INTENT["recipe"]
        a = score(row, row["genui_json"])
        b = score(row, row["genui_json"])
        self.assertEqual(a.quality_0_1, b.quality_0_1)
        self.assertEqual(a.reward, b.reward)
        self.assertTrue(0.0 <= a.quality_0_1 <= 1.0)
        self.assertTrue(-1.0 <= a.reward <= 1.0)

    def test_very_short_and_long_sources_do_not_create_nan(self):
        spec = simple_spec("x")
        for source in ["x", "long source " * 10000]:
            result = score_genui_completion(dump(spec), source)
            self.assertTrue(math.isfinite(result.quality_0_1))
            self.assertTrue(math.isfinite(result.reward))

    def test_wrong_action_url_decreases_reward(self):
        row = BY_INTENT["notification"]
        original = score(row, row["genui_json"])
        mutated = copy.deepcopy(row["genui_json"])
        for element in mutated["elements"].values():
            on = element.get("on", {})
            press = on.get("press") if isinstance(on, dict) else None
            if isinstance(press, dict) and press.get("action") == "openUrl":
                press.setdefault("params", {})["url"] = "https://wrong.invalid.example/path"
        wrong = score(row, mutated)
        self.assertLess(wrong.quality_0_1, original.quality_0_1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
