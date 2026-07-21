from __future__ import annotations

import copy
import json
import sys
import unittest
from collections import OrderedDict
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_quality import score_genui_completion  # noqa: E402


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


def score(row: dict, spec: dict, *, contract=None):
    return score_genui_completion(
        spec,
        row["response_text"],
        intent=row.get("intent_bucket"),
        assets=row.get("assets"),
        expected_ui_contract=contract,
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
            "body": {"type": "Text", "props": {"text": text, "variant": "body"}, "children": []},
        },
    }


def first_element(spec: dict, element_type: str) -> tuple[str, dict]:
    return next(
        (element_id, element)
        for element_id, element in spec["elements"].items()
        if element.get("type") == element_type
    )


class MetricV4MutationAuditTests(unittest.TestCase):
    def test_01_add_unreachable_cards_does_not_increase(self):
        row = BY_INTENT["booking"]
        original = score(row, row["genui_json"])
        changed = copy.deepcopy(row["genui_json"])
        for index in range(55):
            changed["elements"][f"unreachable_{index}"] = {"type": "Card", "props": {}, "children": []}
        result = score(row, changed)
        self.assertLessEqual(result.quality_0_1, original.quality_0_1)
        self.assertLessEqual(result.cap_0_1, 0.40)

    def test_02_add_reachable_empty_card_does_not_increase(self):
        row = BY_INTENT["booking"]
        original = score(row, row["genui_json"])
        changed = copy.deepcopy(row["genui_json"])
        changed["elements"]["empty"] = {"type": "Card", "props": {}, "children": []}
        changed["elements"][changed["root"]]["children"].append("empty")
        self.assertLessEqual(score(row, changed).quality_0_1, original.quality_0_1)

    def test_03_duplicate_visible_heading_decreases(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        _, heading = next(
            (element_id, element)
            for element_id, element in changed["elements"].items()
            if element.get("type") == "Text" and element.get("props", {}).get("variant") in {"h1", "h2"}
        )
        changed["elements"]["duplicate_heading"] = copy.deepcopy(heading)
        changed["elements"][changed["root"]]["children"].append("duplicate_heading")
        self.assertLess(score(row, changed).quality_0_1, score(row, row["genui_json"]).quality_0_1)

    def test_04_unsupported_visible_claim_decreases(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        changed["elements"]["unsupported_claim"] = {
            "type": "Text",
            "props": {"text": "Guaranteed teleportation included with every booking", "variant": "body"},
            "children": [],
        }
        changed["elements"][changed["root"]]["children"].append("unsupported_claim")
        self.assertLess(score(row, changed).quality_0_1, score(row, row["genui_json"]).quality_0_1)

    def test_05_unrelated_action_decreases(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        changed["elements"]["unrelated_action"] = {
            "type": "Button",
            "props": {"label": "Unrelated"},
            "children": [],
            "on": {"press": {"action": "openUrl", "params": {"url": "https://unrelated.invalid/"}}},
        }
        changed["elements"][changed["root"]]["children"].append("unrelated_action")
        self.assertLess(score(row, changed).quality_0_1, score(row, row["genui_json"]).quality_0_1)

    def test_06_wrong_action_destinations_decrease(self):
        row = BY_INTENT["notification"]
        changed = copy.deepcopy(row["genui_json"])
        for element in changed["elements"].values():
            on = element.get("on")
            if not isinstance(on, dict):
                continue
            for candidate in on.values():
                candidates = candidate if isinstance(candidate, list) else [candidate]
                for action in candidates:
                    if isinstance(action, dict) and action.get("action") == "openUrl":
                        action.setdefault("params", {})["url"] = "https://wrong.invalid/"
        self.assertLess(score(row, changed).quality_0_1, score(row, row["genui_json"]).quality_0_1)

    def test_07_drop_table_row_decreases(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        _, table = first_element(changed, "Table")
        path = str(table["props"].get("statePath") or "").lstrip("/")
        changed["state"][path] = changed["state"][path][:-1]
        self.assertLess(score(row, changed).quality_0_1, score(row, row["genui_json"]).quality_0_1)

    def test_08_drop_table_column_decreases(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        _, table = first_element(changed, "Table")
        removed = table["props"]["columns"].pop()["key"]
        path = str(table["props"].get("statePath") or "").lstrip("/")
        for item in changed["state"].get(path, []):
            if isinstance(item, dict):
                item.pop(removed, None)
        self.assertLess(score(row, changed).quality_0_1, score(row, row["genui_json"]).quality_0_1)

    def test_09_split_paragraph_into_words_does_not_increase(self):
        source = "A clear paragraph with enough words to represent one useful content block."
        original = simple_spec(source)
        changed = simple_spec(source)
        changed["elements"].pop("body")
        changed["elements"]["root"]["children"] = []
        for index, word in enumerate(source.split()):
            element_id = f"word_{index}"
            changed["elements"][element_id] = {"type": "Text", "props": {"text": word, "variant": "body"}, "children": []}
            changed["elements"]["root"]["children"].append(element_id)
        self.assertLessEqual(
            score_genui_completion(changed, source).quality_0_1,
            score_genui_completion(original, source).quality_0_1,
        )

    def test_10_decorative_icon_inflation_decreases(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        for index in range(40):
            element_id = f"decorative_{index}"
            changed["elements"][element_id] = {
                "type": "Icon",
                "props": {"name": "genuicraft:star", "accessibility": {"decorative": True}},
                "children": [],
            }
            changed["elements"][changed["root"]]["children"].append(element_id)
        self.assertLess(score(row, changed).quality_0_1, score(row, row["genui_json"]).quality_0_1)

    def test_11_missing_reference_decreases_and_caps(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        changed["elements"][changed["root"]]["children"].append("missing_element")
        result = score(row, changed)
        self.assertLessEqual(result.cap_0_1, 0.25)

    def test_12_root_cycle_decreases_and_caps(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        changed["elements"][changed["root"]]["children"].append(changed["root"])
        result = score(row, changed)
        self.assertLessEqual(result.cap_0_1, 0.25)

    def test_13_missing_declared_root_is_fatal(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        changed["root"] = "does_not_exist"
        self.assertLessEqual(score(row, changed).quality_0_1, 0.10)

    def test_14_disconnected_valid_subtree_decreases(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        root_children = changed["elements"][changed["root"]]["children"]
        self.assertGreater(len(root_children), 1)
        root_children.pop()
        self.assertLess(score(row, changed).quality_0_1, score(row, row["genui_json"]).quality_0_1)

    def test_15_identifier_renaming_is_invariant(self):
        row = BY_INTENT["recipe"]
        spec = row["genui_json"]
        mapping = {old: f"renamed_{index}" for index, old in enumerate(spec["elements"])}
        changed = copy.deepcopy(spec)
        changed["root"] = mapping[spec["root"]]
        changed["elements"] = {
            mapping[old]: {
                **copy.deepcopy(element),
                "children": [mapping.get(child, child) for child in element.get("children", [])],
            }
            for old, element in spec["elements"].items()
        }
        self.assertAlmostEqual(score(row, changed).quality_0_1, score(row, spec).quality_0_1, places=12)

    def test_16_element_map_order_is_invariant(self):
        row = BY_INTENT["recipe"]
        changed = copy.deepcopy(row["genui_json"])
        changed["elements"] = OrderedDict(reversed(list(changed["elements"].items())))
        self.assertAlmostEqual(score(row, changed).quality_0_1, score(row, row["genui_json"]).quality_0_1, places=12)

    def test_17_adding_required_charts_and_formula_increases(self):
        source = "Quarterly results require three bar charts. Formula: revenue - costs = profit."
        contract = {
            "content_units": [source],
            "required_roles": {"chart": 3, "formula": 1},
            "expected_role_counts": {"chart": 3, "formula": 1},
        }
        original = simple_spec(source)
        changed = copy.deepcopy(original)
        changed["state"]["series"] = [{"quarter": "Q1", "value": 10}, {"quarter": "Q2", "value": 12}]
        for index in range(3):
            element_id = f"chart_{index}"
            changed["elements"][element_id] = {
                "type": "Chart",
                "props": {
                    "columns": [{"key": "quarter", "label": "Quarter"}, {"key": "value", "label": "Value"}],
                    "statePath": "/series",
                    "xKey": "quarter",
                    "yKey": "value",
                },
                "children": [],
            }
            changed["elements"]["root"]["children"].append(element_id)
        changed["elements"]["formula"] = {"type": "Formula", "props": {"text": "revenue - costs = profit"}, "children": []}
        changed["elements"]["root"]["children"].append("formula")
        self.assertGreater(
            score_genui_completion(changed, source, expected_ui_contract=contract).quality_0_1,
            score_genui_completion(original, source, expected_ui_contract=contract).quality_0_1,
        )

    def test_18_remove_required_image_decreases(self):
        source = "Product photo"
        contract = {
            "content_units": [source],
            "media": [{"kind": "Image", "url": "https://example.com/product.png", "alt": "Product photo"}],
            "required_roles": {"image": 1},
        }
        original = simple_spec(source)
        original["elements"]["image"] = {"type": "Image", "props": {"url": "https://example.com/product.png", "alt": "Product photo"}, "children": []}
        original["elements"]["root"]["children"].append("image")
        changed = copy.deepcopy(original)
        changed["elements"]["root"]["children"].remove("image")
        self.assertLess(
            score_genui_completion(changed, source, expected_ui_contract=contract).quality_0_1,
            score_genui_completion(original, source, expected_ui_contract=contract).quality_0_1,
        )

    def test_19_remove_required_action_decreases(self):
        row = BY_INTENT["booking"]
        changed = copy.deepcopy(row["genui_json"])
        _, button = first_element(changed, "Button")
        button.pop("on", None)
        self.assertLess(score(row, changed).quality_0_1, score(row, row["genui_json"]).quality_0_1)

    def test_20_equivalent_root_stack_to_column_is_near_invariant(self):
        source = "Hello world"
        original = simple_spec(source)
        changed = copy.deepcopy(original)
        changed["elements"]["root"]["type"] = "Column"
        changed["elements"]["root"]["props"].pop("direction")
        delta = abs(
            score_genui_completion(changed, source).quality_0_1
            - score_genui_completion(original, source).quality_0_1
        )
        self.assertLess(delta, 0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
