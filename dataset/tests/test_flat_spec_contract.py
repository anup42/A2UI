import sys
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.flat_spec_contract import (
    build_fallback_flat_spec,
    coerce_and_validate,
    validate_flat_spec,
)
from pipeline.flat_spec_semantics import iter_renderer_references


class FlatSpecContractTests(unittest.TestCase):
    def test_fallback_is_valid(self) -> None:
        spec = build_fallback_flat_spec("Hello world")
        validation = validate_flat_spec(spec)
        self.assertTrue(validation.is_valid, msg=validation.error)
        self.assertEqual(spec["root"], "root")

    def test_legacy_messages_are_converted(self) -> None:
        legacy = [
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["title"]},
                        {"id": "title", "component": "Text", "text": "Hello", "variant": "h2"},
                    ],
                },
            }
        ]
        result = coerce_and_validate(legacy)
        self.assertTrue(result.is_valid, msg=result.error)
        self.assertTrue(result.converted_from_legacy)
        self.assertEqual(result.spec["elements"]["title"]["props"]["text"], "Hello")

    def test_invalid_child_reference_is_rejected(self) -> None:
        bad_spec = {
            "root": "root",
            "state": {},
            "elements": {
                "root": {"type": "Column", "props": {}, "children": ["missing"]},
            },
        }
        validation = validate_flat_spec(bad_spec)
        self.assertFalse(validation.is_valid)
        self.assertIn("missing child", validation.error or "")

    def test_tabs_modal_and_template_references_share_renderer_inventory(self) -> None:
        tabs = {
            "type": "Tabs",
            "props": {
                "tabs": [
                    {"child": "first"},
                    {"content": "second"},
                    {"id": "third"},
                    {"element": "fourth"},
                ],
                "template": "template",
                "itemTemplate": "item_template",
                "child": "property_child",
            },
            "children": ["ordinary"],
            "repeat": {"statePath": "/items", "template": "repeat_template"},
        }
        references = {
            (item.target_id, item.source_path, item.reference_kind)
            for item in iter_renderer_references(tabs)
        }
        for target in (
            "ordinary",
            "template",
            "item_template",
            "property_child",
            "repeat_template",
            "first",
            "second",
            "third",
            "fourth",
        ):
            self.assertTrue(any(item[0] == target for item in references), target)

        modal = {
            "type": "Modal",
            "props": {"trigger": "open", "content": "content"},
            "children": [],
        }
        self.assertEqual(
            {item.reference_kind for item in iter_renderer_references(modal)},
            {"modal_trigger", "modal_content"},
        )

    def test_missing_tabs_and_modal_references_are_rejected(self) -> None:
        spec = {
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
                    "props": {"trigger": "trigger", "content": "missing"},
                    "children": [],
                },
                "trigger": {
                    "type": "Button",
                    "props": {"label": "Open"},
                    "children": [],
                },
            },
        }
        validation = validate_flat_spec(spec)
        self.assertFalse(validation.is_valid)
        self.assertIn("props.content", validation.error or "")

    def test_legacy_function_call_shape_is_rejected(self) -> None:
        bad_spec = {
            "root": "root",
            "state": {},
            "elements": {
                "root": {
                    "type": "Button",
                    "props": {"label": "Open"},
                    "children": [],
                    "on": {
                        "press": {
                            "functionCall": {
                                "call": "openUrl",
                                "args": {"url": "https://example.com"},
                            }
                        }
                    },
                }
            },
        }
        validation = validate_flat_spec(bad_spec)
        self.assertFalse(validation.is_valid)
        self.assertIn("legacy functionCall", validation.error or "")

    def test_action_shorthand_is_normalized(self) -> None:
        spec = {
            "root": "root",
            "state": {},
            "elements": {
                "root": {
                    "type": "column",
                    "props": {},
                    "children": ["cta"],
                },
                "cta": {
                    "type": "button",
                    "props": {"label": "Open"},
                    "children": [],
                    "on": {
                        "click": {
                            "action": "openurl",
                            "url": "https://example.com",
                        }
                    },
                },
            },
        }
        result = coerce_and_validate(spec)
        self.assertTrue(result.is_valid, msg=result.error)
        press = result.spec["elements"]["cta"]["on"]["press"]
        self.assertEqual(press["action"], "openUrl")
        self.assertEqual(press["params"]["url"], "https://example.com")
        self.assertEqual(result.spec["elements"]["root"]["type"], "Stack")
        self.assertEqual(
            result.spec["elements"]["root"]["props"]["direction"],
            "vertical",
        )

    def test_repeat_and_watch_keys_are_normalized(self) -> None:
        spec = {
            "root": "root",
            "state": {"rows": [{"id": "1"}]},
            "elements": {
                "root": {
                    "type": "Column",
                    "props": {},
                    "children": ["list"],
                    "watch": {"state.form.valid": {"action": "validate"}},
                },
                "list": {
                    "type": "List",
                    "props": {},
                    "children": ["item"],
                    "repeat": {"path": "rows", "key": "id"},
                },
                "item": {"type": "Text", "props": {"text": "x"}, "children": []},
            },
        }
        result = coerce_and_validate(spec)
        self.assertTrue(result.is_valid, msg=result.error)
        self.assertIn("/form/valid", result.spec["elements"]["root"]["watch"])
        self.assertEqual(result.spec["elements"]["list"]["repeat"]["statePath"], "/rows")

    def test_email_preview_is_supported_and_canonicalized(self) -> None:
        spec = {
            "root": "email",
            "state": {},
            "elements": {
                "email": {
                    "type": "email_preview",
                    "props": {
                        "subject": "Follow-up on launch plan",
                        "to": "founder@example.com",
                        "body": ["Hi team,", "Here is the next step."],
                        "signature": "Anup",
                    },
                    "children": [],
                }
            },
        }
        result = coerce_and_validate(spec)
        self.assertTrue(result.is_valid, msg=result.error)
        self.assertEqual(result.spec["elements"]["email"]["type"], "EmailPreview")

    def test_chart_contract_is_aligned_across_schema_prompt_python_and_android(self) -> None:
        spec = {
            "root": "chart",
            "state": {"rows": [{"label": "Q1", "value": 10}]},
            "elements": {
                "chart": {
                    "type": "bar_chart",
                    "props": {
                        "columns": [
                            {"key": "label", "label": "Quarter"},
                            {"key": "value", "label": "Value"},
                        ],
                        "statePath": "/rows",
                        "xKey": "label",
                        "yKey": "value",
                    },
                    "children": [],
                }
            },
        }
        result = coerce_and_validate(spec)
        self.assertTrue(result.is_valid, msg=result.error)
        self.assertEqual(result.spec["elements"]["chart"]["type"], "Chart")

        schema = json.loads((ROOT / "schema" / "genui_flatspec.schema.json").read_text(encoding="utf-8"))
        type_enum = schema["$defs"]["element"]["properties"]["type"]["enum"]
        self.assertIn("Chart", type_enum)

        prompt = (ROOT / "prompts" / "genui_gen_mobile_flatspec_v11.md").read_text(encoding="utf-8")
        self.assertIn("`Chart` props", prompt)

        renderer_registry = (
            ROOT.parent
            / "android"
            / "app"
            / "src"
            / "main"
            / "java"
            / "com"
            / "samsung"
            / "genuicraft"
            / "renderer"
            / "flat"
            / "compose"
            / "FlatRenderRegistry.kt"
        ).read_text(encoding="utf-8")
        self.assertIn('"chart" to { c -> RenderChart', renderer_registry)


if __name__ == "__main__":
    unittest.main()
