import sys
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
        self.assertEqual(result.spec["elements"]["root"]["type"], "Column")

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


if __name__ == "__main__":
    unittest.main()
