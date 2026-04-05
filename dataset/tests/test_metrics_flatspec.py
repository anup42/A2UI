import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.metrics import compute_ui_metrics


class MetricsFlatSpecTests(unittest.TestCase):
    def test_metrics_extract_from_flat_spec(self) -> None:
        response_text = """
Summary:
Best option available.
Quick Actions:
- [Button: Open Offer] https://example.com/deal
""".strip()
        spec = {
            "root": "root",
            "state": {},
            "elements": {
                "root": {"type": "Column", "props": {}, "children": ["title", "cta"]},
                "title": {
                    "type": "Text",
                    "props": {"text": "Summary", "variant": "h2"},
                    "children": [],
                },
                "cta": {
                    "type": "Button",
                    "props": {"label": "Open Offer"},
                    "children": [],
                    "on": {
                        "press": {
                            "action": "openUrl",
                            "params": {"url": "https://example.com/deal"},
                        }
                    },
                },
            },
        }

        metrics = compute_ui_metrics(response_text, spec)
        self.assertGreaterEqual(metrics.get("component_count", 0.0), 3.0)
        self.assertGreaterEqual(metrics.get("actionable_elements", 0.0), 1.0)
        self.assertGreaterEqual(metrics.get("action_coverage", 0.0), 1.0)

    def test_heading_and_repeat_table_detection_for_flat_spec(self) -> None:
        response_text = """
Gross Income Breakdown
Frequency | Gross Amount
Hourly | $36.06
Weekly | $1,442.31
Quick Actions:
- [Button: Open Details] https://example.com/a
""".strip()
        spec = {
            "root": "root",
            "state": {
                "rows": [
                    {"id": "r1", "frequency": "Hourly", "amount": "$36.06", "url": "https://example.com/a"},
                    {"id": "r2", "frequency": "Weekly", "amount": "$1,442.31", "url": "https://example.com/b"},
                ]
            },
            "elements": {
                "root": {"type": "Column", "props": {}, "children": ["heading", "list", "cta"]},
                "heading": {
                    "type": "Text",
                    "props": {"text": "Gross Income Breakdown", "fontSize": "titleLarge", "fontWeight": "bold"},
                    "children": [],
                },
                "list": {
                    "type": "List",
                    "props": {},
                    "children": ["row"],
                    "repeat": {"statePath": "/rows", "key": "id"},
                },
                "row": {
                    "type": "Row",
                    "props": {},
                    "children": ["f", "a"],
                },
                "f": {"type": "Text", "props": {"text": {"$bindItem": "frequency"}}, "children": []},
                "a": {"type": "Text", "props": {"text": {"$bindItem": "amount"}}, "children": []},
                "cta": {
                    "type": "Button",
                    "props": {"label": "Open Details"},
                    "children": [],
                    "on": {"press": {"action": "openUrl", "params": {"url": {"$bindItem": "url"}}}},
                },
            },
        }

        metrics = compute_ui_metrics(response_text, spec)
        self.assertGreaterEqual(metrics.get("table_pattern_detected", 0.0), 1.0)
        self.assertGreater(metrics.get("table_cell_coverage", 0.0), 0.0)
        self.assertGreater(metrics.get("section_heading_coverage", 0.0), 0.0)
        self.assertGreaterEqual(metrics.get("action_coverage", 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
