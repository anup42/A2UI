from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(DATASET_ROOT / "scripts"))

from capture_android_run_screenshots import write_native_render_checks  # noqa: E402
from pipeline.genui_quality import aggregate_v4_records, score_record_v4 as score_record  # noqa: E402


def simple_record(ui_id: str = "sample") -> dict:
    return {
        "ui_id": ui_id,
        "response_text": "Hello world",
        "intent_bucket": "documentation",
        "genui_json": {
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
                    "props": {"text": "Hello world", "variant": "body"},
                    "children": [],
                },
            },
        },
    }


class MetricV4PipelineIntegrationTests(unittest.TestCase):
    def test_android_capture_manifest_publishes_attempted_pass_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            output_dir = run_dir / "android_device_rendered"
            output_dir.mkdir()
            rows = [
                {
                    "ui_id": "pass",
                    "query_id": "q1",
                    "response_id": "r1",
                    "screenshot": "01_pass.png",
                    "ok": True,
                    "full_height_px": 1200,
                    "tile_count": 2,
                },
                {
                    "ui_id": "fail",
                    "query_id": "q2",
                    "response_id": "r2",
                    "screenshot": "02_fail.png",
                    "ok": False,
                    "full_height_px": 0,
                    "tile_count": 0,
                },
            ]
            (output_dir / "capture_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            self.assertEqual(write_native_render_checks(run_dir, output_dir), 2)
            published = [
                json.loads(line)
                for line in (run_dir / "native_render_checks.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertTrue(published[0]["renderer_check_result"]["attempted"])
            self.assertTrue(published[0]["renderer_check_result"]["ok"])
            self.assertFalse(published[1]["renderer_check_result"]["ok"])

    def test_failed_native_attempt_activates_render_cap(self):
        record = simple_record()
        native_failure = {
            "ui_id": "sample",
            "renderer_check_result": {
                "adapter": "android_native_flat_renderer",
                "attempted": True,
                "ok": False,
            },
        }
        result = score_record(record, render_row=native_failure)
        self.assertFalse(result.evidence["render_ok"])
        self.assertLessEqual(result.cap_0_1, 0.30)
        self.assertTrue(any(cap["name"] == "render_failure" for cap in result.active_caps))

    def test_generic_browser_screenshot_is_not_native_smoke_evidence(self):
        record = simple_record()
        browser_row = {
            "ui_id": "sample",
            "renderer_check_result": {
                "adapter": "playwright_browser",
                "attempted": True,
                "ok": False,
            },
        }
        result = score_record(record, render_row=browser_row)
        self.assertIsNone(result.evidence["render_ok"])
        self.assertFalse(any(cap["name"] == "render_failure" for cap in result.active_caps))

    def test_aggregate_headline_is_mean_of_per_sample_scores(self):
        first = simple_record("one")
        second = simple_record("two")
        second["genui_json"] = {"root": "missing", "state": {}, "elements": {}}
        first_score = score_record(first).quality_0_100
        second_score = score_record(second).quality_0_100
        aggregate = aggregate_v4_records([first, second])
        self.assertAlmostEqual(
            aggregate["quality_0_100"]["mean"],
            (first_score + second_score) / 2.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
