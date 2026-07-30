from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_ROOT.parent
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.flat_spec_contract import _TYPE_CANONICAL_MAP  # noqa: E402
from pipeline.genui_quality import (  # noqa: E402
    RewardInflationMonitor,
    load_default_reward_config,
    make_genui_grpo_reward,
    score_genui_completion_v4 as score_genui_completion,
)
from pipeline.genui_quality.evidence import RENDERER_SEMANTICS  # noqa: E402


def _spec(*, include_icon: bool = False, decorative_count: int = 0) -> dict:
    children = ["body"]
    elements = {
        "root": {
            "type": "Stack",
            "props": {"direction": "vertical"},
            "children": children,
        },
        "body": {
            "type": "Text",
            "props": {"text": "Hello world"},
            "children": [],
        },
    }
    if include_icon:
        children.append("calendar_icon")
        elements["calendar_icon"] = {
            "type": "Icon",
            "props": {"name": "calendar", "description": "Calendar"},
            "children": [],
        }
    for index in range(decorative_count):
        element_id = f"divider_{index}"
        children.append(element_id)
        elements[element_id] = {"type": "Divider", "props": {}, "children": []}
    return {"root": "root", "state": {}, "elements": elements}


def _dump(spec: dict) -> str:
    return json.dumps(spec, separators=(",", ":"))


class CoreComplianceTests(unittest.TestCase):
    def test_reward_breakdown_is_frozen(self):
        result = score_genui_completion(_dump(_spec()), "Hello world")
        with self.assertRaises(FrozenInstanceError):
            result.reward = 0.0  # type: ignore[misc]

    def test_renderer_inventory_covers_contract_and_native_only_aliases(self):
        for alias, canonical in _TYPE_CANONICAL_MAP.items():
            self.assertEqual(RENDERER_SEMANTICS.aliases.get(alias), canonical)
        self.assertEqual(RENDERER_SEMANTICS.aliases["logoutput"], "ConsoleLog")
        self.assertEqual(RENDERER_SEMANTICS.aliases["log_output"], "ConsoleLog")
        self.assertIn("Chart", RENDERER_SEMANTICS.canonical_types)

    def test_required_icon_participates_in_media_fidelity(self):
        source = "Hello world\nMedia: Icon = calendar Alt = Calendar"
        present = score_genui_completion(_dump(_spec(include_icon=True)), source)
        missing = score_genui_completion(_dump(_spec()), source)
        present_media = present.atomics["fidelity"]["media_fidelity"]
        missing_media = missing.atomics["fidelity"]["media_fidelity"]
        self.assertIsNotNone(present_media)
        self.assertGreater(float(present_media), float(missing_media))
        self.assertGreater(present.quality_0_100, missing.quality_0_100)


class GrpoObservabilityTests(unittest.TestCase):
    def test_monitor_alerts_on_growth_without_fidelity_gain(self):
        monitor = RewardInflationMonitor(
            component_growth_threshold=0.10,
            length_growth_threshold=0.10,
            min_fidelity_gain=0.01,
            ema_alpha=0.25,
        )
        first = monitor.observe(
            component_count=2,
            completion_length=100,
            fidelity=0.8,
        )
        second = monitor.observe(
            component_count=4,
            completion_length=150,
            fidelity=0.8,
        )
        self.assertEqual(first["comparable_window"], 0.0)
        self.assertEqual(second["comparable_window"], 1.0)
        self.assertEqual(second["alert_component_inflation_without_fidelity_gain"], 1.0)
        self.assertEqual(second["alert_length_inflation_without_fidelity_gain"], 1.0)

    def test_grpo_callback_emits_inflation_alert_metrics(self):
        monitor = RewardInflationMonitor(
            component_growth_threshold=0.10,
            length_growth_threshold=0.10,
            min_fidelity_gain=0.01,
            ema_alpha=0.25,
        )
        reward = make_genui_grpo_reward(
            load_default_reward_config(),
            model_checkpoint="test-checkpoint",
            inflation_monitor=monitor,
        )
        logged: dict[str, list[float]] = {}

        def log_metric(name: str, value: float) -> None:
            logged.setdefault(name, []).append(float(value))

        reward([_dump(_spec())], ["Hello world"], log_metric=log_metric)
        reward(
            [_dump(_spec(decorative_count=3))],
            ["Hello world"],
            log_metric=log_metric,
        )
        self.assertEqual(
            logged["genui/alert_component_inflation_without_fidelity_gain"][-1],
            1.0,
        )
        self.assertEqual(
            logged["genui/alert_length_inflation_without_fidelity_gain"][-1],
            1.0,
        )


class DashboardComplianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_path = DATASET_ROOT / "scripts" / "dataset_dashboard.py"
        spec = importlib.util.spec_from_file_location("metric_v4_dashboard", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load dataset dashboard module")
        cls.dashboard = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.dashboard
        spec.loader.exec_module(cls.dashboard)

    def tearDown(self) -> None:
        self.dashboard.set_dashboard_metric_version("legacy")

    def test_dashboard_selects_requested_headline_and_preserves_both(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "aggregates.json").write_text(
                json.dumps(
                    {
                        "legacy_structural_richness_score": 41.0,
                        "overall_score": 41.0,
                        "genui_quality_v4": {
                            "quality_0_100": {"mean": 82.0},
                            "calibration_status": "uncalibrated_engineering_score",
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "ui_id": "ui-1",
                "response_text": "Hello world",
                "genui_json": _spec(),
                "metrics": {
                    "legacy_structural_richness_score": 41.0,
                    "overall_score": 41.0,
                    "genui_quality_v4": 82.0,
                    "genui_quality_v4_dimensions": {
                        "integrity": 1.0,
                        "fidelity": 0.8,
                    },
                },
            }
            (run_dir / "genui.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            self.assertEqual(self.dashboard.run_score(run_dir, "legacy"), 41.0)
            self.assertEqual(self.dashboard.run_score(run_dir, "v4"), 82.0)
            self.dashboard.set_dashboard_metric_version("v4")
            scanned = self.dashboard.scan_run("local", "Local", run_dir)
            self.assertEqual(scanned["overall_score"], 82.0)
            self.assertEqual(scanned["legacy_structural_richness_score"], 41.0)
            self.assertEqual(scanned["genui_quality_v4_score"], 82.0)
            self.assertEqual(scanned["calibration_status"], "uncalibrated_engineering_score")
            self.assertEqual(scanned["metric_avgs"]["genui_quality_v4_fidelity"], 0.8)

    def test_dashboard_html_labels_v4_as_uncalibrated(self):
        self.assertIn("uncalibrated engineering score", self.dashboard.INDEX_HTML)
        self.assertIn("not an equal-interval quality percentage", self.dashboard.INDEX_HTML)


if __name__ == "__main__":
    unittest.main(verbosity=2)
