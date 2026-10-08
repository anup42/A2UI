"""Host-only receipt gate tests; no device, model or build calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location("fold7_review", Path(__file__).with_name("review_fold7_case.py"))
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


class BeforeRenderFailureGateTest(unittest.TestCase):
    def setUp(self):
        scratch = review.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="f7_", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.before_path = self.root / "BXP-018/before/r01/receipt.json"
        self.before_path.parent.mkdir(parents=True)
        self.source = self.before_path.parent / "input/BXP-018/output.a2ui.json"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"[]")
        self.args = SimpleNamespace(case="BXP-018", phase="after", source=None, source_info=None,
            allow_before_render_failures=False, output=self.root, action="replay", attempt="r01",
            adb="unused", serial="R3CY30QFWLP", max_vertical_swipes=30, max_horizontal_swipes=8, timeout=30)
        self.write_before()

    def write_before(self, failed=()):
        self.before = {"case": "BXP-018", "phase": "before", "attempt": "r01", "startedAtUtc": "2026-10-09",
            "status": "failed" if failed else "collected", "automatedChecksSatisfied": not bool(failed),
            "checks": {key: key not in failed for key in review.REPLAY_CHECK_NAMES},
            "source": {"archivedPath": self.source.relative_to(self.root).as_posix(),
                "sha256": review.digest(self.source), "replayMode": "json"}}
        self.save_before()

    def save_before(self):
        self.before_path.write_text(json.dumps(self.before), encoding="utf-8")

    def after(self):
        _, source = review.choose_source(self.root, self.args)
        return {"status": "collected", "automatedChecksSatisfied": True, "source": source,
            "checks": {key: True for key in review.REPLAY_CHECK_NAMES}}

    def test_passing_baseline_needs_no_flag_and_pins_exact_receipt(self):
        result = self.after()
        self.assertNotIn("beforeRenderFailureOverride", result["source"])
        self.assertEqual(review.digest(self.before_path), result["source"]["pinnedBeforeReceiptSha256"])
        self.assertTrue(review.pinned_before_evidence(self.root, result, self.before_path, self.before))

    def test_known_failures_require_flag_and_preserve_failed_receipt(self):
        for failed in ({"renderedWithoutIssues"}, {"allTableColumnsObserved"}, review.BEFORE_RENDER_FAILURE_CHECKS):
            with self.subTest(failed=failed):
                self.write_before(failed)
                original = self.before_path.read_bytes()
                self.args.allow_before_render_failures = False
                with self.assertRaises(ValueError): review.choose_source(self.root, self.args)
                self.args.allow_before_render_failures = True
                result = self.after()
                self.assertEqual(sorted(failed), result["source"]["beforeRenderFailureOverride"]["overriddenChecks"])
                self.assertTrue(review.pinned_before_evidence(self.root, result, self.before_path, self.before))
                self.assertEqual(original, self.before_path.read_bytes())
                self.assertEqual("failed", self.before["status"])

    def test_other_false_missing_unknown_and_nonboolean_checks_are_rejected(self):
        self.args.allow_before_render_failures = True
        for key in review.REPLAY_CHECK_NAMES - review.BEFORE_RENDER_FAILURE_CHECKS:
            with self.subTest(failed=key):
                self.write_before({"renderedWithoutIssues", key})
                with self.assertRaises(ValueError): review.choose_source(self.root, self.args)
        for mutation in ("missing", "unknown", "nonboolean"):
            with self.subTest(mutation=mutation):
                self.write_before(review.BEFORE_RENDER_FAILURE_CHECKS)
                if mutation == "missing": del self.before["checks"]["frozenCorpus"]
                if mutation == "unknown": self.before["checks"]["unknownCheck"] = False
                if mutation == "nonboolean": self.before["checks"]["noInference"] = 1
                self.save_before()
                with self.assertRaises(ValueError): review.choose_source(self.root, self.args)

    def test_changed_archived_source_and_different_caller_source_are_rejected(self):
        self.write_before(review.BEFORE_RENDER_FAILURE_CHECKS)
        self.args.allow_before_render_failures = True
        other = self.root / "other.json"
        other.write_bytes(b"[{}]")
        self.args.source = other
        with self.assertRaises(ValueError): review.choose_source(self.root, self.args)
        self.args.source = None
        self.source.write_bytes(b"[{}]")
        with self.assertRaises(ValueError): review.choose_source(self.root, self.args)

    def test_flag_is_after_only(self):
        self.args.phase = "before"
        self.args.source = self.source
        self.args.allow_before_render_failures = True
        with self.assertRaises(ValueError): review.choose_source(self.root, self.args)
        parsed = review.parser().parse_args(["replay", "--case", "BXP-018", "--phase", "after", "--allow-before-render-failures"])
        self.assertTrue(parsed.allow_before_render_failures)

    def test_reporting_requires_exact_override_receipt_pin_source_hash_and_after_checks(self):
        self.write_before(review.BEFORE_RENDER_FAILURE_CHECKS)
        self.args.allow_before_render_failures = True
        valid = self.after()
        for key in review.REPLAY_CHECK_NAMES:
            with self.subTest(after_failed=key):
                value = json.loads(json.dumps(valid))
                value["checks"][key] = False
                self.assertFalse(review.pinned_before_evidence(self.root, value, self.before_path, self.before))
        for field in ("pinnedBeforeReceiptSha256", "sha256"):
            value = json.loads(json.dumps(valid)); value["source"][field] = "0" * 64
            self.assertFalse(review.pinned_before_evidence(self.root, value, self.before_path, self.before))
        value = json.loads(json.dumps(valid))
        value["source"]["beforeRenderFailureOverride"]["overriddenChecks"] = ["renderedWithoutIssues"]
        self.assertFalse(review.pinned_before_evidence(self.root, value, self.before_path, self.before))
        value = json.loads(json.dumps(valid)); del value["checks"]["capturesComplete"]
        self.assertFalse(review.pinned_before_evidence(self.root, value, self.before_path, self.before))

    def test_collect_does_not_waive_after_render_or_column_failures(self):
        self.write_before(review.BEFORE_RENDER_FAILURE_CHECKS)
        self.args.allow_before_render_failures = True
        source_digest = review.digest(self.source)
        class FakeAdb:
            def __init__(self, *args): pass
            def absent(self, *args): pass
            def command(self, *args, **kwargs): return ""
            def file_hash(self, *args): return source_digest
        def instrument(adb, invocation, remote, target, timeout, receipt):
            artifacts = target / "artifacts"; case = artifacts / "BXP-018"; case.mkdir(parents=True)
            (case / "initial.png").write_bytes(b"fixture"); (case / "initial.xml").write_text("<hierarchy/>")
            report = {"id": "BXP-018", "sourceJsonSha256": source_digest, "status": "rendered",
                "issues": ["Missing authored date"], "verticalEndObserved": True, "verticalLimitReached": False,
                "captures": [{"name": "initial", "screenshot": True}], "tables": [{"missingColumns": ["Date"]}]}
            (artifacts / "replay_results.json").write_text(json.dumps([report]))
            (artifacts / "replay_config.json").write_text(json.dumps({"cases": ["BXP-018"],
                "corpusSha256": review.digest(review.CORPUS), "inferenceEvaluated": False}))
            (artifacts / "replay_summary.json").write_text(json.dumps({"modelCalls": 0}))
            receipt["instrumentationPassed"] = True
        with patch.object(review, "Adb", FakeAdb), patch.object(review, "provenance", return_value={}), \
                patch.object(review, "instrument", side_effect=instrument), patch.object(review, "build_report"):
            with self.assertRaises(ValueError): review.collect(self.args)
        failed = review.read(self.root / "BXP-018/after/r01/receipt.json")
        self.assertEqual("failed", failed["status"])
        self.assertFalse(failed["automatedChecksSatisfied"])
        self.assertFalse(failed["checks"]["renderedWithoutIssues"])
        self.assertFalse(failed["checks"]["allTableColumnsObserved"])
        self.assertFalse(review.pinned_before_evidence(self.root, failed, self.before_path, self.before))


if __name__ == "__main__":
    unittest.main()
