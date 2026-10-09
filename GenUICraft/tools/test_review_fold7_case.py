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
            adb="unused", serial="R3CY30QFWLP", max_vertical_swipes=30, max_horizontal_swipes=8,
            repeat_table_sweeps_per_viewport=False, verify_tab_views=False, timeout=30)
        self.write_before()

    def write_before(self, failed=(), repeat_sweeps=None, verify_tabs=None):
        checks = review.LEGACY_REPLAY_CHECK_NAMES if repeat_sweeps is None else review.VIEWPORT_REPLAY_CHECK_NAMES
        if verify_tabs is not None:
            checks = checks | {"requestedTabViews"}
        self.before = {"case": "BXP-018", "phase": "before", "attempt": "r01", "startedAtUtc": "2026-10-09",
            "status": "failed" if failed else "collected", "automatedChecksSatisfied": not bool(failed),
            "checks": {key: key not in failed for key in checks},
            "source": {"archivedPath": self.source.relative_to(self.root).as_posix(),
                "sha256": review.digest(self.source), "replayMode": "json"}}
        if repeat_sweeps is not None:
            self.before["repeatTableSweepsPerViewport"] = repeat_sweeps
        if verify_tabs is not None:
            self.before["verifyTabViews"] = verify_tabs
        self.save_before()

    def save_before(self):
        self.before_path.write_text(json.dumps(self.before), encoding="utf-8")

    def after(self):
        _, source = review.choose_source(self.root, self.args)
        return {"status": "collected", "automatedChecksSatisfied": True, "source": source,
            "repeatTableSweepsPerViewport": self.args.repeat_table_sweeps_per_viewport,
            "verifyTabViews": self.args.verify_tab_views,
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
        for key in review.LEGACY_REPLAY_CHECK_NAMES - review.BEFORE_RENDER_FAILURE_CHECKS:
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
        self.assertFalse(parsed.repeat_table_sweeps_per_viewport)
        self.assertFalse(parsed.verify_tab_views)
        opted_in = review.parser().parse_args(["replay", "--case", "BXP-029", "--phase", "after",
            "--repeat-table-sweeps-per-viewport"])
        self.assertTrue(opted_in.repeat_table_sweeps_per_viewport)
        tabs = review.parser().parse_args(["replay", "--case", "BXP-035", "--phase", "after", "--verify-tab-views"])
        self.assertTrue(tabs.verify_tab_views)

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
            self.assertEqual("false", invocation[invocation.index("repeatTableSweepsPerViewport") + 1])
            self.assertEqual("false", invocation[invocation.index("verifyTabViews") + 1])
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
        self.assertFalse(failed["repeatTableSweepsPerViewport"])
        self.assertFalse(failed["checks"]["renderedWithoutIssues"])
        self.assertFalse(failed["checks"]["allTableColumnsObserved"])
        self.assertFalse(review.pinned_before_evidence(self.root, failed, self.before_path, self.before))

    def test_legacy_and_extended_receipts_compare_without_mutating_the_baseline(self):
        for before_flag in (None, False, True):
            with self.subTest(before_flag=before_flag):
                self.write_before(repeat_sweeps=before_flag)
                original = self.before_path.read_bytes()
                for after_flag in (None, False, True):
                    with self.subTest(after_flag=after_flag):
                        self.args.repeat_table_sweeps_per_viewport = after_flag is True
                        after = self.after()
                        del after["verifyTabViews"]
                        del after["checks"]["requestedTabViews"]
                        if after_flag is None:
                            del after["repeatTableSweepsPerViewport"]
                            del after["checks"]["requestedViewportTableSweeps"]
                        self.assertTrue(review.pinned_before_evidence(self.root, after, self.before_path, self.before))
                        self.assertEqual(original, self.before_path.read_bytes())

    def test_viewport_sweep_request_requires_exact_schema_and_boolean_flag(self):
        self.args.allow_before_render_failures = True
        for flag in (False, True):
            for mutation in ("missing_check", "unknown_check"):
                with self.subTest(flag=flag, mutation=mutation):
                    self.write_before(repeat_sweeps=flag)
                    if mutation == "missing_check": del self.before["checks"]["requestedViewportTableSweeps"]
                    else: self.before["checks"]["unknownCheck"] = True
                    self.save_before()
                    with self.assertRaises(ValueError): review.choose_source(self.root, self.args)
        for flag in (None, 0, 1, "false", "true"):
            with self.subTest(nonboolean_flag=flag):
                self.write_before(repeat_sweeps=False)
                self.before["repeatTableSweepsPerViewport"] = flag
                self.save_before()
                with self.assertRaises(ValueError): review.choose_source(self.root, self.args)
        for check in (False, True):
            with self.subTest(legacy_extra_check=check):
                self.write_before()
                self.before["checks"]["requestedViewportTableSweeps"] = check
                self.save_before()
                with self.assertRaises(ValueError): review.choose_source(self.root, self.args)

    def test_extended_schema_keeps_every_after_check_mandatory(self):
        for flag in (False, True):
            with self.subTest(flag=flag):
                self.write_before({"requestedViewportTableSweeps"}, repeat_sweeps=flag)
                self.args.allow_before_render_failures = True
                with self.assertRaises(ValueError): review.choose_source(self.root, self.args)
                self.write_before(review.BEFORE_RENDER_FAILURE_CHECKS, repeat_sweeps=flag)
                self.args.repeat_table_sweeps_per_viewport = flag
                valid = self.after()
                self.assertTrue(review.pinned_before_evidence(self.root, valid, self.before_path, self.before))
                for key in review.REPLAY_CHECK_NAMES:
                    for mutation in ("false", "missing", "nonboolean"):
                        with self.subTest(key=key, mutation=mutation):
                            after = json.loads(json.dumps(valid))
                            if mutation == "missing": del after["checks"][key]
                            else: after["checks"][key] = False if mutation == "false" else 1
                            self.assertFalse(review.pinned_before_evidence(self.root, after, self.before_path, self.before))
                after = json.loads(json.dumps(valid)); after["checks"]["unknownCheck"] = True
                self.assertFalse(review.pinned_before_evidence(self.root, after, self.before_path, self.before))
                after = json.loads(json.dumps(valid)); del after["repeatTableSweepsPerViewport"]
                self.assertFalse(review.pinned_before_evidence(self.root, after, self.before_path, self.before))

    def tab_fixture(self, case_dir=None):
        self.source.write_text(json.dumps([{"updateComponents": {"components": [
            {"id": "d", "component": "Tabs", "tabs": [
                {"title": "Week 1", "child": "e"}, {"title": "Week 2", "child": "f"}]},
            {"id": "e", "component": "Text", "text": "First authored content"},
            {"id": "f", "component": "Text", "text": "Second authored content"}]}}]))
        case_dir = case_dir or self.root / "states"
        case_dir.mkdir(parents=True, exist_ok=True)
        captures, evidence = [], []
        def capture(name, label, requested=True):
            texts = ["Week 1", "Week 2", "Content for " + label]
            data = {"name": name, "screenshot": True, "visibleTexts": texts}
            if requested: data["requestedTabLabel"] = label
            captures.append(data)
            (case_dir / (name + ".png")).write_bytes(b"fixture")
            (case_dir / (name + ".xml")).write_text(
                f'<hierarchy><node package="{review.PACKAGE}" class="android.view.View" scrollable="true" bounds="[0,0][600,100]">' + ''.join(
                    f'<node package="{review.PACKAGE}" class="android.view.View" clickable="{str(tab != label).lower()}" '
                    f'selected="{str(tab == label).lower()}" bounds="[{index * 250},0][{(index + 1) * 250},100]">'
                    f'<node package="{review.PACKAGE}" class="android.widget.TextView" text="{tab}" clickable="false" selected="false" '
                    f'bounds="[{index * 250 + 30},25][{index * 250 + 200},75]"/></node>'
                    for index, tab in enumerate(("Week 1", "Week 2"))) + '</node>' +
                f'<node package="{review.PACKAGE}" class="android.view.View" clickable="true" selected="true" bounds="[0,200][600,700]">'
                f'<node package="{review.PACKAGE}" text="Week 1" bounds="[20,240][190,290]"/></node></hierarchy>')
            return {"capture": name, "selectedLabel": label, "visibleTexts": texts}
        capture("tabs_1_top_1", "Week 1", requested=False)
        for index, (label, child) in enumerate((('Week 1', 'e'), ('Week 2', 'f'))):
            prefix = f"tabs_1_{index + 1}"
            if index: capture(prefix + "_top_1", "Week 1")
            pages = [capture(prefix + "_selected", label), capture(prefix + "_vertical_1", label)]
            evidence.append({"tabsId": "d", "authoredTabIndex": index, "selectedLabel": label,
                "authoredChild": child, "selectedStateObserved": True, "scrollEndObserved": True, "pages": pages})
        capture("tabs_1_restored", "Week 1")
        return self.source, case_dir, {"verifyTabViews": True}, {
            "tabViewsVerified": True, "tabViewEvidence": evidence, "captures": captures}

    def test_tab_evidence_proves_all_authored_states_pages_and_restoration(self):
        source, case_dir, config, report = self.tab_fixture()
        self.assertTrue(review.tab_view_evidence_complete(source, case_dir, config, report))
        self.assertEqual([('d', [('Week 1', 'e'), ('Week 2', 'f')])], review.authored_tab_targets(source))
        for mutation in ("missing_state", "wrong_label", "wrong_child", "not_selected", "not_at_end",
                         "missing_pages", "repeated_page", "missing_text", "config_ignored", "not_verified", "wrong_restore",
                         "selected_xml", "restored_xml", "missing_png", "missing_xml"):
            with self.subTest(mutation=mutation):
                source, case_dir, config, report = self.tab_fixture()
                state = report["tabViewEvidence"][0]
                if mutation == "missing_state": report["tabViewEvidence"].pop()
                if mutation == "wrong_label": state["selectedLabel"] = "Week 2"
                if mutation == "wrong_child": state["authoredChild"] = "f"
                if mutation == "not_selected": state["selectedStateObserved"] = False
                if mutation == "not_at_end": state["scrollEndObserved"] = False
                if mutation == "missing_pages": state["pages"] = state["pages"][:1]
                if mutation == "repeated_page": state["pages"][1] = state["pages"][0]
                if mutation == "missing_text": state["pages"][0]["visibleTexts"] = []
                if mutation == "config_ignored": config["verifyTabViews"] = False
                if mutation == "not_verified": report["tabViewsVerified"] = False
                if mutation == "wrong_restore": report["captures"][-1]["requestedTabLabel"] = "Week 2"
                if mutation in ("selected_xml", "restored_xml"):
                    name = "tabs_1_1_selected" if mutation == "selected_xml" else "tabs_1_restored"
                    (case_dir / (name + ".xml")).write_text((case_dir / "tabs_1_2_selected.xml").read_text())
                if mutation in ("missing_png", "missing_xml"):
                    suffix = ".png" if mutation == "missing_png" else ".xml"
                    (case_dir / ("tabs_1_2_selected" + suffix)).unlink()
                self.assertFalse(review.tab_view_evidence_complete(source, case_dir, config, report))

    def test_selected_tab_xml_resolves_split_semantics_and_excludes_identical_table_labels(self):
        _, case_dir, _, _ = self.tab_fixture()
        labels = {"Week 1", "Week 2"}
        self.assertEqual({"Week 1"}, review.observed_selected_tab_labels(case_dir / "tabs_1_1_selected.xml", labels))
        self.assertEqual({"Week 2"}, review.observed_selected_tab_labels(case_dir / "tabs_1_2_selected.xml", labels))
        merged = case_dir / "merged.xml"
        merged.write_text(f'<hierarchy><node package="{review.PACKAGE}" class="android.widget.HorizontalScrollView" bounds="[0,0][600,100]">'
            f'<node package="{review.PACKAGE}" text="Week 1" selected="true" clickable="true" bounds="[0,0][250,100]"/>'
            f'<node package="{review.PACKAGE}" text="Week 2" selected="false" clickable="true" bounds="[250,0][500,100]"/>'
            '</node></hierarchy>')
        self.assertEqual({"Week 1"}, review.observed_selected_tab_labels(merged, labels))
        merged.write_text(merged.read_text().replace('bounds="[0,0][600,100]"', 'bounds="[0,0][600,900]"'))
        self.assertEqual(set(), review.observed_selected_tab_labels(merged, labels))

    def test_full_fit_tab_strip_requires_authored_sibling_owners_and_one_selection(self):
        import xml.etree.ElementTree as ET
        _, case_dir, _, _ = self.tab_fixture()
        path = case_dir / "full_fit.xml"
        text = (case_dir / "tabs_1_1_selected.xml").read_text().replace('scrollable="true"', 'scrollable="false"')
        path.write_text(text)
        labels = {"Week 1", "Week 2"}
        self.assertEqual({"Week 1"}, review.observed_selected_tab_labels(path, labels))
        for mutation in ("no_selection", "two_selected", "row_buttons", "single_owner", "duplicate_labels", "overlapping_owners"):
            with self.subTest(mutation=mutation):
                root = ET.fromstring(text)
                strip = root.find("node")
                first, second = list(strip)
                if mutation == "no_selection": first.set("selected", "false")
                if mutation == "two_selected": second.set("selected", "true")
                if mutation == "row_buttons":
                    first.set("class", "android.widget.Button"); second.set("class", "android.widget.Button")
                if mutation == "single_owner": strip.remove(second)
                if mutation == "duplicate_labels": second.find("node").set("text", "Week 1")
                if mutation == "overlapping_owners": second.set("bounds", "[200,0][500,100]")
                path.write_text(ET.tostring(root, encoding="unicode"))
                self.assertEqual(set(), review.observed_selected_tab_labels(path, labels))

    def test_tab_flag_preserves_exact_prior_receipt_schemas_and_cannot_be_waived(self):
        for repeat in (None, False, True):
            for tabs in (None, False):
                with self.subTest(repeat=repeat, tabs=tabs):
                    self.write_before(repeat_sweeps=repeat, verify_tabs=tabs)
                    self.assertEqual([], review.before_replay_overrides(self.before))
        self.write_before({"requestedTabViews"}, verify_tabs=True)
        with self.assertRaises(ValueError): review.before_replay_overrides(self.before, True)
        for invalid in (None, 0, 1, "false"):
            self.write_before(verify_tabs=False)
            self.before["verifyTabViews"] = invalid
            with self.assertRaises(ValueError): review.before_replay_overrides(self.before)
        self.write_before(verify_tabs=False)
        del self.before["checks"]["requestedTabViews"]
        with self.assertRaises(ValueError): review.before_replay_overrides(self.before)
        self.write_before()
        self.before["checks"]["requestedTabViews"] = True
        with self.assertRaises(ValueError): review.before_replay_overrides(self.before)

    def test_tab_optin_rejects_unsupported_or_ambiguous_input_before_device_access(self):
        source, _, _, _ = self.tab_fixture()
        payload = review.read(source)
        payload[0]["updateComponents"]["components"][0]["tabs"][1]["title"] = "Week 1"
        source.write_text(json.dumps(payload))
        with self.assertRaises(ValueError): review.authored_tab_targets(source)
        self.source = self.root / "output.express"
        self.source.write_text('<a2ui>root=Text("Saved Express")</a2ui>')
        self.write_before()
        self.args.verify_tab_views = True
        with patch.object(review, "Adb") as adb:
            with self.assertRaises(ValueError): review.collect(self.args)
            adb.assert_not_called()

    def test_collect_records_failure_when_requested_tab_state_is_omitted(self):
        self.tab_fixture()
        self.write_before()
        self.args.verify_tab_views = True
        source_digest = review.digest(self.source)
        class FakeAdb:
            def __init__(self, *args): pass
            def absent(self, *args): pass
            def command(self, *args, **kwargs): return ""
            def file_hash(self, *args): return source_digest
        def instrument(adb, invocation, remote, target, timeout, receipt):
            self.assertEqual("true", invocation[invocation.index("verifyTabViews") + 1])
            artifacts = target / "artifacts"
            _, _, config, report = self.tab_fixture(artifacts / self.args.case)
            report.update({"id": self.args.case, "sourceJsonSha256": source_digest, "status": "rendered",
                "issues": [], "verticalEndObserved": True, "verticalLimitReached": False, "tables": []})
            report["tabViewEvidence"].pop()
            config.update({"cases": [self.args.case], "corpusSha256": review.digest(review.CORPUS), "inferenceEvaluated": False})
            (artifacts / "replay_results.json").write_text(json.dumps([report]))
            (artifacts / "replay_config.json").write_text(json.dumps(config))
            (artifacts / "replay_summary.json").write_text(json.dumps({"modelCalls": 0}))
            receipt["instrumentationPassed"] = True
        with patch.object(review, "Adb", FakeAdb), patch.object(review, "provenance", return_value={}), \
                patch.object(review, "instrument", side_effect=instrument), patch.object(review, "build_report"):
            with self.assertRaises(ValueError): review.collect(self.args)
        failed = review.read(self.root / "BXP-018/after/r01/receipt.json")
        self.assertTrue(failed["verifyTabViews"])
        self.assertEqual("failed", failed["status"])
        self.assertEqual(["requestedTabViews"], [key for key, value in failed["checks"].items() if value is False])
        self.assertFalse(review.pinned_before_evidence(self.root, failed, self.before_path, self.before))

    def test_pinned_tab_receipt_rechecks_xml_even_when_all_recorded_checks_pass(self):
        self.tab_fixture()
        self.write_before()
        self.args.verify_tab_views = True
        after = self.after()
        target = self.root / "BXP-018/after/r01"
        staged = target / "input/BXP-018/output.a2ui.json"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(self.source.read_bytes())
        after["case"] = self.args.case
        after["source"]["archivedPath"] = staged.relative_to(self.root).as_posix()
        _, case_dir, config, report = self.tab_fixture(target / "artifacts/BXP-018")
        (target / "artifacts/replay_config.json").write_text(json.dumps(config))
        after["result"] = report
        self.assertTrue(review.pinned_before_evidence(self.root, after, self.before_path, self.before))
        (case_dir / "tabs_1_restored.xml").write_text((case_dir / "tabs_1_2_selected.xml").read_text())
        self.assertTrue(all(after["checks"].values()))
        self.assertFalse(review.pinned_before_evidence(self.root, after, self.before_path, self.before))


    def repair_fixture(self, phase="before", recovered="First repaired output", kind="GENERATED_DSL_REPAIR"):
        target = self.root / f"BXP-018/{phase}/r01"
        raw = target / "input/BXP-018/output.express"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b'<a2ui>root=Text("Unchanged raw")</a2ui>')
        case_dir = target / "artifacts/BXP-018"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "source.output.express").write_bytes(raw.read_bytes())
        (case_dir / "recovered.output.express").write_text(recovered)
        (case_dir / "recovered.output.a2ui.json").write_text(json.dumps([{"repairFixture": recovered}]))
        source = {"sha256": review.digest(raw), "replayMode": review.EXPRESS_REPAIR_ONLY,
            "archivedPath": raw.relative_to(self.root).as_posix()}
        config = {"replayMode": review.EXPRESS_REPAIR_ONLY, "kind": "generated_dsl_repair_no_fallback"}
        report = dict(config, generatedDslRepairAccepted=True, repairKind=kind, sourceExpressSha256=review.digest(raw),
            recoveredExpressSha256=review.digest(case_dir / "recovered.output.express"),
            recoveredJsonSha256=review.digest(case_dir / "recovered.output.a2ui.json"), sourceIntegrityAccepted=False)
        summary = dict(config, sourceTextFallbacks=0, repairCounts={kind: 1}, repairRejected=0, renderFailures=0)
        (target / "artifacts/replay_config.json").write_text(json.dumps(config))
        checks, artifacts = review.express_repair_evidence(self.root, case_dir, source, config, report, summary)
        receipt = {"case": "BXP-018", "phase": phase, "attempt": "r01", "startedAtUtc": "2026-10-09",
            "source": source, "result": report, "summary": summary, "repairArtifacts": artifacts,
            "repeatTableSweepsPerViewport": False, "verifyTabViews": False,
            "status": "collected", "automatedChecksSatisfied": True,
            "checks": dict({key: True for key in review.REPLAY_CHECK_NAMES}, **checks)}
        path = target / "receipt.json"
        path.write_text(json.dumps(receipt))
        return path, receipt, case_dir, config

    def test_repair_only_mode_requires_raw_express_and_after_inherits_the_pinned_mode(self):
        path, before, _, _ = self.repair_fixture()
        args = SimpleNamespace(case="BXP-018", phase="after", source=None, source_info=None,
            allow_before_render_failures=False, replay_mode=None)
        raw, source = review.choose_source(self.root, args)
        self.assertEqual("output.express", raw.name)
        self.assertEqual(review.EXPRESS_REPAIR_ONLY, source["replayMode"])
        self.assertEqual(review.digest(path), source["pinnedBeforeReceiptSha256"])
        args.replay_mode = "express"
        with self.assertRaises(ValueError): review.choose_source(self.root, args)
        args.phase = "before"; args.source = self.source; args.replay_mode = review.EXPRESS_REPAIR_ONLY
        with self.assertRaises(ValueError): review.choose_source(self.root, args)
        parsed = review.parser().parse_args(["replay", "--case", "BXP-046", "--phase", "before",
            "--replay-mode", "express_repair_only"])
        self.assertEqual(review.EXPRESS_REPAIR_ONLY, parsed.replay_mode)

    def test_repair_only_evidence_accepts_nonfallback_kinds_and_rejects_bad_hashes_or_policy(self):
        for kind in ("NONE", "STRUCTURAL", "GENERATED_DSL_REPAIR"):
            _, receipt, _, _ = self.repair_fixture(kind=kind)
            self.assertTrue(review.receipt_express_repair_complete(self.root, receipt))
            self.assertFalse(receipt["result"]["sourceIntegrityAccepted"])
        for mutation in ("fallback", "wrong_mode", "rejected", "fallback_count", "missing_recovery", "raw_changed", "recovered_changed"):
            with self.subTest(mutation=mutation):
                _, receipt, case_dir, config = self.repair_fixture()
                report, summary = receipt["result"], receipt["summary"]
                if mutation == "fallback": report["repairKind"] = "SOURCE_TEXT_FALLBACK"
                if mutation == "wrong_mode": config["replayMode"] = "express_repair"
                if mutation == "rejected": report["generatedDslRepairAccepted"] = False
                if mutation == "fallback_count": summary["sourceTextFallbacks"] = 1
                if mutation == "missing_recovery": (case_dir / "recovered.output.a2ui.json").unlink()
                if mutation == "raw_changed": (case_dir / "source.output.express").write_bytes(b"Changed raw")
                if mutation == "recovered_changed": (case_dir / "recovered.output.express").write_bytes(b"Changed repaired output")
                checks, _ = review.express_repair_evidence(self.root, case_dir, receipt["source"], config, report, summary)
                self.assertFalse(all(checks.values()))

    def test_same_raw_comparison_retains_different_repaired_hashes_and_never_waives_repair_failures(self):
        before_path, before, _, _ = self.repair_fixture("before", recovered="Older compiler output")
        _, after, case_dir, _ = self.repair_fixture("after", recovered="Newer compiler output")
        after["source"].update(pinnedBeforeReceipt=before_path.relative_to(self.root).as_posix(),
            pinnedBeforeReceiptSha256=review.digest(before_path))
        comparison = review.express_repair_comparison(before, after)
        self.assertTrue(comparison["sameRawInput"])
        self.assertFalse(comparison["sameRecoveredExpress"])
        self.assertFalse(comparison["sameRecoveredJson"])
        self.assertTrue(review.pinned_before_evidence(self.root, after, before_path, before))
        for check in review.EXPRESS_REPAIR_CHECK_NAMES:
            broken = json.loads(json.dumps(before))
            broken["checks"][check] = False; broken["status"] = "failed"; broken["automatedChecksSatisfied"] = False
            with self.assertRaises(ValueError): review.before_replay_overrides(broken, True)
            broken = json.loads(json.dumps(after)); broken["checks"][check] = False
            self.assertFalse(review.pinned_before_evidence(self.root, broken, before_path, before))
        (case_dir / "recovered.output.a2ui.json").write_text("[]")
        self.assertTrue(all(after["checks"].values()))
        self.assertFalse(review.pinned_before_evidence(self.root, after, before_path, before))


    def progress_fixture(self, repair_mode=True):
        before_path, before, _, _ = self.repair_fixture("before", recovered="Before repaired output")
        after_path, after, after_case, _ = self.repair_fixture("after", recovered="After repaired output")
        native = self.root / "native_generation/generated_r1/BXP-018"
        native.mkdir(parents=True, exist_ok=True)
        raw = native / "output.express"; raw.write_bytes((self.root / before["source"]["archivedPath"]).read_bytes())
        final = native / "a2ui.json"; final.write_text('[{"prepared": "original final"}]')
        (native / "result.json").write_text(json.dumps({"rawStrictValid": False, "repairKind": "GENERATED_DSL_REPAIR"}))
        if not repair_mode:
            for receipt, path in ((before, before_path), (after, after_path)):
                staged = path.parent / "input/BXP-018/output.a2ui.json"; staged.write_bytes(final.read_bytes())
                receipt["source"].update(replayMode="json", sha256=review.digest(final), archivedPath=staged.relative_to(self.root).as_posix())
                for check in review.EXPRESS_REPAIR_CHECK_NAMES: del receipt["checks"][check]
        before_path.write_text(json.dumps(before))
        after["source"].update(pinnedBeforeReceipt=before_path.relative_to(self.root).as_posix(), pinnedBeforeReceiptSha256=review.digest(before_path))
        capture = after_case / "initial.png"; capture.write_bytes(b"host-only fixture")
        xml = after_case / "initial.xml"; xml.write_text("<hierarchy/>")
        after["capturedFiles"] = [{"path": p.relative_to(self.root).as_posix(), "sha256": review.digest(p)} for p in (capture, xml)]
        if repair_mode: after["repairComparison"] = review.express_repair_comparison(before, after)
        after_path.write_text(json.dumps(after))
        manifest = {"cases": [{"id": "BXP-018", "runId": "generated_r1", "recordedGenerationStatus": "valid",
            "raw": "generated_r1/BXP-018/output.express", "rawSha256": review.digest(raw),
            "final": "generated_r1/BXP-018/a2ui.json", "finalSha256": review.digest(final), "result": "generated_r1/BXP-018/result.json"}],
            "priorValidationReferences": {"cases": []}}
        (self.root / "native_generation/manifest.json").write_text(json.dumps(manifest))
        (self.root / "review_findings.json").write_text('{"cases": {}}')
        annotation = {"status": "accepted", "notes": ["Explicit fixture review"],
            "reviewedAfterReceipt": after_path.relative_to(self.root).as_posix(), "reviewedAfterReceiptSha256": review.digest(after_path),
            "sourceSha256": after["source"]["sha256"]}
        (self.root / "BXP-018/review.json").write_text(json.dumps(annotation))
        corpus = self.root / "corpus.jsonl"
        corpus.write_text("\n".join(json.dumps({"id": f"BXP-{index:03}", "domain": "generic"}) for index in range(1, 51)))
        path = Path(__file__).resolve().parents[1] / "validation/20261009_fold7_visual_review/refresh_progress.py"
        spec = importlib.util.spec_from_file_location("repair_progress_tests", path)
        progress = importlib.util.module_from_spec(spec); spec.loader.exec_module(progress)
        progress.REVIEW = self.root; progress.WORKSPACE = review.ROOT; progress.CORPUS = corpus; progress._receipt_checks = review
        return progress, manifest, before_path, after_path, after, final

    def test_progress_repair_receipt_selects_manifest_raw_and_keeps_original_final_separate(self):
        progress, _, before_path, after_path, _, final = self.progress_fixture()
        original = {path: review.digest(path) for path in (before_path, after_path, final)}
        result = progress.update(self.root / "progress_out")
        record = next(c for c in result["cases"] if c["id"] == "BXP-018")
        self.assertEqual("accepted", record["visualStatus"])
        self.assertEqual(review.EXPRESS_REPAIR_ONLY, record["selectedSource"]["format"])
        self.assertEqual([review.digest(final)], [v["sha256"] for v in record["preparedInputs"]])
        self.assertEqual(1, result["availableCurrentModelDocuments"])
        self.assertEqual(1, result["casesWithVerifiedRepairOutputs"])
        self.assertNotEqual(record["selectedSource"]["sha256"], record["repairedOutput"]["sha256"])
        self.assertNotEqual(review.digest(final), record["repairedOutput"]["sha256"])
        self.assertFalse(record["repairComparison"]["sameRecoveredJson"])
        self.assertTrue(record["acceptanceChecks"]["sourceMatchesManifestPinnedRaw"])
        self.assertTrue(record["acceptanceChecks"]["recoveredDocumentHashesMatch"])
        self.assertTrue(all(review.digest(path) == digest for path, digest in original.items()))

    def test_progress_legacy_json_receipt_stays_canonical_without_selecting_raw(self):
        progress, _, _, _, _, final = self.progress_fixture(repair_mode=False)
        result = progress.update(self.root / "progress_out")
        record = next(c for c in result["cases"] if c["id"] == "BXP-018")
        self.assertEqual("accepted", record["visualStatus"])
        self.assertEqual(review.digest(final), record["selectedSource"]["sha256"])
        self.assertIsNone(record["repairedOutput"])
        self.assertNotIn("sourceMatchesManifestPinnedRaw", record["acceptanceChecks"])
        self.assertEqual(0, result["casesWithVerifiedRepairOutputs"])
        self.assertEqual(1, result["availableCurrentModelDocuments"])

    def test_progress_rejects_manifest_raw_hash_or_case_path_mismatch_without_json_fallback(self):
        for mutation in ("wrong_hash", "wrong_case_path", "missing_raw"):
            with self.subTest(mutation=mutation):
                progress, manifest, _, _, _, final = self.progress_fixture()
                raw = self.root / "native_generation" / manifest["cases"][0]["raw"]
                if mutation == "wrong_hash": manifest["cases"][0]["rawSha256"] = "0" * 64
                if mutation == "wrong_case_path":
                    other = self.root / "native_generation/generated_r1/BXP-019/output.express"
                    other.parent.mkdir(parents=True, exist_ok=True); other.write_bytes(raw.read_bytes())
                    manifest["cases"][0]["raw"] = "generated_r1/BXP-019/output.express"
                if mutation == "missing_raw": raw.unlink()
                (self.root / "native_generation/manifest.json").write_text(json.dumps(manifest))
                result = progress.update(self.root / "progress_out")
                record = next(c for c in result["cases"] if c["id"] == "BXP-018")
                self.assertIsNone(record["selectedSource"])
                self.assertIsNone(record["repairedOutput"])
                self.assertNotEqual("accepted", record["visualStatus"])
                self.assertEqual([review.digest(final)], [v["sha256"] for v in record["preparedInputs"]])
                self.assertEqual(1, result["availableCurrentModelDocuments"])

    def test_progress_never_credits_raw_as_prepared_final_and_rechecks_repaired_artifact(self):
        progress, _, _, _, after, final = self.progress_fixture()
        prepared = [{"path": "canonical", "sha256": review.digest(final), "format": "json"}]
        raw = [{"path": "raw", "sha256": after["source"]["sha256"], "format": review.EXPRESS_REPAIR_ONLY}]
        ordinary = {"source": {"replayMode": "express", "sha256": raw[0]["sha256"]}}
        self.assertEqual(prepared[0], progress.select_saved_source(ordinary, prepared, raw))
        self.assertNotEqual(raw[0], progress.select_saved_source(ordinary, prepared, raw))
        (self.root / after["repairArtifacts"]["recoveredJson"]["path"]).write_text("[]")
        result = progress.update(self.root / "progress_out")
        record = next(c for c in result["cases"] if c["id"] == "BXP-018")
        self.assertIsNone(record["repairedOutput"])
        self.assertFalse(record["acceptanceChecks"]["recoveredDocumentHashesMatch"])
        self.assertNotEqual("accepted", record["visualStatus"])
        self.assertEqual(1, result["availableCurrentModelDocuments"])


class AnnotationReceiptProvenanceTest(unittest.TestCase):
    def setUp(self):
        scratch = review.ROOT / ".tmp"
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="f7_annotation_", dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "BXP-040"
        self.directory.mkdir()
        self.args = SimpleNamespace(output=self.root, case="BXP-040", status="needs_work",
            note=["Readable recovery; duplicate sections still need work."], notes_file=None, reviewer="codex")

    def receipt(self, attempt, started, status="collected", passed=True):
        path = self.directory / "after" / attempt / "receipt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"case": "BXP-040", "phase": "after", "attempt": attempt, "startedAtUtc": started,
            "status": status, "automatedChecksSatisfied": passed,
            "checks": {"renderedWithoutIssues": passed}, "source": {"sha256": attempt.ljust(64, "0")}}
        review.save(path, value)
        return path, value

    def annotate(self):
        with patch.object(review, "build_report"):
            review.annotate(self.args)
        return review.read(self.directory / "review.json")

    def test_needs_work_pins_newest_failed_after_without_mutating_receipts(self):
        old_path, _ = self.receipt("r01", "2026-10-09T01:00:00Z")
        new_path, newest = self.receipt("r02", "2026-10-09T02:00:00Z", status="failed", passed=False)
        original = {path: path.read_bytes() for path in (old_path, new_path)}
        result = self.annotate()
        self.assertEqual("needs_work", result["status"])
        self.assertEqual(self.args.note, result["notes"])
        self.assertEqual(new_path.relative_to(self.root).as_posix(), result["reviewedAfterReceipt"])
        self.assertEqual(review.digest(new_path), result["reviewedAfterReceiptSha256"])
        self.assertEqual(newest["source"]["sha256"], result["sourceSha256"])
        self.assertIn("automated checks need not pass", result["scope"])
        self.assertEqual(original, {path: path.read_bytes() for path in original})

    def test_needs_work_before_only_has_no_invented_after_provenance(self):
        result = self.annotate()
        self.assertEqual("needs_work", result["status"])
        for field in ("reviewedAfterReceipt", "reviewedAfterReceiptSha256", "sourceSha256"):
            self.assertNotIn(field, result)

    def test_accepted_still_rejects_newest_failed_or_incomplete_after(self):
        self.args.status = "accepted"
        self.receipt("r01", "2026-10-09T01:00:00Z")
        for status, passed in (("failed", False), ("collected", False), ("failed", True), ("running", True)):
            with self.subTest(status=status, passed=passed):
                path, _ = self.receipt("r02", "2026-10-09T02:00:00Z", status=status, passed=passed)
                original = path.read_bytes()
                with self.assertRaises(ValueError): self.annotate()
                self.assertEqual(original, path.read_bytes())
                self.assertFalse((self.directory / "review.json").exists())

    def test_accepted_requires_notes_and_pins_passing_after(self):
        self.args.status = "accepted"
        path, receipt = self.receipt("r01", "2026-10-09T01:00:00Z")
        self.args.note = []
        with self.assertRaises(ValueError): self.annotate()
        self.args.note = ["All after captures inspected."]
        result = self.annotate()
        self.assertEqual(path.relative_to(self.root).as_posix(), result["reviewedAfterReceipt"])
        self.assertEqual(review.digest(path), result["reviewedAfterReceiptSha256"])
        self.assertEqual(receipt["source"]["sha256"], result["sourceSha256"])
        self.assertNotIn("automated checks need not pass", result["scope"])


if __name__ == "__main__":
    unittest.main()
