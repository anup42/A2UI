import copy
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pipeline.source_quality import (
    QueryQualityIndex, QualityQueryWriter, assess_source_quality,
    asset_verification_metadata, calculate, source_contract_prompt,
)
from pipeline.stage1_queries import run_stage1
from pipeline.stage2_responses import run_stage2


def contract(**fields):
    return {"version": 1, "verification": {"status": "reviewed", "reference": "test-fixture-v1"}, **fields}


def query(**fields):
    return {"query_id": "q_000001", "intent": "calculation", "query_text": "Add the supplied values.", **fields}


class SourceQualityTests(unittest.TestCase):
    def test_detects_real_audit_bill_error_without_rewriting_text(self):
        q = query(source_contract=contract(calculations=[{
            "id": "bill", "op": "sum", "values": ["342.50", "51.375", "27.40"],
            "result_label": "Total", "tolerance": "0.005"}]))
        original = copy.deepcopy(q)
        bad = assess_source_quality(q, "Total: $423.55")
        good = assess_source_quality(q, "| **Total** | **$421.28** |")
        self.assertEqual("exclude", bad["training_eligibility"])
        self.assertEqual("eligible", good["training_eligibility"])
        self.assertEqual(original, q)
        self.assertEqual("not_performed", good["prose_fact_verification"])

    def test_contradictory_summary_is_not_hidden_by_correct_table(self):
        q = query(source_contract=contract(calculations=[{
            "id": "grade", "op": "weighted_mean", "values": [4, 3, 2, 1, 0],
            "weights": [8, 15, 12, 10, 5], "result_label": "GPA"}]))
        out = assess_source_quality(q, "GPA: 2.42\n| GPA | 2.22 |")
        self.assertEqual("failed", out["status"])
        self.assertEqual(["2.42", "2.22"], out["checks"][0]["observed"])

    def test_missing_binding_is_not_treated_as_verified(self):
        q = query(source_contract=contract(calculations=[{
            "id": "sum", "op": "sum", "values": [1, 2], "result_label": "Total"}]))
        self.assertEqual("failed", assess_source_quality(q, "The answer is probably three.")["status"])

    def test_loan_formula_catches_reversed_cost_comparison_inputs(self):
        a = calculate({"op": "loan_interest", "amount": 20000, "period_rate": "0.004166666666666666666666666667", "periods": 48})
        b = calculate({"op": "loan_interest", "amount": 20000, "period_rate": "0.002916666666666666666666666667", "periods": 60})
        self.assertAlmostEqual(2108.1218278, float(a), places=6)
        self.assertAlmostEqual(1830.0939643, float(b), places=6)
        self.assertGreater(a, b)
        self.assertEqual(Decimal(100), calculate({"op": "loan_payment", "amount": 1200, "period_rate": 0, "periods": 12}))

    def test_arithmetic_language_cannot_execute_expressions(self):
        for spec in ({"op": "eval", "values": ["__import__('os')"]},
                     {"op": "sum", "values": ["NaN"]},
                     {"op": "ratio", "a": 1, "b": 0},
                     {"op": "loan_payment", "amount": 1, "period_rate": .1, "periods": 99999999}):
            with self.subTest(spec=spec), self.assertRaises((ValueError, ArithmeticError)):
                calculate(spec)

    def test_fact_role_change_is_detected_not_only_ip_token_presence(self):
        q = query(query_text="Affected IP 192.168.1.45", source_contract=contract(facts=[{
            "id": "host", "value": "192.168.1.45", "label": "Affected IP", "provenance": {"kind": "query"}}]))
        self.assertEqual("failed", assess_source_quality(q, "| Source IP | 192.168.1.45 |")["status"])
        self.assertEqual("checks_passed", assess_source_quality(q, "| Affected IP | 192.168.1.45 |")["status"])

    def test_fact_claim_not_in_query_fails_provenance(self):
        q = query(source_contract=contract(facts=[{"id": "server", "value": "Prod-04", "provenance": {"kind": "query"}}]))
        self.assertEqual("failed", assess_source_quality(q, "Prod-04")["status"])

    def test_schedule_checks_window_and_overlap(self):
        q = query(source_contract=contract(constraints=[{"id": "day", "kind": "schedule", "window_start": "09:00", "window_end": "18:00"}]))
        for bad in ["18:00-20:00 Dinner", "09:00-13:00 Coding\n12:00-14:00 Lunch", "9 AM coding"]:
            with self.subTest(bad=bad):
                self.assertEqual("failed", assess_source_quality(q, bad)["status"])
        self.assertEqual("checks_passed", assess_source_quality(q, "09:00-13:00 Coding\n16:00-18:00 Dinner")["status"])

    def test_budget_and_forbidden_constraint(self):
        q = query(source_contract=contract(constraints=[
            {"id": "budget", "kind": "number_range", "label": "Cost", "min": 0, "max": 2000},
            {"id": "outdoor", "kind": "forbidden_text", "value": "Indoor/Hybrid"}]))
        out = assess_source_quality(q, "Cost: 2070\nActivity: Indoor/Hybrid")
        self.assertEqual(2, sum(not c["passed"] for c in out["checks"]))

    def test_legacy_placeholder_urls_request_review_but_are_not_rejected(self):
        out = assess_source_quality(query(), "Action: [Button: Export] https://example.com/export")
        self.assertEqual("needs_review", out["status"])
        self.assertEqual("review", out["training_eligibility"])
        self.assertNotIn("error", [f["severity"] for f in out["findings"]])

    def test_explicit_mock_action_is_allowed_without_network(self):
        q = query(source_contract=contract(modality="interactive_tool", actions=[{
            "label": "Export", "destination": "https://example.com/export", "mode": "mock"}]))
        self.assertEqual("eligible", assess_source_quality(q, "Action: [Button: Export] https://example.com/export")["training_eligibility"])

    def test_missing_interface_actions_do_not_pass_on_prose_coverage(self):
        q = query(source_contract=contract(modality="interactive_tool", actions=[{
            "label": "Copy Text", "destination": "action://copy/scan", "mode": "capability",
            "capability_id": "copy_text", "parameters": {"text": "fixture"}}]))
        self.assertEqual("failed", assess_source_quality(q, "The interface has a Copy Text button.")["status"])
        good = assess_source_quality(q, "Action: [Button: Copy Text] action://copy/scan")
        self.assertEqual("checks_passed", good["status"])

    def test_wrong_destination_and_malformed_url_fail(self):
        q = query(source_contract=contract(actions=[{"label": "Rates", "destination": "https://www.myhammer.de/", "mode": "navigation"}]))
        for url in ["https://www.myhamster.de/", "https://.jsdelivr.net/a", "https://host.test/en /"]:
            with self.subTest(url=url):
                self.assertEqual("failed", assess_source_quality(q, f"Action: [Button: Rates] <{url}>")["status"])

    def test_empty_or_malformed_contract_cannot_verify_source(self):
        self.assertEqual("needs_review", assess_source_quality(query(source_contract=contract()), "Hello")["status"])
        for malformed in [[], {"version": 99}, contract(modality=[]), contract(calculations=[{"op": []}]), contract(actions=[{"label": "x", "destination": "app://x", "mode": []}])]:
            with self.subTest(malformed=malformed):
                self.assertEqual("failed", assess_source_quality(query(source_contract=malformed), "Hello")["status"])

    def test_truncated_and_unresolved_action_destinations_are_hard_failures(self):
        cases = {
            "https://www.glassdoor.co.in/Salaries/analyst-salary-SRCH_...": "truncated_http_destination",
            "https://example.test/path/…": "truncated_http_destination",
            "[URL_2...]": "unresolved_destination_placeholder",
            "[URL_2]": "unresolved_destination_placeholder",
            "https://example.test/[ACTION_URL_3]": "unresolved_destination_placeholder",
        }
        for destination, code in cases.items():
            with self.subTest(destination=destination):
                result = assess_source_quality(query(), f"Action: [Button: Open] {destination}")
                self.assertEqual("exclude", result["training_eligibility"])
                self.assertIn(code, [finding["code"] for finding in result["findings"]])

    def test_url_checks_do_not_reject_query_ellipsis_or_symbolic_media(self):
        for destination in ("https://example.test/search?q=...", "https://example.test/search?q=…",
                            "https://example.test/a%2E%2E%2Eb", "https://[::1]/page",
                            "mailto:person@example.test", "tel:+1234567890", "action://copy/text"):
            with self.subTest(destination=destination):
                result = assess_source_quality(query(), f"Media: Icon=[ICON_URL_1]\nAction: [Button: Open] {destination}")
                self.assertNotEqual("failed", result["status"])

    def test_response_self_attestation_is_not_a_source_contract(self):
        response = json.dumps({"source_contract": contract(calculations=[{
            "id": "sum", "op": "sum", "values": [1, 2], "result_label": "Total"}])}) + "\nTotal: 3"
        result = assess_source_quality(query(), response)
        self.assertFalse(result["contract_reviewed"])
        self.assertEqual("needs_review", result["status"])

    def test_assets_are_separate_from_training_eligibility(self):
        offline = asset_verification_metadata(3, 0, 3, offline=True)
        self.assertEqual("skipped_offline", offline["verification_state"])
        self.assertFalse(offline["visual_ready"])
        self.assertFalse(offline["training_assets_required"])
        self.assertEqual("processing_error", asset_verification_metadata(2, 0, 2, offline=False, processing_error=True)["verification_state"])
        self.assertEqual("not_applicable", asset_verification_metadata(0, 0, 0, offline=True)["verification_state"])

    def test_scenario_duplicates_group_for_split_isolation_across_intents(self):
        index = QueryQualityIndex()
        first = index.add(query(query_text="Design a simple interface for a QR scanner with buttons to open links and copy text."))
        second = index.add(query(query_id="q_2", intent="Navigation", query_text="Design a simple interface for a QR scanner with buttons to open links and copy text now."))
        self.assertTrue(second["query_quality"]["near_duplicate_candidates"])
        self.assertEqual(first["scenario_family_id"], second["scenario_family_id"])
        self.assertIn("Recent other-intent scenarios", index.prompt_context("Travel"))
        self.assertNotEqual(index.add(query(query_id="q_3", query_text="Hello"))["scenario_family_id"], index.add(query(query_id="q_4", query_text="Goodbye"))["scenario_family_id"])

    def test_all_writer_paths_annotate_fallback_and_prompt_hash(self):
        sink = Mock()
        writer = QualityQueryWriter(sink, QueryQualityIndex(), "query.md", "prompt")
        writer.append(query(source="fallback"))
        row = sink.append.call_args.args[0]
        self.assertEqual("exclude", row["query_quality"]["training_eligibility"])
        self.assertEqual(64, len(row["gen"]["prompt_template_sha256"]))


class Stage2SourceIntegrationTests(unittest.TestCase):
    def test_mocked_stage1_appends_quality_in_serial_and_cyclic_paths(self):
        for cycle in (0, 1):
            with self.subTest(cycle=cycle), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)
                (p / "intents.txt").write_text("calculation\n", encoding="utf8")
                (p / "prompt.md").write_text("# query_gen_test_v3\nIntent {intent}; count {k}", encoding="utf8")
                generated = [{"query_text": "Calculate the mean of 1, 2 and 3.", "difficulty": "easy", "tags": ["math"]}]
                adapter = SimpleNamespace(spec=SimpleNamespace(name="mock", provider="mock", model="mock", supports_json_mode=False),
                                          generate=Mock(return_value=SimpleNamespace(text=json.dumps(generated), raw={}, latency_ms=0, input_tokens=0, output_tokens=0, provider="mock", model="mock", error=None)))
                cache = Mock(); cache.get.return_value = None
                run_stage1(p/"intents.txt", p/"prompt.md", adapter, p, p/"queries.jsonl", 1, 1, 1, .1, 100, Mock(), cache, logging.getLogger("source-test"), max_total=1, fill_missing_with_fallback=False, intent_cycle_size=cycle)
                row = json.loads((p/"queries.jsonl").read_text(encoding="utf8"))
                self.assertIn("query_quality", row)
                self.assertIn("scenario_family_id", row)
                self.assertEqual("query_gen_test_v3", row["gen"]["prompt_version"])
                self.assertIn("Scenario diversity guard", adapter.generate.call_args.kwargs["prompt"])

    def test_mocked_stage2_preserves_source_and_annotates_offline_assets(self):
        # Stubbed adapter and transport: this test must never run model inference.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            q = query(source_contract=contract(calculations=[{"id": "sum", "op": "sum", "values": [1, 2], "result_label": "Total"}]), scenario_family_id="fixture-family")
            (p / "queries.jsonl").write_text(json.dumps(q) + "\n", encoding="utf8")
            (p / "prompt.md").write_text("Answer {query_text}", encoding="utf8")
            response = "Total: 3\nMedia: Icon=https://example.com/icon.svg"
            adapter = SimpleNamespace(spec=SimpleNamespace(name="mock", provider="mock", model="mock", supports_json_mode=False),
                                      generate=Mock(return_value=SimpleNamespace(text=response, raw={}, latency_ms=0, input_tokens=0, output_tokens=0, provider="mock", model="mock", error=None)))
            cache = Mock(); cache.get.return_value = None
            env = {"DATASET_OFFLINE_MODE": "1", "INTERNET": "0", "STAGE2_ASSET_WORKERS": "1", "STAGE2_REAL_ASSET_RETRY": "0"}
            with patch.dict(os.environ, env), patch("pipeline.stage2_responses._load_local_icon_context", return_value=None), patch("pipeline.stage2_responses._commons_enrichment_enabled", return_value=False), patch("pipeline.stage2_responses.urlopen", side_effect=AssertionError("network forbidden")):
                run_stage2(p/"queries.jsonl", p/"prompt.md", None, adapter, p/"responses.jsonl", 1, 1, 1, False, False, [.1], 100, 1, Mock(), cache, logging.getLogger("source-test"), max_total=1)
            row = json.loads((p/"responses.jsonl").read_text(encoding="utf8"))
            self.assertEqual("checks_passed", row["source_quality"]["status"])
            self.assertEqual("skipped_offline", row["asset_stats"]["verification_state"])
            self.assertEqual("fixture-family", row["scenario_family_id"])
            self.assertEqual(q["source_contract"], row["source_contract"])
            self.assertEqual(response, row["response_text"])
            self.assertIn("Source correctness policy", adapter.generate.call_args.kwargs["prompt"])


if __name__ == "__main__":
    unittest.main()
