"""Evidence and replay tests for repairs of the known v10 manual-review holds."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/scripts"))
import prepare_space_v11 as preparation
import space_v11_known_hold_repairs as repairs
from space_v11_repairs import residual_gates


class KnownHoldRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(os.environ.get("A2UI_V10_POLICY_REPO", "D:/git/anup-code/A2UI"))
        path = repo / "training/reports/v10_manual100_20260916/samples.jsonl"
        if not path.is_file():
            raise unittest.SkipTest("Requires the explicit original v10 audited samples")
        preparation.init(repo)
        cls.samples = {r["sample"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}

    def row(self, case):
        from ir_training.data.express_preparation import TASK_PREFIX
        sample = self.samples[case]
        return {"id": sample["id"], "row_id": sample["id"], "source_id": sample["source_id"],
                "response_text": sample["source"], "completion": sample["completion"],
                "messages": [{"role": "user", "content": TASK_PREFIX + sample["source"]},
                             {"role": "assistant", "content": sample["completion"]}],
                "metadata": {"assigned_split": sample["split"], "archive_recovery": {
                    "original_source_sha256": sample["lineage"]["original_source_sha256"],
                    "original_target_sha256": sample["lineage"]["original_target_sha256"]}}}

    def fixed(self, case):
        from ir_training.data.express_preparation import serialize_checked
        row = self.row(case)
        original = deepcopy(row)
        fixed, proofs, reasons = repairs.repair_known_hold(row, case)
        self.assertEqual(row, original)
        self.assertEqual(reasons, [])
        self.assertEqual(len(proofs), 1)
        self.assertTrue(proofs[0]["changes"])
        self.assertEqual(fixed["id"], row["id"])
        self.assertEqual(fixed["source_id"], row["source_id"])
        self.assertEqual(fixed["metadata"]["assigned_split"], row["metadata"]["assigned_split"])
        self.assertEqual(proofs[0]["source_factual_status"], "needs_review")
        checked = serialize_checked(fixed["completion"], "root-first")
        self.assertEqual(residual_gates(fixed["response_text"], fixed["completion"], checked), [])
        replay = repairs.repair_known_hold(row, case)
        self.assertEqual(replay, (fixed, proofs, reasons))
        self.assertEqual(repairs.repair_known_hold(fixed, case), (fixed, [], []))
        return fixed, checked.graph, proofs[0]

    def test_all_ten_supported_cases_validate_and_replay(self):
        for case in (2, 28, 33, 46, 51, 73, 82, 85, 88, 100):
            with self.subTest(case=case):
                self.fixed(case)

    def test_changed_source_is_never_silently_repaired(self):
        row = self.row(2)
        row["response_text"] = row["response_text"].replace("22:30", "20:30")
        fixed, proofs, reasons = repairs.repair_known_hold(row, 2)
        self.assertEqual(fixed, row)
        self.assertEqual(proofs, [])
        self.assertEqual(reasons, ["known_hold_source_hash_not_audited"])

    def test_changed_target_table_evidence_fails_closed(self):
        row = self.row(28)
        row["completion"] = row["completion"].replace("Clinical Grade", "Unsupported Accuracy")
        row["messages"][-1]["content"] = row["completion"]
        fixed, proofs, reasons = repairs.repair_known_hold(row, 28)
        self.assertEqual(fixed, row)
        self.assertEqual(proofs, [])
        self.assertIn("transposed_table_correspondence_not_unique", reasons[0])

    def test_mismatched_chat_binding_fails_closed(self):
        row = self.row(33)
        row["messages"][-2]["content"] += " changed"
        fixed, proofs, reasons = repairs.repair_known_hold(row, 33)
        self.assertEqual(fixed, row)
        self.assertFalse(proofs)
        self.assertIn("messages_not_bound", reasons[0])

    def test_requested_pie_is_held_and_never_substituted(self):
        row = self.row(84)
        fixed, proofs, reasons = repairs.repair_known_hold(row, 84)
        self.assertEqual(fixed, row)
        self.assertFalse(proofs)
        self.assertEqual(reasons, ["known_hold_requested_pie_chart_outside_bar_column_renderer_contract"])

    def test_ambiguous_annuity_and_source_review_holds_are_unchanged(self):
        for case in (3, 10, 15, 34, 38, 48, 65, 71, 79, 89, 96, 97):
            row = self.row(case)
            fixed, proofs, reasons = repairs.repair_known_hold(row, case)
            self.assertEqual(fixed, row)
            self.assertFalse(proofs)
            self.assertEqual(reasons, ["known_hold_requires_source_review"])

    def test_matrix_headers_and_cells_match_original_orientation(self):
        fixed, graph, _ = self.fixed(28)
        headers, cells = repairs.source_tables(fixed["response_text"])[0]
        tables = repairs.graph_tables(graph)
        self.assertEqual(len(tables), 1)
        _, table, rows = tables[0]
        columns = table["props"]["columns"]
        self.assertEqual([c["label"] for c in columns], headers)
        self.assertEqual([[row[c["key"]] for c in columns] for row in rows], cells)
        self.assertEqual(table["props"]["preferredPresentation"], "table")

    def test_source_bullet_positions_and_option_descriptors_restored(self):
        fixed, _, _ = self.fixed(33)
        for fragment in ("Signature: In the main signature block.", "Date Signed: Adjacent to the signature.",
                         "Initials: At the bottom of each page."):
            self.assertIn(fragment, fixed["completion"])
        fixed, _, _ = self.fixed(46)
        for fragment in ("Top-tier cargo and MPG", "Industry-leading reliability", "Best-in-class handling"):
            self.assertIn(fragment, fixed["completion"])

    def test_attribution_buttons_preserve_three_exact_source_bindings(self):
        from ir_training.data.archive_semantic_review import event_urls
        _, graph, _ = self.fixed(85)
        actual = {e["props"]["label"]: event_urls(e)[0]["params"]["url"]
                  for e in graph["elements"].values() if e["type"] == "Button" and event_urls(e)}
        expected = {"TradeMe Property": "[ACTION_URL_6]", "Realestate.co.nz": "[ACTION_URL_4]",
                    "QV New Zealand": "[ACTION_URL_5]"}
        self.assertEqual({label: actual[label] for label in expected}, expected)

    def test_latent_action_removed_only_for_unsupported_entity(self):
        _, graph, proof = self.fixed(88)
        rows = repairs.graph_tables(graph)[0][2]
        self.assertEqual(rows[0]["actionLabel"], "Book Hotel A")
        self.assertEqual(rows[1]["actionLabel"], "Book Hotel B")
        self.assertNotIn("actionLabel", rows[2])
        self.assertNotIn("bookingUrl", rows[2])
        self.assertEqual(len(proof["changes"]), 2)

    def test_unit_metric_rename_updates_state_and_all_column_references(self):
        fixed, graph, proof = self.fixed(51)
        self.assertNotIn("conversion", (fixed["response_text"] + fixed["completion"]).lower())
        table = next(e for _, e, _ in repairs.graph_tables(graph)
                     if any(c["label"] == "Revenue per Lead" for c in e["props"]["columns"]))
        self.assertIn("Revenue per Lead", table["props"]["numericColumns"])
        self.assertEqual(graph["state"]["performance_data"][0]["Revenue per Lead"], "$2,000")
        self.assertTrue(proof["source_corrected"])

    def test_calculated_corrections_preserve_original_numeric_inputs(self):
        fixed, _, proof = self.fixed(2)
        self.assertIn("11.1%", fixed["response_text"])
        self.assertNotIn("10.4%", fixed["completion"])
        self.assertEqual(proof["evidence"]["exact_fraction"], "100/9")
        fixed, _, proof = self.fixed(73)
        self.assertEqual(proof["evidence"]["totals"], {"Offer A": 89250, "Offer B": 92000, "Offer C": 89600})
        self.assertNotIn("Offer C has the highest potential total cash", fixed["completion"])
        fixed, _, proof = self.fixed(100)
        self.assertEqual(proof["evidence"]["interval_hours"], [30.5, 18.75])
        self.assertIn("30.5 hours", fixed["completion"])
        self.assertIn("49.25 Hours", fixed["completion"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
