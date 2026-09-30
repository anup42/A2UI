"""Source-evidence, reference-identity and replay regressions for v11 repairs."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/scripts"))
import prepare_space_v11 as prepare
import space_v11_repairs as repairs


def text_graph(text, identifier="root"):
    return {"root": identifier, "state": {}, "elements": {
        identifier: {"type": "Text", "props": {"text": text}, "children": []}}}


class RepairsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(os.environ.get("A2UI_V10_POLICY_REPO", "D:/git/anup-code/A2UI"))
        if not (repo / "training/src/ir_training/data/archive_semantic_review.py").is_file():
            raise unittest.SkipTest("Requires v10 policy repository")
        prepare.init(repo)

    def assert_clean(self, result):
        self.assertEqual(result["issues"], [])
        self.assertTrue(result["idempotent"])
        self.assertEqual(repairs.residual_gates(result["source"], result["checked"].text,
                                              result["checked"], result["url_map"]), [])
        replay = repairs.repair_chain(result["source"], result["graph"], result["url_map"])
        self.assertEqual(replay["changes"], {})
        self.assertEqual(replay["proofs"], [])
        self.assertEqual(replay["graph"], result["graph"])

    def test_type_named_element_does_not_crash_or_change(self):
        graph = text_graph("Pack your passport and charger.", "type")
        graph["root"] = "root"
        graph["elements"]["root"] = {"type": "Stack", "props": {}, "children": ["type"]}
        graph["state"]["type"] = {"type": ["harmless"]}
        result = repairs.repair_chain("Pack your passport and charger.", graph, {})
        self.assert_clean(result)
        self.assertEqual(result["graph"], graph)
        self.assertEqual(len(result["implementation"]["original_sha256"]), 64)
        self.assertNotEqual(result["implementation"]["original_sha256"], result["implementation"]["derived_sha256"])

    def test_guarded_semantic_policy_parity_without_collision(self):
        from ir_training.data.archive_semantic_review import repair_graph
        source = "Keep all delivery records until the warranty expires and retain the original invoice safely."
        graph = text_graph("Keep all delivery records and retain the original invoice safely.")
        self.assertEqual(repair_graph(source, graph, {}), repairs._semantic_function()[0](source, graph, {}))

    def test_unique_source_clause_restored_with_exact_proofs(self):
        source = "Keep all delivery records until the warranty expires and retain the original invoice safely."
        shortened = "Keep all delivery records and retain the original invoice safely."
        graph = text_graph(shortened)
        result = repairs.repair_chain(source, graph, {})
        self.assert_clean(result)
        self.assertEqual(graph["elements"]["root"]["props"]["text"], shortened)
        self.assertEqual(result["graph"]["elements"]["root"]["props"]["text"], source)
        proof = next(p for p in result["proofs"] if p["location"] == "/graph/elements/root/props/text")
        self.assertEqual(proof["before"], shortened)
        self.assertEqual(proof["after"], source)
        self.assertEqual(proof["source_sha256"], prepare.sha(source))
        self.assertIn("unique_source_clause_restoration", result["changes"])

    def test_does_not_invent_missing_prose(self):
        source = "Keep all delivery records until the warranty expires and retain the original invoice safely."
        result = repairs.repair_chain(source, text_graph("Welcome"), {})
        self.assertEqual(result["graph"], text_graph("Welcome"))
        self.assertIn("source_clause_gap", {item["code"] for item in result["issues"]})

    def test_literal_list_has_source_grounded_visible_content(self):
        source = "Pack passport, charger and tickets."
        graph = {"root": "root", "state": {}, "elements": {
            "root": {"type": "List", "props": {"items": ["Pack passport", "charger", "tickets"]}, "children": []}}}
        result = repairs.repair_chain(source, graph, {})
        self.assert_clean(result)
        self.assertIn("source_supported_literal_list_content", result["resolutions"])

    def test_literal_list_exception_does_not_allow_empty_or_invented_content(self):
        for items in ([], ["buy a spaceship"]):
            graph = {"root": "root", "state": {}, "elements": {
                "root": {"type": "List", "props": {"items": items}, "children": []}}}
            result = repairs.repair_chain("Pack passport and charger.", graph, {})
            self.assertIn("layout_only", {item["code"] for item in result["issues"]})

    def test_literal_list_source_evidence_requires_whole_word_boundaries(self):
        graph = {"root": "root", "state": {}, "elements": {
            "root": {"type": "List", "props": {"items": ["12"]}, "children": []}}}
        self.assertFalse(repairs._literal_lists_supported("There are 123 records.", graph))

    def test_mixed_reference_masking_preserves_exact_identities(self):
        from ir_training.data.url_preprocess import restore_url_placeholders
        source = "Open https://example.org/correct for details."
        graph = text_graph("Open [URL_1] for details.")
        mapping = {"[URL_1]": "https://example.org/correct"}
        normalized, mixed = repairs.normalize_for_repair(source, graph, mapping)
        self.assertTrue(mixed)
        self.assertEqual(restore_url_placeholders(normalized.response_text, normalized.url_map), source)
        self.assertEqual(restore_url_placeholders(normalized.canonical_graph, normalized.url_map),
                         restore_url_placeholders(graph, mapping))
        self.assert_clean(repairs.repair_chain(normalized.response_text, normalized.canonical_graph, normalized.url_map))

    def test_existing_closed_placeholder_map_is_retained(self):
        source = "Open [URL_1] for details."
        mapping = {"[URL_1]": {"url": "https://example.org/correct", "role": "source"}}
        normalized, mixed = repairs.normalize_for_repair(source, text_graph(source), mapping)
        self.assertFalse(mixed)
        self.assertEqual(normalized.url_map, mapping)
        self.assertEqual(normalized.response_text, source)

    def test_unsupported_extra_media_removed_without_inventing_a_reference(self):
        source = "Pack your passport and charger."
        graph = {"root": "root", "state": {}, "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["text", "image"]},
            "text": {"type": "Text", "props": {"text": source}, "children": []},
            "image": {"type": "Image", "props": {"src": "[IMAGE_URL_9]"}, "children": []}}}
        result = repairs.repair_chain(source, graph, {})
        self.assert_clean(result)
        self.assertNotIn("image", result["graph"]["elements"])
        proof = next(p for p in result["proofs"] if p["location"] == "/graph/elements/image")
        self.assertFalse(proof["after_exists"])
        self.assertTrue(proof["before_exists"])

    def test_known_distinct_media_destinations_are_never_rebound(self):
        source = "Media: Image [IMAGE_URL_1]"
        graph = {"root": "root", "state": {}, "elements": {
            "root": {"type": "Image", "props": {"src": "[IMAGE_URL_9]"}, "children": []}}}
        mapping = {"[IMAGE_URL_1]": "https://example.org/supplied.jpg",
                   "[IMAGE_URL_9]": "https://example.org/different.jpg"}
        result = repairs.repair_chain(source, graph, mapping)
        self.assertTrue(result["issues"])
        self.assertEqual(result["graph"], graph)
        self.assertNotIn("placeholder_unique_media_rebindings", result["changes"])

    def test_recovery_source_token_rebinding_still_supported_without_known_conflict(self):
        source = "Media: Image [IMAGE_URL_1]"
        graph = {"root": "root", "state": {}, "elements": {
            "root": {"type": "Image", "props": {"src": "[IMAGE_URL_9]"}, "children": []}}}
        result = repairs.repair_chain(source, graph, {})
        self.assertEqual(result["graph"]["elements"]["root"]["props"]["src"], "[IMAGE_URL_1]")
        self.assertIn("placeholder_unique_media_rebindings", result["changes"])

    def test_requested_button_label_recovered_from_exact_binding(self):
        source = "Action: [Button: Read instructions] [ACTION_URL_1]"
        graph = {"root": "root", "state": {}, "elements": {
            "root": {"type": "Button", "props": {"label": "Go"}, "children": [],
                     "on": {"press": {"action": "openUrl", "params": {"url": "[ACTION_URL_1]"}}}}}}
        result = repairs.repair_chain(source, graph, {})
        self.assert_clean(result)
        self.assertEqual(result["graph"]["elements"]["root"]["props"]["label"], "Read instructions")
        self.assertEqual(result["graph"]["elements"]["root"]["on"], graph["elements"]["root"]["on"])

    def test_json_pointer_diff_escapes_keys_and_distinguishes_absence(self):
        result = repairs._diff({"x/y": {"~": None}}, {"x/y": {"new": None}})
        self.assertEqual({item["location"] for item in result}, {"/x~1y/~0", "/x~1y/new"})
        self.assertEqual(sum(item["before_exists"] for item in result), 1)
        self.assertEqual(sum(item["after_exists"] for item in result), 1)

    def test_unique_short_literal_expansion_preserves_old_text_and_numbers(self):
        source = "The profile and targets preserved from your request:"
        graph = text_graph("Profile and targets preserved from your request:")
        result = repairs.repair_chain(source, graph, {})
        self.assert_clean(result)
        self.assertEqual(result["graph"]["elements"]["root"]["props"]["text"], source)
        self.assertIn("unique_short_source_literal_expansion", result["changes"])
        unchanged = repairs.repair_chain("The original budget is 25 dollars.", text_graph("The original budget is 50 dollars."), {})
        self.assertEqual(unchanged["graph"], text_graph("The original budget is 50 dollars."))

    def test_short_expansion_refuses_ambiguous_source_lines(self):
        graph = text_graph("keep the box in storage")
        source = "Please keep the box in storage\nNever keep the box in storage"
        fixed, proofs = repairs._restore_small_literal_expansions(source, graph)
        self.assertEqual(fixed, graph)
        self.assertEqual(proofs, [])

    def test_omitted_markdown_enumeration_is_not_a_content_repair(self):
        graph = text_graph("Keep your records in storage.")
        fixed, proofs = repairs._restore_small_literal_expansions("1. Keep your records in storage.", graph)
        self.assertEqual(fixed, graph)
        self.assertEqual(proofs, [])

    def test_short_expansion_does_not_duplicate_heading_separated_by_icon(self):
        body = "Perform 800m or 1km repeats at a 4:20-4:25 pace to build anaerobic capacity."
        graph = {"root": "root", "state": {}, "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["heading", "icon", "body"]},
            "heading": {"type": "Text", "props": {"text": "Interval Training", "variant": "h3"}, "children": []},
            "icon": {"type": "Icon", "props": {"name": "running"}, "children": []},
            "body": {"type": "Text", "props": {"text": body}, "children": []}}}
        fixed, proofs = repairs._restore_small_literal_expansions("- **Interval Training**: " + body, graph)
        self.assertEqual(fixed, graph)
        self.assertEqual(proofs, [])

    def test_unrelated_qualifier_elsewhere_does_not_suppress_literal_restoration(self):
        graph = {"root": "root", "state": {}, "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["other", "group"]},
            "other": {"type": "Text", "props": {"text": "Only"}, "children": []},
            "group": {"type": "Card", "props": {}, "children": ["body"]},
            "body": {"type": "Text", "props": {"text": "Approve delivery after the checks."}, "children": []}}}
        fixed, proofs = repairs._restore_small_literal_expansions("Only approve delivery after the checks.", graph)
        self.assertEqual(fixed["elements"]["body"]["props"]["text"], "Only approve delivery after the checks.")
        self.assertEqual(len(proofs), 1)

    def test_chart_declaration_is_not_promoted_or_added_to_existing_title(self):
        title = "Electricity Usage and Cost Trend Jan-Mar 2026"
        graph = text_graph(title)
        fixed, proofs = repairs._restore_small_literal_expansions("Chart: " + title, graph)
        self.assertEqual(fixed, graph)
        self.assertEqual(proofs, [])
        graph = self.gap_graph("Monthly cash flow", "X-axis: Month\nY-axis: USD")
        source = "## Monthly cash flow\nChart: " + title + "\nX-axis: Month\nY-axis: USD"
        fixed, proofs = repairs._restore_anchored_prose_gaps(source, graph)
        self.assertEqual(fixed, graph)
        self.assertEqual(proofs, [])

    def test_short_expansion_preserves_adjacent_list_heading_and_button_labels(self):
        for label, body, left_kind, right_kind in (
            ("Pros", "No travel and keep your remote flexibility", "Text", "List"),
            ("Verify", "confirms photo match and active ID status", "Button", "Text"),
        ):
            with self.subTest(label=label):
                graph = {"root": "root", "state": {}, "elements": {
                    "root": {"type": "Stack", "props": {}, "children": ["label", "body"]},
                    "label": {"type": left_kind, "props": {"label" if left_kind == "Button" else "text": label}, "children": []},
                    "body": {"type": right_kind, "props": {"items": [body]} if right_kind == "List" else {"text": body}, "children": []}}}
                fixed, proofs = repairs._restore_small_literal_expansions(label + ": " + body, graph)
                self.assertEqual(fixed, graph)
                self.assertEqual(proofs, [])
    def gap_graph(self, left, right, left_kind="Text", right_kind="Text"):
        def element(value, kind):
            key = "code" if kind == "CodeBlock" else "text"
            props = {key: value}
            if kind == "Text": props["variant"] = "h3"
            return {"type": kind, "props": props, "children": []}
        return {"root": "root", "state": {}, "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["left", "right"]},
            "left": element(left, left_kind), "right": element(right, right_kind)}}

    def test_prose_between_two_exact_unique_adjacent_anchors_gets_small_text(self):
        source = "### Recovery verification\nAfter any approved action:\n```bash\ncat /proc/mdstat\n```"
        graph = self.gap_graph("Recovery verification", "cat /proc/mdstat", right_kind="CodeBlock")
        result = repairs.repair_chain(source, graph, {})
        self.assert_clean(result)
        children = result["graph"]["elements"]["root"]["children"]
        self.assertEqual((children[0], children[-1]), ("left", "right"))
        self.assertEqual(len(children), 3)
        self.assertEqual(result["graph"]["elements"][children[1]]["props"]["text"], "After any approved action:")
        self.assertIn("adjacent_unique_source_prose_restoration", result["changes"])
        self.assertTrue(any(p["policy_proofs"][0]["left_source_span"] for p in result["proofs"] if p["stage"] == "adjacent_unique_source_prose_restoration"))

    def test_prose_gap_extends_existing_body_text_when_possible(self):
        source = "Use original equipment.\n\nFor trail use with articulation:\n* Check vehicle clearance.\n* Keep sufficient margin."
        graph = self.gap_graph("Use original equipment.", "Check vehicle clearance.")
        graph["elements"]["left"]["props"].pop("variant")
        graph["elements"]["right"] = {"type": "List", "props": {"items": ["Check vehicle clearance.", "Keep sufficient margin."]}, "children": []}
        result = repairs.repair_chain(source, graph, {})
        self.assert_clean(result)
        self.assertEqual(len(result["graph"]["elements"]), 3)
        self.assertEqual(result["graph"]["elements"]["left"]["props"]["text"], "Use original equipment.\n\nFor trail use with articulation:")

    def test_gap_after_code_block_is_literal_and_ordered(self):
        source = "```\nwsl --shutdown\n```\nThen start containers again.\n\nWindows event logs"
        graph = self.gap_graph("wsl --shutdown", "Windows event logs", left_kind="CodeBlock")
        result = repairs.repair_chain(source, graph, {})
        self.assert_clean(result)
        child = result["graph"]["elements"]["root"]["children"][1]
        self.assertEqual(result["graph"]["elements"][child]["props"]["text"], "Then start containers again.")

    def test_gap_rule_refuses_ambiguous_anchors_multiline_declarations_and_hidden_nodes(self):
        graph = self.gap_graph("First section", "Next section")
        for middle in ("Action: [Button: Go] [URL_1]", "* A new bullet item", "A table | has cells", "First prose line.\nSecond prose line.", "### Missing heading", "A result = a formula", "First section"):
            source = "First section\n" + middle + "\nNext section"
            fixed, proof = repairs._restore_anchored_prose_gaps(source, graph)
            self.assertEqual(fixed, graph, middle)
            self.assertEqual(proof, [], middle)
        graph["elements"]["left"]["visible"] = False
        fixed, proof = repairs._restore_anchored_prose_gaps("First section\nSmall missing note.\nNext section", graph)
        self.assertEqual(fixed, graph)
        self.assertEqual(proof, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
