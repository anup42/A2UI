"""Offline integration of synthetic sources, separate review and Stage 3 gates."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from llm.base import LLMResult
from llm.factory import load_model_specs
from pipeline.training_augmentation import (
    CATEGORIES,
    DEFAULT_TEACHER,
    bounded_teacher_environment,
    generate_training_augmentations,
    load_donors,
    read_jsonl,
    validate_review,
)
from utils.config import load_yaml


def teacher_spec():
    return next(spec for spec in load_model_specs(load_yaml(DATASET_ROOT / "configs/models.yaml")) if spec.name == DEFAULT_TEACHER)


def approved_review(category):
    return {"category": category, "approved": True, "coherent": True,
            "category_satisfied": True, "synthetic_provenance_clear": True, "issues": []}


class FakeTeacher:
    def __init__(self, review_override=None, source_override=None):
        self.spec = teacher_spec()
        self.calls = []
        self.review_override = review_override
        self.source_override = source_override

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["prompt"]
        context = json.loads(prompt.split("\n")[1])
        category = context["category"]
        if prompt.startswith("TASK: GENERATE_SOURCE"):
            value = {"category": category, "synthetic": True,
                     "response_text": "Synthetic changed source: " + category + " " + str(kwargs["seed"])}
            if self.source_override:
                value = self.source_override(value, context)
        else:
            value = approved_review(category)
            if self.review_override:
                value = self.review_override(value)
        return LLMResult(text=json.dumps(value), raw={}, latency_ms=1, input_tokens=20, output_tokens=20,
                         cost_usd=None, model=self.spec.model, provider=self.spec.provider,
                         finish_reason="stop", completion_complete=True)


def stage3_row(response):
    return {"response_id": response["response_id"], "query_id": response["query_id"],
            "response_text": response["response_text"], "source_format": "a2ui_express_v1",
            "record_status": "accepted", "completion": "<a2ui>NEW_TEACHER_GENERATED_TARGET</a2ui>",
            "a2ui_express": "<a2ui>NEW_TEACHER_GENERATED_TARGET</a2ui>",
            "canonical_graph": {"root": "new_root", "state": {}, "elements": {}},
            "training_acceptance": {"eligible": True, "blocking_reasons": [], "review_reasons": []},
            "validation": {"schema_valid_strict": True, "standard_a2ui_valid": True},
            "gen": {"completion_complete": True, "error": None}}


class AugmentationGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.donors_path = self.root / "donors.jsonl"
        self.output = self.root / "out"

    def donors(self, count=9, mutate=None):
        rows = [{"donor_id": f"train/id:{index}", "split": "train", "source_group_id": f"family-{index}",
                 "response_text": f"Original source {index}", "completion": "FORBIDDEN_DONOR_TARGET"} for index in range(count)]
        if mutate:
            mutate(rows)
        self.donors_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return rows

    def stage3(self, mutate=None, record_calls=None):
        def runner(**kwargs):
            if record_calls is not None:
                record_calls.append(kwargs)
            rows = [stage3_row(response) for response in read_jsonl(kwargs["responses_path"])]
            if mutate:
                mutate(rows)
            kwargs["genui_path"].write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return runner

    def run_generator(self, **kwargs):
        return generate_training_augmentations(self.donors_path, self.output,
            adapter=kwargs.pop("adapter", FakeTeacher()), stage3_runner=kwargs.pop("stage3_runner", self.stage3()), **kwargs)

    def test_all_categories_new_sources_new_labels_and_manifest_provenance(self):
        donors = self.donors()
        calls = []
        teacher = FakeTeacher()
        manifest = self.run_generator(adapter=teacher, stage3_runner=self.stage3(record_calls=calls), max_new_samples=9, seed=42)
        rows = read_jsonl(self.output / "accepted_genui.jsonl")
        self.assertEqual([row["augmentation"]["category"] for row in rows], list(CATEGORIES))
        self.assertEqual(len(teacher.calls), 18)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["metric_version"], "v5_4")
        self.assertEqual(calls[0]["ir_formats"], ["a2ui_express_v1"])
        self.assertEqual(manifest["attempted_rows"], 9)
        self.assertEqual(manifest["accepted_rows"], 9)
        for category in CATEGORIES:
            self.assertEqual(manifest["coverage"][category], {"attempted": 1, "source_approved": 1, "accepted": 1})
        by_id = {donor["donor_id"]: donor for donor in donors}
        for row in rows:
            provenance = row["augmentation"]
            donor = by_id[provenance["donor_id"]]
            self.assertEqual(provenance["source_group_id"], donor["source_group_id"])
            self.assertEqual(provenance["teacher_model"], DEFAULT_TEACHER)
            self.assertEqual(provenance["seed"], 42)
            self.assertTrue(provenance["synthetic"])
            self.assertEqual(provenance["source_sha256"], hashlib.sha256(row["response_text"].encode()).hexdigest())
            self.assertNotEqual(row["response_text"], donor["response_text"])
            self.assertNotEqual(row["completion"], donor["completion"])
        self.assertNotIn("FORBIDDEN_DONOR_TARGET", json.dumps(teacher.calls))
        self.assertEqual(manifest["donors_sha256"], hashlib.sha256(self.donors_path.read_bytes()).hexdigest())
        self.assertEqual(manifest["accepted_genui_sha256"], hashlib.sha256((self.output / "accepted_genui.jsonl").read_bytes()).hexdigest())
        self.assertTrue(manifest["code_hashes"])
        self.assertEqual(len(read_jsonl(self.output / "augmentation_audit.jsonl")), 18)

    def test_attempts_cap_and_one_variant_per_donor(self):
        self.donors(2)
        manifest = self.run_generator(max_new_samples=90)
        self.assertEqual(manifest["attempted_rows"], 2)

    def test_deterministic_schedule(self):
        self.donors()
        self.run_generator(max_new_samples=9, seed=47)
        first = read_jsonl(self.output / "responses.jsonl")
        self.output = self.root / "second"
        self.run_generator(max_new_samples=9, seed=47)
        self.assertEqual(first, read_jsonl(self.output / "responses.jsonl"))

    def test_strict_stage3_rejections_even_if_status_accepted(self):
        self.donors()
        def mutate(rows):
            rows[0]["training_acceptance"]["eligible"] = False
            rows[1]["training_acceptance"]["eligible"] = False
            rows[1]["training_acceptance"]["review_reasons"] = ["dynamic_evidence_incomplete"]
            rows[2]["training_acceptance"]["blocking_reasons"] = ["production_invalid"]
            rows[3]["validation"]["schema_valid_strict"] = False
            rows[4]["validation"]["standard_a2ui_valid"] = False
            rows[5]["gen"]["completion_complete"] = False
            rows[6]["gen"]["error"] = "server failure"
            rows[7]["response_text"] = "wrong source binding"
        manifest = self.run_generator(max_new_samples=9, stage3_runner=self.stage3(mutate))
        self.assertEqual(manifest["accepted_rows"], 1)
        rejected = [row for row in read_jsonl(self.output / "augmentation_audit.jsonl") if row["status"] == "stage3_rejected"]
        self.assertEqual(len(rejected), 8)

    def test_missing_duplicate_unknown_stage3_ids_are_not_admitted(self):
        self.donors(3)
        def mutate(rows):
            rows.pop(0)
            rows.append(dict(rows[0]))
            rows.append(dict(rows[-1], response_id="unknown"))
        manifest = self.run_generator(max_new_samples=3, stage3_runner=self.stage3(mutate))
        self.assertEqual(manifest["accepted_rows"], 1)
        audits = read_jsonl(self.output / "augmentation_audit.jsonl")
        self.assertTrue(any("unknown_source_binding" in row.get("reasons", []) for row in audits))

    def test_zero_accepted_keeps_diagnostics_and_fails(self):
        self.donors(1)
        with self.assertRaisesRegex(RuntimeError, "No eligible"):
            self.run_generator(adapter=FakeTeacher(review_override=lambda value: dict(value, coherent=False)))
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["attempted_rows"], 1)
        self.assertEqual((self.output / "accepted_genui.jsonl").read_text(), "")
        self.assertEqual(read_jsonl(self.output / "augmentation_audit.jsonl")[0]["status"], "source_rejected")

    def test_source_review_malformed_verdicts_fail_closed(self):
        good = approved_review(CATEGORIES[0])
        variants = [{}, dict(good, approved="true"), dict(good, issues=None), dict(good, category="wrong"),
                    dict(good, category_satisfied=False), dict(good, synthetic_provenance_clear=False),
                    dict(good, issues=["inconsistent counts"])]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate_review(variant, CATEGORIES[0])

    def test_non_object_review_rejected_before_stage3(self):
        self.donors(1)
        calls = []
        with self.assertRaises(RuntimeError):
            self.run_generator(adapter=FakeTeacher(review_override=lambda _: ["yes"]), stage3_runner=self.stage3(record_calls=calls))
        self.assertEqual(calls, [])

    def test_unchanged_source_rejected_without_review_or_stage3(self):
        self.donors(1)
        teacher = FakeTeacher(source_override=lambda value, context: dict(value, response_text=context["donor_response_text"]))
        with self.assertRaises(RuntimeError):
            self.run_generator(adapter=teacher)
        self.assertEqual(len(teacher.calls), 1)

    def test_donor_validation_nontraining_duplicates_and_malformed(self):
        mutations = [lambda rows: rows[0].update(split="validation"),
                     lambda rows: rows[0].pop("source_group_id"),
                     lambda rows: rows[0].update(response_text=" "),
                     lambda rows: rows[1].update(donor_id=rows[0]["donor_id"]),
                     lambda rows: rows[1].update(response_text=rows[0]["response_text"])]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.donors(2, mutate)
                with self.assertRaises(ValueError):
                    load_donors(self.donors_path)
        self.donors_path.write_text("{broken\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Malformed JSONL"):
            load_donors(self.donors_path)

    def test_bad_teacher_and_nonempty_output_fail_before_calls(self):
        self.donors(1)
        teacher = FakeTeacher()
        with self.assertRaisesRegex(ValueError, "registered teacher"):
            self.run_generator(adapter=teacher, teacher_model="unregistered")
        self.output.mkdir()
        (self.output / "existing").touch()
        with self.assertRaisesRegex(ValueError, "empty"):
            self.run_generator(adapter=teacher)
        self.assertEqual(teacher.calls, [])

    def test_timeout_and_retry_environment_bounded_and_restored(self):
        with patch.dict(os.environ, {"LOCAL_VLLM_TIMEOUT_SECONDS": "999999", "LOCAL_VLLM_RETRY_CONNECTION_ERRORS": "1"}):
            with bounded_teacher_environment():
                self.assertEqual(os.environ["LOCAL_VLLM_TIMEOUT_SECONDS"], "180")
                self.assertEqual(os.environ["LOCAL_VLLM_RETRY_CONNECTION_ERRORS"], "0")
                self.assertEqual(os.environ["LOCAL_VLLM_RETRY_RESULT_ERRORS"], "0")
            self.assertEqual(os.environ["LOCAL_VLLM_TIMEOUT_SECONDS"], "999999")

    def test_stage3_crash_fails_with_audit_and_manifest(self):
        self.donors(1)
        def crash(**_kwargs):
            raise RuntimeError("mock stage3 crash")
        with self.assertRaisesRegex(RuntimeError, "mock stage3 crash"):
            self.run_generator(stage3_runner=crash)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")
        self.assertEqual(read_jsonl(self.output / "augmentation_audit.jsonl")[-1]["status"], "stage3_rejected")

    def test_incomplete_generation_and_malformed_review_never_reach_stage3(self):
        self.donors(1)
        for failure in ("source_incomplete", "review_incomplete", "review_json"):
            with self.subTest(failure=failure):
                self.output = self.root / failure
                class BadTeacher(FakeTeacher):
                    def __init__(inner, failure_kind):
                        super().__init__()
                        inner.failure_kind = failure_kind

                    def generate(inner, **kwargs):
                        result = super(BadTeacher, inner).generate(**kwargs)
                        is_review = kwargs["prompt"].startswith("TASK: REVIEW_SOURCE")
                        if inner.failure_kind == "source_incomplete" and not is_review or inner.failure_kind == "review_incomplete" and is_review:
                            result.completion_complete = False
                        if inner.failure_kind == "review_json" and is_review:
                            result.text = "Not JSON"
                        return result
                stage3_calls = []
                with self.assertRaises(RuntimeError):
                    self.run_generator(adapter=BadTeacher(failure), stage3_runner=self.stage3(record_calls=stage3_calls))
                self.assertEqual(stage3_calls, [])

    def test_malformed_stage3_file_audits_every_source_rejection(self):
        self.donors(2)
        def malformed(**kwargs):
            kwargs["genui_path"].write_text("{broken\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Malformed JSONL"):
            self.run_generator(stage3_runner=malformed)
        audits = read_jsonl(self.output / "augmentation_audit.jsonl")
        self.assertEqual(len([row for row in audits if row["status"] == "stage3_rejected"]), 2)

    def test_real_stage3_produces_current_express_record_contract(self):
        from pipeline.stage3_genui import run_stage3
        self.donors(1)
        source = "Synthetic example: blue lantern."
        class RealStage3Teacher(FakeTeacher):
            def generate(inner, **kwargs):
                if kwargs["prompt"].startswith("TASK:"):
                    return super(RealStage3Teacher, inner).generate(**kwargs)
                completion = '<a2ui>\nroot=Text(' + json.dumps(source) + ')\n</a2ui>'
                raw = {"choices": [{"finish_reason": "stop", "message": {"content": completion, "reasoning_content": "checked synthetic rendering"}}]}
                return LLMResult(text=completion, raw=raw, latency_ms=1, input_tokens=20, output_tokens=30,
                    cost_usd=None, model=inner.spec.model, provider=inner.spec.provider,
                    completion_complete=True, finish_reason="stop", reasoning_text="checked synthetic rendering",
                    reasoning_source="message.reasoning_content")
        teacher = RealStage3Teacher(source_override=lambda value, _: dict(value, response_text=source))
        manifest = self.run_generator(adapter=teacher, stage3_runner=run_stage3, max_new_samples=1)
        row = read_jsonl(self.output / "accepted_genui.jsonl")[0]
        self.assertEqual(manifest["accepted_rows"], 1)
        self.assertEqual(row["a2ui_express"], row["completion"])
        self.assertEqual(row["response_text"], source)
        self.assertTrue(row["gen"]["completion_complete"])
        self.assertTrue(row["training_acceptance"]["eligible"])
        self.assertTrue(row["validation"]["standard_a2ui_valid"])
        self.assertIsInstance(row["compiled_a2ui"], list)
        self.assertNotIn("genui_json", row)


if __name__ == "__main__":
    unittest.main()
