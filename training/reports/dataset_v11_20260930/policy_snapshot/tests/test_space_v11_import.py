"""Adversarial tests for archive safety and the new Stage 3 admission boundary."""
from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"training/scripts"))
import project_space_archive as project
import prepare_space_v11 as prepare


class ArchiveTests(unittest.TestCase):
    def make_tar(self,path,members):
        with tarfile.open(path,"w:gz") as tar:
            for name,data,kind in members:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.size = len(data) if kind == tarfile.REGTYPE else 0
                if kind == tarfile.SYMTYPE: info.linkname = "elsewhere"
                tar.addfile(info,io.BytesIO(data) if kind == tarfile.REGTYPE else None)

    def test_historical_targets_are_not_extra_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/"data.tar.gz"
            row={"completion":"exact target", "response_text":"exact source", "record_status":"accepted", "training_acceptance":{"eligible":True}}
            raw=(json.dumps(row)+"\n").encode()
            self.make_tar(archive,[("dataset_muse_glimmer_100k_r1/genui.jsonl",raw,tarfile.REGTYPE),("dataset_muse_glimmer_100k_r1/old/genui.jsonl",raw,tarfile.REGTYPE)])
            project.project(archive,root/"out")
            manifest=json.loads((root/"out/archive_manifest.json").read_text())
            self.assertEqual(len(manifest["selected"]),1)
            projected=json.loads((root/"out/dataset_muse_glimmer_100k_r1/genui.training.jsonl").read_text())
            self.assertEqual(projected["completion"],row["completion"])
            self.assertEqual(projected["archive_record"]["sha256"],hashlib.sha256(raw).hexdigest())

    def test_traversal_and_selected_symlink_fail_closed(self):
        for name,kind in [("dataset_muse_glimmer_100k_r1/../../outside",tarfile.REGTYPE),("dataset_muse_glimmer_100k_r1/genui.jsonl",tarfile.SYMTYPE)]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);archive=root/"data.tar.gz"
                self.make_tar(archive,[(name,b"{}\n",kind)])
                with self.assertRaises(ValueError): project.project(archive,root/"out")
                self.assertFalse((root/"out").exists())

    def test_corrupt_gzip_crc_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/"data.tar.gz"
            self.make_tar(archive,[("dataset_muse_glimmer_100k_r1/queries.jsonl",b"{}\n",tarfile.REGTYPE)])
            data=bytearray(archive.read_bytes());data[-8]^=1;archive.write_bytes(data)
            with self.assertRaises((OSError,tarfile.ReadError)): project.project(archive,root/"out")
            self.assertFalse((root/"out").exists())


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo=Path(os.environ.get("A2UI_V10_POLICY_REPO","D:/git/anup-code/A2UI"))
        if not (repo/"training/src/ir_training/data/archive_semantic_review.py").is_file():
            raise unittest.SkipTest("Requires explicit v10 policy repository")
        prepare.init(repo,{"scaffold":{"messages":[{"role":"system","content":"fixture"}],"shared_prompt":{}}})

    def row(self):
        from ir_training.data.express_preparation import serialize_checked
        source="Pack your passport and charger."
        target='<a2ui>\nroot=Text("Pack your passport and charger.")\n</a2ui>'
        checked=serialize_checked(target,"root-first")
        return {"ui_id":"u1","response_id":"r1","query_id":"q1","response_text":source,"stage2_source":source,
                "completion":target,"a2ui_express":target,"canonical_graph":checked.graph,"canonical_graph_hash":checked.semantic_sha256,
                "semantic_hash":checked.semantic_sha256,"reference_source_sha256":prepare.sha(source),"reference_map":{},
                "source_format":"a2ui_express_v1","target_format":"a2ui_express_v1","record_status":"accepted","training_acceptance":{"eligible":True},
                "archive_record":{"member":"dataset_muse_glimmer_100k_r1/genui.jsonl","line":1,"sha256":"fixture"},
                "query":{"query_text":"What should I pack?"},"source_quality":{},"gen":{"completion_complete":True},"join_errors":[]}

    def test_accepts_unchanged_grounded_target(self):
        row=self.row();output,event,_=prepare.new_item(("dataset_muse_glimmer_100k_r1",row))
        self.assertIsNotNone(output,event)
        self.assertEqual(output["response_text"],row["response_text"])
        from ir_training.data.express_preparation import serialize_checked
        self.assertEqual(serialize_checked(output["completion"],"root-first").graph,row["canonical_graph"])
        self.assertFalse(output["repair"]["applied"])

    def test_accepted_status_does_not_override_ineligible_flag(self):
        row=self.row();row["training_acceptance"]={"eligible":False,"review_reasons":["fidelity"]}
        output,event,_=prepare.new_item(("run",row))
        self.assertIsNone(output)
        self.assertIn("generator_training_ineligible",event["reasons"])

    def test_stage2_mismatch_rejected(self):
        row=self.row();row["stage2_source"]="Different source."
        output,event,_=prepare.new_item(("run",row))
        self.assertIsNone(output)
        self.assertIn("stage2_source_mismatch",event["reasons"])

    def test_source_exclusion_overrides_target_eligibility(self):
        row=self.row();row["source_quality"]={"training_eligibility":"exclude"}
        output,event,_=prepare.new_item(("run",row))
        self.assertIsNone(output)
        self.assertIn("source_quality_blocked",event["reasons"])

    def test_invented_reference_rejected(self):
        row=self.row();row["completion"]=row["a2ui_express"]='<a2ui>\nroot=Image("[IMAGE_URL_1]")\n</a2ui>'
        output,event,_=prepare.new_item(("run",row))
        self.assertIsNone(output)
        self.assertIn("target_reference_absent_from_source",event["reasons"])

    def test_canonical_graph_content_disagreement_rejected(self):
        row=self.row();row["canonical_graph"]=copy.deepcopy(row["canonical_graph"])
        row["canonical_graph"]["elements"]["root"]["props"]["text"]="Different target."
        output,event,_=prepare.new_item(("run",row))
        self.assertIsNone(output)
        self.assertIn("canonical_graph_mismatch",event["reasons"])

    def test_mixed_raw_source_and_masked_target_preserves_reference_identity(self):
        from ir_training.data.url_preprocess import restore_url_placeholders
        graph={"root":"a","state":{},"elements":{"a":{"type":"Button","props":{"label":"Read"},"on":{"press":{"action":"openUrl","params":{"url":"[URL_1]"}}}}}}
        source='Action: [Button: Read] https://example.org/resource'
        mapping={'[URL_1]':'https://example.org/resource'}
        normalized,mixed=prepare.normalize_references(source,graph,mapping)
        self.assertTrue(mixed)
        self.assertEqual(restore_url_placeholders(normalized.response_text,normalized.url_map),source)
        self.assertEqual(restore_url_placeholders(normalized.canonical_graph,normalized.url_map),restore_url_placeholders(graph,mapping))

    def test_mapped_but_invented_url_is_still_rejected(self):
        graph={"root":"a","state":{},"elements":{"a":{"type":"Image","props":{"src":"[URL_1]"}}}}
        with self.assertRaisesRegex(ValueError,'target_reference_absent_from_source'):
            prepare.normalize_references('No image was supplied.',graph,{'[URL_1]':'https://invented.example/image.jpg'})


if __name__ == "__main__":
    unittest.main(verbosity=2)
