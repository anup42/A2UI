"""Offline source-only Stage 3 admission and resumable archive checks."""

import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset/scripts"))

from chatgpt_fresh_labels import label_shard
from test_chatgpt_stage3 import FakeCodex
from test_openai_stage3_batch import GOOD, SOURCE


@pytest.mark.parametrize(("label", "expected"), [(GOOD, 1), (GOOD.replace("$120", "$999"), 0)])
def test_fresh_stage3_uses_real_labels_and_preserves_strict_holds(tmp_path, label, expected):
    rows = [{"id": "new-training-source", "response_text": SOURCE,
             "metadata": {"assigned_split": "train", "query_text": "Compare Alder and Birch."}}]
    client = FakeCodex(text=label)
    output = tmp_path / "fresh"
    report = label_shard(rows, output, client, model="gpt-6.1-sol", workers=2)
    assert (report["accepted"], report["held"]) == (expected, 1 - expected)
    assert client.calls == 1
    assert not (output / "work").exists()
    with tarfile.open(report["archive"], "r:gz") as archive:
        assert json.load(archive.extractfile("input/sources.jsonl")) == rows[0]
        assessment = json.load(archive.extractfile("p/assessments.jsonl"))
        assert assessment["eligible"] is bool(expected)
        records = [member for member in archive if member.name.startswith("p/chatgpt_records/")]
        assert len(records) == 1
        assert json.load(archive.extractfile(records[0]))["payload"]["text"] == label
        training = archive.extractfile("p/pilot_accepted.jsonl").read()
        if expected:
            example = json.loads(training)
            assert example["id"] == rows[0]["id"]
            assert example["response_text"] == SOURCE
            assert example["messages"][-1]["content"] == example["a2ui_express"]
            assert "$120" in example["a2ui_express"]
        else:
            assert training == b""
    assert label_shard(rows, output, client, model="gpt-6.1-sol", workers=2) == report
    with pytest.raises(ValueError, match="changed"):
        label_shard([{**rows[0], "response_text": "Changed."}], output, client, model="gpt-6.1-sol")
    assert client.calls == 1


def test_unbound_source_is_archived_without_dispatch(tmp_path):
    client = FakeCodex()
    row = {"id": "fresh-unbound", "response_text": "Register: [URL_1]",
           "metadata": {"query": {"query_text": "Show where to register."}}}
    report = label_shard([row], tmp_path / "fresh", client)
    assert report["dispatched"] == report["accepted"] == client.calls == 0
    assert report["source_held"] == report["held"] == 1
    with tarfile.open(report["archive"], "r:gz") as archive:
        hold = json.load(archive.extractfile("input/source_holds.jsonl"))
        assert any("unresolved_source_reference" in reason for reason in hold["reasons"])
