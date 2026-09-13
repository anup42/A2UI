"""Small CPU checks for the read-only September data-audit helpers."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import zlib

import pytest


@pytest.fixture
def audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts/audits/full_data_tokens_20260913.py"
    spec = importlib.util.spec_from_file_location("full_data_token_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_histogram_percentiles_and_strict_thresholds(audit_module):
    from collections import Counter
    result = audit_module.distribution(Counter({10: 1, 2048: 2, 2049: 1}))
    assert result["p50"] == 2048
    assert result["p99"] == 2049
    assert result["over"]["2048"] == 1


def test_canonical_join_binds_coordinates_and_uses_changed_target(audit_module, tmp_path):
    m = audit_module
    raw_csv = tmp_path / "token_lengths.csv"
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("fixture", encoding="utf-8")
    train_raw, val_raw, val_canonical = "training", "old target", "new target"
    fields = ["split", "line", "target_sha256", "target_tokens", "completion_tokens", "new_reconstructed_prompt_tokens"]
    with raw_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, raw in (("train", train_raw), ("val", val_raw)):
            writer.writerow(dict(zip(fields, [split, 1, m.sha(raw), len(raw), len(raw)+2, 100])))
    summary = {"row_lengths_sha256": m.file_sha(raw_csv), "tokenizer": {"sha256": m.file_sha(tokenizer_path)},
               "lengths": {s: {"target_tokens": {"count": 1}} for s in ("train", "val")}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    db_path = tmp_path / "ir.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript("CREATE TABLE rows(split,line,target_sha,binding_error); CREATE TABLE targets(sha,valid,features,canonical_zlib);")
    for split, raw, canonical in (("train", train_raw, train_raw), ("val", val_raw, val_canonical)):
        connection.execute("INSERT INTO rows VALUES(?,?,?,?)", (split, 1, m.sha(raw), "[]" if split == "train" else '["fixture_binding_error"]'))
        connection.execute("INSERT INTO targets VALUES(?,?,?,?)", (m.sha(raw), 1,
                           json.dumps({"canonical_equals_raw": raw == canonical, "canonical_sha256": m.sha(canonical)}),
                           zlib.compress(canonical.encode("utf-8"))))
    connection.commit()
    connection.close()
    args = SimpleNamespace(output_dir=tmp_path, rows_output=raw_csv, tokenizer=tokenizer_path, canonical_db=db_path)
    fake_tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: SimpleNamespace(ids=list(text)))
    m.canonical_audit(args, fake_tokenizer)
    result = json.loads((tmp_path / "canonical_summary.json").read_text(encoding="utf-8"))
    assert result["counts"]["train"]["canonical_equals_raw"] == 1
    assert result["counts"]["train"].get("binding_error", 0) == 0
    assert result["counts"]["val"]["canonical_differs_raw"] == 1
    assert result["counts"]["val"]["binding_error"] == 1
    assert result["lengths"]["val"]["canonical_target_tokens"]["max"] == len(val_canonical)
    connection = sqlite3.connect(db_path)
    connection.execute("DELETE FROM targets WHERE sha=?", (m.sha(val_raw),))
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="Strict audit incomplete"):
        m.canonical_audit(args, fake_tokenizer)
