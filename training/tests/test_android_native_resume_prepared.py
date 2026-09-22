from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.eval.android_native_quality import (
    _validated_prepared_root,
    _verified_receipt_files,
    file_sha256,
)
from ir_training.train.mobile_resume import POLICY


def _continuation(tmp_path: Path):
    run = (tmp_path / "new-run").resolve()
    run.mkdir()
    prepared = (tmp_path / "source-run/prepared").resolve()
    prepared.mkdir(parents=True)
    source_config = (
        tmp_path / "source-run/launch/resolved_training_config.yaml"
    ).resolve()
    source_config.parent.mkdir(parents=True)
    source_config.write_text("source: true\n")
    report = source_config.parent / "preparation_report.json"
    report.write_text("{}")
    plan = {
        "options": {"output_dir": str(run)},
        "paths": {"prepared": str(prepared)},
        "resume": {"prepared": str(prepared), "source_config": str(source_config)},
    }
    config = {
        "run": {"dataset_dir": str(prepared)},
        "training": {"resume_policy": POLICY},
    }
    return run, prepared, source_config, report, plan, config


def test_continuation_accepts_only_exact_external_prepared_and_receipt_bindings(
    tmp_path,
):
    run, prepared, source_config, report, plan, config = _continuation(tmp_path)
    split = prepared / "train.jsonl"
    split.write_text('{"id":1}\n')
    audit = run / "data_audit.json"
    audit.write_text("{}")
    receipt = {
        "files": {
            str(audit): file_sha256(audit),
            str(split): file_sha256(split),
            str(source_config): file_sha256(source_config),
            str(report): file_sha256(report),
        }
    }

    root, exact_files = _validated_prepared_root(
        plan=plan, config=config, run=run, original_root=str(run)
    )
    assert root == prepared
    verified = _verified_receipt_files(
        receipt,
        original_root=str(run),
        current_root=run,
        required_prefix=root,
        allowed_external_roots=(root,),
        allowed_external_files=exact_files,
    )
    assert verified == {
        str(path): file_sha256(path) for path in (split, source_config, report)
    }


@pytest.mark.parametrize(
    "tamper", ["source_config", "preparation_report", "missing_prepared"]
)
def test_continuation_hash_checks_external_evidence_and_requires_prepared_files(
    tmp_path, tamper
):
    run, prepared, source_config, report, _, _ = _continuation(tmp_path)
    split = prepared / "train.jsonl"
    split.write_text("{}\n")
    receipt = {
        "files": {
            str(path): file_sha256(path) for path in (split, source_config, report)
        }
    }
    if tamper == "missing_prepared":
        receipt["files"].pop(str(split))
    else:
        path = source_config if tamper == "source_config" else report
        path.write_text("changed")
    with pytest.raises(ValueError, match="changed|binds no files"):
        _verified_receipt_files(
            receipt,
            original_root=str(run),
            current_root=run,
            required_prefix=prepared,
            allowed_external_roots=(prepared,),
            allowed_external_files=(source_config, report),
        )


@pytest.mark.parametrize("field", ["plan", "resume", "config"])
def test_continuation_rejects_disagreeing_prepared_paths(tmp_path, field):
    run, _, _, _, plan, config = _continuation(tmp_path)
    wrong = str((tmp_path / "other-prepared").resolve())
    if field == "plan":
        plan["paths"]["prepared"] = wrong
    elif field == "resume":
        plan["resume"]["prepared"] = wrong
    else:
        config["run"]["dataset_dir"] = wrong
    with pytest.raises(ValueError, match="paths disagree"):
        _validated_prepared_root(
            plan=plan, config=config, run=run, original_root=str(run)
        )


def test_prepare_receipt_rejects_unlisted_external_file(tmp_path):
    run, prepared, source_config, _, plan, config = _continuation(tmp_path)
    split = prepared / "train.jsonl"
    split.write_text("{}\n")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("not allowed")
    root, exact_files = _validated_prepared_root(
        plan=plan, config=config, run=run, original_root=str(run)
    )
    receipt = {
        "files": {
            str(split): file_sha256(split),
            str(source_config): file_sha256(source_config),
            str(unrelated): file_sha256(unrelated),
        }
    }
    with pytest.raises(ValueError, match="outside the official-mobile run"):
        _verified_receipt_files(
            receipt,
            original_root=str(run),
            current_root=run,
            required_prefix=root,
            allowed_external_roots=(root,),
            allowed_external_files=exact_files,
        )


def test_fresh_run_behavior_remains_run_local(tmp_path):
    run = (tmp_path / "fresh").resolve()
    prepared = run / "prepared"
    prepared.mkdir(parents=True)
    plan = {"paths": {"prepared": str(prepared)}}
    config = {"run": {"dataset_dir": str(prepared)}, "training": {}}
    root, external = _validated_prepared_root(
        plan=plan, config=config, run=run, original_root=str(run)
    )
    assert root == prepared and external == ()

    plan["paths"]["prepared"] = str(tmp_path / "outside")
    with pytest.raises(
        ValueError, match="outside the official-mobile run|inside the run"
    ):
        _validated_prepared_root(
            plan=plan, config=config, run=run, original_root=str(run)
        )
