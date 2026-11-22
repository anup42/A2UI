"""Regression checks for Stage 3 attempt storage on Windows."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import generation_audit


@pytest.mark.parametrize("path_length", [259, 260, 263])
def test_windows_attempt_ledger_keeps_readable_exclusive_artifacts(tmp_path, monkeypatch, path_length):
    monkeypatch.setattr(generation_audit, "platform", "win32")
    invocation = "a" * 32
    task = {"ui_id": "ui-1", "generation_invocation_id": invocation, "response_text": "Fit: comfortable"}
    original = Path("generation_attempts") / f"ui-1_{invocation}" / "000_initial.json"
    prefix = "run-\U0001f9ea"
    unpadded = tmp_path / prefix / "artifacts" / original
    padding = path_length - len(str(unpadded.absolute()).encode("utf-16-le")) // 2
    assert padding >= 0
    artifacts = tmp_path / (prefix + "p" * padding) / "artifacts"
    kwargs = {"prompt": "Render the source", "system": None, "seed": 42, "temperature": 0, "max_tokens": 2048}

    completion = '<a2ui>\nroot=Text("Fit: comfortable")\n</a2ui>'
    first = generation_audit.record_attempt(task, artifacts, phase="initial", text=completion, **kwargs)
    retry = generation_audit.record_attempt(task, artifacts, phase="retry_1", text="retry output", **kwargs)
    artifact = Path(first["artifact"])
    assert not artifact.is_absolute()
    assert len(str((artifacts / artifact).absolute()).encode("utf-16-le")) // 2 < 260
    payload = json.loads((artifacts / artifact).read_text(encoding="utf-8"))
    assert payload["phase"] == "initial"
    if path_length == 259:
        assert artifact == original
    else:
        assert artifact.parent == original.parent.parent
        assert payload["ui_id"] == task["ui_id"]
        assert payload["generation_invocation_id"] == invocation
    assert first["artifact"] != retry["artifact"]
    retry_payload = json.loads((artifacts / retry["artifact"]).read_text(encoding="utf-8"))
    assert retry_payload["completion"] == "retry output"
    assert retry_payload["attempt_index"] == 1

    duplicate = {**task, "generation_attempts": []}
    with pytest.raises(FileExistsError):
        generation_audit.record_attempt(duplicate, artifacts, phase="initial", text="overwrite", **kwargs)
    assert duplicate["generation_attempts"] == []
    assert json.loads((artifacts / artifact).read_text(encoding="utf-8"))["completion"] == completion
