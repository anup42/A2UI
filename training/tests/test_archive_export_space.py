import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import finalize_archive_boundaries as finalizer


def test_space_check_reserves_headroom_and_does_not_create_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        finalizer, "disk_usage", lambda _: SimpleNamespace(free=1024**3)
    )
    output = tmp_path / "nested" / "new-copy"
    with pytest.raises(OSError, match="No output directory was created"):
        finalizer.check_export_space(output, 100)
    assert not output.parent.exists()


def test_space_check_accepts_sufficient_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        finalizer, "disk_usage", lambda _: SimpleNamespace(free=2 * 1024**3)
    )
    result = finalizer.check_export_space(tmp_path / "copy", 1024**3)
    assert result["required_bytes"] == result["available_bytes_before_export"]
    assert result["headroom_bytes"] == 1024**3
