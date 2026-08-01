"""Guards DoD #2: a renderer refactor must not break the v5.4 metric pipeline.

`identity_v5_4` used to hash a single hardcoded path,
`.../renderer/FlatSpecRenderer.kt`. Renaming, splitting or moving that one file
made `metric_fingerprint_v5_4()` raise `FileNotFoundError` and stopped the whole
v5.4 pipeline — a landmine under exactly the decomposition work DoD #3 requires.

It now hashes a sorted manifest of `renderer/**/*.kt`. These tests pin the
properties that make that safe, so the landmine cannot be reintroduced.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.genui_quality import identity_v5_4 as identity  # noqa: E402
from pipeline.genui_quality.config_v5_4 import RewardConfigV54  # noqa: E402


def test_manifest_is_non_empty_and_covers_subpackages() -> None:
    manifest = identity.android_renderer_manifest()
    assert manifest, "Renderer manifest is empty; the glob or path is wrong"
    # The decomposition moves files into flat/domain, flat/parse, ... Those must
    # be hashed too, or the fingerprint would stop tracking renderer behaviour.
    assert any("/" in path for path in manifest), (
        "No nested renderer sources found. If the decomposition moved files out "
        "of renderer/**, update ANDROID_RENDERER_DIR in the same commit."
    )


def test_fingerprint_computes() -> None:
    assert identity.metric_fingerprint_v5_4(RewardConfigV54())


def _with_temp_renderer_dir(mutate) -> str:
    """Copies the renderer tree, applies `mutate`, returns the manifest hash."""
    real = identity.ANDROID_RENDERER_DIR
    tmp_root = Path(tempfile.mkdtemp())
    tmp = tmp_root / "renderer"
    shutil.copytree(real, tmp)
    try:
        mutate(tmp)
        identity.ANDROID_RENDERER_DIR = tmp
        return identity.android_renderer_manifest_hash()
    finally:
        identity.ANDROID_RENDERER_DIR = real
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_hash_is_location_independent() -> None:
    """Paths are recorded relative, so a checkout location cannot change scores."""
    assert _with_temp_renderer_dir(lambda _: None) == (
        identity.android_renderer_manifest_hash()
    )


def test_hash_is_insensitive_to_line_endings() -> None:
    """A contributor with core.autocrlf=true must not get a different score."""

    def to_crlf(root: Path) -> None:
        for source in root.rglob("*.kt"):
            data = source.read_bytes()
            source.write_bytes(data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    assert _with_temp_renderer_dir(to_crlf) == identity.android_renderer_manifest_hash()


def test_splitting_a_renderer_file_does_not_raise_and_changes_the_hash() -> None:
    """The exact regression: the old code raised FileNotFoundError here."""

    def split_largest(root: Path) -> None:
        largest = max(root.rglob("*.kt"), key=lambda p: p.stat().st_size)
        text = largest.read_text(encoding="utf-8")
        half = len(text) // 2
        largest.with_name(largest.stem + "PartA.kt").write_text(
            text[:half], encoding="utf-8"
        )
        largest.with_name(largest.stem + "PartB.kt").write_text(
            text[half:], encoding="utf-8"
        )
        largest.unlink()

    split_hash = _with_temp_renderer_dir(split_largest)
    assert split_hash != identity.android_renderer_manifest_hash(), (
        "Splitting a renderer file must change the hash, otherwise the "
        "fingerprint has stopped tracking renderer content"
    )


def test_missing_renderer_tree_fails_loudly() -> None:
    """Silently hashing nothing would split score identity between checkouts."""
    real = identity.ANDROID_RENDERER_DIR
    identity.ANDROID_RENDERER_DIR = Path(tempfile.mkdtemp()) / "does-not-exist"
    try:
        identity.android_renderer_manifest()
    except FileNotFoundError as error:
        assert "full repository checkout" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError for a missing renderer tree")
    finally:
        identity.ANDROID_RENDERER_DIR = real
