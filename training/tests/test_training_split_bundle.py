"""Small binary fixtures prove bundle integrity and safe restore failure behavior."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _module(Path(__file__).resolve().parents[1] / "scripts" / "build_training_split_bundle.py", "split_bundle_builder")


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    payloads = {
        "train.jsonl": '{"text":"क नमस्ते"}\r\n{"text":"second"}\r\n{"text":"last"}'.encode(),
        "val.jsonl": b'{"text":"validation"}\r\n',
    }
    manifest = {"dataset": "fixture", "status": "offline_verified_candidate", "policy": "fixture-policy", "outputs": {}, "rows": {}}
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
        manifest["outputs"][name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        manifest["rows"][Path(name).stem] = payload.count(b"\n") + (not payload.endswith(b"\n"))
    (directory / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\r\n").encode())
    (directory / "prompt_scaffold.json").write_bytes(b'{"messages":[]}\r\n')
    return directory


@pytest.fixture
def bundle(source, tmp_path):
    path = tmp_path / "bundle"
    builder.build_bundle(source, path, part_bytes=13)
    return path, _module(path / "restore.py", "fixture_restore")


def test_roundtrip_preserves_exact_bytes_and_sidecars(source, bundle, tmp_path):
    path, restore = bundle
    manifest = restore.restore(path, verify_only=True)
    assert len(manifest["files"][0]["parts"]) > 1
    output = tmp_path / "restored"
    restore.restore(path, output)
    for name in ("train.jsonl", "val.jsonl", "manifest.json", "prompt_scaffold.json"):
        assert (output / name).read_bytes() == (source / name).read_bytes()


def test_existing_output_is_never_overwritten(bundle, tmp_path):
    path, restore = bundle
    output = tmp_path / "restored"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError, match="nothing overwritten"):
        restore.restore(path, output)
    assert sentinel.read_text() == "keep"
    assert not list(tmp_path.glob(".restored.partial-*"))


def test_same_length_compressed_tamper_fails_before_publishing_output(bundle, tmp_path):
    path, restore = bundle
    part = path / "train.part-00000.gz"
    payload = bytearray(part.read_bytes())
    payload[4] ^= 1  # Gzip timestamp: decodes identically, still fails the digest.
    part.write_bytes(payload)
    output = tmp_path / "restored"
    with pytest.raises(ValueError, match="Compressed checksum mismatch"):
        restore.restore(path, output)
    assert not output.exists()
    partial = list(tmp_path.glob(".restored.partial-*"))
    assert len(partial) == 1
    assert not (partial[0] / "val.jsonl").exists()


def test_sidecar_tamper_is_rejected_before_any_extract(bundle, tmp_path):
    path, restore = bundle
    sidecar = path / "prompt_scaffold.json"
    sidecar.write_bytes(sidecar.read_bytes().replace(b"messages", b"messagEs"))
    with pytest.raises(ValueError, match="File checksum mismatch"):
        restore.restore(path, tmp_path / "restored")
    assert not list(tmp_path.glob(".restored.partial-*"))


def test_part_path_traversal_is_rejected(bundle):
    path, restore = bundle
    manifest = json.loads((path / "bundle.json").read_text())
    manifest["files"][0]["parts"][0]["name"] = "../train.part-00000.gz"
    (path / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe or out-of-order"):
        restore.restore(path, verify_only=True)


def test_source_binding_prevents_changed_split_metadata(bundle):
    path, restore = bundle
    manifest = json.loads((path / "bundle.json").read_text())
    manifest["files"][0]["rows"] += 1
    (path / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Split differs from source manifest"):
        restore.restore(path, verify_only=True)


def test_expansion_bound_is_enforced_while_decoding(bundle):
    path, restore = bundle
    item = copy.deepcopy(restore.load_manifest(path)["files"][0])
    item["parts"][0]["raw_bytes"] -= 1
    with pytest.raises(ValueError, match="Expanded size exceeds manifest"):
        restore._copy_split(path, item)


def test_wrong_source_checksum_never_publishes_bundle(source, tmp_path):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["train.jsonl"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "failed_bundle"
    with pytest.raises(ValueError, match="Source bytes, rows or checksum differ"):
        builder.build_bundle(source, output, part_bytes=13)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed_bundle.building-*"))


def test_gzip_parts_are_reproducible(source, bundle, tmp_path):
    path, _ = bundle
    second = tmp_path / "bundle_again"
    builder.build_bundle(source, second, part_bytes=13)
    assert (path / "bundle.json").read_bytes() == (second / "bundle.json").read_bytes()
    for first_part in path.glob("*.gz"):
        assert first_part.read_bytes() == (second / first_part.name).read_bytes()
