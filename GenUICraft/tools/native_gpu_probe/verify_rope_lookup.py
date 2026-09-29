"""Independently verify one EXPERIMENTAL, NON-DEPLOYMENT RoPE repack.

Checks the pinned source SHA, every original TFLite buffer, the complete
verbatim target copy backing those buffers, graph rewrite scope, package
sections including unchanged MTP bytes, and the repacker's SHA sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import sys
from pathlib import Path

from ai_edge_litert import schema_py_generated as s


def repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "training" / "src" / "ir_training" / "export" /
                "litertlm_inspector.py").is_file():
            return candidate
    raise RuntimeError("Cannot locate A2UI repository root and training inspector")


sys.path.insert(0, str(repo_root() / "training" / "src"))
from ir_training.export.litertlm_inspector import _FlatBufferReader, _metadata_from_header


EXPECTED_INPUT_SHA = "de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62"
TARGET = "tf_lite_prefill_decode"
GRAPHS = {"decode", "prefill_1024", "prefill_128", "verify"}
MAX_POS = 8192
MAX_FRONT_SIZE = 64 * 1024 * 1024
MAX_PACKAGE_GROWTH = 128 * 1024 * 1024
EXPECTED_INLINE_BUFFERS = 657
EXPECTED_INLINE_BUFFER_BYTES = 777_983_384


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_sections(mapped):
    reader = _FlatBufferReader(mapped)
    return _metadata_from_header(reader, reader.follow(32))[1]


def target_index(sections):
    matches = [i for i, section in enumerate(sections)
               if any(item["key"] == "model_type" and item["value"] == TARGET
                      for item in section["items"])]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one assistant target section, got {matches}")
    return matches[0]


def opcode_names(model):
    names = {int(getattr(s.BuiltinOperator, name)): name
             for name in dir(s.BuiltinOperator)
             if name.isupper() and isinstance(getattr(s.BuiltinOperator, name), int)}
    return [names.get(model.OperatorCodes(i).BuiltinCode(), "?")
            for i in range(model.OperatorCodesLength())]


def graph_counts(model):
    codes = opcode_names(model)
    counts = {}
    for index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(index)
        raw_name = graph.Name()
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        if name not in GRAPHS:
            continue
        ops = [codes[graph.Operators(i).OpcodeIndex()]
               for i in range(graph.OperatorsLength())]
        if name in counts:
            raise ValueError(f"Duplicate assistant graph {name}")
        counts[name] = {"gather": ops.count("GATHER"), "sin": ops.count("SIN"),
                        "cos": ops.count("COS"), "operators": len(ops)}
    if set(counts) != GRAPHS:
        raise ValueError(f"Unexpected assistant graphs: {sorted(counts)}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Original pinned rank-64 package")
    parser.add_argument("output", type=Path, help="Package produced by patch_rope_lookup.py")
    parser.add_argument("--repack-only", action="store_true", help="Expect causal packaging control")
    args = parser.parse_args()
    source, patched = args.input.resolve(), args.output.resolve()
    if source == patched or not source.is_file() or not patched.is_file():
        raise ValueError("Input and output must be two existing, distinct files")
    growth = patched.stat().st_size - source.stat().st_size
    if not 0 <= growth <= MAX_PACKAGE_GROWTH:
        raise ValueError("Output package growth is outside the diagnostic bound")
    source_sha = digest_file(source)
    if source_sha != EXPECTED_INPUT_SHA:
        raise ValueError(f"Source SHA is not the pinned rank-64 model: {source_sha}")
    report_path = patched.with_suffix(patched.suffix + ".report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_mode = "repack_only" if args.repack_only else "rope_lookup"
    if (report.get("input_sha256") != source_sha or
            report.get("mode") != expected_mode or
            report.get("max_position_exclusive") != MAX_POS or
            report.get("new_file_size") != patched.stat().st_size):
        raise ValueError("Repack report does not match this input, mode or output size")
    output_sha = digest_file(patched)
    if output_sha != report.get("output_sha256"):
        raise ValueError("Output SHA does not match repack report")

    with source.open("rb") as fo, patched.open("rb") as fp, \
            mmap.mmap(fo.fileno(), 0, access=mmap.ACCESS_READ) as original, \
            mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as output:
        original_sections, output_sections = package_sections(original), package_sections(output)
        target_i = target_index(original_sections)
        if target_i != target_index(output_sections) or len(original_sections) != len(output_sections):
            raise ValueError("Package section identity changed")
        for index, (old, new) in enumerate(zip(original_sections, output_sections)):
            if (old["data_type_name"] != new["data_type_name"] or
                    old["items"] != new["items"]):
                raise ValueError(f"Section metadata changed at index {index}")
            if index == target_i:
                continue
            before = memoryview(original)[old["begin_offset"]:old["end_offset"]]
            after = memoryview(output)[new["begin_offset"]:new["end_offset"]]
            try:
                if before != after:
                    raise ValueError(f"Non-target section {index} differs, including MTP")
            finally:
                before.release()
                after.release()

        osec, psec = original_sections[target_i], output_sections[target_i]
        ov = memoryview(original)[osec["begin_offset"]:osec["end_offset"]]
        pv = memoryview(output)[psec["begin_offset"]:psec["end_offset"]]
        try:
            om, pm = s.Model.GetRootAsModel(ov, 0), s.Model.GetRootAsModel(pv, 0)
            if pm.BuffersLength() != om.BuffersLength() + 4:
                raise ValueError("Expected exactly four added lookup buffers")
            front_sizes = set()
            count = byte_count = 0
            for index in range(om.BuffersLength()):
                old, new = om.Buffers(index), pm.Buffers(index)
                length = old.DataLength()
                if length == 0:
                    if new.DataLength() != 0 or new.Size() != old.Size() or new.Offset() != old.Offset():
                        raise ValueError(f"Empty/external original buffer {index} changed")
                    continue
                count += 1
                byte_count += length
                original_offset = old._tab.Vector(old._tab.Offset(4))
                if new.DataLength() != 0 or new.Size() != length:
                    raise ValueError(f"Buffer {index} was not externalized with its exact length")
                front_sizes.add(int(new.Offset()) - original_offset)
                if ov[original_offset:original_offset + length] != pv[new.Offset():new.Offset() + length]:
                    raise ValueError(f"Original buffer {index} bytes differ")
            if (count != EXPECTED_INLINE_BUFFERS or
                    byte_count != EXPECTED_INLINE_BUFFER_BYTES or
                    len(front_sizes) != 1):
                raise ValueError(
                    f"Unexpected original buffer coverage: {count} buffers, "
                    f"{byte_count} bytes, front sizes {front_sizes}")
            front_size = next(iter(front_sizes))
            if not 0 < front_size <= MAX_FRONT_SIZE or front_size != report.get("front_size"):
                raise ValueError(f"Metadata front size is invalid: {front_size}")
            if pv[front_size:front_size + len(ov)] != ov:
                raise ValueError("Verbatim copy of the complete original target section differs")
            graphs = graph_counts(pm)
            expected = (0, 2, 2) if args.repack_only else (4, 0, 0)
            if any((graph["gather"], graph["sin"], graph["cos"]) != expected
                   for graph in graphs.values()):
                raise ValueError(f"RoPE graph scope differs: {graphs}")
            result = {"mode": expected_mode, "input_sha256": source_sha,
                      "output_sha256": output_sha, "original_buffers": count,
                      "original_buffer_bytes_checked": byte_count,
                      "complete_original_target_copy_checked": len(ov),
                      "unchanged_non_target_sections": len(original_sections) - 1,
                      "front_size": front_size, "graphs": graphs}
            print(json.dumps(result, indent=2))
            del om, pm
        finally:
            ov.release()
            pv.release()


if __name__ == "__main__":
    main()
