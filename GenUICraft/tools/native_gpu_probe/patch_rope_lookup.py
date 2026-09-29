"""EXPERIMENTAL, NON-DEPLOYMENT RoPE lookup rewrite of one pinned model.

Replaces the four assistant subgraphs' RoPE SIN/COS paths with exact-position
GATHERs into precomputed FLOAT32 tables. All original graph buffer payloads are
kept byte-for-byte as external buffers. The MTP and every other package section
are copied unchanged. This is a diagnostic package, limited to positions
0..8191; it is not a production exporter.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import mmap
import os
import shutil
import struct
import sys
from pathlib import Path

import flatbuffers
import numpy as np
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
MAX_POS = 8192
ALIGN = 16384
MAX_FRONT_SIZE = 64 * 1024 * 1024
MAX_PACKAGE_GROWTH = 128 * 1024 * 1024
TARGET = "tf_lite_prefill_decode"
GRAPHS = {"decode", "prefill_1024", "prefill_128", "verify"}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_n(source, target, n: int, hasher=None):
    remaining = n
    while remaining:
        block = source.read(min(8 * 1024 * 1024, remaining))
        if not block:
            raise EOFError(f"Unexpected EOF, remaining={remaining}")
        target.write(block)
        if hasher is not None:
            hasher.update(block)
        remaining -= len(block)


def code_index(model, builtin: int) -> int:
    for i, code in enumerate(model.operatorCodes):
        if int(code.builtinCode) == builtin:
            return i
    code = s.OperatorCodeT()
    code.builtinCode = builtin
    code.deprecatedBuiltinCode = min(builtin, 127)
    code.version = 1
    model.operatorCodes.append(code)
    return len(model.operatorCodes) - 1


def new_tensor(shape, buffer: int, name: str):
    tensor = s.TensorT()
    tensor.shape = list(shape)
    tensor.type = s.TensorType.FLOAT32
    tensor.buffer = buffer
    tensor.name = name.encode()
    tensor.hasRank = True
    return tensor


def new_lookup_operator(gather_code, table_ti, position_ti, output_ti):
    op = s.OperatorT()
    op.opcodeIndex = gather_code
    op.inputs = [table_ti, position_ti]
    op.outputs = [output_ti]
    op.builtinOptionsType = s.BuiltinOptions.GatherOptions
    op.builtinOptions = s.GatherOptionsT()
    op.builtinOptions.axis = 0
    op.builtinOptions.batchDims = 0
    return op


def new_reshape_operator(reshape_code, input_ti, shape_ti, output_ti, output_shape):
    op = s.OperatorT()
    op.opcodeIndex = reshape_code
    op.inputs = [input_ti, shape_ti]
    op.outputs = [output_ti]
    op.builtinOptionsType = s.BuiltinOptions.ReshapeOptions
    op.builtinOptions = s.ReshapeOptionsT()
    op.builtinOptions.newShape = list(output_shape)
    return op


def section_tables(header: bytearray):
    reader = _FlatBufferReader(header)
    root = reader.follow(32)
    section_meta = reader.follow(reader.table_field(root, 1))
    return reader, reader.vector_tables(reader.table_field(section_meta, 0))


def patch_header(header: bytearray, sections, target_index, new_target_end, delta):
    reader, tables = section_tables(header)
    if len(tables) != len(sections):
        raise ValueError("Header section count changed")
    for i, table in enumerate(tables):
        if i == target_index:
            struct.pack_into("<Q", header, reader.table_field(table, 2), new_target_end)
        elif i > target_index:
            for field_index in (1, 2):
                pos = reader.table_field(table, field_index)
                original = reader.u64(pos)
                struct.pack_into("<Q", header, pos, original + delta)


def pack_model(model) -> bytes:
    builder = flatbuffers.Builder(1024 * 1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    return bytes(builder.Output())


def rewrite_graph(model, raw_model, buffer_regions, max_pos: int, *, repack_only: bool = False):
    gather_code = None if repack_only else code_index(model, s.BuiltinOperator.GATHER)
    reshape_code = code_index(model, s.BuiltinOperator.RESHAPE)
    sin_code = code_index(model, s.BuiltinOperator.SIN)
    cos_code = code_index(model, s.BuiltinOperator.COS)
    table_buffers = {}
    frequencies = {}
    replacements = []

    for gi, graph in enumerate(model.subgraphs):
        graph_name = graph.name.decode() if isinstance(graph.name, bytes) else str(graph.name)
        if graph_name not in GRAPHS:
            continue
        raw_graph = raw_model.Subgraphs(gi)
        producers = {int(ti): oi for oi, op in enumerate(graph.operators)
                     for ti in op.outputs if int(ti) >= 0}
        pos = [int(ti) for ti in graph.inputs
               if b"input_pos" in graph.tensors[int(ti)].name]
        if len(pos) != 1:
            raise ValueError(f"{graph_name}: expected one input_pos tensor, got {pos}")
        position_ti = pos[0]
        input_len = int(graph.tensors[position_ti].shape[0])
        if input_len not in {1, 4, 128, 1024}:
            raise ValueError(f"Unexpected position input length {input_len} in {graph_name}")

        replacements_by_operator = {}
        sin_count = cos_count = 0
        for oi, op in enumerate(graph.operators):
            opcode = int(op.opcodeIndex)
            if opcode not in (sin_code, cos_code):
                continue
            kind = "sin" if opcode == sin_code else "cos"
            if len(op.inputs) != 1 or len(op.outputs) != 1:
                raise ValueError(f"{graph_name}:{oi} unexpected trig arity")
            angle_ti = int(op.inputs[0])
            angle_op = graph.operators[producers[angle_ti]]
            if int(angle_op.opcodeIndex) != reshape_code:
                raise ValueError(f"{graph_name}:{oi} trig input is not RESHAPE")
            shape_ti = int(angle_op.inputs[1])
            mul_ti = int(angle_op.inputs[0])
            mul_op = graph.operators[producers[mul_ti]]
            if int(model.operatorCodes[int(mul_op.opcodeIndex)].builtinCode) != s.BuiltinOperator.MUL:
                raise ValueError(f"{graph_name}:{oi} trig ancestry is not MUL")
            freq_candidates = [int(ti) for ti in mul_op.inputs
                               if buffer_regions[int(raw_graph.Tensors(int(ti)).Buffer())] is not None]
            if len(freq_candidates) != 1:
                raise ValueError(f"{graph_name}:{oi} frequency tensor ambiguous: {freq_candidates}")
            freq_ti = freq_candidates[0]
            freq_tensor = raw_graph.Tensors(freq_ti)
            width = int(graph.tensors[int(op.outputs[0])].shape[-1])
            if width not in (128, 256):
                raise ValueError(f"Unexpected RoPE width {width}")
            buf_i = int(freq_tensor.Buffer())
            region = buffer_regions[buf_i]
            if region is None or region[1] != width * 4:
                raise ValueError(f"Frequency buffer mismatch {graph_name}:{oi} {region}")
            freq = np.frombuffer(raw_model.Buffers(buf_i).DataAsNumpy(), dtype="<f4").copy()
            if width in frequencies and not np.array_equal(freq, frequencies[width]):
                raise ValueError(f"Frequency vector differs for width {width}")
            frequencies[width] = freq

            key = (kind, width)
            if key not in table_buffers:
                buf = s.BufferT()
                buf.data = None
                buf.offset = 1  # Keep uint64 field present on the first pack.
                buf.size = max_pos * width * 4
                model.buffers.append(buf)
                table_buffers[key] = len(model.buffers) - 1
            if repack_only:
                sin_count += kind == "sin"
                cos_count += kind == "cos"
                continue
            table_ti = len(graph.tensors)
            graph.tensors.append(new_tensor([max_pos, width], table_buffers[key],
                                            f"test_rope_{kind}_table_{width}"))
            gather_ti = len(graph.tensors)
            graph.tensors.append(new_tensor([input_len, width], 0,
                                            f"test_rope_{kind}_gather_{width}"))
            old_output_ti = int(op.outputs[0])
            output_shape = [int(x) for x in graph.tensors[old_output_ti].shape]
            if output_shape != [1, input_len, 1, width]:
                raise ValueError(f"Unexpected trig output shape in {graph_name}: {output_shape}")
            replacements_by_operator[oi] = [
                new_lookup_operator(gather_code, table_ti, position_ti, gather_ti),
                new_reshape_operator(reshape_code, gather_ti, shape_ti, old_output_ti, output_shape),
            ]
            replacements.append({"subgraph": graph_name, "old_operator": oi,
                                 "kind": kind, "width": width, "frequency_buffer": buf_i,
                                 "output_tensor": old_output_ti})
            sin_count += kind == "sin"
            cos_count += kind == "cos"
        if (sin_count, cos_count) != (2, 2):
            raise ValueError(f"{graph_name}: expected 2 SIN and 2 COS, got {sin_count}/{cos_count}")
        if not repack_only:
            graph.operators = [new_op for oi, op in enumerate(graph.operators)
                               for new_op in replacements_by_operator.get(oi, [op])]

    expected_replacements = 0 if repack_only else 16
    if len(replacements) != expected_replacements or set(frequencies) != {128, 256}:
        raise ValueError(f"Unexpected RoPE rewrite scope: {len(replacements)} replacements, {set(frequencies)} widths")
    original = model.description.decode() if isinstance(model.description, bytes) else str(model.description or "")
    marker = ("diagnostic_repack_only_external_buffers_maxpos_8192" if repack_only
              else "diagnostic_rope_lookup_fp32_table_maxpos_8192")
    model.description = (original + "; " + marker).encode()
    return table_buffers, frequencies, replacements


def stream_table(target, kind: str, freq, max_pos: int, hasher):
    for start in range(0, max_pos, 256):
        count = min(256, max_pos - start)
        pos = np.arange(start, start + count, dtype=np.float32)[:, None]
        angle = pos * freq[None, :]
        values = np.sin(angle) if kind == "sin" else np.cos(angle)
        block = np.asarray(values, dtype="<f4").tobytes(order="C")
        target.write(block)
        hasher.update(block)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--repack-only", action="store_true",
                    help="Causal control: externalize buffers and append unused tables, without graph operator changes")
    ap.add_argument("--match-front-size", type=int,
                    help="Pad metadata front to the patched package front length for identical external offsets")
    ap.add_argument("--plan-only", action="store_true", help="Build and validate metadata without writing a package")
    args = ap.parse_args()
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input model does not exist: {source}")
    if args.output.is_symlink():
        raise ValueError("Output must not be a symlink")
    output = args.output.resolve(strict=False)
    report_path = output.with_suffix(output.suffix + ".report.json")
    if source == output or output.exists() or report_path.exists() or report_path.is_symlink():
        raise ValueError("Output and report must be new paths distinct from the input")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {output.parent}")
    if args.match_front_size is not None:
        if not args.repack_only:
            raise ValueError("--match-front-size is only valid with --repack-only")
        if not 1 <= args.match_front_size <= MAX_FRONT_SIZE:
            raise ValueError(f"--match-front-size must be in [1, {MAX_FRONT_SIZE}]")
    sha = digest_file(source)
    if sha != EXPECTED_INPUT_SHA:
        raise ValueError(f"Refusing unpinned input SHA {sha}")
    with source.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        reader = _FlatBufferReader(mapped)
        header_end = struct.unpack_from("<Q", mapped, 24)[0]
        _, sections = _metadata_from_header(reader, reader.follow(32))
        target_matches = [(i, x) for i, x in enumerate(sections)
                          if any(y["key"] == "model_type" and y["value"] == TARGET for y in x["items"])]
        if len(target_matches) != 1:
            raise ValueError(f"Expected one assistant target section, got {len(target_matches)}")
        ti, target = target_matches[0]
        if ti + 1 >= len(sections):
            raise ValueError("Expected a following MTP section for preservation check")
        target_begin, target_end = target["begin_offset"], target["end_offset"]
        original_target_size = target_end - target_begin
        old_suffix_begin = sections[ti + 1]["begin_offset"]
        if sections[ti + 1]["data_type_name"] != "TFLiteModel":
            raise ValueError("Unexpected section after target")
        view = memoryview(mapped)[target_begin:target_end]
        raw_model = s.Model.GetRootAsModel(view, 0)
        buffer_regions = []

        def unpack_external(self, buffer):
            if buffer is None:
                buffer_regions.append(None)
                return
            length = int(buffer.DataLength())
            region = (int(buffer._tab.Vector(buffer._tab.Offset(4))), length) if length else None
            buffer_regions.append(region)
            self.data = None
            self.offset = 1 if region else int(buffer.Offset())
            self.size = length if region else int(buffer.Size())

        original_unpack = s.BufferT._UnPack
        try:
            s.BufferT._UnPack = unpack_external
            model = s.ModelT.InitFromObj(raw_model)
        finally:
            s.BufferT._UnPack = original_unpack
        if len(buffer_regions) != len(model.buffers):
            raise ValueError("Buffer inventory count changed during unpack")
        table_buffers, frequencies, replacements = rewrite_graph(
            model, raw_model, buffer_regions, MAX_POS, repack_only=args.repack_only)
        front = pack_model(model)
        raw_front_len = len(front)
        front_len = args.match_front_size if args.match_front_size is not None else raw_front_len
        if front_len < raw_front_len:
            raise ValueError(f"Requested front size {front_len} smaller than packed {raw_front_len}")
        if front_len > MAX_FRONT_SIZE:
            raise ValueError(f"Metadata front exceeds {MAX_FRONT_SIZE} bytes")
        table_order = sorted(table_buffers)
        table_offsets = {}
        cursor = front_len + original_target_size
        for key in table_order:
            table_offsets[key] = cursor
            cursor += MAX_POS * key[1] * 4
        for i, region in enumerate(buffer_regions):
            if region is not None:
                model.buffers[i].offset = front_len + region[0]
        for key, bi in table_buffers.items():
            model.buffers[bi].offset = table_offsets[key]
        front = pack_model(model)
        if len(front) != raw_front_len:
            raise ValueError("Packed metadata length changed after offset fixup")
        front += b"\x00" * (front_len - raw_front_len)
        new_target_size = cursor
        new_target_end = target_begin + new_target_size
        new_suffix_begin = (new_target_end + ALIGN - 1) // ALIGN * ALIGN
        delta = new_suffix_begin - old_suffix_begin
        new_size = mapped.size() + delta
        if delta < 0 or delta > MAX_PACKAGE_GROWTH:
            raise ValueError(f"Package growth outside diagnostic bound: {delta}")
        header = bytearray(mapped[:header_end])
        patch_header(header, sections, ti, new_target_end, delta)
        patched_reader = _FlatBufferReader(header)
        _, updated_sections = _metadata_from_header(patched_reader, patched_reader.follow(32))
        if updated_sections[ti]["end_offset"] != new_target_end or updated_sections[ti + 1]["begin_offset"] != new_suffix_begin:
            raise ValueError("Header patch validation failed")
        del raw_model, model
        view.release()
        if args.plan_only:
            print(json.dumps({"mode": "repack_only" if args.repack_only else "rope_lookup",
                              "front_size": front_len, "packed_front_size": raw_front_len,
                              "new_target_size": new_target_size, "new_file_size": new_size,
                              "replacement_count": len(replacements),
                              "output_would_be": str(output)}, indent=2))
            return
        available = shutil.disk_usage(output.parent).free
        if available < new_size + 128 * 1024 * 1024:
            raise OSError(f"Insufficient free disk: need {new_size + 128 * 1024 * 1024}, have {available}")

    # The package is streamed, with the original 818 MB target appended verbatim
    # as external buffer backing for the new TFLite FlatBuffer metadata.
    prefix_sha = hashlib.sha256()
    old_target_sha = hashlib.sha256()
    suffix_sha = hashlib.sha256()
    table_hashes = {}
    created_output = False
    created_report = False
    try:
        with source.open("rb") as src, output.open("xb") as dst:
            created_output = True
            dst.write(header)
            src.seek(header_end)
            copy_n(src, dst, target_begin - header_end, prefix_sha)
            dst.write(front)
            src.seek(target_begin)
            copy_n(src, dst, original_target_size, old_target_sha)
            for key in table_order:
                h = hashlib.sha256()
                stream_table(dst, key[0], frequencies[key[1]], MAX_POS, h)
                table_hashes[f"{key[0]}_{key[1]}"] = h.hexdigest()
            dst.write(b"\x00" * (new_suffix_begin - new_target_end))
            src.seek(old_suffix_begin)
            copy_n(src, dst, source.stat().st_size - old_suffix_begin, suffix_sha)
        if output.stat().st_size != new_size:
            raise ValueError(f"Output size mismatch {output.stat().st_size} != {new_size}")
        report = {"marker": ("diagnostic_repack_only_external_buffers_maxpos_8192" if args.repack_only
                              else "diagnostic_rope_lookup_fp32_table_maxpos_8192"),
                  "mode": "repack_only" if args.repack_only else "rope_lookup",
                  "input_sha256": sha, "output_sha256": digest_file(output),
                  "max_position_exclusive": MAX_POS, "original_target_sha256": old_target_sha.hexdigest(),
                  "original_non_header_prefix_sha256": prefix_sha.hexdigest(),
                  "preserved_suffix_sha256": suffix_sha.hexdigest(),
                  "table_sha256": table_hashes, "front_size": front_len,
                  "packed_front_size": raw_front_len, "new_target_size": new_target_size,
                  "new_file_size": new_size, "replacement_count": len(replacements),
                  "replacements": replacements}
        with report_path.open("x", encoding="utf-8") as report_file:
            created_report = True
            report_file.write(json.dumps(report, indent=2) + "\n")
    except BaseException:
        # Only new paths created by this invocation can be removed here.
        if created_output:
            output.unlink(missing_ok=True)
        if created_report:
            report_path.unlink(missing_ok=True)
        raise
    print(json.dumps({k: v for k, v in report.items() if k != "replacements"}, indent=2))


if __name__ == "__main__":
    main()
