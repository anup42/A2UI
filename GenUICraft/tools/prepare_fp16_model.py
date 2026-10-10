"""Prepare a SHA-pinned compatible E2B model with bounded target and MTP RoPE.

The FP16 GPU runtime also needs the matching float-intermediate Q/DQ policy.
This command only changes RoPE graph operators. It copies every original model
section as external-buffer backing and leaves every other package section intact.
Positions must remain in [0, 8192); the app must enforce this bound.

Usage:
  python GenUICraft/tools/prepare_fp16_model.py \
    .tmp/fp16_rootcause/r64.litertlm \
    .tmp/fp16_app/model-fp16-corrected.litertlm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ai_edge_litert import schema_py_generated as s

from native_gpu_probe.patch_rope_lookup import (
    ALIGN, MAX_PACKAGE_GROWTH, MAX_POS, code_index, copy_n, digest_file,
    new_lookup_operator, new_reshape_operator, new_tensor, pack_model,
    section_tables, stream_table,
)


def repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "training" / "src" / "ir_training" / "export" /
                "litertlm_inspector.py").is_file():
            return candidate
    raise RuntimeError("Cannot locate A2UI repository root")


sys.path.insert(0, str(repo_root() / "training" / "src"))
from ir_training.export.litertlm_inspector import _FlatBufferReader, _metadata_from_header


EXPECTED_SOURCE_SHA = "de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62"
TRANSFORM_VERSION = "pinned-r64-rope-target-mtp-8192-v1"
NEW_EXPORT_TRANSFORM_VERSION = "e2b-rope-target-mtp-8192-v2"
PRECISION_POLICY = "genuicraft-fp16-rope-qdq-v1"
MAX_FRONT_SIZE = 64 * 1024 * 1024
SECTIONS = {
    "tf_lite_prefill_decode": {
        "graphs": {"decode", "prefill_1024", "prefill_128", "verify"},
        "weight_buffers": 657,
        "weight_bytes": 777_983_384,
    },
    "tf_lite_mtp_drafter": {
        "graphs": {"main"},
        "weight_buffers": 75,
        "weight_bytes": 40_779_664,
    },
}


@dataclass
class PreparedSection:
    section_type: str
    front: bytes
    original_size: int
    table_order: list[tuple[str, int]]
    frequencies: dict[int, np.ndarray]
    replacements: list[dict]
    original_buffer_count: int
    original_weight_buffers: int
    original_weight_bytes: int

    @property
    def payload_size(self) -> int:
        return len(self.front) + self.original_size + sum(
            MAX_POS * width * 4 for _, width in self.table_order)


def model_type(section: dict) -> str | None:
    values = [item["value"] for item in section["items"] if item["key"] == "model_type"]
    return values[0] if len(values) == 1 else None


def prepare_section(mapped: mmap.mmap, section: dict, section_type: str) -> PreparedSection:
    begin, end = section["begin_offset"], section["end_offset"]
    view = memoryview(mapped)[begin:end]
    try:
        raw_model = s.Model.GetRootAsModel(view, 0)
        regions: list[tuple[int, int] | None] = []

        # Object API normally copies hundreds of MB of inline weight data. Keep
        # those bytes in the original section and externalize each buffer.
        def unpack_external(self, buffer):
            if buffer is None:
                regions.append(None)
                return
            length = int(buffer.DataLength())
            region = (int(buffer._tab.Vector(buffer._tab.Offset(4))), length) if length else None
            regions.append(region)
            self.data = None
            self.offset = 1 if region else int(buffer.Offset())
            self.size = length if region else int(buffer.Size())

        original_unpack = s.BufferT._UnPack
        try:
            s.BufferT._UnPack = unpack_external
            model = s.ModelT.InitFromObj(raw_model)
        finally:
            s.BufferT._UnPack = original_unpack
        if len(regions) != len(model.buffers):
            raise ValueError(f"{section_type}: buffer inventory changed")
        count = sum(region is not None for region in regions)
        weight_bytes = sum(region[1] for region in regions if region is not None)
        expected = SECTIONS[section_type]
        if (count, weight_bytes) != (expected["weight_buffers"], expected["weight_bytes"]):
            raise ValueError(f"{section_type}: unexpected original weight coverage: {count}, {weight_bytes}")

        gather_code = code_index(model, s.BuiltinOperator.GATHER)
        reshape_code = code_index(model, s.BuiltinOperator.RESHAPE)
        sin_code = code_index(model, s.BuiltinOperator.SIN)
        cos_code = code_index(model, s.BuiltinOperator.COS)
        table_buffers: dict[tuple[str, int], int] = {}
        frequencies: dict[int, np.ndarray] = {}
        replacements: list[dict] = []
        found_graphs: set[str] = set()
        for gi, graph in enumerate(model.subgraphs):
            name = graph.name.decode() if isinstance(graph.name, bytes) else str(graph.name)
            trig_indices = [oi for oi, op in enumerate(graph.operators)
                            if int(op.opcodeIndex) in (sin_code, cos_code)]
            if not trig_indices:
                continue
            if name not in expected["graphs"] or name in found_graphs:
                raise ValueError(f"{section_type}: unexpected RoPE graph {name}")
            found_graphs.add(name)
            if len(trig_indices) != 4:
                raise ValueError(f"{section_type}/{name}: expected four trigonometric operators")
            position = [int(ti) for ti in graph.inputs
                        if b"input_pos" in graph.tensors[int(ti)].name]
            if len(position) != 1:
                raise ValueError(f"{section_type}/{name}: expected one input_pos")
            pos_ti = position[0]
            input_len = int(graph.tensors[pos_ti].shape[0])
            if input_len not in (1, 4, 128, 1024):
                raise ValueError(f"{section_type}/{name}: unexpected position shape")
            producers = {int(ti): oi for oi, op in enumerate(graph.operators)
                         for ti in op.outputs if int(ti) >= 0}
            raw_graph = raw_model.Subgraphs(gi)
            replacement_ops = {}
            kinds = []
            for oi in trig_indices:
                op = graph.operators[oi]
                kind = "sin" if int(op.opcodeIndex) == sin_code else "cos"
                kinds.append(kind)
                if len(op.inputs) != 1 or len(op.outputs) != 1:
                    raise ValueError(f"{section_type}/{name}:{oi}: unexpected trig arity")
                angle_ti = int(op.inputs[0])
                angle_op = graph.operators[producers[angle_ti]]
                if int(angle_op.opcodeIndex) != reshape_code:
                    raise ValueError(f"{section_type}/{name}:{oi}: trig input is not RESHAPE")
                shape_ti = int(angle_op.inputs[1])
                mul_op = graph.operators[producers[int(angle_op.inputs[0])]]
                if int(model.operatorCodes[int(mul_op.opcodeIndex)].builtinCode) != s.BuiltinOperator.MUL:
                    raise ValueError(f"{section_type}/{name}:{oi}: angle is not from MUL")
                candidates = [int(ti) for ti in mul_op.inputs
                              if regions[int(raw_graph.Tensors(int(ti)).Buffer())] is not None]
                if len(candidates) != 1:
                    raise ValueError(f"{section_type}/{name}:{oi}: ambiguous frequency")
                buf_i = int(raw_graph.Tensors(candidates[0]).Buffer())
                width = int(graph.tensors[int(op.outputs[0])].shape[-1])
                if width not in (128, 256) or regions[buf_i][1] != width * 4:
                    raise ValueError(f"{section_type}/{name}:{oi}: unexpected frequency width")
                freq = np.frombuffer(raw_model.Buffers(buf_i).DataAsNumpy(), dtype="<f4").copy()
                if width in frequencies and not np.array_equal(freq, frequencies[width]):
                    raise ValueError(f"{section_type}/{name}:{oi}: differing width-{width} frequency")
                frequencies[width] = freq
                key = (kind, width)
                if key not in table_buffers:
                    buffer = s.BufferT()
                    buffer.data = None
                    buffer.offset = 1
                    buffer.size = MAX_POS * width * 4
                    model.buffers.append(buffer)
                    table_buffers[key] = len(model.buffers) - 1
                table_ti = len(graph.tensors)
                graph.tensors.append(new_tensor([MAX_POS, width], table_buffers[key],
                                                f"fp16_rope_{kind}_table_{width}"))
                gather_ti = len(graph.tensors)
                graph.tensors.append(new_tensor([input_len, width], 0,
                                                f"fp16_rope_{kind}_gather_{width}"))
                output_ti = int(op.outputs[0])
                output_shape = [int(x) for x in graph.tensors[output_ti].shape]
                if output_shape != [1, input_len, 1, width]:
                    raise ValueError(f"{section_type}/{name}:{oi}: unexpected trig output shape")
                replacement_ops[oi] = [
                    new_lookup_operator(gather_code, table_ti, pos_ti, gather_ti),
                    new_reshape_operator(reshape_code, gather_ti, shape_ti,
                                         output_ti, output_shape),
                ]
                replacements.append({"section_type": section_type, "graph": name,
                                     "operator": oi, "kind": kind, "width": width,
                                     "frequency_buffer": buf_i})
            if sorted(kinds) != ["cos", "cos", "sin", "sin"]:
                raise ValueError(f"{section_type}/{name}: incomplete SIN/COS pair")
            graph.operators = [replacement for oi, op in enumerate(graph.operators)
                               for replacement in replacement_ops.get(oi, [op])]
        if found_graphs != expected["graphs"] or set(table_buffers) != {
                (kind, width) for kind in ("sin", "cos") for width in (128, 256)}:
            raise ValueError(f"{section_type}: incomplete graph/table coverage: {found_graphs}")
        marker = f"{TRANSFORM_VERSION}:{section_type}"
        description = model.description.decode() if isinstance(model.description, bytes) else str(model.description or "")
        model.description = (description + "; " + marker).encode()
        first_front = pack_model(model)
        front_size = len(first_front)
        if not 0 < front_size <= MAX_FRONT_SIZE:
            raise ValueError(f"{section_type}: metadata front exceeds bound")
        table_order = sorted(table_buffers)
        offset = front_size + len(view)
        for key in table_order:
            model.buffers[table_buffers[key]].offset = offset
            offset += MAX_POS * key[1] * 4
        for i, region in enumerate(regions):
            if region is not None:
                model.buffers[i].offset = front_size + region[0]
        front = pack_model(model)
        if len(front) != front_size:
            raise ValueError(f"{section_type}: metadata length changed after offset fixup")
        return PreparedSection(section_type, front, len(view), table_order,
                               frequencies, replacements, len(regions), count,
                               weight_bytes)
    finally:
        view.release()


def patch_header(header: bytearray, target_i: int, drafter_i: int,
                 target_end: int, drafter_begin: int, drafter_end: int) -> None:
    reader, tables = section_tables(header)
    for index, field, value in ((target_i, 2, target_end),
                                (drafter_i, 1, drafter_begin),
                                (drafter_i, 2, drafter_end)):
        position = reader.table_field(tables[index], field)
        if position is None:
            raise ValueError("Missing package section offset field")
        struct.pack_into("<Q", header, position, value)


def write_section(source, destination, original: dict,
                  prepared: PreparedSection, table_hashes: dict) -> None:
    destination.write(prepared.front)
    source.seek(original["begin_offset"])
    copy_n(source, destination, prepared.original_size)
    for kind, width in prepared.table_order:
        digest = hashlib.sha256()
        stream_table(destination, kind, prepared.frequencies[width], MAX_POS, digest)
        table_hashes[f"{prepared.section_type}/{kind}_{width}"] = digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA,
                        help="Independently recorded source SHA; structural/weight checks still apply.")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.output.is_symlink():
        raise ValueError("Output may not be a symlink")
    output = args.output.resolve(strict=False)
    manifest_path = Path(str(output) + ".fp16.json")
    if (source == output or not source.is_file() or output.exists() or
            manifest_path.exists() or manifest_path.is_symlink()):
        raise ValueError("Source and output must be distinct; output and manifest must be new")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent does not exist: {output.parent}")
    source_sha = digest_file(source)
    expected_source_sha = args.expected_source_sha256.lower()
    if (len(expected_source_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_source_sha)):
        raise ValueError("Expected source SHA must be 64 hexadecimal characters")
    if source_sha != expected_source_sha:
        raise ValueError(f"Refusing unpinned source hash {source_sha}")
    transform_version = (TRANSFORM_VERSION if source_sha == EXPECTED_SOURCE_SHA
                         else NEW_EXPORT_TRANSFORM_VERSION)
    with source.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        reader = _FlatBufferReader(mapped)
        header_end = struct.unpack_from("<Q", mapped, 24)[0]
        _, sections = _metadata_from_header(reader, reader.follow(32))
        indices = {kind: [i for i, section in enumerate(sections)
                          if model_type(section) == kind] for kind in SECTIONS}
        if (len(sections) != 12 or indices["tf_lite_prefill_decode"] != [10] or
                indices["tf_lite_mtp_drafter"] != [11]):
            raise ValueError(f"Unexpected pinned package section layout: {indices}")
        ti, di = 10, 11
        target, drafter = sections[ti], sections[di]
        prepared_target = prepare_section(mapped, target, "tf_lite_prefill_decode")
        prepared_drafter = prepare_section(mapped, drafter, "tf_lite_mtp_drafter")
        new_target_end = target["begin_offset"] + prepared_target.payload_size
        new_drafter_begin = (new_target_end + ALIGN - 1) // ALIGN * ALIGN
        new_drafter_end = new_drafter_begin + prepared_drafter.payload_size
        trailer_size = len(mapped) - drafter["end_offset"]
        end_before_pad = new_drafter_end + trailer_size
        new_file_size = (end_before_pad + ALIGN - 1) // ALIGN * ALIGN
        growth = new_file_size - len(mapped)
        if not 0 <= growth <= MAX_PACKAGE_GROWTH:
            raise ValueError(f"Unexpected package growth {growth}")
        header = bytearray(mapped[:header_end])
        patch_header(header, ti, di, new_target_end, new_drafter_begin, new_drafter_end)
        patched_reader = _FlatBufferReader(header)
        _, new_sections = _metadata_from_header(patched_reader, patched_reader.follow(32))
        if (new_sections[ti]["end_offset"] != new_target_end or
                new_sections[di]["begin_offset"] != new_drafter_begin or
                new_sections[di]["end_offset"] != new_drafter_end):
            raise ValueError("Patched section metadata is inconsistent")
        plan = {"source": str(source), "output": str(output),
                "source_sha256": source_sha, "source_bytes": len(mapped),
                "model_bytes": new_file_size, "growth_bytes": growth,
                "target_replacements": len(prepared_target.replacements),
                "drafter_replacements": len(prepared_drafter.replacements),
                "max_context_tokens": MAX_POS}
        if args.plan_only:
            print(json.dumps(plan, indent=2))
            return
        required_free = new_file_size + 128 * 1024 * 1024
        available = shutil.disk_usage(output.parent).free
        if available < required_free:
            raise OSError(f"Need {required_free} free bytes for single output, have {available}")
    table_hashes: dict[str, str] = {}
    created_output = created_manifest = False
    try:
        with source.open("rb") as src, output.open("xb") as dst:
            created_output = True
            dst.write(header)
            src.seek(header_end)
            copy_n(src, dst, target["begin_offset"] - header_end)
            write_section(src, dst, target, prepared_target, table_hashes)
            dst.write(b"\x00" * (new_drafter_begin - new_target_end))
            write_section(src, dst, drafter, prepared_drafter, table_hashes)
            src.seek(drafter["end_offset"])
            copy_n(src, dst, source.stat().st_size - drafter["end_offset"])
            dst.write(b"\x00" * (new_file_size - end_before_pad))
        if output.stat().st_size != new_file_size:
            raise ValueError("Output size differs from planned package size")
        manifest = {
            "schema_version": 1,
            "transform_version": transform_version,
            "precision_policy": PRECISION_POLICY,
            "max_context_tokens": MAX_POS,
            "target_rope_corrected": True,
            "drafter_rope_corrected": True,
            "source_sha256": source_sha,
            "source_bytes": source.stat().st_size,
            "model_sha256": digest_file(output),
            "model_bytes": output.stat().st_size,
            "coverage": {
                "target": {"graphs": sorted(SECTIONS["tf_lite_prefill_decode"]["graphs"]),
                           "replaced_sin_cos": len(prepared_target.replacements),
                           "original_weight_buffers": prepared_target.original_weight_buffers,
                           "original_weight_bytes": prepared_target.original_weight_bytes},
                "drafter": {"graphs": sorted(SECTIONS["tf_lite_mtp_drafter"]["graphs"]),
                            "replaced_sin_cos": len(prepared_drafter.replacements),
                            "original_weight_buffers": prepared_drafter.original_weight_buffers,
                            "original_weight_bytes": prepared_drafter.original_weight_bytes},
            },
            "table_sha256": table_hashes,
        }
        with manifest_path.open("x", encoding="utf-8") as manifest_file:
            created_manifest = True
            json.dump(manifest, manifest_file, indent=2)
            manifest_file.write("\n")
    except BaseException:
        if created_output:
            output.unlink(missing_ok=True)
        if created_manifest:
            manifest_path.unlink(missing_ok=True)
        raise
    print(json.dumps({"model": str(output), "manifest": str(manifest_path),
                      "model_sha256": manifest["model_sha256"],
                      "model_bytes": manifest["model_bytes"],
                      "coverage": manifest["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
