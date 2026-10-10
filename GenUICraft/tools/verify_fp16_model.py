"""Independently verify a SHA-pinned target+MTP FP16 RoPE model package.

Checks the SHA-bound source and output, every original weight buffer, complete
verbatim copies of both rewritten sections, all untouched package sections,
the exact graph rewrites, and every lookup-table value for positions 0..8191.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np
from ai_edge_litert import schema_py_generated as s


def repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "training" / "src" / "ir_training" / "export" /
                "litertlm_inspector.py").is_file():
            return candidate
    raise RuntimeError("Cannot locate A2UI repository root")


sys.path.insert(0, str(repo_root() / "training" / "src"))
from ir_training.export.litertlm_inspector import _FlatBufferReader, _metadata_from_header


SOURCE_SHA = "de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62"
TRANSFORM_VERSION = "pinned-r64-rope-target-mtp-8192-v1"
NEW_EXPORT_TRANSFORM_VERSION = "e2b-rope-target-mtp-8192-v2"
PRECISION_POLICY = "genuicraft-fp16-rope-qdq-v1"
MAX_POS = 8192
ALIGN = 16384
SECTION_RULES = {
    "tf_lite_prefill_decode": {
        "index": 10,
        "graphs": {"decode", "prefill_1024", "prefill_128", "verify"},
        "buffers": 657,
        "bytes": 777_983_384,
        "coverage_key": "target",
    },
    "tf_lite_mtp_drafter": {
        "index": 11,
        "graphs": {"main"},
        "buffers": 75,
        "bytes": 40_779_664,
        "coverage_key": "drafter",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sections_of(mapped: mmap.mmap) -> list[dict]:
    reader = _FlatBufferReader(mapped)
    return _metadata_from_header(reader, reader.follow(32))[1]


def model_type(section: dict) -> str | None:
    values = [item["value"] for item in section["items"] if item["key"] == "model_type"]
    return values[0] if len(values) == 1 else None


def op_code(model, operator) -> int:
    return int(model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode())


def io(operator) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (tuple(int(operator.Inputs(i)) for i in range(operator.InputsLength())),
            tuple(int(operator.Outputs(i)) for i in range(operator.OutputsLength())))


def graph_name(graph) -> str:
    name = graph.Name()
    return name.decode() if isinstance(name, bytes) else str(name)


def tensor_shape(tensor) -> tuple[int, ...]:
    return tuple(int(tensor.Shape(i)) for i in range(tensor.ShapeLength()))


def read_frequency(original, graph, trig_operator, producers: dict[int, int],
                   section_type: str, name: str) -> tuple[np.ndarray, int, int]:
    trig_inputs, trig_outputs = io(trig_operator)
    require(len(trig_inputs) == len(trig_outputs) == 1,
            f"{section_type}/{name}: trig arity changed")
    angle_op = graph.Operators(producers[trig_inputs[0]])
    require(op_code(original, angle_op) == s.BuiltinOperator.RESHAPE,
            f"{section_type}/{name}: trig angle lacks RESHAPE")
    angle_inputs, _ = io(angle_op)
    require(len(angle_inputs) == 2, f"{section_type}/{name}: angle reshape shape missing")
    mul_op = graph.Operators(producers[angle_inputs[0]])
    require(op_code(original, mul_op) == s.BuiltinOperator.MUL,
            f"{section_type}/{name}: trig angle lacks MUL")
    mul_inputs, _ = io(mul_op)
    constants = [(int(graph.Tensors(ti).Buffer()), ti) for ti in mul_inputs
                 if original.Buffers(int(graph.Tensors(ti).Buffer())).DataLength() > 0]
    require(len(constants) == 1, f"{section_type}/{name}: frequency is ambiguous")
    buffer_index, _ = constants[0]
    width = tensor_shape(graph.Tensors(trig_outputs[0]))[-1]
    require(width in (128, 256), f"{section_type}/{name}: unsupported RoPE width")
    freq_buffer = original.Buffers(buffer_index)
    require(freq_buffer.DataLength() == width * 4,
            f"{section_type}/{name}: frequency buffer length differs")
    freq = np.frombuffer(freq_buffer.DataAsNumpy(), dtype="<f4").copy()
    return freq, angle_inputs[1], width


def verify_tables(original, patched, original_graphs: dict[str, object],
                  output_view: memoryview, old_buffers: int, manifest: dict,
                  section_type: str, table_buffers: dict[str, int],
                  table_start: int) -> dict[str, str]:
    expected_frequencies: dict[int, np.ndarray] = {}
    for name, graph in original_graphs.items():
        producers = {int(graph.Operators(i).Outputs(j)): i
                     for i in range(graph.OperatorsLength())
                     for j in range(graph.Operators(i).OutputsLength())
                     if int(graph.Operators(i).Outputs(j)) >= 0}
        for oi in range(graph.OperatorsLength()):
            op = graph.Operators(oi)
            if op_code(original, op) not in (s.BuiltinOperator.SIN, s.BuiltinOperator.COS):
                continue
            freq, _, width = read_frequency(original, graph, op, producers,
                                            section_type, name)
            if width in expected_frequencies:
                require(np.array_equal(freq, expected_frequencies[width]),
                        f"{section_type}: width-{width} frequency differs between graphs")
            expected_frequencies[width] = freq
    require(set(expected_frequencies) == {128, 256},
            f"{section_type}: missing frequency vector")
    expected_order = [("cos", 128), ("cos", 256), ("sin", 128), ("sin", 256)]
    require(set(table_buffers) == {f"{kind}_{width}" for kind, width in expected_order} and
            set(table_buffers.values()) == set(range(old_buffers, old_buffers + 4)),
            f"{section_type}: appended table buffer coverage differs")
    actual_hashes = {}
    next_offset = table_start
    for kind, width in expected_order:
        buffer = patched.Buffers(table_buffers[f"{kind}_{width}"])
        length = MAX_POS * width * 4
        offset = int(buffer.Offset())
        require(buffer.DataLength() == 0 and buffer.Size() == length and
                offset == next_offset and offset <= len(output_view) - length,
                f"{section_type}/{kind}_{width}: invalid table buffer")
        next_offset += length
        digest = hashlib.sha256()
        frequency = expected_frequencies[width]
        for start in range(0, MAX_POS, 256):
            count = min(256, MAX_POS - start)
            positions = np.arange(start, start + count, dtype=np.float32)[:, None]
            angles = positions * frequency[None, :]
            values = np.sin(angles) if kind == "sin" else np.cos(angles)
            expected = np.asarray(values, dtype="<f4").tobytes(order="C")
            begin = offset + start * width * 4
            actual = output_view[begin:begin + len(expected)]
            try:
                require(actual == expected,
                        f"{section_type}/{kind}_{width}: table differs at position {start}")
                digest.update(actual)
            finally:
                actual.release()
        key = f"{section_type}/{kind}_{width}"
        actual_hashes[key] = digest.hexdigest()
        require(manifest["table_sha256"].get(key) == actual_hashes[key],
                f"{key}: table SHA differs from manifest")
    require(next_offset == len(output_view),
            f"{section_type}: unexpected bytes after lookup tables")
    return actual_hashes


def verify_graphs(original, patched, rules: dict, section_type: str) -> tuple[int, dict]:
    require(original.SubgraphsLength() == patched.SubgraphsLength(),
            f"{section_type}: subgraph count changed")
    originals: dict[str, object] = {}
    seen = set()
    replacement_count = 0
    graph_report = {}
    table_buffers: dict[str, int] = {}
    for gi in range(original.SubgraphsLength()):
        before, after = original.Subgraphs(gi), patched.Subgraphs(gi)
        name = graph_name(before)
        require(name == graph_name(after), f"{section_type}: graph name changed at {gi}")
        original_tensors = before.TensorsLength()
        require(after.TensorsLength() == original_tensors + (8 if name in rules["graphs"] else 0),
                f"{section_type}/{name}: tensor count differs")
        for ti in range(original_tensors):
            left, right = before.Tensors(ti), after.Tensors(ti)
            require((left.Name(), left.Type(), left.Buffer(), tensor_shape(left)) ==
                    (right.Name(), right.Type(), right.Buffer(), tensor_shape(right)),
                    f"{section_type}/{name}: original tensor {ti} changed")
        positions = [int(before.Inputs(i)) for i in range(before.InputsLength())
                     if b"input_pos" in before.Tensors(int(before.Inputs(i))).Name()]
        if name in rules["graphs"]:
            require(name not in seen and len(positions) == 1,
                    f"{section_type}/{name}: duplicate graph or missing position input")
            seen.add(name)
            originals[name] = before
            producers = {int(before.Operators(i).Outputs(j)): i
                         for i in range(before.OperatorsLength())
                         for j in range(before.Operators(i).OutputsLength())
                         if int(before.Operators(i).Outputs(j)) >= 0}
        else:
            producers = {}
        ai = 0
        kinds = []
        for oi in range(before.OperatorsLength()):
            left = before.Operators(oi)
            kind = op_code(original, left)
            if kind in (s.BuiltinOperator.SIN, s.BuiltinOperator.COS):
                require(name in rules["graphs"],
                        f"{section_type}/{name}: unhandled trigonometric operator")
                kinds.append("sin" if kind == s.BuiltinOperator.SIN else "cos")
                gather, reshape = after.Operators(ai), after.Operators(ai + 1)
                require(op_code(patched, gather) == s.BuiltinOperator.GATHER and
                        op_code(patched, reshape) == s.BuiltinOperator.RESHAPE,
                        f"{section_type}/{name}:{oi}: missing GATHER+RESHAPE")
                gather_inputs, gather_outputs = io(gather)
                reshape_inputs, reshape_outputs = io(reshape)
                freq, shape_ti, width = read_frequency(original, before, left,
                                                       producers, section_type, name)
                del freq
                require(len(gather_inputs) == 2 and len(gather_outputs) == 1 and
                        gather_inputs[1] == positions[0] and
                        gather_inputs[0] >= original_tensors and
                        gather_outputs[0] >= original_tensors and
                        reshape_inputs == (gather_outputs[0], shape_ti) and
                        reshape_outputs == io(left)[1],
                        f"{section_type}/{name}:{oi}: lookup wiring differs")
                table = after.Tensors(gather_inputs[0])
                table_key = f"{kinds[-1]}_{width}"
                require(tensor_shape(table) == (MAX_POS, width) and
                        table.Type() == s.TensorType.FLOAT32 and
                        table.Buffer() >= original.BuffersLength() and
                        table.Name() == f"fp16_rope_{kinds[-1]}_table_{width}".encode(),
                        f"{section_type}/{name}:{oi}: wrong lookup table")
                if table_key in table_buffers:
                    require(table_buffers[table_key] == table.Buffer(),
                            f"{section_type}/{name}:{oi}: table buffer changed within section")
                table_buffers[table_key] = table.Buffer()
                gather_options = gather.BuiltinOptions()
                reshape_options = reshape.BuiltinOptions()
                require(gather.BuiltinOptionsType() == s.BuiltinOptions.GatherOptions and
                        reshape.BuiltinOptionsType() == s.BuiltinOptions.ReshapeOptions and
                        gather_options is not None and reshape_options is not None,
                        f"{section_type}/{name}:{oi}: lookup options missing")
                parsed_gather = s.GatherOptions()
                parsed_gather.Init(gather_options.Bytes, gather_options.Pos)
                parsed_reshape = s.ReshapeOptions()
                parsed_reshape.Init(reshape_options.Bytes, reshape_options.Pos)
                require(parsed_gather.Axis() == 0 and parsed_gather.BatchDims() == 0 and
                        tuple(int(x) for x in parsed_reshape.NewShapeAsNumpy()) ==
                        tensor_shape(before.Tensors(io(left)[1][0])),
                        f"{section_type}/{name}:{oi}: lookup axis or output shape differs")
                replacement_count += 1
                ai += 2
            else:
                right = after.Operators(ai)
                require(kind == op_code(patched, right) and io(left) == io(right),
                        f"{section_type}/{name}:{oi}: unrelated operator changed")
                ai += 1
        require(ai == after.OperatorsLength(), f"{section_type}/{name}: operator count differs")
        if name in rules["graphs"]:
            require(sorted(kinds) == ["cos", "cos", "sin", "sin"],
                    f"{section_type}/{name}: incomplete SIN/COS replacement")
            graph_report[name] = {"replaced_sin_cos": len(kinds),
                                  "position_input": positions[0]}
    require(seen == rules["graphs"] and replacement_count == 4 * len(rules["graphs"]),
            f"{section_type}: graph coverage incomplete: {seen}")
    return replacement_count, {"graphs": graph_report,
                               "source_graphs": originals,
                               "table_buffers": table_buffers}


def verify_section(source_map: mmap.mmap, output_map: mmap.mmap,
                   old_section: dict, new_section: dict,
                   section_type: str, manifest: dict) -> dict:
    rules = SECTION_RULES[section_type]
    old_view = memoryview(source_map)[old_section["begin_offset"]:old_section["end_offset"]]
    new_view = memoryview(output_map)[new_section["begin_offset"]:new_section["end_offset"]]
    try:
        original = s.Model.GetRootAsModel(old_view, 0)
        patched = s.Model.GetRootAsModel(new_view, 0)
        old_buffers = original.BuffersLength()
        require(patched.BuffersLength() == old_buffers + 4,
                f"{section_type}: expected exactly four added lookup buffers")
        front_sizes = set()
        weights = bytes_checked = 0
        for bi in range(old_buffers):
            before, after = original.Buffers(bi), patched.Buffers(bi)
            length = before.DataLength()
            if not length:
                require(after.DataLength() == 0 and before.Size() == after.Size() and
                        before.Offset() == after.Offset(),
                        f"{section_type}: original empty/external buffer {bi} changed")
                continue
            old_offset = before._tab.Vector(before._tab.Offset(4))
            front_sizes.add(int(after.Offset()) - old_offset)
            require(after.DataLength() == 0 and after.Size() == length and
                    old_view[old_offset:old_offset + length] ==
                    new_view[after.Offset():after.Offset() + length],
                    f"{section_type}: original weight buffer {bi} differs")
            weights += 1
            bytes_checked += length
        require((weights, bytes_checked) == (rules["buffers"], rules["bytes"]) and
                len(front_sizes) == 1, f"{section_type}: original weight coverage differs")
        front_size = next(iter(front_sizes))
        require(0 < front_size <= 64 * 1024 * 1024 and
                new_view[front_size:front_size + len(old_view)] == old_view,
                f"{section_type}: full original section copy differs")
        table_bytes = 4 * MAX_POS * (128 + 256) * 2
        require(len(new_view) == front_size + len(old_view) + table_bytes,
                f"{section_type}: unexpected section payload size")
        replacements, graphs = verify_graphs(original, patched, rules, section_type)
        table_hashes = verify_tables(original, patched, graphs["source_graphs"],
                                     new_view, old_buffers, manifest, section_type,
                                     graphs["table_buffers"], front_size + len(old_view))
        coverage = manifest["coverage"].get(rules["coverage_key"])
        require(coverage == {"graphs": sorted(rules["graphs"]),
                             "replaced_sin_cos": replacements,
                             "original_weight_buffers": weights,
                             "original_weight_bytes": bytes_checked},
                f"{section_type}: manifest graph/weight coverage differs")
        return {"original_weight_buffers": weights,
                "original_weight_bytes_checked": bytes_checked,
                "full_original_section_bytes_checked": len(old_view),
                "replaced_sin_cos": replacements,
                "graphs": graphs["graphs"], "table_sha256": table_hashes}
    finally:
        old_view.release()
        new_view.release()


def verify(source: Path, output: Path, expected_source_sha: str = SOURCE_SHA) -> dict:
    source = source.resolve()
    output = output.resolve()
    require(source != output and source.is_file() and output.is_file(),
            "Expected two existing, distinct model files")
    manifest_path = Path(str(output) + ".fp16.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_source_sha = expected_source_sha.lower()
    require(len(expected_source_sha) == 64 and all(char in "0123456789abcdef" for char in expected_source_sha),
            "Expected source SHA must be 64 hexadecimal characters")
    transform_version = (TRANSFORM_VERSION if expected_source_sha == SOURCE_SHA
                         else NEW_EXPORT_TRANSFORM_VERSION)
    require(manifest.get("schema_version") == 1 and
            manifest.get("transform_version") == transform_version and
            manifest.get("precision_policy") == PRECISION_POLICY and
            manifest.get("max_context_tokens") == MAX_POS and
            manifest.get("target_rope_corrected") is True and
            manifest.get("drafter_rope_corrected") is True,
            "FP16 manifest contract is incomplete")
    require(source.stat().st_size == manifest.get("source_bytes") and
            output.stat().st_size == manifest.get("model_bytes") and
            output.stat().st_size % ALIGN == 0,
            "FP16 manifest size or output alignment differs")
    source_sha = sha256_file(source)
    require(source_sha == expected_source_sha == manifest.get("source_sha256"),
            "Source SHA differs from independently pinned export")
    output_sha = sha256_file(output)
    require(output_sha == manifest.get("model_sha256"),
            "Output SHA differs from manifest")
    with source.open("rb") as original_file, output.open("rb") as patched_file, \
            mmap.mmap(original_file.fileno(), 0, access=mmap.ACCESS_READ) as original_map, \
            mmap.mmap(patched_file.fileno(), 0, access=mmap.ACCESS_READ) as patched_map:
        before_sections, after_sections = sections_of(original_map), sections_of(patched_map)
        require(len(before_sections) == len(after_sections) == 12,
                "Package section count differs")
        for index, (before, after) in enumerate(zip(before_sections, after_sections)):
            require((before["data_type_name"], before["items"]) ==
                    (after["data_type_name"], after["items"]),
                    f"Package section metadata changed at {index}")
            if index < 10:
                require(before["begin_offset"] == after["begin_offset"] and
                        before["end_offset"] == after["end_offset"] and
                        original_map[before["begin_offset"]:before["end_offset"]] ==
                        patched_map[after["begin_offset"]:after["end_offset"]],
                        f"Untouched package section {index} differs")
        require(model_type(before_sections[10]) == model_type(after_sections[10]) ==
                "tf_lite_prefill_decode" and
                model_type(before_sections[11]) == model_type(after_sections[11]) ==
                "tf_lite_mtp_drafter", "Target/drafter section identities differ")
        header_end = struct.unpack_from("<Q", original_map, 24)[0]
        require(original_map[header_end:before_sections[10]["begin_offset"]] ==
                patched_map[header_end:after_sections[10]["begin_offset"]],
                "Unmodified package prefix differs")
        restored_header = bytearray(patched_map[:header_end])
        reader = _FlatBufferReader(restored_header)
        root = reader.follow(32)
        meta = reader.follow(reader.table_field(root, 1))
        tables = reader.vector_tables(reader.table_field(meta, 0))
        for index, field in ((10, 2), (11, 1), (11, 2)):
            where = reader.table_field(tables[index], field)
            original_value = struct.unpack_from("<Q", original_map, where)[0]
            struct.pack_into("<Q", restored_header, where, original_value)
        require(restored_header == original_map[:header_end],
                "Package header has changes beyond the three section offsets")
        require(after_sections[10]["begin_offset"] == before_sections[10]["begin_offset"] and
                after_sections[11]["begin_offset"] % ALIGN == 0,
                "Section start/alignment differs")
        gap = patched_map[after_sections[10]["end_offset"]:after_sections[11]["begin_offset"]]
        require(all(byte == 0 for byte in gap), "Target-to-drafter alignment gap is not zero")
        old_tail = original_map[before_sections[11]["end_offset"]:]
        new_tail = patched_map[after_sections[11]["end_offset"]:]
        require(new_tail[:len(old_tail)] == old_tail and
                all(byte == 0 for byte in new_tail[len(old_tail):]),
                "Original package trailer differs")
        report = {}
        for section_type, rules in SECTION_RULES.items():
            i = rules["index"]
            report[rules["coverage_key"]] = verify_section(
                original_map, patched_map, before_sections[i], after_sections[i],
                section_type, manifest)
    return {"source_sha256": source_sha, "model_sha256": output_sha,
            "model_bytes": output.stat().st_size,
            "unchanged_other_sections": 10,
            "target": report["target"], "drafter": report["drafter"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-source-sha256", default=SOURCE_SHA)
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.output, args.expected_source_sha256), indent=2))


if __name__ == "__main__":
    main()
