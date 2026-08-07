"""Audit observable LiteRT-LM quantization structure without loading weights.

The mobile Gemma 4 artifact is not self-describing: its private recipe and
calibration data are not stored in the LiteRT-LM header.  This read-only audit
therefore reports the evidence that *is* recoverable from embedded TFLite
graphs: low-bit tensor types, per-axis scale layout, integer activation edges,
section roles, and the presence of MTP.  It deliberately does not claim to
recover Google's QAT schedule, observers, calibration corpus, or weight bytes.

The script uses the optional pure-Python ``tflite`` schema bindings.  It does
not run training, mutate the artifact, or materialize model weights.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import mmap
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.export.litertlm_inspector import (  # noqa: E402
    LiteRTLMInspectionError,
    inspect_litertlm,
)


class RecipeAuditError(RuntimeError):
    """Raised when the artifact cannot be audited."""


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _vector(obj: Any, length_method: str, item_method: str) -> list[int]:
    length = getattr(obj, length_method, lambda: 0)() or 0
    method = getattr(obj, item_method, None)
    if method is None:
        return []
    return [int(method(index)) for index in range(int(length))]


def _enum_names(module: Any, class_name: str, fallbacks: dict[int, str] | None = None) -> dict[int, str]:
    cls = getattr(module, class_name, None)
    names = {}
    if cls is not None:
        names.update(
            {
                int(value): name
                for name, value in vars(cls).items()
                if not name.startswith("_") and isinstance(value, int)
            }
        )
    for value, name in (fallbacks or {}).items():
        names.setdefault(value, name)
    return names


def _quantization(tensor: Any) -> dict[str, Any] | None:
    quant = tensor.Quantization()
    if quant is None:
        return None
    scale_count = int(quant.ScaleLength() or 0)
    zero_count = int(quant.ZeroPointLength() or 0)
    if not scale_count and not zero_count:
        return None
    # Scanning zero points is inexpensive compared with mapping a mobile
    # artifact and proves the symmetric-zero-point property observable in the
    # FlatBuffer.  It never reads a weight buffer.
    zero_points = [int(quant.ZeroPoint(i)) for i in range(zero_count)]
    return {
        "scale_count": scale_count,
        "zero_point_count": zero_count,
        "quantized_dimension": int(quant.QuantizedDimension()),
        "zero_points_all_zero": bool(zero_points) and all(value == 0 for value in zero_points),
    }


def _model_type(section: dict[str, Any]) -> str:
    return next(
        (
            str(item.get("value"))
            for item in section.get("items", [])
            if item.get("key") == "model_type"
        ),
        "",
    )


def _low_bit_name(type_value: int, tensor_names: dict[int, str]) -> str:
    return tensor_names.get(type_value, f"unknown:{type_value}")


def _layer_group(name: str) -> str:
    lower = name.lower()
    if "/mlp/" in lower or "/mlp" in lower:
        marker = "mlp"
        layer = ""
        for part in name.split("/"):
            if part.startswith("layer_"):
                layer = part.removeprefix("layer_")
                break
        return f"mlp_layer_{layer}" if layer.isdigit() else marker
    if "per_layer" in lower:
        return "per_layer"
    if "/attn" in lower or "attention" in lower:
        return "self_attention"
    return "other"


def _layer_scope(name: str) -> str:
    match = re.search(r"(?:^|/)layer_(\d+)(?:/|\.)", name)
    return f"layer_{match.group(1)}" if match else "global"


def _operation_family(name: str) -> str:
    lower = name.lower()
    if "decode_softmax" in lower:
        return "lm_head"
    if "mtp_pre_project" in lower or "mtp_pre_proj" in lower:
        return "mtp_pre_project"
    if "mtp_post_project" in lower or "mtp_post_proj" in lower:
        return "mtp_post_project"
    if "per_layer_model_projection" in lower:
        return "per_layer_model_projection"
    if "per_layer_embedding" in lower:
        return "per_layer_embedding"
    if "/mlp/" in lower or "/mlp" in lower:
        return "mlp"
    if "pre_q" in lower or "pre_qkv" in lower or "post_q" in lower or "post_qkv" in lower:
        return "self_attention"
    return "other"


def _audit_graph(model: Any, *, tensor_names: dict[int, str], operator_names: dict[int, str]) -> dict[str, Any]:
    quantized_tensors = collections.Counter()
    layout_histogram = collections.Counter()
    op_histogram = collections.Counter()
    custom_codes: list[str] = []
    fc_weight_bits = collections.Counter()
    fc_weight_groups = collections.Counter()
    fc_activation_inputs = collections.Counter()
    fc_activation_outputs = collections.Counter()
    fc_activation_input_layouts = collections.Counter()
    fc_activation_output_layouts = collections.Counter()
    fc_assignments: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"weight_types": collections.Counter(), "families": collections.defaultdict(collections.Counter)}
    )
    embedding_weight_bits = collections.Counter()
    examples: list[dict[str, Any]] = []

    operator_codes = []
    for index in range(int(model.OperatorCodesLength() or 0)):
        code = model.OperatorCodes(index)
        builtin = int(code.BuiltinCode())
        operator_codes.append((builtin, operator_names.get(builtin, f"unknown:{builtin}")))
        custom = code.CustomCode()
        if custom:
            custom_codes.append(_decode(custom))

    def tensor_record(subgraph: Any, index: int) -> dict[str, Any] | None:
        if index < 0 or index >= int(subgraph.TensorsLength() or 0):
            return None
        tensor = subgraph.Tensors(index)
        type_value = int(tensor.Type())
        quant = _quantization(tensor)
        name = _decode(tensor.Name())
        record = {
            "type": _low_bit_name(type_value, tensor_names),
            "type_value": type_value,
            "shape": [int(tensor.Shape(i)) for i in range(int(tensor.ShapeLength() or 0))],
            "name": name,
            "quantization": quant,
        }
        if quant is not None:
            quantized_tensors[record["type"]] += 1
            layout_histogram[
                f"{record['type']}|scales={quant['scale_count']}|"
                f"zero_points={quant['zero_point_count']}|"
                f"axis={quant['quantized_dimension']}|symmetric={quant['zero_points_all_zero']}"
            ] += 1
        return record

    for subgraph_index in range(int(model.SubgraphsLength() or 0)):
        subgraph = model.Subgraphs(subgraph_index)
        for operator_index in range(int(subgraph.OperatorsLength() or 0)):
            operator = subgraph.Operators(operator_index)
            opcode_index = int(operator.OpcodeIndex())
            builtin = operator_codes[opcode_index][0] if opcode_index < len(operator_codes) else -1
            op_name = operator_names.get(builtin, f"unknown:{builtin}")
            op_histogram[op_name] += 1
            inputs = _vector(operator, "InputsLength", "Inputs")
            outputs = _vector(operator, "OutputsLength", "Outputs")
            if op_name not in {"FULLY_CONNECTED", "EMBEDDING_LOOKUP"}:
                continue
            records = [tensor_record(subgraph, index) for index in inputs]
            output_records = [tensor_record(subgraph, index) for index in outputs]
            if op_name == "EMBEDDING_LOOKUP" and len(records) > 1 and records[1]:
                weight = records[1]
                if weight["type"] in {"INT2", "INT4", "INT8"}:
                    embedding_weight_bits[weight["type"]] += 1
            if op_name == "FULLY_CONNECTED" and len(records) > 1 and records[1]:
                weight = records[1]
                if weight["type"] in {"INT2", "INT4", "INT8"}:
                    fc_weight_bits[weight["type"]] += 1
                    fc_weight_groups[_layer_group(weight["name"])] += 1
                    scope = _layer_scope(weight["name"])
                    family = _operation_family(weight["name"])
                    fc_assignments[scope]["weight_types"][weight["type"]] += 1
                    fc_assignments[scope]["families"][family][weight["type"]] += 1
                    if records and records[0]:
                        fc_activation_inputs[records[0]["type"]] += 1
                        input_quant = records[0].get("quantization")
                        if input_quant:
                            fc_activation_input_layouts[
                                f"{records[0]['type']}|scales={input_quant['scale_count']}|"
                                f"zero_points={input_quant['zero_point_count']}|"
                                f"axis={input_quant['quantized_dimension']}|"
                                f"symmetric={input_quant['zero_points_all_zero']}"
                            ] += 1
                    if output_records and output_records[0]:
                        fc_activation_outputs[output_records[0]["type"]] += 1
                        output_quant = output_records[0].get("quantization")
                        if output_quant:
                            fc_activation_output_layouts[
                                f"{output_records[0]['type']}|scales={output_quant['scale_count']}|"
                                f"zero_points={output_quant['zero_point_count']}|"
                                f"axis={output_quant['quantized_dimension']}|"
                                f"symmetric={output_quant['zero_points_all_zero']}"
                            ] += 1
                    if len(examples) < 8:
                        examples.append(
                            {
                                "weight_type": weight["type"],
                                "weight_shape": weight["shape"],
                                "weight_name": weight["name"],
                                "input_type": records[0]["type"] if records and records[0] else None,
                                "output_type": output_records[0]["type"] if output_records and output_records[0] else None,
                                "input_quantization": records[0]["quantization"] if records and records[0] else None,
                                "weight_quantization": weight["quantization"],
                                "output_quantization": output_records[0]["quantization"] if output_records and output_records[0] else None,
                            }
                        )

    return {
        "subgraph_count": int(model.SubgraphsLength() or 0),
        "operator_histogram": dict(sorted(op_histogram.items())),
        "quantized_tensor_types": dict(sorted(quantized_tensors.items())),
        "quantization_layout_histogram": dict(sorted(layout_histogram.items())),
        "custom_operator_codes": sorted(set(custom_codes)),
        "embedding_weight_types": dict(sorted(embedding_weight_bits.items())),
        "fully_connected_weight_types": dict(sorted(fc_weight_bits.items())),
        "fully_connected_weight_groups": dict(sorted(fc_weight_groups.items())),
        "fully_connected_activation_input_types": dict(sorted(fc_activation_inputs.items())),
        "fully_connected_activation_output_types": dict(sorted(fc_activation_outputs.items())),
        "fully_connected_activation_input_layouts": dict(sorted(fc_activation_input_layouts.items())),
        "fully_connected_activation_output_layouts": dict(sorted(fc_activation_output_layouts.items())),
        "fully_connected_assignments": {
            scope: {
                "weight_types": dict(sorted(record["weight_types"].items())),
                "families": {
                    family: dict(sorted(bits.items()))
                    for family, bits in sorted(record["families"].items())
                },
            }
            for scope, record in sorted(fc_assignments.items())
        },
        "fully_connected_examples": examples,
    }


def audit_artifact(path: str | Path) -> dict[str, Any]:
    """Audit an artifact's observable quantization recipe evidence."""

    artifact = Path(path).expanduser().resolve()
    try:
        package = inspect_litertlm(artifact, inspect_tflite=False)
        from tflite.Model import Model
        builtin_module = importlib.import_module("tflite.BuiltinOperator")
        tensor_module = importlib.import_module("tflite.TensorType")
    except (ImportError, OSError, LiteRTLMInspectionError) as exc:
        raise RecipeAuditError(
            "Recipe audit requires a valid .litertlm and the optional 'tflite' package."
        ) from exc

    tensor_names = _enum_names(
        tensor_module,
        "TensorType",
        {19: "INT2", 20: "UINT4", 21: "FLOAT8_E4M3FN", 22: "FLOAT8_E5M2"},
    )
    operator_names = _enum_names(builtin_module, "BuiltinOperator")
    graphs: list[dict[str, Any]] = []
    with artifact.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            for section in package["sections"]:
                if section["data_type_name"] != "TFLiteModel":
                    continue
                view = memoryview(mapped)[section["begin_offset"] : section["end_offset"]]
                try:
                    graph = _audit_graph(
                        Model.GetRootAsModel(view, 0),
                        tensor_names=tensor_names,
                        operator_names=operator_names,
                    )
                finally:
                    view.release()
                graph["section_index"] = section["index"]
                graph["model_type"] = _model_type(section)
                graphs.append(graph)
        finally:
            mapped.close()

    by_type = {graph["model_type"]: graph for graph in graphs}
    prefill = by_type.get("tf_lite_prefill_decode", {})
    mtp = by_type.get("tf_lite_mtp_drafter", {})
    embedder = by_type.get("tf_lite_embedder", {})
    per_layer = by_type.get("tf_lite_per_layer_embedder", {})
    fc_bits = set(prefill.get("fully_connected_weight_types", {}))
    fc_inputs = set(prefill.get("fully_connected_activation_input_types", {}))
    fc_outputs = set(prefill.get("fully_connected_activation_output_types", {}))
    fc_input_layouts = set(prefill.get("fully_connected_activation_input_layouts", {}))
    fc_output_layouts = set(prefill.get("fully_connected_activation_output_layouts", {}))
    expected_static_a8 = "INT8|scales=1|zero_points=1|axis=0|symmetric=True"
    static_a8_layout = bool(
        (fc_input_layouts | fc_output_layouts)
        and (fc_input_layouts | fc_output_layouts) <= {expected_static_a8}
    )
    observable_mobile_contract = bool(
        {"INT2", "INT4"}.issubset(fc_bits)
        and "INT8" in fc_inputs
        and "INT8" in fc_outputs
        and static_a8_layout
        and embedder.get("embedding_weight_types") == {"INT2": 1}
        and per_layer.get("embedding_weight_types") == {"INT4": 35}
        and "INT4" in mtp.get("fully_connected_weight_types", {})
        and all(not graph.get("custom_operator_codes") for graph in graphs)
    )
    observed_quantized_types = sorted(
        {
            tensor_type
            for graph in graphs
            for tensor_type in graph.get("quantized_tensor_types", {})
        }
    )
    if observable_mobile_contract:
        interpretation = (
            "The artifact exposes a mixed W2/W4/W8 graph with symmetric per-axis "
            "weight scales and INT8 activation edges. This is observably unlike "
            "the public AI Edge Quantizer gemma4_mixed48 W4/W8-afp32 recipe, but "
            "the FlatBuffer does not contain Google's private observers, scales "
            "source, training schedule, or calibration corpus."
        )
    else:
        interpretation = (
            "The artifact does not satisfy the observed Gemma 4 mobile W2/W4/W8-A8 "
            f"contract. Its quantized tensor types are {observed_quantized_types or ['none']}; "
            f"prefill/decode fully-connected weights are {sorted(fc_bits) or ['none']} "
            f"with activation edge types {sorted(fc_inputs | fc_outputs) or ['none']}. "
            "The FlatBuffer does not contain Google's private observers, scales "
            "source, training schedule, or calibration corpus."
        )
    return {
        "artifact": str(artifact),
        "package_version": package["header"]["version"],
        "graphs": graphs,
        "observed_quantized_tensor_types": observed_quantized_types,
        "observable_mobile_contract": observable_mobile_contract,
        "public_gemma4_mixed48_recipe_match": False,
        "exact_private_recipe_recovered": False,
        "private_calibration_recovered": False,
        "training_executed": False,
        "random_weights": False,
        "interpretation": interpretation,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit observable Gemma 4 LiteRT-LM quantization evidence.")
    parser.add_argument("artifact", help="Path to a .litertlm artifact.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--strict-mobile", action="store_true", help="Exit 2 unless observable mobile evidence is present.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = audit_artifact(args.artifact)
    except RecipeAuditError as exc:
        parser.error(str(exc))
        return 2
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.strict_mobile and not report["observable_mobile_contract"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
