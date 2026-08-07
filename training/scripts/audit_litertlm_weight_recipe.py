"""Compare a LiteRT-LM INT8 weight section with its source safetensors.

This read-only audit is useful for a locally exported Gemma 3/FunctionGemma
270M model when the merged BF16 safetensors are still available.  It maps the
TFLite ``FULLY_CONNECTED`` and ``EMBEDDING_LOOKUP`` constants back to common
Hugging Face Gemma names, then checks whether the stored INT8 rows equal the
public AI Edge channelwise rule::

    scale[row] = max(abs(float_weight[row])) / 127
    q[row] = round(float_weight[row] / scale[row]), clipped to [-127, 127]

The script does not mutate either input and does not claim that the result is
Google's private QAT recipe.  An exact result identifies the observable public
dynamic INT8 export rule used by the compared artifact only.
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


class WeightRecipeAuditError(RuntimeError):
    """Raised when the source-weight comparison cannot be completed."""


_INT8_TYPE = 9
_LINEAR_RE = re.compile(
    r"Gemma(?:3|4)DecoderLayer_(?P<layer>\d+)"
    r".*?Linear_(?P<name>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _model_type(section: dict[str, Any]) -> str:
    return next(
        (
            str(item.get("value"))
            for item in section.get("items", [])
            if item.get("key") == "model_type"
        ),
        "",
    )


def _section_by_model_type(package: dict[str, Any], model_type: str) -> dict[str, Any]:
    matches = [
        section
        for section in package.get("sections", [])
        if section.get("data_type_name") == "TFLiteModel"
        and _model_type(section) == model_type
    ]
    if not matches:
        available = [
            _model_type(section)
            for section in package.get("sections", [])
            if section.get("data_type_name") == "TFLiteModel"
        ]
        raise WeightRecipeAuditError(
            f"No TFLite model_type={model_type!r} section. Available: {available}"
        )
    return matches[0]


def _schema_model(data: Any) -> Any:
    for module_name, class_name in (
        ("ai_edge_litert.schema_py_generated", "Model"),
        ("tflite.Model", "Model"),
    ):
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            getter = getattr(model_class, "GetRootAsModel", None) or getattr(model_class, "GetRootAs")
            return getter(data, 0)
        except (ImportError, AttributeError, TypeError, ValueError):
            continue
    raise WeightRecipeAuditError(
        "Install ai-edge-litert or the generated 'tflite' schema bindings."
    )


def _operator_names() -> dict[int, str]:
    module = importlib.import_module("tflite.BuiltinOperator")
    cls = getattr(module, "BuiltinOperator")
    return {
        int(value): name
        for name, value in vars(cls).items()
        if not name.startswith("_") and isinstance(value, int)
    }


def _buffer_view(model: Any, buffer_index: int, section: Any) -> bytes:
    buffer = model.Buffers(buffer_index)
    data_length = int(getattr(buffer, "DataLength", lambda: 0)() or 0)
    if data_length:
        data = buffer.DataAsNumpy()
        if data is not None and len(data) == data_length:
            return bytes(data)
    offset = int(getattr(buffer, "Offset", lambda: 0)() or 0)
    size = int(getattr(buffer, "Size", lambda: 0)() or 0)
    if offset <= 0 or size <= 0 or offset + size > len(section):
        raise WeightRecipeAuditError(
            f"Buffer {buffer_index} has no readable inline or external storage."
        )
    return bytes(section[offset : offset + size])


def canonical_source_key(operator_name: str, tensor_name: str, shape: tuple[int, ...]) -> str | None:
    """Map common LiteRT-Torch Gemma names to Hugging Face safetensor keys."""

    match = _LINEAR_RE.search(tensor_name)
    if match:
        layer = match.group("layer")
        name = match.group("name")
        if name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            return f"model.layers.{layer}.self_attn.{name}.weight"
        return f"model.layers.{layer}.mlp.{name}.weight"
    if operator_name == "EMBEDDING_LOOKUP" and len(shape) == 2:
        return "model.embed_tokens.weight"
    return None


def _source_tensor(source: Any, key: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise WeightRecipeAuditError("The weight audit requires PyTorch.") from exc
    tensor = source.get_tensor(key)
    return tensor.detach().to(dtype=torch.float32, device="cpu").numpy()


def _expected_int8(float_weight: Any) -> tuple[Any, Any]:
    import numpy as np

    weight = np.asarray(float_weight, dtype=np.float32)
    if weight.ndim < 2:
        raise WeightRecipeAuditError(f"Expected a rank-2 weight, got {weight.shape}.")
    maxima = np.max(np.abs(weight), axis=tuple(range(1, weight.ndim)))
    scales = (maxima / np.float32(127.0)).astype(np.float32, copy=False)
    safe_scales = np.where(scales == 0, np.float32(1.0), scales)
    quantized = np.rint(weight / safe_scales.reshape((-1,) + (1,) * (weight.ndim - 1)))
    quantized = np.clip(quantized, -127, 127).astype(np.int8, copy=False)
    return quantized, scales


def audit_artifact(
    artifact: str | Path,
    source_model: str | Path,
    *,
    model_type: str = "tf_lite_prefill_decode",
) -> dict[str, Any]:
    """Audit one TFLite section against a safetensors source model."""

    import numpy as np

    artifact_path = Path(artifact).expanduser().resolve()
    source_path = Path(source_model).expanduser().resolve()
    if not artifact_path.is_file():
        raise WeightRecipeAuditError(f"LiteRT-LM artifact does not exist: {artifact_path}")
    if source_path.is_dir():
        source_path = source_path / "model.safetensors"
    if not source_path.is_file():
        raise WeightRecipeAuditError(f"Source safetensors do not exist: {source_path}")

    try:
        package = inspect_litertlm(artifact_path, inspect_tflite=False)
        section = _section_by_model_type(package, model_type)
        from safetensors import safe_open
        from tflite.Model import Model
    except (ImportError, OSError, LiteRTLMInspectionError) as exc:
        raise WeightRecipeAuditError(
            "Weight recipe audit requires safetensors and generated TFLite bindings."
        ) from exc

    op_names = _operator_names()
    model_type_seen = _model_type(section)
    records: list[dict[str, Any]] = []
    skipped = collections.Counter()
    seen_buffers: set[int] = set()
    with artifact_path.open("rb") as handle, safe_open(str(source_path), framework="pt") as source:
        source_keys = set(source.keys())
        mapped = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            section_view = memoryview(mapped)[section["begin_offset"] : section["end_offset"]]
            model = Model.GetRootAsModel(section_view, 0)
            for subgraph_index in range(int(model.SubgraphsLength() or 0)):
                subgraph = model.Subgraphs(subgraph_index)
                for operator_index in range(int(subgraph.OperatorsLength() or 0)):
                    operator = subgraph.Operators(operator_index)
                    code = model.OperatorCodes(operator.OpcodeIndex())
                    operator_name = op_names.get(int(code.BuiltinCode()), "")
                    if operator_name not in {"FULLY_CONNECTED", "EMBEDDING_LOOKUP"}:
                        continue
                    if int(operator.InputsLength() or 0) < 2:
                        skipped["operator_without_weight_input"] += 1
                        continue
                    tensor = subgraph.Tensors(operator.Inputs(1))
                    buffer_index = int(tensor.Buffer())
                    if buffer_index in seen_buffers:
                        continue
                    seen_buffers.add(buffer_index)
                    type_value = int(tensor.Type())
                    shape = tuple(int(tensor.Shape(i)) for i in range(int(tensor.ShapeLength() or 0)))
                    if type_value != _INT8_TYPE:
                        skipped["non_int8_weight"] += 1
                        continue
                    quantization = tensor.Quantization()
                    if quantization is None:
                        skipped["missing_quantization"] += 1
                        continue
                    scale_count = int(quantization.ScaleLength() or 0)
                    axis = int(quantization.QuantizedDimension())
                    if axis != 0 or len(shape) < 2 or scale_count != shape[0]:
                        skipped["unsupported_scale_layout"] += 1
                        continue
                    zero_points = [int(quantization.ZeroPoint(i)) for i in range(int(quantization.ZeroPointLength() or 0))]
                    raw = np.frombuffer(_buffer_view(model, buffer_index, section_view), dtype=np.int8)
                    expected_count = int(np.prod(shape, dtype=np.int64))
                    if raw.size != expected_count:
                        skipped["unexpected_weight_size"] += 1
                        continue
                    raw = raw.reshape(shape).copy()
                    stored_scales = np.asarray(
                        [quantization.Scale(i) for i in range(scale_count)], dtype=np.float32
                    )
                    tensor_name = _decode(tensor.Name())
                    key = canonical_source_key(operator_name, tensor_name, shape)
                    if key is None:
                        skipped["unmapped_tensor_name"] += 1
                        continue
                    if key not in source_keys:
                        skipped["missing_source_tensor"] += 1
                        continue
                    float_weight = _source_tensor(source, key)
                    transposed = False
                    if tuple(float_weight.shape) != shape:
                        if float_weight.ndim == 2 and tuple(float_weight.T.shape) == shape:
                            float_weight = float_weight.T
                            transposed = True
                        else:
                            skipped["source_shape_mismatch"] += 1
                            continue
                    expected_q, expected_scales = _expected_int8(float_weight)
                    scale_abs_error = float(np.max(np.abs(stored_scales - expected_scales)))
                    scale_exact = bool(np.array_equal(stored_scales, expected_scales))
                    q_exact = bool(np.array_equal(raw, expected_q))
                    records.append(
                        {
                            "operator": operator_name,
                            "subgraph": subgraph_index,
                            "operator_index": operator_index,
                            "tensor_name": tensor_name,
                            "source_key": key,
                            "shape": list(shape),
                            "buffer": buffer_index,
                            "transposed_source": transposed,
                            "scale_count": scale_count,
                            "zero_points_all_zero": bool(zero_points) and all(value == 0 for value in zero_points),
                            "scale_exact": scale_exact,
                            "scale_max_abs_error": scale_abs_error,
                            "quantized_values_exact": q_exact,
                        }
                    )
        finally:
            section_view.release()
            mapped.close()

    exact_scale_count = sum(1 for record in records if record["scale_exact"])
    exact_value_count = sum(1 for record in records if record["quantized_values_exact"])
    all_exact = bool(records) and exact_scale_count == len(records) and exact_value_count == len(records)
    return {
        "artifact": str(artifact_path),
        "source_model": str(source_path),
        "model_type": model_type_seen,
        "source_weight_comparison": {
            "matched_weight_count": len(records),
            "exact_scale_count": exact_scale_count,
            "exact_quantized_value_count": exact_value_count,
            "all_matched_weights_exact": all_exact,
            "records_preview": records[:12],
        },
        "skipped": dict(sorted(skipped.items())),
        "observable_recipe": (
            "dynamic_wi8_afp32 (public alias of channelwise dynamic_wi8c_afp32): "
            "axis-0 INT8 weights with scale=max(abs(source_row))/127 and float32 "
            "activation edges."
            if all_exact
            else "No exact public INT8 source-weight rule was proven for every matched weight."
        ),
        "private_qat_recipe_recovered": False,
        "training_executed": False,
        "exact_official_model_match": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a LiteRT-LM INT8 section with source safetensors."
    )
    parser.add_argument("artifact", help="Path to the .litertlm artifact.")
    parser.add_argument(
        "--source-model",
        required=True,
        help="Merged Hugging Face directory or model.safetensors file.",
    )
    parser.add_argument("--model-type", default="tf_lite_prefill_decode")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero unless every matched weight has exact scales and values.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = audit_artifact(args.artifact, args.source_model, model_type=args.model_type)
    except (OSError, WeightRecipeAuditError) as exc:
        parser.error(str(exc))
        return 2
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.strict and not report["source_weight_comparison"]["all_matched_weights_exact"]:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
