"""Build and quantize a tiny random graph with a public AI Edge recipe.

This is a converter smoke test, not a Gemma training command.  It creates a
small deterministic Keras graph with three fully-connected layers, applies the
public ``ai_edge_quantizer`` Gemma 4 recipe, and checks the resulting TFLite
weight dtypes/scale layout.  The middle layer is named ``per_layer`` so the
published Gemma 4 recipe's W8 override is exercised.

The experiment is intentionally separate from exact mobile-artifact parity:
the public ``gemma4_mixed48`` recipe is W4/W8 channelwise with float32 graph
activations, whereas the official mobile artifact observed locally is W2/W4/W8
with INT8 activation edges.  Random weights can validate converter mechanics,
not Google's learned weights or private calibration.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class RandomTFLiteRecipeError(RuntimeError):
    """Raised when the opt-in synthetic conversion cannot run."""


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


def _vector(obj: Any, length_method: str, item_method: str) -> list[int]:
    length = getattr(obj, length_method, lambda: 0)() or 0
    method = getattr(obj, item_method, None)
    if method is None:
        return []
    return [int(method(index)) for index in range(int(length))]


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    return {
        "execute": bool(args.execute),
        "training_executed": False,
        "random_initialization": True,
        "learned_weight_equivalence": False,
        "recipe": args.recipe,
        "output_dir": str(output_dir),
        "fp32_tflite": str(output_dir / "random_fp32.tflite"),
        "quantized_tflite": str(output_dir / "random_quantized.tflite"),
        "seed": args.seed,
        "official_artifact": str(Path(args.official_artifact).expanduser().resolve()) if args.official_artifact else None,
        "graph": {
            "input_features": 8,
            "hidden_features": 16,
            "per_layer_features": 12,
            "output_features": 4,
            "fully_connected_recipe_scope": "per_layer",
        },
        "expected_public_layout": {
            "default_fully_connected_weight_type": "INT4",
            "per_layer_fully_connected_weight_type": "INT8",
            "weight_granularity": "CHANNELWISE",
            "symmetric_zero_points": True,
        },
    }


def _build_random_model(seed: int):
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise RandomTFLiteRecipeError("--execute requires TensorFlow.") from exc

    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(8,), batch_size=1, name="input")
    hidden = tf.keras.layers.Dense(16, use_bias=False, name="dense")(inputs)
    per_layer = tf.keras.layers.Dense(12, use_bias=False, name="per_layer_projection")(hidden)
    outputs = tf.keras.layers.Dense(4, use_bias=False, name="output")(per_layer)
    model = tf.keras.Model(inputs, outputs)
    for layer in model.layers:
        if not getattr(layer, "weights", None):
            continue
        shape = tuple(int(value) for value in layer.weights[0].shape)
        layer.set_weights([tf.random.stateless_normal(shape, seed=[seed, len(shape)]) .numpy()])
    return model


def _convert_fp32(model: Any, output: Path) -> None:
    try:
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        data = converter.convert()
    except Exception as exc:  # pragma: no cover - TensorFlow/tool version dependent
        raise RandomTFLiteRecipeError(f"TensorFlow to TFLite conversion failed: {exc}") from exc
    output.write_bytes(data)


def _quantize(fp32_path: Path, output: Path, recipe_name: str) -> None:
    try:
        from ai_edge_quantizer import quantizer, recipe
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise RandomTFLiteRecipeError(
            "--execute requires ai-edge-quantizer and its ai-edge-litert dependency."
        ) from exc
    recipe_factory = getattr(recipe, recipe_name, None)
    if recipe_factory is None:
        raise RandomTFLiteRecipeError(f"Unknown ai-edge-quantizer recipe: {recipe_name}")
    try:
        quantizer_instance = quantizer.Quantizer(str(fp32_path))
        quantizer_instance.load_quantization_recipe(recipe_factory()["tf_lite_prefill_decode"])
        quantizer_instance.quantize().export_model(str(output), overwrite=False)
    except Exception as exc:  # pragma: no cover - quantizer/tool version dependent
        raise RandomTFLiteRecipeError(f"AI Edge quantization failed: {exc}") from exc


def _inspect_tflite(path: Path) -> dict[str, Any]:
    try:
        model_module = importlib.import_module("tflite.Model")
        op_module = importlib.import_module("tflite.BuiltinOperator")
        tensor_module = importlib.import_module("tflite.TensorType")
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise RandomTFLiteRecipeError("Inspection requires the optional 'tflite' package.") from exc
    model = model_module.Model.GetRootAsModel(path.read_bytes(), 0)
    op_names = _enum_names(op_module, "BuiltinOperator")
    tensor_names = _enum_names(
        tensor_module,
        "TensorType",
        {19: "INT2", 20: "UINT4", 21: "FLOAT8_E4M3FN", 22: "FLOAT8_E5M2"},
    )
    subgraph = model.Subgraphs(0)
    operator_codes = [
        int(model.OperatorCodes(index).BuiltinCode())
        for index in range(int(model.OperatorCodesLength() or 0))
    ]
    weight_types = collections.Counter()
    weight_layouts = collections.Counter()
    fully_connected = []
    for index in range(int(subgraph.OperatorsLength() or 0)):
        operator = subgraph.Operators(index)
        op_code = operator_codes[int(operator.OpcodeIndex())]
        op_name = op_names.get(op_code, f"unknown:{op_code}")
        if op_name != "FULLY_CONNECTED":
            continue
        inputs = _vector(operator, "InputsLength", "Inputs")
        if len(inputs) < 2 or inputs[1] < 0:
            continue
        weight = subgraph.Tensors(inputs[1])
        type_value = int(weight.Type())
        type_name = tensor_names.get(type_value, f"unknown:{type_value}")
        quant = weight.Quantization()
        scale_count = int(quant.ScaleLength() or 0) if quant else 0
        zero_count = int(quant.ZeroPointLength() or 0) if quant else 0
        zero_points = [int(quant.ZeroPoint(i)) for i in range(zero_count)] if quant else []
        name = _decode(weight.Name())
        if scale_count:
            weight_types[type_name] += 1
            weight_layouts[
                f"{type_name}|scales={scale_count}|zero_points={zero_count}|"
                f"axis={int(quant.QuantizedDimension())}|symmetric={bool(zero_points) and all(v == 0 for v in zero_points)}"
            ] += 1
        fully_connected.append({"weight_type": type_name, "weight_name": name})
    return {
        "operator_count": int(subgraph.OperatorsLength() or 0),
        "fully_connected_weight_types": dict(sorted(weight_types.items())),
        "quantization_layout": dict(sorted(weight_layouts.items())),
        "fully_connected": fully_connected,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _plan(args)
    output_dir = Path(plan["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise RandomTFLiteRecipeError(
            f"Output directory is non-empty: {output_dir}. Choose a new path or pass --force; no files were removed."
        )
    for planned in (Path(plan["fp32_tflite"]), Path(plan["quantized_tflite"])):
        if planned.exists():
            raise RandomTFLiteRecipeError(f"Refusing to overwrite existing output: {planned}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "random_tflite_recipe_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.execute:
        return plan

    model = _build_random_model(args.seed)
    fp32_path = Path(plan["fp32_tflite"])
    quantized_path = Path(plan["quantized_tflite"])
    _convert_fp32(model, fp32_path)
    _quantize(fp32_path, quantized_path, args.recipe)
    observed = _inspect_tflite(quantized_path)
    fully_connected = observed["fully_connected"]
    per_layer = [item for item in fully_connected if "per_layer" in item["weight_name"]]
    other = [item for item in fully_connected if "per_layer" not in item["weight_name"]]
    public_layout_match = bool(
        per_layer
        and all(item["weight_type"] == "INT8" for item in per_layer)
        and other
        and all(item["weight_type"] == "INT4" for item in other)
        and all("axis=0" in key and "symmetric=True" in key for key in observed["quantization_layout"])
    )
    result = {
        **plan,
        "fp32_size": fp32_path.stat().st_size,
        "quantized_size": quantized_path.stat().st_size,
        "observed": observed,
        "public_recipe_layout_match": public_layout_match,
        "exact_official_mobile_reproduction": False,
        "reason": (
            "This deterministic graph only proves that the public AI Edge "
            "recipe applies channelwise INT4 by default and INT8 to a "
            "per_layer scope. It has unrelated topology and random weights; "
            "it cannot reproduce the official Gemma 4 mobile artifact."
        ),
    }
    if args.official_artifact:
        try:
            from audit_litertlm_recipe import audit_artifact
            result["official_artifact_observation"] = audit_artifact(args.official_artifact)
        except Exception as exc:  # pragma: no cover - optional comparison path
            result["official_artifact_observation_error"] = str(exc)
    (output_dir / "random_tflite_recipe_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a random TFLite graph and audit a public AI Edge quantization recipe.")
    parser.add_argument("--output-dir", required=True, help="New/empty directory for the graph and report.")
    parser.add_argument("--recipe", default="gemma4_mixed48", help="Named ai-edge-quantizer recipe (default: gemma4_mixed48).")
    parser.add_argument("--official-artifact", help="Optional .litertlm artifact to include as a separate observable audit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true", help="Actually build and quantize the graph; default is plan-only.")
    parser.add_argument("--force", action="store_true", help="Allow an existing non-empty root; planned files are still never overwritten.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run(args)
    except (OSError, RandomTFLiteRecipeError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
