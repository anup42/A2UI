"""Build a random static W2/W4/W8-A8 graph and compare its observable layout.

This is a converter experiment, not a training or Google-recipe recreation
command.  It builds a tiny deterministic Keras graph whose layer names model
the observable Gemma 4 mobile assignments, calibrates it with deterministic
random inputs, and applies a custom AI Edge Quantizer recipe:

* default fully-connected layers: symmetric channelwise W4;
* language ``layer_15_mlp`` through ``layer_34_mlp``: symmetric channelwise W2;
* ``per_layer_projection``: symmetric channelwise W8;
* all fully-connected activations: symmetric static A8.

The public quantizer policy does not advertise arbitrary static W2 fully
connected operations, so the recipe sets ``skip_checks`` explicitly.  The
result is useful for testing FlatBuffer dtypes, scale layout, and integer
activation edges only.  It must not be presented as Google's private mobile
converter, calibration corpus, QAT schedule, or learned checkpoint.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class MobileContractParityError(RuntimeError):
    """Raised when the opt-in synthetic conversion cannot run."""


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    return {
        "execute": bool(args.execute),
        "training_executed": False,
        "random_initialization": True,
        "learned_weight_equivalence": False,
        "private_recipe_recovered": False,
        "calibration_source": "deterministic random normal inputs",
        "unsafe_policy_override": True,
        "output_dir": str(output_dir),
        "fp32_tflite": str(output_dir / "random_fp32.tflite"),
        "quantized_tflite": str(output_dir / "random_mobile_contract.tflite"),
        "report": str(output_dir / "random_mobile_contract_report.json"),
        "seed": int(args.seed),
        "calibration_samples": int(args.calibration_samples),
        "threads": int(args.threads),
        "official_artifact": (
            str(Path(args.official_artifact).expanduser().resolve())
            if args.official_artifact
            else None
        ),
        "observable_contract": {
            "weight_bits": [2, 4, 8],
            "activation_bits": 8,
            "weight_granularity": "CHANNELWISE",
            "weight_quantized_dimension": 0,
            "symmetric_zero_points": True,
            "default_fc_bits": 4,
            "low_bit_language_mlp_layers": "15-34",
            "layer_15_mlp_bits": 2,
            "per_layer_projection_bits": 8,
        },
    }


def _mobile_recipe() -> list[dict[str, Any]]:
    """Return the observable-contract synthetic recipe.

    The order is intentional: the default rule is followed by narrower regex
    overrides.  ``skip_checks`` is explicit because the public AI Edge policy
    rejects arbitrary static W2 FULLY_CONNECTED configurations.
    """

    def entry(regex: str, bits: int) -> dict[str, Any]:
        return {
            "regex": regex,
            "operation": "FULLY_CONNECTED",
            "algorithm_key": "min_max_uniform_quantize",
            "op_config": {
                "activation_tensor_config": {
                    "num_bits": 8,
                    "symmetric": True,
                    "granularity": "TENSORWISE",
                    "dtype": "INT",
                },
                "weight_tensor_config": {
                    "num_bits": bits,
                    "symmetric": True,
                    "granularity": "CHANNELWISE",
                    "dtype": "INT",
                },
                "compute_precision": "INTEGER",
                "explicit_dequantize": False,
                "skip_checks": True,
                "min_weight_elements": 0,
            },
        }

    return [
        entry(".*", 4),
        entry(r"layer_(?:1[5-9]|2[0-9]|3[0-4])_mlp", 2),
        entry("per_layer_projection", 8),
    ]


def _build_random_model(seed: int) -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - converter environment only
        raise MobileContractParityError("--execute requires TensorFlow.") from exc

    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(8,), batch_size=1, name="input")
    hidden = inputs
    for layer_index in range(35):
        hidden = tf.keras.layers.Dense(
            16,
            use_bias=False,
            name=f"layer_{layer_index}_mlp",
        )(hidden)
    adapter = tf.keras.layers.Dense(10, use_bias=False, name="per_layer_projection")(hidden)
    outputs = tf.keras.layers.Dense(4, use_bias=False, name="output")(adapter)
    model = tf.keras.Model(inputs, outputs)
    for layer_index, layer in enumerate(model.layers):
        if not getattr(layer, "weights", None):
            continue
        shape = tuple(int(value) for value in layer.weights[0].shape)
        values = tf.random.stateless_normal(shape, seed=[seed, layer_index + 1])
        layer.set_weights([values.numpy()])
    return model


def _convert_fp32(model: Any, output: Path) -> None:
    try:
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        output.write_bytes(converter.convert())
    except Exception as exc:  # pragma: no cover - TensorFlow/tool version dependent
        raise MobileContractParityError(f"TensorFlow to TFLite conversion failed: {exc}") from exc


def _quantize(fp32_path: Path, output: Path, *, calibration_samples: int, threads: int) -> None:
    try:
        from ai_edge_quantizer import quantizer
        from ai_edge_quantizer.utils import tfl_interpreter_utils
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise MobileContractParityError(
            "--execute requires ai-edge-quantizer, ai-edge-litert, and their runtime dependencies."
        ) from exc
    try:
        quantizer_instance = quantizer.Quantizer(str(fp32_path))
        quantizer_instance.load_quantization_recipe(_mobile_recipe())
        calibration_data = tfl_interpreter_utils.create_random_normal_input_data(
            quantizer_instance._float_model_buffer,  # pylint: disable=protected-access
            num_samples=calibration_samples,
        )
        calibration_result = quantizer_instance.calibrate(calibration_data, num_threads=threads)
        result = quantizer_instance.quantize(calibration_result, enable_progress_report=False)
        output.write_bytes(result.quantized_model)
    except Exception as exc:  # pragma: no cover - quantizer/tool version dependent
        raise MobileContractParityError(f"Static mobile-contract quantization failed: {exc}") from exc


def _enum_names(module: Any, class_name: str, fallbacks: dict[int, str] | None = None) -> dict[int, str]:
    cls = getattr(module, class_name, None)
    names: dict[int, str] = {}
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


def _quantization(tensor: Any) -> dict[str, Any] | None:
    quant = tensor.Quantization()
    if quant is None:
        return None
    scale_count = int(quant.ScaleLength() or 0)
    zero_count = int(quant.ZeroPointLength() or 0)
    if not scale_count and not zero_count:
        return None
    zero_points = [int(quant.ZeroPoint(index)) for index in range(zero_count)]
    return {
        "scale_count": scale_count,
        "zero_point_count": zero_count,
        "quantized_dimension": int(quant.QuantizedDimension()),
        "zero_points_all_zero": bool(zero_points) and all(value == 0 for value in zero_points),
    }


def _inspect_tflite(path: Path) -> dict[str, Any]:
    try:
        model_module = importlib.import_module("tflite.Model")
        op_module = importlib.import_module("tflite.BuiltinOperator")
        tensor_module = importlib.import_module("tflite.TensorType")
    except ImportError as exc:  # pragma: no cover - converter environment only
        raise MobileContractParityError("Inspection requires the optional 'tflite' schema bindings.") from exc

    model = model_module.Model.GetRootAsModel(path.read_bytes(), 0)
    op_names = _enum_names(op_module, "BuiltinOperator")
    tensor_names = _enum_names(
        tensor_module,
        "TensorType",
        {19: "INT2", 20: "UINT4", 21: "FLOAT8_E4M3FN", 22: "FLOAT8_E5M2"},
    )
    operator_codes = [
        int(model.OperatorCodes(index).BuiltinCode())
        for index in range(int(model.OperatorCodesLength() or 0))
    ]
    weight_types = collections.Counter()
    activation_inputs = collections.Counter()
    activation_outputs = collections.Counter()
    layouts = collections.Counter()
    examples: list[dict[str, Any]] = []
    fc_count = 0
    for subgraph_index in range(int(model.SubgraphsLength() or 0)):
        subgraph = model.Subgraphs(subgraph_index)
        for operator_index in range(int(subgraph.OperatorsLength() or 0)):
            operator = subgraph.Operators(operator_index)
            opcode_index = int(operator.OpcodeIndex())
            if opcode_index >= len(operator_codes):
                continue
            if op_names.get(operator_codes[opcode_index], "") != "FULLY_CONNECTED":
                continue
            inputs = _vector(operator, "InputsLength", "Inputs")
            outputs = _vector(operator, "OutputsLength", "Outputs")
            if len(inputs) < 2 or inputs[1] < 0:
                continue
            weight = subgraph.Tensors(inputs[1])
            weight_type = tensor_names.get(int(weight.Type()), f"unknown:{int(weight.Type())}")
            weight_quant = _quantization(weight)
            if weight_quant is None:
                continue
            fc_count += 1
            weight_types[weight_type] += 1
            layouts[
                f"{weight_type}|scales={weight_quant['scale_count']}|"
                f"zero_points={weight_quant['zero_point_count']}|"
                f"axis={weight_quant['quantized_dimension']}|"
                f"symmetric={weight_quant['zero_points_all_zero']}"
            ] += 1

            def tensor_type(index: int) -> tuple[str, dict[str, Any] | None] | None:
                if index < 0 or index >= int(subgraph.TensorsLength() or 0):
                    return None
                tensor = subgraph.Tensors(index)
                return tensor_names.get(int(tensor.Type()), f"unknown:{int(tensor.Type())}"), _quantization(tensor)

            input_record = tensor_type(inputs[0])
            output_record = tensor_type(outputs[0]) if outputs else None
            if input_record:
                activation_inputs[input_record[0]] += 1
            if output_record:
                activation_outputs[output_record[0]] += 1
            if len(examples) < 16:
                examples.append(
                    {
                        "weight_type": weight_type,
                        "weight_shape": [int(weight.Shape(index)) for index in range(int(weight.ShapeLength() or 0))],
                        "weight_name": _decode(weight.Name()),
                        "input_type": input_record[0] if input_record else None,
                        "input_quantization": input_record[1] if input_record else None,
                        "output_type": output_record[0] if output_record else None,
                        "output_quantization": output_record[1] if output_record else None,
                    }
                )

    expected_weight_layout = all(
        "axis=0" in key and "symmetric=True" in key for key in layouts
    )
    expected_activation_layout = all(
        record.get("input_quantization", {}).get("scale_count") == 1
        and record.get("input_quantization", {}).get("zero_point_count") == 1
        and record.get("input_quantization", {}).get("zero_points_all_zero") is True
        and record.get("output_quantization", {}).get("scale_count") == 1
        and record.get("output_quantization", {}).get("zero_point_count") == 1
        and record.get("output_quantization", {}).get("zero_points_all_zero") is True
        for record in examples
        if record.get("input_type") == "INT8" and record.get("output_type") == "INT8"
    )
    observed_contract = bool(
        fc_count > 0
        and {"INT2", "INT4", "INT8"}.issubset(weight_types)
        and activation_inputs == {"INT8": fc_count}
        and activation_outputs == {"INT8": fc_count}
        and expected_weight_layout
        and expected_activation_layout
    )
    return {
        "subgraph_count": int(model.SubgraphsLength() or 0),
        "fully_connected_count": fc_count,
        "fully_connected_weight_types": dict(sorted(weight_types.items())),
        "fully_connected_activation_input_types": dict(sorted(activation_inputs.items())),
        "fully_connected_activation_output_types": dict(sorted(activation_outputs.items())),
        "quantization_layout": dict(sorted(layouts.items())),
        "fully_connected_examples": examples,
        "observable_contract": observed_contract,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _plan(args)
    output_dir = Path(plan["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise MobileContractParityError(
            f"Output directory is non-empty: {output_dir}. Choose a new path or pass --force; no files were removed."
        )
    planned_paths = [
        Path(plan["fp32_tflite"]),
        Path(plan["quantized_tflite"]),
        Path(plan["report"]),
    ]
    if any(path.exists() for path in planned_paths):
        raise MobileContractParityError("Refusing to overwrite an existing planned output; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "random_mobile_contract_plan.json").write_text(
        json.dumps({**plan, "recipe": _mobile_recipe()}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.execute:
        return plan

    model = _build_random_model(args.seed)
    fp32_path = Path(plan["fp32_tflite"])
    quantized_path = Path(plan["quantized_tflite"])
    _convert_fp32(model, fp32_path)
    _quantize(
        fp32_path,
        quantized_path,
        calibration_samples=args.calibration_samples,
        threads=args.threads,
    )
    observed = _inspect_tflite(quantized_path)
    result: dict[str, Any] = {
        **plan,
        "fp32_size": fp32_path.stat().st_size,
        "quantized_size": quantized_path.stat().st_size,
        "recipe": _mobile_recipe(),
        "synthetic_observation": observed,
        "synthetic_contract_match": bool(observed["observable_contract"]),
        "exact_official_mobile_reproduction": False,
        "reason": (
            "The random graph proves that a custom static W2/W4/W8-A8 layout can "
            "be serialized by the installed AI Edge Quantizer when policy checks "
            "are explicitly bypassed. It has unrelated topology and random weights, "
            "so it cannot recover Google's private recipe or checkpoint."
        ),
    }
    if args.official_artifact:
        try:
            from audit_litertlm_recipe import audit_artifact

            result["official_artifact_observation"] = audit_artifact(args.official_artifact)
        except Exception as exc:  # pragma: no cover - optional comparison path
            result["official_artifact_observation_error"] = str(exc)
    Path(plan["report"]).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and inspect a random static W2/W4/W8-A8 TFLite contract graph."
    )
    parser.add_argument("--output-dir", required=True, help="New/empty directory for graph and report.")
    parser.add_argument("--official-artifact", help="Optional official .litertlm for independent audit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-samples", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Build, calibrate, and quantize the random graph.")
    parser.add_argument("--force", action="store_true", help="Allow an existing directory; never overwrite planned files.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.calibration_samples < 1 or args.threads < 1:
        parser.error("--calibration-samples and --threads must be positive")
    try:
        result = run(args)
    except (OSError, MobileContractParityError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
