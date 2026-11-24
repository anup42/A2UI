"""Package the 270M QAT W8 text graph with its FP32 GPU activation preference."""
from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from unittest.mock import patch

from ir_training.export.litertlm_mtp import find_model_section


def probe_fp32_text_builder() -> dict:
    """Verify the prefill/decode binding and installed packager before training."""
    import litert_lm_builder as builder

    packager = importlib.import_module("litert_torch.generative.export_hf.core.litert_lm_builder")
    model_type = builder.TfLiteModelType.PREFILL_DECODE
    if getattr(packager, "litertlm_builder", None) is not builder or model_type.value != "tf_lite_prefill_decode":
        raise ValueError("Unsupported 270M W8 LiteRT Torch package builder binding.")
    inspect.signature(builder.LitertLmFileBuilder.add_tflite_model).bind(
        None, "model.tflite", model_type, prefer_activation_type="fp32")
    return {"litert_torch": importlib.metadata.version("litert-torch"),
            "litert_lm_builder": importlib.metadata.version("litert-lm-builder"),
            "model_type": model_type.value, "prefer_activation_type": "fp32"}


def export_with_fp32_text_activations(export, settings: dict) -> None:
    """Use the public builder option missing from LiteRT Torch 0.9.4's export API."""
    import litert_lm_builder as builder

    if settings["quantization_recipe"] != "dynamic_wi8_afp32" or settings["experimental_use_fp16"] is not False:
        raise ValueError("The 270M FP32 GPU activation preference requires W8/AFP32 conversion.")

    class FP32TextBuilder(builder.LitertLmFileBuilder):
        def add_tflite_model(self, tflite_model_path, model_type, **options):
            if model_type == builder.TfLiteModelType.PREFILL_DECODE:
                if options.get("prefer_activation_type") not in (None, "fp32"):
                    raise ValueError("Converter requested conflicting text activation precision.")
                options["prefer_activation_type"] = "fp32"
            return super().add_tflite_model(tflite_model_path, model_type, **options)

    # Conversion runs in an isolated worker. Delegate all graph conversion and
    # serialization to the pinned exporter and restore the builder on failure.
    with patch.object(builder, "LitertLmFileBuilder", FP32TextBuilder):
        export(**settings)


def verify_fp32_text_activations(inspection: dict) -> None:
    section = find_model_section(inspection, "tf_lite_prefill_decode")
    preferences = [entry for entry in section["items"] if entry["key"] == "prefer_activation_type"]
    if preferences != [{"key": "prefer_activation_type", "type": "StringValue", "value": "fp32"}]:
        raise ValueError("270M W8 package must declare one FP32 text activation preference.")
