"""Audited orchestration of the existing converter for the full-QAT lane only.

There is no alternate converter here: LiteRT Torch still loads, traces,
quantizes and packages the model. Scoped hooks verify the exact loaded state,
all physical constants, and the packaged section bytes before publishing a
success manifest. The official retained-scale LoRA exporter never enters here.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
from typing import Any

from ir_training.common.progress import Progress
from ir_training.export import full_parameter_serialization as serialization
from ir_training.export.gemma4_text_compat import gemma4_text_export_context
from ir_training.export.litertlm_inspector import inspect_litertlm
from ir_training.export.litertlm_mtp import find_model_section

SECTION_TYPES = {
    "target": "tf_lite_prefill_decode",
    "token_embedder": "tf_lite_embedder",
    "per_layer_embedder": "tf_lite_per_layer_embedder",
}
FLOAT_FILES = {
    "model.tflite": "target",
    "embedder.tflite": "token_embedder",
    "per_layer_embedder.tflite": "per_layer_embedder",
}


def check_serialization_dependencies() -> dict[str, str]:
    """Probe the exact physical-schema and quantization APIs before training."""
    required = {"ai-edge-quantizer": "0.9.0", "ai-edge-litert": "2.2.0"}
    for name, expected in required.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Full-parameter serialization requires {name}=={expected}; found {actual}")
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
    if not all(hasattr(schema.TensorType, f"INT{bits}") for bits in (2, 4, 8)):
        raise ValueError("Physical TFLite schema is missing W2/W4/W8 types")
    if not callable(uniform_quantize_tensor.uniform_quantize):
        raise TypeError("Pinned deterministic quantization API is unavailable")
    return required


def validate_serialization_report(report: dict, artifact_sha256: str) -> None:
    """Downstream receipts require the complete proof, not a success flag alone."""
    inventory = report.get("checkpoint_inventory") or {}
    evidence = report.get("evidence") or {}
    loaded = evidence.get("loaded") or {}
    physical = evidence.get("physical") or {}
    package = evidence.get("package") or {}
    floating = physical.get("float_audit") or {}
    if (report.get("verified") is not True or report.get("parameter_count") != 541
            or report.get("policy") != "full_checkpoint_physical_w248_v1"
            or len(inventory) != 541
            or not {"model.embed_tokens.weight", "model.embed_tokens_per_layer.weight"} <= set(inventory)
            or any(part.get("verified") is not True for part in (loaded, physical, floating, package))
            or any(part.get("parameter_count") != 541 for part in (loaded, physical, floating))
            or loaded.get("value_hashes") != {key: value.get("value_sha256") for key, value in inventory.items()}
            or set(physical.get("parameters") or {}) != set(inventory)
            or set(floating.get("mappings") or {}) != set(inventory)
            or set(package.get("sections") or {}) != set(SECTION_TYPES)
            or package.get("artifact_sha256") != artifact_sha256):
        raise ValueError("Full-parameter export lacks complete source-to-package serialization proof")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_packaged_sections(artifact: Path, paths: dict[str, Path]) -> dict:
    """The builder must package the very TFLite bytes that passed the audit."""
    if set(paths) != set(SECTION_TYPES):
        raise ValueError("Full export must bind target and both embedding sections")
    report = inspect_litertlm(artifact, include_hashes=True, inspect_tflite=False)
    models = [s for s in report["sections"] if s.get("data_type_name") == "TFLiteModel"]
    if len(models) != 3 or any(s.get("data_type_name") == "TFLiteWeights" for s in report["sections"]):
        raise ValueError("Full export requires exactly three self-contained model sections; no MTP/extra weights")
    bindings = {}
    for role, model_type in SECTION_TYPES.items():
        section = find_model_section(report, model_type)
        path = paths[role]
        digest = _sha(path)
        if section.get("size") != path.stat().st_size or section.get("sha256") != digest:
            raise ValueError(f"Packaged {role} differs from the physically audited TFLite")
        bindings[role] = {"model_type": model_type, "sha256": digest, "size": path.stat().st_size}
    return {"verified": True, "sections": bindings, "artifact_sha256": _sha(artifact)}


@contextlib.contextmanager
def _audited_converter(model_dir: Path, output_dir: Path, recipe_path: Path, inventory: dict):
    library = importlib.import_module("litert_torch.generative.export_hf.core.export_lib")
    builder = importlib.import_module("litert_torch.generative.export_hf.core.litert_lm_builder")
    original_load = library.load_model
    original_quantize = library.maybe_quantize_model
    original_package = builder.package_model
    state: dict[str, Any] = {"loaded": None, "physical": None, "package": None}
    floats: dict[str, Path] = {}
    quantized: dict[str, Path] = {}
    input_hashes: dict[str, str] = {}
    recipe_sha = _sha(recipe_path)

    def checked_load(model_path, export_config, **kwargs):
        if (Path(model_path).resolve() != model_dir.resolve() or state["loaded"] is not None
                or getattr(export_config, "use_random_weights", False)
                or kwargs.get("trust_remote_code", False) or kwargs.get("auto_model_override") is not None):
            raise ValueError("Full export must load the selected local checkpoint exactly once, without overrides")
        artifacts = original_load(model_path, export_config, **kwargs)
        from ir_training.export.gemma4_text_compat import _require_model
        _require_model(artifacts.model)
        with Progress("Verify every loaded FP32 checkpoint parameter before conversion", unit="stage"):
            state["loaded"] = serialization.verify_loaded_model(artifacts.model, inventory)
        return artifacts

    def checked_quantize(model_path, quantization_recipe=None):
        path = Path(model_path).resolve()
        role = FLOAT_FILES.get(path.name)
        if (state["loaded"] is None or role is None or role in floats
                or not path.is_relative_to(output_dir.resolve())
                or Path(str(quantization_recipe)).resolve() != recipe_path.resolve()
                or _sha(recipe_path) != recipe_sha):
            raise ValueError("Unexpected/mutated full export quantizer input or section")
        floats[role] = path
        input_hashes[role] = _sha(path)
        result = Path(original_quantize(model_path, quantization_recipe)).resolve()
        if result == path or not result.is_file() or not result.is_relative_to(output_dir.resolve()):
            raise ValueError("Full W248 export did not produce a separate quantized section")
        if _sha(path) != input_hashes[role]:
            raise ValueError("Quantization mutated its floating reference graph")
        quantized[role] = result
        return str(result)

    def checked_package(source_artifacts, export_config, exported_artifacts):
        additional = exported_artifacts.additional_model_paths or {}
        if (not exported_artifacts.prefill_decode_model_path or not exported_artifacts.embedder_model_path
                or set(additional) != {"per_layer_embedder"} or not additional["per_layer_embedder"]):
            raise ValueError("Full export package must contain target and both embedding graphs")
        expected_paths = {
            "target": Path(exported_artifacts.prefill_decode_model_path).resolve(),
            "token_embedder": Path(exported_artifacts.embedder_model_path).resolve(),
            "per_layer_embedder": Path(additional["per_layer_embedder"]).resolve(),
        }
        if (expected_paths != quantized or set(floats) != set(SECTION_TYPES)
                or state["package"] is not None or _sha(recipe_path) != recipe_sha
                or any(_sha(path) != input_hashes[role] for role, path in floats.items())):
            raise ValueError("Full export package inputs differ from the checked converter outputs")
        with Progress("Prove all trained parameters in float and W248 physical constants", unit="stage"):
            state["physical"] = serialization.audit_quantized_sections(floats, quantized, inventory)
        if state["physical"].get("verified") is not True:
            raise ValueError("Full checkpoint serialization did not pass")
        audited_hashes = {role: _sha(path) for role, path in quantized.items()}
        result = original_package(source_artifacts, export_config, exported_artifacts)
        if any(_sha(path) != audited_hashes[role] for role, path in quantized.items()):
            raise ValueError("Audited model section changed during packaging")
        artifact = Path(result.litert_lm_model_path)
        state["package"] = verify_packaged_sections(artifact, quantized)
        return result

    library.load_model = checked_load
    library.maybe_quantize_model = checked_quantize
    builder.package_model = checked_package
    try:
        yield state
        if any(state[name] is None for name in ("loaded", "physical", "package")):
            raise ValueError("Converter skipped a mandatory full-parameter serialization gate")
    finally:
        library.load_model = original_load
        library.maybe_quantize_model = original_quantize
        builder.package_model = original_package


def export_full_parameter_checkpoint(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert with complete physical provenance, never with the 205-code patcher."""
    versions = check_serialization_dependencies()
    model_dir, output_dir = Path(kwargs["model"]), Path(kwargs["output_dir"])
    recipe = Path(kwargs["quantization_recipe"])
    with Progress("Index selected full FP32 checkpoint for serialization proof", unit="stage"):
        inventory = serialization.build_checkpoint_inventory(model_dir)
    checked_kwargs = dict(kwargs)
    # Keep separate pre- and post-quantization references for audit/review.
    # Fusing independent weights obscures the one-to-one source proof.
    checked_kwargs.update(keep_temporary_files=True, fuse_gate_up=False, fuse_qkv=False,
                          use_rope_composite=False, experimental_use_mixed_precision=False,
                          experimental_use_fp16=False, use_random_weights=False)
    with (gemma4_text_export_context() as compatibility,
          _audited_converter(model_dir, output_dir, recipe, inventory) as evidence):
        export = importlib.import_module("litert_torch.generative.export_hf.export")
        export.export(**checked_kwargs)
    result = {"verified": True, "policy": "full_checkpoint_physical_w248_v1",
            "parameter_count": len(inventory), "compatibility": compatibility,
            "checkpoint_inventory": inventory, "evidence": evidence,
            "export_kwargs": checked_kwargs, "recipe_sha256": _sha(recipe),
            "serialization_versions": versions, "runtime_tested": False}
    validate_serialization_report(result, evidence["package"]["artifact_sha256"])
    return result
