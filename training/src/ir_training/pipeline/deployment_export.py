"""Export the current dense review-training checkpoint, never an official graph transplant.

Each converter invocation belongs in an isolated CPU export environment.  CUDA
training and LiteRT-LM GPU inference are separate stages, not conversion claims.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ir_training.common.progress import Progress, log
from ir_training.train.resume_contract import resolve_export_training_lineage

PROFILES = {"e2b": {"gemma4", "gemma4_text"}, "270m": {"gemma3", "gemma3_text"}}
VARIANTS = ("w32", "w16", "w8", "w4")
W16_RECIPE = "weight_only_fp16"
W16_RECIPE_PATH = Path(__file__).resolve().parents[3] / "configs/export/weight_only_fp16.json"
E2B_W4_RECIPE = "gemma4_mixed48_b32"
E2B_W4_RECIPE_PATH = Path(__file__).resolve().parents[3] / "configs/export/gemma4_mixed48_b32_flat.json"


def deployment_variants(profile: str) -> dict[str, dict[str, Any]]:
    if profile not in PROFILES:
        raise ValueError("Deployment profile must be e2b or 270m")
    return {
        "w32": {"weight_bits": 32, "kind": "fp32", "recipe": "none", "experimental": False},
        "w16": {"weight_bits": 16, "kind": "fp16", "recipe": W16_RECIPE, "experimental": True},
        "w8": {"weight_bits": 8, "kind": "int8", "recipe": "dynamic_wi8_afp32", "experimental": False},
        "w4": {"weight_bits": 4, "kind": "mixed_w4_w8" if profile == "e2b" else "int4_block32",
               "recipe": E2B_W4_RECIPE if profile == "e2b" else "dynamic_wi4b32_afp32", "experimental": True},
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _local_file(directory: Path, name: str, *, allow_bound_symlink: bool = False) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Missing or escaping bound file {name!r} in {directory}")
    result = (directory / name).resolve()
    if (not allow_bound_symlink and not result.is_relative_to(directory.resolve())) or not result.is_file():
        raise ValueError(f"Missing or escaping bound file {name!r} in {directory}")
    return result


def _python(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Use an existing absolute Python executable, not a PATH alias: {value}")
    # Resolving a Linux venv's python symlink to /usr/bin/python loses the venv
    # and imports the wrong converter dependencies. Preserve the invoked path.
    return str(path.absolute())


def strict_prefix_rows(path: Path, limit: int = 3) -> list[dict[str, Any]]:
    """Read only a bounded nonblank prefix; malformed prefix rows never vanish."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("Prefix limit must be a positive integer")
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON in template-check prefix") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected an object in template-check prefix")
            rows.append(row)
            if len(rows) == limit:
                break
    return rows


def verify_deployment_template(tokenizer: Any, *, chat_template_kwargs: dict[str, Any] | None = None,
                               examples: list[list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Check the actual source tokenizer before training and again after merge."""
    from ir_training.train.prepared_binding import value_sha256

    source = Path(__file__).resolve().parents[3] / "configs/export/gemma4_e2b_training_minijinja.jinja"
    template = source.read_text(encoding="utf-8")
    checked_examples = [
        [{"role": "user", "content": "  Unicode Ω and\nline breaks  "}],
        [{"role": "system", "content": "Follow the response-to-IR contract."},
         {"role": "user", "content": "Response: compact table and cards"}],
        [{"role": "user", "content": "Demonstration response"},
         {"role": "assistant", "content": "<a2ui>demonstration</a2ui>"},
         {"role": "user", "content": "New response"}],
        *(examples or []),
    ]
    # Review preparation explicitly sets enable_thinking=False. At the later
    # merge gate use the bound config instead, not an invented serving policy.
    kwargs = {"enable_thinking": False} if chat_template_kwargs is None else dict(chat_template_kwargs)
    options = dict(kwargs, tokenize=False, add_generation_prompt=True)
    for index, messages in enumerate(checked_examples):
        original = tokenizer.apply_chat_template(messages, **options)
        deployment = tokenizer.apply_chat_template(messages, chat_template=template, **options)
        if original != deployment:
            raise ValueError(f"MiniJinja deployment template differs from the training prompt at parity example {index}; refusing export")
    return {"passed": True, "tested_prompts": len(checked_examples),
            "sha256": file_sha256(source), "source_chat_template_sha256": value_sha256(tokenizer.chat_template),
            "chat_template_kwargs": kwargs}


def build_deployment_export_plan(*, profile: str, training_config_path: Path,
                                 checkpoint_dir: Path, output_dir: Path,
                                 training_python: str | Path, exporter_python: str | Path,
                                 model_dir: Path | None = None,
                                 preparation_config_path: Path | None = None,
                                 cache_length: int = 8192, max_input_tokens: int = 4096,
                                 max_new_tokens: int = 2048) -> dict[str, Any]:
    """Build subprocess contracts, including the BEFORE-training exporter probe.

    Future selected-checkpoint/config paths may not exist during a plan.  The
    prepare subprocess validates them against actual saved training provenance.
    """
    variants = deployment_variants(profile)
    if type(cache_length) is not int or cache_length < max_input_tokens + max_new_tokens:
        raise ValueError("Export cache_length must cover max_input_tokens + max_new_tokens")
    for value in (max_input_tokens, max_new_tokens):
        if type(value) is not int or value <= 0:
            raise ValueError("Generation limits must be positive integers")
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "deployment_export.py"
    output = output_dir.resolve()
    merged = output / "merged_hf"
    training_python, exporter_python = _python(training_python), _python(exporter_python)
    common = ["--profile", profile, "--cache-length", str(cache_length)]
    probe = [exporter_python, "-u", str(script), "probe", *common,
             "--model-dir", str((model_dir or checkpoint_dir).resolve()),
             "--report", str(output / "exporter_preflight.json")]
    prepare = [training_python, "-u", str(script), "prepare", "--profile", profile,
               "--training-config", str(training_config_path.resolve()),
               "--checkpoint", str(checkpoint_dir.resolve()), "--output-dir", str(merged)]
    if preparation_config_path is not None:
        prepare.extend(["--preparation-config", str(preparation_config_path.resolve())])
    for name, spec in variants.items():
        folder = output / "variants" / name
        spec.update({"output_dir": str(folder), "artifact": str(folder / "model.litertlm"),
                     "command": [exporter_python, "-u", str(script), "convert", *common,
                                 "--variant", name, "--model-dir", str(merged), "--output-dir", str(folder)]})
    return {"profile": profile, "source_checkpoint": str(checkpoint_dir.resolve()),
            "training_config": str(training_config_path.resolve()), "merged_model_dir": str(merged),
            "probe_command": probe, "prepare_command": prepare, "variants": variants,
            "environment": {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
            "conversion_device": "cpu", "official_retained_scale_export": False,
            "mtp_exported": False, "requires_merged_hf_and_real_litertlm_evaluation": True,
            "limits": {"cache_length": cache_length, "max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens}}


def export_kwargs(profile: str, variant: str, model_dir: Path, output_dir: Path, cache_length: int) -> dict[str, Any]:
    spec = deployment_variants(profile)[variant]
    kwargs: dict[str, Any] = {
        "model": str(model_dir.resolve()), "output_dir": str(output_dir.resolve()),
        "task": "text_generation", "quantization_recipe": spec["recipe"],
        "cache_length": cache_length, "prefill_lengths": [128],
        "enable_gpu_dynamic_prefill": True, "enable_gpu_dynamic_cache": True,
        "externalize_embedder": profile == "e2b", "bundle_litert_lm": True,
        "use_jinja_template": True, "experimental_use_fp16": False,
        "keep_temporary_files": False, "trust_remote_code": False,
    }
    if variant == "w16":
        # W16 describes physical weight storage, not FP16 activations/KV cache.
        # The fp16 flag alone leaves weights FP32. The whole-graph mixed pass
        # breaks Gemma4's RMSNorm composite types and misses external embedders.
        # AEQ's public FLOAT_CASTING recipe applies to every exported model,
        # including E2B's token and per-layer embedders, with FP32 boundaries.
        kwargs["quantization_recipe"] = str(W16_RECIPE_PATH)
        kwargs["experimental_use_mixed_precision"] = False
    elif profile == "e2b" and variant == "w4":
        # The named Gemma4 recipe returns a LiteRT-LM section->recipe mapping,
        # not a single TFLite recipe list. Use its checked equivalent operator
        # rules here; passing the mapping to Quantizer directly raises TypeError.
        kwargs["quantization_recipe"] = str(E2B_W4_RECIPE_PATH)
    if profile == "e2b":
        template = model_dir / "deployment_chat_template.jinja"
        kwargs["jinja_chat_template_override"] = str(template.resolve())
    return kwargs


def probe_w16_recipe() -> dict[str, Any]:
    """Validate the installed quantizer's weight-casting API without a model."""
    from ai_edge_quantizer import qtyping, recipe_manager

    payload = W16_RECIPE_PATH.read_bytes()
    manager = recipe_manager.RecipeManager()
    manager.load_quantization_recipe(json.loads(payload))
    operations = (qtyping.TFLOperationName.FULLY_CONNECTED, qtyping.TFLOperationName.EMBEDDING_LOOKUP)
    for operation in operations:
        algorithm, config = manager.get_quantization_configs(operation, "a2ui_w16_probe")
        weight = config.weight_tensor_config
        if (algorithm != "float_casting" or weight is None or weight.num_bits != 16
                or weight.dtype != qtyping.TensorDataType.FLOAT
                or config.compute_precision != qtyping.ComputePrecision.FLOAT
                or config.activation_tensor_config is not None
                or config.explicit_dequantize is not True or config.skip_checks
                or config.min_weight_elements != 0):
            raise ValueError(f"Installed quantizer/recipe cannot guarantee W16 weight-only casting for {operation}")
    if manager.need_calibration():
        raise ValueError("W16 float casting must not require calibration")
    return {"path": str(W16_RECIPE_PATH), "sha256": hashlib.sha256(payload).hexdigest(),
            "algorithm": "float_casting", "weight_storage": "FLOAT16",
            "activation_contract": "FLOAT32", "calibration_required": False,
            "validated_operations": [str(op.value) for op in operations]}


def probe_e2b_w4_recipe(mapping: Any, *, recipe_path: Path | None = None) -> dict[str, Any]:
    """Prove the flat TFLite recipe preserves the installed Gemma4 LM policy.

    This is intentionally NOT a generic dictionary flattener. The two embedding
    sections must share a policy, and their rules must be disjoint from the FC
    rules in prefill/decode. Changed/unknown mappings fail before conversion.
    """
    from ai_edge_quantizer import qtyping, recipe_manager
    from ai_edge_quantizer.utils import recipe_utils

    sections = {"tf_lite_prefill_decode", "tf_lite_embedder", "tf_lite_per_layer_embedder"}
    if not isinstance(mapping, dict) or set(mapping) != sections:
        raise ValueError("E2B W4 expects the upstream three-section Gemma4 recipe mapping")
    for section, rules in mapping.items():
        operation = "FULLY_CONNECTED" if section == "tf_lite_prefill_decode" else "EMBEDDING_LOOKUP"
        if (not isinstance(rules, list) or not rules
                or any(not isinstance(rule, dict) or rule.get("operation") != operation for rule in rules)):
            raise ValueError(f"E2B W4 cannot safely flatten changed operator rules for {section}")
    if mapping["tf_lite_embedder"] != mapping["tf_lite_per_layer_embedder"]:
        raise ValueError("E2B W4 embedding section policies differ; require explicit section routing")
    recipe_path = (recipe_path or E2B_W4_RECIPE_PATH).resolve()
    if recipe_path.suffix != ".json":
        raise ValueError("E2B W4 export requires a JSON recipe file, not a named package mapping")
    payload = recipe_path.read_bytes()
    flat_recipe = json.loads(payload)
    expected = mapping["tf_lite_prefill_decode"] + mapping["tf_lite_embedder"]
    if flat_recipe != expected:
        raise ValueError("Repository E2B W4 recipe differs from the installed upstream Gemma4 policy")
    # Exercise the same public file resolver and rule loader used by
    # Quantizer.load_quantization_recipe(str_path), not only factory()/json.loads.
    loaded_recipe = recipe_utils.resolve_recipe(str(recipe_path))
    if loaded_recipe != flat_recipe or file_sha256(recipe_path) != hashlib.sha256(payload).hexdigest():
        raise ValueError("E2B W4 JSON recipe resolution differs from the checked file")
    manager = recipe_manager.RecipeManager()
    manager.load_quantization_recipe(loaded_recipe)
    checks = [
        (qtyping.TFLOperationName.FULLY_CONNECTED, "model.layers.0.mlp.up_proj", 4, "BLOCKWISE_32"),
        (qtyping.TFLOperationName.FULLY_CONNECTED, "per_layer_model_projection", 8, "CHANNELWISE"),
        (qtyping.TFLOperationName.EMBEDDING_LOOKUP, "embed_tokens", 4, "BLOCKWISE_32"),
        (qtyping.TFLOperationName.EMBEDDING_LOOKUP, "per_layer_embedder", 4, "BLOCKWISE_32"),
    ]
    for operation, scope, bits, granularity in checks:
        algorithm, config = manager.get_quantization_configs(operation, scope)
        weight = config.weight_tensor_config
        if (algorithm != "min_max_uniform_quantize" or weight is None or weight.num_bits != bits
                or weight.dtype != qtyping.TensorDataType.INT or weight.granularity != granularity
                or config.activation_tensor_config is not None
                or config.compute_precision != qtyping.ComputePrecision.INTEGER
                or config.explicit_dequantize is not False or config.skip_checks
                or config.min_weight_elements != 0):
            raise ValueError(f"Installed quantizer cannot apply E2B W4 policy at {scope}")
    if manager.need_calibration():
        raise ValueError("E2B mixed W4/W8 dynamic recipe must not require calibration")
    mapping_bytes = json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"path": str(recipe_path), "sha256": hashlib.sha256(payload).hexdigest(),
            "upstream_recipe": E2B_W4_RECIPE, "upstream_mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
            "transport": "equivalent_flat_operator_rules", "sections": sorted(sections),
            "representation": "json_file", "file_loading_tested": True,
            "rule_count": len(flat_recipe), "calibration_required": False,
            "policy": {"fully_connected": "INT4_BLOCK32", "per_layer_fully_connected": "INT8_CHANNELWISE",
                       "token_embedder": "INT4_BLOCK32", "per_layer_embedder": "INT4_BLOCK32"}}


def probe_exporter(*, profile: str, model_dir: Path, cache_length: int = 8192) -> dict[str, Any]:
    """Check actual installed APIs/recipes/model routing without loading weights.

    This is compatibility screening, not a claim a real model converted or ran.
    In particular do not accept generic Gemma4 exportables which omit its
    additional per-layer embedding model.
    """
    variants = deployment_variants(profile)
    local_config = _json(model_dir / "config.json")
    model_type = local_config.get("model_type")
    if model_type not in PROFILES[profile]:
        raise ValueError(f"Profile {profile} cannot export model_type={model_type!r}")
    from ai_edge_quantizer import recipe
    from litert_torch.generative.export_hf.core.exportable_module_config import (
        ExportableModuleConfig,
    )
    from litert_torch.generative.export_hf.model_ext import exportables
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

    # Import the public export entry point now to discover incompatible transitive
    # packages before training consumes GPU time.
    export_module = importlib.import_module("litert_torch.generative.export_hf.export")
    if not callable(getattr(export_module, "export", None)):
        raise TypeError("Installed LiteRT Torch has no export_hf.export callable")
    fields = {field.name for field in dataclasses.fields(ExportableModuleConfig)}
    required = set(export_kwargs(profile, "w16", model_dir, model_dir / "unused", cache_length))
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"Installed exporter lacks requested options: {missing}; use a compatible isolated export environment")
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    try:
        model_class = AutoModelForCausalLM._model_mapping[type(config)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Installed Transformers cannot load {model_type} as a causal LM") from exc
    if profile == "e2b":
        probe_config = ExportableModuleConfig(**export_kwargs(profile, "w32", model_dir, model_dir / "unused", cache_length))
        custom = exportables.get_prefill_decode_exportables(config, probe_config)
        additional = exportables.get_additional_exportables(config)
        if not custom or "per_layer_embedder" not in (additional or {}):
            raise ValueError(
                f"Installed exporter does not support complete E2B per-layer embedding routing for {model_type!r}. "
                "Use the full dense gemma4 HF seed or an upstream exporter with explicit gemma4_text support; "
                "do not relabel the config or transplant weights into the official QAT graph.")
    template_parity = None
    if profile == "e2b":
        with Progress("Check actual E2B tokenizer and deployment template before training", unit="stage"):
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
            template_parity = verify_deployment_template(tokenizer)
    recipes = {}
    recipe_files = {}
    for spec in variants.values():
        name = spec["recipe"]
        if name == "none":
            continue
        if name == W16_RECIPE:
            recipe_files[name] = probe_w16_recipe()
            recipes[name] = True
            continue
        factory = getattr(recipe, name, None)
        if not callable(factory):
            raise TypeError(f"Installed AI Edge Quantizer lacks required recipe {name}")
        resolved = factory()
        if profile == "e2b" and name == E2B_W4_RECIPE:
            actual_recipe = export_kwargs(profile, "w4", model_dir, model_dir / "unused", cache_length)["quantization_recipe"]
            recipe_files[name] = probe_e2b_w4_recipe(resolved, recipe_path=Path(actual_recipe))
        recipes[name] = True
    versions = {}
    for name in ("litert-torch", "ai-edge-quantizer", "ai-edge-litert", "transformers", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "distribution metadata unavailable"
    return {"status": "passed", "screening_only": True, "model_loaded": False,
            "conversion_tested": False, "runtime_gpu_tested": False, "model_type": model_type,
            "model_class": model_class.__name__, "profile": profile, "recipes": recipes,
            "recipe_files": recipe_files,
            "config_sha256": file_sha256(model_dir / "config.json"), "versions": versions,
            "python": sys.executable, "experimental_variants": ["w16", "w4"],
            "template_parity": template_parity}


def verify_checkpoint_source(training_config: Path, checkpoint: Path, profile: str, *,
                             preparation_config: Path | None = None) -> dict[str, Any]:
    """Bind every saved checkpoint/tokenizer file and the original dense seed."""
    original = preparation_config or training_config.parent / "training_config.yaml"
    if preparation_config is None and training_config != original:
        report_path = training_config.parent / "preparation_report.json"
        report = _json(report_path) if report_path.is_file() else {}
        # Preserve support for explicitly named, preparation-bound configs.
        # A resume config instead needs the sibling original config as anchor.
        if report.get("training_config_sha256") == file_sha256(training_config):
            original = training_config
    original = original.resolve()
    binding = resolve_export_training_lineage(original, checkpoint,
        training_config=training_config.resolve() if training_config.resolve() != original else None)
    config, metadata = binding["config"], binding["metadata"]
    if type(metadata.get("checkpoint_step")) is not int or metadata["checkpoint_step"] <= 0:
        raise ValueError("Selected checkpoint has no positive optimizer-step provenance")
    if config.get("qat", {}).get("enabled") or config.get("model", {}).get("mobile_training_seed_manifest"):
        raise ValueError("Dense deployment export does not accept retained-scale or fake-QAT training; use its separate verified pipeline")
    saved = metadata.get("checkpoint_adapter_files")
    if not isinstance(saved, list) or not saved:
        raise ValueError("Selected checkpoint lacks hashed checkpoint and tokenizer inventory")
    expected = {}
    with Progress("Verify selected checkpoint weights and tokenizer", unit="file", total=len(saved)) as progress:
        for item in saved:
            path = _local_file(checkpoint, item["path"])
            if file_sha256(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
                raise ValueError(f"Checkpoint changed after training: {path.name}")
            expected[path.name] = item["sha256"]
            progress.advance()
    # Reject unbound extra model shards/tokenizer files which HF could select.
    from ir_training.train.callbacks import _checkpoint_adapter_manifest
    actual = _checkpoint_adapter_manifest(checkpoint, role="deployment")
    if {item["path"]: item["sha256"] for item in actual["files"]} != expected:
        raise ValueError("Checkpoint file inventory differs from saved provenance")
    base = Path(config["model"]["model_source"]).resolve()
    model_type = _json(base / "config.json").get("model_type")
    if model_type not in PROFILES[profile]:
        raise ValueError(f"Base model type {model_type!r} does not match {profile}")
    preparation = binding["preparation"]
    with Progress("Verify original dense model before deployment merge", unit="file", total=len(preparation["model_files"])) as progress:
        for name, expected_hash in preparation["model_files"].items():
            # Standard HF snapshots reference ../blobs through symlinks. This
            # ORIGINAL source has already been bound by a full content hash;
            # permit that layout, never an escaping manifest path such as ../x.
            if file_sha256(_local_file(base, name, allow_bound_symlink=True)) != expected_hash:
                raise ValueError(f"Original dense model changed after training: {name}")
            progress.advance()
    return {**{key: value for key, value in binding.items() if key != "preparation"}, "base_model_dir": str(base),
            "checkpoint_kind": metadata["checkpoint_kind"], "files": expected,
            "training_metadata_sha256": file_sha256(checkpoint / "training_metadata.json")}


def prepare_deployment_checkpoint(*, profile: str, training_config: Path, checkpoint: Path, output_dir: Path,
                                  preparation_config: Path | None = None) -> dict[str, Any]:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Deployment merged model directory must be fresh: {output_dir}")
    binding = verify_checkpoint_source(training_config, checkpoint, profile, preparation_config=preparation_config)
    training_config = Path(binding["training_config"])
    config = binding["config"]
    model_config = config["model"]
    from ir_training.eval.prepared_contract import (
        checked_preparation_manifest,
        verify_loaded_evaluation_tokenizer,
    )
    from ir_training.models.registry import create_adapter
    tokenizer_cfg = dict(model_config, tokenizer_source=str(checkpoint))
    tokenizer = create_adapter(tokenizer_cfg).load_tokenizer()
    manifest = checked_preparation_manifest(Path(config["run"]["dataset_dir"]))
    verify_loaded_evaluation_tokenizer(tokenizer, model_config, {"tokenizer": manifest["tokenizer"]})
    if binding["checkpoint_kind"] == "lora_adapter":
        from ir_training.export.merge_lora import merge_lora_adapter
        log("Merge selected dense LoRA checkpoint on CPU; source weights remain unchanged")
        merge_lora_adapter(base_model_id=str(model_config["model_id"]), adapter_dir=checkpoint,
                           output_dir=output_dir, model_loader=model_config.get("model_loader", "auto_causal_lm"),
                           dtype=model_config.get("dtype", "bfloat16"), trust_remote_code=False,
                           training_config_path=training_config, base_model_source=binding["base_model_dir"],
                           processor_model_id=binding["base_model_dir"])
    elif binding["checkpoint_kind"] == "full_model":
        output_dir.mkdir(parents=True, exist_ok=True)
        with Progress("Copy selected full checkpoint", unit="file", total=len(binding["files"])) as progress:
            for name in binding["files"]:
                shutil.copy2(_local_file(checkpoint, name), output_dir / name)
                progress.advance()
    else:
        raise ValueError(f"Unsupported checkpoint kind: {binding['checkpoint_kind']}")
    # Save the verified tokenizer, not a merge helper's permissive base fallback.
    tokenizer.save_pretrained(str(output_dir))
    output_tokenizer = create_adapter(dict(model_config, tokenizer_source=str(output_dir))).load_tokenizer()
    verify_loaded_evaluation_tokenizer(output_tokenizer, model_config, {"tokenizer": manifest["tokenizer"]})
    template_parity = None
    if profile == "e2b":
        source = Path(__file__).resolve().parents[3] / "configs/export/gemma4_e2b_training_minijinja.jinja"
        # Read only the bounded prepared prefix, not the entire training JSONL.
        rows = strict_prefix_rows(Path(config["run"]["dataset_dir"]) / "train.jsonl")
        examples = []
        for row in rows:
            messages = list(row.get("messages") or [])
            if messages:
                if messages[-1].get("role") == "assistant":
                    messages.pop()
                examples.append(messages)
        template_parity = verify_deployment_template(tokenizer, chat_template_kwargs=model_config.get("chat_template_kwargs") or {}, examples=examples)
        # Preserve the exact bytes verified above, including platform line endings.
        shutil.copy2(source, output_dir / "deployment_chat_template.jinja")
        if file_sha256(output_dir / "deployment_chat_template.jinja") != template_parity["sha256"]:
            raise ValueError("Deployment template serialization changed after parity validation")
    for name in ("manifest.json", "shared_prompt.json", "inference_prompt.json", "prompt_scaffolds.json"):
        source = Path(config["run"]["dataset_dir"]) / name
        if source.is_file():
            shutil.copy2(source, output_dir / ("prepared_" + name))
    shutil.copy2(training_config, output_dir / "training_config.yaml")
    shutil.copy2(checkpoint / "training_metadata.json", output_dir / "source_training_metadata.json")
    report = {key: value for key, value in binding.items() if key not in {"config", "metadata"}}
    report.update({"profile": profile, "source_checkpoint": str(checkpoint.resolve()),
                   "merged_model_dir": str(output_dir.resolve()), "template_parity": template_parity,
                   "official_retained_scale_export": False, "mtp_exported": False,
                   "checkpoint_step": binding["metadata"].get("checkpoint_step"),
                   "requires_merged_checkpoint_evaluation": True})
    with Progress("Bind merged model and deployment prompt files", unit="stage"):
        report["merged_files"] = {path.name: file_sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()}
    _write(output_dir / "deployment_source.json", report)
    return report


def inspect_variant_precision(package: dict[str, Any], profile: str, variant: str) -> dict[str, Any]:
    """Inspect physical matrix weights, following storage casts/dequantization.

    FLOAT32 activations and INT32 shape constants are not model weight precision.
    Unknown FC weight producers are rejected rather than inferred from filenames.
    """
    deployment_variants(profile)[variant]
    counts: Counter[str] = Counter()
    fc_counts: Counter[str] = Counter()
    unresolved = []
    graphs = package.get("graphs") or []
    if not graphs or any(graph.get("available") is not True for graph in graphs):
        raise ValueError("Actual package graph inspection is unavailable")
    for graph in graphs:
        buffers = {item["index"]: item.get("logical_size", 0) for item in graph.get("buffers", [])}
        for subgraph in graph.get("subgraphs", []):
            tensors = subgraph.get("tensors", [])
            producers = {index: op for op in subgraph.get("operators", []) for index in op.get("outputs", [])}

            def physical(index: int, seen: set[int], tensors=tensors, buffers=buffers, producers=producers) -> dict[str, Any] | None:
                if index in seen or index < 0 or index >= len(tensors):
                    return None
                seen.add(index)
                tensor = tensors[index]
                if buffers.get(tensor.get("buffer"), 0) > 0:
                    return tensor
                producer = producers.get(index) or {}
                if producer.get("builtin_name") in {"DEQUANTIZE", "CAST", "RESHAPE", "TRANSPOSE"} and producer.get("inputs"):
                    return physical(producer["inputs"][0], seen)
                return None

            for op in subgraph.get("operators", []):
                name = op.get("builtin_name")
                if name not in {"FULLY_CONNECTED", "EMBEDDING_LOOKUP", "GATHER"}:
                    continue
                position = 0 if name == "GATHER" else 1
                inputs = op.get("inputs") or []
                weight = physical(inputs[position], set()) if len(inputs) > position else None
                if weight is None:
                    if name == "FULLY_CONNECTED":
                        unresolved.append({"section": graph.get("section_index"), "inputs": inputs})
                    continue
                if name == "GATHER" and len(weight.get("shape", [])) < 2:
                    continue
                dtype = weight.get("type_name", "unknown")
                counts[dtype] += 1
                if name == "FULLY_CONNECTED":
                    fc_counts[dtype] += 1
    if unresolved or not fc_counts:
        raise ValueError(f"Cannot prove physical model weight precision: unresolved FC inputs={unresolved[:3]}, FC weights={dict(fc_counts)}")
    allowed = {"w32": {"FLOAT32"}, "w16": {"FLOAT16"}, "w8": {"INT8"},
               "w4": {"INT4", "UINT4", "INT8"} if profile == "e2b" else {"INT4", "UINT4"}}[variant]
    observed = set(counts)
    correct = observed <= allowed and (variant != "w4" or bool(observed & {"INT4", "UINT4"}))
    if not correct:
        raise ValueError(f"Requested {variant} physical weights {sorted(allowed)}, actual package has {dict(counts)}")
    return {"verified": True, "method": "physical_constant_weights_used_by_fully_connected_and_embeddings",
            "weight_type_counts": dict(counts), "fully_connected_weight_type_counts": dict(fc_counts),
            "activation_precision_not_conflated": True,
            "scope": "FC and embedding weight storage; not all graph constants or arithmetic precision"}


def validate_deployment_export_output(plan: dict[str, Any], variant: str) -> dict[str, Any]:
    """Root stage contract: require hashes and actual graph precision proof."""
    folder = Path(plan["variants"][variant]["output_dir"])
    manifest_path = folder / "export_manifest.json"
    inspection_path = folder / "package_inspection.json"
    report = _json(manifest_path)
    artifact = Path(plan["variants"][variant]["artifact"])
    if report.get("variant") != variant or report.get("profile") != plan["profile"]:
        raise ValueError("Export result profile/variant binding differs from plan")
    if report.get("artifact") != str(artifact.resolve()) or report.get("sha256") != file_sha256(artifact):
        raise ValueError("Export artifact path/hash differs from manifest")
    if report.get("inspection_sha256") != file_sha256(inspection_path):
        raise ValueError("Package inspection changed after export")
    precision = inspect_variant_precision(_json(inspection_path), plan["profile"], variant)
    if precision != report.get("actual_precision"):
        raise ValueError("Export precision summary differs from inspected physical weights")
    source = Path(plan["merged_model_dir"]) / "deployment_source.json"
    if file_sha256(source) != report.get("source_manifest_sha256"):
        raise ValueError("Export source manifest differs from merged checkpoint")
    files = [str(artifact), str(manifest_path), str(inspection_path)]
    recipe_file = report.get("quantization_recipe_file")
    if recipe_file is not None:
        recipe_path = _local_file(folder, recipe_file["name"])
        if file_sha256(recipe_path) != recipe_file["sha256"]:
            raise ValueError("Export quantization recipe changed after conversion")
        files.append(str(recipe_path))
    return {"artifact": str(artifact), "manifest": str(manifest_path), "inspection": str(inspection_path),
            "sha256": report["sha256"], "actual_precision": precision,
            "files": files}


def convert_deployment_variant(*, profile: str, variant: str, model_dir: Path, output_dir: Path,
                               cache_length: int = 8192) -> dict[str, Any]:
    source = _json(model_dir / "deployment_source.json")
    if source.get("profile") != profile or source.get("official_retained_scale_export") is not False:
        raise ValueError("Deployment model lacks matching dense source provenance")
    if not source.get("merged_files"):
        raise ValueError("Deployment model lacks hashed merged weights and tokenizer")
    with Progress(f"Verify merged source for {variant}", unit="stage"):
        actual_names = {path.name for path in model_dir.iterdir() if path.is_file() and path.name != "deployment_source.json"}
        if actual_names != set(source["merged_files"]):
            raise ValueError("Merged deployment file inventory changed")
        for name, digest in source["merged_files"].items():
            if file_sha256(_local_file(model_dir, name)) != digest:
                raise ValueError(f"Merged deployment source changed: {name}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Variant destination must be fresh: {output_dir}")
    preflight = probe_exporter(profile=profile, model_dir=model_dir, cache_length=cache_length)
    if profile == "e2b":
        parity = source.get("template_parity") or {}
        if parity.get("passed") is not True or parity.get("sha256") != file_sha256(model_dir / "deployment_chat_template.jinja"):
            raise ValueError("Deployment chat-template parity is missing or changed")
    export = importlib.import_module("litert_torch.generative.export_hf.export")
    kwargs = export_kwargs(profile, variant, model_dir, output_dir, cache_length)
    recipe_file = None
    local_recipe = W16_RECIPE_PATH if variant == "w16" else (
        E2B_W4_RECIPE_PATH if profile == "e2b" and variant == "w4" else None)
    if local_recipe is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        recipe_path = output_dir / f"{variant}_quantization_recipe.json"
        shutil.copy2(local_recipe, recipe_path)
        digest = file_sha256(recipe_path)
        recipe_name = deployment_variants(profile)[variant]["recipe"]
        if digest != preflight["recipe_files"][recipe_name]["sha256"]:
            raise ValueError(f"{variant.upper()} quantization recipe changed after preflight")
        kwargs["quantization_recipe"] = str(recipe_path.resolve())
        recipe_file = {"name": recipe_path.name, "sha256": digest}
        if profile == "e2b" and variant == "w4":
            from ai_edge_quantizer import recipe
            # Verify the exact destination file passed to export.export before
            # the expensive base conversion starts. Preserve W16's working path.
            copied_probe = probe_e2b_w4_recipe(recipe.gemma4_mixed48_b32(), recipe_path=recipe_path)
            if copied_probe["sha256"] != digest:
                raise ValueError("E2B W4 quantization recipe changed while validating its snapshot")
            preflight["recipe_files"][E2B_W4_RECIPE] = copied_probe
            log(f"E2B W4 recipe file verified: {kwargs['quantization_recipe']}; sha256={digest}")
    if variant == "w16":
        log("W16: FLOAT16 stored FC/embedding weights; FLOAT32 activations, RMSNorm and KV cache; "
            "whole-graph mixed precision disabled")
    elif profile == "e2b" and variant == "w4":
        log("E2B W4: equivalent flat Gemma4 recipe; INT4 block-32 FC/embedding weights, "
            "INT8 per-layer FC projections; physical package inspection required")
    log(f"Convert {profile} {variant}: {deployment_variants(profile)[variant]['kind']}; CPU conversion, no GPU speed claim")
    with Progress(f"LiteRT Torch conversion {variant}", unit="stage"):
        export.export(**kwargs)
    artifacts = list(output_dir.rglob("*.litertlm"))
    if len(artifacts) != 1 or artifacts[0].stat().st_size < 8:
        raise ValueError(f"Expected one nonempty real .litertlm artifact, found {len(artifacts)}")
    artifact = artifacts[0]
    canonical = output_dir / "model.litertlm"
    if artifact.resolve() != canonical.resolve():
        if canonical.exists():
            raise ValueError("Ambiguous canonical LiteRT-LM destination")
        artifact.replace(canonical)
    from ir_training.export.litertlm_inspector import inspect_litertlm
    with Progress(f"Inspect real LiteRT-LM package and weight precision {variant}", unit="stage"):
        inspection = inspect_litertlm(canonical, include_hashes=False, include_graph_details=True)
        _write(output_dir / "package_inspection.json", inspection)
        actual_precision = inspect_variant_precision(inspection, profile, variant)
    result = {"status": "exported_not_yet_evaluated", "profile": profile, "variant": variant,
              **deployment_variants(profile)[variant], "artifact": str(canonical.resolve()),
              "size_bytes": canonical.stat().st_size, "sha256": file_sha256(canonical),
              "source_manifest_sha256": file_sha256(model_dir / "deployment_source.json"),
              "exporter_preflight": preflight, "export_kwargs": kwargs,
              "runtime_gpu_tested": False, "actual_precision": actual_precision,
              "inspection_sha256": file_sha256(output_dir / "package_inspection.json"),
              "official_retained_scale_export": False, "mtp_exported": False}
    if recipe_file is not None:
        result["quantization_recipe_file"] = recipe_file
    _write(output_dir / "export_manifest.json", result)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "prepare", "convert"))
    parser.add_argument("--profile", required=True, choices=tuple(PROFILES))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--preparation-config", type=Path, help="Original prepared config when --training-config is a bound resume config")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--cache-length", type=int, default=8192)
    args = parser.parse_args()
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    required = {"probe": ("model_dir", "report"), "prepare": ("training_config", "checkpoint", "output_dir"),
                "convert": ("model_dir", "output_dir", "variant")}[args.stage]
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"{args.stage} requires --{name.replace('_', '-')}")
    # Must be set before importing any torch/TensorFlow converter dependency.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.stage == "probe":
        result = probe_exporter(profile=args.profile, model_dir=args.model_dir, cache_length=args.cache_length)
        _write(args.report, result)
    elif args.stage == "prepare":
        result = prepare_deployment_checkpoint(profile=args.profile, training_config=args.training_config,
                                               checkpoint=args.checkpoint, output_dir=args.output_dir,
                                               preparation_config=args.preparation_config)
    else:
        result = convert_deployment_variant(profile=args.profile, variant=args.variant, model_dir=args.model_dir,
                                             output_dir=args.output_dir, cache_length=args.cache_length)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
