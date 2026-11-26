"""Opt-in SM8850 LiteRT-LM export from the selected, dense trained checkpoints."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import mmap
import platform
import random
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import tomllib

from ir_training.common.config import load_yaml
from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.data.bixby50 import validate_source_only_rows
from ir_training.eval.golden_set import load_fixed_golden_rows
from ir_training.eval.prepared_contract import checked_preparation_manifest, file_sha256
from ir_training.export.litertlm_inspector import inspect_litertlm
from ir_training.export.litertlm_mtp import find_model_section
from ir_training.pipeline.deployment_recovery import deployment_lock

TRAINING = Path(__file__).resolve().parents[3]
BIXBY = TRAINING / "data/eval/bixby50_v1"
TEMPLATE = TRAINING / "configs/export/gemma4_e2b_training_minijinja.jinja"
SOC = "SM8850"
TARGET = "tf_lite_prefill_decode"
EMBEDDER = "tf_lite_embedder"
PER_LAYER = "tf_lite_per_layer_embedder"
PACKAGE = "model.litertlm"
REQUESTS = "bixby50_requests.jsonl"
SPLIT = "bixby50.jsonl"
VERSIONS = {
    "litert-torch": "0.9.4",
    "transformers": "5.16.1",
    "ai-edge-litert": "2.2.0",
    "ai-edge-litert-sdk-qualcomm": "2.2.0",
    "litert-lm-builder": "0.17.0",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_binding(
    profile: str, checkpoint: Path, config_path: Path
) -> tuple[dict, dict]:
    """Bind the existing training/merge receipts before allowing HF to load weights."""
    config = load_yaml(config_path)
    if (checkpoint / "adapter_config.json").exists() or not list(
        checkpoint.glob("model*.safetensors")
    ):
        raise ValueError("Expected a dense trained safetensors checkpoint")
    if profile == "270m":
        from ir_training.qat.gemma270m_w8 import validate_selected_270m_qat
        from ir_training.train.resume_contract import _verify_source_inventory

        receipt_path = checkpoint / "training_metadata.json"
        receipt = read_json(receipt_path)
        validate_selected_270m_qat(config, receipt, checkpoint)
        if (receipt.get("qat") or {}).get("true_fake_quant") is not True:
            raise ValueError("Missing 270M training-time QAT evidence")
        _verify_source_inventory(checkpoint, receipt)
    elif profile == "e2b":
        receipt_path = checkpoint / "qat_mtp_merge_metadata.json"
        receipt = read_json(receipt_path)
        if (
            read_json(checkpoint / "config.json").get("model_type") != "gemma4_text"
            or receipt.get("manifest_version") != 4
            or receipt.get("continued_qat_performed") is not True
            or receipt.get("qat_profile") != "gemma4_e2b_mobile_retained_scale_wna8o8"
            or not all(
                (receipt.get(key) or {}).get("verified") is True
                for key in (
                    "mobile_training_seed",
                    "mobile_qparams",
                    "training_run_metadata",
                )
            )
        ):
            raise ValueError(
                "Expected the official retained-mobile E2B QAT LoRA merge receipt"
            )
        weights = sorted(checkpoint.glob("model*.safetensors"))
        index = checkpoint / "model.safetensors.index.json"
        if index.is_file():
            weights.append(index)
        expected = receipt["merged_model_files"]
        actual = [
            {"path": p.name, "size": p.stat().st_size, "sha256": file_sha256(p)}
            for p in weights
        ]
        if sorted(actual, key=lambda x: x["path"]) != sorted(
            expected, key=lambda x: x["path"]
        ):
            raise ValueError(
                "E2B merged checkpoint inventory differs from its signed-off merge"
            )
    else:
        raise ValueError(f"Unknown NPU export profile {profile!r}")
    if file_sha256(config_path) != receipt.get("training_config_sha256"):
        raise ValueError("Training config differs from checkpoint provenance")
    return config, {
        "checkpoint_receipt_sha256": file_sha256(receipt_path),
        "config_sha256": file_sha256(checkpoint / "config.json"),
        "training_config_sha256": receipt["training_config_sha256"],
    }


def training_calibration(
    prepared: Path,
    manifest: dict,
    adapter,
    tokenizer,
    cache_length: int,
    reserved_paths: list[Path],
):
    """Deterministic sample of distinct training sources; never use evaluation prompts for calibration."""
    from ir_training.data.audit_filter import (
        _identity_keys_overlap,
        _row_identity_keys,
        _source_hashes,
        load_reserved_cohorts,
    )

    path = prepared / "train.jsonl"
    digest = file_sha256(path)
    if digest != manifest["splits"]["train"]["output_sha256"]:
        raise ValueError("Calibration train split differs from its preparation receipt")
    reserved = load_reserved_cohorts(reserved_paths)
    sample = []
    seen = set()
    rng = random.Random(8850)
    for row in read_jsonl(path):
        source = row.get("raw_response_text") or row["response_text"]
        source_digest = hashlib.sha256(source.encode("utf-8")).digest()
        if source_digest in seen:
            continue
        seen.add(source_digest)
        slot = len(sample) if len(sample) < 32 else rng.randrange(len(seen))
        if slot < 32:
            if (
                _identity_keys_overlap(_row_identity_keys(row), reserved["identities"])
                or _source_hashes(row) & reserved["responses"]
            ):
                raise ValueError("Held-out source in calibration training data")
            if slot == len(sample):
                sample.append(row)
            else:
                sample[slot] = row
    if len(sample) != 32:
        raise ValueError("Expected 32 distinct training sources for NPU calibration")
    prompts = []
    for row in sample:
        if row["messages"][-1]["role"] != "assistant":
            raise ValueError("Expected a supervised training row for calibration")
        prompt = adapter.format_example(
            row, tokenizer=tokenizer, include_assistant=False
        )
        if (
            len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + 32
            > cache_length
        ):
            raise ValueError("Calibration prompt exceeds the static context budget")
        prompts.append({"text": prompt})
    return prompts, {
        "train_sha256": digest,
        "sample_ids": [row["id"] for row in sample],
        "decode_steps": 32,
    }


def prepare_requests(profile: str, checkpoint: Path, config: dict, cache_length: int):
    from ir_training.eval.prepared_contract import (
        verify_golden_preparation,
        verify_loaded_evaluation_tokenizer,
    )
    from ir_training.models.registry import create_adapter
    from ir_training.pipeline.deployment_export import verify_deployment_template

    prepared = Path(config["run"]["dataset_dir"])
    manifest = checked_preparation_manifest(prepared)
    if profile == "270m":
        binding = (
            read_json(checkpoint / "training_metadata.json").get(
                "prepared_dataset_binding"
            )
            or {}
        )
        if binding.get("manifest_sha256") != file_sha256(prepared / "manifest.json"):
            raise ValueError("Prepared data differs from the selected 270M checkpoint")
    cohorts = config["final_evaluation_datasets"]
    reserved_paths = []
    for name in ("golden32", "golden35", "bixby50"):
        path = Path(cohorts[name]["split_path"])
        if file_sha256(path) != cohorts[name]["split_sha256"]:
            raise ValueError(f"Bound {name} split changed")
        reserved_paths.append(path)
    split = Path(cohorts["bixby50"]["split_path"])
    max_prompt = int(config["golden_eval"]["max_input_tokens"])
    binding = verify_golden_preparation(
        prepared,
        split,
        required_rows=50,
        max_sequence=int(config["training"]["max_seq_length"]),
        max_prompt=max_prompt,
        expected_kind="source_only_holdout",
    )
    if any(cohorts["bixby50"].get(key) != value for key, value in binding.items()):
        raise ValueError("Prepared Bixby50 differs from the original training run")
    rows = load_fixed_golden_rows(split, required_rows=50)
    validate_source_only_rows(rows, read_json(BIXBY / "benchmark_manifest.json"))
    model = dict(
        config["model"], tokenizer_source=str(checkpoint), model_source=str(checkpoint)
    )
    adapter = create_adapter(model)
    tokenizer = adapter.load_tokenizer()
    verify_loaded_evaluation_tokenizer(tokenizer, model, manifest)
    if profile == "e2b":
        verify_deployment_template(
            tokenizer,
            chat_template_kwargs=model.get("chat_template_kwargs"),
            examples=[row["messages"] for row in rows],
        )
    max_new = int(model["max_output_tokens"])
    if cache_length > int(model["max_context_tokens"]):
        raise ValueError(
            "Requested static cache exceeds the checkpoint's trained model context"
        )
    if max_new <= 0 or max_new >= cache_length:
        raise ValueError(
            "Invalid trained generation budget for selected NPU cache length"
        )
    requests = []
    for row in sorted(rows, key=lambda item: item["id"]):
        prompt = adapter.format_example(
            row, tokenizer=tokenizer, include_assistant=False
        )
        ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if not ids or len(ids) > max_prompt or len(ids) + max_new > cache_length:
            raise ValueError(
                f"{row['id']} requires {len(ids)} prompt + {max_new} output tokens "
                f"(prompt limit {max_prompt}, cache {cache_length})"
            )
        requests.append({"id": row["id"], "prompt": prompt, "input_ids": ids})
    calibration, calibration_proof = training_calibration(
        prepared, manifest, adapter, tokenizer, cache_length, reserved_paths
    )
    return (
        requests,
        calibration,
        {
            "cache_length": cache_length,
            "max_input_tokens": max_prompt,
            "max_output_tokens": max_new,
            "calibration": calibration_proof,
            "prepared_manifest_sha256": file_sha256(prepared / "manifest.json"),
            "bixby50_split_sha256": file_sha256(split),
            "bixby50_manifest_sha256": binding["manifest_sha256"],
        },
    )


def _dispatch_signatures(graph, schema) -> dict[str, int]:
    signatures = {}
    for i in range(graph.SignatureDefsLength()):
        sig = graph.SignatureDefs(i)
        name = sig.SignatureKey().decode()
        section = graph.Subgraphs(sig.SubgraphIndex())
        count = 0
        for j in range(section.OperatorsLength()):
            op = graph.OperatorCodes(section.Operators(j).OpcodeIndex())
            if (
                op.BuiltinCode() == schema.BuiltinOperator.CUSTOM
                and op.CustomCode() == b"DISPATCH_OP"
            ):
                count += 1
        if not count or name in signatures:
            raise ValueError(f"Invalid or uncompiled target signature {name!r}")
        signatures[name] = count
    if not {"prefill_128", "decode"} <= signatures.keys():
        raise ValueError("Missing compiled prefill or decode signature")
    return signatures


def inspect_npu_package(path: Path, profile: str) -> dict:
    from ai_edge_litert import schema_py_generated as schema
    from litert_lm_builder.runtime.proto import executor_metadata_pb2

    package = inspect_litertlm(path, inspect_tflite=False)
    required = [TARGET, EMBEDDER, "tf_lite_aux"]
    if profile == "e2b":
        required.append(PER_LAYER)
    sections = {name: find_model_section(package, name) for name in required}
    tflite = [s for s in package["sections"] if s["data_type_name"] == "TFLiteModel"]
    if len(tflite) != len(required) or any(
        not s["ordered_after_previous"] or not s["size"] for s in package["sections"]
    ):
        raise ValueError(
            "Expected a target-only split-cache package without duplicate or empty sections"
        )
    types = [s["data_type_name"] for s in package["sections"]]
    if (
        types.count("LlmMetadataProto") != 1
        or types.count("ExecutorMetadataProto") != 1
        or sum(types.count(t) for t in ("SP_Tokenizer", "HF_Tokenizer_Zlib")) < 1
    ):
        raise ValueError("Missing runtime metadata or tokenizer")
    helpers = {EMBEDDER: ("embedder",), "tf_lite_aux": ("cache_update", "mask", "rope")}
    if profile == "e2b":
        helpers[PER_LAYER] = ("per_layer_embedder",)
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        metadata = next(
            s
            for s in package["sections"]
            if s["data_type_name"] == "ExecutorMetadataProto"
        )
        executor = executor_metadata_pb2.ExecutorMetadata.FromString(
            mapped[metadata["begin_offset"] : metadata["end_offset"]]
        ).llm_executor_metadata
        # E2B's final 20 layers share KV. Gemma 3 270M owns KV at every layer.
        layers = 15 if profile == "e2b" else 18
        expected_kv = {
            f"kv_cache_{kind}_{i}" for i in range(layers) for kind in ("k", "v")
        }
        if (
            executor.max_history_size <= 0
            or len(executor.state_buffers) != len(expected_kv)
            or {buf.decode_input_name for buf in executor.state_buffers} != expected_kv
        ):
            raise ValueError("Unexpected SM8850 executor KV cache layout")
        for name, section in sections.items():
            with memoryview(mapped)[
                section["begin_offset"] : section["end_offset"]
            ] as view:
                if bytes(view[4:8]) != b"TFL3":
                    raise ValueError(f"Invalid FlatBuffer in {name}")
                graph = schema.Model.GetRootAsModel(view, 0)
                if name == TARGET:
                    signatures = _dispatch_signatures(graph, schema)
                else:
                    expected = {
                        signature
                        for role in helpers[name]
                        for signature in (f"prefill_{role}_128", f"decode_{role}")
                    }
                    available = {
                        graph.SignatureDefs(i).SignatureKey().decode()
                        for i in range(graph.SignatureDefsLength())
                    }
                    if not expected <= available:
                        raise ValueError(
                            f"{name} lacks runtime signatures {sorted(expected - available)}"
                        )
    return {
        "sections": required,
        "dispatch_ops_per_signature": signatures,
        "compiled_cache_length": executor.max_history_size,
    }


def _quantize_sections(*, input_litertlm, output_litertlm, **settings):
    """Keep the original package manifest, tokenizer and executor metadata through static PTQ."""
    import litert_lm_builder
    from ai_edge_litert import schema_py_generated as schema
    from litert_lm_builder import litertlm_peek
    from litert_torch.generative.export_hf.experimental.calib.quantize import quantize

    with tempfile.TemporaryDirectory(dir=Path(output_litertlm).parent) as tmp:
        unpacked = Path(tmp)
        litertlm_peek.peek_litertlm_file(str(input_litertlm), tmp, sys.stdout)
        manifest = unpacked / "model.toml"
        with manifest.open("rb") as stream:
            sections = {
                s["model_type"]: unpacked / s["data_path"]
                for s in tomllib.load(stream)["section"]
                if s["section_type"] == "TFLiteModel"
            }
        main, aux = sections["prefill_decode"], sections["aux"]
        quantize(model_path=str(main), aux_model_path=str(aux), **settings)
        for path in (main, aux):
            path.with_name(path.name + ".quantized.tflite").replace(path)
        with (
            main.open("rb") as stream,
            mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
        ):
            _check_static_quantization(schema.Model.GetRootAsModel(mapped, 0), schema)
        builder = litert_lm_builder.LitertLmFileBuilder.from_toml_file(str(manifest))
        with Path(output_litertlm).open("wb") as output:
            builder.build(output)


def _check_static_quantization(graph, schema):
    """Require physical W8 constants and calibrated A16 tensors on every target signature."""
    if not graph.SignatureDefsLength():
        raise ValueError("Static quantization produced no target signatures")
    for i in range(graph.SignatureDefsLength()):
        sig = graph.SignatureDefs(i)
        subgraph = graph.Subgraphs(sig.SubgraphIndex())
        weight = activation = False
        for j in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(j)
            quant = tensor.Quantization()
            if not quant or not quant.ScaleLength():
                continue
            buf = graph.Buffers(tensor.Buffer())
            stored = bool(buf.DataLength() or buf.Size())
            weight |= stored and tensor.Type() == schema.TensorType.INT8
            activation |= not stored and tensor.Type() == schema.TensorType.INT16
            if weight and activation:
                break
        if not (weight and activation):
            raise ValueError(
                f"Static W8/A16 tensors missing from {sig.SignatureKey()!r}"
            )


def _convert(
    profile: str,
    checkpoint: Path,
    work: Path,
    cache_length: int,
    calibration: list[dict],
):
    from litert_torch.generative.export_hf import export, export_utils
    from litert_torch.generative.export_hf.core import export_lib

    from ir_training.export.gemma4_text_compat import gemma4_text_export_context

    # Calibrate the float main graph. The host lookup tables can use W8 directly.
    def quantize_lookups(original):
        def wrapped(source, config, artifacts):
            return original(
                source,
                dataclasses.replace(config, quantization_recipe="dynamic_wi8_afp32"),
                artifacts,
            )

        return wrapped

    options = {}
    calib_dir = work / "calibration_inputs"
    write_jsonl(calib_dir / "train.jsonl", calibration)
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(
                export_utils,
                "calib_func",
                functools.partial(export_utils.calib_func, enable_formatting=False),
            )
        )
        stack.enter_context(
            patch.object(export_utils, "quant_func", _quantize_sections)
        )
        stack.enter_context(patch.object(tempfile, "tempdir", str(work)))
        if profile == "e2b":
            stack.enter_context(gemma4_text_export_context(qualcomm_npu=True))
            options["jinja_chat_template_override"] = str(TEMPLATE)
        for name in ("export_embedder_model", "export_additional_models"):
            stack.enter_context(
                patch.object(
                    export_lib, name, quantize_lookups(getattr(export_lib, name))
                )
            )
        export.export(
            model=str(checkpoint),
            output_dir=str(work),
            task="text_generation",
            aot_backend="qualcomm",
            aot_soc_model=SOC,
            use_litert_lm_compiler=True,
            calibration_dataset_dir=str(calib_dir),
            calibration_eval_task_names="train",
            max_calibration_decode_steps=32,
            quantization_recipe="",
            cache_length=cache_length,
            prefill_lengths=[128],
            externalize_embedder=True,
            split_cache=True,
            enable_dynamic_shape=False,
            enable_gpu_dynamic_cache=False,
            enable_gpu_dynamic_prefill=False,
            export_vision_encoder=False,
            export_audio_encoder=False,
            bundle_litert_lm=True,
            trust_remote_code=False,
            **options,
        )


def export_checkpoint(
    profile: str,
    checkpoint: Path,
    training_config: Path,
    output: Path,
    cache_length: int,
) -> Path:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError(
            "Qualcomm AOT compilation requires the isolated Linux x86-64 export environment"
        )
    for dist, expected in VERSIONS.items():
        actual = importlib.metadata.version(dist)
        if actual != expected:
            raise RuntimeError(f"Expected {dist}=={expected}, found {actual}")
    checkpoint, training_config, output = (
        checkpoint.resolve(),
        training_config.resolve(),
        output.resolve(),
    )
    if output.is_relative_to(checkpoint) or cache_length <= 128 or cache_length % 128:
        raise ValueError(
            "Use a separate output directory and a static cache length divisible by 128"
        )
    with deployment_lock(output):
        if output.exists():
            raise FileExistsError(output)
        config, provenance = checkpoint_binding(profile, checkpoint, training_config)
        requests, calibration, limits = prepare_requests(
            profile, checkpoint, config, cache_length
        )
        with tempfile.TemporaryDirectory(
            dir=output.parent, prefix=f".{output.name}-"
        ) as temp:
            staging = Path(temp)
            _convert(profile, checkpoint, staging, cache_length, calibration)
            artifact = staging / PACKAGE
            inspection = inspect_npu_package(artifact, profile)
            if inspection["compiled_cache_length"] != cache_length:
                raise ValueError(
                    "Compiled executor cache differs from the requested context length"
                )
            if (
                checkpoint_binding(profile, checkpoint, training_config)[1]
                != provenance
            ):
                raise ValueError("Checkpoint changed during export")
            deliverable = staging / "deliverable"
            deliverable.mkdir()
            artifact.rename(deliverable / PACKAGE)
            write_jsonl(deliverable / REQUESTS, requests)
            split = Path(config["final_evaluation_datasets"]["bixby50"]["split_path"])
            for source, dest, digest in (
                (split, SPLIT, limits["bixby50_split_sha256"]),
                (
                    split.parent / "manifest.json",
                    "manifest.json",
                    limits["bixby50_manifest_sha256"],
                ),
            ):
                shutil.copyfile(source, deliverable / dest)
                if file_sha256(deliverable / dest) != digest:
                    raise ValueError(
                        f"Bixby50 scoring context changed during export: {dest}"
                    )
            receipt = {
                "format": "a2ui-sm8850-npu-v1",
                "soc": SOC,
                "profile": profile,
                "mtp": False,
                "compiler_input": "static_a16w8_main_w8_lookup",
                "versions": VERSIONS,
                **provenance,
                **limits,
                **inspection,
                "model_sha256": file_sha256(deliverable / PACKAGE),
                "requests_sha256": file_sha256(deliverable / REQUESTS),
            }
            (deliverable / "export.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            deliverable.rename(output)
    return output / PACKAGE
