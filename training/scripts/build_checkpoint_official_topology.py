"""Quantize a merged Gemma checkpoint into an official LiteRT-LM topology.

This is the production counterpart to
``build_converter_random_topology_injection_parity.py``.  The official package
is treated as a read-only graph/template authority.  Float projection weights
from a merged QAT+LoRA checkpoint are mapped to every unique official
FC/embedding constant, quantized by the public AI Edge Quantizer with the
officially observed bit width, and copied into a byte-for-byte copy of the
official graph.

The operation is deliberately narrow. It accepts only projection-only LoRA
training bound to the declared public training seed. Every mapped target
FC/embedding constant is regenerated from the merged checkpoint; learned
constants outside that inventory remain from the official package. Gemma 4
therefore also requires a fail-closed retained-constant evidence contract.
For Gemma 4, the training checkpoint must descend from the hash-bound public
mobile reconstruction manifest; the incompatible public Q4_0 seed is rejected.
The tokenizer, separate token/per-layer embedder sections, audio/vision
sections, and default MTP drafter are preserved byte-for-byte. A fail-closed
manifest records every source key, shape transform, quantizer layout, graph
fingerprint, and package boundary before a final package is accepted.

This does not recover Google's private trainer, calibration corpus, or learned
constants.  It does reproduce the released graph/operators/layout while
allowing trained projection weights to differ, which is the artifact property
required for equivalent GPU delegation.  Actual throughput and MTP acceptance
remain device gates.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_gemma4_mobile_checkpoint_parity import _source_key as _gemma4_source_key
from audit_litertlm_remote_source_recipe import (
    generic_gemma3_source_key,
    read_local_safetensors_header,
)
from build_converter_random_inventory_parity import (
    _build_random_float_tflite,
    _counter,
    _recipe,
)
from build_converter_random_topology_injection_parity import (
    _compact_converter_layout,
    _constants_digest,
    _converter_batch_ranges,
    _converter_weight_records,
    _file_range_sha256,
    _graph_report,
    _non_mapped_state_digest,
    _official_constants_digest,
    _package_boundary_report,
    _patch_official_constants,
    _runtime_allocate_report,
    _sha256,
    _write_injected_package,
)
from build_converter_topology_parity import (
    _quantize as _topology_quantize,
)
from build_converter_topology_parity import (
    _read_section,
    _section_by_model_type,
)
from build_fresh_random_quantized_graph import _extract_inventory
from ir_training.common.config import load_yaml
from ir_training.export.litertlm_inspector import inspect_litertlm
from ir_training.qat.fake_quant import QATSpec
from ir_training.qat.mobile_training_seed import (
    OFFICIAL_MOBILE_MODEL_ID,
    verify_configured_mobile_training_seed,
)
from ir_training.qat.retained_constants import verify_retained_constant_contract


class CheckpointTopologyError(RuntimeError):
    """Raised when a checkpoint cannot safely inhabit an official topology."""


_FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "gemma4_e2b": {
        "model_type": "tf_lite_prefill_decode",
        "expected_inventory": 277,
        "model_id_marker": "gemma-4-e2b",
        "mtp_model_type": "tf_lite_mtp_drafter",
    },
    "gemma3_270m": {
        "model_type": "TF_LITE_PREFILL_DECODE",
        "expected_inventory": 127,
        "model_id_marker": "gemma-3-270m",
        "mtp_model_type": None,
    },
}
_FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _canonical_family(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "gemma4": "gemma4_e2b",
        "gemma4_e2b": "gemma4_e2b",
        "gemma_4_e2b": "gemma4_e2b",
        "gemma270m": "gemma3_270m",
        "gemma3_270m": "gemma3_270m",
        "gemma_3_270m": "gemma3_270m",
    }
    if normalized not in aliases:
        raise CheckpointTopologyError(
            f"Unsupported family {value!r}; choose gemma4_e2b or gemma3_270m."
        )
    return aliases[normalized]


class SafetensorCheckpoint:
    """Header-indexed, lazy safetensors reader for merged HF checkpoints."""

    def __init__(self, checkpoint: str | Path):
        self.path = Path(checkpoint).expanduser().resolve()
        self.key_to_shard: dict[str, Path] = {}
        self.entries: dict[str, dict[str, Any]] = {}
        self.shards: list[Path] = []
        self._stack: contextlib.ExitStack | None = None
        self._handles: dict[Path, Any] = {}
        self._discover()

    def _discover(self) -> None:
        if self.path.is_file():
            if self.path.suffix.lower() != ".safetensors":
                raise CheckpointTopologyError(
                    f"Checkpoint file must be .safetensors: {self.path}"
                )
            shards = [self.path]
            declared_map: dict[str, str] | None = None
        elif self.path.is_dir():
            index_path = self.path / "model.safetensors.index.json"
            declared_map = None
            if index_path.is_file():
                try:
                    payload = json.loads(index_path.read_text(encoding="utf-8"))
                    weight_map = payload.get("weight_map")
                except (OSError, json.JSONDecodeError) as exc:
                    raise CheckpointTopologyError(
                        f"Could not read safetensors index {index_path}: {exc}"
                    ) from exc
                if not isinstance(weight_map, dict) or not weight_map:
                    raise CheckpointTopologyError(
                        f"Safetensors index has no non-empty weight_map: {index_path}"
                    )
                declared_map = {str(key): str(value) for key, value in weight_map.items()}
                shards = sorted({(self.path / value).resolve() for value in declared_map.values()})
            else:
                preferred = [self.path / "model.safetensors"]
                shards = [item for item in preferred if item.is_file()]
                if not shards:
                    shards = sorted(self.path.glob("model-*.safetensors"))
                if not shards:
                    shards = sorted(
                        item
                        for item in self.path.glob("*.safetensors")
                        if not item.name.startswith("adapter_model")
                    )
        else:
            raise CheckpointTopologyError(f"Merged checkpoint does not exist: {self.path}")

        if not shards:
            raise CheckpointTopologyError(
                f"No merged model safetensors were found under {self.path}."
            )
        missing = [item for item in shards if not item.is_file()]
        if missing:
            raise CheckpointTopologyError(
                "Safetensors index references missing shards: "
                + ", ".join(str(item) for item in missing[:8])
            )

        for shard in shards:
            try:
                header = read_local_safetensors_header(shard)
            except Exception as exc:
                raise CheckpointTopologyError(
                    f"Could not read safetensors header {shard}: {exc}"
                ) from exc
            for key, entry in header.items():
                if key.startswith("__"):
                    continue
                if not isinstance(entry, dict):
                    continue
                if key in self.entries:
                    raise CheckpointTopologyError(
                        f"Tensor key {key!r} occurs in more than one shard."
                    )
                dtype = str(entry.get("dtype") or "").upper()
                shape = tuple(int(value) for value in entry.get("shape", []))
                offsets = entry.get("data_offsets")
                if dtype not in _DTYPE_BYTES or not isinstance(offsets, list | tuple) or len(offsets) != 2:
                    raise CheckpointTopologyError(
                        f"Tensor {key!r} has an invalid safetensors header entry."
                    )
                start, end = (int(offsets[0]), int(offsets[1]))
                payload_size = int(header["__a2ui_file_size__"]) - 8 - int(
                    header["__a2ui_header_size__"]
                )
                expected_bytes = int(np.prod(shape, dtype=np.int64)) * _DTYPE_BYTES[dtype]
                if start < 0 or end < start or end > payload_size or end - start != expected_bytes:
                    raise CheckpointTopologyError(
                        f"Tensor {key!r} has invalid data offsets {offsets}; "
                        f"expected {expected_bytes} bytes inside payload size {payload_size}."
                    )
                enriched = dict(entry)
                enriched["shard"] = str(shard)
                self.entries[key] = enriched
                self.key_to_shard[key] = shard

        if declared_map is not None:
            for key, relative in declared_map.items():
                expected = (self.path / relative).resolve()
                observed = self.key_to_shard.get(key)
                if observed is None:
                    raise CheckpointTopologyError(
                        f"Index key {key!r} is absent from declared shard {expected}."
                    )
                if observed != expected:
                    raise CheckpointTopologyError(
                        f"Index key {key!r} resolved to {observed}, expected {expected}."
                    )
        self.shards = list(shards)

    @property
    def keys(self) -> set[str]:
        return set(self.entries)

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "tensor_count": len(self.entries),
            "shard_count": len(self.shards),
            "shards": [
                {"path": str(item), "size": int(item.stat().st_size)}
                for item in self.shards
            ],
        }

    def __enter__(self) -> Self:
        self._stack = contextlib.ExitStack()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._handles.clear()

    def load_float32(self, key: str) -> np.ndarray:
        if key not in self.key_to_shard:
            raise CheckpointTopologyError(f"Checkpoint tensor is unavailable: {key}")
        if self._stack is None:
            raise CheckpointTopologyError(
                "SafetensorCheckpoint must be used as a context manager when loading tensors."
            )
        shard = self.key_to_shard[key]
        handle = self._handles.get(shard)
        if handle is None:
            try:
                from safetensors import safe_open
            except ImportError as exc:  # pragma: no cover - conversion environment only
                raise CheckpointTopologyError(
                    "Checkpoint execution requires safetensors and PyTorch."
                ) from exc
            handle = self._stack.enter_context(
                safe_open(str(shard), framework="pt", device="cpu")
            )
            self._handles[shard] = handle
        try:
            tensor = handle.get_tensor(key)
        except Exception as exc:
            raise CheckpointTopologyError(f"Could not load {key!r}: {exc}") from exc
        if not bool(getattr(tensor, "is_floating_point", lambda: False)()):
            raise CheckpointTopologyError(
                f"Source tensor {key!r} is not floating point; packed checkpoints "
                "must be dequantized/merged before this converter path."
            )
        return np.ascontiguousarray(tensor.detach().float().cpu().numpy())


def _key_candidates(canonical: str, family: str) -> list[str]:
    variants = [canonical]
    if family == "gemma4_e2b" and canonical.endswith(".weight"):
        # The ordinary Transformers Gemma 4 model keeps projection weights in
        # Gemma4ClippableLinear.linear, while the public packed mobile model
        # flattens that wrapper. Accept both state-dict forms.
        variants.append(canonical[: -len(".weight")] + ".linear.weight")
    if canonical.startswith("model."):
        variants.extend(
            value[len("model.") :] for value in list(variants) if value.startswith("model.")
        )
    else:
        variants.append("model." + canonical)
    if family == "gemma4_e2b" and canonical == "lm_head.weight":
        variants.extend(
            [
                "model.language_model.embed_tokens.weight",
                "language_model.embed_tokens.weight",
            ]
        )
    expanded: list[str] = []
    for value in variants:
        expanded.extend([value, "base_model.model." + value])
    return list(dict.fromkeys(expanded))


def _canonical_inventory_keys(
    records: list[dict[str, Any]], family: str, model_type: str
) -> list[str | None]:
    if family == "gemma4_e2b":
        return [
            _gemma4_source_key(
                model_type,
                str(record.get("official_tensor_name") or ""),
                int(record["ordinal"]),
            )[0]
            for record in records
        ]

    result: list[str | None] = []
    fc_ordinal = 0
    for record in records:
        operator = str(record.get("operator") or "").upper()
        if operator == "EMBEDDING_LOOKUP":
            result.append("model.embed_tokens.weight")
        elif operator == "FULLY_CONNECTED":
            result.append(
                generic_gemma3_source_key(
                    fc_ordinal,
                    tuple(int(value) for value in record["shape"]),
                )
            )
            fc_ordinal += 1
        else:
            result.append(None)
    return result


def _qat_module_name_for_inventory_key(
    canonical_key: str, family: str
) -> str:
    """Map an export/checkpoint key to the base module named during QAT."""

    if family == "gemma4_e2b" and canonical_key == "lm_head.weight":
        # The official decode head is tied to the language embedding in the
        # supported checkpoint. The compiler resolves this same alias through
        # ``_key_candidates``; audit the QAT precision of that source module.
        return "model.language_model.embed_tokens"
    if canonical_key.endswith(".weight"):
        return canonical_key[: -len(".weight")]
    return canonical_key


def _qat_inventory_precision_report(
    records: list[dict[str, Any]],
    *,
    family: str,
    model_type: str,
    training_config: str | Path | None,
) -> dict[str, Any]:
    """Compare every official weight bit-width with its training-time QAT rule."""

    result: dict[str, Any] = {
        "training_config": str(Path(training_config).expanduser().resolve())
        if training_config
        else None,
        "official_inventory_count": len(records),
        "canonical_key_count": 0,
        "compared_count": 0,
        "matched_count": 0,
        "mismatch_count": 0,
        "official_bit_histogram": {},
        "qat_bit_histogram": {},
        "assignments": [],
        "mismatches": [],
        "exact": False,
    }
    if not records:
        result["error"] = "Official target inventory is empty."
        return result
    if training_config is None:
        result["error"] = "Training config is required for QAT precision coverage."
        return result
    config_path = Path(training_config).expanduser().resolve()
    if not config_path.is_file():
        result["error"] = f"Training config does not exist: {config_path}"
        return result
    try:
        spec = QATSpec.from_config(load_yaml(config_path))
        canonical_keys = _canonical_inventory_keys(records, family, model_type)
    except Exception as exc:  # noqa: BLE001 - precision coverage is a hard gate
        result["error"] = f"Could not resolve QAT precision coverage: {exc}"
        return result

    official_histogram: dict[str, int] = {}
    qat_histogram: dict[str, int] = {}
    assignments: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for record, canonical_key in zip(records, canonical_keys):
        official_bits = int(record["bits"])
        official_label = str(official_bits)
        official_histogram[official_label] = (
            official_histogram.get(official_label, 0) + 1
        )
        module_name = (
            _qat_module_name_for_inventory_key(canonical_key, family)
            if canonical_key
            else None
        )
        qat_bits = spec.weight_bits_for_module(module_name) if module_name else None
        qat_label = "excluded" if qat_bits is None else str(qat_bits)
        qat_histogram[qat_label] = qat_histogram.get(qat_label, 0) + 1
        matches = bool(module_name and qat_bits == official_bits)
        assignment = {
            "ordinal": int(record["ordinal"]),
            "canonical_source_key": canonical_key,
            "qat_module_name": module_name,
            "official_bits": official_bits,
            "qat_bits": qat_bits,
            "match": matches,
        }
        assignments.append(assignment)
        if not matches:
            mismatches.append(assignment)

    canonical_count = sum(
        1 for assignment in assignments if assignment["canonical_source_key"]
    )
    matched_count = sum(1 for assignment in assignments if assignment["match"])
    result.update(
        {
            "canonical_key_count": canonical_count,
            "compared_count": len(assignments),
            "matched_count": matched_count,
            "mismatch_count": len(mismatches),
            "official_bit_histogram": dict(
                sorted(official_histogram.items(), key=lambda item: int(item[0]))
            ),
            "qat_bit_histogram": dict(
                sorted(
                    qat_histogram.items(),
                    key=lambda item: (item[0] == "excluded", item[0]),
                )
            ),
            "assignments": assignments,
            "mismatches": mismatches,
            "exact": bool(
                len(assignments) == len(records)
                and canonical_count == len(records)
                and not mismatches
            ),
        }
    )
    return result


def _resolve_inventory_mappings(
    records: list[dict[str, Any]],
    *,
    family: str,
    model_type: str,
    checkpoint: SafetensorCheckpoint,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_keys = _canonical_inventory_keys(records, family, model_type)
    mappings: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for record, canonical in zip(records, canonical_keys):
        ordinal = int(record["ordinal"])
        if not canonical:
            issues.append(
                {
                    "code": "unmapped_official_tensor",
                    "ordinal": ordinal,
                    "tensor_name": record.get("official_tensor_name"),
                }
            )
            continue
        resolved = next(
            (key for key in _key_candidates(canonical, family) if key in checkpoint.entries),
            None,
        )
        if resolved is None:
            issues.append(
                {
                    "code": "missing_checkpoint_tensor",
                    "ordinal": ordinal,
                    "canonical_source_key": canonical,
                    "expected_shape": record["shape"],
                }
            )
            continue
        entry = checkpoint.entries[resolved]
        source_shape = tuple(int(value) for value in entry.get("shape", []))
        expected_shape = tuple(int(value) for value in record["shape"])
        transposed = False
        if source_shape != expected_shape:
            if len(source_shape) == 2 and source_shape[::-1] == expected_shape:
                transposed = True
            else:
                issues.append(
                    {
                        "code": "checkpoint_shape_mismatch",
                        "ordinal": ordinal,
                        "source_key": resolved,
                        "source_shape": list(source_shape),
                        "expected_shape": list(expected_shape),
                    }
                )
                continue
        dtype = str(entry.get("dtype") or "").upper()
        if dtype not in _FLOAT_DTYPES:
            issues.append(
                {
                    "code": "checkpoint_tensor_not_float",
                    "ordinal": ordinal,
                    "source_key": resolved,
                    "dtype": dtype,
                }
            )
            continue
        mappings.append(
            {
                "ordinal": ordinal,
                "operator": record["operator"],
                "bits": int(record["bits"]),
                "official_shape": list(expected_shape),
                "official_tensor_name": record.get("official_tensor_name"),
                "canonical_source_key": canonical,
                "source_key": resolved,
                "source_shape": list(source_shape),
                "source_dtype": dtype,
                "transpose": transposed,
                "source_shard": str(checkpoint.key_to_shard[resolved]),
            }
        )
    return mappings, issues


def _training_scope_report(
    training_config: str | Path | None,
    *,
    family: str,
    official_base_model_id: str | None,
    mobile_training_seed_manifest: str | Path | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "training_config_present": False,
        "official_base_model_id_declared": bool(official_base_model_id),
        "base_model_id_matches_declared_official_base": False,
        "qat_lora_sft": False,
        "qat_enabled": False,
        "projection_only_lora": False,
        "only_base_layers": False,
        "effective_merged_weight_qat": False,
        "zero_lora_dropout": False,
        "embedding_qat_matches_inventory": False,
        "public_ai_edge_weight_ranges": False,
        "precision_matches_official_layout": False,
        "family_training_seed_supported": family != "gemma4_e2b",
        "mobile_training_seed_manifest_matches_config": family != "gemma4_e2b",
        "mobile_training_seed_verified": family != "gemma4_e2b",
        "exact_checkpoint_key_loading": family != "gemma4_e2b",
        "per_layer_embedding_group_size_matches": family != "gemma4_e2b",
    }
    result: dict[str, Any] = {
        "path": str(Path(training_config).expanduser().resolve())
        if training_config
        else None,
        "official_base_model_id": official_base_model_id,
        "checks": checks,
        "supported_projection_only_transplant": False,
    }
    if training_config is None:
        return result
    path = Path(training_config).expanduser().resolve()
    if not path.is_file():
        result["error"] = f"Training config does not exist: {path}"
        return result
    try:
        config = load_yaml(path)
    except Exception as exc:  # noqa: BLE001 - convert config failures into a gate
        result["error"] = f"Could not load training config: {exc}"
        return result
    checks["training_config_present"] = True
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    training = (
        config.get("training") if isinstance(config.get("training"), dict) else {}
    )
    lora = config.get("lora") if isinstance(config.get("lora"), dict) else {}
    qat = config.get("qat") if isinstance(config.get("qat"), dict) else {}
    model_id = str(model.get("model_id") or "")
    result["training_model_id"] = model_id
    checks["base_model_id_matches_declared_official_base"] = bool(
        official_base_model_id and model_id == str(official_base_model_id)
    )
    if family == "gemma4_e2b":
        checks["family_training_seed_supported"] = model_id == OFFICIAL_MOBILE_MODEL_ID
        configured_manifest = model.get("mobile_training_seed_manifest")
        configured_path = (
            (ROOT / str(configured_manifest)).resolve()
            if configured_manifest
            else None
        )
        supplied_path = (
            Path(mobile_training_seed_manifest).expanduser().resolve()
            if mobile_training_seed_manifest
            else None
        )
        checks["mobile_training_seed_manifest_matches_config"] = bool(
            configured_path and supplied_path and configured_path == supplied_path
        )
        mobile_training_seed = verify_configured_mobile_training_seed(
            model, base=ROOT, require_materialized=True
        )
        checks["mobile_training_seed_verified"] = bool(
            mobile_training_seed["verified"]
        )
        checks["exact_checkpoint_key_loading"] = bool(
            model.get("require_exact_checkpoint_keys") is True
        )
        try:
            checks["per_layer_embedding_group_size_matches"] = (
                QATSpec.from_config(config).group_size_for_module(
                    "language_model.embed_tokens_per_layer"
                )
                == 256
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            checks["per_layer_embedding_group_size_matches"] = False
        result["mobile_training_seed"] = mobile_training_seed
    checks["qat_lora_sft"] = str(training.get("method") or "") == "qat_lora_sft"
    checks["qat_enabled"] = bool(qat.get("enabled", False))
    modules_to_save = lora.get("modules_to_save", [])
    target_modules = lora.get("target_modules", "auto")
    target_aliases = {"", "auto", "peft", "peft-default", "peft_default"}
    projection_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "q_proj.linear",
        "k_proj.linear",
        "v_proj.linear",
        "o_proj.linear",
        "gate_proj.linear",
        "up_proj.linear",
        "down_proj.linear",
    }
    if isinstance(target_modules, list):
        normalized_targets = [str(item).strip() for item in target_modules]
        supported_targets = bool(normalized_targets) and all(
            any(
                target == suffix or target.endswith("." + suffix)
                for suffix in projection_suffixes
            )
            for target in normalized_targets
        )
    else:
        normalized_targets = str(target_modules or "").strip().lower()
        supported_targets = normalized_targets in target_aliases
    checks["projection_only_lora"] = bool(
        not modules_to_save and supported_targets
    )
    result["lora_target_modules"] = normalized_targets
    checks["only_base_layers"] = bool(qat.get("only_base_layers", False))
    checks["effective_merged_weight_qat"] = bool(
        qat.get("effective_merged_weight", False)
    )
    try:
        checks["zero_lora_dropout"] = float(lora.get("dropout", -1.0)) == 0.0
    except (TypeError, ValueError):
        checks["zero_lora_dropout"] = False
    checks["embedding_qat_matches_inventory"] = bool(
        qat.get("quantize_embeddings", False)
        and not qat.get("exclude_modules", [])
    )
    checks["public_ai_edge_weight_ranges"] = (
        str(qat.get("quantizer") or "").strip().lower() == "ste_ai_edge"
    )
    weight_bits = int(qat.get("weight_bits", 0) or 0)
    activation_bits = int(qat.get("activation_bits", 0) or 0)
    if family == "gemma4_e2b":
        checks["precision_matches_official_layout"] = bool(
            weight_bits in {4, 8}
            and activation_bits == 8
            and qat.get("schema_path")
        )
    else:
        # The released 270M Q8 graph has INT8 per-row weights and FP32
        # activation edges.  >=16 disables activation fake quantization.
        checks["precision_matches_official_layout"] = bool(
            weight_bits == 8 and activation_bits >= 16
        )
    result["supported_projection_only_transplant"] = bool(all(checks.values()))
    result["preserved_constant_contract"] = (
        "Every mapped target FC/embedding constant may differ. All learned constants "
        "outside that inventory are retained from the hash-bound official package; "
        "this mutation scope can preserve graph/layout identity, but production also "
        "requires a separate retained-constant compatibility proof."
    )
    return result


def _merge_provenance_report(
    checkpoint: Path,
    training_config: str | Path | None,
    *,
    official_base_model_id: str | None,
    mobile_training_seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed_required = bool(mobile_training_seed and mobile_training_seed.get("required"))
    metadata_path = (
        checkpoint / "qat_mtp_merge_metadata.json"
        if checkpoint.is_dir()
        else checkpoint.parent / "qat_mtp_merge_metadata.json"
    )
    checks = {
        "metadata_present": metadata_path.is_file(),
        "manifest_v3_or_newer": False,
        "base_model_matches": False,
        "training_config_hash_matches": False,
        "qat_lora_sft": False,
        "qat_enabled": False,
        "effective_merged_weight_qat": False,
        "zero_lora_dropout": False,
        "training_run_metadata_verified": False,
        "continued_qat_performed": False,
        "merge_did_not_fake_qat": False,
        "adapter_hashes_recorded": False,
        "merged_checkpoint_hashes_match": False,
        "floating_merge_requires_quantization": False,
        "assistant_unmodified": False,
        "mobile_training_seed_matches": not seed_required,
        "base_model_source_matches_seed": not seed_required,
        "exact_checkpoint_key_loading": not seed_required,
    }
    result: dict[str, Any] = {
        "path": str(metadata_path),
        "checks": checks,
        "verified": False,
    }
    if not metadata_path.is_file():
        return result
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"Could not read merge provenance: {exc}"
        return result
    if not isinstance(metadata, dict):
        result["error"] = "Merge provenance must be a JSON object."
        return result
    result["metadata"] = metadata
    checks["manifest_v3_or_newer"] = int(metadata.get("manifest_version", 0) or 0) >= 3
    checks["base_model_matches"] = bool(
        official_base_model_id
        and str(metadata.get("base_model_id") or "") == str(official_base_model_id)
    )
    if seed_required:
        recorded_seed = (
            metadata.get("mobile_training_seed")
            if isinstance(metadata.get("mobile_training_seed"), dict)
            else {}
        )
        checks["manifest_v3_or_newer"] = (
            int(metadata.get("manifest_version", 0) or 0) >= 4
        )
        checks["mobile_training_seed_matches"] = bool(
            mobile_training_seed
            and mobile_training_seed.get("verified") is True
            and recorded_seed.get("verified") is True
            and str(recorded_seed.get("manifest_sha256") or "").lower()
            == str(mobile_training_seed.get("manifest_sha256") or "").lower()
            and str(recorded_seed.get("transformation_plan_sha256") or "").lower()
            == str(
                mobile_training_seed.get("transformation_plan_sha256") or ""
            ).lower()
        )
        seed_output = (
            mobile_training_seed.get("output")
            if isinstance(mobile_training_seed.get("output"), dict)
            else {}
        )
        checks["base_model_source_matches_seed"] = bool(
            str(metadata.get("base_model_source") or "")
            == str(seed_output.get("directory") or "")
        )
        checks["exact_checkpoint_key_loading"] = bool(
            metadata.get("exact_checkpoint_keys_required") is True
        )
    if training_config is not None:
        config_path = Path(training_config).expanduser().resolve()
        if config_path.is_file():
            expected_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            result["expected_training_config_sha256"] = expected_config_hash
            checks["training_config_hash_matches"] = (
                str(metadata.get("training_config_sha256") or "").lower()
                == expected_config_hash
            )
    checks["qat_lora_sft"] = str(metadata.get("training_method") or "") == "qat_lora_sft"
    checks["qat_enabled"] = bool(metadata.get("qat_enabled", False))
    checks["effective_merged_weight_qat"] = bool(
        metadata.get("qat_effective_merged_weight", False)
    )
    try:
        checks["zero_lora_dropout"] = float(metadata.get("lora_dropout", -1.0)) == 0.0
    except (TypeError, ValueError):
        checks["zero_lora_dropout"] = False
    run_metadata = (
        metadata.get("training_run_metadata")
        if isinstance(metadata.get("training_run_metadata"), dict)
        else {}
    )
    checks["training_run_metadata_verified"] = run_metadata.get("verified") is True
    checks["continued_qat_performed"] = metadata.get("continued_qat_performed") is True
    checks["merge_did_not_fake_qat"] = metadata.get("merge_performed_qat") is False
    adapter_files = metadata.get("adapter_files")
    checks["adapter_hashes_recorded"] = bool(
        isinstance(adapter_files, list)
        and adapter_files
        and all(
            isinstance(item, dict)
            and int(item.get("size", 0) or 0) > 0
            and len(str(item.get("sha256") or "")) == 64
            for item in adapter_files
        )
    )
    merged_model_files = metadata.get("merged_model_files")
    verification: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    if isinstance(merged_model_files, list) and merged_model_files:
        checkpoint_root = checkpoint if checkpoint.is_dir() else checkpoint.parent
        for item in merged_model_files:
            if not isinstance(item, dict):
                verification.append({"valid": False, "reason": "entry_not_object"})
                continue
            relative = Path(str(item.get("path") or ""))
            safe_relative = bool(
                str(relative)
                and not relative.is_absolute()
                and ".." not in relative.parts
            )
            candidate = (checkpoint_root / relative).resolve()
            within_checkpoint = bool(
                safe_relative and candidate.is_relative_to(checkpoint_root.resolve())
            )
            expected_size = int(item.get("size", 0) or 0)
            expected_sha = str(item.get("sha256") or "").lower()
            observed_size = (
                int(candidate.stat().st_size)
                if within_checkpoint and candidate.is_file()
                else None
            )
            observed_sha = (
                _file_range_sha256(candidate, 0, observed_size)
                if observed_size is not None
                else None
            )
            valid = bool(
                within_checkpoint
                and expected_size > 0
                and len(expected_sha) == 64
                and observed_size == expected_size
                and observed_sha == expected_sha
            )
            declared_paths.add(str(relative).replace("\\", "/"))
            verification.append(
                {
                    "path": str(relative).replace("\\", "/"),
                    "expected_size": expected_size,
                    "observed_size": observed_size,
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed_sha,
                    "valid": valid,
                }
            )
    if checkpoint.is_dir():
        actual_paths = {
            item.relative_to(checkpoint).as_posix()
            for item in checkpoint.glob("model*.safetensors")
            if item.is_file()
        }
        index_path = checkpoint / "model.safetensors.index.json"
        if index_path.is_file():
            actual_paths.add(index_path.relative_to(checkpoint).as_posix())
    else:
        actual_paths = {checkpoint.name} if checkpoint.is_file() else set()
    checks["merged_checkpoint_hashes_match"] = bool(
        verification
        and all(item.get("valid", False) for item in verification)
        and len(declared_paths) == len(verification)
        and declared_paths == actual_paths
    )
    result["merged_model_file_verification"] = verification
    checks["floating_merge_requires_quantization"] = bool(
        metadata.get("requires_post_merge_quantization", False)
        and not metadata.get("packed_int4_output", False)
    )
    checks["assistant_unmodified"] = not bool(
        metadata.get("mtp_assistant_trained_or_modified", False)
    )
    result["verified"] = bool(all(checks.values()))
    return result


def build_plan(
    artifact: str | Path,
    checkpoint: str | Path,
    *,
    family: str,
    model_type: str | None = None,
    training_config: str | Path | None = None,
    retained_constant_contract: str | Path | None = None,
    mobile_training_seed_manifest: str | Path | None = None,
    official_base_model_id: str | None = None,
    official_artifact_sha256: str | None = None,
    output_dir: str | Path | None = None,
    package_output: str | Path | None = None,
) -> dict[str, Any]:
    normalized_family = _canonical_family(family)
    spec = _FAMILY_SPECS[normalized_family]
    selected_model_type = str(model_type or spec["model_type"])
    artifact_path = Path(artifact).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    package_section: dict[str, Any] | None = None
    checkpoint_index: SafetensorCheckpoint | None = None
    mappings: list[dict[str, Any]] = []
    declared_artifact_sha256 = str(official_artifact_sha256 or "").strip().lower()
    observed_artifact_sha256: str | None = None

    if len(declared_artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in declared_artifact_sha256
    ):
        issues.append(
            {
                "code": "official_artifact_sha256_required",
                "message": "Declare the exact released package SHA-256 (64 lowercase hex characters).",
            }
        )

    if not artifact_path.is_file():
        issues.append(
            {"code": "missing_official_artifact", "path": str(artifact_path)}
        )
    else:
        try:
            observed_artifact_sha256 = _file_range_sha256(
                artifact_path, 0, artifact_path.stat().st_size
            )
            if (
                len(declared_artifact_sha256) == 64
                and observed_artifact_sha256 != declared_artifact_sha256
            ):
                issues.append(
                    {
                        "code": "official_artifact_sha256_mismatch",
                        "expected": declared_artifact_sha256,
                        "observed": observed_artifact_sha256,
                    }
                )
            package = inspect_litertlm(artifact_path, inspect_tflite=False)
            package_section = _section_by_model_type(package, selected_model_type)
            _, records = _extract_inventory(
                artifact_path,
                selected_model_type,
                include_embeddings=True,
                max_weights=None,
            )
        except Exception as exc:  # noqa: BLE001 - inventory failures must be reported
            issues.append({"code": "official_inventory_error", "error": str(exc)})
    expected_inventory = int(spec["expected_inventory"])
    if records and len(records) != expected_inventory:
        issues.append(
            {
                "code": "unexpected_official_inventory_count",
                "expected": expected_inventory,
                "observed": len(records),
            }
        )

    try:
        checkpoint_index = SafetensorCheckpoint(checkpoint_path)
    except CheckpointTopologyError as exc:
        issues.append({"code": "checkpoint_index_error", "error": str(exc)})
    if records and checkpoint_index is not None:
        mappings, mapping_issues = _resolve_inventory_mappings(
            records,
            family=normalized_family,
            model_type=selected_model_type,
            checkpoint=checkpoint_index,
        )
        issues.extend(mapping_issues)
        if len(mappings) != len(records):
            issues.append(
                {
                    "code": "incomplete_checkpoint_mapping",
                    "official_inventory": len(records),
                    "mapped": len(mappings),
                }
            )

    training_scope = _training_scope_report(
        training_config,
        family=normalized_family,
        official_base_model_id=official_base_model_id,
        mobile_training_seed_manifest=mobile_training_seed_manifest,
    )
    if not training_scope["supported_projection_only_transplant"]:
        issues.append(
            {
                "code": "unsupported_training_mutation_scope",
                "checks": training_scope["checks"],
            }
        )
    retained_constant_compatibility = verify_retained_constant_contract(
        retained_constant_contract,
        family=normalized_family,
        training_model_id=str(training_scope.get("training_model_id") or ""),
    )
    if not retained_constant_compatibility["verified"]:
        issues.append(
            {
                "code": "retained_constant_compatibility_unverified",
                "checks": retained_constant_compatibility["checks"],
                "production_status": retained_constant_compatibility.get(
                    "production_status"
                ),
                "message": (
                    "Refusing to mix checkpoint-derived projection weights with "
                    "unverified learned constants retained from the official package."
                ),
            }
        )
    qat_precision_coverage = _qat_inventory_precision_report(
        records,
        family=normalized_family,
        model_type=selected_model_type,
        training_config=training_config,
    )
    if records and not qat_precision_coverage["exact"]:
        issues.append(
            {
                "code": "qat_official_inventory_precision_mismatch",
                "matched": qat_precision_coverage["matched_count"],
                "official_inventory": qat_precision_coverage[
                    "official_inventory_count"
                ],
                "mismatches": qat_precision_coverage["mismatches"],
                "error": qat_precision_coverage.get("error"),
            }
        )
    merge_provenance = _merge_provenance_report(
        checkpoint_path,
        training_config,
        official_base_model_id=official_base_model_id,
        mobile_training_seed=training_scope.get("mobile_training_seed"),
    )
    if not merge_provenance["verified"]:
        issues.append(
            {
                "code": "unverified_merge_provenance",
                "checks": merge_provenance["checks"],
            }
        )

    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else ROOT / "outputs" / "checkpoint_official_topology"
    )
    output_package = (
        Path(package_output).expanduser().resolve()
        if package_output
        else output_root / f"{normalized_family}_trained_official_topology.litertlm"
    )
    if output_package == artifact_path:
        issues.append(
            {
                "code": "output_would_overwrite_official_artifact",
                "path": str(output_package),
            }
        )
    return {
        "family": normalized_family,
        "artifact": str(artifact_path),
        "official_artifact_sha256_expected": declared_artifact_sha256 or None,
        "official_artifact_sha256_observed": observed_artifact_sha256,
        "checkpoint": str(checkpoint_path),
        "model_type": selected_model_type,
        "expected_inventory_count": expected_inventory,
        "official_inventory_count": len(records),
        "package_section": package_section,
        "checkpoint_index": checkpoint_index.describe()
        if checkpoint_index is not None
        else None,
        "mapping_count": len(mappings),
        "mappings": mappings,
        "training_scope": training_scope,
        "mobile_training_seed": training_scope.get("mobile_training_seed"),
        "retained_constant_compatibility": retained_constant_compatibility,
        "qat_precision_coverage": qat_precision_coverage,
        "merge_provenance": merge_provenance,
        "output_dir": str(output_root),
        "package_output": str(output_package),
        "training_executed": False,
        "private_google_recipe_recovered": False,
        "official_graph_is_template_authority": True,
        "ready": not issues,
        "issues": issues,
        "remaining_runtime_gates": [
            "Android GPU full-delegation parity",
            "bounded warm decode throughput parity",
            *(
                ["MTP draft acceptance and MTP-on throughput parity"]
                if normalized_family == "gemma4_e2b"
                else []
            ),
        ],
    }


def _mapping_preview(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        key: mapping.get(key)
        for key in (
            "ordinal",
            "operator",
            "bits",
            "official_shape",
            "official_tensor_name",
            "canonical_source_key",
            "source_key",
            "source_shape",
            "source_dtype",
            "transpose",
            "source_shard",
            "source_float32_sha256",
        )
    }


def run(
    artifact: str | Path,
    checkpoint: str | Path,
    *,
    family: str,
    model_type: str | None,
    training_config: str | Path,
    retained_constant_contract: str | Path | None,
    official_base_model_id: str,
    official_artifact_sha256: str,
    output_dir: str | Path,
    package_output: str | Path,
    mobile_training_seed_manifest: str | Path | None = None,
    calibration_samples: int = 2,
    threads: int = 1,
    converter_batch_size: int | None = None,
    retain_intermediates: bool = False,
    runtime_allocate: bool = False,
    runtime_threads: int = 2,
    runtime_without_default_delegates: bool = False,
) -> dict[str, Any]:
    plan = build_plan(
        artifact,
        checkpoint,
        family=family,
        model_type=model_type,
        training_config=training_config,
        retained_constant_contract=retained_constant_contract,
        mobile_training_seed_manifest=mobile_training_seed_manifest,
        official_base_model_id=official_base_model_id,
        official_artifact_sha256=official_artifact_sha256,
        output_dir=output_dir,
        package_output=package_output,
    )
    if not plan["ready"]:
        raise CheckpointTopologyError(
            "Checkpoint topology plan is not executable: "
            + json.dumps(plan["issues"], ensure_ascii=False)
        )
    output_root = Path(plan["output_dir"])
    output_package = Path(plan["package_output"])
    if output_root.exists() and any(output_root.iterdir()):
        raise CheckpointTopologyError(f"Output directory is non-empty: {output_root}")
    if output_package.exists():
        raise CheckpointTopologyError(
            f"Refusing to overwrite package output: {output_package}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_path = Path(plan["artifact"])
    selected_model_type = str(plan["model_type"])
    package = inspect_litertlm(artifact_path, inspect_tflite=False)
    section = _section_by_model_type(package, selected_model_type)
    official_bytes = _read_section(artifact_path, section)
    _, official_records = _extract_inventory(
        artifact_path,
        selected_model_type,
        include_embeddings=True,
        max_weights=None,
    )
    mapping_by_ordinal = {
        int(item["ordinal"]): dict(item) for item in plan["mappings"]
    }
    chosen_batch_size, batch_ranges = _converter_batch_ranges(
        official_records, converter_batch_size
    )
    checkpoint_index = SafetensorCheckpoint(plan["checkpoint"])
    converter_records: list[dict[str, Any]] = []
    converter_batches: list[dict[str, Any]] = []
    converter_digest = hashlib.sha256()
    quantized_paths: list[Path] = []
    recipe: list[dict[str, Any]] = []
    loaded_mappings: dict[int, dict[str, Any]] = {}

    try:
        with checkpoint_index:
            for batch_index, (batch_start, batch_end) in enumerate(batch_ranges):
                batch_records = official_records[batch_start:batch_end]

                def provide_weight(
                    ordinal: int, _record: dict[str, Any]
                ) -> np.ndarray:
                    mapping = mapping_by_ordinal.get(int(ordinal))
                    if mapping is None:
                        raise CheckpointTopologyError(
                            f"No source mapping exists for official ordinal {ordinal}."
                        )
                    values = checkpoint_index.load_float32(str(mapping["source_key"]))
                    if bool(mapping["transpose"]):
                        values = np.ascontiguousarray(values.T)
                    expected = tuple(int(value) for value in mapping["official_shape"])
                    if tuple(values.shape) != expected:
                        raise CheckpointTopologyError(
                            f"Loaded tensor {mapping['source_key']!r} has shape "
                            f"{tuple(values.shape)}, expected {expected}."
                        )
                    if not bool(np.all(np.isfinite(values))):
                        raise CheckpointTopologyError(
                            f"Loaded tensor {mapping['source_key']!r} contains NaN/Inf."
                        )
                    mapping["source_float32_sha256"] = hashlib.sha256(
                        memoryview(values).cast("B")
                    ).hexdigest()
                    loaded_mappings[int(ordinal)] = mapping
                    return values

                float_bytes, _ = _build_random_float_tflite(
                    batch_records,
                    seed=0,
                    ordinal_offset=batch_start,
                    weight_provider=provide_weight,
                    graph_description=(
                        "Merged checkpoint float inventory for official-topology transplant"
                    ),
                )
                batch_recipe = _recipe(batch_records, ordinal_offset=batch_start)
                recipe.extend(batch_recipe)
                float_path = output_root / f"checkpoint_fp32_batch_{batch_index:04d}.tflite"
                quantized_path = (
                    output_root / f"checkpoint_quantized_batch_{batch_index:04d}.tflite"
                )
                if retain_intermediates:
                    float_path.write_bytes(float_bytes)
                try:
                    _topology_quantize(
                        float_path if retain_intermediates else float_bytes,
                        quantized_path,
                        batch_recipe,
                        calibration_samples=calibration_samples,
                        threads=threads,
                    )
                except Exception as exc:
                    raise CheckpointTopologyError(
                        f"Public AI Edge quantization failed for batch {batch_index}: {exc}"
                    ) from exc
                quantized_bytes = quantized_path.read_bytes()
                observed = _converter_weight_records(quantized_bytes)
                expected_ordinals = list(range(batch_start, batch_end))
                observed_ordinals = [int(item["ordinal"]) for item in observed]
                if observed_ordinals != expected_ordinals:
                    raise CheckpointTopologyError(
                        f"Converter batch {batch_index} returned ordinals "
                        f"{observed_ordinals}, expected {expected_ordinals}."
                    )
                converter_records.extend(observed)
                converter_digest.update(quantized_bytes)
                quantized_paths.append(quantized_path)
                converter_batches.append(
                    {
                        "index": batch_index,
                        "start_ordinal": batch_start,
                        "end_ordinal_exclusive": batch_end,
                        "weight_count": batch_end - batch_start,
                        "float_size": len(float_bytes),
                        "quantized_size": len(quantized_bytes),
                        "float_path": str(float_path) if retain_intermediates else None,
                        "quantized_path": str(quantized_path)
                        if retain_intermediates
                        else None,
                    }
                )
    finally:
        gc.collect()

    (output_root / "checkpoint_quantization_recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    injected_bytes, injection = _patch_official_constants(
        official_bytes, official_records, converter_records
    )
    official_graph = _graph_report(official_bytes)
    injected_graph = _graph_report(injected_bytes)
    converter_constants = _constants_digest(converter_records)
    injected_constants = _official_constants_digest(injected_bytes, official_records)
    official_non_mapped_state = _non_mapped_state_digest(
        official_bytes, official_records
    )
    injected_non_mapped_state = _non_mapped_state_digest(
        injected_bytes, official_records
    )
    official_structural = official_graph["graph"].get("structural_sha256")
    injected_structural = injected_graph["graph"].get("structural_sha256")
    official_layout = official_graph["graph"].get("quantization_layout_sha256")
    injected_layout = injected_graph["graph"].get("quantization_layout_sha256")
    official_topology = official_graph["graph_without_buffer_indices"].get(
        "structural_sha256"
    )
    injected_topology = injected_graph["graph_without_buffer_indices"].get(
        "structural_sha256"
    )
    official_layout_no_buffers = official_graph["graph_without_buffer_indices"].get(
        "quantization_layout_sha256"
    )
    injected_layout_no_buffers = injected_graph["graph_without_buffer_indices"].get(
        "quantization_layout_sha256"
    )
    official_execution_contract = official_graph["graph"].get(
        "execution_contract_sha256"
    )
    injected_execution_contract = injected_graph["graph"].get(
        "execution_contract_sha256"
    )
    official_execution_contract_no_buffers = official_graph[
        "graph_without_buffer_indices"
    ].get("execution_contract_sha256")
    injected_execution_contract_no_buffers = injected_graph[
        "graph_without_buffer_indices"
    ].get("execution_contract_sha256")
    execution_contracts_complete = bool(
        official_graph["graph"].get("execution_contract_complete")
        and injected_graph["graph"].get("execution_contract_complete")
        and official_graph["graph_without_buffer_indices"].get(
            "execution_contract_complete"
        )
        and injected_graph["graph_without_buffer_indices"].get(
            "execution_contract_complete"
        )
    )
    gates = {
        "training_scope_supported": bool(
            plan["training_scope"]["supported_projection_only_transplant"]
        ),
        "retained_constant_compatibility_verified": bool(
            plan["retained_constant_compatibility"]["verified"]
        ),
        "merge_provenance_verified": bool(plan["merge_provenance"]["verified"]),
        "complete_checkpoint_mapping": len(loaded_mappings) == len(official_records),
        "expected_official_inventory": len(official_records)
        == int(plan["expected_inventory_count"]),
        "converter_inventory_count": len(converter_records) == len(official_records),
        "converter_quantization_layout": _counter(official_records)
        == _counter(converter_records),
        "converter_constants_transferred_exactly": converter_constants
        == injected_constants,
        "non_mapped_state_byte_exact": bool(
            official_non_mapped_state["sha256"]
            == injected_non_mapped_state["sha256"]
        ),
        "official_graph_structure": bool(
            official_structural and official_structural == injected_structural
        ),
        "official_quantization_layout": bool(
            official_layout and official_layout == injected_layout
        ),
        "official_execution_topology_ignoring_buffer_indices": bool(
            official_topology and official_topology == injected_topology
        ),
        "official_layout_ignoring_buffer_indices": bool(
            official_layout_no_buffers
            and official_layout_no_buffers == injected_layout_no_buffers
        ),
        "execution_contract_complete": execution_contracts_complete,
        "official_execution_contract": bool(
            execution_contracts_complete
            and official_execution_contract
            and official_execution_contract == injected_execution_contract
        ),
        "official_execution_contract_ignoring_buffer_indices": bool(
            execution_contracts_complete
            and official_execution_contract_no_buffers
            and official_execution_contract_no_buffers
            == injected_execution_contract_no_buffers
        ),
        "section_size_unchanged": len(injected_bytes) == len(official_bytes),
    }
    final_gate_pass = bool(all(gates.values()))
    package_boundary = _package_boundary_report(
        artifact_path, section, injected_bytes
    )
    result: dict[str, Any] = {
        **{key: value for key, value in plan.items() if key != "mappings"},
        "training_executed": False,
        "checkpoint_weights_loaded": len(loaded_mappings),
        "checkpoint_mappings": [
            _mapping_preview(loaded_mappings[index])
            for index in sorted(loaded_mappings)
        ],
        "converter": {
            "public_ai_edge_quantizer_executed": True,
            "batch_size": int(chosen_batch_size),
            "batch_count": len(converter_batches),
            "batches": converter_batches,
            "combined_model_sha256": converter_digest.hexdigest(),
            "weight_count": len(converter_records),
            "layout": _compact_converter_layout(converter_records),
            "constants_sha256": converter_constants,
        },
        "injection": {
            **injection,
            "section_size": len(injected_bytes),
            "section_sha256": _sha256(injected_bytes),
            "constants_sha256": injected_constants,
        },
        "official_section": official_graph,
        "candidate_section": injected_graph,
        "non_mapped_state": {
            "official": official_non_mapped_state,
            "candidate": injected_non_mapped_state,
        },
        "package_boundary": package_boundary,
        "gates": gates,
        "final_artifact_gate_pass": final_gate_pass,
        "exact_official_graph_ops_layout_match": bool(
            gates["official_graph_structure"]
            and gates["official_quantization_layout"]
            and gates["official_execution_topology_ignoring_buffer_indices"]
            and gates["official_layout_ignoring_buffer_indices"]
            and gates["official_execution_contract"]
            and gates["official_execution_contract_ignoring_buffer_indices"]
        ),
        "learned_weights_expected_to_differ": True,
        "private_google_recipe_recovered": False,
        "android_gpu_validation_required": True,
    }
    if runtime_allocate:
        result["host_runtime"] = _runtime_allocate_report(
            official_bytes,
            injected_bytes,
            threads=runtime_threads,
            without_default_delegates=runtime_without_default_delegates,
        )
        gates["host_runtime_allocation_match"] = bool(
            result["host_runtime"].get("allocation_match")
        )
        final_gate_pass = bool(all(gates.values()))
        result["final_artifact_gate_pass"] = final_gate_pass

    report_path = output_root / "checkpoint_official_topology_report.json"
    if not final_gate_pass:
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise CheckpointTopologyError(
            "Exact-topology artifact gates failed; no .litertlm package was written."
        )

    partial_output = output_package.with_name(output_package.name + ".partial")
    if partial_output.exists():
        raise CheckpointTopologyError(
            f"Refusing to overwrite quarantined partial package: {partial_output}"
        )
    _write_injected_package(artifact_path, section, injected_bytes, partial_output)
    output_size = partial_output.stat().st_size
    begin = int(section["begin_offset"])
    size = int(section["size"])
    suffix_size = int(artifact_path.stat().st_size) - begin - size
    output_prefix = _file_range_sha256(partial_output, 0, begin)
    output_suffix = _file_range_sha256(partial_output, begin + size, suffix_size)
    output_sha256 = _file_range_sha256(partial_output, 0, output_size)
    outside_unchanged = bool(
        output_size == artifact_path.stat().st_size
        and output_prefix == package_boundary["prefix_sha256"]
        and output_suffix == package_boundary["suffix_sha256"]
    )
    result["package_boundary"].update(
        {
            "quarantined_partial": str(partial_output),
            "output_size": int(output_size),
            "output_sha256": output_sha256,
            "output_prefix_sha256": output_prefix,
            "output_suffix_sha256": output_suffix,
            "bytes_outside_selected_section_verified_unchanged": outside_unchanged,
        }
    )
    if plan["family"] == "gemma4_e2b":
        mtp_section = _section_by_model_type(
            package, str(_FAMILY_SPECS["gemma4_e2b"]["mtp_model_type"])
        )
        mtp_begin = int(mtp_section["begin_offset"])
        mtp_size = int(mtp_section["size"])
        official_mtp_sha = _file_range_sha256(artifact_path, mtp_begin, mtp_size)
        output_mtp_sha = _file_range_sha256(partial_output, mtp_begin, mtp_size)
        result["mtp_preservation"] = {
            "model_type": _FAMILY_SPECS["gemma4_e2b"]["mtp_model_type"],
            "offset": mtp_begin,
            "size": mtp_size,
            "official_sha256": official_mtp_sha,
            "output_sha256": output_mtp_sha,
            "byte_exact": official_mtp_sha == output_mtp_sha,
            "assistant_trained": False,
        }
        outside_unchanged = bool(
            outside_unchanged and result["mtp_preservation"]["byte_exact"]
        )
    result["gates"]["package_outside_target_byte_exact"] = outside_unchanged
    result["final_artifact_gate_pass"] = bool(all(result["gates"].values()))
    if not result["final_artifact_gate_pass"]:
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise CheckpointTopologyError(
            "Package boundary verification failed; the unpromoted file remains "
            f"quarantined at {partial_output}."
        )

    partial_output.replace(output_package)
    result["package_boundary"]["output"] = str(output_package)
    result["package_boundary"]["quarantined_partial"] = None

    cleanup: dict[str, Any] = {
        "requested": not retain_intermediates,
        "removed": [],
        "locked": [],
    }
    if not retain_intermediates:
        gc.collect()
        for item in quantized_paths:
            try:
                item.unlink(missing_ok=True)
                cleanup["removed"].append(str(item))
            except PermissionError:
                cleanup["locked"].append(str(item))
    result["temporary_fixture_cleanup"] = cleanup
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Map a merged QAT+LoRA checkpoint through the public AI Edge "
            "quantizer into an official LiteRT-LM graph/template."
        )
    )
    parser.add_argument("artifact", help="Read-only official .litertlm package.")
    parser.add_argument("--checkpoint", required=True, help="Merged HF checkpoint dir/file.")
    parser.add_argument(
        "--family", required=True, choices=("gemma4_e2b", "gemma3_270m")
    )
    parser.add_argument("--model-type")
    parser.add_argument("--training-config", required=True)
    parser.add_argument(
        "--retained-constant-contract",
        help=(
            "Evidence YAML proving constants retained from the official package "
            "are compatible with the declared training seed (required for Gemma 4)."
        ),
    )
    parser.add_argument(
        "--mobile-training-seed-manifest",
        help=(
            "Hash manifest for the reconstructed Gemma 4 mobile BF16 seed "
            "(required for Gemma 4)."
        ),
    )
    parser.add_argument("--official-base-model-id", required=True)
    parser.add_argument("--official-artifact-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-output")
    parser.add_argument("--calibration-samples", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--converter-batch-size", type=int)
    parser.add_argument("--retain-intermediates", action="store_true")
    parser.add_argument("--runtime-allocate", action="store_true")
    parser.add_argument("--runtime-threads", type=int, default=2)
    parser.add_argument("--runtime-without-default-delegates", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually load checkpoint tensors, quantize, and write the candidate.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.calibration_samples < 1 or args.threads < 1 or args.runtime_threads < 1:
        parser.error("sample/thread counts must be positive")
    if args.converter_batch_size is not None and args.converter_batch_size < 1:
        parser.error("--converter-batch-size must be positive")
    package_output = args.package_output or str(
        Path(args.output_dir) / f"{args.family}_trained_official_topology.litertlm"
    )
    try:
        if args.execute:
            result = run(
                args.artifact,
                args.checkpoint,
                family=args.family,
                model_type=args.model_type,
                training_config=args.training_config,
                retained_constant_contract=args.retained_constant_contract,
                mobile_training_seed_manifest=args.mobile_training_seed_manifest,
                official_base_model_id=args.official_base_model_id,
                official_artifact_sha256=args.official_artifact_sha256,
                output_dir=args.output_dir,
                package_output=package_output,
                calibration_samples=args.calibration_samples,
                threads=args.threads,
                converter_batch_size=args.converter_batch_size,
                retain_intermediates=args.retain_intermediates,
                runtime_allocate=args.runtime_allocate,
                runtime_threads=args.runtime_threads,
                runtime_without_default_delegates=args.runtime_without_default_delegates,
            )
        else:
            result = build_plan(
                args.artifact,
                args.checkpoint,
                family=args.family,
                model_type=args.model_type,
                training_config=args.training_config,
                retained_constant_contract=args.retained_constant_contract,
                mobile_training_seed_manifest=args.mobile_training_seed_manifest,
                official_base_model_id=args.official_base_model_id,
                official_artifact_sha256=args.official_artifact_sha256,
                output_dir=args.output_dir,
                package_output=package_output,
            )
    except (OSError, ValueError, CheckpointTopologyError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not args.execute:
        print("Plan only: no training, tensor loading, quantization, or package write was run.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
