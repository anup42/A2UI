"""Metadata-only diagnostics for the pinned DeepSpeed 0.19.7 ZeRO-2 layout.

The reviewed attributes come from DeepSpeed 0.19.7's ZeRO-1/2 implementation
and tensor-fragment adapter:
https://github.com/deepspeedai/DeepSpeed/blob/v0.19.7/deepspeed/runtime/zero/stage_1_and_2.py
https://github.com/deepspeedai/DeepSpeed/blob/v0.19.7/deepspeed/utils/tensor_fragment.py

This module never reads tensor values, clones tensors, builds a state dict,
performs a reduction/collective, or assembles a full gradient. Logical bytes
describe views; physical totals deduplicate aliases by underlying storage.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _unavailable(reason: str = "attribute absent") -> dict[str, Any]:
    return {"available": False, "reason": reason}


def _value(owner: Any, name: str) -> Any:
    return getattr(owner, name, None) if owner is not None else None


def _tensor_metadata(tensor: Any) -> tuple[dict[str, Any], tuple[Any, ...] | None, int | None]:
    if tensor is None or not callable(getattr(tensor, "numel", None)):
        return _unavailable("not a tensor-like object"), None, None
    try:
        numel = int(tensor.numel())
        element_size = int(tensor.element_size())
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        return _unavailable(f"tensor metadata failed: {type(exc).__name__}"), None, None
    logical_bytes = numel * element_size
    result = {
        "available": True,
        "numel": numel,
        "element_size_bytes": element_size,
        "logical_view_bytes": logical_bytes,
        "dtype": str(getattr(tensor, "dtype", None)),
        "device": str(getattr(tensor, "device", None)),
    }
    try:
        storage = tensor.untyped_storage()
        storage_bytes = int(storage.nbytes())
        storage_key = (str(getattr(tensor, "device", None)), int(storage.data_ptr()), storage_bytes)
        result["physical_storage_bytes"] = storage_bytes
        result["physical_storage_available"] = True
        return result, storage_key, storage_bytes
    except (AttributeError, TypeError, ValueError, RuntimeError):
        result["physical_storage_available"] = False
        return result, None, None


def _tensor_collection(
    values: Any,
) -> tuple[dict[str, Any], list[tuple[Any, ...]], int]:
    if values is None:
        return _unavailable(), [], 0
    # A Tensor is iterable, but iterating it creates dimension views and can be
    # catastrophic for a very large partition. Treat tensor-like objects as a
    # single metadata item before considering container iteration.
    if callable(getattr(values, "numel", None)):
        tensors = [values]
    else:
        try:
            tensors = list(values)
        except TypeError:
            return _unavailable("attribute is neither tensor-like nor iterable"), [], 0
    items, storage_keys, logical_bytes, logical_numel, missing_storage = [], [], 0, 0, 0
    for tensor in tensors:
        metadata, storage_key, _ = _tensor_metadata(tensor)
        items.append(metadata)
        if metadata.get("available"):
            logical_numel += metadata["numel"]
            logical_bytes += metadata["logical_view_bytes"]
        if storage_key is not None:
            storage_keys.append(storage_key)
        elif metadata.get("available"):
            missing_storage += 1
    return {
        "available": True,
        "tensor_count": len(items),
        "logical_numel": logical_numel,
        "logical_view_bytes": logical_bytes,
        "physical_storage_metadata_missing_count": missing_storage,
        "tensors": items,
    }, storage_keys, logical_bytes


def _owned_parameters(values: Any) -> dict[str, Any]:
    if values is None:
        return _unavailable()
    groups = list(values)
    per_group = []
    for group in groups:
        tensors = list(group)
        per_group.append({
            "parameter_count": len(tensors),
            "logical_numel": sum(int(tensor.numel()) for tensor in tensors),
        })
    return {"available": True, "groups": per_group}


def _group_value(values: Any, index: int) -> Any:
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return _unavailable()
    value = values[index]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return str(value)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _gradient_metadata(parameter: Any, mapping: Any) -> dict[str, Any]:
    getter = getattr(mapping, "get_lp_grad_fragment", None)
    index = getattr(parameter, "_index_in_param_group", None)
    if not callable(getter) or type(index) is not int:
        return _unavailable("gradient fragment API unavailable")
    try:
        gradient = getter(index)
    except (AttributeError, KeyError, IndexError, TypeError, RuntimeError) as exc:
        return _unavailable(f"gradient fragment lookup failed: {type(exc).__name__}")
    return _tensor_metadata(gradient)[0]


def _parameter_fragments(
    parameters: list[tuple[str, Any]], *, include_gradients: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    fragments, model_tensors, local_fragments = [], [], []
    gradient_numel = gradient_bytes = gradient_count = 0
    for name, parameter in parameters:
        parameter_metadata, _, _ = _tensor_metadata(parameter)
        model_tensors.append({"name": name, **parameter_metadata})
        mapping = getattr(parameter, "_hp_mapping", None)
        if mapping is None:
            fragments.append({"name": name, "available": False, "reason": "no local fragment mapping"})
            continue
        address = getattr(mapping, "lp_fragment_address", None)
        start, count = getattr(address, "start", None), getattr(address, "numel", None)
        fragment = {
            "name": name,
            "available": type(start) is int and type(count) is int,
            "logical_interval": {"start": start, "numel": count},
        }
        if fragment["available"]:
            local_fragments.append({"name": name, "fragment_numel": count})
        if include_gradients:
            gradient = _gradient_metadata(parameter, mapping)
            fragment["gradient_fragment"] = gradient
            if gradient.get("available"):
                gradient_count += 1
                gradient_numel += gradient["numel"]
                gradient_bytes += gradient["logical_view_bytes"]
        fragments.append(fragment)
    model_tensors.sort(key=lambda item: item.get("logical_view_bytes", -1), reverse=True)
    local_fragments.sort(key=lambda item: item["fragment_numel"], reverse=True)
    return fragments, model_tensors[:20], local_fragments[:20], {
        "tensor_count": gradient_count,
        "logical_numel": gradient_numel,
        "logical_view_bytes": gradient_bytes,
    }


def _optimizer_state(raw_optimizer: Any) -> tuple[dict[str, Any], list[tuple[Any, ...]]]:
    if raw_optimizer is None:
        return _unavailable("raw optimizer absent"), []
    groups = getattr(raw_optimizer, "param_groups", None)
    state = getattr(raw_optimizer, "state", None)
    if not isinstance(groups, list) or not hasattr(state, "get"):
        return _unavailable("raw optimizer groups/state unavailable"), []
    result, keys, missing_storage = [], [], 0
    aggregates = {
        name: {"tensor_count": 0, "logical_numel": 0, "logical_view_bytes": 0}
        for name in ("master", "exp_avg", "exp_avg_sq")
    }
    for group_index, group in enumerate(groups):
        entries = []
        for parameter in group.get("params", []):
            master, key, _ = _tensor_metadata(parameter)
            if master.get("available"):
                for field in ("logical_numel", "logical_view_bytes"):
                    aggregates["master"][field] += master[
                        "numel" if field == "logical_numel" else field
                    ]
                aggregates["master"]["tensor_count"] += 1
            if key is not None:
                keys.append(key)
            elif master.get("available"):
                missing_storage += 1
            item = {"master": master}
            parameter_state = state.get(parameter, {})
            for name in ("exp_avg", "exp_avg_sq"):
                metadata, state_key, _ = _tensor_metadata(parameter_state.get(name))
                item[name] = metadata
                if metadata.get("available"):
                    aggregates[name]["tensor_count"] += 1
                    aggregates[name]["logical_numel"] += metadata["numel"]
                    aggregates[name]["logical_view_bytes"] += metadata["logical_view_bytes"]
                if state_key is not None:
                    keys.append(state_key)
                elif metadata.get("available"):
                    missing_storage += 1
            entries.append(item)
        result.append({"group_index": group_index, "entries": entries})
    return {
        "available": True,
        "aggregates": aggregates,
        "physical_storage_metadata_missing_count": missing_storage,
        "groups": result,
    }, keys


def _physical_summary(
    storage_records: Iterable[tuple[Any, ...]], *, missing_metadata_count: int,
    unavailable_categories: list[str],
) -> dict[str, Any]:
    unique = {record for record in storage_records}
    return {
        "method": "unique_untyped_storage_data_ptr_device_and_nbytes",
        "unique_storage_count": len(unique),
        "known_unique_physical_storage_bytes": sum(record[2] for record in unique),
        "storage_metadata_missing_count": missing_metadata_count,
        "unavailable_inventory_categories": unavailable_categories,
        "complete_for_inventory_collections": missing_metadata_count == 0 and not unavailable_categories,
        "warning": (
            "Only tensor_collections, raw_adamw and current_ipg_buckets contribute storage here. "
            "Parameter/fragment descriptions and unavailable categories are excluded. "
            "This is not complete process or CUDA-device physical memory."
        ),
    }


def _ipg_bucket_metadata(zero_optimizer: Any) -> tuple[dict[str, Any], list[tuple[Any, ...]]]:
    buckets = _value(zero_optimizer, "ipg_buckets")
    if not hasattr(buckets, "items"):
        return _unavailable(), []
    result, keys, missing_storage = {}, [], 0
    for dtype, bucket in buckets.items():
        metadata, bucket_keys, _ = _tensor_collection(getattr(bucket, "buffer", None))
        result[str(dtype)] = {
            "buffer": metadata,
            "index": _json_value(getattr(bucket, "index", None)),
        }
        keys.extend(bucket_keys)
        missing_storage += metadata.get("physical_storage_metadata_missing_count", 0)
    return {
        "available": True,
        "physical_storage_metadata_missing_count": missing_storage,
        "by_dtype": result,
    }, keys


def summarize_zero_partitions(
    zero_optimizer: Any,
    parameters: list[tuple[str, Any]],
    *,
    include_gradients: bool = False,
) -> dict[str, Any]:
    """Return JSON-serializable rank-local ZeRO metadata without tensor work."""
    group_fields = ("partition_count", "partition_size", "groups_padding", "first_offset")
    group_count = max(
        (len(value) for name in group_fields
         if isinstance((value := _value(zero_optimizer, name)), (list, tuple))),
        default=0,
    )
    groups = [
        {"group_index": index, **{
            name: _group_value(_value(zero_optimizer, name), index) for name in group_fields
        }}
        for index in range(group_count)
    ]
    storage_records: list[tuple[Any, ...]] = []
    collections = {}
    for label, attribute in (
        ("replicated_flat_parameters", "bit16_groups_flat"),
        ("fp32_master_partitions", "single_partition_of_fp32_groups"),
        ("all_partition_views", "parallel_partitioned_bit16_groups"),
        ("current_gradient_partition_buffers", "grads_in_partition"),
    ):
        value = _value(zero_optimizer, attribute)
        if attribute == "parallel_partitioned_bit16_groups" and value is not None:
            value = [tensor for group in value for tensor in group]
        metadata, keys, _ = _tensor_collection(value)
        collections[label] = metadata
        storage_records.extend(keys)
    optimizer, optimizer_keys = _optimizer_state(_value(zero_optimizer, "optimizer"))
    storage_records.extend(optimizer_keys)
    ipg_buckets, ipg_keys = _ipg_bucket_metadata(zero_optimizer)
    storage_records.extend(ipg_keys)
    fragments, largest_model, largest_fragments, gradient_aggregate = _parameter_fragments(
        parameters, include_gradients=include_gradients
    )
    missing_storage = sum(
        collection.get("physical_storage_metadata_missing_count", 0)
        for collection in collections.values()
        if collection.get("available")
    )
    missing_storage += optimizer.get("physical_storage_metadata_missing_count", 0)
    missing_storage += ipg_buckets.get("physical_storage_metadata_missing_count", 0)
    unavailable_categories = [name for name, collection in {
        **collections, "raw_adamw": optimizer, "current_ipg_buckets": ipg_buckets,
    }.items() if not collection.get("available")]
    return {
        "schema_version": 1,
        "scope": "rank_local_metadata_only",
        "include_gradient_fragment_metadata": include_gradients,
        "groups": groups if groups else _unavailable("no partition group metadata"),
        "tensor_collections": collections,
        "raw_adamw": optimizer,
        "ownership": {
            "params_in_partition": _owned_parameters(_value(zero_optimizer, "params_in_partition")),
            "params_not_in_partition": _owned_parameters(_value(zero_optimizer, "params_not_in_partition")),
        },
        "bucket_policy": {
            name: (_json_value(_value(zero_optimizer, name))
                   if _value(zero_optimizer, name) is not None else _unavailable())
            for name in (
                "partition_gradients", "contiguous_gradients", "overlap_comm",
                "reduce_bucket_size", "allgather_bucket_size",
                "gradient_accumulation_dtype", "use_separate_grad_accum",
            )
        },
        "current_ipg_buckets": ipg_buckets,
        "parameter_fragments": fragments,
        "gradient_fragment_aggregate": (
            gradient_aggregate if include_gradients else _unavailable("gradient metadata not requested")
        ),
        "largest_model_tensors": largest_model,
        "largest_local_fragment_views": largest_fragments,
        "physical_storage_summary": _physical_summary(
            storage_records, missing_metadata_count=missing_storage,
            unavailable_categories=unavailable_categories,
        ),
    }
