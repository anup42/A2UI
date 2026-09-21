from __future__ import annotations

from types import SimpleNamespace

import torch
from ir_training.train.sharded_partition_diagnostics import summarize_zero_partitions


class _Mapping:
    def __init__(self, start, numel, gradient):
        self.lp_fragment_address = SimpleNamespace(start=start, numel=numel)
        self.gradient = gradient

    def get_lp_grad_fragment(self, index):
        assert index == 0
        return self.gradient


def _fixture():
    module = torch.nn.Parameter(torch.zeros(8))
    module._index_in_param_group = 0
    module._hp_mapping = _Mapping(0, 4, module[:4])
    flat = torch.zeros(16)
    master = torch.nn.Parameter(torch.zeros(4))
    raw = SimpleNamespace(
        param_groups=[{"params": [master]}],
        state={master: {"exp_avg": torch.zeros(4), "exp_avg_sq": torch.zeros(4)}},
    )
    zero = SimpleNamespace(
        partition_count=[4], partition_size=[4], groups_padding=[1], first_offset=[0],
        bit16_groups_flat=[flat], parallel_partitioned_bit16_groups=[[flat[:4], flat[4:8]]],
        single_partition_of_fp32_groups=[master], grads_in_partition=[module[:4]],
        params_in_partition=[[module]], params_not_in_partition=[[]], optimizer=raw,
        partition_gradients=True, contiguous_gradients=True, overlap_comm=False,
        reduce_bucket_size=5_000_000, allgather_bucket_size=5_000_000,
        gradient_accumulation_dtype=torch.float32, use_separate_grad_accum=False,
        ipg_buckets={torch.float32: SimpleNamespace(buffer=[flat[:2]], index=2)},
    )
    return zero, [("weight", module)]


def test_partition_summary_is_json_serializable_and_comparison_ready():
    import json

    zero, parameters = _fixture()
    report = summarize_zero_partitions(zero, parameters, include_gradients=True)
    json.dumps(report)
    assert report["groups"] == [{
        "group_index": 0, "partition_count": 4, "partition_size": 4,
        "groups_padding": 1, "first_offset": 0,
    }]
    fragment = report["parameter_fragments"][0]
    assert fragment["logical_interval"] == {"start": 0, "numel": 4}
    assert fragment["gradient_fragment"]["numel"] == 4
    assert report["raw_adamw"]["groups"][0]["entries"][0]["exp_avg_sq"]["numel"] == 4
    assert report["current_ipg_buckets"]["by_dtype"]["torch.float32"]["index"] == 2
    assert report["largest_model_tensors"][0]["name"] == "weight"
    assert report["largest_local_fragment_views"][0] == {"name": "weight", "fragment_numel": 4}
    assert report["gradient_fragment_aggregate"]["logical_numel"] == 4
    aggregates = report["raw_adamw"]["aggregates"]
    assert aggregates["master"]["logical_numel"] == 4
    assert aggregates["exp_avg"]["logical_view_bytes"] == 16


def test_aliases_are_not_double_counted_as_physical_storage():
    zero, parameters = _fixture()
    report = summarize_zero_partitions(zero, parameters)
    flat = zero.bit16_groups_flat[0]
    logical = report["tensor_collections"]["replicated_flat_parameters"]["logical_view_bytes"]
    assert logical == flat.numel() * flat.element_size()
    # The flat tensor and both partition views share one storage.
    assert report["physical_storage_summary"]["unique_storage_count"] < 8


def test_missing_optional_attributes_are_explicitly_unavailable():
    parameter = torch.nn.Parameter(torch.ones(2))
    report = summarize_zero_partitions(SimpleNamespace(), [("p", parameter)])
    assert report["groups"]["available"] is False
    assert report["tensor_collections"]["current_gradient_partition_buffers"]["available"] is False
    assert report["bucket_policy"]["reduce_bucket_size"]["available"] is False
    assert report["parameter_fragments"][0]["available"] is False
    physical = report["physical_storage_summary"]
    assert physical["complete_for_inventory_collections"] is False
    assert "raw_adamw" in physical["unavailable_inventory_categories"]


def test_diagnostics_do_not_clone_or_extract_state(monkeypatch):
    zero, parameters = _fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("tensor value/copy operation is forbidden")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    zero.optimizer.state_dict = forbidden
    summarize_zero_partitions(zero, parameters, include_gradients=True)


class _HugeSingleTensor:
    dtype = "torch.float32"
    device = "cuda:1"

    def numel(self):
        return 9_000_000_000

    def element_size(self):
        return 4

    def __iter__(self):
        raise AssertionError("a single tensor-like object must never be iterated")


def test_single_huge_gradient_partition_is_one_metadata_item():
    zero = SimpleNamespace(grads_in_partition=_HugeSingleTensor())
    report = summarize_zero_partitions(zero, [])
    collection = report["tensor_collections"]["current_gradient_partition_buffers"]
    assert collection["tensor_count"] == 1
    assert collection["logical_numel"] == 9_000_000_000
    assert collection["logical_view_bytes"] == 36_000_000_000
    assert collection["physical_storage_metadata_missing_count"] == 1
    physical = report["physical_storage_summary"]
    assert physical["complete_for_inventory_collections"] is False
    assert physical["storage_metadata_missing_count"] == 1


def test_local_fragment_ranking_excludes_unmapped_parameters():
    mapped = torch.nn.Parameter(torch.zeros(5))
    mapped._index_in_param_group = 0
    mapped._hp_mapping = _Mapping(1, 2, mapped[1:3])
    unmapped = torch.nn.Parameter(torch.zeros(100))
    report = summarize_zero_partitions(
        SimpleNamespace(), [("large_unmapped", unmapped), ("small_mapped", mapped)]
    )
    assert report["largest_model_tensors"][0]["name"] == "large_unmapped"
    assert report["largest_local_fragment_views"] == [
        {"name": "small_mapped", "fragment_numel": 2}
    ]
