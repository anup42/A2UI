from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

flatbuffers = pytest.importorskip("flatbuffers")
schema = pytest.importorskip("ai_edge_litert.schema_py_generated")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_fresh_random_quantized_graph
import build_random_official_topology_parity


def _shared_weight_model() -> bytearray:
    records = [
        {
            "ordinal": 0,
            "operator": "FULLY_CONNECTED",
            "official_subgraph": 0,
            "official_operator_index": 0,
            "official_buffer": 1,
            "bits": 8,
            "shape": [4, 3],
            "type_value": 9,
        }
    ]
    base = build_fresh_random_quantized_graph._build_fresh_tflite(records, seed=7)
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(base, 0))
    alias = copy.deepcopy(model.subgraphs[0])
    alias.name = b"shared_weight_alias"
    model.subgraphs.append(alias)
    builder = flatbuffers.Builder(len(base) * 2 + 1024)
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    return bytearray(builder.Output())


def test_topology_randomizer_updates_every_shared_weight_alias():
    section = _shared_weight_model()
    report = build_random_official_topology_parity._randomize_section(
        section, seed=42, verify_weight_encoding=True
    )
    model = build_random_official_topology_parity._schema_model(section)
    scales = [
        np.asarray(
            model.Subgraphs(subgraph_index)
            .Tensors(1)
            .Quantization()
            .ScaleAsNumpy(),
            dtype=np.float32,
        ).copy()
        for subgraph_index in range(2)
    ]

    assert report["randomized_buffer_count"] == 1
    assert report["randomized_weight_alias_count"] == 2
    assert report["shared_weight_buffer_count"] == 1
    assert report["alias_count_histogram"] == {2: 1}
    assert report["weight_encoding"]["roundtrip_verified_buffer_count"] == 1
    assert np.array_equal(scales[0], scales[1])
