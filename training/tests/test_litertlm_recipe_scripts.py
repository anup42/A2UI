from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_litertlm_recipe  # noqa: E402
import audit_litertlm_remote_source_recipe  # noqa: E402
import audit_litertlm_weight_recipe  # noqa: E402
import audit_gemma4_mobile_checkpoint_parity  # noqa: E402
import audit_gemma4_mobile_projection_parity  # noqa: E402
import audit_gemma4_mtp_assistant_parity  # noqa: E402
import build_fresh_random_quantized_graph  # noqa: E402
import build_converter_random_inventory_parity  # noqa: E402
import build_converter_random_topology_injection_parity  # noqa: E402
import build_random_official_topology_parity  # noqa: E402
import build_random_mobile_contract_parity  # noqa: E402
import build_random_tflite_recipe_parity  # noqa: E402


def test_mobile_recipe_audit_groups_observable_tensor_names():
    assert audit_litertlm_recipe._layer_group("layer_14/mlp/gating") == "mlp_layer_14"
    assert audit_litertlm_recipe._layer_group("layer_2/attn/q") == "self_attention"
    assert audit_litertlm_recipe._layer_group("per_layer_projection") == "per_layer"
    assert audit_litertlm_recipe._layer_scope("x/layer_15/mlp/linear") == "layer_15"
    assert audit_litertlm_recipe._operation_family("x/layer_15/mlp/linear") == "mlp"
    assert audit_litertlm_recipe._operation_family("x/layer_0/per_layer_embedding_gate") == "per_layer_embedding"


def test_random_recipe_plan_is_explicitly_non_training(tmp_path):
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        recipe="gemma4_mixed48",
        official_artifact=None,
        seed=42,
        execute=False,
        force=False,
    )

    plan = build_random_tflite_recipe_parity._plan(args)

    assert plan["training_executed"] is False
    assert plan["random_initialization"] is True
    assert plan["expected_public_layout"]["default_fully_connected_weight_type"] == "INT4"
    assert plan["expected_public_layout"]["per_layer_fully_connected_weight_type"] == "INT8"


def test_mobile_contract_recipe_is_explicit_and_static():
    recipe = build_random_mobile_contract_parity._mobile_recipe()

    assert [entry["op_config"]["weight_tensor_config"]["num_bits"] for entry in recipe] == [4, 2, 8]
    assert all(entry["op_config"]["activation_tensor_config"]["num_bits"] == 8 for entry in recipe)
    assert all(entry["op_config"]["skip_checks"] is True for entry in recipe)


def test_mobile_contract_plan_is_explicitly_non_training(tmp_path):
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        official_artifact=None,
        seed=42,
        calibration_samples=2,
        threads=1,
        execute=False,
        force=False,
    )

    plan = build_random_mobile_contract_parity._plan(args)

    assert plan["training_executed"] is False
    assert plan["private_recipe_recovered"] is False
    assert plan["observable_contract"]["weight_bits"] == [2, 4, 8]


def test_official_topology_plan_is_explicitly_non_training(tmp_path):
    args = argparse.Namespace(
        artifact=str(tmp_path / "official.litertlm"),
        output=str(tmp_path / "random.tflite"),
        model_type="tf_lite_mtp_drafter",
        seed=7,
        execute=False,
    )

    plan = build_random_official_topology_parity._plan(args)

    assert plan["random_weights"] is True
    assert plan["training_executed"] is False
    assert plan["private_recipe_recovered"] is False
    assert plan["exact_official_model_match"] is False
    assert plan["runtime_allocate"] is False


def test_official_topology_low_bit_packing_matches_ai_edge_order():
    # Four 2-bit signed values [0, 1, -2, -1] occupy one byte in the public
    # AI Edge order: least-significant group first.
    assert build_random_official_topology_parity._pack_low_bit(
        [0, 1, -2, -1], 2
    ).tolist() == [0xE4]
    assert build_random_official_topology_parity._pack_low_bit(
        [-2, -1, 0], 2
    ).tolist() == [0x0E]
    values = [-2, -1, 0, 1, 2, -3, 3, 0]
    packed = build_random_official_topology_parity._pack_low_bit(values, 4)
    assert build_random_official_topology_parity._unpack_low_bit(
        packed, 4, len(values)
    ).tolist() == values


def test_official_topology_reads_external_buffer_offsets():
    class FakeBuffer:
        def DataLength(self):
            return 0

        def Offset(self):
            return 3

        def Size(self):
            return 4

    class FakeModel:
        def Buffers(self, index):
            assert index == 1
            return FakeBuffer()

    payload = bytearray(b"0123456789")
    view, storage, offset, size = build_random_official_topology_parity._buffer_view(
        FakeModel(), 1, payload
    )

    assert storage == "external"
    assert (offset, size) == (3, 4)
    assert bytes(view) == b"3456"


def test_weight_recipe_audit_maps_gemma_linear_names_and_public_formula():
    import numpy as np

    name = "prefix/Gemma3DecoderLayer_4/path/Linear_gate_proj;"
    assert audit_litertlm_weight_recipe.canonical_source_key(
        "FULLY_CONNECTED", name, (2, 3)
    ) == "model.layers.4.mlp.gate_proj.weight"
    assert audit_litertlm_weight_recipe.canonical_source_key(
        "EMBEDDING_LOOKUP", "arith.constant", (8, 3)
    ) == "model.embed_tokens.weight"

    float_weight = np.asarray([[1.0, -2.0, 0.5], [0.25, -0.5, 0.75]], dtype=np.float32)
    quantized, scales = audit_litertlm_weight_recipe._expected_int8(float_weight)
    assert np.allclose(scales, [2.0 / 127.0, 0.75 / 127.0], rtol=0.0, atol=1e-9)
    assert quantized.tolist() == [[64, -127, 32], [42, -85, 127]]


def test_remote_recipe_audit_maps_canonical_gemma3_operator_order():
    assert audit_litertlm_remote_source_recipe.generic_gemma3_source_key(
        0, (1024, 640)
    ) == "model.layers.0.self_attn.q_proj.weight"
    assert audit_litertlm_remote_source_recipe.generic_gemma3_source_key(
        3, (640, 1024)
    ) == "model.layers.0.self_attn.o_proj.weight"
    assert audit_litertlm_remote_source_recipe.generic_gemma3_source_key(
        6, (640, 2048)
    ) == "model.layers.0.mlp.down_proj.weight"
    assert audit_litertlm_remote_source_recipe.generic_gemma3_source_key(
        7, (1024, 640)
    ) == "model.layers.1.self_attn.q_proj.weight"


def test_remote_recipe_bfloat16_decode_is_exact_for_known_bits():
    import numpy as np

    # 1.0 and -2.0 in bfloat16 little-endian representation.
    raw = np.asarray([0x3F80, 0xC000], dtype="<u2").tobytes()
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    values = bits.view(np.float32)
    assert values.tolist() == [1.0, -2.0]


def test_remote_recipe_resolves_windows_local_source_path(tmp_path):
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"not-a-real-checkpoint")
    assert audit_litertlm_remote_source_recipe._local_source_path(str(source)) == source


def test_fresh_random_graph_uses_full_signed_low_bit_ranges():
    import numpy as np

    packed, scales = build_fresh_random_quantized_graph._random_quantized_weight(
        (4, 8), 2, seed=42, ordinal=0
    )
    assert len(packed) == 4 * 8 // 4
    assert scales.shape == (4,)
    unpacked = build_random_official_topology_parity._unpack_low_bit(packed, 2, 32)
    assert int(unpacked.min()) >= -2
    assert int(unpacked.max()) <= 1
    assert np.all(scales > 0)


def test_converter_inventory_recipe_separates_static_and_dynamic_edges():
    static = {
        "bits": 4,
        "input_type_name": "INT8",
        "output_type_name": "INT8",
    }
    dynamic = {
        "bits": 8,
        "input_type_name": "FLOAT32",
        "output_type_name": "FLOAT32",
    }
    static_config = build_converter_random_inventory_parity._config_for_record(static)
    dynamic_config = build_converter_random_inventory_parity._config_for_record(dynamic)
    assert static_config["weight_tensor_config"]["num_bits"] == 4
    assert "activation_tensor_config" in static_config
    assert "activation_tensor_config" not in dynamic_config


def test_converter_inventory_builder_preserves_embedding_operator_kind():
    # The low-level fixture builder is exercised in the documented conversion
    # environment, where the generated ``tflite`` schema package is installed.
    # Keep the core training test suite usable without that optional dependency.
    pytest.importorskip("flatbuffers")
    pytest.importorskip("tflite")
    records = [
        {
            "operator": "EMBEDDING_LOOKUP",
            "bits": 8,
            "shape": [32, 8],
            "input_type_name": "INT32",
            "output_type_name": "FLOAT32",
        }
    ]
    data, mappings = build_converter_random_inventory_parity._build_random_float_tflite(
        records, seed=7
    )
    recipe = build_converter_random_inventory_parity._recipe(records)

    assert data[4:8] == b"TFL3"
    assert mappings[0]["official_operator"] == "EMBEDDING_LOOKUP"
    assert recipe[0]["operation"] == "EMBEDDING_LOOKUP"


def test_converter_injection_layout_compacts_packed_values():
    compact = build_converter_random_topology_injection_parity._compact_converter_layout(
        [
            {
                "ordinal": 0,
                "operator": "FULLY_CONNECTED",
                "bits": 8,
                "shape": [4, 8],
                "raw": b"packed",
                "scales": [0.1, 0.2, 0.3, 0.4],
                "scale_count": 4,
                "zero_point_count": 4,
                "quantized_dimension": 0,
                "zero_points_all_zero": True,
                "input_type": "FLOAT32",
                "output_type": "FLOAT32",
            }
        ]
    )
    assert compact[0]["shape"] == [4, 8]
    assert "raw" not in compact[0]
    assert "scales" not in compact[0]


def test_converter_injection_reports_random_network_parity_separately_from_values():
    report = build_converter_random_topology_injection_parity._random_quantized_network_match(
        converter_inventory_match=True,
        converter_fc_match=True,
        converter_embedding_match=True,
        official_structure_match=True,
        official_layout_match=True,
        execution_topology_match=True,
        execution_layout_match=True,
    )

    assert report["same"] is True
    assert report["scope"] == "topology_and_quantization_layout_only"
    assert report["not_value_or_model_identity"] is True

    failed = build_converter_random_topology_injection_parity._random_quantized_network_match(
        converter_inventory_match=True,
        converter_fc_match=True,
        converter_embedding_match=True,
        official_structure_match=True,
        official_layout_match=False,
        execution_topology_match=True,
        execution_layout_match=True,
    )
    assert failed["same"] is False
    assert failed["checks"]["official_layout_match"] is False


def test_converter_injection_constant_digest_is_ordered_and_scale_sensitive():
    import numpy as np

    records = [
        {
            "operator": "FULLY_CONNECTED",
            "bits": 4,
            "shape": [2, 4],
            "raw": b"abcd",
            "scales": np.asarray([0.25, 0.5], dtype=np.float32),
        }
    ]
    same = build_converter_random_topology_injection_parity._constants_digest(records)
    changed = build_converter_random_topology_injection_parity._constants_digest(
        [{**records[0], "scales": np.asarray([0.25, 0.5001], dtype=np.float32)}]
    )
    assert same != changed


def test_converter_injection_splits_large_float_inventory_before_flatbuffer_limit():
    records = [
        {"shape": [128, 128]},
        {"shape": [128, 128]},
        {"shape": [128, 128]},
    ]
    chosen, ranges = build_converter_random_topology_injection_parity._converter_batch_ranges(
        records, None
    )

    assert chosen == len(records)
    assert ranges == [(0, len(records))]

    huge = records + [{"shape": [12288, 1536]}] * 40
    huge[0] = {"shape": [262144, 1536]}
    chosen, ranges = build_converter_random_topology_injection_parity._converter_batch_ranges(
        huge, None
    )
    assert chosen == 8
    assert ranges[0][0] == 0
    assert ranges[0][1] <= 8
    assert all(end > start for start, end in ranges)
    assert ranges[-1][1] == len(huge)


def test_converter_injection_package_boundary_replaces_only_selected_section(tmp_path):
    artifact = tmp_path / "official.litertlm"
    artifact.write_bytes(b"prefix" + b"section" + b"suffix")
    section = {"begin_offset": 6, "size": 7}
    injected = b"INJECT!"

    report = build_converter_random_topology_injection_parity._package_boundary_report(
        artifact, section, injected
    )
    output = tmp_path / "injected.litertlm"
    build_converter_random_topology_injection_parity._write_injected_package(
        artifact, section, injected, output
    )

    assert report["bytes_outside_selected_section_unchanged"] is True
    assert output.read_bytes() == b"prefix" + injected + b"suffix"


def test_converter_injection_rebuilds_modelt_graph_and_preserves_constants():
    import numpy as np

    pytest.importorskip("flatbuffers")
    pytest.importorskip("ai_edge_litert.schema_py_generated")
    records = [
        {
            "ordinal": 0,
            "operator": "FULLY_CONNECTED",
            "official_subgraph": 0,
            "official_operator_index": 0,
            "bits": 8,
            "shape": [4, 3],
            "type_value": 9,
        }
    ]
    official = build_fresh_random_quantized_graph._build_fresh_tflite(records, seed=11)
    converter = [
        {
            "ordinal": 0,
            "operator": "FULLY_CONNECTED",
            "bits": 8,
            "shape": [4, 3],
            "raw": bytes(range(12)),
            "scales": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        }
    ]
    rebuilt, details = build_converter_random_topology_injection_parity._rebuild_graph_with_constants(
        official, records, converter
    )

    assert details["modelt_rebuilt"] is True
    assert rebuilt[4:8] == b"TFL3"
    expected = build_converter_random_topology_injection_parity._constants_digest(converter)
    observed = build_converter_random_topology_injection_parity._official_constants_digest(
        rebuilt, records
    )
    assert observed == expected


def test_mtp_assistant_audit_has_explicit_23_weight_mapping():
    keys = audit_gemma4_mtp_assistant_parity._source_keys()
    assert len(keys) == 23
    assert keys[0] == "pre_projection.weight"
    assert keys[1] == "model.layers.0.self_attn.q_proj.weight"
    assert keys[20] == "model.layers.3.mlp.down_proj.weight"
    assert keys[21] == "model.embed_tokens.weight"
    assert keys[22] == "post_projection.weight"


def test_mobile_projection_candidate_rounding_is_signed_and_clipped():
    import numpy as np

    values = np.asarray([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 200.0], dtype=np.float32)
    assert audit_gemma4_mobile_projection_parity._round_codes(
        values, "rint"
    ).tolist() == [-2, -2, 0, 0, 2, 2, 127]
    assert audit_gemma4_mobile_projection_parity._round_codes(
        values, "half_up"
    ).tolist() == [-2, -1, 0, 1, 2, 3, 127]
    assert audit_gemma4_mobile_projection_parity._round_codes(
        values, "ties_to_zero"
    ).tolist() == [-2, -1, 0, 0, 1, 2, 127]


def test_mobile_projection_scale_candidates_keep_float32_shape():
    import numpy as np

    values = np.asarray([[1.0, -2.0], [0.25, -0.5]], dtype=np.float32)
    candidates = audit_gemma4_mobile_projection_parity._scale_candidates(values)
    assert set(candidates) == {
        "max_abs_f32_div_f32",
        "max_abs_f32_div_f64",
        "max_abs_f64_div_f64",
    }
    assert all(item.dtype == np.float32 and item.shape == (2,) for item in candidates.values())


def test_mobile_projection_bfloat16_interval_helpers_accept_consistent_values():
    import numpy as np

    source = np.asarray([[1.0, -2.0], [0.25, -0.5]], dtype=np.float32)
    scales = np.asarray([2.0 / 127.0, 0.5 / 127.0], dtype=np.float32)
    codes = np.asarray([[64, -127], [64, -127]], dtype=np.int8)
    report = audit_gemma4_mobile_projection_parity._bfloat16_interval_report(
        source,
        codes.tobytes(),
        scales,
    )

    assert report["source_precision"] == "BF16"
    assert report["all_scales_are_interval_compatible"] is True
    assert report["all_codes_are_interval_compatible"] is True
    assert report["necessary_condition_only"] is True


def test_mobile_projection_bfloat16_interval_helpers_reject_incompatible_scale():
    import numpy as np

    source = np.asarray([[1.0, -2.0]], dtype=np.float32)
    codes = np.asarray([[64, -127]], dtype=np.int8)
    report = audit_gemma4_mobile_projection_parity._bfloat16_interval_report(
        source,
        codes.tobytes(),
        np.asarray([1.0], dtype=np.float32),
    )

    assert report["all_scales_are_interval_compatible"] is False
    assert report["scale_rows_outside_bfloat16_max_abs_interval"] == 1


def test_mtp_assistant_public_candidate_uses_signed_axis_zero_packing():
    import numpy as np

    raw, scales = audit_gemma4_mtp_assistant_parity._pack_source(
        np.asarray([[1.0, -2.0, 0.5], [0.25, -0.5, 0.75]], dtype=np.float32),
        bits=8,
    )
    assert np.allclose(scales, [2.0 / 127.0, 0.75 / 127.0], rtol=0.0, atol=1e-8)
    assert np.frombuffer(raw, dtype=np.int8).reshape(2, 3).tolist() == [
        [64, -127, 32],
        [42, -85, 127],
    ]


def test_converter_inventory_layout_includes_activation_quantization():
    base = {
        "bits": 4,
        "shape": [8, 4],
        "scale_count": 8,
        "zero_point_count": 8,
        "quantized_dimension": 0,
        "input_type": "INT8",
        "output_type": "INT8",
        "input_quantization": {
            "scale_count": 1,
            "zero_point_count": 1,
            "quantized_dimension": 0,
            "zero_points_all_zero": True,
        },
        "output_quantization": {
            "scale_count": 1,
            "zero_point_count": 1,
            "quantized_dimension": 0,
            "zero_points_all_zero": True,
        },
    }
    changed = dict(base)
    changed["output_quantization"] = {
        **base["output_quantization"],
        "zero_points_all_zero": False,
    }
    assert build_converter_random_inventory_parity._layout_key(
        base
    ) != build_converter_random_inventory_parity._layout_key(changed)


def test_gemma4_mobile_checkpoint_signed_code_transform():
    # The public mobile checkpoint uses unsigned-offset codes.  The supplied
    # LiteRT artifact uses the same low-bit code order in signed form.
    assert audit_gemma4_mobile_checkpoint_parity._signed_litert_bytes(
        bytes([0xF7, 0xA9]), 4
    ) == bytes([0x7F, 0x21])
    assert audit_gemma4_mobile_checkpoint_parity._signed_litert_bytes(
        bytes([0xE4]), 2
    ) == bytes([0x4E])
    assert audit_gemma4_mobile_checkpoint_parity._signed_litert_bytes(
        bytes([0x80, 0xFF]), 8
    ) == bytes([0x80, 0xFF])


def test_gemma4_mobile_checkpoint_source_key_mapping():
    assert audit_gemma4_mobile_checkpoint_parity._source_key(
        "tf_lite_embedder", "embedder.lookup_embedding_table/composite", 0
    ) == ("model.language_model.embed_tokens.embedding_quantized", None)
    assert audit_gemma4_mobile_checkpoint_parity._source_key(
        "tf_lite_per_layer_embedder", "per_layer_embedder.lookup_embedding_table/composite7", 7
    ) == ("model.language_model.embed_tokens_per_layer.embedding_quantized", 7)
    assert audit_gemma4_mobile_checkpoint_parity._source_key(
        "tf_lite_prefill_decode", "layer_3/mlp/gating_einsum1", 0
    ) == ("model.language_model.layers.3.mlp.gate_proj.weight", None)
