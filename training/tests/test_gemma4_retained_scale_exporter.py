from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_gemma4_retained_scale_litertlm as exporter


@pytest.mark.parametrize(
    ("bits", "codes", "expected"),
    [
        (2, [-2, -1, 0, 1], bytes([0x4E])),
        (4, [-8, 7, -1, 0], bytes([0x78, 0x0F])),
        (8, [-127, -1, 0, 127], bytes([0x81, 0xFF, 0x00, 0x7F])),
    ],
)
def test_litert_packer_uses_signed_twos_complement_not_source_offsets(
    bits, codes, expected
):
    values = np.asarray(codes, dtype=np.int8)
    packed = exporter._pack_litert_codes(values, bits)

    assert packed == expected
    np.testing.assert_array_equal(
        exporter._unpack_litert_codes(packed, bits, len(codes)), values
    )


def test_litert_packer_rejects_codes_outside_full_low_bit_range():
    with pytest.raises(exporter.RetainedScaleExportError, match="exceed"):
        exporter._pack_litert_codes(np.asarray([-3, 0], dtype=np.int8), 2)


def test_official_artifact_identity_rejects_self_consistent_unpinned_hash():
    arbitrary = "f" * 64
    rejected = exporter._official_artifact_sha_report(arbitrary, arbitrary)
    accepted = exporter._official_artifact_sha_report(
        exporter.OFFICIAL_LITERTLM_SHA256,
        exporter.OFFICIAL_LITERTLM_SHA256,
    )

    assert rejected["verified"] is False
    assert rejected["checks"]["declared_observed_sha_match"] is True
    assert accepted["verified"] is True


def test_official_source_keys_normalize_to_text_only_hf_namespace():
    assert (
        exporter._normalize_official_key(
            "model.language_model.layers.3.self_attn.q_proj.weight"
        )
        == "model.layers.3.self_attn.q_proj.weight"
    )
    assert (
        exporter._normalize_official_key(
            "model.language_model.per_layer_model_projection.weight"
        )
        == "model.per_layer_model_projection.weight"
    )
    assert exporter._normalize_official_key("lm_head.weight") == "lm_head.weight"
    # Shared-KV K/V tensors are absent from the text-only materialized inventory.
    assert (
        exporter._normalize_official_key(
            "model.language_model.layers.20.self_attn.k_proj.weight"
        )
        is None
    )


def _mutable_key_bits() -> dict[str, int]:
    result: dict[str, int] = {}
    for layer in range(35):
        attention = (
            ("q_proj", "k_proj", "v_proj", "o_proj")
            if layer < 15
            else (
                "q_proj",
                "o_proj",
            )
        )
        for projection in attention:
            result[f"model.layers.{layer}.self_attn.{projection}.weight"] = 4
        mlp_bits = 4 if layer < 15 else 2
        for projection in ("gate_proj", "up_proj", "down_proj"):
            result[f"model.layers.{layer}.mlp.{projection}.weight"] = mlp_bits
    return result


def _source_key(hf_key: str) -> str:
    if hf_key == "lm_head.weight":
        return hf_key
    if hf_key.startswith("model."):
        return "model.language_model." + hf_key[len("model.") :]
    raise AssertionError(hf_key)


class _FakeQParams:
    def __init__(self, inventory):
        self.inventory = inventory

    def trainable_projection_weight_keys(self):
        return tuple(sorted(_mutable_key_bits()))


def test_scope_is_exact_205_projection_bijection_and_frozen_72(monkeypatch):
    mutable = _mutable_key_bits()
    frozen = {
        key: (2 if key == "lm_head.weight" else 8)
        for key in exporter._expected_frozen_keys()
    }
    key_bits = {**mutable, **frozen}
    ordered_keys = sorted(key_bits)
    records = [
        {
            "ordinal": index,
            "official_buffer": 1000 + index,
            "bits": key_bits[key],
            "shape": [2, 4],
        }
        for index, key in enumerate(ordered_keys)
    ]
    inventory = {
        key: {
            "bits": bits,
            "weight_shape": [2, 4],
            "scale_shape": [2, 1],
            "group_size": None,
        }
        for key, bits in mutable.items()
    }
    monkeypatch.setattr(
        exporter,
        "_canonical_inventory_keys",
        lambda _records, _family, _model_type: [
            _source_key(key) for key in ordered_keys
        ],
    )

    report, mutable_records, frozen_records = exporter._scope_report(
        records, _FakeQParams(inventory)
    )

    assert report["verified"] is True
    assert len(mutable_records) == 205
    assert len(frozen_records) == 72
    assert report["mutable_bit_histogram"] == {2: 60, 4: 145}
    assert report["frozen_bit_histogram"] == {2: 1, 8: 71}
    assert len({item["official_buffer"] for item in mutable_records}) == 205


def test_scope_rejects_count_preserving_wrong_frozen_key(monkeypatch):
    mutable = _mutable_key_bits()
    frozen = sorted(exporter._expected_frozen_keys())
    frozen[-1] = "model.layers.34.unexpected_frozen.weight"
    key_bits = {
        **mutable,
        **{key: (2 if key == "lm_head.weight" else 8) for key in frozen},
    }
    ordered_keys = sorted(key_bits)
    records = [
        {
            "ordinal": index,
            "official_buffer": index + 1,
            "bits": key_bits[key],
            "shape": [2, 4],
        }
        for index, key in enumerate(ordered_keys)
    ]
    inventory = {
        key: {
            "bits": bits,
            "weight_shape": [2, 4],
            "scale_shape": [2, 1],
            "group_size": None,
        }
        for key, bits in mutable.items()
    }
    monkeypatch.setattr(
        exporter,
        "_canonical_inventory_keys",
        lambda _records, _family, _model_type: [
            _source_key(key) for key in ordered_keys
        ],
    )

    with pytest.raises(
        exporter.RetainedScaleExportError, match="frozen_key_inventory_72"
    ):
        exporter._scope_report(records, _FakeQParams(inventory))


def test_projection_quantization_uses_retained_scale_ties_even_and_reports_clipping():
    values = np.asarray(
        [[-1.25, -0.75, -0.25, 0.25, 0.75, 1.25, -4.0, 4.0]],
        dtype=np.float32,
    )
    scales = np.asarray([[0.5]], dtype=np.float32)

    raw, report = exporter._quantize_projection(
        values, scales, bits=2, working_set_bytes=64
    )
    codes = exporter._unpack_litert_codes(raw, 2, values.size)

    # np.rint is ties-to-even, followed by W2 clipping to [-2, 1].
    assert codes.tolist() == [-2, -2, 0, 0, 1, 1, -2, 1]
    # Before clipping, ties-to-even produces [-2,-2,0,0,2,2,-8,8].
    assert report["clipped_low_count"] == 1
    assert report["clipped_high_count"] == 3
    assert report["litert_pack_roundtrip"] is True


def test_normalized_lora_module_handles_peft_and_clippable_wrappers():
    assert (
        exporter._normalized_lora_module(
            "base_model.model.model.layers.4.self_attn.q_proj"
        )
        == "model.layers.4.self_attn.q_proj"
    )


def test_adapter_mapping_accepts_only_plain_exact_lora_pairs(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 1,
                "lora_alpha": 1,
                "fan_in_fan_out": False,
                "use_rslora": False,
                "use_dora": False,
                "rank_pattern": {},
                "alpha_pattern": {},
                "bias": "none",
                "lora_bias": False,
                "modules_to_save": None,
            }
        ),
        encoding="utf-8",
    )
    module = "base_model.model.model.layers.0.self_attn.q_proj"

    class FakeAdapter:
        def __init__(self):
            self.entries = {
                f"{module}.lora_A.weight": {"shape": [1, 2], "dtype": "F32"},
                f"{module}.lora_B.weight": {"shape": [3, 1], "dtype": "F32"},
            }

        @staticmethod
        def describe():
            return {"tensor_count": 2}

    class FakeQParams:
        def __init__(self):
            self.inventory = {
                "model.layers.0.self_attn.q_proj.weight": {"weight_shape": [3, 2]}
            }

    records = [{"hf_weight_key": "model.layers.0.self_attn.q_proj.weight"}]
    monkeypatch.setattr(exporter, "EXPECTED_MUTABLE_COUNT", 1)

    report, mapping, scaling = exporter._adapter_mapping_report(
        adapter_dir,
        FakeAdapter(),
        records,
        FakeQParams(),
        {"lora": {"r": 1, "alpha": 1}},
    )

    assert report["verified"] is True
    assert mapping[records[0]["hf_weight_key"]]["a"].endswith("lora_A.weight")
    assert scaling == 1.0

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    config["use_dora"] = True
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(exporter.RetainedScaleExportError, match="use_dora_false"):
        exporter._adapter_mapping_report(
            adapter_dir,
            FakeAdapter(),
            records,
            FakeQParams(),
            {"lora": {"r": 1, "alpha": 1}},
        )


def test_a8_report_checks_every_mutable_weight_alias_edge(monkeypatch):
    class Tensor:
        def __init__(self, scale_hex):
            self.scale_hex = scale_hex

        @staticmethod
        def Type():
            return 9

    class Operator:
        @staticmethod
        def InputsLength():
            return 2

        @staticmethod
        def Inputs(index):
            return (0, 1)[index]

        @staticmethod
        def OutputsLength():
            return 1

        @staticmethod
        def Outputs(_index):
            return 2

        @staticmethod
        def OpcodeIndex():
            return 0

    class Subgraph:
        def __init__(self, input_hex, output_hex):
            self.tensors = [Tensor(input_hex), Tensor("weight"), Tensor(output_hex)]

        @staticmethod
        def OperatorsLength():
            return 1

        @staticmethod
        def Operators(_index):
            return Operator()

        def Tensors(self, index):
            return self.tensors[index]

    class Code:
        @staticmethod
        def BuiltinCode():
            return exporter._FULLY_CONNECTED

    class Model:
        def __init__(self):
            self.subgraphs = [
                Subgraph("01020304", "05060708"),
                Subgraph("01020304", "05060708"),
            ]

        def SubgraphsLength(self):
            return len(self.subgraphs)

        def Subgraphs(self, index):
            return self.subgraphs[index]

        @staticmethod
        def OperatorCodes(_index):
            return Code()

    key = "model.layers.0.self_attn.q_proj.weight"
    record = {"hf_weight_key": key}
    model = Model()
    monkeypatch.setattr(exporter, "_schema_model", lambda _section: model)
    monkeypatch.setattr(
        exporter,
        "_official_weight_alias_groups",
        lambda _model, _records: [
            {
                "aliases": [
                    {"subgraph_index": 0, "tensor_index": 1},
                    {"subgraph_index": 1, "tensor_index": 1},
                ]
            }
        ],
    )
    monkeypatch.setattr(
        exporter,
        "_quantization_vectors",
        lambda tensor: {"scales_hex": tensor.scale_hex},
    )
    monkeypatch.setattr(exporter, "EXPECTED_MUTABLE_FC_ALIAS_EDGES", 2)
    monkeypatch.setattr(exporter, "EXPECTED_MUTABLE_A8_RECORDS", 4)

    class QParams:
        def __init__(self):
            self.inventory = {
                key: {
                    "input_activation_scale_f32_le_hex": "01020304",
                    "output_activation_scale_f32_le_hex": "05060708",
                }
            }

    report = exporter._activation_a8_report(b"fake", [record], QParams())

    assert report["verified"] is True
    assert report["fc_edge_count"] == 2
    assert report["record_count"] == 4
    assert (
        exporter._normalized_lora_module(
            "base_model.model.model.layers.4.self_attn.q_proj.linear"
        )
        == "model.layers.4.self_attn.q_proj"
    )


def test_base_lora_reconstruction_reports_numeric_code_and_norm_parity():
    base = np.asarray([[0.0, 0.5], [1.0, -0.5]], dtype=np.float32)
    lora_a = np.asarray([[1.0, -1.0]], dtype=np.float32)
    lora_b = np.asarray([[0.25], [-0.5]], dtype=np.float32)
    candidate = base + lora_b @ lora_a
    scale = np.asarray([[0.25], [0.5]], dtype=np.float32)
    candidate_raw, _ = exporter._quantize_projection(
        candidate, scale, bits=4, working_set_bytes=128
    )

    report = exporter._base_lora_projection_parity(
        candidate,
        base,
        lora_a,
        lora_b,
        scale,
        candidate_dtype="F32",
        base_dtype="F32",
        adapter_dtype="F32",
        scaling=1.0,
        bits=4,
        candidate_raw=candidate_raw,
        working_set_bytes=128,
    )

    assert report["candidate_base_dtype_match"] is True
    assert report["numerical_exact"] is True
    assert report["numerical_close"] is True
    assert report["retained_scale_code_exact"] is True
    assert report["norms_finite"] is True
    assert report["delta_nonzero"] is True
    assert report["delta_to_base_l2_ratio"] > 0.0


def test_base_lora_reconstruction_tracks_bfloat16_merge_rounding():
    torch = pytest.importorskip("torch")
    base = np.asarray([[0.101, -0.203], [0.307, -0.409]], dtype=np.float32)
    lora_a = np.asarray([[0.113, -0.127]], dtype=np.float32)
    lora_b = np.asarray([[0.137], [-0.149]], dtype=np.float32)
    base_bf16 = torch.from_numpy(base).to(torch.bfloat16)
    delta_bf16 = (torch.from_numpy(lora_b) @ torch.from_numpy(lora_a)).to(
        torch.bfloat16
    )
    candidate_t = base_bf16.clone()
    candidate_t.add_(delta_bf16)
    candidate = candidate_t.float().numpy()
    scale = np.asarray([[0.01], [0.01]], dtype=np.float32)
    candidate_raw, _ = exporter._quantize_projection(
        candidate, scale, bits=4, working_set_bytes=128
    )

    report = exporter._base_lora_projection_parity(
        candidate,
        base_bf16.float().numpy(),
        lora_a,
        lora_b,
        scale,
        candidate_dtype="BF16",
        base_dtype="BF16",
        adapter_dtype="F32",
        scaling=1.0,
        bits=4,
        candidate_raw=candidate_raw,
        working_set_bytes=128,
    )

    assert report["candidate_base_bfloat16"] is True
    assert report["numerical_exact"] is True
    assert report["retained_scale_code_exact"] is True


def test_restore_official_payload_proof_detects_any_change_outside_selected_buffers(
    monkeypatch,
):
    offsets = {11: (4, 2), 12: (10, 2)}

    def fake_groups(_model, records):
        return [
            {
                "buffer_index": int(record["official_buffer"]),
                "aliases": [{"tensor": object()}],
            }
            for record in records
        ]

    def fake_view(_model, buffer_index, section):
        offset, size = offsets[int(buffer_index)]
        return memoryview(section)[offset : offset + size], "external", offset, size

    monkeypatch.setattr(exporter, "_schema_model", lambda section: object())
    monkeypatch.setattr(exporter, "_official_weight_alias_groups", fake_groups)
    monkeypatch.setattr(exporter, "_buffer_view", fake_view)
    selected = [{"official_buffer": 11}, {"official_buffer": 12}]
    official = bytes(range(20))
    candidate = bytearray(official)
    candidate[4:6] = b"\xaa\xbb"
    candidate[10:12] = b"\xcc\xdd"

    exact = exporter._restore_official_payloads(official, candidate, selected)
    assert exact["restored_official_payload_target_byte_exact"] is True

    candidate[0] ^= 0xFF
    forbidden = exporter._restore_official_payloads(official, candidate, selected)
    assert forbidden["restored_official_payload_target_byte_exact"] is False


def test_buffer_diff_requires_trained_code_change_and_rejects_frozen_change(
    monkeypatch,
):
    offsets = {11: (4, 2), 12: (10, 2)}

    def fake_groups(_model, records):
        return [
            {
                "buffer_index": int(record["official_buffer"]),
                "aliases": [{"tensor": object()}],
            }
            for record in records
        ]

    def fake_view(_model, buffer_index, section):
        offset, size = offsets[int(buffer_index)]
        return memoryview(section)[offset : offset + size], "external", offset, size

    monkeypatch.setattr(exporter, "_schema_model", lambda section: object())
    monkeypatch.setattr(exporter, "_official_weight_alias_groups", fake_groups)
    monkeypatch.setattr(exporter, "_buffer_view", fake_view)
    monkeypatch.setattr(exporter, "EXPECTED_FROZEN_COUNT", 1)
    records = [{"official_buffer": 11}, {"official_buffer": 12}]
    mutable = [records[0]]
    official = bytes(range(20))

    unchanged = exporter._buffer_diff_report(
        official, bytearray(official), records, mutable
    )
    assert unchanged["at_least_one_trained_code_changed"] is False
    assert unchanged["frozen_72_byte_exact"] is True

    trained = bytearray(official)
    trained[4] ^= 0xFF
    changed = exporter._buffer_diff_report(official, trained, records, mutable)
    assert changed["at_least_one_trained_code_changed"] is True
    assert changed["changed_buffers_within_expected_205"] is True

    frozen = bytearray(official)
    frozen[10] ^= 0xFF
    rejected = exporter._buffer_diff_report(official, frozen, records, mutable)
    assert rejected["changed_buffers_within_expected_205"] is False
    assert rejected["frozen_72_byte_exact"] is False


def test_exclusive_package_write_and_publish_never_overwrite(tmp_path):
    official = tmp_path / "official.bin"
    official.write_bytes(b"prefixOLDsuffix")
    partial = tmp_path / "candidate.partial"
    output = tmp_path / "candidate.litertlm"
    section = {"begin_offset": 6, "size": 3}

    exporter._write_package_exclusive(official, section, b"NEW", partial)
    assert partial.read_bytes() == b"prefixNEWsuffix"
    with pytest.raises(FileExistsError):
        exporter._write_package_exclusive(official, section, b"NEW", partial)

    exporter._link_no_clobber(partial, output)
    assert output.read_bytes() == partial.read_bytes()
    with pytest.raises(exporter.RetainedScaleExportError, match="overwrite"):
        exporter._link_no_clobber(partial, output)


def test_buffer_compare_and_restore_are_chunk_bounded():
    source = np.arange(41, dtype=np.uint8)
    target = bytearray(41)

    assert exporter._buffer_views_equal(source, source.copy(), chunk_size=7)
    assert not exporter._buffer_views_equal(source, source[:-1], chunk_size=7)
    changed = source.copy()
    changed[29] ^= 0xFF
    assert not exporter._buffer_views_equal(source, changed, chunk_size=7)
    exporter._copy_buffer_view(source, target, chunk_size=7)
    assert bytes(target) == source.tobytes()


class _MetadataQParams:
    contract_sha256 = "a" * 64
    scale_storage_sha256 = "b" * 64

    def trainable_projection_weight_keys(self):
        return tuple(sorted(_mutable_key_bits()))


def test_best_adapter_provenance_requires_retained_best_golden_metadata(tmp_path):
    adapter = tmp_path / "best_golden_checkpoint"
    adapter.mkdir()
    adapter_file = adapter / "adapter_model.safetensors"
    adapter_file.write_bytes(b"adapter")
    config = tmp_path / "resolved_training_config.yaml"
    config.write_text("qat: {}\n", encoding="utf-8")
    keys = sorted(_mutable_key_bits())
    bindings = {
        f"module_{index}": {"weight_key": key} for index, key in enumerate(keys)
    }
    adapter_record = {
        "path": adapter_file.name,
        "size": adapter_file.stat().st_size,
        "sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
    }
    metadata = {
        "training_metadata_version": 4,
        "checkpoint_role": "best_golden",
        "training_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "best_golden_eval": {
            "metric": "generation_reward_v5_4_avg",
            "metric_value": 80.0,
        },
        "adapter_checkpoints": [{"role": "best_golden", "files": [adapter_record]}],
        "numeric_preflight": {
            "passed": True,
            "greedy_generation": {"passed": True},
        },
        "qat": {
            "wrapped_effective_lora_count": 205,
            "retained_qparams_binding_count": 205,
            "retained_qparams_bindings": bindings,
            "retained_qparams": {
                "contract_sha256": "a" * 64,
                "scale_storage_sha256": "b" * 64,
            },
            "spec": {
                "scale_mode": "retained_mobile",
                "fixed_scale_required": True,
                "fixed_activation_scale_required": True,
                "effective_lora_only": True,
                "ste_gradient": "clipped",
                "quantize_embeddings": False,
                "expected_effective_lora_modules": 205,
            },
        },
        "mobile_training_seed": {
            "verified": True,
            "manifest_sha256": "c" * 64,
            "transformation_plan_sha256": "d" * 64,
        },
    }
    (adapter / "training_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    seed_report = {
        "manifest_sha256": "c" * 64,
        "transformation_plan_sha256": "d" * 64,
    }

    report = exporter._best_adapter_provenance_report(
        adapter, config, qparams=_MetadataQParams(), seed_report=seed_report
    )
    assert report["verified"] is True
    assert report["checks"]["legacy_metadata_rejected"] is True

    metadata["qat"]["spec"]["scale_mode"] = "dynamic"
    (adapter / "training_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    rejected = exporter._best_adapter_provenance_report(
        adapter, config, qparams=_MetadataQParams(), seed_report=seed_report
    )
    assert rejected["verified"] is False
    assert rejected["checks"]["legacy_metadata_rejected"] is False
