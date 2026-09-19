from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_fresh_random_quantized_graph as inventory_builder
import build_gemma4_retained_scale_litertlm as exporter


def _package_sections():
    target = {
        "index": 1, "data_type_name": "TFLiteModel",
        "items": [{"key": "model_type", "value": exporter.TARGET_MODEL_TYPE}],
        "begin_offset": 20, "end_offset": 40, "size": 20,
    }
    mtp = {
        "index": 0, "data_type_name": "TFLiteModel",
        "items": [{"key": "model_type", "value": exporter.MTP_MODEL_TYPE}],
        "begin_offset": 10, "end_offset": 20, "size": 10,
    }
    return {"sections": [mtp, target]}, target, mtp


def test_real_inventory_return_is_a_section_and_requires_separate_package_inspection(
    tmp_path, monkeypatch
):
    official = tmp_path / "official.litertlm"
    official.write_bytes(b"P" * 100)
    package, target, mtp = _package_sections()
    inspections = []

    def inspect(path, *, inspect_tflite):
        assert path == official
        assert inspect_tflite is False
        inspections.append(path)
        return copy.deepcopy(package)

    monkeypatch.setattr(inventory_builder, "inspect_litertlm", inspect)
    monkeypatch.setattr(inventory_builder, "_operator_names", dict)
    monkeypatch.setattr(
        inventory_builder, "_schema_model",
        lambda _data: SimpleNamespace(SubgraphsLength=lambda: 0),
    )
    # Exercise the real extractor control flow, mocking only the tiny package
    # and graph readers. Do not replace its (section, records) return contract.
    section, records = inventory_builder._extract_inventory(
        official, exporter.TARGET_MODEL_TYPE, include_embeddings=True, max_weights=None
    )
    assert section == target
    assert "sections" not in section
    assert records == []
    monkeypatch.setattr(exporter, "inspect_litertlm", inspect)

    resolved_package, resolved_target, resolved_mtp = exporter._inspect_official_model_sections(
        official, section
    )

    assert resolved_package == package
    assert resolved_target == target
    assert resolved_mtp == mtp
    assert len(inspections) == 2


@pytest.mark.parametrize("model_type", [exporter.TARGET_MODEL_TYPE, exporter.MTP_MODEL_TYPE])
@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_official_sections_fail_closed_when_missing_or_ambiguous(monkeypatch, model_type, change):
    package, target, mtp = _package_sections()
    section = target if model_type == exporter.TARGET_MODEL_TYPE else mtp
    if change == "missing":
        package["sections"].remove(section)
    else:
        package["sections"].append(copy.deepcopy(section))
    monkeypatch.setattr(exporter, "inspect_litertlm", lambda *_args, **_kwargs: package)
    with pytest.raises(exporter.RetainedScaleExportError, match="official target/MTP sections"):
        exporter._inspect_official_model_sections(Path("unused"), target)


@pytest.mark.parametrize("change", ["package", "range", "metadata"])
def test_official_sections_require_exact_inventory_target_identity(monkeypatch, change):
    package, target, _ = _package_sections()
    section = copy.deepcopy(target)
    if change == "package":
        section = package
    elif change == "range":
        section["begin_offset"] += 1
    else:
        section["items"][0]["value"] = exporter.MTP_MODEL_TYPE
    monkeypatch.setattr(exporter, "inspect_litertlm", lambda *_args, **_kwargs: package)
    with pytest.raises(exporter.RetainedScaleExportError, match="inventory section differs"):
        exporter._inspect_official_model_sections(Path("unused"), section)


def test_official_sections_preserve_canonical_inspection_failure(monkeypatch):
    def inspect(*_args, **_kwargs):
        raise exporter.LiteRTLMInspectionError("Invalid section range")

    monkeypatch.setattr(exporter, "inspect_litertlm", inspect)
    with pytest.raises(exporter.RetainedScaleExportError, match="Invalid section range"):
        exporter._inspect_official_model_sections(Path("unused"), {})


def test_export_plan_uses_full_package_report_not_277_weight_section(tmp_path, monkeypatch):
    official = tmp_path / "official.litertlm"
    config = tmp_path / "train.yaml"
    seed_manifest = tmp_path / "seed.json"
    qparams_path = tmp_path / "qparams.json"
    for path in (official, config, seed_manifest, qparams_path):
        path.write_bytes(b"fixture")
    zero, merged, adapter = (tmp_path / name for name in ("seed", "merged", "adapter"))
    for path in (zero, merged, adapter):
        path.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    package, target, mtp = _package_sections()
    records = [{"ordinal": index} for index in range(277)]
    mutable, frozen = records[:205], records[205:]
    qparams = SimpleNamespace(
        path=qparams_path, contract_sha256="q" * 64,
        storage_path=tmp_path / "scales.safetensors", scale_storage_sha256="s" * 64,
        report={"verified": True}, summary=lambda: {"verified": True},
    )
    monkeypatch.setattr(exporter, "_sha256_file", lambda *_a, **_k: exporter.OFFICIAL_LITERTLM_SHA256)
    monkeypatch.setattr(exporter, "_config_report", lambda *_a, **_k: (
        {"verified": True, "path": str(config), "sha256": "c" * 64}, {"model": {}}
    ))
    monkeypatch.setattr(exporter, "verify_configured_mobile_training_seed", lambda *_a, **_k: {
        "verified": True, "output": {"directory": str(zero)},
    })
    monkeypatch.setattr(exporter, "MobileQParams", lambda *_a, **_k: qparams)
    monkeypatch.setattr(exporter, "_extract_inventory", lambda *_a, **_k: (target, records))

    def inspect(path, *, inspect_tflite):
        assert path == official and inspect_tflite is False
        return copy.deepcopy(package)

    monkeypatch.setattr(exporter, "inspect_litertlm", inspect)

    def scope(actual_records, actual_qparams):
        assert actual_records is records and actual_qparams is qparams
        assert len(actual_records) == 277
        return {"verified": True}, mutable, frozen

    monkeypatch.setattr(exporter, "_scope_report", scope)
    monkeypatch.setattr(exporter, "SafetensorCheckpoint", lambda path: SimpleNamespace(path=path))
    monkeypatch.setattr(exporter, "_checkpoint_mapping_report", lambda *_a, **_k: ({"verified": True}, {}))
    monkeypatch.setattr(exporter, "_adapter_mapping_report", lambda *_a, **_k: ({"verified": True}, {}, 1.0))
    selection = {"verified": True, "selected": {"metric": "unique_source_generation_reward_v5_4_avg"}}
    adapter_files = [{"path": "adapter_model.safetensors", "size": 7, "sha256": "a" * 64}]
    monkeypatch.setattr(exporter, "_best_adapter_provenance_report", lambda *_a, **_k: {
        "verified": True, "adapter_files": adapter_files, "golden_selection": selection,
    })
    monkeypatch.setattr(exporter, "_merge_provenance_report", lambda *_a, **_k: {
        "verified": True,
        "metadata": {"adapter_files": adapter_files, "training_run_metadata": {"golden_selection": selection}},
    })
    output = tmp_path / "export"
    plan, context = exporter._build_plan(
        official_litertlm=official, official_artifact_sha256=exporter.OFFICIAL_LITERTLM_SHA256,
        checkpoint=merged, adapter_checkpoint=adapter, training_config=config,
        mobile_training_seed_manifest=seed_manifest, mobile_qparams_contract=qparams_path,
        zero_adapter_checkpoint=zero, output_dir=output, output_litertlm=None, report=None,
    )

    assert plan["plan_passed"] is True
    assert plan["executed"] is False
    assert plan["passed"] is False  # A plan is not an executed export gate.
    assert plan["checks"]["target_and_mtp_sections_present"] is True
    assert plan["target_section"] == target and plan["mtp_section"] == mtp
    assert context["package"] == package
    assert context["records"] is records
    assert not output.exists()


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
    tokenizer_file = adapter / "tokenizer.json"
    tokenizer_file.write_text('{"fixture": true}', encoding="utf-8")
    config = tmp_path / "resolved_training_config.yaml"
    config.write_text(
        "qat: {}\n"
        "golden_eval:\n"
        "  dataset_dir: /bound/golden100\n"
        "  split: all\n"
        "  max_rows: 100\n"
        "  required_rows: 100\n"
        "  require_exact_rows: true\n"
        "  require_unique_rows: true\n"
        "  metric_version: dual\n"
        "  metric_for_best_model: generation_reward_v5_4_avg\n",
        encoding="utf-8",
    )
    keys = sorted(_mutable_key_bits())
    bindings = {
        f"module_{index}": {"weight_key": key} for index, key in enumerate(keys)
    }
    adapter_record = {
        "path": adapter_file.name,
        "size": adapter_file.stat().st_size,
        "sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
    }
    tokenizer_record = {
        "path": tokenizer_file.name,
        "size": tokenizer_file.stat().st_size,
        "sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
    }
    metadata = {
        "training_metadata_version": 4,
        "checkpoint_role": "best_golden",
        "checkpoint_step": 500,
        "training_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "best_golden_eval": {
            "metric": "generation_reward_v5_4_avg",
            "metric_value": 80.0,
            "step": 500,
        },
        "golden_eval": {
            "dataset_dir": "/bound/golden100",
            "split": "all",
            "max_rows": 100,
            "required_rows": 100,
            "require_exact_rows": True,
            "require_unique_rows": True,
            "metric_version": "dual",
            "metric_for_best_model": "generation_reward_v5_4_avg",
        },
        "adapter_checkpoints": [
            {
                "role": "best_golden",
                "files": [adapter_record, tokenizer_record],
            }
        ],
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
    assert report["checks"]["adapter_hashes_self_bound"] is True
    assert report["checks"]["golden_selection_binding_matches"] is True

    metadata["best_golden_eval"]["metric"] = (
        "unique_source_generation_reward_v5_4_avg"
    )
    (adapter / "training_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    mismatched_selection = exporter._best_adapter_provenance_report(
        adapter, config, qparams=_MetadataQParams(), seed_report=seed_report
    )
    assert mismatched_selection["verified"] is False
    assert (
        mismatched_selection["checks"]["golden_selection_binding_matches"]
        is False
    )

    metadata["best_golden_eval"]["metric"] = "generation_reward_v5_4_avg"
    metadata["qat"]["spec"]["scale_mode"] = "dynamic"
    (adapter / "training_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    rejected = exporter._best_adapter_provenance_report(
        adapter, config, qparams=_MetadataQParams(), seed_report=seed_report
    )
    assert rejected["verified"] is False
    assert rejected["checks"]["legacy_metadata_rejected"] is False


def test_merged_selection_legacy_fallback_is_ordinary_golden100_only():
    ordinary = {
        "verified": True,
        "selected": {
            "metric": "generation_reward_v5_4_avg",
            "metric_value": 80.0,
            "step": 500,
        },
    }
    legacy_merge = {
        "metadata": {
            "training_run_metadata": {
                "verified": True,
                "checks": {
                    "best_golden_v5_4_selected": True,
                    "portable_launcher_artifacts_bound": True,
                },
            }
        }
    }
    assert exporter._merged_golden_selection_matches(legacy_merge, ordinary)

    repeated = copy.deepcopy(ordinary)
    repeated["selected"]["metric"] = (
        "unique_source_generation_reward_v5_4_avg"
    )
    assert not exporter._merged_golden_selection_matches(legacy_merge, repeated)

    exact_merge = copy.deepcopy(legacy_merge)
    exact_merge["metadata"]["training_run_metadata"]["golden_selection"] = repeated
    assert exporter._merged_golden_selection_matches(exact_merge, repeated)
