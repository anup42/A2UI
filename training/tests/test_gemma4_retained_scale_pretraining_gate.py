from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_gemma4_retained_scale_litertlm as exporter
import verify_gemma4_retained_scale_pretraining as gate


class _Quantization:
    @staticmethod
    def ScaleAsNumpy():
        return np.asarray([0.25], dtype=np.float32)

    @staticmethod
    def ZeroPointAsNumpy():
        return np.asarray([0], dtype=np.int64)

    @staticmethod
    def MinAsNumpy():
        return np.asarray([], dtype=np.float32)

    @staticmethod
    def MaxAsNumpy():
        return np.asarray([], dtype=np.float32)

    @staticmethod
    def QuantizedDimension():
        return 0


class _Tensor:
    def __init__(self, name, shape, dtype=9, quantized=False):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.quantized = quantized

    def Name(self):
        return self.name.encode()

    def ShapeLength(self):
        return len(self.shape)

    def Shape(self, index):
        return self.shape[index]

    def ShapeSignatureLength(self):
        return len(self.shape)

    def ShapeSignature(self, index):
        return self.shape[index]

    def Type(self):
        return self.dtype

    def Quantization(self):
        return _Quantization() if self.quantized else None


class _Subgraph:
    def __init__(self, name, tensors, inputs, outputs):
        self.name = name
        self.tensors = tensors
        self.inputs = inputs
        self.outputs = outputs

    def Name(self):
        return self.name.encode()

    def Tensors(self, index):
        return self.tensors[index]

    def InputsLength(self):
        return len(self.inputs)

    def Inputs(self, index):
        return self.inputs[index]

    def OutputsLength(self):
        return len(self.outputs)

    def Outputs(self, index):
        return self.outputs[index]


class _TensorMap:
    def __init__(self, name, tensor_index):
        self.name = name
        self.tensor_index = tensor_index

    def Name(self):
        return self.name.encode()

    def TensorIndex(self):
        return self.tensor_index


class _Signature:
    def __init__(self, key, subgraph_index, inputs, outputs):
        self.key = key
        self.subgraph_index = subgraph_index
        self.inputs = [_TensorMap(*item) for item in inputs]
        self.outputs = [_TensorMap(*item) for item in outputs]

    def SignatureKey(self):
        return self.key.encode()

    def SubgraphIndex(self):
        return self.subgraph_index

    def InputsLength(self):
        return len(self.inputs)

    def Inputs(self, index):
        return self.inputs[index]

    def OutputsLength(self):
        return len(self.outputs)

    def Outputs(self, index):
        return self.outputs[index]


class _Model:
    def __init__(self, subgraphs, signatures):
        self.subgraphs = subgraphs
        self.signatures = signatures

    def SubgraphsLength(self):
        return len(self.subgraphs)

    def Subgraphs(self, index):
        return self.subgraphs[index]

    def SignatureDefsLength(self):
        return len(self.signatures)

    def SignatureDefs(self, index):
        return self.signatures[index]


def test_static_kv_cache_contract_extracts_named_prefill_decode_boundaries(monkeypatch):
    prefill = _Subgraph(
        "prefill_graph",
        [
            _Tensor("tokens", [1, -1], dtype=2),
            _Tensor("present_key_cache", [1, 4, -1, 16], quantized=True),
            _Tensor("present_value_cache", [1, 4, -1, 16], quantized=True),
        ],
        [0],
        [1, 2],
    )
    decode = _Subgraph(
        "decode_graph",
        [
            _Tensor("token", [1, 1], dtype=2),
            _Tensor("past_key_cache", [1, 4, -1, 16], quantized=True),
            _Tensor("past_value_cache", [1, 4, -1, 16], quantized=True),
            _Tensor("present_key_cache", [1, 4, -1, 16], quantized=True),
            _Tensor("present_value_cache", [1, 4, -1, 16], quantized=True),
        ],
        [0, 1, 2],
        [3, 4],
    )
    model = _Model(
        [prefill, decode],
        [
            _Signature(
                "prefill",
                0,
                [("tokens", 0)],
                [("kv_cache_key", 1), ("kv_cache_value", 2)],
            ),
            _Signature(
                "decode",
                1,
                [("token", 0), ("kv_cache_key", 1), ("kv_cache_value", 2)],
                [("kv_cache_key_out", 3), ("kv_cache_value_out", 4)],
            ),
        ],
    )
    monkeypatch.setattr(gate, "_schema_model", lambda _section: model)

    report = gate._static_kv_cache_contract(b"synthetic")

    assert report["established"] is True
    assert report["prefill_decode_stages_identified"] is True
    assert report["prefill_decode_mapping_established"] is False
    assert report["named_cache_input_count"] == 2
    assert report["named_cache_output_count"] == 4
    assert {item["cache_role"] for item in report["records"]} == {"key", "value"}
    assert all(item["dtype_name"] == "INT8" for item in report["records"])
    assert all(item["qparams_present"] for item in report["records"])
    assert all(len(item["qparams_sha256"]) == 64 for item in report["records"])
    assert report["runtime_simulation_performed"] is False
    assert report["runtime_cache_correctness_established"] is False


def test_static_kv_cache_contract_reports_not_established_when_names_absent(monkeypatch):
    model = _Model(
        [
            _Subgraph(
                "serve",
                [
                    _Tensor("tokens", [1, 8], dtype=2),
                    _Tensor("cache_position", [1], dtype=2),
                ],
                [0, 1],
                [0],
            )
        ],
        [
            _Signature(
                "serve",
                0,
                [("tokens", 0), ("cache_position", 1)],
                [("logits", 0)],
            )
        ],
    )
    monkeypatch.setattr(gate, "_schema_model", lambda _section: model)

    report = gate._static_kv_cache_contract(b"synthetic")

    assert report["established"] is False
    assert report["status"] == "not_established_unidentified"
    assert report["records"] == []
    assert report["runtime_simulation_performed"] is False


def _patch_synthetic_run(monkeypatch, tmp_path, *, one_buffer_differs=False):
    official = tmp_path / "official.litertlm"
    official.write_bytes(b"P" * 100)
    config = tmp_path / "train.yaml"
    config.write_text("model: {}\n", encoding="utf-8")
    manifest = tmp_path / "seed.json"
    manifest.write_text("{}", encoding="utf-8")
    qparams_path = tmp_path / "qparams.json"
    qparams_path.write_text("{}", encoding="utf-8")
    zero = tmp_path / "seed"
    zero.mkdir()

    monkeypatch.setattr(
        exporter,
        "_sha256_file",
        lambda path, **_kwargs: exporter.OFFICIAL_LITERTLM_SHA256,
    )
    monkeypatch.setattr(
        exporter,
        "_config_report",
        lambda *_args, **_kwargs: (
            {"verified": True, "checks": {}, "path": str(config), "sha256": "c" * 64},
            {"model": {}},
        ),
    )
    monkeypatch.setattr(
        gate,
        "verify_configured_mobile_training_seed",
        lambda *_args, **_kwargs: {
            "verified": True,
            "checks": {},
            "manifest_sha256": "m" * 64,
            "output": {"directory": str(zero)},
        },
    )

    class QParams:
        def __init__(self, *_args, **_kwargs):
            self.report = {"verified": True}

        @staticmethod
        def summary():
            return {"verified": True}

    class Reader:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(gate, "MobileQParams", QParams)
    monkeypatch.setattr(gate, "SafetensorCheckpoint", Reader)
    records = [{"ordinal": index} for index in range(exporter.EXPECTED_TARGET_COUNT)]
    mutable = [{"hf_weight_key": "w", "official_buffer": 0}]
    target = {
        "index": 0, "data_type_name": "TFLiteModel",
        "items": [{"key": "model_type", "value": exporter.TARGET_MODEL_TYPE}],
        "begin_offset": 10, "end_offset": 18, "size": 8,
    }
    mtp = {
        "index": 1, "data_type_name": "TFLiteModel",
        "items": [{"key": "model_type", "value": exporter.MTP_MODEL_TYPE}],
        "begin_offset": 50, "end_offset": 60, "size": 10,
    }
    monkeypatch.setattr(
        gate,
        "_extract_inventory",
        lambda *_args, **_kwargs: (
            target,  # Real helper contract: ONE section, not a package report.
            records,
        ),
    )
    monkeypatch.setattr(
        exporter,
        "_scope_report",
        lambda *_args, **_kwargs: ({"verified": True}, mutable, []),
    )
    def inspect_package(path, *, inspect_tflite):
        assert path == official
        assert inspect_tflite is False
        return {"sections": [mtp, target]}

    # Do not mock section selection: exercise the canonical lookup against
    # the full report with the target deliberately not at position zero.
    monkeypatch.setattr(exporter, "inspect_litertlm", inspect_package)
    monkeypatch.setattr(
        exporter,
        "_checkpoint_mapping_report",
        lambda *_args, **_kwargs: ({"verified": True}, {"w": "w"}),
    )
    monkeypatch.setattr(gate, "_read_section", lambda *_args: b"official")
    monkeypatch.setattr(gate, "_enriched_records", lambda value: value)
    weight_report = {"sha256": "w" * 64, "mutable_retained_scales_exact": True}
    a8_report = {"sha256": "a" * 64, "retained_a8_contract_exact": True}
    all_report = {"sha256": "q" * 64}
    monkeypatch.setattr(exporter, "_weight_qparams_report", lambda *_args: weight_report)
    monkeypatch.setattr(exporter, "_activation_a8_report", lambda *_args: a8_report)
    monkeypatch.setattr(exporter, "_all_quantization_digest", lambda *_args: all_report)
    telemetry = [
        {
            "official_buffer": index,
            "differs_from_official_codes": one_buffer_differs and index == 0,
        }
        for index in range(exporter.EXPECTED_MUTABLE_COUNT)
    ]
    monkeypatch.setattr(
        exporter,
        "_quantize_and_patch",
        lambda *_args, **_kwargs: (
            bytearray(b"official"),
            {
                "verified": True,
                "checks": {"processed_205": True},
                "telemetry": telemetry,
            },
        ),
    )
    monkeypatch.setattr(
        exporter,
        "_buffer_diff_report",
        lambda *_args: {
            "frozen_72_byte_exact": True,
            "changed_buffer_count": 0,
        },
    )
    monkeypatch.setattr(
        exporter,
        "_graph_identity_report",
        lambda *_args: {"verified": True},
    )
    monkeypatch.setattr(
        gate,
        "_static_kv_cache_contract",
        lambda *_args: {
            "established": False,
            "status": "not_established_unidentified",
            "runtime_simulation_performed": False,
        },
    )
    monkeypatch.setattr(
        exporter,
        "_file_range_sha256",
        lambda *_args: "k" * 64,
    )
    return official, config, manifest, qparams_path, zero


def test_run_proves_actual_205_buffer_noop_and_writes_only_report(tmp_path, monkeypatch):
    official, config, manifest, qparams_path, zero = _patch_synthetic_run(
        monkeypatch, tmp_path
    )
    report_path = tmp_path / "reports" / "pretraining.json"

    result = gate.run(
        official_litertlm=official,
        official_artifact_sha256=exporter.OFFICIAL_LITERTLM_SHA256,
        training_config=config,
        mobile_training_seed_manifest=manifest,
        mobile_qparams_contract=qparams_path,
        zero_adapter_checkpoint=zero,
        report=report_path,
        working_set_bytes=1024,
    )

    assert result["passed"] is True
    assert result["gate_status"] == "PASSED"
    assert result["checks"]["unique_materialized_code_buffers_205"] is True
    assert result["checks"]["every_materialized_code_buffer_matches_official"] is True
    assert result["checks"]["frozen_72_byte_exact"] is True
    assert result["checks"]["graph_layout_execution_identity"] is True
    assert result["package_copy_materialized"] is False
    assert result["training_started"] is False
    assert result["model_execution_performed"] is False
    assert official.read_bytes() == b"P" * 100
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
    assert list(tmp_path.rglob("*.litertlm")) == [official]


def test_run_fails_closed_when_any_materialized_seed_code_buffer_differs(
    tmp_path, monkeypatch
):
    official, config, manifest, qparams_path, zero = _patch_synthetic_run(
        monkeypatch, tmp_path, one_buffer_differs=True
    )

    with pytest.raises(
        exporter.RetainedScaleExportError,
        match="every_materialized_code_buffer_matches_official",
    ):
        gate.run(
            official_litertlm=official,
            official_artifact_sha256=exporter.OFFICIAL_LITERTLM_SHA256,
            training_config=config,
            mobile_training_seed_manifest=manifest,
            mobile_qparams_contract=qparams_path,
            zero_adapter_checkpoint=zero,
            working_set_bytes=1024,
        )


def test_cli_returns_success_only_for_passed_gate(monkeypatch, capsys):
    monkeypatch.setattr(gate, "run", lambda **_kwargs: {"passed": True})

    code = gate.main(
        [
            "--official-litertlm",
            "official.litertlm",
            "--official-artifact-sha256",
            exporter.OFFICIAL_LITERTLM_SHA256,
            "--training-config",
            "train.yaml",
            "--mobile-training-seed-manifest",
            "seed.json",
            "--mobile-qparams-contract",
            "qparams.json",
            "--zero-adapter-checkpoint",
            "seed",
        ]
    )

    assert code == 0
    assert "PASSED: materialized retained-scale seed" in capsys.readouterr().out
