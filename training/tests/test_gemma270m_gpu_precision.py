import sys
from types import SimpleNamespace

import pytest
from ir_training.export.gemma270m_gpu_precision import (
    export_with_fp32_text_activations,
    probe_fp32_text_builder,
    verify_fp32_text_activations,
)
from ir_training.export.litertlm_inspector import inspect_litertlm
from ir_training.export.litertlm_mtp import find_model_section
from ir_training.pipeline import deployment_export


def test_only_270m_qat_probe_requires_pinned_packager_api(tmp_path, monkeypatch):
    builder = pytest.importorskip("litert_lm_builder")
    packager_name = "litert_torch.generative.export_hf.core.litert_lm_builder"
    monkeypatch.setitem(sys.modules, packager_name, SimpleNamespace(litertlm_builder=builder))
    monkeypatch.setattr(deployment_export, "_probe_exporter", lambda **kw: {"status": "passed"})
    args = {"profile": "270m", "model_dir": tmp_path, "selected_variants": ("w8",)}
    evidence = deployment_export.probe_exporter(**args, require_270m_qat=True)["text_gpu_activation"]
    assert evidence["model_type"] == "tf_lite_prefill_decode"
    assert evidence["prefer_activation_type"] == "fp32"
    assert evidence["litert_lm_builder"]
    monkeypatch.setitem(sys.modules, packager_name, SimpleNamespace(litertlm_builder=None))
    with pytest.raises(ValueError, match="builder binding"):
        probe_fp32_text_builder()
    assert deployment_export.probe_exporter(**args) == {"status": "passed"}
    monkeypatch.setitem(sys.modules, packager_name, SimpleNamespace(litertlm_builder=builder))
    monkeypatch.setattr(builder.LitertLmFileBuilder, "add_tflite_model", lambda self, path, kind: None)
    with pytest.raises(TypeError, match="prefer_activation_type"):
        probe_fp32_text_builder()


def test_w8_export_sets_header_for_text_only_and_preserves_payloads(tmp_path):
    builder = pytest.importorskip("litert_lm_builder")
    original = builder.LitertLmFileBuilder
    graph = tmp_path / "model.tflite"
    graph.write_bytes(b"same test graph")
    settings = {"quantization_recipe": "dynamic_wi8_afp32", "experimental_use_fp16": False}

    def package(**kwargs):
        assert kwargs == settings
        result = builder.LitertLmFileBuilder()
        result.add_tflite_model(str(graph), builder.TfLiteModelType.PREFILL_DECODE)
        result.add_tflite_model(str(graph), builder.TfLiteModelType.EMBEDDER)
        with (tmp_path / "model.litertlm").open("wb") as stream:
            result.build(stream)

    def inspect():
        return inspect_litertlm(tmp_path / "model.litertlm", include_hashes=True, inspect_tflite=False)

    package(**settings)
    baseline = inspect()
    with pytest.raises(ValueError, match="FP32 text activation"):
        verify_fp32_text_activations(baseline)
    export_with_fp32_text_activations(package, settings)
    exported = inspect()
    verify_fp32_text_activations(exported)
    assert [s["sha256"] for s in exported["sections"]] == [s["sha256"] for s in baseline["sections"]]
    assert (find_model_section(exported, "tf_lite_embedder")["items"]
            == find_model_section(baseline, "tf_lite_embedder")["items"])
    assert builder.LitertLmFileBuilder is original

    def conflicting(**kwargs):
        builder.LitertLmFileBuilder().add_tflite_model(
            str(graph), builder.TfLiteModelType.PREFILL_DECODE, prefer_activation_type="fp16")

    with pytest.raises(ValueError, match="conflicting"):
        export_with_fp32_text_activations(conflicting, settings)
    assert builder.LitertLmFileBuilder is original
