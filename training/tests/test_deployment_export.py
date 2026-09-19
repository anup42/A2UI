from __future__ import annotations

import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest
import yaml
from ir_training.pipeline import deployment_export as de


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_export_plan_uses_isolated_python_and_cpu_conversion(tmp_path):
    plan = de.build_deployment_export_plan(profile="e2b", training_config_path=tmp_path / "future.yaml",
        checkpoint_dir=tmp_path / "future_checkpoint", output_dir=tmp_path / "deployment",
        model_dir=tmp_path / "seed", training_python=sys.executable, exporter_python=sys.executable)
    assert set(plan["variants"]) == {"w32", "w16", "w8", "w4"}
    assert plan["probe_command"][3:6] == ["probe", "--profile", "e2b"]
    assert str(tmp_path / "seed") in plan["probe_command"]
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert plan["official_retained_scale_export"] is False
    assert plan["variants"]["w4"]["kind"] == "mixed_w4_w8"
    assert plan["variants"]["w16"]["experimental"] is True
    assert not (tmp_path / "deployment").exists()


@pytest.mark.parametrize("limit", [0, 4096, True])
def test_export_plan_rejects_short_or_invalid_context(tmp_path, limit):
    with pytest.raises(ValueError, match="cache_length"):
        de.build_deployment_export_plan(profile="270m", training_config_path=tmp_path / "config",
            checkpoint_dir=tmp_path / "checkpoint", output_dir=tmp_path / "out",
            training_python=sys.executable, exporter_python=sys.executable, cache_length=limit)


def test_plan_rejects_path_aliases(tmp_path):
    with pytest.raises(ValueError, match="absolute Python"):
        de.build_deployment_export_plan(profile="270m", training_config_path=tmp_path / "config",
            checkpoint_dir=tmp_path / "checkpoint", output_dir=tmp_path / "out",
            training_python="python", exporter_python=sys.executable)


def test_explicit_none_avoids_upstream_default_int8(tmp_path):
    for variant in ("w32", "w16"):
        kwargs = de.export_kwargs("e2b", variant, tmp_path, tmp_path / "out", 8192)
        assert kwargs["quantization_recipe"] == ("none" if variant == "w32" else str(de.W16_RECIPE_PATH))
        assert kwargs["externalize_embedder"] is True
        assert kwargs["experimental_use_fp16"] is False
        if variant == "w16":
            assert kwargs["experimental_use_mixed_precision"] is False
        assert kwargs["prefill_lengths"] == [128]
        assert kwargs["enable_gpu_dynamic_prefill"] is True
    assert de.deployment_variants("270m")["w4"]["recipe"] == "dynamic_wi4b32_afp32"


def package(weights, *, cast=False, activation="FLOAT32"):
    tensors = [{"type_name": activation, "buffer": 0, "shape": [1, 32]}]
    operators = []
    buffers = [{"index": 0, "logical_size": 0}]
    for dtype in weights:
        index = len(tensors)
        tensors.append({"type_name": dtype, "buffer": len(buffers), "shape": [32, 32]})
        buffers.append({"index": len(buffers), "logical_size": 1024})
        if cast:
            tensors.append({"type_name": activation, "buffer": 0, "shape": [32, 32]})
            operators.append({"builtin_name": "DEQUANTIZE", "inputs": [index], "outputs": [index + 1]})
            index += 1
        operators.append({"builtin_name": "FULLY_CONNECTED", "inputs": [0, index], "outputs": []})
    return {"graphs": [{"available": True, "section_index": 1, "buffers": buffers,
                        "subgraphs": [{"tensors": tensors, "operators": operators}]}]}


@pytest.mark.parametrize("variant,weights", [("w32", ["FLOAT32"]), ("w16", ["FLOAT16"]), ("w8", ["INT8"]), ("w4", ["INT4", "INT8"])])
def test_precision_uses_physical_weights_not_float32_activations(variant, weights):
    result = de.inspect_variant_precision(package(weights, cast=True), "e2b", variant)
    assert result["verified"] is True
    assert set(result["weight_type_counts"]) == set(weights)


@pytest.mark.parametrize("variant,weights", [("w32", ["INT8"]), ("w16", ["FLOAT32"]), ("w8", ["INT4"]), ("w4", ["INT8"])])
def test_wrong_precision_fails_despite_requested_filename(variant, weights):
    with pytest.raises(ValueError, match="actual package"):
        de.inspect_variant_precision(package(weights), "e2b", variant)


def test_270m_w4_rejects_mixed_e2b_recipe():
    with pytest.raises(ValueError, match="actual package"):
        de.inspect_variant_precision(package(["INT4", "INT8"]), "270m", "w4")


def test_unknown_or_missing_graphs_fail_closed():
    with pytest.raises(ValueError, match="unavailable"):
        de.inspect_variant_precision({}, "e2b", "w32")
    value = package(["FLOAT16"])
    value["graphs"][0]["buffers"][1]["logical_size"] = 0
    with pytest.raises(ValueError, match="Cannot prove"):
        de.inspect_variant_precision(value, "e2b", "w16")


def install_fake_exporter(monkeypatch, tmp_path, *, text_type=False, support_text=False, missing_field=None, missing_recipe=None):
    write_json(tmp_path / "config.json", {"model_type": "gemma4_text" if text_type else "gemma4"})
    Config = type("FakeConfig", (), {})
    config = Config()
    config.model_type = "gemma4_text" if text_type else "gemma4"
    hf = types.ModuleType("transformers")
    hf.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: config)
    hf.AutoModelForCausalLM = types.SimpleNamespace(_model_mapping={Config: type("FakeModel", (), {})})
    hf.PreTrainedTokenizerFast = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: TemplateTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", hf)
    fields = [(name, object, dataclasses.field(default=None)) for name in de.export_kwargs("e2b", "w16", tmp_path, tmp_path, 8192) if name != missing_field]
    config_mod = types.ModuleType("litert_torch.generative.export_hf.core.exportable_module_config")
    config_mod.ExportableModuleConfig = dataclasses.make_dataclass("ExportableModuleConfig", fields)
    monkeypatch.setitem(sys.modules, config_mod.__name__, config_mod)
    model_ext = types.ModuleType("litert_torch.generative.export_hf.model_ext")
    model_ext.exportables = types.SimpleNamespace(
        get_prefill_decode_exportables=lambda *args: ("prefill", "decode") if not text_type or support_text else None,
        get_additional_exportables=lambda *args: {"per_layer_embedder": True} if not text_type or support_text else {})
    monkeypatch.setitem(sys.modules, model_ext.__name__, model_ext)
    quantizer = types.ModuleType("ai_edge_quantizer")
    quantizer.recipe = types.SimpleNamespace(**{name: dict for name in ("dynamic_wi8_afp32", "gemma4_mixed48_b32") if name != missing_recipe})
    monkeypatch.setitem(sys.modules, "ai_edge_quantizer", quantizer)
    monkeypatch.setattr(de, "probe_w16_recipe", lambda: {
        "path": str(de.W16_RECIPE_PATH), "sha256": de.file_sha256(de.W16_RECIPE_PATH)})
    monkeypatch.setattr(de, "probe_e2b_w4_recipe", lambda mapping, *, recipe_path=None: {
        "path": str(recipe_path or de.E2B_W4_RECIPE_PATH), "sha256": de.file_sha256(recipe_path or de.E2B_W4_RECIPE_PATH)})
    export = types.ModuleType("litert_torch.generative.export_hf.export")
    export.export = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, export.__name__, export)


def test_probe_is_no_weights_screening_and_checks_dataclass_fp16(monkeypatch, tmp_path):
    install_fake_exporter(monkeypatch, tmp_path)
    result = de.probe_exporter(profile="e2b", model_dir=tmp_path)
    assert result["status"] == "passed"
    assert result["model_loaded"] is False
    assert result["runtime_gpu_tested"] is False
    assert result["recipes"]["gemma4_mixed48_b32"] is True
    assert result["recipes"][de.W16_RECIPE] is True
    assert result["recipe_files"][de.W16_RECIPE]["sha256"] == de.file_sha256(de.W16_RECIPE_PATH)
    assert result["recipe_files"][de.E2B_W4_RECIPE]["sha256"] == de.file_sha256(de.E2B_W4_RECIPE_PATH)
    assert result["template_parity"]["passed"] is True
    assert result["template_parity"]["tested_prompts"] == 3


class TemplateTokenizer:
    chat_template = "test source template"

    def __init__(self, *, mismatch=False):
        self.mismatch = mismatch
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "source" if self.mismatch and "chat_template" not in kwargs else "same formatted prompt"


def test_early_probe_rejects_actual_tokenizer_template_mismatch(monkeypatch, tmp_path):
    install_fake_exporter(monkeypatch, tmp_path)
    sys.modules["transformers"].PreTrainedTokenizerFast.from_pretrained = lambda *args, **kwargs: TemplateTokenizer(mismatch=True)
    with pytest.raises(ValueError, match="differs from the training prompt"):
        de.probe_exporter(profile="e2b", model_dir=tmp_path)


def test_template_parity_checks_unicode_fewshot_and_exact_training_kwargs():
    tokenizer = TemplateTokenizer()
    example = [{"role": "user", "content": "extra prepared input"}]
    result = de.verify_deployment_template(tokenizer, chat_template_kwargs={"enable_thinking": False, "custom_flag": "bound"}, examples=[example])
    assert result["tested_prompts"] == 4
    assert len(tokenizer.calls) == 8
    assert "Ω" in tokenizer.calls[0][0][0]["content"]
    assert [message["role"] for message in tokenizer.calls[4][0]] == ["user", "assistant", "user"]
    assert tokenizer.calls[-1][0] == example
    assert all(kwargs["custom_flag"] == "bound" and kwargs["enable_thinking"] is False for _, kwargs in tokenizer.calls)
    assert all(kwargs["add_generation_prompt"] is True and kwargs["tokenize"] is False for _, kwargs in tokenizer.calls)


def test_prefix_reads_only_three_rows_and_does_not_parse_malformed_tail(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('\ufeff\n{"n": 1}\n\n{"n": 2}\n{"n": 3}\nnot JSON at all\n', encoding="utf-8")
    assert de.strict_prefix_rows(path) == [{"n": 1}, {"n": 2}, {"n": 3}]


@pytest.mark.parametrize("prefix,error", [('not JSON\n', ValueError), ('[]\n', TypeError)])
def test_prefix_rejects_invalid_rows_instead_of_silently_skipping(tmp_path, prefix, error):
    path = tmp_path / "train.jsonl"
    path.write_text(prefix, encoding="utf-8")
    with pytest.raises(error, match="train.jsonl:1"):
        de.strict_prefix_rows(path)


def test_real_tiny_e2b_auto_loader_lora_merge_preserves_export_wrapper(tmp_path):
    """Exercise the actual training/merge loader, no downloads or GPU weights."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    peft = pytest.importorskip("peft", minversion="0.19.0")
    from ir_training.models.hf_loading import load_hf_model

    text_config = transformers.Gemma4TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, global_head_dim=16, max_position_embeddings=64,
        vocab_size_per_layer_input=32, hidden_size_per_layer_input=4,
        layer_types=["full_attention", "full_attention"], num_kv_shared_layers=1,
    )
    config = transformers.Gemma4Config(text_config=text_config, vision_config=None, audio_config=None)
    model = transformers.AutoModelForCausalLM.from_config(config)
    assert isinstance(model, transformers.Gemma4ForConditionalGeneration)
    assert model.config.model_type == "gemma4"
    base, adapter, output = tmp_path / "base", tmp_path / "adapter", tmp_path / "merged"
    model.save_pretrained(base, safe_serialization=True)
    loader_config = {"model_loader": "auto_causal_lm", "dtype": "float32", "device_map": None,
                     "require_exact_checkpoint_keys": True}
    loaded = load_hf_model(str(base), loader_config)
    original_keys = set(loaded.state_dict())
    original_weight = loaded.model.language_model.layers[0].self_attn.q_proj.weight.detach().clone()
    adapted = peft.get_peft_model(loaded, peft.LoraConfig(r=2, lora_alpha=4,
        target_modules=r"model\.language_model\.layers\.\d+\.self_attn\.q_proj", task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.125)
    adapted.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
    restored = peft.PeftModel.from_pretrained(load_hf_model(str(base), loader_config), adapter)
    merged = restored.merge_and_unload()
    assert isinstance(merged, transformers.Gemma4ForConditionalGeneration)
    assert merged.config.model_type == "gemma4"
    assert merged.config.text_config.model_type == "gemma4_text"
    assert set(merged.state_dict()) == original_keys
    assert not torch.equal(merged.model.language_model.layers[0].self_attn.q_proj.weight, original_weight)
    merged.save_pretrained(output, safe_serialization=True)
    serialized = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert serialized["model_type"] == "gemma4"
    assert serialized["text_config"]["model_type"] == "gemma4_text"
    reloaded = load_hf_model(str(output), loader_config)
    assert isinstance(reloaded, transformers.Gemma4ForConditionalGeneration)
    assert set(reloaded.state_dict()) == original_keys
    for key, expected in merged.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], expected), key


@pytest.mark.parametrize("options,error", [({"text_type": True}, "per-layer embedding"),
    ({"missing_field": "experimental_use_fp16"}, "requested options"),
    ({"missing_recipe": "gemma4_mixed48_b32"}, "required recipe")])
def test_probe_rejects_missing_export_capabilities(monkeypatch, tmp_path, options, error):
    install_fake_exporter(monkeypatch, tmp_path, **options)
    with pytest.raises((ValueError, TypeError), match=error):
        de.probe_exporter(profile="e2b", model_dir=tmp_path)


def test_probe_supports_future_explicit_text_type_exporter(monkeypatch, tmp_path):
    install_fake_exporter(monkeypatch, tmp_path, text_type=True, support_text=True)
    assert de.probe_exporter(profile="e2b", model_dir=tmp_path)["status"] == "passed"


def test_selected_checkpoint_rejects_changed_config_before_importing_frameworks(tmp_path):
    config = tmp_path / "training_config.yaml"
    config.write_text(yaml.safe_dump({"qat": {"enabled": False}}), encoding="utf-8")
    write_json(tmp_path / "checkpoint/training_metadata.json", {"training_config_sha256": "wrong"})
    with pytest.raises(ValueError, match="bound training config"):
        de.verify_checkpoint_source(config, tmp_path / "checkpoint", "e2b")


def test_bound_file_cannot_escape_directory(tmp_path):
    (tmp_path / "outside").write_text("data", encoding="utf-8")
    inner = tmp_path / "inner"
    inner.mkdir()
    with pytest.raises(ValueError, match="escaping"):
        de._local_file(inner, "../outside")
    with pytest.raises(ValueError, match="escaping"):
        de._local_file(inner, "../outside", allow_bound_symlink=True)


def test_hash_bound_original_snapshot_symlink_is_allowed(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"pinned source weights")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    try:
        (snapshot / "model.safetensors").symlink_to(blob)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this host")
    assert de._local_file(snapshot, "model.safetensors", allow_bound_symlink=True) == blob
    assert de.file_sha256(de._local_file(snapshot, "model.safetensors", allow_bound_symlink=True)) == de.file_sha256(blob)
    with pytest.raises(ValueError, match="escaping"):
        de._local_file(snapshot, "model.safetensors")


def test_output_validation_rechecks_precision_and_hashes(tmp_path):
    folder = tmp_path / "w16"
    artifact = folder / "model.litertlm"
    folder.mkdir()
    artifact.write_bytes(b"test-only-contract-artifact")
    inspection = package(["FLOAT16"], cast=True)
    write_json(folder / "package_inspection.json", inspection)
    write_json(tmp_path / "merged/deployment_source.json", {"source_checkpoint": "pinned"})
    report = {"profile": "e2b", "variant": "w16", "artifact": str(artifact.resolve()),
              "sha256": de.file_sha256(artifact), "inspection_sha256": de.file_sha256(folder / "package_inspection.json"),
              "source_manifest_sha256": de.file_sha256(tmp_path / "merged/deployment_source.json"),
              "actual_precision": de.inspect_variant_precision(inspection, "e2b", "w16")}
    write_json(folder / "export_manifest.json", report)
    plan = {"profile": "e2b", "merged_model_dir": str(tmp_path / "merged"),
            "variants": {"w16": {"output_dir": str(folder), "artifact": str(artifact)}}}
    assert de.validate_deployment_export_output(plan, "w16")["actual_precision"]["verified"]
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="path/hash"):
        de.validate_deployment_export_output(plan, "w16")


def source_fixture(tmp_path):
    base, checkpoint, fit = tmp_path / "base", tmp_path / "checkpoint", tmp_path / "fit"
    write_json(base / "config.json", {"model_type": "gemma3_text"})
    (base / "model.safetensors").write_bytes(b"base fixture")
    write_json(checkpoint / "config.json", {"model_type": "gemma3_text"})
    (checkpoint / "model.safetensors").write_bytes(b"trained fixture")
    write_json(checkpoint / "tokenizer.json", {"test": "verified tokenizer"})
    fit.mkdir()
    config = {"run": {"dataset_dir": str(tmp_path / "data")},
              "model": {"family": "gemma", "model_id": "google/gemma-3-270m-it", "model_source": str(base)},
              "training": {"method": "full_sft"}, "qat": {"enabled": False}}
    config_path = fit / "training_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    write_json(fit / "preparation_report.json", {"training_config_sha256": de.file_sha256(config_path),
        "model_files": {path.name: de.file_sha256(path) for path in base.iterdir()}})
    files = [{"path": path.name, "sha256": de.file_sha256(path), "size_bytes": path.stat().st_size} for path in checkpoint.iterdir()]
    write_json(checkpoint / "training_metadata.json", {"training_config_sha256": de.file_sha256(config_path),
        "checkpoint_step": 1000, "checkpoint_kind": "full_model", "checkpoint_adapter_files": files})
    return config_path, checkpoint, base


def test_source_verifies_checkpoint_and_base_hashes(tmp_path):
    config_path, checkpoint, base = source_fixture(tmp_path)
    result = de.verify_checkpoint_source(config_path, checkpoint, "270m")
    assert result["checkpoint_kind"] == "full_model"
    assert result["metadata"]["checkpoint_step"] == 1000
    (base / "model.safetensors").write_bytes(b"changed base")
    with pytest.raises(ValueError, match="Original dense model changed"):
        de.verify_checkpoint_source(config_path, checkpoint, "270m")


def test_source_rejects_unbound_shard(tmp_path):
    config_path, checkpoint, _ = source_fixture(tmp_path)
    (checkpoint / "model-00002.safetensors").write_bytes(b"unbound")
    with pytest.raises(ValueError, match="inventory differs"):
        de.verify_checkpoint_source(config_path, checkpoint, "270m")


def test_materialize_full_checkpoint_preserves_training_assets_and_step(monkeypatch, tmp_path):
    config_path, checkpoint, base = source_fixture(tmp_path)
    import ir_training.eval.prepared_contract as contract
    from ir_training.models import registry

    class Tokenizer:
        def save_pretrained(self, target):
            write_json(Path(target) / "tokenizer.json", {"test": "verified tokenizer"})

    sources = []

    def create(config):
        sources.append(config["tokenizer_source"])
        return types.SimpleNamespace(load_tokenizer=Tokenizer)

    monkeypatch.setattr(registry, "create_adapter", create)
    monkeypatch.setattr(contract, "checked_preparation_manifest", lambda path: {"tokenizer": {"bound": True}})
    monkeypatch.setattr(contract, "verify_loaded_evaluation_tokenizer", lambda *args: None)
    output = tmp_path / "merged"
    result = de.prepare_deployment_checkpoint(profile="270m", training_config=config_path, checkpoint=checkpoint, output_dir=output)
    assert result["checkpoint_step"] == 1000
    assert (output / "model.safetensors").read_bytes() == b"trained fixture"
    assert (base / "model.safetensors").read_bytes() == b"base fixture"
    assert sources == [str(checkpoint), str(output)]
    assert de.file_sha256(output / "training_config.yaml") == de.file_sha256(config_path)
    assert "model.safetensors" in result["merged_files"]
    assert result["official_retained_scale_export"] is False
    with pytest.raises(ValueError, match="fresh"):
        de.prepare_deployment_checkpoint(profile="270m", training_config=config_path, checkpoint=checkpoint, output_dir=output)
