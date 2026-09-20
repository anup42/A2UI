from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.export import full_parameter_export as export


def test_package_gate_binds_all_three_exact_section_payloads(tmp_path):
    builder = pytest.importorskip("litert_lm_builder.litertlm_builder")

    paths = {role: tmp_path / f"{role}.tflite" for role in export.SECTION_TYPES}
    bundle = builder.LitertLmFileBuilder()
    for role, kind in (
        ("target", builder.TfLiteModelType.PREFILL_DECODE),
        ("token_embedder", builder.TfLiteModelType.EMBEDDER),
        ("per_layer_embedder", builder.TfLiteModelType.PER_LAYER_EMBEDDER),
    ):
        paths[role].write_bytes(f"audited-{role}".encode())
        bundle.add_tflite_model(str(paths[role]), kind)
    artifact = tmp_path / "model.litertlm"
    with artifact.open("wb") as stream:
        bundle.build(stream)
    assert export.verify_packaged_sections(artifact, paths)["verified"] is True
    paths["per_layer_embedder"].write_bytes(b"wrong embedding")
    with pytest.raises(ValueError, match="Packaged per_layer_embedder"):
        export.verify_packaged_sections(artifact, paths)


def _proof():
    keys = ["model.embed_tokens.weight", "model.embed_tokens_per_layer.weight"] + [
        f"p{i}" for i in range(504)
    ] + [
        f"model.layers.{i}.layer_scalar" for i in range(35)
    ]
    inventory = {
        key: {
            "shape": [1],
            "value_sha256": hashlib.sha256(key.encode()).hexdigest(),
        }
        for key in keys
    }
    full = {"verified": True, "state_tensor_count": 541}
    return {
        "verified": True,
        "state_tensor_count": 541,
        "named_parameter_count": 506,
        "persistent_buffer_count": 35,
        "policy": "full_checkpoint_physical_w248_v1",
        "checkpoint_inventory": inventory,
        "evidence": {
            "loaded": {
                **full,
                "value_hashes": {
                    key: value["value_sha256"] for key, value in inventory.items()
                },
            },
            "physical": {
                **full,
                "state_tensors": dict.fromkeys(keys),
                "float_audit": {**full, "mappings": dict.fromkeys(keys)},
            },
            "package": {
                "verified": True,
                "sections": dict.fromkeys(export.SECTION_TYPES),
                "artifact_sha256": "artifact",
            },
        },
    }


@pytest.mark.parametrize(
    "mutation", ["embedding", "loaded", "physical", "float", "package"]
)
def test_downstream_proof_rejects_partial_evidence(mutation):
    report = _proof()
    export.validate_serialization_report(report, "artifact")
    if mutation == "embedding":
        report["checkpoint_inventory"].pop("model.embed_tokens_per_layer.weight")
    elif mutation == "loaded":
        report["evidence"]["loaded"]["value_hashes"]["p0"] = "wrong"
    elif mutation == "physical":
        report["evidence"]["physical"]["state_tensors"].pop("p0")
    elif mutation == "float":
        report["evidence"]["physical"]["float_audit"]["verified"] = False
    else:
        report["evidence"]["package"]["artifact_sha256"] = "wrong"
    with pytest.raises(ValueError, match="complete source-to-package"):
        export.validate_serialization_report(report, "artifact")


@pytest.mark.parametrize("inventory", [["not-a-mapping"], {"bad": None}])
def test_downstream_proof_rejects_malformed_inventory_as_value_error(inventory):
    report = _proof()
    report["checkpoint_inventory"] = inventory
    with pytest.raises(ValueError, match="complete source-to-package"):
        export.validate_serialization_report(report, "artifact")


def _fake_modules(monkeypatch):
    library = types.SimpleNamespace(
        load_model=lambda *_a, **_k: types.SimpleNamespace(model=object()),
        maybe_quantize_model=lambda *_a, **_k: None,
    )
    builder = types.SimpleNamespace(package_model=lambda *_a, **_k: None)
    actual_import = export.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.endswith("core.export_lib"):
            return library
        if name.endswith("core.litert_lm_builder"):
            return builder
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(export.importlib, "import_module", fake_import)
    return library, builder


def test_scoped_audit_restores_every_hook_on_failure(tmp_path, monkeypatch):
    library, builder = _fake_modules(monkeypatch)
    originals = library.load_model, library.maybe_quantize_model, builder.package_model
    recipe = tmp_path / "recipe.json"
    recipe.write_text("[]")
    with (
        pytest.raises(ValueError, match="skipped a mandatory"),
        export._audited_converter(tmp_path, tmp_path, recipe, {}),
    ):
        assert library.load_model is not originals[0]
    assert (
        library.load_model,
        library.maybe_quantize_model,
        builder.package_model,
    ) == originals


def test_wrong_loaded_parameter_stops_before_quantization(tmp_path, monkeypatch):
    from ir_training.export import gemma4_text_compat

    library, _ = _fake_modules(monkeypatch)
    recipe = tmp_path / "recipe.json"
    recipe.write_text("[]")
    monkeypatch.setattr(gemma4_text_compat, "_require_model", lambda _model: None)

    def reject(*_a):
        raise ValueError("loaded weights differ")

    monkeypatch.setattr(export.serialization, "verify_loaded_model", reject)
    with (
        pytest.raises(ValueError, match="loaded weights differ"),
        export._audited_converter(tmp_path, tmp_path, recipe, {}),
    ):
        library.load_model(tmp_path, types.SimpleNamespace())


def _successful_fake_task_sequence(tmp_path, monkeypatch):
    from ir_training.export import gemma4_text_compat

    library, builder = _fake_modules(monkeypatch)
    recipe = tmp_path / "recipe.json"
    recipe.write_text("[]")
    model = object()
    library.load_model = lambda *_a, **_k: types.SimpleNamespace(model=model)

    def quantize(model_path, _recipe):
        source = Path(model_path)
        result = source.with_name(source.stem + "_quantized.tflite")
        result.write_bytes(b"quantized-" + source.read_bytes())
        return str(result)

    library.maybe_quantize_model = quantize

    def package(_source, _config, _exported):
        artifact = tmp_path / "model.litertlm"
        artifact.write_bytes(b"package")
        return types.SimpleNamespace(litert_lm_model_path=str(artifact))

    builder.package_model = package
    # Reinstall the fake modules so _audited_converter captures these originals.
    monkeypatch.setattr(gemma4_text_compat, "_require_model", lambda _model: None)
    monkeypatch.setattr(
        export.serialization,
        "verify_loaded_model",
        lambda actual, inventory: {
            "verified": actual is model,
            "state_tensor_count": len(inventory),
        },
    )
    monkeypatch.setattr(
        export.serialization,
        "audit_quantized_sections",
        lambda floats, quantized, inventory: {
            "verified": set(floats) == set(export.SECTION_TYPES)
            and set(quantized) == set(export.SECTION_TYPES),
            "state_tensor_count": len(inventory),
        },
    )
    monkeypatch.setattr(
        export,
        "verify_packaged_sections",
        lambda artifact, paths: {
            "verified": artifact.read_bytes() == b"package",
            "sections": dict.fromkeys(paths),
            "artifact_sha256": export._sha(artifact),
        },
    )
    originals = library.load_model, library.maybe_quantize_model, builder.package_model
    return library, builder, originals, recipe


def test_fake_pinned_task_sequence_reaches_every_gate_and_restores(
    tmp_path, monkeypatch
):
    library, builder, originals, recipe = _successful_fake_task_sequence(
        tmp_path, monkeypatch
    )
    with export._audited_converter(tmp_path, tmp_path, recipe, {}) as state:
        artifacts = library.load_model(
            tmp_path,
            types.SimpleNamespace(use_random_weights=False),
            trust_remote_code=False,
            auto_model_override=None,
        )
        quantized = {}
        for filename, role in export.FLOAT_FILES.items():
            source = tmp_path / filename
            source.write_bytes(f"float-{role}".encode())
            quantized[role] = library.maybe_quantize_model(source, str(recipe))
        packaged = types.SimpleNamespace(
            prefill_decode_model_path=quantized["target"],
            embedder_model_path=quantized["token_embedder"],
            additional_model_paths={
                "per_layer_embedder": quantized["per_layer_embedder"]
            },
        )
        result = builder.package_model(artifacts, object(), packaged)
        assert Path(result.litert_lm_model_path).is_file()
    assert all(state[name]["verified"] for name in ("loaded", "physical", "package"))
    assert (
        library.load_model,
        library.maybe_quantize_model,
        builder.package_model,
    ) == originals


def test_duplicate_load_is_rejected_and_hooks_restore(tmp_path, monkeypatch):
    library, builder, originals, recipe = _successful_fake_task_sequence(
        tmp_path, monkeypatch
    )
    options = types.SimpleNamespace(use_random_weights=False)
    with (
        pytest.raises(ValueError, match="exactly once"),
        export._audited_converter(tmp_path, tmp_path, recipe, {}),
    ):
        library.load_model(tmp_path, options)
        library.load_model(tmp_path, options)
    assert (
        library.load_model,
        library.maybe_quantize_model,
        builder.package_model,
    ) == originals


def test_wrong_recipe_is_rejected_and_hooks_restore(tmp_path, monkeypatch):
    library, builder, originals, recipe = _successful_fake_task_sequence(
        tmp_path, monkeypatch
    )
    wrong = tmp_path / "wrong.json"
    wrong.write_text("[]")
    source = tmp_path / "model.tflite"
    source.write_bytes(b"float-target")
    with (
        pytest.raises(ValueError, match="quantizer input or section"),
        export._audited_converter(tmp_path, tmp_path, recipe, {}),
    ):
        library.load_model(tmp_path, types.SimpleNamespace(use_random_weights=False))
        library.maybe_quantize_model(source, str(wrong))
    assert (
        library.load_model,
        library.maybe_quantize_model,
        builder.package_model,
    ) == originals


def test_full_probe_opt_in_preserves_ordinary_route(monkeypatch, tmp_path):
    import json
    from contextlib import contextmanager

    from ir_training.export import gemma4_text_compat
    from ir_training.pipeline import deployment_export

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4_text"}))
    calls = []

    @contextmanager
    def context():
        calls.append("entered")
        yield {"active": True}
        calls.append("exited")

    monkeypatch.setattr(gemma4_text_compat, "gemma4_text_export_context", context)
    monkeypatch.setattr(export, "check_serialization_dependencies", dict)
    monkeypatch.setattr(
        deployment_export, "_probe_exporter", lambda **_kw: {"status": "passed"}
    )
    normal = deployment_export.probe_exporter(
        profile="e2b", model_dir=tmp_path, selected_variants=("w248",)
    )
    assert normal == {"status": "passed"} and not calls
    result = deployment_export.probe_exporter(
        profile="e2b",
        model_dir=tmp_path,
        selected_variants=("w248",),
        full_parameter_export=True,
    )
    assert result["all_parameter_serialization_required"] is True
    assert calls == ["entered", "exited"]
    with pytest.raises(ValueError, match="unchanged gemma4_text"):
        deployment_export.probe_exporter(
            profile="e2b",
            model_dir=tmp_path,
            selected_variants=("w4",),
            full_parameter_export=True,
        )


def _deployment_proof_fixture(tmp_path, monkeypatch, *, plan_trigger, source_trigger):
    from ir_training.export import gemma4_mixed248
    from ir_training.pipeline import deployment_export

    folder = tmp_path / "w248"
    merged = tmp_path / "merged"
    folder.mkdir()
    merged.mkdir()
    artifact = folder / "model.litertlm"
    artifact.write_bytes(b"full-proof-bound-artifact")
    inspection = folder / "package_inspection.json"
    inspection.write_text("{}")
    source = merged / "deployment_source.json"
    source.write_text(
        json.dumps(
            {"full_qat_contract": {"verified": True} if source_trigger else None}
        )
    )
    proof_path = folder / "full_parameter_serialization.json"
    proof = _proof()
    proof["evidence"]["package"]["artifact_sha256"] = export._sha(artifact)
    proof_path.write_text(json.dumps(proof))
    precision = {"verified": True, "test": "unrelated precision gate"}
    monkeypatch.setattr(
        deployment_export,
        "inspect_variant_precision",
        lambda *_args, **_kwargs: precision,
    )
    recipe = folder / "w248_quantization_recipe.json"
    recipe.write_text(json.dumps(gemma4_mixed248.canonical_recipe()))
    manifest = {
        "profile": "e2b",
        "variant": "w248",
        "artifact": str(artifact.resolve()),
        "sha256": export._sha(artifact),
        "inspection_sha256": export._sha(inspection),
        "source_manifest_sha256": export._sha(source),
        "actual_precision": precision,
        **deployment_export.deployment_variants("e2b", ("w248",))["w248"],
        "mtp_exported": False,
        "official_retained_scale_export": False,
        "runtime_gpu_tested": False,
        "status": "exported_not_yet_evaluated",
        "quantization_recipe_file": {
            "name": recipe.name,
            "sha256": export._sha(recipe),
        },
        "full_parameter_serialization": {
            "verified": True,
            "state_tensor_count": 541,
            "named_parameter_count": 506,
            "persistent_buffer_count": 35,
            "path": str(proof_path.resolve()),
            "sha256": export._sha(proof_path),
        },
    }
    manifest_path = folder / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    plan = {
        "profile": "e2b",
        "merged_model_dir": str(merged),
        "variants": {"w248": {"output_dir": str(folder), "artifact": str(artifact)}},
        "full_qat_contract": {"verified": True} if plan_trigger else None,
    }
    return deployment_export, plan, manifest, manifest_path, proof, proof_path


@pytest.mark.parametrize(
    ("plan_trigger", "source_trigger"), [(True, False), (False, True)]
)
def test_output_validator_requires_complete_real_proof_for_either_trigger(
    tmp_path, monkeypatch, plan_trigger, source_trigger
):
    deployment_export, plan, _, _, _, _ = _deployment_proof_fixture(
        tmp_path,
        monkeypatch,
        plan_trigger=plan_trigger,
        source_trigger=source_trigger,
    )
    result = deployment_export.validate_deployment_export_output(plan, "w248")
    assert result["sha256"] == export._sha(Path(result["artifact"]))
    assert any(
        path.endswith("full_parameter_serialization.json") for path in result["files"]
    )


@pytest.mark.parametrize(
    ("plan_trigger", "source_trigger"), [(True, False), (False, True)]
)
def test_output_validator_rejects_absent_or_modified_proof(
    tmp_path, monkeypatch, plan_trigger, source_trigger
):
    deployment_export, plan, manifest, manifest_path, _, proof_path = (
        _deployment_proof_fixture(
            tmp_path,
            monkeypatch,
            plan_trigger=plan_trigger,
            source_trigger=source_trigger,
        )
    )
    proof_path.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        deployment_export.validate_deployment_export_output(plan, "w248")

    proof_path.write_text(json.dumps(_proof()))
    manifest["full_parameter_serialization"]["sha256"] = export._sha(proof_path)
    manifest_path.write_text(json.dumps(manifest))
    proof_path.write_text(proof_path.read_text() + " ")
    with pytest.raises(ValueError, match="missing or changed"):
        deployment_export.validate_deployment_export_output(plan, "w248")


@pytest.mark.parametrize(
    ("plan_trigger", "source_trigger"), [(True, False), (False, True)]
)
def test_output_validator_rejects_hash_bound_incomplete_inventory(
    tmp_path, monkeypatch, plan_trigger, source_trigger
):
    deployment_export, plan, manifest, manifest_path, proof, proof_path = (
        _deployment_proof_fixture(
            tmp_path,
            monkeypatch,
            plan_trigger=plan_trigger,
            source_trigger=source_trigger,
        )
    )
    proof["checkpoint_inventory"].pop("p0")
    proof_path.write_text(json.dumps(proof))
    manifest["full_parameter_serialization"]["sha256"] = export._sha(proof_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete source-to-package"):
        deployment_export.validate_deployment_export_output(plan, "w248")


def test_output_validator_rejects_full_parameter_w16(monkeypatch, tmp_path):
    from ir_training.pipeline import deployment_export

    folder = tmp_path / "w16"
    merged = tmp_path / "merged"
    folder.mkdir()
    merged.mkdir()
    artifact = folder / "model.litertlm"
    artifact.write_bytes(b"invalid-full-w16")
    inspection = folder / "package_inspection.json"
    inspection.write_text("{}")
    source = merged / "deployment_source.json"
    source.write_text(json.dumps({"full_qat_contract": {"verified": True}}))
    precision = {"verified": True}
    monkeypatch.setattr(
        deployment_export,
        "inspect_variant_precision",
        lambda *_args, **_kwargs: precision,
    )
    manifest = {
        "profile": "e2b",
        "variant": "w16",
        "artifact": str(artifact.resolve()),
        "sha256": export._sha(artifact),
        "inspection_sha256": export._sha(inspection),
        "source_manifest_sha256": export._sha(source),
        "actual_precision": precision,
    }
    (folder / "export_manifest.json").write_text(json.dumps(manifest))
    plan = {
        "profile": "e2b",
        "merged_model_dir": str(merged),
        "variants": {"w16": {"output_dir": str(folder), "artifact": str(artifact)}},
        "full_qat_contract": {"verified": True},
    }
    with pytest.raises(ValueError, match="requires E2B W248"):
        deployment_export.validate_deployment_export_output(plan, "w16")
