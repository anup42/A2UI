"""CUDA/NVIDIA UUID spelling compatibility without loosening GPU allocation checks."""
import json
from copy import deepcopy
from dataclasses import replace

import pytest
import test_golden_deployment as fixtures
from ir_training.pipeline import golden_deployment as deployment

options = fixtures.options
export_validator = fixtures.export_validator


def profile(*values):
    return {"selected_devices": [{"uuid": value, "launch_identifier": str(i)} for i, value in enumerate(values)]}


def mixed_inventory():
    result = fixtures.inventory()
    for i, device in enumerate(result["devices"]):
        if i % 2:
            device["uuid"] = device["uuid"].removeprefix("GPU-")
    return result


def test_normalizes_only_selected_ids_without_mutating_profile():
    selected = profile(fixtures.gpu_uuid(3).removeprefix("GPU-").upper(), " " + fixtures.gpu_uuid(1) + " ")
    selected["inventory"] = fixtures.inventory()
    before = deepcopy(selected)
    assert deployment._litert_allowed_gpu_uuids(selected) == [fixtures.gpu_uuid(3), fixtures.gpu_uuid(1)]
    assert selected == before


@pytest.mark.parametrize("value", [None, "", " ", 12, True, [], {}, "N/A", "0", "GPU-", "GPU-not-a-uuid",
                                  "12345678", "GPU-GPU-12345678-abcd-4321-abcd-000000000000"])
def test_bad_id_is_not_silently_dropped_among_valid_ids(value):
    with pytest.raises(ValueError, match="complete NVIDIA GPU UUID"):
        deployment._litert_allowed_gpu_uuids(profile(fixtures.gpu_uuid(0), value))


@pytest.mark.parametrize("selected", [None, [], {}, "GPU-test"])
def test_selected_list_must_be_nonempty(selected):
    with pytest.raises(ValueError, match="selected NVIDIA GPU UUIDs"):
        deployment._litert_allowed_gpu_uuids({"selected_devices": selected})


@pytest.mark.parametrize("devices", [[{}], [None], ["bad"], [{"uuid": None}, {}]])
def test_missing_device_identity_is_an_error(devices):
    with pytest.raises(ValueError, match="every selected GPU"):
        deployment._litert_allowed_gpu_uuids({"selected_devices": devices})


def test_duplicates_detected_after_prefix_and_case_normalization():
    with pytest.raises(ValueError, match="unique"):
        deployment._litert_allowed_gpu_uuids(profile(fixtures.gpu_uuid(0), fixtures.gpu_uuid(0)[4:].upper()))


@pytest.mark.parametrize("field,value", [
    ("uuid", "MIG-12345678-abcd-4321-abcd-000000000000"),
    ("uuid", "MIG-GPU-12345678-abcd-4321-abcd-000000000000/1/0"),
    ("launch_identifier", "MIG-12345678-abcd-4321-abcd-000000000000"),
])
def test_mig_is_not_relabelled_as_a_physical_gpu(field, value):
    selected = profile(fixtures.gpu_uuid(0)[4:])
    selected["selected_devices"][0][field] = value
    with pytest.raises(ValueError, match="MIG selection is not supported"):
        deployment._litert_allowed_gpu_uuids(selected)


@pytest.mark.parametrize("indices", [[1, 3], [1, 3, 5, 7], list(range(8))])
def test_full_deployment_handoffs_normalized_subset_at_2_4_8_gpus(options, export_validator, indices):
    probe = mixed_inventory()
    before = deepcopy(probe)
    options = replace(options, base=replace(options.base, devices=",".join(map(str, indices))))
    calls = []
    result = deployment.run_deployment(
        options, execute=True,
        command_runner=fixtures.mock_runner(calls, expected_gpu_uuids=[fixtures.gpu_uuid(i) for i in indices]),
        pipeline_runner=fixtures.fake_pipeline, writer_factory=fixtures.Writer, gpu_probe=lambda: probe,
    )
    assert result["status"] == "complete" and len(result["results"]) == 14
    assert probe == before
    saved = json.loads((options.base.output_dir / "gpu_preflight.json").read_text())
    assert saved["selected_devices"] == [probe["devices"][i] for i in indices]
    assert saved["cuda_visible_devices"] == ",".join(map(str, indices))
    assert len([argv for argv, _, _ in calls if "--preflight" in argv]) == 1
    assert len([argv for argv, _, _ in calls if "--builtin-gpu" in argv]) == 8


def test_missing_uuid_stops_before_subprocesses_or_training(options):
    probe = mixed_inventory()
    probe["devices"][3]["uuid"] = None
    calls = []
    with pytest.raises(ValueError, match="every selected GPU"):
        deployment.run_deployment(
            options, execute=True, command_runner=fixtures.mock_runner(calls),
            pipeline_runner=lambda *a, **k: pytest.fail("Training must not run"),
            writer_factory=fixtures.Writer, gpu_probe=lambda: probe,
        )
    assert not calls


def test_recovery_normalizes_both_handoffs_without_rewriting_bound_profile(options, export_validator):
    with pytest.raises(RuntimeError, match="simulated"):
        deployment.run_deployment(
            options, execute=True, command_runner=fixtures.mock_runner([], failure="w32_golden32"),
            pipeline_runner=fixtures.fake_pipeline, writer_factory=fixtures.Writer, gpu_probe=mixed_inventory,
        )
    saved = options.base.output_dir / "gpu_preflight.json"
    before = saved.read_bytes()
    calls = []
    result = deployment.run_deployment(
        replace(options, resume_run=True), execute=True, command_runner=fixtures.mock_runner(calls),
        pipeline_runner=lambda *a, **k: pytest.fail("Completed training must not rerun"),
        writer_factory=fixtures.Writer, gpu_probe=mixed_inventory,
    )
    assert result["status"] == "complete"
    assert saved.read_bytes() == before
    assert len([argv for argv, _, _ in calls if "--preflight" in argv]) == 1
    assert len([argv for argv, _, _ in calls if "--builtin-gpu" in argv]) == 8
