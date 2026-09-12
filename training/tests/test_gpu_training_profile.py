from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.train.gpu_profile import build_gpu_profile, detect_cuda_devices, select_devices, verify_gpu_profile, visible_launch_profile


def inventory(count: int, *, mask: str | None = None, name: str = "NVIDIA H100 80GB HBM3", memory_gib: int = 80):
    cuda = SimpleNamespace(
        is_available=lambda: count > 0,
        device_count=lambda: count,
        get_device_properties=lambda index: SimpleNamespace(
            name=name, total_memory=memory_gib * 1024**3, major=9, minor=0, uuid=f"GPU-{index}",
        ),
    )
    return detect_cuda_devices(torch_module=SimpleNamespace(cuda=cuda), environment={} if mask is None else {"CUDA_VISIBLE_DEVICES": mask})


@pytest.mark.parametrize("count", [2, 4, 8])
@pytest.mark.parametrize("model,micro", [("e2b", 2), ("270m", 4)])
def test_h100_profiles_use_all_gpus_with_stable_batch(count, model, micro):
    profile = build_gpu_profile(inventory(count), model=model, cpu_count=64)
    assert profile["world_size"] == count
    assert profile["cuda_visible_devices"] == ",".join(map(str, range(count)))
    assert profile["microbatch"] == micro
    assert profile["effective_batch_size"] == 32
    assert count * micro * profile["gradient_accumulation_steps"] == 32
    assert profile["dtype"] == "bfloat16" and profile["tf32"]
    assert profile["attn_implementation"] == "sdpa"
    assert not profile["benchmark_verified"]
    verify_gpu_profile(profile, inventory(count))


@pytest.mark.parametrize("mask", ["2,5", "GPU-0,GPU-1", "MIG-aaa,MIG-bbb"])
def test_scheduler_mask_survives_logical_selection(mask):
    observed = inventory(2, mask=mask)
    assert build_gpu_profile(observed, model="e2b")["cuda_visible_devices"] == mask
    subset = build_gpu_profile(observed, model="e2b", devices="1")
    assert subset["cuda_visible_devices"] == mask.split(",")[1]
    assert subset["selected_devices"][0]["visible_index"] == 1


def test_uuid_selection_stays_inside_visible_inventory():
    observed = inventory(2, mask="MIG-aaa,MIG-bbb")
    assert select_devices(observed, "MIG-bbb")[0]["launch_identifier"] == "MIG-bbb"
    assert select_devices(observed, "GPU-0")[0]["visible_index"] == 0
    for invalid in ("2", "GPU-unassigned", "0,GPU-0", "0,0"):
        with pytest.raises(ValueError):
            select_devices(observed, invalid)


def test_legacy_launch_ids_cannot_escape_scheduler_mask(monkeypatch):
    observed = inventory(2, mask="2,5")
    monkeypatch.setattr("ir_training.train.gpu_profile.detect_cuda_devices", lambda: observed)
    assert visible_launch_profile(model="e2b")["cuda_visible_devices"] == "2,5"
    assert visible_launch_profile(model="e2b", launch_ids="5")["cuda_visible_devices"] == "5"
    with pytest.raises(ValueError, match="outside"):
        visible_launch_profile(model="e2b", launch_ids="0")
    with pytest.raises(ValueError, match="disagrees"):
        visible_launch_profile(model="e2b", launch_ids="2,5", num_gpus=1)
    with pytest.raises(ValueError, match="count"):
        visible_launch_profile(model="e2b", num_gpus=4)


def test_zero_gpu_and_inconsistent_masks_fail():
    with pytest.raises(ValueError, match="No CUDA-visible"):
        inventory(0)
    with pytest.raises(ValueError, match="disagrees"):
        inventory(2, mask="GPU-0")


def test_batch_overrides_cap_auto_microbatch_and_never_change_learning_rate():
    profile = build_gpu_profile(inventory(8), model="270m", effective_batch=16)
    assert profile["microbatch"] == 2
    assert profile["gradient_accumulation_steps"] == 1
    assert "learning_rate" not in profile
    profile = build_gpu_profile(inventory(4), model="e2b", effective_batch=16, microbatch=1)
    assert profile["gradient_accumulation_steps"] == 4
    with pytest.raises(ValueError, match="divisible"):
        build_gpu_profile(inventory(3), model="e2b")
    with pytest.raises(ValueError, match="microbatch"):
        build_gpu_profile(inventory(8), model="e2b", effective_batch=16, microbatch=4)


def test_other_hardware_and_small_h100_partitions_use_conservative_profile():
    for observed in (inventory(4, name="NVIDIA A100"), inventory(4, memory_gib=20)):
        profile = build_gpu_profile(observed, model="e2b", cpu_count=4)
        assert profile["microbatch"] == 1
        assert profile["effective_batch_size"] == 16
        assert profile["dataloader_num_workers"] == 0


@pytest.mark.parametrize("change", ["count", "mask", "memory", "uuid", "name"])
def test_launch_revalidation_rejects_stale_hardware(change):
    observed = inventory(2)
    profile = build_gpu_profile(observed, model="e2b")
    refreshed = copy.deepcopy(observed)
    if change == "count":
        refreshed["visible_gpu_count"] = 4
    elif change == "mask":
        refreshed["inherited_cuda_visible_devices"] = "1,2"
    elif change == "memory":
        refreshed["devices"][0]["total_memory_bytes"] = 20 * 1024**3
    elif change == "uuid":
        refreshed["devices"][0]["uuid"] = "GPU-replaced"
    else:
        refreshed["devices"][0]["name"] = "Different GPU"
    with pytest.raises(ValueError, match="changed"):
        verify_gpu_profile(profile, refreshed)


def test_launch_revalidates_runtime_mask_against_hardware_profile(tmp_path, monkeypatch):
    import yaml
    path = ROOT / "scripts" / "launch_review_training.py"
    spec = importlib.util.spec_from_file_location("launch_gpu_profile_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = inventory(2, mask="GPU-0,GPU-1")
    profile = build_gpu_profile(observed, model="e2b")
    monkeypatch.setattr("ir_training.train.gpu_profile.detect_cuda_devices", lambda: observed)
    config_path = tmp_path / "config.yaml"
    config = {"runtime": {"world_size": 2, "cuda_visible_devices": "GPU-0,GPU-1", "gpu_profile": profile}}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    module.verify_launch_gpu_binding(config_path)
    config["runtime"]["world_size"] = 4
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        module.verify_launch_gpu_binding(config_path)


@pytest.mark.parametrize("count", [2, 4, 8])
def test_preparation_wires_hardware_logging_and_generation_without_model_load(tmp_path, monkeypatch, count):
    path = ROOT / "scripts" / "prepare_review_training.py"
    spec = importlib.util.spec_from_file_location("prepare_gpu_profile_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = inventory(count)
    monkeypatch.setattr("ir_training.train.gpu_profile.detect_cuda_devices", lambda: observed)
    monkeypatch.delenv("A2UI_TENSORBOARD_ROOT", raising=False)
    monkeypatch.setattr(module, "verify_prepared", lambda *args, **kwargs: {
        "tokenizer": {"chat_template_kwargs": {"enable_thinking": False}},
        "benchmark": {"revision": "fixture", "benchmark_kind": "explicit_repeated_case"},
    })
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture, never loaded")
    data = tmp_path / "data"
    data.mkdir()
    golden = data / "golden32.jsonl"
    golden.write_text("", encoding="utf-8")
    args = SimpleNamespace(profile="e2b", model_dir=model, dataset_dir=data, golden_file=golden,
        output_dir=tmp_path / "planned_run", devices="auto", microbatch=None, effective_batch=None,
        epochs=1, steps=20, max_seq_length=4096, resume=None, qv_baseline=False, qat=False)
    config, report = module.build_config(args)
    assert config["runtime"]["world_size"] == count
    assert config["training"]["per_device_train_batch_size"] == 2
    assert config["training"]["gradient_accumulation_steps"] == 32 // (count * 2)
    assert config["training"]["tensorboard_root"] == "/tensorboard"
    assert config["training"]["learning_rate"] == 0.00002
    assert config["training"]["eval_steps"] == 20
    assert config["golden_eval"]["interval"] == 2
    assert config["golden_eval"]["metric_for_best_model"] == "unique_source_generation_reward_v5_4_avg"
    assert config["golden_eval"]["max_new_tokens"] == 2048
    assert config["golden_eval"]["tensorboard"] is True
    assert not report["model_loaded"] and not report["training_executed"]
    assert not args.output_dir.exists()


def test_canonical_270m_materializes_ddp_profile_and_reuses_it_for_later_stages(tmp_path, monkeypatch):
    import yaml
    from ir_training.common.config import load_yaml
    from ir_training.pipeline.gemma270m_multiformat import build_pipeline_plan, _materialize_resolved_training_config

    pipeline_path = ROOT / "configs/pipelines/gemma3_270m_a2ui_express_multiformat.yaml"
    pipeline = load_yaml(pipeline_path)
    source = load_yaml(ROOT / "configs/models/gemma3_270m_a2ui_express_qat.yaml")
    source["run"]["output_dir"] = str(tmp_path / "runs")
    source_path = tmp_path / "training.yaml"
    source_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    profile = build_gpu_profile(inventory(4, mask="GPU-0,GPU-1,GPU-2,GPU-3"), model="270m")
    pipeline["pipeline"]["training"].update(config=str(source_path), host_gpu_profile=profile)
    pipeline["pipeline"]["output_dir"] = str(tmp_path / "outputs")
    plan = build_pipeline_plan(pipeline, run_id_override="gpu_fixture")
    assert "--nproc_per_node=4" in plan["training"]["command"]
    assert plan["training"]["command"][1:3] == ["-m", "torch.distributed.run"]
    _materialize_resolved_training_config(plan)
    resolved = load_yaml(plan["training"]["resolved_config"])
    assert resolved["runtime"]["world_size"] == 4
    assert resolved["training"]["expected_effective_batch_size"] == 32
    assert resolved["training"]["gradient_accumulation_steps"] == 2
    assert resolved["qat"] == source["qat"]
    assert resolved["training"]["learning_rate"] == source["training"]["learning_rate"]
    del pipeline["pipeline"]["training"]["host_gpu_profile"]
    def no_detection():
        raise AssertionError("Later export/evaluation plan must not probe GPUs")
    monkeypatch.setattr("ir_training.train.gpu_profile.detect_cuda_devices", no_detection)
    resumed_plan = build_pipeline_plan(pipeline, run_id_override="gpu_fixture")
    assert resumed_plan["training"]["resolved_config_sha256"] == plan["training"]["resolved_config_sha256"]
    assert resumed_plan["training"]["command"] == plan["training"]["command"]
