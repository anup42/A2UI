"""CPU-only plan/config tests for the separate all-parameter QAT lane."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.pipeline import full_parameter_qat as workflow
from ir_training.pipeline.full_parameter_qat import FullParameterQATOptions
from ir_training.train.gpu_profile import build_gpu_profile


def _write(path: Path, value: bytes = b"fixture; never loaded") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


@pytest.fixture
def options(tmp_path: Path) -> FullParameterQATOptions:
    model = tmp_path / "reconstructed-mobile-seed"
    inputs = tmp_path / "inputs"
    model.mkdir()
    inputs.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "mobile_training_seed_manifest.json",
        "mobile_qparams.json",
    ):
        (model / name).write_text("{}", encoding="utf-8")
    for name in ("model.safetensors", "mobile_qparams.safetensors"):
        _write(model / name)
    for name in ("train.jsonl", "val.jsonl"):
        (inputs / name).write_text("{}\n", encoding="utf-8")
    return FullParameterQATOptions(
        model_dir=model,
        input_dir=inputs,
        output_dir=tmp_path / "full-qat-run",
        exporter_python=Path(sys.executable),
        steps=20,
        eval_steps=5,
        golden_every_steps=10,
        progress_seconds=0.01,
        stage_timeout_seconds=30,
        generation_timeout_seconds=30,
    )


def _inventory(count: int, *, name: str = "NVIDIA H100 80GB HBM3", gib: int = 80):
    devices = [
        {
            "visible_index": index,
            "launch_identifier": str(index),
            "uuid": f"GPU-{index}",
            "name": name,
            "total_memory_bytes": gib * 1024**3,
            "compute_capability": [9, 0],
        }
        for index in range(count)
    ]
    return {
        "version": 1,
        "inherited_cuda_visible_devices": None,
        "visible_gpu_count": count,
        "devices": devices,
    }


def test_plan_is_offline_separate_and_explicit_about_experimental_topology(options):
    plan = workflow.build_plan(options)

    assert not options.output_dir.exists()
    assert plan["workflow"] == "e2b_all_parameter_qat_v1"
    assert plan["training_contract"] == {
        "parameter_scope": "all_unique_text_model_parameters",
        "master_parameter_dtype": "float32",
        "autocast": "bfloat16",
        "optimizer": "adafactor",
        "quantizer": "ste_ai_edge",
        "quantized_modules": "all fully-connected weights and both token embeddings",
        "activation_quantization": False,
        "retained_mobile_scales": False,
        "lora": False,
    }
    assert plan["export"]["variant"] == "w248"
    assert plan["export"]["official_static_mobile_layout"] is False
    assert plan["mtp"] == {"training": False, "inference": False, "exported": False}
    assert plan["selection"] == {
        "cohort": "golden32",
        "metric": workflow.SELECTOR,
        "golden35_used": False,
        "bixby50_used": False,
    }
    assert plan["stages"] == [
        "assets",
        "prepare",
        "configure",
        "preflight",
        "training",
        "best_golden32",
        "final_golden35",
        "final_bixby50",
        "export",
        "scorecard",
    ]
    assert Path(plan["paths"]["config"]) == options.output_dir / "fit/training_config.yaml"
    assert Path(plan["paths"]["preparation_report"]) == options.output_dir / "fit/preparation_report.json"


@pytest.mark.parametrize(
    "missing",
    [
        "mobile_training_seed_manifest.json",
        "mobile_qparams.json",
        "mobile_qparams.safetensors",
        "model.safetensors",
    ],
)
def test_plan_rejects_incomplete_reconstructed_mobile_seed(options, missing):
    (options.model_dir / missing).unlink()
    with pytest.raises(FileNotFoundError, match="mobile-seed|no safetensor"):
        workflow.build_plan(options)
    assert not options.output_dir.exists()


def test_execute_requires_separate_experimental_acknowledgement(options):
    with pytest.raises(ValueError, match="allow-experimental-export"):
        workflow.run_pipeline(options, execute=True)
    assert not options.output_dir.exists()


@pytest.mark.parametrize("count,accumulation", [(2, 16), (4, 8), (8, 4)])
def test_full_recipe_keeps_fp32_all_parameter_qat_and_safe_h100_batch(
    options, count, accumulation
):
    plan = workflow.build_plan(options)
    profile = build_gpu_profile(_inventory(count), model="e2b", cpu_count=64)
    report = {
        "tokenizer": {"chat_template_kwargs": {"enable_thinking": True}},
        "final_evaluation_datasets": {
            "golden35": {"selection_role": "final_only_holdout"},
            "bixby50": {"selection_role": "final_only_holdout"},
        },
    }

    config = workflow.training_config(plan, profile, report)

    assert config["run"]["purpose"] == workflow.WORKFLOW
    assert config["run"]["output_dir"] == plan["paths"]["training"]
    assert config["golden_eval"]["output_dir"] == str(
        options.output_dir / "fit/training/periodic_golden32"
    )
    assert config["training"]["logging_dir"] == str(
        options.output_dir / "fit/training/tensorboard"
    )
    assert config["golden_eval"]["best_checkpoint_dir"] == plan["paths"][
        "best_checkpoint"
    ]
    assert "lora" not in config
    assert "qat_mtp" not in config
    assert config["model"]["dtype"] == "float32"
    assert config["training"]["method"] == "full_finetune_qat"
    assert config["training"]["full_parameter_training"] is True
    assert config["training"]["mixed_precision"] == "bf16"
    assert config["training"]["optim"] == "adafactor"
    assert config["training"]["per_device_train_batch_size"] == 1
    assert config["training"]["gradient_accumulation_steps"] == accumulation
    assert config["training"]["expected_effective_batch_size"] == 32
    assert config["qat"]["scale_mode"] == "dynamic"
    assert config["qat"]["quantize_embeddings"] is True
    assert config["qat"]["activation_bits"] == 32
    assert config["qat"]["effective_lora_only"] is False
    assert config["preflight"]["numeric_policy"] == "all_parameter_qat_safety_v1"


def test_generated_writable_paths_are_isolated_between_runs(options):
    second = replace(options, output_dir=options.output_dir.parent / "full-qat-run-2")
    report = {"tokenizer": {}, "final_evaluation_datasets": {}}
    profile = build_gpu_profile(_inventory(2), model="e2b", cpu_count=64)

    plans = [workflow.build_plan(item) for item in (options, second)]
    configs = [workflow.training_config(plan, profile, report) for plan in plans]

    for item, plan, config in zip((options, second), plans, configs, strict=True):
        run = item.output_dir.resolve()
        assert Path(config["run"]["output_dir"]).is_relative_to(run)
        assert Path(config["golden_eval"]["output_dir"]).is_relative_to(run)
        assert Path(config["training"]["logging_dir"]).is_relative_to(run)
        assert config["golden_eval"]["best_checkpoint_dir"] == plan["paths"][
            "best_checkpoint"
        ]

    assert configs[0]["golden_eval"]["output_dir"] != configs[1]["golden_eval"][
        "output_dir"
    ]
    assert configs[0]["training"]["logging_dir"] != configs[1]["training"][
        "logging_dir"
    ]


@pytest.mark.parametrize(
    "inventory,expected",
    [
        (_inventory(1), "2, 4, or 8"),
        (_inventory(3), "2, 4, or 8"),
        (_inventory(2, name="NVIDIA A100 80GB"), "H100"),
        (_inventory(2, gib=78), "79 GiB"),
    ],
)
def test_gpu_gate_rejects_unsupported_full_parameter_hosts(inventory, expected):
    count = inventory["visible_gpu_count"]
    profile = build_gpu_profile(
        inventory,
        model="e2b",
        effective_batch=32 if 32 % count == 0 else 30,
        cpu_count=64,
    )
    with pytest.raises(ValueError, match=expected):
        workflow.validate_h100_profile(profile)


def test_gpu_gate_rejects_explicit_microbatch_above_one():
    profile = build_gpu_profile(
        _inventory(2), model="e2b", microbatch=2, effective_batch=32, cpu_count=64
    )
    with pytest.raises(ValueError, match="microbatch to 1"):
        workflow.validate_h100_profile(profile)


def test_evaluations_all_use_selected_full_checkpoint_with_qat_enabled(options):
    plan = workflow.build_plan(options)
    command = workflow.evaluation_command(plan, "bixby50", "final_bixby50")

    assert command[command.index("--checkpoint") + 1] == plan["paths"]["best_checkpoint"]
    assert command[command.index("--checkpoint-kind") + 1] == "merged"
    assert command[command.index("--qat-mode") + 1] == "on"
    assert command[command.index("--required-rows") + 1] == "50"
    assert "--require-prepared-contract" in command
    assert "--require-gpu" in command


def test_export_request_is_dense_e2b_w248_and_never_lora_merge(options):
    plan = workflow.build_plan(replace(options, allow_experimental_export=True))
    export = workflow.checkpoint_export_options(plan)

    assert export.profile == "e2b"
    assert export.fit_dir == options.output_dir / "fit"
    assert export.checkpoint == options.output_dir / "fit/training/best_golden_checkpoint"
    assert export.variants == ("w248",)
    assert export.allow_experimental_formats is True


def test_assets_probe_w248_exporter_before_training(options, monkeypatch):
    plan = workflow.build_plan(options)
    options.output_dir.mkdir()
    calls = []

    import ir_training.qat.mobile_qparams as qparams
    import ir_training.qat.mobile_training_seed as seed

    monkeypatch.setattr(
        seed,
        "verify_configured_mobile_training_seed",
        lambda *args, **kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        qparams,
        "verify_mobile_qparams_contract",
        lambda *args, **kwargs: {"verified": True},
    )

    def runner(command, log, environment, **kwargs):
        calls.append((command, environment))
        report = Path(command[command.index("--report") + 1])
        report.parent.mkdir(parents=True)
        report.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "profile": "e2b",
                    "variants": ["w248"],
                    "model_loaded": False,
                    "conversion_tested": False,
                    "full_parameter_export": True,
                    "all_parameter_serialization_required": True,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(workflow, "run_bounded_command", runner)
    files = workflow._assets(plan)

    assert len(calls) == 1
    command, environment = calls[0]
    assert command[0] == str(options.exporter_python)
    assert command[command.index("--variants") + 1 :] == ["w248"]
    assert "--full-parameter-export" in command
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert options.output_dir / "exporter_preflight/exporter_preflight.json" in files


def test_gpu_environment_uses_only_bound_selected_devices(options, monkeypatch):
    plan = workflow.build_plan(options)
    config = {
        "runtime": {
            "gpu_profile": build_gpu_profile(
                _inventory(4), model="e2b", devices="1,3", effective_batch=32, cpu_count=64
            )
        }
    }
    path = Path(plan["paths"]["config"])
    path.parent.mkdir(parents=True)
    import yaml

    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("A2UI_CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("A2UI_EXCLUDE_CUDA_DEVICES", "0")

    environment = workflow._gpu_environment(plan)

    assert environment["CUDA_VISIBLE_DEVICES"] == "1,3"
    assert environment["A2UI_SKIP_CUDA_DEVICE_NORMALIZE"] == "1"
    assert "A2UI_CUDA_VISIBLE_DEVICES" not in environment
    assert "A2UI_EXCLUDE_CUDA_DEVICES" not in environment


def test_full_lane_defaults_expandable_allocator_before_child_launch(options, monkeypatch):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    environment = workflow._environment(workflow.build_plan(options))
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "PYTORCH_ALLOC_CONF" not in environment
    # Environment construction must not change the caller or the LoRA lane.
    assert "PYTORCH_CUDA_ALLOC_CONF" not in workflow.os.environ


@pytest.mark.parametrize("settings", [
    {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"},
    {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
    {"PYTORCH_ALLOC_CONF": "", "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
])
def test_full_lane_preserves_explicit_allocator_settings(options, monkeypatch, settings):
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    environment = workflow._environment(workflow.build_plan(options))
    assert {name: environment[name] for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
            if name in environment} == settings


def test_preflight_receipt_requires_one_disposable_step_from_every_rank(options):
    plan = workflow.build_plan(options)
    profile = build_gpu_profile(_inventory(2), model="e2b", cpu_count=64)
    config_path = Path(plan["paths"]["config"])
    config_path.parent.mkdir(parents=True)
    import yaml

    config_path.write_text(
        yaml.safe_dump({"runtime": {"gpu_profile": profile}}), encoding="utf-8"
    )
    training = Path(plan["paths"]["training"])
    training.mkdir()
    for rank in range(2):
        (training / f"full_optimizer_preflight_rank{rank}.json").write_text(
            json.dumps(
                {
                    "training_config_sha256": workflow.sha256(config_path),
                    "rank": rank,
                    "world_size": 2,
                    "probe": {
                        "passed": True,
                        "disposable_optimizer_steps": 1,
                        "checkpoint_writes": 0,
                        "model_must_not_be_reused": True,
                    },
                }
            ),
            encoding="utf-8",
        )

    reports = workflow._optimizer_preflight_files(plan)
    assert [path.name for path in reports] == [
        "full_optimizer_preflight_rank0.json",
        "full_optimizer_preflight_rank1.json",
    ]

    payload = json.loads(reports[1].read_text(encoding="utf-8"))
    payload["probe"]["disposable_optimizer_steps"] = 0
    reports[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Rank 1 optimizer preflight"):
        workflow._optimizer_preflight_files(plan)


def test_best_checkpoint_validation_uses_dense_export_hash_contract(options, monkeypatch):
    plan = workflow.build_plan(options)
    checkpoint = Path(plan["paths"]["best_checkpoint"])
    checkpoint.mkdir(parents=True)
    (checkpoint / "training_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_role": "best_golden",
                "best_golden_eval": {"metric": workflow.SELECTOR},
            }
        ),
        encoding="utf-8",
    )
    calls = []
    import ir_training.pipeline.deployment_export as deployment

    def verify(config, artifact, profile):
        calls.append((config, artifact, profile))
        return {"full_qat_contract": {"verified": True}}

    monkeypatch.setattr(deployment, "verify_checkpoint_source", verify)
    result = workflow._validate_best_checkpoint(plan)

    assert result == {"verified": True}
    assert calls == [
        (Path(plan["paths"]["config"]), checkpoint, "e2b")
    ]


def test_cli_refuses_execute_without_experimental_acknowledgement(tmp_path):
    script = ROOT / "scripts/run_full_parameter_qat_pipeline.py"
    spec = importlib.util.spec_from_file_location("full_qat_cli", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with pytest.raises(SystemExit, match="2"):
        module.main(
            [
                "--model-dir",
                str(tmp_path / "model"),
                "--input-dir",
                str(tmp_path / "input"),
                "--output-dir",
                str(tmp_path / "output"),
                "--exporter-python",
                sys.executable,
                "--execute",
            ]
        )
