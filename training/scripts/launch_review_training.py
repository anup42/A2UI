"""Print a portable GPU launch plan; --execute explicitly starts it on that host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ir_training.common.config import load_yaml
from ir_training.common.progress import Progress
from ir_training.train.recipe import validate_effective_batch, validate_sft_recipe
from ir_training.train.gpu_profile import training_environment, verify_gpu_profile
from prepare_review_training import sha256, verify_prepared


def verify_launch_binding(config_path: Path) -> None:
    report_path = config_path.parent / "preparation_report.json"
    if not report_path.is_file():
        raise ValueError("Execution requires the report produced by prepare_review_training.py; regenerate a bound run plan.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("training_config_sha256") != sha256(config_path):
        raise ValueError("Training config changed after preparation; regenerate the run plan instead of editing it.")
    config = load_yaml(config_path)
    final_datasets = config.get("final_evaluation_datasets") or {}
    golden35 = (final_datasets.get("golden35") or {}).get("split_path")
    bixby50 = (final_datasets.get("bixby50") or {}).get("split_path")
    refreshed = verify_prepared(Path(config["run"]["dataset_dir"]), Path(config["golden_eval"]["split_path"]),
        max_sequence=config["training"]["max_seq_length"], max_prompt=config["golden_eval"]["max_input_tokens"],
        golden35=Path(golden35) if golden35 is not None else None,
        bixby50=Path(bixby50) if bixby50 is not None else None)
    for key in ("dataset_manifest_sha256", "golden_sha256", "tokenizer"):
        if refreshed[key] != report.get(key):
            raise ValueError(f"Prepared launch binding changed: {key}")
    if final_datasets:
        if refreshed.get("final_evaluation_datasets") != final_datasets or final_datasets != report.get("final_evaluation_datasets"):
            raise ValueError("Prepared launch binding changed: final_evaluation_datasets")
    model_dir = Path(config["model"]["model_source"])
    if not report.get("model_files"):
        raise ValueError("Preparation report has no bound model files.")
    for index, (name, digest) in enumerate(report["model_files"].items(), start=1):
        print(f"Verify model/tokenizer file {index}/{len(report['model_files'])}: {name}", flush=True)
        if not (model_dir / name).is_file() or sha256(model_dir / name) != digest:
            raise ValueError(f"Model/tokenizer bundle changed since preparation: {name}")


def launch_plan(config_path: Path, *, preflight_only: bool = False) -> tuple[list[str], dict[str, str]]:
    config = load_yaml(config_path)
    validate_sft_recipe(config)
    runtime = config.get("runtime") or {}
    devices = str(runtime.get("cuda_visible_devices", "")).strip()
    selected = [part.strip() for part in devices.split(",") if part.strip()]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Resolve an explicit unique device list with prepare_review_training.py first.")
    if int(runtime.get("world_size", len(selected))) != len(selected):
        raise ValueError("runtime.world_size disagrees with CUDA device list.")
    validate_effective_batch(config["training"], len(selected))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(selected)}", str(ROOT / "scripts/train_sft.py"), "--config", str(config_path.resolve())]
    if preflight_only:
        command.append("--preflight-only")
    environment = training_environment(
        {"cuda_visible_devices": ",".join(selected)},
        tensorboard_root=str(config["training"].get("tensorboard_root") or "/tensorboard"),
    )
    return command, environment


def verify_launch_gpu_binding(config_path: Path) -> None:
    config = load_yaml(config_path)
    runtime = config.get("runtime") or {}
    profile = runtime.get("gpu_profile")
    verify_gpu_profile(profile)
    if runtime.get("world_size") != profile["world_size"] or runtime.get("cuda_visible_devices") != profile["cuda_visible_devices"]:
        raise ValueError("GPU inventory disagrees with the resolved worker count or CUDA mask.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true", help="GPU checks only; all-parameter QAT also runs a disposable optimizer step, never saves a trained checkpoint.")
    parser.add_argument("--execute", action="store_true", help="Actually launch GPU workers. Omit to print only.")
    args = parser.parse_args()
    command, environment = launch_plan(args.config, preflight_only=args.preflight_only)
    print(json.dumps({"command": command, "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"], "execute": args.execute}, indent=2), flush=True)
    if args.execute:
        with Progress("Verify GPU launch inventory", unit="stage"):
            verify_launch_gpu_binding(args.config.resolve())
        with Progress("Verify launch dataset/model bindings", unit="stage"):
            verify_launch_binding(args.config.resolve())
        config = load_yaml(args.config)
        profile = (config.get("runtime") or {}).get("gpu_profile") or {}
        print(json.dumps({"cpu_and_gpu_execution": {
            key: profile.get(key) for key in ("world_size", "available_cpu_count", "microbatch", "effective_batch_size",
                "gradient_accumulation_steps", "dataloader_num_workers", "total_dataloader_workers", "cpu_oversubscribed")
        }, "thread_environment": {key: environment[key] for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM")}}, indent=2), flush=True)
        if profile.get("cpu_oversubscribed"):
            print("WARNING: selected DDP ranks plus DataLoader workers exceed the detected CPU budget; reduce --dataloader-workers or allocate more CPUs.", flush=True)
        completed = subprocess.run(command, env=environment, cwd=ROOT.parent, check=False)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
