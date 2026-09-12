"""Resolve a review recipe from GPU metadata and files; no model load or training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.common.config import load_yaml
from ir_training.train.recipe import validate_effective_batch, validate_sft_recipe


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source_identity(row: dict) -> tuple[str, str]:
    source = str(row.get("response_text") or "").strip()
    if not source:
        raise ValueError("Prepared row is missing canonical response_text.")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    query = row.get("query_id") or metadata.get("query_id") or row.get("source_id")
    if not query:
        raise ValueError("Prepared row is missing its query/source ID.")
    return str(query), hashlib.sha256(" ".join(source.split()).encode()).hexdigest()


def verify_prepared(dataset: Path, golden: Path, *, max_sequence: int, max_prompt: int) -> dict:
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = manifest.get("validation") or {}
    if not all(validation.get(key) is True for key in ("strict_express", "wire_schema", "semantic_roundtrip")) or validation.get("root_reachability") != 1.0:
        raise ValueError("Dataset manifest must certify strict Express, wire schema, semantic roundtrip and complete root reachability.")
    tokenizer = manifest.get("tokenizer") or {}
    if tokenizer.get("max_seq_length") != max_sequence or not tokenizer.get("vocabulary_sha256"):
        raise ValueError("Prepare splits with the selected tokenizer and the recipe's --max-seq-length first.")
    splits = manifest.get("splits") or {}
    train_ids, train_sources = set(), set()
    val_ids, val_sources = set(), set()
    counts = {}
    for name in ("train", "val"):
        path = dataset / f"{name}.jsonl"
        if sha256(path) != (splits.get(name) or {}).get("output_sha256"):
            raise ValueError(f"Prepared {name} hash differs from its manifest.")
        identities = [source_identity(row) for row in rows(path)]
        if not identities:
            raise ValueError(f"Empty {name} split")
        counts[name] = len(identities)
        ids = {query for query, _ in identities if query}
        sources = {source for _, source in identities}
        if name == "train":
            train_ids, train_sources = ids, sources
        elif ids & train_ids or sources & train_sources:
            raise ValueError("Train/val overlap by query ID or normalized response text; rebuild splits before training.")
        else:
            val_ids, val_sources = ids, sources
    golden_manifest = json.loads((golden.parent / "manifest.json").read_text(encoding="utf-8"))
    golden_validation = golden_manifest.get("validation") or {}
    if not all(golden_validation.get(key) is True for key in ("strict_express", "wire_schema", "semantic_roundtrip")) or golden_validation.get("root_reachability") != 1.0:
        raise ValueError("Golden manifest lacks strict target validation.")
    if manifest.get("scaffold_count") != 1 or golden_manifest.get("scaffold_count") != 1:
        raise ValueError("The review baseline requires one shared prompt scaffold; separate mixed-prompt experiments.")
    for directory, evidence in ((dataset, manifest), (golden.parent, golden_manifest)):
        if sha256(directory / "prompt_scaffolds.json") != evidence.get("prompt_scaffolds_sha256"):
            raise ValueError("Saved prompt scaffold changed after preparation.")
    if manifest["prompt_scaffolds_sha256"] != golden_manifest["prompt_scaffolds_sha256"]:
        raise ValueError("Training and Golden prompt scaffolds differ.")
    golden_split = (golden_manifest.get("splits") or {}).get(golden.stem) or {}
    if sha256(golden) != golden_split.get("output_sha256"):
        raise ValueError("Golden file must match a checked preparation manifest.")
    if golden_split.get("quarantined_rows", 0):
        raise ValueError("Golden preparation quarantined rows. Regenerate invalid targets through Stage 3; preserve all 32 cases.")
    from ir_training.eval.golden_set import benchmark_contract_for_split, load_fixed_golden_rows
    gold = load_fixed_golden_rows(golden, required_rows=32, require_exact_rows=True, require_unique_rows=True)
    benchmark = benchmark_contract_for_split(golden, gold)
    identities = [source_identity(row) for row in gold]
    required_unique = 31 if (benchmark or {}).get("benchmark_kind") == "explicit_repeated_case" else 32
    if len(gold) != 32 or len(set(identities)) != required_unique or len({key for key, _ in identities}) != required_unique or len({source for _, source in identities}) != required_unique:
        raise ValueError(f"Golden32 requires exactly {required_unique} unique query IDs and sources in this benchmark revision.")
    if any((query and query in train_ids) or source in train_sources for query, source in identities):
        raise ValueError("Golden32 overlaps training by query ID or normalized source response.")
    if any(query in val_ids or source in val_sources for query, source in identities):
        raise ValueError("Golden32 overlaps validation. Filter the ordinary validation split against reserved Golden cohorts first.")
    for excluded in (benchmark or {}).get("excluded_sources", []):
        if excluded.get("query_id") in train_ids | val_ids or excluded.get("response_sha256") in train_sources | val_sources:
            raise ValueError("Replaced Golden source is still reserved and must not enter train/val.")
    gt = golden_manifest.get("tokenizer") or {}
    if any(gt.get(key) != tokenizer.get(key) for key in ("vocabulary_sha256", "chat_template_sha256", "chat_template_kwargs")):
        raise ValueError("Train and Golden tokenizer/chat-template fingerprints differ.")
    maxima = golden_split.get("max_accepted_token_lengths") or {}
    if int(maxima.get("prompt_tokens", max_prompt + 1)) > max_prompt:
        raise ValueError("Golden prompt exceeds inference context. Reprepare with --max-input-tokens.")
    return {"split_rows": counts, "golden_rows": 32, "golden_unique_sources": required_unique,
            "benchmark": benchmark, "dataset_manifest_sha256": sha256(manifest_path), "golden_sha256": sha256(golden), "tokenizer": tokenizer}


def build_config(args: argparse.Namespace) -> tuple[dict, dict]:
    import os
    from ir_training.train.gpu_profile import build_gpu_profile, detect_cuda_devices

    profile = "gemma4_e2b_a2ui_express_review_sft.yaml" if args.profile == "e2b" else "gemma3_270m_a2ui_express_review_sft.yaml"
    config = copy.deepcopy(load_yaml(ROOT / "configs/models" / profile))
    model_dir, dataset, golden = args.model_dir.resolve(strict=True), args.dataset_dir.resolve(strict=True), args.golden_file.resolve(strict=True)
    if not (model_dir / "config.json").is_file() or not any(model_dir.glob("*.safetensors")):
        raise ValueError("--model-dir must contain the dense HF model config and safetensors weights.")
    if not (model_dir / "tokenizer_config.json").is_file():
        raise ValueError("The local model bundle must include its tokenizer and chat template.")
    gpu_profile = build_gpu_profile(
        detect_cuda_devices(), model=args.profile, devices=getattr(args, "devices", "auto"),
        microbatch=getattr(args, "microbatch", None), effective_batch=getattr(args, "effective_batch", None),
        dataloader_workers=getattr(args, "dataloader_workers", None),
    )
    gpu_profile["gradient_checkpointing"] = bool(getattr(args, "gradient_checkpointing", True))
    gpu_profile["attn_implementation"] = getattr(args, "attn_implementation", "sdpa")
    world_size = gpu_profile["world_size"]
    effective_batch = gpu_profile["effective_batch_size"]
    run_dir = args.output_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Choose a new run directory: {run_dir}")
    config["run"].update(id=run_dir.name, dataset_dir=str(dataset), output_dir=str(run_dir / "training"), prepared_manifest_required=True)
    config["runtime"] = {"cuda_visible_devices": gpu_profile["cuda_visible_devices"], "world_size": world_size, "gpu_profile": gpu_profile}
    config["model"].update(model_source=str(model_dir), tokenizer_source=str(model_dir), dtype=gpu_profile["dtype"],
                          attn_implementation=gpu_profile["attn_implementation"])
    training = config["training"]
    tensorboard_root = os.environ.get("A2UI_TENSORBOARD_ROOT") or "/tensorboard"
    training.update(per_device_train_batch_size=gpu_profile["microbatch"], gradient_accumulation_steps=gpu_profile["gradient_accumulation_steps"],
                    expected_effective_batch_size=effective_batch, epochs=args.epochs, max_seq_length=args.max_seq_length,
                    tensorboard_root=tensorboard_root, tensorboard_subdir="training", report_to="tensorboard",
                    logging_dir=str(Path(tensorboard_root) / run_dir.name / "training"),
                    tf32=gpu_profile["tf32"], dataloader_num_workers=gpu_profile["dataloader_num_workers"],
                    dataloader_pin_memory=True, gradient_checkpointing=gpu_profile["gradient_checkpointing"],
                    gradient_checkpointing_kwargs={"use_reentrant": False})
    if gpu_profile["dataloader_num_workers"] > 0:
        training.update(dataloader_persistent_workers=True, dataloader_prefetch_factor=2)
    eval_steps = int(getattr(args, "eval_steps", 500))
    golden_every_steps = int(getattr(args, "golden_every_steps", 1000))
    if eval_steps <= 0 or golden_every_steps <= 0 or golden_every_steps % eval_steps:
        raise ValueError("--golden-every-steps must be a positive integer multiple of --eval-steps.")
    training.update(eval_steps=eval_steps, save_steps=eval_steps)
    config["golden_eval"].update(interval=golden_every_steps // eval_steps, requested_every_optimizer_steps=golden_every_steps)
    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        training.update(max_steps=args.steps, eval_steps=min(eval_steps, args.steps), save_steps=min(eval_steps, args.steps))
    if args.resume is not None:
        training["resume_from_checkpoint"] = str(args.resume.resolve(strict=True))
    if args.qv_baseline:
        if args.profile != "e2b":
            raise ValueError("--qv-baseline is an E2B LoRA ablation")
        config["lora"].update(r=16, alpha=16, target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q|v)_proj(?:\.linear)?")
    if args.qat:
        if args.profile != "270m":
            raise ValueError("E2B mobile QAT requires the separate retained-scale launcher and verified seed contract.")
        training.update(method="full_finetune_qat", learning_rate=0.000005)
        template = load_yaml(ROOT / "configs/models/gemma3_270m_a2ui_express_qat.yaml")
        config["qat"] = copy.deepcopy(template["qat"])
        config["qat"]["profile"] = "gemma3_270m_wi8_afp32_full_finetune"
    max_new_tokens = int(getattr(args, "max_new_tokens", 2048))
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    config["golden_eval"].update(split_path=str(golden), output_dir=str(run_dir / "golden_eval"), max_input_tokens=args.max_seq_length,
                                 max_new_tokens=max_new_tokens, tensorboard=True)
    config["model"]["max_output_tokens"] = max_new_tokens
    if args.max_seq_length + config["golden_eval"]["max_new_tokens"] > config["model"]["max_context_tokens"]:
        raise ValueError("Prompt + generation budget exceeds model context")
    validate_sft_recipe(config)
    if args.qat:
        from ir_training.qat.workflow import validate_qat_config
        errors = [issue.message for issue in validate_qat_config(config) if issue.severity == "error"]
        if errors:
            raise ValueError("; ".join(errors))
    validate_effective_batch(training, world_size)
    report = verify_prepared(dataset, golden, max_sequence=args.max_seq_length, max_prompt=args.max_seq_length)
    if (report.get("benchmark") or {}).get("benchmark_kind") == "explicit_repeated_case":
        config["golden_eval"]["metric_for_best_model"] = "unique_source_generation_reward_v5_4_avg"
    config["model"]["chat_template_kwargs"] = report["tokenizer"].get("chat_template_kwargs") or {}
    report.update(training_executed=False, model_loaded=False, profile=args.profile, effective_batch=effective_batch, gpu_profile=gpu_profile,
                  model_config_sha256=sha256(model_dir / "config.json"), source_config=profile,
                  model_files={path.name: sha256(path) for path in sorted(model_dir.iterdir()) if path.is_file() and (path.suffix in {".json", ".safetensors", ".model", ".jinja"})})
    return config, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("e2b", "270m"), required=True)
    for name in ("model-dir", "dataset-dir", "golden-file", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--devices", default="auto", help="Use all CUDA-visible GPUs (default), or comma-separated visible logical indices/exact GPU or MIG UUIDs. Scheduler masks are preserved.")
    parser.add_argument("--effective-batch", type=int, help="Global examples per optimizer update; default 32 on H100 >=70 GiB, otherwise 16. Learning rate is unchanged.")
    parser.add_argument("--microbatch", type=int, help="Override per-GPU microbatch; H100 defaults: E2B 2, 270M 4, capped to divide the effective batch.")
    parser.add_argument("--dataloader-workers", type=int, help="Workers per GPU process; default bounds worker count by host CPU count.")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Golden generation budget. Token-limit failures remain scored; this never truncates source records.")
    parser.add_argument("--eval-steps", type=int, default=500, help="Validation-loss and checkpoint cadence in optimizer updates.")
    parser.add_argument("--golden-every-steps", type=int, default=1000, help="Full Golden generation cadence; must be divisible by --eval-steps. Final weights are always evaluated.")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--qv-baseline", action="store_true")
    parser.add_argument("--qat", action="store_true", help="270M full-model W8 QAT; use the selected SFT full checkpoint as --model-dir.")
    args = parser.parse_args()
    config, report = build_config(args)
    import yaml
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "training_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report["training_config_sha256"] = sha256(config_path)
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"config": str(config_path), "training_executed": False, "next": [sys.executable, str(ROOT / "scripts/launch_review_training.py"), "--config", str(config_path)]}, indent=2))


if __name__ == "__main__":
    main()
