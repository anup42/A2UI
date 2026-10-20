"""Read-only, fail-closed context for mining offline GRPO training preferences.

This does not load model weights, generate candidates, or adopt evaluation
outputs. Historical rewards are not trusted: the caller rescores raw training
completions with the current source-fidelity admission policy.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, resolve_path, training_root

_WORKFLOW = "qat_lora_grpo_v1"
_AUDIT_NAME = re.compile(r"grpo_periodic_rewards\.rank(0|[1-9][0-9]*)\.jsonl\Z")
_UNBOUND = object()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")  # noqa: TRY004
    return result


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _exact_path(value: Any, expected: Path, name: str) -> Path:
    if not isinstance(value, str) or not value or resolve_path(value, training_root()).resolve() != expected:
        raise ValueError(f"{name} must bind this run's exact {expected}")
    return expected


def _load_tokenizer(path: Path, config: dict):
    """Mirror ModelAdapter's configured tokenizer path without network access.

    E2B's reconstructed text seed deliberately uses PreTrainedTokenizerFast,
    while 270M normally uses AutoTokenizer. Keep configured processor fallback
    and padding semantics, but never enable remote code or Hub downloads.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    if config.get("trust_remote_code"):
        raise ValueError("Offline preference preparation does not permit remote tokenizer code")
    kwargs = {"local_files_only": True, "trust_remote_code": False}

    def processor_tokenizer():
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(path), **kwargs)
        return getattr(processor, "tokenizer", processor)

    loader = str(config.get("tokenizer_loader", "auto_tokenizer")).strip().lower()
    if loader in {"auto_processor", "processor"}:
        tokenizer = processor_tokenizer()
    else:
        try:
            if loader in {"pretrained_tokenizer_fast", "tokenizer_fast", "fast"}:
                tokenizer = PreTrainedTokenizerFast.from_pretrained(str(path), **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(str(path),
                    use_fast=bool(config.get("use_fast_tokenizer", True)), **kwargs)
        except Exception:
            if not bool(config.get("processor_fallback", False)):
                raise
            tokenizer = processor_tokenizer()
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    padding = str(config.get("padding_side", "right")).strip().lower()
    if padding in {"left", "right"}:
        tokenizer.padding_side = padding
    return tokenizer


def _load_training_rows(path: Path, tokenizer: Any, model_config: dict):
    from ir_training.train.qat_grpo import _script_module
    return _script_module("train_grpo").load_training_dataset(
        str(path), None, tokenizer=tokenizer, prompt_source="prepared",
        chat_template_kwargs=model_config.get("chat_template_kwargs") or {})


def load_preference_run(run_dir: Path) -> dict[str, Any]:
    """Load only the sealed train split and periodic train/eval audit files.

    The caller must filter audit records to ``mode=train`` and verify their
    source IDs and exact rendered prompt hashes. Audit paths are intentionally
    nonrecursive: runtime probes, Golden/Bixby outputs and arbitrary JSONL
    files can never be selected through this entry point.
    """
    root = Path(run_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("GRPO run must be an existing directory")
    plan_path = root / "qat_grpo_plan.json"
    config_path = root / "configs/grpo_training.yaml"
    report_path = root / "configs/preparation_report.json"
    receipt_path = root / "stage_receipts/configure.json"
    reward_path = root / "configs/grpo_reward.yaml"
    profile_path = root / "configs/gpu_profile.json"
    prepared = root / "prepared"
    output = root / "training"
    for directory in (root / "configs", root / "stage_receipts", prepared, output):
        if not directory.is_dir() or directory.resolve() != directory:
            raise ValueError(f"GRPO input directories must not redirect through links: {directory}")

    files: dict[str, str] = {}

    def bind(path: Path, expected: Any = _UNBOUND) -> str:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required GRPO evidence is not a regular file: {path}")
        digest = _sha(path)
        if expected is not _UNBOUND and (not isinstance(expected, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected) is None or digest != expected):
            raise ValueError(f"GRPO evidence changed after preparation: {path.name}")
        files[str(path.resolve())] = digest
        return digest

    plan_hash = bind(plan_path)
    plan = _object(plan_path)
    options, paths = plan.get("options") or {}, plan.get("paths") or {}
    if plan.get("schema_version") != 1 or plan.get("workflow") != _WORKFLOW:
        raise ValueError("Preference mining requires an official QAT LoRA GRPO run")
    _exact_path(options.get("output_dir"), root, "plan output")
    for key, expected in (("config", config_path), ("prepared", prepared),
                          ("reward_config", reward_path), ("gpu_profile", profile_path)):
        _exact_path(paths.get(key), expected, f"plan {key}")
    family = options.get("family")
    if family not in {"e2b", "270m"}:
        raise ValueError("Preference mining supports only E2B and 270M")

    bind(receipt_path)
    receipt = _object(receipt_path)
    sealed = receipt.get("files")
    if (receipt.get("stage") != "configure" or receipt.get("plan_sha256") != plan_hash
            or not isinstance(sealed, dict)):
        raise ValueError("Configure receipt does not bind the original GRPO plan")
    for path in (config_path, report_path, reward_path, profile_path):
        expected = sealed.get(str(path))
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"Configure receipt does not seal {path.name}")
        bind(path, expected)
    config, report = load_yaml(config_path), _object(report_path)
    if report.get("training_config_sha256") != files[str(config_path)]:
        raise ValueError("Training config differs from its preparation report")
    grpo, training, run, model = (config.get(name) or {} for name in ("grpo", "training", "run", "model"))
    mode = grpo.get("objective", "grpo")
    if (run.get("purpose") != _WORKFLOW or training.get("method") != "qat_lora_grpo"
            or mode not in {"grpo", "grpo_sft"} or mode != options.get("objective", "grpo")
            or grpo.get("family") != family):
        raise ValueError("Config must match the E2B/270M GRPO or GRPO rehearsal plan")
    from ir_training.train.qat_grpo_contract import validate_qat_grpo_config
    validate_qat_grpo_config(config)
    _exact_path(run.get("dataset_dir"), prepared, "prepared training data")
    _exact_path(run.get("output_dir"), output, "training output")
    _exact_path(grpo.get("reward_config"), reward_path, "reward config")

    limits = {"max_prompt_length": _positive(grpo.get("max_prompt_length"), "prompt budget"),
              "max_completion_length": _positive(grpo.get("max_completion_length"), "completion budget"),
              "max_sequence_length": _positive(training.get("max_seq_length"), "sequence budget")}
    profile = _object(profile_path)
    effective_batch = _positive(profile.get("effective_batch_size"), "effective batch")
    actual_batch = (_positive(profile.get("world_size"), "world size")
                    * _positive(training.get("per_device_train_batch_size"), "microbatch")
                    * _positive(training.get("gradient_accumulation_steps"), "accumulation"))
    if (effective_batch != actual_batch or training.get("expected_effective_batch_size") != effective_batch
            or options.get("effective_batch") != effective_batch):
        raise ValueError("Saved effective batch does not match the original GRPO configuration")

    final = config.get("final_evaluation_datasets")
    if not isinstance(final, dict) or final != report.get("final_evaluation_datasets"):
        raise ValueError("All reserved evaluation cohorts must match the preparation report")
    for name, role in (("golden32", "development_checkpoint_selection"),
                       ("golden35", "final_only_holdout"), ("bixby50", "final_only_holdout")):
        entry = final.get(name)
        if not isinstance(entry, dict) or entry.get("selection_role") != role:
            raise ValueError(f"Required reserved cohort is missing or has an invalid role: {name}")
        split = prepared / f"{name}.jsonl"
        _exact_path(entry.get("split_path"), split, name)
        bind(split, entry.get("split_sha256"))
    _exact_path((config.get("golden_eval") or {}).get("split_path"),
                prepared / "golden32.jsonl", "Golden selection")
    from ir_training.train.qat_grpo import verify_prepared_training_contract
    prepared_contract = verify_prepared_training_contract(config)
    if (prepared_contract.get("dataset_manifest_sha256") != report.get("dataset_manifest_sha256")
            or prepared_contract.get("final_evaluation_datasets") != final):
        raise ValueError("Prepared train/holdout contract differs from the sealed run")
    manifest_path = prepared / "manifest.json"
    bind(manifest_path, report.get("dataset_manifest_sha256"))
    manifest = _object(manifest_path)
    for name in ("train", "val"):
        bind(prepared / f"{name}.jsonl", (manifest.get("splits") or {}).get(name, {}).get("output_sha256"))
    from ir_training.data.audit_filter import load_reserved_cohorts
    reserved = load_reserved_cohorts([prepared / f"{name}.jsonl" for name in ("golden32", "golden35", "bixby50")])
    for record in reserved["evidence"]:
        bind(Path(record["path"]), record["sha256"])
    # Freeze small prompt artifacts too, including every file already sealed by
    # the checked manifest. The training contract above validates their meaning.
    for name in ("prompt_scaffolds", "shared_prompt", "inference_prompt", "source_prompt_scaffolds"):
        if f"{name}_sha256" in manifest:
            bind(prepared / f"{name}.json", manifest[f"{name}_sha256"])

    tokenizer_source = resolve_path(model.get("tokenizer_source") or model.get("model_source") or "", training_root()).resolve(strict=True)
    if (not tokenizer_source.is_dir() or str(tokenizer_source) != options.get("model_dir")
            or resolve_path(model.get("model_source") or "", training_root()).resolve() != tokenizer_source):
        raise ValueError("Tokenizer must be the original local base bound by this run")
    model_files = report.get("model_files")
    if not isinstance(model_files, dict) or "tokenizer_config.json" not in model_files:
        raise ValueError("Preparation report lacks the original local tokenizer binding")
    tokenizer_files = {}
    for name, digest in model_files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid original model/tokenizer filename")
        if Path(name).suffix in {".json", ".model", ".jinja"}:
            tokenizer_files[name] = bind(tokenizer_source / name, digest)
    if "config.json" not in tokenizer_files:
        raise ValueError("Preparation report lacks the original local model configuration")
    from ir_training.train.sft import _collect_position_limit_candidates
    position_limits: list[int] = []
    _collect_position_limit_candidates(_object(tokenizer_source / "config.json"), position_limits)
    context = min(position_limits) if position_limits else model.get("max_context_tokens")
    if context is not None and limits["max_sequence_length"] > _positive(context, "model context"):
        raise ValueError("DPO supervised sequence budget exceeds the original model context")
    tokenizer = _load_tokenizer(tokenizer_source, model)
    from ir_training.train.prepared_binding import verify_tokenizer_binding
    tokenizer_binding = verify_tokenizer_binding(prepared, tokenizer, model, required=True)
    train_rows = _load_training_rows(prepared / "train.jsonl", tokenizer, model)
    forbidden_rows = _load_training_rows(prepared / "val.jsonl", tokenizer, model)

    audit_paths = sorted(output.glob("grpo_periodic_rewards.rank*.jsonl"))
    if not audit_paths:
        raise ValueError("No periodic GRPO reward audits were saved; evaluation outputs cannot substitute")
    for path in audit_paths:
        if _AUDIT_NAME.fullmatch(path.name) is None or path.resolve().parent != output:
            raise ValueError(f"Invalid periodic audit path: {path}")
        bind(path)
    for path, expected in files.items():
        if _sha(Path(path)) != expected:
            raise ValueError(f"GRPO input changed while loading preference context: {path}")
    return {"train_rows": train_rows, "forbidden_rows": forbidden_rows, "tokenizer": tokenizer,
            "limits": limits, "reward_config": reward_path, "audit_paths": audit_paths,
            "effective_batch": effective_batch, "family": family,
            "evidence": {"schema_version": 1, "source_run": str(root), "family": family,
                "objective": mode, "files": files, "tokenizer_files": tokenizer_files,
                "prepared_contract": prepared_contract, "tokenizer_binding": tokenizer_binding,
                "limits": limits, "effective_batch": effective_batch,
                "candidate_scope": "periodic GRPO audits only; train mode required by pair admission",
                "model_weights_loaded": False}}
