#!/usr/bin/env python3
"""Plan, preflight, or explicitly launch one fresh Gemma 4 mobile QAT run.

The launcher is intentionally dry-run by default.  It never resumes or reuses a
run directory and it refuses to execute unless the scale-preserving checkpoint
validator is installed and every declared preflight succeeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml, resolve_path  # noqa: E402
from ir_training.qat.mobile_training_seed import (  # noqa: E402
    OFFICIAL_MOBILE_SAFETENSORS_SHA256,
)


DEFAULT_CONFIG = ROOT / "configs" / "models" / "gemma4_e2b_mobile_seed_ir_qat_sft.yaml"
SCALE_VALIDATOR = ROOT / "scripts" / "validate_gemma4_mobile_scale_preserving_qat.py"
ARCHITECTURE_VALIDATOR = ROOT / "scripts" / "validate_gemma4_mobile_seed_architecture.py"
STATIC_QAT_VALIDATOR = ROOT / "scripts" / "validate_qat_training.py"
TRAIN_ENTRYPOINT = ROOT / "scripts" / "train_sft.py"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class PortableTrainingLaunchError(RuntimeError):
    """Raised when a portable launch cannot proceed without ambiguity."""


def _utc_run_id() -> str:
    return "gemma4_e2b_mobile_qat_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _training_limit_summary(training: dict[str, Any]) -> dict[str, Any]:
    configured = "max_steps" in training
    raw_max_steps = training.get("max_steps") if configured else None
    valid = bool(
        not configured or (type(raw_max_steps) is int and raw_max_steps > 0)
    )
    return {
        "mode": (
            "max_optimizer_steps"
            if configured and valid
            else "num_train_epochs" if not configured else "invalid"
        ),
        "max_optimizer_steps": raw_max_steps if configured and valid else None,
        "configured_max_steps": raw_max_steps,
        "num_train_epochs": training.get("epochs", 2),
        "max_steps_overrides_epochs": configured and valid,
        "valid": valid,
    }


def _resolve_training_path(value: Any) -> Path:
    return resolve_path(str(value), ROOT)


def _resolve_packed_source(
    config: dict[str, Any], override: str | Path | None
) -> tuple[Path | None, str, str | None]:
    if override is not None and str(override).strip():
        path = Path(str(override)).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve(), "command_line_override", None

    model = _section(config, "model")
    manifest_path = _resolve_training_path(
        model.get("mobile_training_seed_manifest", "")
    )
    if not manifest_path.is_file():
        return (
            None,
            "seed_manifest",
            "The mobile seed manifest is unavailable, so its packed source path "
            "cannot be resolved. Supply --source-safetensors.",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest.get("source") if isinstance(manifest, dict) else None
        source_value = source.get("safetensors") if isinstance(source, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, "seed_manifest", f"Could not read packed source from seed manifest: {exc}"
    if source_value is None or not str(source_value).strip():
        return (
            None,
            "seed_manifest",
            "The mobile seed manifest does not declare source.safetensors. "
            "Supply --source-safetensors.",
        )
    path = Path(str(source_value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve(), "seed_manifest", None


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _command_text(command: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(command)) if os.name == "nt" else shlex.join(command)


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _resolve_run_config(
    source_config: dict[str, Any],
    *,
    run_id: str,
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    config = copy.deepcopy(source_config)
    run = config.setdefault("run", {})
    training = config.setdefault("training", {})
    golden = config.setdefault("golden_eval", {})
    if not all(isinstance(value, dict) for value in (run, training, golden)):
        raise PortableTrainingLaunchError(
            "run, training, and golden_eval must be YAML objects."
        )

    training_dir = (run_root / "trainer").resolve()
    tensorboard_dir = (run_root / "tensorboard").resolve()
    golden_output_dir = (run_root / "golden100").resolve()
    best_checkpoint_dir = (run_root / "best_golden_checkpoint").resolve()
    launch_dir = (run_root / "launch").resolve()
    run["launch_plan_path"] = str(launch_dir / "launch_plan.json")
    run["preflight_report_path"] = str(launch_dir / "preflight_report.json")

    run["id"] = run_id
    run["output_dir"] = str(training_dir)
    training["logging_dir"] = str(tensorboard_dir)
    golden["output_dir"] = str(golden_output_dir)
    golden["best_checkpoint_dir"] = str(best_checkpoint_dir)
    return config, {
        "run_root": run_root.resolve(),
        "training_dir": training_dir,
        "tensorboard_dir": tensorboard_dir,
        "golden_output_dir": golden_output_dir,
        "best_checkpoint_dir": best_checkpoint_dir,
        "launch_dir": launch_dir,
        "resolved_config": launch_dir / "resolved_training_config.yaml",
        "launch_plan": launch_dir / "launch_plan.json",
        "preflight_report": launch_dir / "preflight_report.json",
        "model_numeric_preflight_log": launch_dir
        / "preflight"
        / "model_numeric_preflight.log",
        "training_log": launch_dir / "training.log",
    }


def _validate_launch_contract(
    config: dict[str, Any],
    *,
    source_config_path: Path,
    paths: dict[str, Path],
    num_gpus: int,
    packed_source_path: Path | None,
    packed_source_error: str | None,
    packed_source_identity: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    model = _section(config, "model")
    training = _section(config, "training")
    qat = _section(config, "qat")
    golden = _section(config, "golden_eval")
    run = _section(config, "run")

    def require(code: str, condition: bool, message: str) -> None:
        if not condition:
            issues.append({"code": code, "message": message})

    require(
        "wrong_model_identity",
        str(model.get("model_id") or "")
        == "google/gemma-4-E2B-it-qat-mobile-transformers",
        "The launcher accepts only the official packed-mobile Gemma 4 E2B identity.",
    )
    require(
        "missing_local_model_source",
        bool(str(model.get("model_source") or "").strip()),
        "model.model_source must name the materialized BF16 mobile reconstruction.",
    )
    require(
        "missing_mobile_seed_manifest",
        bool(str(model.get("mobile_training_seed_manifest") or "").strip()),
        "model.mobile_training_seed_manifest is required.",
    )
    require(
        "missing_mobile_qparams_contract",
        bool(str(model.get("mobile_qparams_contract") or "").strip()),
        "model.mobile_qparams_contract must bind the retained codes/scales.",
    )
    require(
        "qat_disabled",
        qat.get("enabled") is True,
        "qat.enabled must remain true.",
    )
    require(
        "wrong_scale_mode",
        str(qat.get("scale_mode") or "") == "retained_mobile",
        "qat.scale_mode must be retained_mobile; abs-max scale recomputation is unsafe.",
    )
    require(
        "fixed_scale_not_required",
        qat.get("fixed_scale_required") is True,
        "qat.fixed_scale_required must remain true.",
    )
    require(
        "fixed_activation_scale_not_required",
        qat.get("fixed_activation_scale_required") is True,
        "qat.fixed_activation_scale_required must remain true.",
    )
    require(
        "effective_lora_only_not_required",
        qat.get("effective_lora_only") is True,
        "qat.effective_lora_only must remain true.",
    )
    require(
        "wrong_effective_lora_scope",
        int(qat.get("expected_effective_lora_modules", 0) or 0) == 205,
        "qat.expected_effective_lora_modules must remain exactly 205.",
    )
    require(
        "wrong_ste_gradient",
        str(qat.get("ste_gradient") or "") == "clipped",
        "qat.ste_gradient must be clipped at the representable retained-scale range.",
    )
    preflight = _section(config, "preflight")
    require(
        "zero_adapter_parity_not_required",
        preflight.get("require_zero_adapter_parity") is True,
        "preflight.require_zero_adapter_parity must remain true.",
    )
    require(
        "initial_loss_gate_not_required",
        preflight.get("require_initial_loss_gate") is True,
        "preflight.require_initial_loss_gate must remain true.",
    )
    try:
        min_top1_probe_match = float(
            preflight.get("min_top1_probe_match", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        min_top1_probe_match = 0.0
    require(
        "weak_top1_numeric_gate",
        min_top1_probe_match >= 0.90,
        "preflight.min_top1_probe_match must be at least 0.90.",
    )
    try:
        greedy_rows = int(preflight.get("greedy_probe_rows", 0) or 0)
        greedy_new_tokens = int(
            preflight.get("greedy_probe_new_tokens", 0) or 0
        )
        minimum_greedy_tokens = int(
            preflight.get("min_greedy_tokens", 0) or 0
        )
        minimum_greedy_prefix = int(
            preflight.get("min_baseline_qat_greedy_prefix_tokens", 0) or 0
        )
    except (TypeError, ValueError):
        greedy_rows = greedy_new_tokens = minimum_greedy_tokens = 0
        minimum_greedy_prefix = 0
    require(
        "missing_greedy_generation_gate",
        preflight.get("require_greedy_determinism") is True
        and greedy_rows >= 1
        and greedy_new_tokens >= 8
        and minimum_greedy_tokens >= 8
        and minimum_greedy_tokens <= greedy_new_tokens
        and minimum_greedy_prefix >= 8
        and minimum_greedy_prefix <= greedy_new_tokens,
        "A repeated nontrivial zero-adapter greedy-generation gate is required.",
    )
    require(
        "wrong_training_method",
        str(training.get("method") or "") == "qat_lora_sft",
        "training.method must be qat_lora_sft.",
    )
    require(
        "wrong_training_dtype",
        str(model.get("dtype") or "").strip().lower() in {"bf16", "bfloat16"},
        "The retained mobile run must request BF16; the environment gate forbids silent fallback.",
    )
    require(
        "bnb_4bit_enabled",
        model.get("load_in_4bit") is False,
        "BitsAndBytes/NF4 loading is not retained-scale mobile QAT.",
    )
    require(
        "resume_not_refused",
        training.get("refuse_resume") is True,
        "training.refuse_resume must remain true for this fresh-run-only workflow.",
    )
    require(
        "nonzero_lora_dropout",
        float(_section(config, "lora").get("dropout", -1.0)) == 0.0,
        "Effective-weight QAT requires lora.dropout=0.0.",
    )
    require(
        "missing_ddp_requirement",
        training.get("require_ddp_for_multi_gpu") is True,
        "training.require_ddp_for_multi_gpu must remain true.",
    )
    require(
        "golden100_not_strict",
        golden.get("enabled") is True
        and int(golden.get("required_rows", 0) or 0) == 100
        and golden.get("require_exact_rows") is True
        and golden.get("require_unique_rows") is True,
        "Golden evaluation must require exactly 100 unique held-out rows.",
    )
    require(
        "wrong_best_metric",
        str(golden.get("metric_for_best_model") or "")
        == "generation_reward_v5_4_avg",
        "Best-checkpoint selection must use generation_reward_v5_4_avg.",
    )
    require(
        "golden_not_every_eval",
        str(golden.get("trigger") or "") == "evaluate"
        and int(golden.get("interval", 0) or 0) == 1,
        "Golden-100 v5.4 must run at every Trainer evaluation event.",
    )
    try:
        eval_steps = int(training.get("eval_steps", 0) or 0)
        save_steps = int(training.get("save_steps", 0) or 0)
    except (TypeError, ValueError):
        eval_steps = save_steps = 0
    training_limit = _training_limit_summary(training)
    require(
        "invalid_max_steps",
        training_limit["valid"],
        "When configured, training.max_steps must be a positive integer "
        "optimizer-step limit; booleans, zero, negative, fractional, and "
        "string values are rejected.",
    )
    require(
        "checkpoint_eval_cadence_mismatch",
        eval_steps > 0 and save_steps == eval_steps,
        "training.save_steps must equal positive training.eval_steps so each "
        "new Golden best receives self-bound checkpoint provenance immediately.",
    )
    max_optimizer_steps = training_limit["max_optimizer_steps"]
    require(
        "bounded_run_cadence_exceeds_max_steps",
        max_optimizer_steps is None
        or (eval_steps <= max_optimizer_steps and save_steps <= max_optimizer_steps),
        "For a bounded training.max_steps run, training.eval_steps and "
        "training.save_steps must not exceed the optimizer-step limit.",
    )
    require(
        "invalid_gpu_count",
        num_gpus >= 1,
        "--num-gpus must be at least one.",
    )
    require(
        "packed_source_not_materialized",
        packed_source_error is None
        and packed_source_path is not None
        and packed_source_path.is_file(),
        packed_source_error
        or "The exact official packed model.safetensors is unavailable. "
        "Supply its local path with --source-safetensors.",
    )
    require(
        "packed_source_identity_mismatch",
        packed_source_identity is not None
        and packed_source_identity.get("sha256")
        == OFFICIAL_MOBILE_SAFETENSORS_SHA256,
        "The selected packed model.safetensors does not match the pinned official "
        f"SHA-256 {OFFICIAL_MOBILE_SAFETENSORS_SHA256}.",
    )
    require(
        "source_config_inside_output",
        not source_config_path.is_relative_to(paths["run_root"]),
        "The immutable source config must not live inside this run's output directory.",
    )
    require(
        "run_root_already_exists",
        not paths["run_root"].exists(),
        "The selected run ID already exists; this launcher never resumes or reuses it.",
    )
    require(
        "output_path_mismatch",
        str(run.get("output_dir")) == str(paths["training_dir"])
        and str(training.get("logging_dir")) == str(paths["tensorboard_dir"])
        and str(golden.get("output_dir")) == str(paths["golden_output_dir"])
        and str(golden.get("best_checkpoint_dir"))
        == str(paths["best_checkpoint_dir"]),
        "All mutable outputs must be isolated under the unique run root.",
    )
    issues.extend(_artifact_contract_issues(config))
    return issues


def _artifact_contract_issues(config: dict[str, Any]) -> list[dict[str, Any]]:
    model = _section(config, "model")
    run = _section(config, "run")
    golden = _section(config, "golden_eval")
    issues: list[dict[str, Any]] = []

    def missing(code: str, message: str) -> None:
        issues.append({"code": code, "message": message})

    model_source = _resolve_training_path(model.get("model_source", ""))
    seed_manifest = _resolve_training_path(
        model.get("mobile_training_seed_manifest", "")
    )
    qparams = _resolve_training_path(model.get("mobile_qparams_contract", ""))
    dataset_dir = _resolve_training_path(run.get("dataset_dir", ""))
    golden_dir = _resolve_training_path(golden.get("dataset_dir", ""))

    if not model_source.is_dir():
        missing(
            "mobile_model_source_not_materialized",
            f"Materialized BF16 mobile model source is missing: {model_source}",
        )
    if not seed_manifest.is_file():
        missing(
            "mobile_seed_manifest_not_materialized",
            f"Mobile seed manifest is missing: {seed_manifest}",
        )
    if not qparams.is_file():
        missing(
            "mobile_qparams_not_materialized",
            f"Retained mobile qparams contract is missing: {qparams}",
        )
    training_identities: set[str] = set()
    split_identity_sets: dict[str, set[str]] = {}
    for name in ("train.jsonl", "val.jsonl"):
        split_path = dataset_dir / name
        if not split_path.is_file() or split_path.stat().st_size <= 0:
            missing(
                f"training_{name.removesuffix('.jsonl')}_missing",
                f"Prepared training split is missing or empty: {split_path}",
            )
            continue
        try:
            identities = set(_jsonl_identities(split_path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            missing(
                f"training_{name.removesuffix('.jsonl')}_unreadable",
                f"Could not validate prepared split {split_path}: {exc}",
            )
            continue
        split_identity_sets[name] = identities
        training_identities.update(identities)
    if split_identity_sets.get("train.jsonl", set()) & split_identity_sets.get(
        "val.jsonl", set()
    ):
        missing(
            "training_validation_identity_overlap",
            "Prepared train and validation splits share source identities.",
        )

    golden_path = golden_dir / f"{str(golden.get('split', 'all')).strip() or 'all'}.jsonl"
    if not golden_path.is_file():
        missing(
            "golden100_missing",
            f"Immutable Golden-100 split is missing: {golden_path}",
        )
        return issues
    try:
        identities = _jsonl_identities(golden_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        missing(
            "golden100_unreadable",
            f"Could not validate Golden-100 split {golden_path}: {exc}",
        )
        return issues
    if len(identities) != 100:
        missing(
            "golden100_wrong_count",
            f"Golden split must contain exactly 100 non-empty JSON rows; observed {len(identities)}.",
        )
    if len(set(identities)) != len(identities):
        missing(
            "golden100_duplicate_identity",
            "Golden split contains duplicate source/response/row identities.",
        )
    overlap = training_identities & set(identities)
    if overlap:
        missing(
            "golden100_training_identity_overlap",
            f"Golden-100 shares {len(overlap)} source identities with train/validation.",
        )
    return issues


def _jsonl_identities(path: Path) -> list[str]:
    identities: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {index + 1} is not a JSON object")
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            identity = next(
                (
                    str(value).strip()
                    for value in (
                        row.get("source_id"),
                        row.get("response_id"),
                        row.get("id"),
                        metadata.get("query_id"),
                    )
                    if value is not None and str(value).strip()
                ),
                f"row-index:{index}",
            )
            identities.append(identity)
    return identities


def _bound_artifacts(
    config: dict[str, Any],
    *,
    packed_source_identity: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    model = _section(config, "model")
    run = _section(config, "run")
    golden = _section(config, "golden_eval")
    dataset_dir = _resolve_training_path(run.get("dataset_dir", ""))
    golden_dir = _resolve_training_path(golden.get("dataset_dir", ""))
    golden_name = f"{str(golden.get('split', 'all')).strip() or 'all'}.jsonl"
    candidates = {
        "mobile_seed_manifest": _resolve_training_path(
            model.get("mobile_training_seed_manifest", "")
        ),
        "mobile_qparams_contract": _resolve_training_path(
            model.get("mobile_qparams_contract", "")
        ),
        "training_train": dataset_dir / "train.jsonl",
        "training_val": dataset_dir / "val.jsonl",
        "golden100": golden_dir / golden_name,
    }
    result: dict[str, dict[str, Any]] = {}
    for role, path in candidates.items():
        if path.is_file():
            result[role] = _file_identity(path)
    if packed_source_identity is not None:
        result["official_packed_source"] = dict(packed_source_identity)
    return result


def _verify_bound_artifacts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for role, expected in dict(plan.get("bound_artifacts") or {}).items():
        path = Path(str(expected.get("path") or ""))
        if not path.is_file():
            issues.append({"role": role, "code": "missing", "path": str(path)})
            continue
        observed_size = int(path.stat().st_size)
        observed_sha256 = _sha256_file(path)
        if (
            observed_size != int(expected.get("size_bytes", -1))
            or observed_sha256 != str(expected.get("sha256") or "")
        ):
            issues.append(
                {
                    "role": role,
                    "code": "identity_changed_after_plan",
                    "path": str(path),
                    "expected_size_bytes": expected.get("size_bytes"),
                    "observed_size_bytes": observed_size,
                    "expected_sha256": expected.get("sha256"),
                    "observed_sha256": observed_sha256,
                }
            )
    return issues


def _preflight_commands(
    *,
    resolved_config: Path,
    preflight_dir: Path,
    num_gpus: int,
    packed_source_path: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "id": "cuda_bf16_environment",
            "command": [
                sys.executable,
                str(Path(__file__).resolve()),
                "--environment-check",
                "--num-gpus",
                str(num_gpus),
            ],
        },
        {
            "id": "static_qat_profile",
            "command": [
                sys.executable,
                str(STATIC_QAT_VALIDATOR),
                "--config",
                str(resolved_config),
            ],
        },
        {
            "id": "mobile_seed_architecture",
            "command": [
                sys.executable,
                str(ARCHITECTURE_VALIDATOR),
                "--training-config",
                str(resolved_config),
                "--output",
                str(preflight_dir / "mobile_seed_architecture.json"),
            ],
        },
        {
            "id": "scale_preserving_qat",
            "command": [
                sys.executable,
                str(SCALE_VALIDATOR),
                "--config",
                str(resolved_config),
                "--source-safetensors",
                str(packed_source_path),
                "--strict",
                "--output",
                str(preflight_dir / "scale_preserving_qat.json"),
            ],
        },
        {
            "id": "model_numeric_preflight",
            "command": [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={int(num_gpus)}",
                str(TRAIN_ENTRYPOINT),
                "--config",
                str(resolved_config),
                "--preflight-only",
            ],
        },
    ]


def build_launch_plan(
    source_config_path: str | Path,
    *,
    run_id: str,
    runs_root: str | Path,
    num_gpus: int,
    source_safetensors: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    source_path = Path(source_config_path).expanduser().resolve()
    if not source_path.is_file():
        raise PortableTrainingLaunchError(f"Training config does not exist: {source_path}")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise PortableTrainingLaunchError(
            "--run-id must use only letters, digits, '.', '_' or '-' and be at most 96 characters."
        )
    if int(num_gpus) < 1:
        raise PortableTrainingLaunchError("--num-gpus must be at least one.")

    root = Path(runs_root).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    run_root = (root / run_id).resolve()
    if run_root.parent != root.resolve():
        raise PortableTrainingLaunchError("Resolved run root escaped --runs-root.")

    source_config = load_yaml(source_path)
    packed_source_path, packed_source_origin, packed_source_error = (
        _resolve_packed_source(source_config, source_safetensors)
    )
    packed_source_identity = (
        _file_identity(packed_source_path)
        if packed_source_path is not None and packed_source_path.is_file()
        else None
    )
    resolved_config, paths = _resolve_run_config(
        source_config, run_id=run_id, run_root=run_root
    )
    issues = _validate_launch_contract(
        resolved_config,
        source_config_path=source_path,
        paths=paths,
        num_gpus=int(num_gpus),
        packed_source_path=packed_source_path,
        packed_source_error=packed_source_error,
        packed_source_identity=packed_source_identity,
    )
    preflight_dir = paths["launch_dir"] / "preflight"
    preflights = _preflight_commands(
        resolved_config=paths["resolved_config"],
        preflight_dir=preflight_dir,
        num_gpus=int(num_gpus),
        packed_source_path=(
            packed_source_path
            if packed_source_path is not None
            else Path("MISSING_OFFICIAL_PACKED_SOURCE")
        ),
    )
    train_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(num_gpus)}",
        str(TRAIN_ENTRYPOINT),
        "--config",
        str(paths["resolved_config"]),
    ]
    plan = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "fresh_run_only_no_resume",
        "run_id": run_id,
        "run_intent": {
            "purpose": _section(resolved_config, "run").get("purpose", "training"),
            "quality_promotion_eligible": _section(resolved_config, "run").get(
                "quality_promotion_eligible", True
            ),
            "serialization_validation_eligible": _section(
                resolved_config, "run"
            ).get(
                "serialization_validation_eligible", True
            ),
        },
        "source_config": str(source_path),
        "source_config_sha256": _sha256_file(source_path),
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "tracked_worktree_dirty": _git_dirty(),
        },
        "num_gpus": int(num_gpus),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_limit": _training_limit_summary(
            _section(resolved_config, "training")
        ),
        "packed_source": {
            "origin": packed_source_origin,
            "path": str(packed_source_path) if packed_source_path else None,
            "error": packed_source_error,
        },
        "bound_artifacts": _bound_artifacts(
            resolved_config,
            packed_source_identity=packed_source_identity,
        ),
        "paths": {name: str(path) for name, path in paths.items()},
        "checks": {
            "no_resume_supported": True,
            "run_root_must_not_exist": True,
            "scale_validator_required": True,
            "scale_validator_present": SCALE_VALIDATOR.is_file(),
            "contract_ok": not issues,
            "issues": issues,
        },
        "preflights": preflights,
        "training_command": train_command,
        "tensorboard_command": [
            "tensorboard",
            "--logdir",
            str(paths["tensorboard_dir"]),
        ],
    }
    return plan, resolved_config, paths


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(value).split("+", 1)[0])
    return tuple(int(part) for part in parts[:3])


def _check_training_environment(num_gpus: int) -> dict[str, Any]:
    try:
        import peft
        import torch
        import transformers
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Could not import torch/transformers/peft: {exc}",
        }

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    devices: list[dict[str, Any]] = []
    for index in range(count):
        try:
            with torch.cuda.device(index):
                bf16 = bool(torch.cuda.is_bf16_supported())
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "bf16_supported": bf16,
                }
            )
        except Exception as exc:
            devices.append(
                {
                    "index": index,
                    "bf16_supported": False,
                    "error": str(exc),
                }
            )
    selected = devices[: int(num_gpus)]
    checks = {
        "cuda_available": available,
        "cuda_build_present": bool(getattr(torch.version, "cuda", None)),
        "enough_visible_gpus": count >= int(num_gpus),
        "selected_gpus_support_bf16": len(selected) == int(num_gpus)
        and all(item.get("bf16_supported") is True for item in selected),
        "transformers_minimum": _version_tuple(transformers.__version__)
        >= (5, 10, 1),
        "peft_minimum": _version_tuple(peft.__version__) >= (0, 19, 0),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_count": count,
        "requested_gpu_count": int(num_gpus),
        "devices": devices,
    }


def _normalize_gpu_ids(value: str | None, *, num_gpus: int) -> str | None:
    if value is None or not value.strip():
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    if any(not item.isdigit() for item in values) or len(set(values)) != len(values):
        raise PortableTrainingLaunchError(
            "--gpu-ids must be a comma-separated list of unique nonnegative integers."
        )
    if len(values) != int(num_gpus):
        raise PortableTrainingLaunchError(
            f"--gpu-ids contains {len(values)} devices but --num-gpus is {num_gpus}."
        )
    return ",".join(values)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency error path
        raise PortableTrainingLaunchError(
            "PyYAML is required; install training/requirements-gemma4-qat.txt."
        ) from exc
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run_live(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {_command_text(command)}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("Command:\n" + _command_text(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return int(process.wait())


def _declared_output_paths(command: list[str]) -> list[Path]:
    outputs: list[Path] = []
    for index, value in enumerate(command):
        if value == "--output" and index + 1 < len(command):
            outputs.append(Path(command[index + 1]).expanduser().resolve())
        elif value.startswith("--output="):
            outputs.append(Path(value.split("=", 1)[1]).expanduser().resolve())
    return outputs


def _run_preflights(
    plan: dict[str, Any], *, paths: dict[str, Path]
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for item in plan["preflights"]:
        command = [str(value) for value in item["command"]]
        log_path = paths["launch_dir"] / "preflight" / f"{item['id']}.log"
        exit_code = _run_live(command, cwd=REPO_ROOT, log_path=log_path)
        declared_outputs = _declared_output_paths(command)
        output_identities = [
            (
                {"present": True, **_file_identity(output_path)}
                if output_path.is_file()
                else {"present": False, "path": str(output_path)}
            )
            for output_path in declared_outputs
        ]
        outputs_present = all(
            identity.get("present") is True for identity in output_identities
        )
        passed = exit_code == 0 and outputs_present
        report = {
            "id": item["id"],
            "command": command,
            "log": {"present": True, **_file_identity(log_path)},
            "declared_outputs": output_identities,
            "exit_code": exit_code,
            "passed": passed,
        }
        reports.append(report)
        if not passed:
            break
    result = {
        "schema_version": 1,
        "run_id": plan["run_id"],
        "all_passed": len(reports) == len(plan["preflights"])
        and all(item["passed"] for item in reports),
        "reports": reports,
    }
    paths["preflight_report"].write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def _reserve_run(
    plan: dict[str, Any], resolved_config: dict[str, Any], paths: dict[str, Path]
) -> None:
    run_root = paths["run_root"]
    if run_root.exists():
        raise PortableTrainingLaunchError(
            f"Run directory already exists; resume/reuse is intentionally unsupported: {run_root}"
        )
    # Never clean a partially reserved run automatically. It is safer to keep
    # diagnostic evidence and require a new run ID than to make this launcher a
    # recursive-delete path on an unfamiliar training host.
    paths["launch_dir"].mkdir(parents=True, exist_ok=False)
    _write_yaml(paths["resolved_config"], resolved_config)
    resolved_config_path = paths["resolved_config"]
    plan.setdefault("bound_artifacts", {})["resolved_training_config"] = {
        "path": str(resolved_config_path.resolve()),
        "size_bytes": int(resolved_config_path.stat().st_size),
        "sha256": _sha256_file(resolved_config_path),
    }
    paths["launch_plan"].write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default=_utc_run_id())
    parser.add_argument(
        "--runs-root", default="training/runs/portable_gemma4_mobile_qat"
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--source-safetensors",
        help=(
            "Local exact official packed model.safetensors. Optional only while "
            "the source path recorded in the seed manifest remains valid."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        help="Optional physical CUDA indices, for example 0,1,3; count must equal --num-gpus.",
    )
    parser.add_argument(
        "--environment-check",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Reserve a fresh run directory and execute gates, but do not train.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Reserve a fresh run, run all gates, then launch torchrun.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        selected_gpu_ids = _normalize_gpu_ids(
            args.gpu_ids, num_gpus=args.num_gpus
        )
        if selected_gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu_ids
        if args.environment_check:
            report = _check_training_environment(args.num_gpus)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["ok"] else 2
        plan, resolved_config, paths = build_launch_plan(
            args.config,
            run_id=args.run_id,
            runs_root=args.runs_root,
            num_gpus=args.num_gpus,
            source_safetensors=args.source_safetensors,
        )
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        if not args.preflight and not args.execute:
            print(
                "Plan only: no output directory was created, no model was loaded, "
                "and no training was run."
            )
            return 0 if plan["checks"]["contract_ok"] else 2
        if not plan["checks"]["contract_ok"]:
            raise PortableTrainingLaunchError(
                "Launch contract failed: "
                + ", ".join(
                    str(item["code"]) for item in plan["checks"]["issues"]
                )
            )
        if not SCALE_VALIDATOR.is_file():
            raise PortableTrainingLaunchError(
                "Scale-preserving QAT validator is missing. Refusing to execute the "
                "old abs-max QAT path: "
                + str(SCALE_VALIDATOR)
            )

        _reserve_run(plan, resolved_config, paths)
        preflight = _run_preflights(plan, paths=paths)
        if not preflight["all_passed"]:
            raise PortableTrainingLaunchError(
                f"Preflight failed; inspect {paths['preflight_report']}"
            )
        if args.preflight:
            print("All preflights passed. Training was not started.")
            return 0

        artifact_changes = _verify_bound_artifacts(plan)
        if artifact_changes:
            raise PortableTrainingLaunchError(
                "A bound launch input changed after planning/preflight: "
                + json.dumps(artifact_changes, ensure_ascii=False)
            )

        exit_code = _run_live(
            [str(value) for value in plan["training_command"]],
            cwd=REPO_ROOT,
            log_path=paths["training_log"],
        )
        if exit_code != 0:
            raise PortableTrainingLaunchError(
                f"Training exited with code {exit_code}; inspect {paths['training_log']}"
            )
        print(
            "Training command completed. This does not promote, merge, quantize, "
            "export, or deploy the checkpoint."
        )
        return 0
    except (OSError, ValueError, PortableTrainingLaunchError) as exc:
        print(f"Gemma 4 portable training launch failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
