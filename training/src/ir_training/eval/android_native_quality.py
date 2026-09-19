"""Contracts and Android runner for official-mobile native quality evaluation.

This module is intentionally post-export only. It neither selects a checkpoint
nor loads model weights. The only model-side host dependency is the tokenizer
saved in the already-selected checkpoint, used to reproduce and hash the exact
HF prompts that were previously evaluated.
"""

from __future__ import annotations

import hashlib
import json
import math
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml
from ir_training.eval.generate import build_prediction_record
from ir_training.eval.golden_set import load_fixed_golden_rows
from ir_training.eval.prepared_contract import (
    checked_preparation_manifest,
    verify_golden_preparation,
)
from ir_training.generation_policy import (
    closing_sentinel_end,
    sha256_text,
    stop_express_completion,
)
from ir_training.pipeline.official_mobile import NO_OP_CHECKS, SELECTOR, WORKFLOW

COHORTS = {"golden32": 32, "golden35": 35, "bixby50": 50}
SUPPORTED_WORKFLOWS = {
    "e2b_retained_mobile_golden_bixby_no_mtp_v1",
    "e2b_retained_mobile_golden_bixby_no_mtp_v2",
    WORKFLOW,
}
DEVICE_STAGE_ROOT = "/data/local/tmp/official_mobile_native_quality"
TEST_CLASS = "com.samsung.genuicraft.OfficialMobileNativeQualityProbeTest"
RUNNER = "com.samsung.genuicraft.test/androidx.test.runner.AndroidJUnitRunner"
APP_PACKAGE = "com.samsung.genuicraft"
EVENT_RE = re.compile(r"a2ui_native_event=(\{.*\})")
SHA_RE = re.compile(r"^\s*([0-9a-fA-F]{64})(?:\s+|$)")
THREADTIME_RE = re.compile(
    r"^\S+\s+\S+\s+(?P<pid>\d+)\s+\d+\s+\S\s+(?P<tag>[^:]+):\s?(?P<message>.*)$"
)


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def read_jsonl_strict(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{number}")
            rows.append(value)
    return rows


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_nonoverlapping_paths(
    run_dir: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    run = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"Official-mobile run directory is missing: {run}")
    if run == output or _is_relative_to(output, run) or _is_relative_to(run, output):
        raise ValueError(
            "--output-dir and --run-dir must be disjoint (neither may contain the other)"
        )
    if output.exists():
        raise FileExistsError(
            f"Choose a fresh native-quality output directory: {output}"
        )
    return run, output


def relocate_run_path(
    value: str | Path, *, original_root: str | Path, current_root: Path
) -> Path:
    # Receipts can be moved from a POSIX training host to a Windows Android
    # host (or conversely). Compare the saved spelling lexically before asking
    # the local pathlib implementation to interpret it.
    raw = str(value).replace("\\", "/")
    old = str(original_root).replace("\\", "/").rstrip("/")
    raw_fold, old_fold = raw.casefold(), old.casefold()
    if raw_fold == old_fold:
        relative_text = ""
    elif raw_fold.startswith(old_fold + "/"):
        relative_text = raw[len(old) + 1 :]
    elif not raw.startswith("/") and not re.match(r"^[A-Za-z]:/", raw):
        relative_text = raw
    else:
        raise ValueError(f"Bound path is outside the official-mobile run: {value}")
    parts = [part for part in relative_text.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"Bound path escapes the official-mobile run: {value}")
    return current_root.joinpath(*parts).resolve()


def _verified_receipt_files(
    receipt: dict[str, Any],
    *,
    original_root: str | Path,
    current_root: Path,
    required_prefix: Path | None = None,
) -> dict[str, str]:
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Stage receipt has no file bindings")
    verified: dict[str, str] = {}
    for old_path, expected in files.items():
        current = relocate_run_path(
            old_path, original_root=original_root, current_root=current_root
        )
        if required_prefix is not None and not _is_relative_to(
            current, required_prefix
        ):
            continue
        if not current.is_file():
            raise FileNotFoundError(
                f"Receipt-bound file is missing after run relocation: {current}"
            )
        observed = file_sha256(current)
        if observed != expected:
            raise ValueError(f"Receipt-bound file changed: {current}")
        verified[str(current)] = observed
    if required_prefix is not None and not verified:
        raise ValueError(f"Stage receipt binds no files under {required_prefix}")
    return verified


def inspect_completed_run(run_dir: str | Path) -> dict[str, Any]:
    """Verify the completed run, selected checkpoint, prepared data and export."""
    run = Path(run_dir).expanduser().resolve()
    plan_path = run / "official_mobile_plan.json"
    manifest_path = run / "official_mobile_manifest.json"
    plan, manifest = read_json(plan_path), read_json(manifest_path)
    if plan.get("workflow") not in SUPPORTED_WORKFLOWS or manifest.get(
        "workflow"
    ) != plan.get("workflow"):
        raise ValueError(
            "Run is not a supported completed official retained-mobile no-MTP workflow "
            f"({sorted(SUPPORTED_WORKFLOWS)})"
        )
    if manifest.get("status") != "complete" or manifest.get("plan") != plan:
        raise ValueError(
            "Native quality requires a complete manifest bound to the saved plan"
        )
    completed = manifest.get("completed")
    is_v2 = plan.get("workflow") == "e2b_retained_mobile_golden_bixby_no_mtp_v2"
    required_stages = [
        "prepare",
        "configure",
        *(["no_op_export"] if is_v2 else []),
        "training",
        *[f"best_{name}" for name in COHORTS],
        "export",
    ]
    if not isinstance(completed, dict) or any(
        stage not in completed for stage in required_stages
    ):
        raise ValueError(
            f"Run is missing completed prerequisite stages: {required_stages}"
        )
    plan_digest = file_sha256(plan_path)
    for stage in required_stages:
        saved = read_json(run / "stage_receipts" / f"{stage}.json")
        if (
            saved != completed[stage]
            or saved.get("stage") != stage
            or saved.get("plan_sha256") != plan_digest
        ):
            raise ValueError(f"Stage receipt is not bound to this run: {stage}")

    original_root = str(plan["options"]["output_dir"])
    paths = {
        name: relocate_run_path(value, original_root=original_root, current_root=run)
        for name, value in plan.get("paths", {}).items()
    }
    required_paths = (
        "config",
        "best_checkpoint",
        "export_report",
        "litertlm",
        *(["no_op_export_report"] if is_v2 else []),
    )
    if any(name not in paths for name in required_paths):
        raise ValueError(f"Official-mobile plan lacks required paths: {required_paths}")
    if not paths["best_checkpoint"].is_dir():
        raise FileNotFoundError(paths["best_checkpoint"])
    if not any(
        (paths["best_checkpoint"] / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    ):
        raise FileNotFoundError(
            "Selected checkpoint has no saved tokenizer; exact host prompt reproduction is unavailable"
        )

    receipt_bindings: dict[str, dict[str, str]] = {}
    receipt_bindings["prepare"] = _verified_receipt_files(
        completed["prepare"],
        original_root=original_root,
        current_root=run,
        required_prefix=run / "prepared",
    )
    receipt_bindings["configure"] = _verified_receipt_files(
        completed["configure"],
        original_root=original_root,
        current_root=run,
        required_prefix=run,
    )
    config_digest = file_sha256(paths["config"])
    if receipt_bindings["configure"].get(str(paths["config"])) != config_digest:
        raise ValueError(
            "Configure receipt is not bound to the resolved training config"
        )
    preparation_report = paths["config"].parent / "preparation_report.json"
    if (
        receipt_bindings["configure"].get(str(preparation_report))
        != file_sha256(preparation_report)
        or read_json(preparation_report).get("training_config_sha256") != config_digest
    ):
        raise ValueError(
            "Preparation report is not bound to the resolved training config"
        )
    receipt_bindings["training"] = _verified_receipt_files(
        completed["training"],
        original_root=original_root,
        current_root=run,
        required_prefix=paths["best_checkpoint"],
    )
    if is_v2:
        receipt_bindings["no_op_export"] = _verified_receipt_files(
            completed["no_op_export"],
            original_root=original_root,
            current_root=run,
            required_prefix=run,
        )
        no_op = read_json(paths["no_op_export_report"])
        if (
            no_op.get("passed") is not True
            or no_op.get("gate_status") != "PASSED"
            or no_op.get("mode") != "retained_scale_pretraining_noop_v1"
            or not isinstance(no_op.get("checks"), dict)
            or not NO_OP_CHECKS.issubset(no_op["checks"])
            or not all(value is True for value in no_op["checks"].values())
        ):
            raise ValueError(
                "v2 run lacks its passing retained-scale pretraining no-op export gate"
            )
        config_identity = no_op.get("resolved_training_config_identity") or {}
        if (
            config_identity.get("verified") is not True
            or relocate_run_path(
                config_identity.get("path", ""),
                original_root=original_root,
                current_root=run,
            )
            != paths["config"]
            or config_identity.get("sha256") != config_digest
        ):
            raise ValueError("No-op export gate is not bound to the resolved config")
        for field, path_key, hash_key, expected in (
            (
                "mobile_training_seed",
                "path",
                "manifest_sha256",
                str(plan["options"]["model_dir"]).rstrip("/\\")
                + "/mobile_training_seed_manifest.json",
            ),
            (
                "mobile_qparams",
                "contract_path",
                "contract_sha256",
                str(plan["options"]["model_dir"]).rstrip("/\\")
                + "/mobile_qparams.json",
            ),
        ):
            identity = no_op.get(field) or {}
            if (
                identity.get("verified") is not True
                or str(identity.get(path_key, "")).replace("\\", "/").casefold()
                != expected.replace("\\", "/").casefold()
                or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get(hash_key, "")))
            ):
                raise ValueError(f"No-op export gate has an invalid {field} identity")
    for cohort in COHORTS:
        prefix = run / "evaluations" / f"best_{cohort}"
        receipt_bindings[f"best_{cohort}"] = _verified_receipt_files(
            completed[f"best_{cohort}"],
            original_root=original_root,
            current_root=run,
            required_prefix=prefix,
        )
    receipt_bindings["export"] = _verified_receipt_files(
        completed["export"],
        original_root=original_root,
        current_root=run,
        required_prefix=paths["export_report"].parent,
    )

    export = read_json(paths["export_report"])
    model_sha = file_sha256(paths["litertlm"])
    if export.get("passed") is not True or export.get("executed") is not True:
        raise ValueError("Retained-scale export report is not an executed pass")
    if export.get("output_sha256") != model_sha:
        raise ValueError("Exported LiteRT-LM bytes differ from the export report")
    if (
        relocate_run_path(
            export.get("output_litertlm", ""),
            original_root=original_root,
            current_root=run,
        )
        != paths["litertlm"]
    ):
        raise ValueError("Export report names a different LiteRT-LM artifact")
    if (
        relocate_run_path(
            export.get("adapter_checkpoint", ""),
            original_root=original_root,
            current_root=run,
        )
        != paths["best_checkpoint"]
    ):
        raise ValueError("Export report is not bound to the selected best checkpoint")
    if (export.get("adapter_identity") or {}).get("verified") is not True:
        raise ValueError("Export report did not verify selected adapter identity")
    runtime = read_json(run / "deployment_runtime.json")
    if (
        runtime.get("model_sha256") != model_sha
        or runtime.get("required_mtp_enabled") is not False
    ):
        raise ValueError(
            "Deployment runtime identity or no-MTP policy differs from this artifact"
        )

    config = load_yaml(paths["config"])
    max_input_tokens = int(config["training"]["max_seq_length"])
    max_new_tokens = int(config["model"]["max_output_tokens"])
    if max_input_tokens != int(plan["options"]["max_seq_length"]):
        raise ValueError("Plan and resolved config disagree on input token limit")
    if max_new_tokens != int(plan["options"]["max_new_tokens"]):
        raise ValueError("Plan and resolved config disagree on output token limit")
    if max_new_tokens > 4096:
        raise ValueError(
            "Android native probe supports at most 4096 generated tokens per case"
        )
    if (
        plan.get("mtp", {}).get("training") is not False
        or plan.get("mtp", {}).get("inference") is not False
    ):
        raise ValueError(
            "This evaluator accepts only the no-MTP official-mobile workflow"
        )

    prepared_manifest = checked_preparation_manifest(run / "prepared")
    cohorts: dict[str, Any] = {}
    for name, count in COHORTS.items():
        split = run / "prepared" / f"{name}.jsonl"
        rows = load_fixed_golden_rows(
            split,
            max_rows=count,
            required_rows=count,
            require_exact_rows=True,
            require_unique_rows=True,
        )
        hf_dir = run / "evaluations" / f"best_{name}"
        result = read_json(hf_dir / "evaluation_result.json")
        predictions = read_jsonl_strict(hf_dir / "predictions.jsonl")
        if result.get("row_count") != count or result.get("qat_applied") is not True:
            raise ValueError(
                f"HF {name} result is incomplete or did not use retained fake QAT"
            )
        if len(predictions) != count or [row.get("id") for row in predictions] != [
            row.get("id") for row in rows
        ]:
            raise ValueError(
                f"HF {name} predictions do not exactly cover the prepared cohort"
            )
        kind = {
            "golden32": "explicit_repeated_case",
            "golden35": "fixed_strict_subset",
            "bixby50": "source_only_holdout",
        }[name]
        prepared = verify_golden_preparation(
            run / "prepared",
            split,
            required_rows=count,
            max_sequence=max_input_tokens,
            max_prompt=max_input_tokens,
            expected_kind=kind,
        )
        saved_prepared = json.loads(json.dumps(result.get("prepared_contract")))
        if isinstance(saved_prepared, dict) and "split_path" in saved_prepared:
            saved_prepared["split_path"] = str(
                relocate_run_path(
                    saved_prepared["split_path"],
                    original_root=original_root,
                    current_root=run,
                )
            )
        if saved_prepared != prepared:
            raise ValueError(
                f"HF {name} result is not bound to the current prepared contract"
            )
        if name == "bixby50" and (
            prepared.get("reference_available") is not False
            or any(row.get("reference_available") is not False for row in predictions)
        ):
            raise ValueError("Bixby50 must remain source-only with no reference IR")
        cohorts[name] = {
            "count": count,
            "split": split,
            "rows": rows,
            "prepared_contract": prepared,
            "hf_dir": hf_dir,
            "hf_result": result,
            "hf_predictions": predictions,
        }
    return {
        "run_dir": run,
        "original_root": original_root,
        "plan_path": plan_path,
        "plan_sha256": plan_digest,
        "plan": plan,
        "manifest_path": manifest_path,
        "paths": paths,
        "config": config,
        "model_sha256": model_sha,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "prepared_manifest_sha256": file_sha256(run / "prepared" / "manifest.json"),
        "prepared_manifest": prepared_manifest,
        "cohorts": cohorts,
        "receipt_bindings": receipt_bindings,
    }


def build_bound_requests(binding: dict[str, Any]) -> list[dict[str, Any]]:
    """Reproduce prompts with the checkpoint tokenizer and match prior HF hashes."""
    from ir_training.eval.litert_gpu import build_gpu_requests

    config = json.loads(json.dumps(binding["config"]))
    config["model"]["tokenizer_source"] = str(binding["paths"]["best_checkpoint"])
    all_requests: list[dict[str, Any]] = []
    for cohort, details in binding["cohorts"].items():
        config["prepared_evaluation_contract"] = details["prepared_contract"]
        requests = build_gpu_requests(
            details["rows"],
            model=binding["paths"]["litertlm"],
            model_config=config,
            max_input_tokens=binding["max_input_tokens"],
            max_new_tokens=binding["max_new_tokens"],
            mtp_enabled=False,
        )
        hf_by_id = {str(row["id"]): row for row in details["hf_predictions"]}
        for request in requests:
            golden_id = str(request["id"])
            hf = hf_by_id[golden_id]
            runtime = hf.get("runtime") if isinstance(hf.get("runtime"), dict) else {}
            if runtime.get("prompt_sha256") != request["prompt_sha256"]:
                raise ValueError(
                    f"HF/native prompt hash mismatch before device execution: {request['id']}"
                )
            expected_ids_hash = sha256_text(json.dumps(request["expected_input_ids"]))
            if runtime.get("input_token_ids_sha256") != expected_ids_hash:
                raise ValueError(
                    f"HF/native host token hash mismatch before device execution: {request['id']}"
                )
            policy = runtime.get("generation_policy") or {}
            expected_policy = {
                "version": "a2ui-envelope-v1",
                "stop_strings": ["</a2ui>"],
                "stop_scope": "generated_tokens_outside_quoted_literals",
                "max_new_tokens": binding["max_new_tokens"],
                "add_special_tokens": False,
            }
            if any(policy.get(key) != value for key, value in expected_policy.items()):
                raise ValueError(
                    f"HF/native generation-policy mismatch: {request['id']}"
                )
            eos_ids = policy.get("eos_token_ids")
            if (
                not isinstance(eos_ids, list)
                or not eos_ids
                or any(type(value) is not int or value < 0 for value in eos_ids)
            ):
                raise ValueError(
                    f"HF generation policy lacks bound EOS IDs: {request['id']}"
                )
            policy_sha = sha256_text(json.dumps(policy, sort_keys=True))
            if runtime.get("generation_policy_sha256") != policy_sha:
                raise ValueError(
                    f"HF generation-policy digest changed: {request['id']}"
                )
            request.update(
                cohort=cohort,
                golden_id=golden_id,
                id=f"{cohort}::{golden_id}",
                hf_prompt_sha256=runtime["prompt_sha256"],
                hf_input_token_ids_sha256=runtime["input_token_ids_sha256"],
                hf_generation_policy=policy,
                hf_generation_policy_sha256=policy_sha,
                host_input_token_ids=list(request.pop("expected_input_ids")),
                host_input_token_ids_sha256=expected_ids_hash,
                native_input_token_ids=None,
                native_output_token_ids=None,
                native_token_ids_available=False,
                native_token_ids_unavailable_reason=(
                    "LiteRT Android 0.16.1 public Conversation API exposes neither tokenize nor generated token IDs"
                ),
                chat_template_sha256=details["prepared_contract"]["tokenizer"][
                    "chat_template_sha256"
                ],
                tokenizer_vocabulary_sha256=details["prepared_contract"]["tokenizer"][
                    "vocabulary_sha256"
                ],
                chat_template_kwargs=details["prepared_contract"]["tokenizer"][
                    "chat_template_kwargs"
                ],
                prompt_scaffolds_sha256=details["prepared_contract"][
                    "prompt_scaffolds_sha256"
                ],
            )
            all_requests.append(request)
    expected = sum(COHORTS.values())
    ids = [str(row["id"]) for row in all_requests]
    if len(all_requests) != expected or len(set(ids)) != expected:
        raise ValueError(f"Native batch requires {expected} globally unique case IDs")
    return all_requests


def device_requests(requests: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = (
        "id",
        "formatted_prompt",
        "prompt_sha256",
        "messages",
        "chat_template_kwargs",
        "max_input_tokens",
        "max_new_tokens",
    )
    return [{key: row[key] for key in allowed} for row in requests]


def parse_delegation_evidence(logcat: str) -> dict[str, Any]:
    """Reuse the established parity parser without duplicating its log grammar."""
    import importlib.util

    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "benchmark_android_litertlm_gpu_parity.py"
    )
    spec = importlib.util.spec_from_file_location("a2ui_android_gpu_parity", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module.parse_logcat_evidence(logcat)


def scope_logcat_to_probe_pid(logcat: str) -> tuple[str, list[int]]:
    """Exclude delegation messages from unrelated device processes."""
    parsed = [
        match for line in logcat.splitlines() if (match := THREADTIME_RE.match(line))
    ]
    probe_pids = {
        int(match.group("pid"))
        for match in parsed
        if match.group("tag").strip() == "OfficialMobileNative"
        and "a2ui_native_event=" in match.group("message")
    }
    if len(probe_pids) != 1:
        raise ValueError(
            f"Expected one OfficialMobileNative instrumentation PID in logcat; observed {sorted(probe_pids)}"
        )
    pid = next(iter(probe_pids))
    return "\n".join(
        match.group(0) for match in parsed if int(match.group("pid")) == pid
    ), [pid]


def validate_native_outputs(
    requests: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    manifest: dict[str, Any],
    delegation: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_ids = [str(row["id"]) for row in requests]
    observed_ids = [str(row.get("id")) for row in outputs]
    if observed_ids != expected_ids or len(set(observed_ids)) != len(observed_ids):
        raise ValueError(
            "Native outputs must exactly cover all 117 requests in original order"
        )
    if manifest.get("success") is not True or manifest.get("completed_cases") != len(
        requests
    ):
        raise ValueError("Android batch manifest is incomplete")
    if (
        manifest.get("requested_backend") != "GPU"
        or manifest.get("mtp_enabled") is not False
    ):
        raise ValueError("Android batch did not retain the GPU/no-MTP contract")
    if (
        manifest.get("raw_session") is not False
        or manifest.get("conversation_template_applied") is not True
        or manifest.get("runtime_rendered_prompt_required") is not True
        or manifest.get("engine_instance_count") != 1
        or manifest.get("fresh_conversation_per_case") is not True
        or manifest.get("stop_policy_version") != "a2ui-envelope-v1"
        or manifest.get("sampler")
        != {
            "do_sample": False,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 0.0,
            "seed": 42,
        }
    ):
        raise ValueError("Android batch did not require exact runtime-rendered prompts")
    if delegation.get("all_gpu_subgraphs_fully_delegated") is not True:
        raise ValueError(
            "LiteRT logcat does not prove full GPU delegation; native quality fails closed"
        )
    by_id = {str(row["id"]): row for row in requests}
    merged: list[dict[str, Any]] = []
    for output in outputs:
        request = by_id[str(output["id"])]
        if (
            output.get("prompt_transport_verified") is not True
            or output.get("prompt_sha256") != request["prompt_sha256"]
        ):
            raise ValueError(
                f"Android prompt transport identity failed: {output.get('id')}"
            )
        if (
            output.get("runtime_template_prompt_verified") is not True
            or output.get("runtime_rendered_prompt_sha256") != request["prompt_sha256"]
        ):
            raise ValueError(
                f"Android runtime template prompt identity failed: {output.get('id')}"
            )
        if (
            output.get("requested_backend") != "GPU"
            or output.get("mtp_enabled") is not False
        ):
            raise ValueError(
                f"Android row violated GPU/no-MTP contract: {output.get('id')}"
            )
        if output.get("sampler") != {
            "do_sample": False,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 0.0,
            "seed": 42,
        }:
            raise ValueError(f"Android sampler contract changed: {output.get('id')}")
        if (
            output.get("stop_policy_version") != "a2ui-envelope-v1"
            or output.get("raw_output_scope")
            != "chunks_actually_returned_by_runtime; continuation_after_stop_unobserved"
        ):
            raise ValueError(f"Android stop policy changed: {output.get('id')}")
        raw = output.get("raw_generated_text")
        if not isinstance(raw, str):
            raise TypeError(f"Android row lacks raw generated text: {output.get('id')}")
        token_count = output.get("native_output_token_count")
        if (
            type(token_count) is not int
            or token_count < 0
            or token_count > int(request["max_new_tokens"])
        ):
            raise ValueError(f"Android output token bound failed: {output.get('id')}")
        has_boundary = closing_sentinel_end(raw) is not None
        if (output.get("stop_reason") == "closing_sentinel") != has_boundary:
            raise ValueError(
                f"Android stop boundary disagrees with raw text: {output.get('id')}"
            )
        merged.append(
            {
                **output,
                "cohort": request["cohort"],
                "formatted_prompt": request["formatted_prompt"],
                "serving_stopped_text": stop_express_completion(raw),
                "host_input_token_ids": request["host_input_token_ids"],
                "host_input_token_ids_sha256": request["host_input_token_ids_sha256"],
                "hf_generation_policy": request["hf_generation_policy"],
                "hf_generation_policy_sha256": request["hf_generation_policy_sha256"],
                "native_input_token_ids": None,
                "native_output_token_ids": None,
                "native_token_ids_available": False,
                "native_token_ids_unavailable_reason": request[
                    "native_token_ids_unavailable_reason"
                ],
                "chat_template_sha256": request["chat_template_sha256"],
                "tokenizer_vocabulary_sha256": request["tokenizer_vocabulary_sha256"],
                "chat_template_kwargs": request["chat_template_kwargs"],
                "prompt_scaffolds_sha256": request["prompt_scaffolds_sha256"],
            }
        )
    return merged


def build_native_predictions(
    binding: dict[str, Any], runtime_rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    by_id = {str(row["id"]): row for row in runtime_rows}
    result: dict[str, list[dict[str, Any]]] = {}
    for cohort, details in binding["cohorts"].items():
        predictions = []
        for golden in details["rows"]:
            runtime = by_id[f"{cohort}::{golden['id']}"]
            raw = runtime["raw_generated_text"]
            record = build_prediction_record(
                golden,
                raw,
                runtime={
                    key: value
                    for key, value in runtime.items()
                    if key
                    not in {
                        "id",
                        "raw_generated_text",
                        "serving_stopped_text",
                        "formatted_prompt",
                    }
                },
            )
            if record["generated_text"] != runtime["serving_stopped_text"]:
                raise ValueError(f"Serving-stop policy disagreement: {golden['id']}")
            predictions.append(record)
        if len(predictions) != details["count"]:
            raise ValueError(f"Incomplete native cohort: {cohort}")
        result[cohort] = predictions
    return result


def comparison_rows(
    binding: dict[str, Any],
    native_aggregates: dict[str, dict[str, Any]],
    checkpoint_aggregates: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for cohort, count in COHORTS.items():
        hf = (
            checkpoint_aggregates[cohort]
            if checkpoint_aggregates is not None
            else binding["cohorts"][cohort]["hf_result"]["aggregate"]
        )
        native = native_aggregates[cohort]
        row = {
            "cohort": cohort,
            "required_rows": count,
            "checkpoint_rows": binding["cohorts"][cohort]["hf_result"]["row_count"],
            "native_rows": count,
            "checkpoint_generation_reward_v5_4_avg": hf.get(
                "generation_reward_v5_4_avg"
            ),
            "native_generation_reward_v5_4_avg": native.get(
                "generation_reward_v5_4_avg"
            ),
            "checkpoint_schema_valid_strict_rate": hf.get("schema_valid_strict_rate"),
            "native_schema_valid_strict_rate": native.get("schema_valid_strict_rate"),
            "checkpoint_selector": hf.get(SELECTOR) if cohort == "golden32" else None,
            "native_selector_diagnostic": native.get(SELECTOR)
            if cohort == "golden32"
            else None,
            "selection_role": "checkpoint_selection"
            if cohort == "golden32"
            else "final_only_holdout",
            "reference_available": cohort != "bixby50",
            "checkpoint_score_origin": "recomputed_with_native_scorer_binding"
            if checkpoint_aggregates is not None
            else "saved_hf_evaluation",
        }
        rows.append(row)
    return rows


def render_results(
    rows: list[dict[str, Any]], *, model_sha256: str, report_status: str
) -> str:
    def metric(value: Any, percent: bool = False) -> str:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value) * 100:.2f}%" if percent else f"{float(value):.2f}"

    table = [
        "| Cohort | Coverage | Checkpoint reward v5.4 | Native reward v5.4 | Checkpoint strict | Native strict | Role |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['cohort']} | {row['native_rows']}/{row['required_rows']} | "
            f"{metric(row['checkpoint_generation_reward_v5_4_avg'])} | "
            f"{metric(row['native_generation_reward_v5_4_avg'])} | "
            f"{metric(row['checkpoint_schema_valid_strict_rate'], True)} | "
            f"{metric(row['native_schema_valid_strict_rate'], True)} | {row['selection_role']} |"
        )
    return "\n".join(
        [
            "# Official-mobile native Android quality",
            "",
            f"Status: {report_status}",
            f"Exported model SHA-256: `{model_sha256}`",
            "",
            *table,
            "",
            "Checkpoint columns are recomputed from the immutable retained-fake-QAT HF predictions with the same v5.4 scorer weights used for the fresh Android LiteRT GPU predictions.",
            "Golden32 selected the checkpoint before export. Native results never select or train a checkpoint.",
            "Bixby50 is source-only: its score covers generated IR validity/source representation, not reference-IR accuracy.",
            "Native token IDs are unavailable in the pinned Android LiteRT 0.16.1 public API; host/HF prompt token IDs and hashes are retained, but native token parity is not claimed.",
            "Full GPU delegation is a separate logcat-backed gate and CPU fallback is not allowed.",
            "",
        ]
    )


class AdbNativeBatchRunner:
    def __init__(
        self,
        *,
        adb: str,
        serial: str,
        app_package: str = APP_PACKAGE,
        runner: str = RUNNER,
        test_class: str = TEST_CLASS,
        command_runner: Callable[
            ..., subprocess.CompletedProcess[str]
        ] = subprocess.run,
    ) -> None:
        if not serial.strip() or serial.strip().startswith("<"):
            raise ValueError("Device execution requires an explicit --serial")
        self.adb = adb
        self.serial = serial.strip()
        self.app_package = app_package
        self.runner = runner
        self.test_class = test_class
        self.command_runner = command_runner
        package = r"(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*"
        class_name = rf"{package}(?:\$[A-Za-z_]\w*)*"
        if not re.fullmatch(package, app_package):
            raise ValueError(f"Unsafe Android package name: {app_package!r}")
        if not re.fullmatch(rf"{package}/{class_name}", runner):
            raise ValueError(f"Unsafe instrumentation component: {runner!r}")
        if not re.fullmatch(class_name, test_class):
            raise ValueError(f"Unsafe instrumentation test class: {test_class!r}")

    def adb_command(
        self, *args: str, timeout: float = 60, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = self.command_runner(
            [self.adb, "-s", self.serial, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if check and result.returncode:
            raise RuntimeError(
                f"ADB command failed ({result.returncode}): {' '.join(args)}\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )
        return result

    def preflight(self) -> dict[str, Any]:
        state = self.adb_command("get-state").stdout.strip()
        if state != "device":
            raise RuntimeError(f"ADB serial {self.serial} is not ready: {state!r}")
        app = self.adb_command("shell", "pm", "path", self.app_package).stdout.strip()
        test_package = self.runner.partition("/")[0]
        test = self.adb_command("shell", "pm", "path", test_package).stdout.strip()
        if not app.startswith("package:") or not test.startswith("package:"):
            raise RuntimeError(
                "Target app and instrumentation APK must already be installed"
            )
        return {
            "serial": self.serial,
            "state": state,
            "app_package": self.app_package,
            "test_package": test_package,
            "app_installed": True,
            "test_installed": True,
        }

    def _sha256(self, device_path: str) -> str:
        output = self.adb_command(
            "shell", "sha256sum", device_path, timeout=1800
        ).stdout
        match = SHA_RE.match(output)
        if not match:
            raise RuntimeError(f"Could not parse device SHA-256: {output[:300]!r}")
        return match.group(1).lower()

    def _instrument(
        self,
        command: list[str],
        *,
        log_path: Path,
        total_timeout: float,
        case_timeout: float,
        progress_seconds: float,
    ) -> dict[str, Any]:
        for name, value in (
            ("total_timeout", total_timeout),
            ("case_timeout", case_timeout),
            ("progress_seconds", progress_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def reader() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        started = case_started = last_progress = time.monotonic()
        current_id: str | None = "engine_initialization"
        last_event: dict[str, Any] | None = None
        try:
            with log_path.open("x", encoding="utf-8") as log:
                finished = False
                while not finished:
                    now = time.monotonic()
                    if now - started > total_timeout:
                        raise TimeoutError(
                            f"Android native batch exceeded {total_timeout:g}s"
                        )
                    if now - case_started > case_timeout:
                        raise TimeoutError(
                            f"Android native case {current_id} exceeded {case_timeout:g}s"
                        )
                    if now - last_progress >= progress_seconds:
                        print(
                            f"Android native heartbeat: current={current_id or 'engine'} elapsed={now - started:.0f}s",
                            flush=True,
                        )
                        last_progress = now
                    try:
                        line = lines.get(timeout=min(0.5, progress_seconds))
                    except queue.Empty:
                        continue
                    if line is None:
                        finished = True
                        continue
                    log.write(line)
                    log.flush()
                    match = EVENT_RE.search(line)
                    if match:
                        event = json.loads(match.group(1))
                        last_event = event
                        if event.get("phase") == "case_start":
                            current_id = str(event.get("id"))
                            case_started = time.monotonic()
                        elif event.get("phase") == "case_done":
                            current_id = "between_cases"
                            case_started = time.monotonic()
                        elif event.get("phase") == "engine_ready":
                            current_id = "before_first_case"
                            case_started = time.monotonic()
                process.wait(timeout=5)
                if process.returncode:
                    raise RuntimeError(
                        f"Android instrumentation exited {process.returncode}; see {log_path}"
                    )
                if not last_event or last_event.get("phase") != "finished":
                    raise RuntimeError(
                        "Android instrumentation exited without a finished event"
                    )
                return last_event
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            # Stopping the target process is the only fail-closed way to end a
            # native decode after the host-side adb client times out. It does
            # not clear or overwrite application data.
            self.adb_command("shell", "am", "force-stop", self.app_package, check=False)
            raise
        finally:
            if process.stdout:
                process.stdout.close()
            thread.join(timeout=2)

    def _require_absent(self, path: str, *, run_as: bool = False) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", path):
            raise ValueError(f"Unsafe device path: {path!r}")
        payload = f"test ! -e {shlex.quote(path)} && printf A2UI_ABSENT"
        remote = f"sh -c {shlex.quote(payload)}"
        if run_as:
            remote = f"run-as {self.app_package} {remote}"
        result = self.adb_command(
            "shell",
            remote,
            check=False,
        )
        if result.returncode != 0 or result.stdout != "A2UI_ABSENT":
            raise RuntimeError(
                f"Could not positively prove fresh/absent device path: {path}"
            )

    def _start_scoped_logcat(
        self, *, label: str, max_bytes: int = 8 * 1024 * 1024
    ) -> tuple[subprocess.Popen[str], threading.Thread, dict[str, Any]]:
        """Stream logcat, retaining only this run's process and bounded bytes."""
        process = subprocess.Popen(
            [self.adb, "-s", self.serial, "logcat", "-v", "threadtime"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        state: dict[str, Any] = {
            "pid": None,
            "lines": [],
            "bytes": 0,
            "overflow": False,
        }

        def reader() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                match = THREADTIME_RE.match(line)
                if not match:
                    continue
                pid = int(match.group("pid"))
                if (
                    state["pid"] is None
                    and match.group("tag").strip() == "OfficialMobileNative"
                    and f"run_label={label}" in match.group("message")
                ):
                    state["pid"] = pid
                if state["pid"] == pid:
                    size = len(line.encode("utf-8", errors="replace"))
                    if state["bytes"] + size > max_bytes:
                        state["overflow"] = True
                    elif not state["overflow"]:
                        state["lines"].append(line)
                        state["bytes"] += size

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        return process, thread, state

    @staticmethod
    def _stop_scoped_logcat(
        process: subprocess.Popen[str], thread: threading.Thread, state: dict[str, Any]
    ) -> str:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        thread.join(timeout=5)
        if process.stdout:
            process.stdout.close()
        if state["pid"] is None:
            raise RuntimeError("Scoped logcat never observed this native run label")
        if state["overflow"]:
            raise RuntimeError(
                "Scoped native logcat exceeded its bounded 8 MiB retention"
            )
        return "".join(state["lines"])

    def run(
        self,
        *,
        model: Path,
        model_sha256: str,
        device_requests_path: Path,
        output_dir: Path,
        total_timeout: float,
        case_timeout: float,
        progress_seconds: float,
    ) -> dict[str, Any]:
        preflight = self.preflight()
        run_id = uuid.uuid4().hex
        stage = f"{DEVICE_STAGE_ROOT}/{run_id}"
        if not re.fullmatch(
            r"/data/local/tmp/official_mobile_native_quality/[0-9a-f]{32}", stage
        ):
            raise AssertionError("Unsafe device staging path")
        label = f"native_quality_{run_id}"
        device_model = f"{stage}/model.litertlm"
        device_requests_path_value = f"{stage}/requests.jsonl"
        report_relative = f"files/official_mobile_native_quality_reports/{label}.jsonl"
        manifest_relative = (
            f"files/official_mobile_native_quality_reports/{label}.manifest.json"
        )
        cache_relative = f"cache/official_mobile_native_quality/{label}"
        cleanup: dict[str, Any] = {"attempted": False, "verified": False}
        stage_created = False
        app_paths_proved_fresh = False
        instrumentation_started = False
        logcat_process: subprocess.Popen[str] | None = None
        logcat_thread: threading.Thread | None = None
        logcat_state: dict[str, Any] | None = None
        scoped_logcat_text = ""

        def capture(*, required: bool) -> str:
            if scoped_logcat_text:
                (output_dir / "device_logcat.txt").write_text(
                    scoped_logcat_text, encoding="utf-8"
                )
            for relative, destination in (
                (report_relative, output_dir / "device_outputs.jsonl"),
                (manifest_relative, output_dir / "device_manifest.json"),
            ):
                result = self.adb_command(
                    "exec-out",
                    "run-as",
                    self.app_package,
                    "cat",
                    relative,
                    timeout=300,
                    check=required,
                )
                if result.returncode == 0 and result.stdout:
                    destination.write_text(
                        result.stdout, encoding="utf-8", newline="\n"
                    )
                elif required:
                    raise RuntimeError(
                        f"Required Android report is unavailable: {relative}"
                    )
            return scoped_logcat_text

        try:
            self._require_absent(stage)
            self._require_absent(report_relative, run_as=True)
            self._require_absent(manifest_relative, run_as=True)
            self._require_absent(cache_relative, run_as=True)
            app_paths_proved_fresh = True
            self.adb_command("shell", "mkdir", "-p", stage)
            stage_created = True
            self.adb_command("push", str(model), device_model, timeout=7200)
            self.adb_command(
                "push",
                str(device_requests_path),
                device_requests_path_value,
                timeout=600,
            )
            self.adb_command(
                "shell", "chmod", "0644", device_model, device_requests_path_value
            )
            staged_sha = self._sha256(device_model)
            if staged_sha != model_sha256:
                raise RuntimeError(
                    "Staged Android model hash differs from exported host artifact"
                )
            logcat_process, logcat_thread, logcat_state = self._start_scoped_logcat(
                label=label
            )
            command = [
                self.adb,
                "-s",
                self.serial,
                "shell",
                "am",
                "instrument",
                "-w",
                "-r",
                "-e",
                "class",
                self.test_class,
                "-e",
                "modelPath",
                device_model,
                "-e",
                "requestPath",
                device_requests_path_value,
                "-e",
                "label",
                label,
                "-e",
                "caseTimeoutSeconds",
                str(max(1, math.ceil(case_timeout))),
                self.runner,
            ]
            instrumentation_started = True
            self._instrument(
                command,
                log_path=output_dir / "instrumentation.log",
                total_timeout=total_timeout,
                case_timeout=case_timeout,
                progress_seconds=progress_seconds,
            )
            scoped_logcat_text = self._stop_scoped_logcat(
                logcat_process, logcat_thread, logcat_state
            )
            logcat_process = None
            logcat = capture(required=True)
            post_sha = self._sha256(device_model)
            if post_sha != model_sha256:
                raise RuntimeError("Android staged model changed during inference")
            scoped_logcat, probe_pids = scope_logcat_to_probe_pid(logcat)
            delegation = parse_delegation_evidence(scoped_logcat)
            delegation["probe_process_ids"] = probe_pids
            delegation["logcat_scope"] = "OfficialMobileNative instrumentation PID only"
            return {
                "preflight": preflight,
                "run_id": run_id,
                "device_stage": stage,
                "device_model_sha256_before": staged_sha,
                "device_model_sha256_after": post_sha,
                "device_outputs": read_jsonl_strict(
                    output_dir / "device_outputs.jsonl"
                ),
                "device_manifest": read_json(output_dir / "device_manifest.json"),
                "delegation": delegation,
            }
        except BaseException:
            if (
                logcat_process is not None
                and logcat_thread is not None
                and logcat_state is not None
            ):
                try:
                    scoped_logcat_text = self._stop_scoped_logcat(
                        logcat_process, logcat_thread, logcat_state
                    )
                except Exception as log_error:  # noqa: BLE001 - preserve original failure
                    (output_dir / "diagnostic_logcat_error.txt").write_text(
                        f"{type(log_error).__name__}: {log_error}\n", encoding="utf-8"
                    )
                logcat_process = None
            # Preserve whatever the test flushed before a native error/timeout.
            # These are diagnostics only; partial rows are never scored.
            try:
                capture(required=False)
            except Exception as capture_error:  # noqa: BLE001 - preserve the original native failure
                (output_dir / "diagnostic_capture_error.txt").write_text(
                    f"{type(capture_error).__name__}: {capture_error}\n",
                    encoding="utf-8",
                )
            raise
        finally:
            cleanup["attempted"] = True
            cleanup_errors: list[str] = []
            try:
                if stage_created:
                    self.adb_command("shell", "rm", "-rf", stage, timeout=600)
                if app_paths_proved_fresh and instrumentation_started:
                    self.adb_command(
                        "shell",
                        "run-as",
                        self.app_package,
                        "rm",
                        "-f",
                        report_relative,
                        manifest_relative,
                    )
                    self.adb_command(
                        "shell", "run-as", self.app_package, "rm", "-rf", cache_relative
                    )
                self._require_absent(stage)
                self._require_absent(report_relative, run_as=True)
                self._require_absent(manifest_relative, run_as=True)
                self._require_absent(cache_relative, run_as=True)
                stage_absent = report_absent = manifest_absent = cache_absent = True
            except Exception as cleanup_error:  # noqa: BLE001 - publish explicit cleanup failure
                cleanup_errors.append(
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                stage_absent = report_absent = manifest_absent = cache_absent = False
            cleanup.update(
                stage_absent=stage_absent,
                report_absent=report_absent,
                manifest_absent=manifest_absent,
                cache_absent=cache_absent,
                verified=all(
                    (stage_absent, report_absent, manifest_absent, cache_absent)
                ),
                errors=cleanup_errors,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "device_cleanup.json").write_text(
                json.dumps(cleanup, indent=2) + "\n", encoding="utf-8"
            )
            if not cleanup["verified"] and sys.exc_info()[0] is None:
                raise RuntimeError(
                    "Unique Android test staging cleanup could not be verified"
                )


def resolve_adb(explicit: str | None) -> str:
    value = explicit or shutil.which("adb")
    if not value:
        raise FileNotFoundError("adb was not found; pass --adb explicitly")
    return str(Path(value).expanduser().resolve())
