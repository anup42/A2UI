"""Read-only gates for GRPO post-training continuation and fresh SFT reruns.

Continuation never resumes an optimizer, reruns preparation, deletes partial
outputs, or adopts an unrecorded stage. Existing receipts remain immutable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from ir_training.common.config import training_root

CONTRACT_NAME = "grpo_continuation_contract.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")  # noqa: TRY004 - artifact contract
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def continuation_code_identity() -> dict[str, Any]:
    """Bind local training/export/evaluation implementations, including dirty edits.

    A Git SHA alone cannot detect modified or new source files. Hash the Python
    trees used by this workflow, not environment/site-packages or model files.
    The identity is deliberately conservative: a code update needs a new run,
    not an assertion that old results came from the updated implementation.
    """
    root = training_root().parent
    bindings = {}
    for directory in (root / "training/src", root / "training/scripts", root / "dataset/src"):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*.py")):
            bindings[path.relative_to(root).as_posix()] = _sha(path)
    if not bindings:
        raise ValueError("No workflow code was available for continuation binding")
    return {"policy": "python_source_trees_sha256_v1", "files": bindings, "sha256": _content_sha(bindings)}


def build_continuation_contract(plan: dict[str, Any]) -> dict[str, Any]:
    """Persist this in configs/ before sealing the configure-stage receipt."""
    from ir_training.train.qat_grpo_reward import reward_implementation_identity

    return {
        "schema_version": 1,
        "policy": "post_training_only_v1",
        "plan_content_sha256": _content_sha(plan),
        "code": continuation_code_identity(),
        "reward": reward_implementation_identity(Path(plan["paths"]["reward_config"])),
    }


def _pending_stage_paths(plan: dict[str, Any], stage: str) -> list[Path]:
    output = Path(plan["options"]["output_dir"])
    paths = plan["paths"]
    result = [output / f"stage_receipts/{stage}.json",
              output / f"logs/{stage}.log", output / f"logs/{stage}_worker.log"]
    if stage.startswith("best_"):
        result.append(output / "evaluations" / stage)
    elif stage == "merge":
        result.append(Path(paths["merged"]))
    elif stage == "export":
        result.append(Path(paths["export_dir"]))
    elif stage == "native_quality":
        result.append(output.parent / f"{output.name}_native_quality")
    else:
        raise ValueError(f"Continuation cannot run pre-training stage: {stage}")
    return result


def verify_post_training_continuation(output_dir: str | Path) -> dict[str, Any]:
    """Return the immutable plan/state and pending stages, without writes or ML IO."""
    from ir_training.pipeline import official_mobile
    from ir_training.pipeline.golden_training import GOLDENS
    from ir_training.pipeline.qat_grpo import MANIFEST_NAME, PLAN_NAME, WORKFLOW

    output = Path(output_dir).expanduser().resolve(strict=True)
    plan_path = output / PLAN_NAME
    plan = _read(plan_path)
    state = _read(output / MANIFEST_NAME)
    if (plan.get("workflow") != WORKFLOW or state.get("workflow") != WORKFLOW
            or state.get("plan") != plan or plan.get("schema_version") != 1
            or Path(plan["options"]["output_dir"]).resolve() != output):
        raise ValueError("Continuation requires this run's unchanged GRPO plan and manifest")
    if state.get("status") not in {"failed", "awaiting_native_evaluation"}:
        raise ValueError("Continuation requires a stopped failed/awaiting-native run, not a running or complete run")
    expected = ["assets", "prepare", "configure",
                *(["no_op_export"] if plan["options"]["family"] == "e2b" else []),
                "runtime_preflight", "training", *[f"best_{name}" for name in GOLDENS],
                "merge", "export", "native_quality"]
    stages = plan.get("stages")
    completed = state.get("completed")
    if stages != expected or not isinstance(completed, dict) or list(completed) != stages[:len(completed)]:
        raise ValueError("Continuation requires an exact ordered prefix of canonical stage receipts")
    if "training" not in completed:
        raise ValueError("Continuation is post-training only; optimizer/preparation resume is unsupported")
    pending = stages[len(completed):]
    if not pending:
        raise ValueError("No post-training stages remain")
    if (state.get("active_stage") not in {None, pending[0]}
            or (state["status"] == "awaiting_native_evaluation" and pending != ["native_quality"])):
        raise ValueError("Continuation status/active stage does not match the completed stage prefix")
    digest = _sha(plan_path)
    for stage, receipt in completed.items():
        saved = _read(output / f"stage_receipts/{stage}.json")
        if (receipt != saved or receipt.get("stage") != stage
                or receipt.get("plan_sha256") != digest
                or not isinstance(receipt.get("files"), dict) or not receipt["files"]):
            raise ValueError(f"Continuation receipt is not bound to the immutable plan: {stage}")
        official_mobile._verify_bindings(receipt["files"])
    contract_path = output / "configs" / CONTRACT_NAME
    if str(contract_path) not in completed["configure"]["files"]:
        raise ValueError("Run has no sealed continuation contract; start a fresh run from SFT")
    if _read(contract_path) != build_continuation_contract(plan):
        raise ValueError("GRPO code, reward implementation, or immutable plan changed; continuation is unsafe")
    official_mobile._verify_bindings(plan["source_adapter_files"])
    if plan["options"].get("objective", "grpo") == "dpo":
        from ir_training.pipeline.qat_grpo import _verify_preference_binding
        _verify_preference_binding(plan)
        if not set(plan["preference_files"]).issubset(completed["configure"]["files"]):
            raise ValueError("Continuation requires the sealed DPO preference source")
    if _sha(Path(plan["options"]["sft_config"])) != plan["source_config_sha256"]:
        raise ValueError("Source SFT configuration changed")
    if _sha(Path(plan["options"]["official_litertlm"])) != plan["official_artifact_sha256"]:
        raise ValueError("Pinned official model package changed")
    preparation_path = output / "configs/preparation_report.json"
    if str(preparation_path) not in completed["configure"]["files"]:
        raise ValueError("Continuation requires a sealed preparation/model identity report")
    model_files = _read(preparation_path).get("model_files")
    if not isinstance(model_files, dict) or not model_files:
        raise ValueError("Continuation cannot verify original model files")
    model_dir = Path(plan["options"]["model_dir"])
    for name, expected_sha in model_files.items():
        if Path(name).name != name or _sha(model_dir / name) != expected_sha:
            raise ValueError(f"Original model file changed or is invalid: {name}")
    existing = [str(path) for stage in pending for path in _pending_stage_paths(plan, stage) if path.exists()]
    if existing:
        raise ValueError("Partial pending-stage artifacts/logs are preserved. Archive them explicitly before "
                         "continuing; do not edit completed receipts or artifacts: " + ", ".join(existing))
    return {"plan": plan, "state": state, "pending_stages": pending, "plan_sha256": digest}


def options_from_previous_run(old_run: str | Path, new_output: str | Path,
                              input_dir: str | Path | None = None):
    """Start fresh from the original SFT adapter, never a previous GRPO adapter.

    Reuse hyperparameters and source paths only. Raw pairs are re-prepared by
    the normal pipeline with the current prompt/reward implementation.
    """
    from ir_training.pipeline.qat_grpo import PLAN_NAME, WORKFLOW, QATGRPOOptions

    previous = Path(old_run).expanduser().resolve(strict=True)
    output = Path(new_output).expanduser().resolve()
    if output.exists():
        raise ValueError("Fresh GRPO rerun requires a new, nonexistent output directory")
    if output.is_relative_to(previous) or previous.is_relative_to(output):
        raise ValueError("Fresh output must be disjoint from the previous GRPO run")
    plan = _read(previous / PLAN_NAME)
    if plan.get("workflow") != WORKFLOW or plan.get("schema_version") != 1:
        raise ValueError("Previous run is not a supported QAT LoRA GRPO plan")
    old = plan.get("options")
    if not isinstance(old, dict):
        raise ValueError("Previous GRPO plan has no options")  # noqa: TRY004 - artifact contract
    if Path(old["output_dir"]).resolve() != previous:
        raise ValueError("Previous GRPO plan belongs to another output directory")
    allowed = {field.name for field in fields(QATGRPOOptions)}
    unknown = set(old) - allowed
    if unknown:
        raise ValueError(f"Previous GRPO plan has unsupported options: {sorted(unknown)}")
    raw_source = input_dir if input_dir is not None else old.get("input_dir")
    if raw_source is None:
        raise ValueError("Previous run used prepared input; supply the original raw train.jsonl/val.jsonl directory")
    raw = Path(raw_source).expanduser().resolve(strict=True)
    if raw.is_relative_to(previous):
        raise ValueError("Fresh preparation must use original raw pairs, not the previous run's prepared directory")
    for name in ("train.jsonl", "val.jsonl"):
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)
    values = dict(old)
    values.update(input_dir=raw, prepared_input_dir=None, output_dir=output,
                  rollout_microbatch=1, rollout_qat_weight_cache=False)
    for key in ("model_dir", "sft_config", "sft_checkpoint", "official_litertlm", "exporter_python",
                "source_safetensors", "dpo_preferences"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    return QATGRPOOptions(**values)
