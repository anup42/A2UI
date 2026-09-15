"""Explicit post-training recovery: never restart training or trust stale evidence."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from ir_training.common.progress import Progress, log


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected recovery evidence object: {path}")  # noqa: TRY004 - malformed persisted file
    return value


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _semantic_plan(value):
    # Logging/deadlines may change without modifying the saved training config.
    ignored = {"timeouts", "tensorboard_detail", "logging_steps", "progress_seconds"}
    if isinstance(value, dict):
        return {key: _semantic_plan(item) for key, item in value.items() if key not in ignored}
    if isinstance(value, list):
        return [_semantic_plan(item) for item in value]
    return value


def verify_completed_files(completed: dict, *, verified: dict | None = None, allow_empty: tuple = ()) -> None:
    verified = {} if verified is None else verified
    if not isinstance(completed, dict) or not completed:
        raise ValueError("No completed stage evidence is available for recovery")
    for stage, entry in completed.items():
        files = entry.get("files") if isinstance(entry, dict) else None
        if not isinstance(files, dict) or (not files and stage not in allow_empty):
            raise ValueError(f"Completed stage has no bound files: {stage}")
        for name, expected in files.items():
            path = Path(name)
            if not path.is_absolute() or not path.is_file() or not isinstance(expected, str):
                raise ValueError(f"Missing/invalid completed recovery evidence: {name}")
            if name not in verified:
                verified[name] = _digest(path)
            if verified[name] != expected:
                raise ValueError(f"Completed recovery evidence changed: {name}")


def validate_resume(state: dict, plan: dict) -> None:
    if state.get("status") not in {"failed", "running", "complete"}:
        raise ValueError("--resume-run requires an existing deployment manifest")
    if _semantic_plan(state.get("plan")) != _semantic_plan(plan):
        raise ValueError("Resume options differ from the saved deployment. Reuse the original command and paths; only logging/deadline options may change.")
    completed = state.get("completed") or {}
    required = "full_training_and_checkpoint_evaluation"
    if required not in completed:
        raise ValueError("--resume-run only recovers AFTER completed full training and checkpoint evaluation; it will not restart training/tuning or retry their holdouts")
    training = Path(plan["training"]["options"]["output_dir"])
    with Progress("Verify retained deployment and training evidence", unit="stage"):
        verified = {}
        verify_completed_files(completed, verified=verified)
        nested = _json(training / "pipeline_manifest.json")
        if nested.get("status") != "complete":
            raise ValueError("Retained training pipeline is not complete")
        verify_completed_files(nested.get("completed"), verified=verified, allow_empty=("preflight",))
        if "merge" in completed:
            merged = Path((state.get("artifact_directories") or {}).get("merged", plan["export"]["merged_model_dir"]))
            if not merged.resolve().is_relative_to(Path(plan["output_dir"]).resolve()):
                raise ValueError("Retained merged model escapes the deployment output")
            source = _json(merged / "deployment_source.json")
            files = source.get("merged_files")
            if not isinstance(files, dict) or not files:
                raise ValueError("Retained merged model lacks its bound weight/tokenizer inventory")
            actual = {item.name for item in merged.iterdir() if item.is_file() and item.name != "deployment_source.json"}
            if actual != set(files):
                raise ValueError("Retained merged model file inventory changed")
            bound = {}
            for name, digest in files.items():
                path = merged / name
                if Path(name).name != name or path.is_symlink():
                    raise ValueError("Unsafe retained merged model file")
                bound[str(path)] = digest
            verify_completed_files({"merged_weights": {"files": bound}}, verified=verified)
    log("Recovery evidence verified; completed training, exports and evaluations will not rerun")


@contextmanager
def deployment_lock(output: Path):
    """Kernel lock released on process exit; the empty marker is not a stale lock."""
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.parent / f".{output.name}.deployment.lock"
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        acquired = False
        try:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                raise RuntimeError(f"Another deployment process is using {output}; do not start concurrent recovery") from exc
            yield
        finally:
            if acquired:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
