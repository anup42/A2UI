from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from importlib.metadata import version, PackageNotFoundError

from utils.config import load_yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_release_version(root: Path) -> str:
    version_path = root / "VERSION"
    if not version_path.exists():
        return "0.0.0"
    raw = version_path.read_text(encoding="utf-8").strip()
    return raw or "0.0.0"


def load_component_versions(root: Path) -> dict[str, str]:
    path = root / "versions" / "components.yaml"
    if not path.exists():
        return {}
    data = load_yaml(path)
    components = data.get("components") if isinstance(data, dict) else None
    if not isinstance(components, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in components.items():
        if key is None:
            continue
        out[str(key)] = str(value)
    return out


def load_compatibility(root: Path) -> dict[str, Any]:
    path = root / "versions" / "components.yaml"
    if not path.exists():
        return {}
    data = load_yaml(path)
    compatibility = data.get("compatibility") if isinstance(data, dict) else None
    if not isinstance(compatibility, dict):
        return {}
    return compatibility


def load_version_bundle(root: Path) -> dict[str, Any]:
    return {
        "release": load_release_version(root),
        "components": load_component_versions(root),
        "compatibility": load_compatibility(root),
    }


def format_version_report(root: Path) -> str:
    bundle = load_version_bundle(root)
    lines = [f"release: {bundle['release']}"]
    components = bundle.get("components", {}) or {}
    if components:
        lines.append("components:")
        for name in sorted(components.keys()):
            lines.append(f"  {name}: {components[name]}")
    compatibility = bundle.get("compatibility", {}) or {}
    if compatibility:
        lines.append("compatibility:")
        for name in sorted(compatibility.keys()):
            lines.append(f"  {name}: {compatibility[name]}")
    return "\n".join(lines)


def _run_git(root: Path, args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value


def get_git_commit(root: Path) -> str | None:
    return _run_git(root, ["rev-parse", "HEAD"])


def get_git_branch(root: Path) -> str | None:
    return _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])


def get_git_dirty(root: Path) -> bool | None:
    status = _run_git(root, ["status", "--porcelain"])
    if status is None:
        return None
    return bool(status)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        h.update(str(path).encode("utf-8"))
        h.update(b"\n")
        digest = sha256_file(path)
        if digest is not None:
            h.update(digest.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_run_manifest(
    root: Path,
    run_id: str,
    stage: int | str,
    model_spec: Any,
    run_paths: Any,
    run_cfg_path: Path,
    models_cfg_path: Path,
    argv: list[str] | None = None,
    effective_run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    release = load_release_version(root)
    components = load_component_versions(root)
    compatibility = load_compatibility(root)

    config_files = [run_cfg_path, models_cfg_path]
    config_hashes = {
        str(path.relative_to(root)) if path.is_absolute() and str(path).startswith(str(root)) else str(path): sha256_file(path)
        for path in config_files
    }

    rel = lambda p: str(p.relative_to(root)) if p.is_absolute() and str(p).startswith(str(root)) else str(p)

    manifest: dict[str, Any] = {
        "manifest_version": 2,
        "phase_invocation_id": uuid4().hex,
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "stage": stage,
        "repo": {
            "release_version": release,
            "components": components,
            "compatibility": compatibility,
            "git_commit": get_git_commit(root),
            "git_branch": get_git_branch(root),
            "git_dirty": get_git_dirty(root),
        },
        "model": {
            "name": getattr(model_spec, "name", None),
            "provider": getattr(model_spec, "provider", None),
            "model": getattr(model_spec, "model", None),
        },
        "config": {
            "files": config_hashes,
            "combined_sha256": sha256_files(config_files),
        },
        "paths": {
            "run_dir": rel(getattr(run_paths, "run_dir", Path(""))),
            "queries_path": rel(getattr(run_paths, "queries_path", Path(""))),
            "responses_path": rel(getattr(run_paths, "responses_path", Path(""))),
            "genui_path": rel(getattr(run_paths, "genui_path", Path(""))),
            "metrics_path": rel(getattr(run_paths, "metrics_path", Path(""))),
            "aggregates_path": rel(getattr(run_paths, "aggregates_path", Path(""))),
            "artifacts_dir": rel(getattr(run_paths, "artifacts_dir", Path(""))),
            "manifest_path": rel(getattr(run_paths, "manifest_path", getattr(run_paths, "run_dir", Path("")) / "run_manifest.json")),
        },
        "command": {
            "argv": argv if argv is not None else sys.argv,
            "cwd": os.getcwd(),
        },
    }
    # Only non-secret generation controls enter provenance. Do not dump os.environ.
    runtime_keys = (
        "A2UI_CALL_SLEEP_SECONDS", "A2UI_STAGE1_INTENT_BATCH_SIZE", "A2UI_STAGE1_INTENT_CYCLE_SIZE",
        "A2UI_MAX_REPAIR_ATTEMPTS", "A2UI_MAX_ATTEMPTS", "A2UI_QUERY_TEMPERATURE",
        "A2UI_RESPONSE_TEMPERATURE", "A2UI_RESPONSE_TEMPERATURES", "A2UI_GENUI_TEMPERATURE",
        "A2UI_GENUI_MAX_TOKENS", "A2UI_GENUI_PROMPT_MAX_TOKENS", "A2UI_STAGE3_BATCH_SIZE",
        "A2UI_STAGE2_BATCH_SIZE", "A2UI_STAGE1_BATCH_SIZE", "STAGE3_FINAL_REGEN_ATTEMPTS",
        "LOCAL_VLLM_MAX_MODEL_LEN", "LOCAL_VLLM_MAX_OUTPUT_TOKENS", "LOCAL_VLLM_ENABLE_THINKING",
        "LOCAL_VLLM_BATCH_PARALLELISM", "LOCAL_VLLM_PARALLEL_REQUESTS", "LOCAL_VLLM_TOP_P",
        "LOCAL_VLLM_TOP_K", "LOCAL_VLLM_MIN_P", "LOCAL_VLLM_CHAT_TEMPLATE_KWARGS",
        "LOCAL_STAGE3_PROMPT_MAX_TOKENS", "LOCAL_STAGE3_PROMPT_TOKEN_MULTIPLIER",
        "VLLM_MAX_MODEL_LEN", "VLLM_TENSOR_PARALLEL_SIZE", "VLLM_MAX_NUM_SEQS",
        "VLLM_MAX_NUM_BATCHED_TOKENS", "VLLM_GPU_MEMORY_UTILIZATION",
    )
    manifest["runtime_overrides"] = {key: os.environ[key] for key in runtime_keys if key in os.environ}
    manifest["config"]["effective_run_config"] = effective_run_config
    source_files = sorted((root / "src").rglob("*.py"))
    contract_files = sorted((root / "schema").glob("*.json"))
    prompt_files = sorted((root / "prompts").glob("*.md"))
    for env_key in ("A2UI_STAGE1_PROMPT_FILE", "A2UI_STAGE3_PROMPT_FILE"):
        configured = os.environ.get(env_key)
        if configured:
            candidate = Path(configured)
            prompt_files.append(candidate if candidate.is_absolute() else root / candidate)
    manifest["implementation_files"] = {rel(path): sha256_file(path) for path in source_files}
    manifest["contract_files"] = {rel(path): sha256_file(path) for path in contract_files + prompt_files}
    packages = {}
    for name in ("vllm", "torch", "transformers", "jsonschema", "google-genai"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    manifest["runtime"] = {"python": sys.version, "packages": packages}
    return manifest


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    phases = path.parent / "phase_manifests"
    phases.mkdir(parents=True, exist_ok=True)
    # Preserve a pre-upgrade manifest once before maintaining the compatibility pointer.
    if path.exists():
        previous = path.read_bytes()
        old_hash = hashlib.sha256(previous).hexdigest()
        legacy = phases / f"previous_{old_hash}.json"
        if not legacy.exists():
            with legacy.open("xb") as handle:
                handle.write(previous)
    snapshot = dict(manifest)
    snapshot["phase_invocation_id"] = snapshot.get("phase_invocation_id") or uuid4().hex
    phase = str(snapshot.get("stage", "unknown")).replace("/", "_").replace("\\", "_")
    phase_path = phases / f"stage_{phase}_{snapshot['phase_invocation_id']}.json"
    snapshot["phase_manifest_path"] = str(phase_path.relative_to(path.parent))
    encoded = json.dumps(snapshot, indent=2, ensure_ascii=False).encode("utf-8")
    with phase_path.open("xb") as handle:
        handle.write(encoded)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
