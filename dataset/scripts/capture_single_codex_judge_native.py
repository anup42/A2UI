#!/usr/bin/env python3
"""Resumable native evidence capture for Single-Codex Judge v2.

This script only controls the coexisting ``.judgecapture`` package. It never
uninstalls, clears, downgrades, or launches the data-bearing production package.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = "com.samsung.genuicraft.judgecapture"
DATA_BEARING_PACKAGE = "com.samsung.genuicraft"
ACTIVITY = "com.samsung.genuicraft.DatasetRenderCaptureActivity"
BATCH_SIZE = 50
MAX_INFRASTRUCTURE_ATTEMPTS = 3

PROFILES: tuple[dict[str, Any], ...] = (
    {
        "name": "compact",
        "width_dp": None,
        "height_dp": None,
        "size_px": None,
        "density_dpi": None,
    },
    {
        "name": "medium_700dp",
        "width_dp": 700,
        "height_dp": 1000,
        "size_px": "1400x2000",
        "density_dpi": 320,
    },
    {
        "name": "expanded_900dp",
        "width_dp": 900,
        "height_dp": 1200,
        "size_px": "1800x2400",
        "density_dpi": 320,
    },
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class Adb:
    def __init__(self, executable: str, serial: str) -> None:
        self.executable = executable
        self.serial = serial

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout: float = 60.0,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [self.executable, "-s", self.serial, *arguments]
        result = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode:
            raise RuntimeError(
                f"adb command failed ({result.returncode}): "
                f"{' '.join(command)}\n{result.stdout[-2000:]}"
            )
        return result

    def shell(
        self,
        *arguments: str,
        timeout: float = 60.0,
        check: bool = True,
    ) -> str:
        return self.run(
            ["shell", *arguments], timeout=timeout, check=check
        ).stdout.strip()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout.strip()


def _renderer_source_fingerprint() -> tuple[str, list[dict[str, str]]]:
    files = sorted(
        (
            REPO_ROOT
            / "android"
            / "app"
            / "src"
            / "main"
            / "java"
            / "com"
            / "samsung"
            / "genuicraft"
            / "renderer"
        ).rglob("*.kt")
    )
    inventory = [
        {
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _hash_file(path),
        }
        for path in files
    ]
    return _hash_bytes(_canonical_json(inventory).encode()), inventory


def _extract_override(output: str, label: str) -> str | None:
    match = re.search(rf"{re.escape(label)}:\s*([^\r\n]+)", output)
    return match.group(1).strip() if match else None


def _display_baseline(adb: Adb) -> dict[str, Any]:
    size_output = adb.shell("wm", "size")
    density_output = adb.shell("wm", "density")
    return {
        "wm_size_output": size_output,
        "wm_density_output": density_output,
        "override_size": _extract_override(size_output, "Override size"),
        "override_density": _extract_override(
            density_output, "Override density"
        ),
        "font_scale": adb.shell(
            "settings", "get", "system", "font_scale"
        ),
        "locale": (
            adb.shell("getprop", "persist.sys.locale", check=False)
            or adb.shell("getprop", "ro.product.locale", check=False)
        ),
        "ui_mode": adb.shell("cmd", "uimode", "night", check=False),
    }


def _restore_display(adb: Adb, baseline: Mapping[str, Any]) -> None:
    override_size = baseline.get("override_size")
    override_density = baseline.get("override_density")
    if override_size:
        adb.shell("wm", "size", str(override_size))
    else:
        adb.shell("wm", "size", "reset")
    if override_density:
        adb.shell("wm", "density", str(override_density))
    else:
        adb.shell("wm", "density", "reset")


def _apply_profile(
    adb: Adb,
    profile: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    _restore_display(adb, baseline)
    if profile["size_px"] is not None:
        adb.shell("wm", "size", str(profile["size_px"]))
    if profile["density_dpi"] is not None:
        adb.shell("wm", "density", str(profile["density_dpi"]))
    time.sleep(1.0)
    size_output = adb.shell("wm", "size")
    density_output = adb.shell("wm", "density")
    effective_size = (
        _extract_override(size_output, "Override size")
        or _extract_override(size_output, "Physical size")
        or ""
    )
    effective_density_raw = (
        _extract_override(density_output, "Override density")
        or _extract_override(density_output, "Physical density")
        or "0"
    )
    density = int(re.sub(r"\D", "", effective_density_raw) or "0")
    match = re.fullmatch(r"(\d+)x(\d+)", effective_size)
    width_px = int(match.group(1)) if match else 0
    height_px = int(match.group(2)) if match else 0
    return {
        **dict(profile),
        "effective_size_px": effective_size,
        "effective_density_dpi": density,
        "effective_width_dp": (
            width_px * 160.0 / density if density else None
        ),
        "effective_height_dp": (
            height_px * 160.0 / density if density else None
        ),
    }


def _tabs_state_variants(
    record: Mapping[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    spec = record.get("genui_json")
    if not isinstance(spec, Mapping):
        return []
    elements = spec.get("elements")
    if not isinstance(elements, Mapping):
        return []
    changes: list[tuple[str, str, int]] = []
    for element_id, element in sorted(elements.items()):
        if not isinstance(element, Mapping):
            continue
        if str(element.get("type") or "").casefold() != "tabs":
            continue
        props = element.get("props")
        if not isinstance(props, Mapping):
            continue
        tabs = props.get("tabs")
        if not isinstance(tabs, Sequence) or isinstance(tabs, (str, bytes)):
            continue
        targets: list[str] = []
        for tab in tabs:
            if not isinstance(tab, Mapping):
                continue
            target = next(
                (
                    str(tab[key])
                    for key in ("child", "content", "id", "element")
                    if str(tab.get(key) or "").strip()
                ),
                "",
            )
            if target:
                targets.append(target)
        initial = str(props.get("activeTabId") or "")
        if not initial and targets:
            initial = targets[0]
        for tab_index, target in enumerate(targets):
            if target != initial:
                changes.append((str(element_id), target, tab_index))
    variants: list[tuple[str, str, dict[str, Any]]] = []
    for variant_index, (element_id, target, tab_index) in enumerate(
        changes[:4],
        start=1,
    ):
        copied = json.loads(_canonical_json(record))
        copied["genui_json"]["elements"][element_id].setdefault(
            "props", {}
        )["activeTabId"] = target
        variants.append(
            (
                f"tabs_{variant_index}",
                (
                    f"Tabs element {element_id} selected renderer child "
                    f"{target} at tab index {tab_index}"
                ),
                copied,
            )
        )
    return variants


def _set_json_pointer(
    root: dict[str, Any],
    pointer: str,
    value: Any,
) -> bool:
    if not pointer.startswith("/"):
        return False
    tokens = [
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer.split("/")[1:]
    ]
    if not tokens or len(tokens) > 16 or any(not token for token in tokens):
        return False
    current: Any = root
    for token in tokens[:-1]:
        if not isinstance(current, dict) or token not in current:
            return False
        current = current[token]
    if not isinstance(current, dict) or tokens[-1] not in current:
        return False
    if current[tokens[-1]] == value:
        return False
    current[tokens[-1]] = value
    return True


def _set_state_action_variants(
    record: Mapping[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Apply bounded renderer-supported setState actions as capture states."""

    spec = record.get("genui_json")
    if not isinstance(spec, Mapping):
        return []
    elements = spec.get("elements")
    state = spec.get("state")
    if not isinstance(elements, Mapping) or not isinstance(state, Mapping):
        return []
    changes: list[tuple[str, str, str, Any]] = []
    for element_id, element in sorted(elements.items()):
        if not isinstance(element, Mapping):
            continue
        handlers = element.get("on")
        if not isinstance(handlers, Mapping):
            continue
        for event_name, action in sorted(handlers.items()):
            if not isinstance(action, Mapping):
                continue
            if str(action.get("action") or "").casefold() != "setstate":
                continue
            params = action.get("params")
            if not isinstance(params, Mapping):
                continue
            path = str(params.get("path") or "")
            if not path.startswith("/") or "value" not in params:
                continue
            changes.append(
                (
                    str(element_id),
                    str(event_name),
                    path,
                    params["value"],
                )
            )
    variants: list[tuple[str, str, dict[str, Any]]] = []
    for element_id, event_name, path, value in changes:
        copied = json.loads(_canonical_json(record))
        copied_state = copied.get("genui_json", {}).get("state")
        if not isinstance(copied_state, dict):
            continue
        if not _set_json_pointer(copied_state, path, value):
            continue
        variants.append(
            (
                f"internal_{len(variants) + 1}",
                (
                    f"Renderer setState action from element {element_id} "
                    f"event {event_name} applied to {path}"
                ),
                copied,
            )
        )
        if len(variants) == 4:
            break
    return variants


def _append_set_state_capture_inputs(
    benchmark_dir: Path,
    root: Path,
    inputs: list[dict[str, Any]],
) -> bool:
    if any(
        str(row.get("state_id") or "").startswith("internal_")
        for row in inputs
    ):
        return False
    records = _read_jsonl(benchmark_dir / "selected_genui.jsonl")
    variants: dict[int, list[dict[str, Any]]] = defaultdict(list)
    descriptions: dict[int, dict[str, str]] = defaultdict(dict)
    for record in records:
        for index, (_, description, variant) in enumerate(
            _set_state_action_variants(record),
            start=1,
        ):
            variants[index].append(variant)
            descriptions[index][str(record["ui_id"])] = description
    for index in sorted(variants):
        path = root / f"internal_state_{index}.jsonl"
        _write_jsonl(path, variants[index])
        description_path = (
            root / f"internal_state_{index}_descriptions.json"
        )
        _atomic_write_json(description_path, descriptions[index])
        inputs.append(
            {
                "state_id": f"internal_{index}",
                "state_description": "Alternate renderer internal state",
                "description_path": str(description_path),
                "path": str(path),
                "record_count": len(variants[index]),
                "sha256": _hash_file(path),
                "profiles": ["compact"],
                "required": True,
            }
        )
    return bool(variants)


def _materialize_capture_inputs(
    benchmark_dir: Path,
) -> list[dict[str, Any]]:
    root = benchmark_dir / "capture_inputs"
    manifest_path = root / "capture_input_manifest.jsonl"
    if manifest_path.exists():
        rows = _read_jsonl(manifest_path)
        rebased = False
        for row in rows:
            for key in ("path", "description_path"):
                raw = row.get(key)
                if not raw or Path(str(raw)).exists():
                    continue
                candidate = root / Path(str(raw)).name
                if not candidate.exists():
                    raise FileNotFoundError(raw)
                row[key] = str(candidate)
                rebased = True
            input_path = Path(str(row["path"]))
            if _hash_file(input_path) != row["sha256"]:
                raise ValueError(
                    f"capture input hash mismatch after relocation: {input_path}"
                )
        extended = _append_set_state_capture_inputs(
            benchmark_dir,
            root,
            rows,
        )
        if rebased or extended:
            _write_jsonl(manifest_path, rows)
        return rows
    root.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(benchmark_dir / "selected_genui.jsonl")
    inputs: list[dict[str, Any]] = []

    base_path = root / "initial.jsonl"
    _write_jsonl(base_path, records)
    inputs.append(
        {
            "state_id": "initial",
            "state_description": "Renderer default initial state",
            "path": str(base_path),
            "record_count": len(records),
            "sha256": _hash_file(base_path),
            "profiles": [profile["name"] for profile in PROFILES],
            "required": True,
        }
    )
    variants: dict[int, list[dict[str, Any]]] = defaultdict(list)
    descriptions: dict[int, dict[str, str]] = defaultdict(dict)
    for record in records:
        for index, (state_id, description, variant) in enumerate(
            _tabs_state_variants(record),
            start=1,
        ):
            variants[index].append(variant)
            descriptions[index][str(record["ui_id"])] = description
    for index in sorted(variants):
        path = root / f"tabs_state_{index}.jsonl"
        _write_jsonl(path, variants[index])
        description_path = root / f"tabs_state_{index}_descriptions.json"
        _atomic_write_json(description_path, descriptions[index])
        inputs.append(
            {
                "state_id": f"tabs_{index}",
                "state_description": "Alternate renderer Tabs state",
                "description_path": str(description_path),
                "path": str(path),
                "record_count": len(variants[index]),
                "sha256": _hash_file(path),
                "profiles": ["compact"],
                "required": True,
            }
        )
    _append_set_state_capture_inputs(benchmark_dir, root, inputs)
    _write_jsonl(manifest_path, inputs)
    return inputs


def _installed_apk_hash(
    adb: Adb,
    package: str,
    scratch_dir: Path,
) -> tuple[str, str]:
    path_output = adb.shell("pm", "path", package)
    remote_paths = [
        line.removeprefix("package:").strip()
        for line in path_output.splitlines()
        if line.startswith("package:")
    ]
    base_path = next(
        (path for path in remote_paths if path.endswith("/base.apk")),
        remote_paths[0] if remote_paths else "",
    )
    if not base_path:
        raise RuntimeError(f"package is not installed: {package}")
    local = scratch_dir / "installed_base.apk"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    adb.run(["pull", base_path, str(local)], timeout=180)
    return _hash_file(local), base_path


def _package_snapshot(adb: Adb, package: str) -> dict[str, Any]:
    paths = adb.shell("pm", "path", package, check=False).splitlines()
    dumpsys = adb.shell("dumpsys", "package", package, check=False)
    selected = [
        line.strip()
        for line in dumpsys.splitlines()
        if any(
            key in line
            for key in (
                "versionCode=",
                "versionName=",
                "firstInstallTime=",
                "lastUpdateTime=",
                "dataDir=",
            )
        )
    ]
    return {
        "package": package,
        "apk_paths": sorted(paths),
        "package_identity_lines": selected,
    }


def _write_provenance(
    benchmark_dir: Path,
    adb: Adb,
    *,
    package: str,
    apk_path: Path,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    destination = benchmark_dir / "native_provenance.json"
    if destination.exists():
        value = json.loads(destination.read_text(encoding="utf-8"))
        if value.get("local_apk_sha256") != _hash_file(apk_path):
            raise ValueError("capture APK changed after provenance was frozen")
        return value
    renderer_hash, renderer_inventory = _renderer_source_fingerprint()
    installed_hash, installed_path = _installed_apk_hash(
        adb, package, benchmark_dir / "capture_scratch"
    )
    local_hash = _hash_file(apk_path)
    schemas = [
        REPO_ROOT / "dataset" / "schema" / "genui_flatspec.schema.json",
        REPO_ROOT
        / "dataset"
        / "schema"
        / "expected_ui_contract.schema.json",
    ]
    value = {
        "schema_version": "genui_native_capture_provenance.v2",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain=v1"),
        "package": package,
        "coexists_with_data_bearing_package": "com.samsung.genuicraft",
        "local_apk_path": str(apk_path),
        "local_apk_sha256": local_hash,
        "installed_apk_path": installed_path,
        "installed_apk_sha256": installed_hash,
        "apk_install_parity": local_hash == installed_hash,
        "renderer_source_fingerprint": renderer_hash,
        "renderer_source_inventory": renderer_inventory,
        "schema_hashes": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): _hash_file(
                path
            )
            for path in schemas
        },
        "device": {
            "serial": adb.serial,
            "manufacturer": adb.shell("getprop", "ro.product.manufacturer"),
            "model": adb.shell("getprop", "ro.product.model"),
            "device": adb.shell("getprop", "ro.product.device"),
            "sdk": adb.shell("getprop", "ro.build.version.sdk"),
            "release": adb.shell("getprop", "ro.build.version.release"),
            "build_fingerprint": adb.shell(
                "getprop", "ro.build.fingerprint"
            ),
        },
        "display_baseline": dict(baseline),
        "parity_established": local_hash == installed_hash,
    }
    _atomic_write_json(destination, value)
    return value


def _remote_exists(adb: Adb, path: str) -> bool:
    output = adb.shell(
        "sh",
        "-c",
        f'if [ -f "{path}" ]; then echo yes; else echo no; fi',
        check=False,
    )
    return output.strip().endswith("yes")


def _private_exists(adb: Adb, package: str, relative_path: str) -> bool:
    result = adb.run(
        [
            "shell",
            "run-as",
            package,
            "test",
            "-f",
            relative_path,
        ],
        check=False,
    )
    return result.returncode == 0


def _pull_private_directory(
    adb: Adb,
    *,
    package: str,
    relative_directory: str,
    local_destination: Path,
) -> str:
    local_destination.parent.mkdir(parents=True, exist_ok=True)
    archive = local_destination.parent / f".{local_destination.name}.tar"
    extract_root = local_destination.parent / (
        f".{local_destination.name}.extract"
    )
    for scratch in (archive,):
        if scratch.exists():
            scratch.unlink()
    if extract_root.exists():
        resolved_extract = extract_root.resolve()
        if not resolved_extract.is_relative_to(
            local_destination.parent.resolve()
        ):
            raise ValueError("unsafe private-output extraction path")
        shutil.rmtree(extract_root)
    parent = str(Path(relative_directory).parent).replace("\\", "/")
    name = Path(relative_directory).name
    command = [
        adb.executable,
        "-s",
        adb.serial,
        "exec-out",
        "run-as",
        package,
        "tar",
        "-cf",
        "-",
        "-C",
        parent,
        name,
    ]
    with archive.open("wb") as output:
        result = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            "failed to pull private capture output: "
            + result.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    extract_root.mkdir()
    with tarfile.open(archive, "r:") as bundle:
        for member in bundle.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("unsafe path in private capture archive")
        bundle.extractall(extract_root, filter="data")
    extracted = extract_root / name
    if not extracted.is_dir():
        raise RuntimeError(
            f"private capture archive lacks expected directory: {name}"
        )
    if local_destination.exists():
        raise FileExistsError(local_destination)
    extracted.rename(local_destination)
    archive.unlink()
    extract_root.rmdir()
    return "private run-as tar pull completed"


def _capture_remote_batch(
    adb: Adb,
    *,
    package: str,
    local_input: Path,
    remote_key: str,
    local_destination: Path,
    settle_ms: int,
    capture_full_height: bool,
) -> tuple[bool, str]:
    private_root = adb.shell("run-as", package, "pwd")
    private_relative_input = (
        f"files/codex_judge_v2/inputs/{remote_key}.jsonl"
    )
    remote_input = f"{private_root}/{private_relative_input}"
    transfer_input = f"/data/local/tmp/{package}.{remote_key}.jsonl"
    private_relative_output = (
        f"files/codex_judge_v2/outputs/{remote_key}"
    )
    remote_output = f"{private_root}/{private_relative_output}"
    if _private_exists(
        adb,
        package,
        f"{private_relative_output}/_COMPLETE.json",
    ):
        message = _pull_private_directory(
            adb,
            package=package,
            relative_directory=private_relative_output,
            local_destination=local_destination,
        )
        return True, "recovered completed remote batch; " + message
    adb.shell(
        "run-as",
        package,
        "mkdir",
        "-p",
        "files/codex_judge_v2/inputs",
    )
    adb.shell(
        "run-as",
        package,
        "rm",
        "-rf",
        private_relative_output,
    )
    adb.run(["push", str(local_input), transfer_input], timeout=180)
    try:
        adb.shell(
            "run-as",
            package,
            "cp",
            transfer_input,
            private_relative_input,
        )
    finally:
        adb.shell("rm", "-f", transfer_input, check=False)
    adb.shell("am", "force-stop", package)
    component = f"{package}/{ACTIVITY}"
    result = adb.shell(
        "am",
        "start",
        "-W",
        "-n",
        component,
        "--es",
        "input_jsonl_path",
        remote_input,
        "--es",
        "output_dir",
        remote_output,
        "--ei",
        "start_index",
        "0",
        "--ei",
        "max_count",
        "-1",
        "--el",
        "settle_ms",
        str(settle_ms),
        "--ez",
        "capture_full_height",
        "true" if capture_full_height else "false",
        timeout=60,
        check=False,
    )
    if "Error:" in result or "Exception" in result:
        return False, result[-2000:]
    deadline = time.monotonic() + 60.0 * max(
        2.0, len(_read_jsonl(local_input)) / 4.0
    )
    while time.monotonic() < deadline:
        if _private_exists(
            adb,
            package,
            f"{private_relative_output}/_COMPLETE.json",
        ):
            message = _pull_private_directory(
                adb,
                package=package,
                relative_directory=private_relative_output,
                local_destination=local_destination,
            )
            return True, message
        if _private_exists(
            adb,
            package,
            f"{private_relative_output}/_ERROR.txt",
        ):
            return False, f"capture activity wrote {remote_output}/_ERROR.txt"
        time.sleep(2.0)
    return False, "timed out waiting for capture completion"


def _capture_batches(
    benchmark_dir: Path,
    adb: Adb,
    *,
    package: str,
    inputs: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    settle_ms: int,
) -> None:
    progress_path = benchmark_dir / "native_capture_progress.jsonl"
    progress = _read_jsonl(progress_path) if progress_path.exists() else []
    completed_keys = {
        str(row["batch_key"])
        for row in progress
        if row.get("status") == "complete"
    }
    for profile in profiles:
        effective = _apply_profile(adb, profile, baseline)
        for input_row in inputs:
            if profile["name"] not in input_row["profiles"]:
                continue
            state_id = str(input_row["state_id"])
            records = _read_jsonl(Path(input_row["path"]))
            for batch_index, start in enumerate(
                range(0, len(records), BATCH_SIZE)
            ):
                rows = records[start : start + BATCH_SIZE]
                batch_key = (
                    f"{profile['name']}__{state_id}__"
                    f"{batch_index:04d}"
                )
                if batch_key in completed_keys:
                    continue
                batch_input_dir = benchmark_dir / "capture_inputs" / "batches"
                batch_input = batch_input_dir / f"{batch_key}.jsonl"
                if not batch_input.exists():
                    _write_jsonl(batch_input, rows)
                elif _hash_file(batch_input) != _hash_bytes(
                    (
                        "".join(_canonical_json(row) + "\n" for row in rows)
                    ).encode()
                ):
                    raise ValueError(f"immutable batch input drift: {batch_input}")
                destination = (
                    benchmark_dir
                    / "native_capture"
                    / "batches"
                    / profile["name"]
                    / state_id
                    / f"batch_{batch_index:04d}"
                )
                host_complete = destination / "_HOST_COMPLETE.json"
                if host_complete.exists():
                    completed_keys.add(batch_key)
                    continue
                last_error = ""
                capture_full_height = (
                    profile["name"] == "compact" and state_id == "initial"
                )
                for attempt in range(1, MAX_INFRASTRUCTURE_ATTEMPTS + 1):
                    temporary_destination = destination.with_name(
                        destination.name + f".attempt_{attempt}"
                    )
                    if temporary_destination.exists():
                        resolved_temporary = temporary_destination.resolve()
                        if not resolved_temporary.is_relative_to(
                            benchmark_dir.resolve()
                        ):
                            raise ValueError(
                                "refusing to remove capture scratch outside "
                                "the benchmark directory"
                            )
                        shutil.rmtree(temporary_destination)
                    ok, message = _capture_remote_batch(
                        adb,
                        package=package,
                        local_input=batch_input,
                        remote_key=f"{batch_key}_attempt_{attempt}",
                        local_destination=temporary_destination,
                        settle_ms=settle_ms,
                        capture_full_height=capture_full_height,
                    )
                    if ok:
                        if destination.exists():
                            raise FileExistsError(destination)
                        temporary_destination.rename(destination)
                        _atomic_write_json(
                            host_complete,
                            {
                                "batch_key": batch_key,
                                "attempt": attempt,
                                "profile": effective,
                                "state_id": state_id,
                                "capture_full_height": capture_full_height,
                                "input_sha256": _hash_file(batch_input),
                                "completed_at": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                        progress.append(
                            {
                                "batch_key": batch_key,
                                "status": "complete",
                                "attempt": attempt,
                                "record_count": len(rows),
                                "completed_at": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            }
                        )
                        _write_jsonl(progress_path, progress)
                        completed_keys.add(batch_key)
                        break
                    last_error = message
                    progress.append(
                        {
                            "batch_key": batch_key,
                            "status": "infrastructure_retry",
                            "attempt": attempt,
                            "error": last_error,
                            "record_count": len(rows),
                            "recorded_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    )
                    _write_jsonl(progress_path, progress)
                else:
                    raise RuntimeError(
                        f"capture infrastructure failed three times for "
                        f"{batch_key}: {last_error}"
                    )


def _consolidate_manifest(
    benchmark_dir: Path,
    inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    description_by_state: dict[str, dict[str, str]] = {}
    required_by_state: dict[str, bool] = {}
    for row in inputs:
        state_id = str(row["state_id"])
        required_by_state[state_id] = bool(row["required"])
        path = row.get("description_path")
        description_by_state[state_id] = (
            json.loads(Path(path).read_text(encoding="utf-8"))
            if path
            else {}
        )
    rows: list[dict[str, Any]] = []
    for host_complete in sorted(
        (benchmark_dir / "native_capture" / "batches").rglob(
            "_HOST_COMPLETE.json"
        )
    ):
        host = json.loads(host_complete.read_text(encoding="utf-8"))
        batch_dir = host_complete.parent
        activity_manifest = batch_dir / "capture_manifest.jsonl"
        if not activity_manifest.exists():
            raise FileNotFoundError(activity_manifest)
        profile = host["profile"]
        state_id = str(host["state_id"])
        for activity_row in _read_jsonl(activity_manifest):
            ui_id = str(activity_row["ui_id"])
            native_render_ok = bool(activity_row.get("native_render_ok"))
            candidate_failure = not native_render_ok
            common = {
                "ui_id": ui_id,
                "query_id": activity_row.get("query_id"),
                "response_id": activity_row.get("response_id"),
                "viewport_profile": profile["name"],
                "viewport_width_dp": profile.get("effective_width_dp"),
                "viewport_height_dp": profile.get("effective_height_dp"),
                "density_dpi": profile.get("effective_density_dpi"),
                "state_id": state_id,
                "state_description": description_by_state.get(
                    state_id, {}
                ).get(
                    ui_id,
                    "Renderer default initial state"
                    if state_id == "initial"
                    else "Alternate renderer state",
                ),
                "required": required_by_state.get(state_id, True),
                "native_render_ok": native_render_ok,
                "native_render_error": activity_row.get(
                    "native_render_error"
                ),
                "failure_class": (
                    "candidate" if candidate_failure else None
                ),
                "batch_key": host["batch_key"],
                "capture_attempt": host["attempt"],
            }
            captures = [
                ("initial_viewport", "viewport_screenshot", "viewport_ok")
            ]
            if bool(activity_row.get("full_height_attempted")):
                captures.append(("full_height", "screenshot", "ok"))
            for capture_kind, key, ok_key in captures:
                local = batch_dir / str(activity_row[key])
                image_captured = local.is_file() and (
                    bool(activity_row.get(ok_key))
                    if capture_kind == "initial_viewport"
                    else True
                )
                capture_ok = (
                    image_captured and native_render_ok
                )
                width_px: int | None = None
                height_px: int | None = None
                if image_captured:
                    with Image.open(local) as image:
                        width_px, height_px = image.size
                rows.append(
                    {
                        **common,
                        "capture_kind": capture_kind,
                        "ok": capture_ok,
                        "image_captured": image_captured,
                        "local_path": str(local.resolve()),
                        "screenshot_sha256": (
                            _hash_file(local)
                            if local.exists()
                            else None
                        ),
                        "width_px": width_px,
                        "height_px": height_px,
                        "tile_count": (
                            activity_row.get("tile_count")
                            if capture_kind == "full_height"
                            else 1
                        ),
                    }
                )
    manifest_path = benchmark_dir / "native_capture_manifest.jsonl"
    _write_jsonl(manifest_path, rows)
    return rows


def _capture_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["ui_id"]),
        str(row["viewport_profile"]),
        str(row["state_id"]),
        str(row["capture_kind"]),
    )


def _expected_capture_rows(
    inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for input_row in inputs:
        state_id = str(input_row["state_id"])
        for record in _read_jsonl(Path(input_row["path"])):
            ui_id = str(record["ui_id"])
            for profile in input_row["profiles"]:
                kinds = ["initial_viewport"]
                if profile == "compact" and state_id == "initial":
                    kinds.append("full_height")
                for kind in kinds:
                    expected.append(
                        {
                            "ui_id": ui_id,
                            "viewport_profile": profile,
                            "state_id": state_id,
                            "capture_kind": kind,
                            "required": bool(input_row["required"]),
                        }
                    )
    return expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--apk-path", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument("--settle-ms", type=int, default=900)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    benchmark_dir = Path(args.benchmark_dir).resolve()
    apk_path = Path(args.apk_path).resolve()
    if not apk_path.exists():
        raise FileNotFoundError(apk_path)
    if args.package == DATA_BEARING_PACKAGE:
        raise ValueError(
            "refusing to use the data-bearing production package for capture"
        )
    adb = Adb(args.adb, args.serial)
    state = adb.run(["get-state"]).stdout.strip()
    if state != "device":
        raise RuntimeError(
            f"selected Android target is not ready: {args.serial} {state}"
        )
    baseline = _display_baseline(adb)
    guard_path = benchmark_dir / "data_bearing_app_guard.json"
    production_before = _package_snapshot(adb, DATA_BEARING_PACKAGE)
    if guard_path.exists():
        prior_guard = json.loads(guard_path.read_text(encoding="utf-8"))
        if prior_guard.get("before") != production_before:
            raise RuntimeError(
                "data-bearing app identity changed since capture started"
            )
    else:
        _atomic_write_json(
            guard_path,
            {
                "schema_version": "genui_data_bearing_app_guard.v2",
                "before": production_before,
                "verified_untouched": None,
            },
        )
    provenance = _write_provenance(
        benchmark_dir,
        adb,
        package=args.package,
        apk_path=apk_path,
        baseline=baseline,
    )
    if not provenance["parity_established"]:
        raise RuntimeError("installed APK does not match the checkout build")
    inputs = _materialize_capture_inputs(benchmark_dir)
    try:
        _capture_batches(
            benchmark_dir,
            adb,
            package=args.package,
            inputs=inputs,
            profiles=PROFILES,
            baseline=baseline,
            settle_ms=args.settle_ms,
        )
    finally:
        _restore_display(adb, baseline)
    manifest = _consolidate_manifest(benchmark_dir, inputs)
    production_after = _package_snapshot(adb, DATA_BEARING_PACKAGE)
    production_untouched = production_after == production_before
    _atomic_write_json(
        guard_path,
        {
            "schema_version": "genui_data_bearing_app_guard.v2",
            "before": production_before,
            "after": production_after,
            "verified_untouched": production_untouched,
        },
    )
    if not production_untouched:
        raise RuntimeError(
            "data-bearing app package identity changed during capture"
        )
    expected = _expected_capture_rows(inputs)
    actual_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in manifest:
        actual_counts[_capture_key(row)] += 1
    duplicates = [
        {
            "ui_id": key[0],
            "viewport_profile": key[1],
            "state_id": key[2],
            "capture_kind": key[3],
            "count": count,
        }
        for key, count in sorted(actual_counts.items())
        if count != 1
    ]
    missing = [
        row
        for row in expected
        if actual_counts.get(_capture_key(row), 0) == 0
    ]
    _write_jsonl(
        benchmark_dir / "native_capture_missing.jsonl",
        missing,
    )
    _write_jsonl(
        benchmark_dir / "native_capture_duplicates.jsonl",
        duplicates,
    )
    required_failures = [
        row
        for row in manifest
        if row["required"] and not row["image_captured"]
    ]
    summary = {
        "capture_row_count": len(manifest),
        "expected_capture_row_count": len(expected),
        "successful_capture_count": sum(bool(row["ok"]) for row in manifest),
        "required_failure_count": len(required_failures),
        "missing_capture_count": len(missing),
        "duplicate_capture_key_count": len(duplicates),
        "candidate_render_failure_count": sum(
            row["failure_class"] == "candidate" for row in manifest
        ),
        "manifest": str(
            benchmark_dir / "native_capture_manifest.jsonl"
        ),
        "display_restored": True,
        "data_bearing_package_untouched": True,
        "internal_state_policy": (
            "Up to four alternate Tabs selections are captured. The current "
            "native Modal renderer exposes trigger and content together in "
            "the initial state, so it has no separate open-state capture."
        ),
    }
    _atomic_write_json(
        benchmark_dir / "native_capture_summary.json", summary
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not required_failures and not missing and not duplicates else 2


if __name__ == "__main__":
    raise SystemExit(main())
