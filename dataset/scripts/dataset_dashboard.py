#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import getpass
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dataset_dashboard.sources.json"
DEFAULT_EXAMPLE_CONFIG = ROOT / "configs" / "dataset_dashboard.sources.example.json"
DEFAULT_MIRROR_DIR = ROOT / "data" / "dashboard_mirror"
MANIFEST_NAME = "sync_manifest.json"
PASSWORD_CACHE: dict[str, str] = {}
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class FileEntry:
    rel: str
    size: int
    mtime: float
    source_path: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp.replace(path)


def resolve_dataset_path(raw: str | None, base: Path = ROOT) -> Path:
    if not raw:
        return base
    expanded = os.path.expandvars(os.path.expanduser(raw))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return base / path


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def read_first_jsonl(path: Path, limit: int = 2000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def safe_source_id(source_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in source_id.strip())
    return cleaned or "source"


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def match_any(rel: str, patterns: list[str]) -> bool:
    rel = rel.replace("\\", "/")
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def match_filter_path(rel: str, patterns: list[str]) -> bool:
    rel = rel.replace("\\", "/")
    candidates = [rel]
    if "/" in rel:
        candidates.append(rel.split("/", 1)[1])
    return any(match_any(candidate, patterns) for candidate in candidates)


def should_include(rel: str, include_globs: list[str], exclude_globs: list[str]) -> bool:
    rel = rel.replace("\\", "/")
    if exclude_globs and match_filter_path(rel, exclude_globs):
        return False
    return not include_globs or match_filter_path(rel, include_globs)


def source_path_mode(source: dict[str, Any]) -> str:
    mode = str(source.get("path_match") or "").strip().lower()
    if mode in {"literal", "glob", "regex"}:
        return mode
    raw = str(source.get("path") or "")
    return "glob" if glob.has_magic(os.path.expandvars(os.path.expanduser(raw))) else "literal"


def root_label(root: Path) -> str:
    return root.name or safe_source_id(str(root))


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def ssh_password(source: dict[str, Any]) -> str:
    return str(source.get("password") or "").strip()


def ssh_cache_key(source: dict[str, Any]) -> str:
    return f"{source.get('user') or ''}@{source.get('host') or ''}:{source.get('port') or 22}"


def source_asks_password(source: dict[str, Any]) -> bool:
    return bool(source.get("ask_password") or source.get("prompt_password"))


def password_for_source(source: dict[str, Any]) -> str:
    password = ssh_password(source)
    if password:
        return password
    if not source_asks_password(source):
        return ""
    key = ssh_cache_key(source)
    if key not in PASSWORD_CACHE:
        PASSWORD_CACHE[key] = getpass.getpass(f"Password for {key}: ")
    return PASSWORD_CACHE[key]


def use_paramiko_ssh(source: dict[str, Any]) -> bool:
    backend = str(source.get("ssh_backend") or "").strip().lower()
    if backend == "paramiko":
        return True
    if backend == "openssh":
        return False
    if source.get("proxy_command") or source.get("proxy_jump"):
        return False
    return bool(ssh_password(source) or source_asks_password(source))


def ssh_transport_mode(source: dict[str, Any]) -> str:
    if not ssh_password(source):
        return "openssh"
    if shutil.which("sshpass"):
        return "sshpass"
    raise RuntimeError(
        f"SSH source {source.get('id')} uses password auth, but sshpass was not found. "
        "Install sshpass or use identity_file key auth."
    )


def ssh_target(source: dict[str, Any]) -> str:
    host = str(source.get("host") or "").strip()
    if not host:
        raise ValueError(f"SSH source {source.get('id')} missing host")
    user = str(source.get("user") or "").strip()
    return f"{user}@{host}" if user else host


def normalized_ssh_options(source: dict[str, Any]) -> list[str]:
    options: list[str] = []
    raw_options = source.get("ssh_options", [])
    if isinstance(raw_options, dict):
        for key, value in raw_options.items():
            if value is None or value == "":
                continue
            options.append(f"{key}={value}")
    elif isinstance(raw_options, list):
        for option in raw_options:
            text = str(option).strip()
            if text:
                options.append(text)
    elif raw_options:
        text = str(raw_options).strip()
        if text:
            options.append(text)

    proxy_jump = str(source.get("proxy_jump") or "").strip()
    if proxy_jump:
        options.append(f"ProxyJump={proxy_jump}")
    proxy_command = str(source.get("proxy_command") or "").strip()
    if proxy_command:
        options.append(f"ProxyCommand={proxy_command}")
    return options


def run_command(command: list[str] | str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    shell = isinstance(command, str)
    return subprocess.run(
        command,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def import_paramiko() -> Any:
    try:
        import paramiko  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Password-based SSH sync requires the optional Python package 'paramiko'. "
            "Install it with: python -m pip install paramiko"
        ) from exc
    return paramiko


def connect_paramiko(source: dict[str, Any]) -> Any:
    paramiko = import_paramiko()
    host = str(source.get("host") or "").strip()
    if not host:
        raise ValueError(f"SSH source {source.get('id')} missing host")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    identity = str(source.get("identity_file") or "").strip()
    connect_kwargs: dict[str, Any] = {
        "hostname": host,
        "username": str(source.get("user") or "").strip() or None,
        "port": int(source.get("port") or 22),
        "timeout": int(source.get("connect_timeout_sec", 30)),
        "look_for_keys": False,
        "allow_agent": False,
    }
    password = password_for_source(source)
    if password:
        connect_kwargs["password"] = password
    if identity:
        connect_kwargs["key_filename"] = os.path.expandvars(os.path.expanduser(identity))
    client.connect(**connect_kwargs)
    return client


def source_enabled(source: dict[str, Any]) -> bool:
    return bool(source.get("enabled", True))


def source_label(source: dict[str, Any]) -> str:
    return str(source.get("label") or source.get("id") or "source")


def source_local_root(source: dict[str, Any], mirror_dir: Path) -> Path:
    source_id = safe_source_id(str(source.get("id") or source_label(source)))
    if (
        source.get("type") == "local"
        and source.get("mirror_local_in_place", False)
        and source_path_mode(source) == "literal"
    ):
        return resolve_dataset_path(str(source.get("path") or ""), ROOT)
    return mirror_dir / source_id


def list_local_files(
    root: Path,
    include_globs: list[str],
    exclude_globs: list[str],
    rel_prefix: str = "",
) -> list[FileEntry]:
    if not root.exists():
        return []
    entries: list[FileEntry] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = rel_posix(path, root)
        if rel_prefix:
            rel = f"{rel_prefix.strip('/')}/{rel}"
        if not should_include(rel, include_globs, exclude_globs):
            continue
        stat = path.stat()
        entries.append(FileEntry(rel=rel, size=stat.st_size, mtime=stat.st_mtime, source_path=str(path)))
    return entries


def regex_local_roots(source: dict[str, Any]) -> list[Path]:
    pattern = str(source.get("path") or "").strip()
    base_raw = str(source.get("path_base") or "").strip()
    if not pattern:
        return []
    if not base_raw:
        raise ValueError(f"local source {source.get('id')} uses path_match=regex but missing path_base")
    base = resolve_dataset_path(base_raw, ROOT)
    if not base.exists():
        return []
    compiled = re.compile(pattern)
    roots: list[Path] = []
    for candidate in base.iterdir():
        if not candidate.is_dir():
            continue
        if compiled.search(candidate.name):
            roots.append(candidate)
    return sorted(roots, key=lambda p: p.as_posix())


def list_local_source_files(source: dict[str, Any], include_globs: list[str], exclude_globs: list[str]) -> list[FileEntry]:
    raw_path = str(source.get("path") or "")
    mode = source_path_mode(source)
    if mode == "literal":
        return list_local_files(resolve_dataset_path(raw_path, ROOT), include_globs, exclude_globs)
    if mode == "glob":
        pattern = str(resolve_dataset_path(raw_path, ROOT))
        roots = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]
    elif mode == "regex":
        roots = regex_local_roots(source)
    else:
        raise ValueError(f"Unsupported path_match for local source: {mode}")
    entries: list[FileEntry] = []
    for root in sorted(roots, key=lambda p: p.as_posix()):
        entries.extend(list_local_files(root, include_globs, exclude_globs, rel_prefix=root_label(root)))
    return entries


def ssh_base_command(source: dict[str, Any]) -> list[str]:
    target = ssh_target(source)
    mode = ssh_transport_mode(source)
    password = ssh_password(source)
    if mode == "sshpass":
        cmd = ["sshpass", "-p", password, "ssh"]
    else:
        cmd = ["ssh"]
    port = source.get("port")
    if port:
        cmd += ["-p", str(port)]
    identity = str(source.get("identity_file") or "").strip()
    if identity:
        cmd += ["-i", os.path.expandvars(os.path.expanduser(identity))]
    for option in normalized_ssh_options(source):
        cmd += ["-o", str(option)]
    cmd.append(target)
    return cmd


def scp_base_command(source: dict[str, Any]) -> list[str]:
    mode = ssh_transport_mode(source)
    password = ssh_password(source)
    if mode == "sshpass":
        cmd = ["sshpass", "-p", password, "scp", "-p"]
    else:
        cmd = ["scp", "-p"]
    port = source.get("port")
    if port:
        cmd += ["-P", str(port)]
    identity = str(source.get("identity_file") or "").strip()
    if identity:
        cmd += ["-i", os.path.expandvars(os.path.expanduser(identity))]
    for option in normalized_ssh_options(source):
        cmd += ["-o", str(option)]
    return cmd


def list_ssh_files(
    source: dict[str, Any],
    include_globs: list[str],
    exclude_globs: list[str],
    client: Any | None = None,
) -> list[FileEntry]:
    remote_root = str(source.get("path") or "").rstrip("/")
    if not remote_root:
        raise ValueError(f"SSH source {source.get('id')} missing path")
    mode = source_path_mode(source)
    path_base = str(source.get("path_base") or "").rstrip("/")
    if mode == "regex" and not path_base:
        raise ValueError(f"SSH source {source.get('id')} uses path_match=regex but missing path_base")
    py = r"""
import os, sys, json, fnmatch, glob, re
root=sys.argv[1]
mode=sys.argv[2]
path_base=sys.argv[3]
include=json.loads(sys.argv[4])
exclude=json.loads(sys.argv[5])
def match_any(rel, patterns):
    return any(fnmatch.fnmatch(rel, p) for p in patterns)
def match_filter_path(rel, patterns):
    candidates=[rel]
    if '/' in rel:
        candidates.append(rel.split('/', 1)[1])
    return any(match_any(candidate, patterns) for candidate in candidates)
def roots_for_mode():
    if mode == 'literal':
        return [(root, '')] if os.path.isdir(root) else []
    if mode == 'glob':
        return [(p, os.path.basename(os.path.normpath(p))) for p in sorted(glob.glob(root)) if os.path.isdir(p)]
    if mode == 'regex':
        compiled=re.compile(root)
        matches=[]
        for name in os.listdir(path_base):
            path=os.path.join(path_base, name)
            if os.path.isdir(path) and compiled.search(name):
                matches.append((path, os.path.basename(os.path.normpath(path))))
        return sorted(matches)
    raise SystemExit('Unsupported path_match: ' + mode)
for scan_root, prefix in roots_for_mode():
    for base, dirs, files in os.walk(scan_root):
        dirs[:] = [d for d in dirs if d not in {'.git','__pycache__'}]
        for name in files:
            path=os.path.join(base,name)
            rel=os.path.relpath(path, scan_root).replace(os.sep, '/')
            if prefix:
                rel=prefix.rstrip('/') + '/' + rel
            if exclude and match_filter_path(rel, exclude):
                continue
            if include and not match_filter_path(rel, include):
                continue
            try:
                st=os.stat(path)
            except OSError:
                continue
            print(json.dumps({'rel': rel, 'size': st.st_size, 'mtime': st.st_mtime, 'source_path': path}, separators=(',',':')))
"""
    remote_command = " ".join(
        [
            "python3",
            "-c",
            shell_quote(py),
            shell_quote(remote_root),
            shell_quote(mode),
            shell_quote(path_base),
            shell_quote(json.dumps(include_globs)),
            shell_quote(json.dumps(exclude_globs)),
        ]
    )
    if client is not None:
        stdin, stdout, stderr = client.exec_command(remote_command, timeout=int(source.get("list_timeout_sec", 300)))
        del stdin
        output = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            raise RuntimeError(error.strip() or output.strip() or "ssh list failed")
    else:
        cmd = ssh_base_command(source) + [remote_command]
        result = run_command(cmd, timeout=int(source.get("list_timeout_sec", 300)))
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ssh list failed")
        output = result.stdout
    entries: list[FileEntry] = []
    for line in output.splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        entries.append(
            FileEntry(
                rel=str(obj["rel"]),
                size=int(obj["size"]),
                mtime=float(obj["mtime"]),
                source_path=str(obj.get("source_path") or ""),
            )
        )
    return entries


def list_command_files(source: dict[str, Any], include_globs: list[str], exclude_globs: list[str]) -> list[FileEntry]:
    command = str(source.get("list_command") or "").strip()
    if not command:
        raise ValueError(f"command source {source.get('id')} missing list_command")
    env = os.environ.copy()
    env["A2UI_DASHBOARD_INCLUDE_GLOBS"] = json.dumps(include_globs)
    env["A2UI_DASHBOARD_EXCLUDE_GLOBS"] = json.dumps(exclude_globs)
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=int(source.get("list_timeout_sec", 300)),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "list_command failed")
    entries: list[FileEntry] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rel = str(obj["rel"])
        if should_include(rel, include_globs, exclude_globs):
            entries.append(FileEntry(rel=rel, size=int(obj.get("size", 0)), mtime=float(obj.get("mtime", 0))))
    return entries


def copy_local_file(source_root: Path, dest_root: Path, entry: FileEntry) -> None:
    src = Path(entry.source_path) if entry.source_path else source_root / entry.rel
    dest = dest_root / entry.rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_ssh_file(source: dict[str, Any], dest_root: Path, entry: FileEntry) -> None:
    remote_root = str(source.get("path") or "").rstrip("/")
    target = ssh_target(source)
    remote_path = entry.source_path or str(PurePosixPath(remote_root) / PurePosixPath(entry.rel))
    dest = dest_root / entry.rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote_spec = f"{target}:{shell_quote(remote_path)}"
    cmd = scp_base_command(source) + [remote_spec, str(dest)]
    result = run_command(cmd, timeout=int(source.get("copy_timeout_sec", 300)))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"scp failed for {entry.rel}")


def copy_paramiko_file(sftp: Any, dest_root: Path, entry: FileEntry) -> None:
    if not entry.source_path:
        raise RuntimeError(f"missing remote source path for {entry.rel}")
    dest = dest_root / entry.rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(entry.source_path, str(dest))


def copy_command_file(source: dict[str, Any], dest_root: Path, rel: str) -> None:
    command = str(source.get("copy_command") or "").strip()
    if not command:
        raise ValueError(f"command source {source.get('id')} missing copy_command")
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    formatted = command.format(rel=rel, dest=str(dest), dest_dir=str(dest.parent))
    result = run_command(formatted, timeout=int(source.get("copy_timeout_sec", 300)))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"copy_command failed for {rel}")


def emit_progress(progress: ProgressCallback | None, **payload: Any) -> None:
    if progress is not None:
        progress(payload)


def sync_source(
    source: dict[str, Any],
    mirror_dir: Path,
    include_globs: list[str],
    exclude_globs: list[str],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    source_id = safe_source_id(str(source.get("id") or source_label(source)))
    source_type = str(source.get("type") or "local").lower()
    dest_root = source_local_root(source, mirror_dir)
    manifest_path = mirror_dir / source_id / f".{MANIFEST_NAME}"
    manifest = load_json(manifest_path, {"files": {}})
    previous: dict[str, Any] = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    client = None
    sftp = None
    emit_progress(
        progress,
        phase="listing",
        source_id=source_id,
        source_label=source_label(source),
        source_type=source_type,
        current_file="",
        listed=0,
        copied=0,
        skipped=0,
        error_count=0,
        processed=0,
        total=0,
        message=f"Listing {source_label(source)}",
    )

    if source_type == "local":
        source_root = resolve_dataset_path(str(source.get("path") or ""), ROOT)
        entries = list_local_source_files(source, include_globs, exclude_globs)
    elif source_type == "ssh":
        source_root = None
        client = connect_paramiko(source) if use_paramiko_ssh(source) else None
        sftp = client.open_sftp() if client is not None else None
        entries = list_ssh_files(source, include_globs, exclude_globs, client=client)
    elif source_type == "command":
        source_root = None
        entries = list_command_files(source, include_globs, exclude_globs)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    emit_progress(
        progress,
        phase="copying",
        source_id=source_id,
        source_label=source_label(source),
        source_type=source_type,
        listed=len(entries),
        total=len(entries),
        processed=0,
        copied=0,
        skipped=0,
        error_count=0,
        message=f"Copying changed files from {source_label(source)}",
    )
    copied = 0
    skipped = 0
    errors: list[str] = []
    current_files: dict[str, Any] = {}
    for index, entry in enumerate(entries, start=1):
        legacy_signature = {"size": entry.size, "mtime": round(entry.mtime, 6)}
        signature = {**legacy_signature, "source_path": entry.source_path or ""}
        current_files[entry.rel] = signature
        if previous.get(entry.rel) in (signature, legacy_signature) and (dest_root / entry.rel).exists():
            skipped += 1
            emit_progress(
                progress,
                phase="copying",
                source_id=source_id,
                source_label=source_label(source),
                current_file=entry.rel,
                listed=len(entries),
                total=len(entries),
                processed=index,
                copied=copied,
                skipped=skipped,
                error_count=len(errors),
                message=f"Skipped unchanged {entry.rel}",
            )
            continue
        try:
            emit_progress(
                progress,
                phase="copying",
                source_id=source_id,
                source_label=source_label(source),
                current_file=entry.rel,
                listed=len(entries),
                total=len(entries),
                processed=index - 1,
                copied=copied,
                skipped=skipped,
                error_count=len(errors),
                message=f"Copying {entry.rel}",
            )
            if source_type == "local":
                assert source_root is not None
                copy_local_file(source_root, dest_root, entry)
            elif source_type == "ssh":
                if sftp is not None:
                    copy_paramiko_file(sftp, dest_root, entry)
                else:
                    copy_ssh_file(source, dest_root, entry)
            else:
                copy_command_file(source, dest_root, entry.rel)
            copied += 1
        except Exception as exc:
            errors.append(f"{entry.rel}: {exc}")
        emit_progress(
            progress,
            phase="copying",
            source_id=source_id,
            source_label=source_label(source),
            current_file=entry.rel,
            listed=len(entries),
            total=len(entries),
            processed=index,
            copied=copied,
            skipped=skipped,
            error_count=len(errors),
            message=f"Processed {index}/{len(entries)} from {source_label(source)}",
        )

    if sftp is not None:
        sftp.close()
    if client is not None:
        client.close()

    write_json(
        manifest_path,
        {
            "source_id": source_id,
            "source_label": source_label(source),
            "source_type": source_type,
            "synced_at": utc_now(),
            "files": current_files,
        },
    )
    emit_progress(
        progress,
        phase="source_done",
        source_id=source_id,
        source_label=source_label(source),
        source_type=source_type,
        current_file="",
        listed=len(entries),
        total=len(entries),
        processed=len(entries),
        copied=copied,
        skipped=skipped,
        error_count=len(errors),
        message=f"Finished {source_label(source)}: copied {copied}, skipped {skipped}, errors {len(errors)}",
    )
    return {
        "source_id": source_id,
        "source_label": source_label(source),
        "source_type": source_type,
        "listed": len(entries),
        "copied": copied,
        "skipped": skipped,
        "errors": errors[:50],
        "error_count": len(errors),
        "mirror_path": str(dest_root),
    }


def load_config(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        config = load_json(config_path, {})
    elif DEFAULT_EXAMPLE_CONFIG.exists():
        config = load_json(DEFAULT_EXAMPLE_CONFIG, {})
    else:
        config = {}
    if "sources" not in config:
        config["sources"] = []
    return config


def run_sync(config: dict[str, Any], mirror_dir: Path, progress: ProgressCallback | None = None) -> dict[str, Any]:
    include_globs = list(config.get("include_globs") or [])
    exclude_globs = list(config.get("exclude_globs") or [])
    results = []
    enabled_sources = [source for source in config.get("sources") or [] if isinstance(source, dict) and source_enabled(source)]
    emit_progress(
        progress,
        running=True,
        phase="starting",
        total_sources=len(enabled_sources),
        source_index=0,
        message=f"Starting sync for {len(enabled_sources)} source(s)",
    )
    for source_index, source in enumerate(enabled_sources, start=1):
        try:
            emit_progress(
                progress,
                running=True,
                phase="source_start",
                total_sources=len(enabled_sources),
                source_index=source_index,
                source_id=safe_source_id(str(source.get("id") or source_label(source))),
                source_label=source_label(source),
                message=f"Starting source {source_index}/{len(enabled_sources)}: {source_label(source)}",
            )
            results.append(sync_source(source, mirror_dir, include_globs, exclude_globs, progress=progress))
        except Exception as exc:
            emit_progress(
                progress,
                running=True,
                phase="source_error",
                total_sources=len(enabled_sources),
                source_index=source_index,
                source_id=safe_source_id(str(source.get("id") or source_label(source))),
                source_label=source_label(source),
                error_count=1,
                message=f"Source failed: {source_label(source)}: {exc}",
            )
            results.append(
                {
                    "source_id": safe_source_id(str(source.get("id") or source_label(source))),
                    "source_label": source_label(source),
                    "source_type": source.get("type", "local"),
                    "listed": 0,
                    "copied": 0,
                    "skipped": 0,
                    "errors": [str(exc)],
                    "error_count": 1,
                    "mirror_path": str(source_local_root(source, mirror_dir)),
                }
            )
    summary = {"synced_at": utc_now(), "results": results}
    write_json(mirror_dir / "last_sync.json", summary)
    emit_progress(
        progress,
        running=False,
        phase="done",
        total_sources=len(enabled_sources),
        source_index=len(enabled_sources),
        listed=sum(int(r.get("listed") or 0) for r in results),
        copied=sum(int(r.get("copied") or 0) for r in results),
        skipped=sum(int(r.get("skipped") or 0) for r in results),
        error_count=sum(int(r.get("error_count") or 0) for r in results),
        current_file="",
        message="Sync complete",
    )
    return summary


def collect_model_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
        provider = str(gen.get("provider") or gen.get("llm_provider") or "").strip()
        model = str(gen.get("model") or "").strip()
        key = "/".join(part for part in [provider, model] if part) or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def dominant_model(counts: dict[str, int]) -> str:
    if not counts:
        return "unknown"
    return next(iter(counts.keys()))


def collect_metric_avgs(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_keys = (
        "overall_score",
        "content_coverage",
        "intent_score",
        "section_heading_coverage",
        "table_cell_coverage",
        "action_coverage",
        "image_presence",
        "icon_presence",
        "markdown_leakage_rate",
    )
    sums: dict[str, float] = {key: 0.0 for key in metric_keys}
    counts: dict[str, int] = {key: 0 for key in metric_keys}
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        for key in metric_keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                sums[key] += float(value)
                counts[key] += 1
    return {key: sums[key] / counts[key] for key in metric_keys if counts[key]}


def collect_intent_quality(rows: list[dict[str, Any]], limit: int = 32) -> dict[str, Any]:
    metric_keys = (
        "content_coverage",
        "intent_score",
        "section_heading_coverage",
        "table_cell_coverage",
        "action_coverage",
        "image_presence",
        "icon_presence",
        "markdown_leakage_rate",
        "component_count",
    )
    stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        intent = str(row.get("intent") or row.get("intent_bucket") or "unknown").strip() or "unknown"
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        record = stats.setdefault(
            intent,
            {
                "count": 0,
                "score_sum": 0.0,
                "score_count": 0,
                "metrics": {key: {"sum": 0.0, "count": 0} for key in metric_keys},
            },
        )
        record["count"] += 1
        score = metrics.get("overall_score")
        if isinstance(score, (int, float)):
            record["score_sum"] += float(score)
            record["score_count"] += 1
        for key in metric_keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                metric = record["metrics"][key]
                metric["sum"] += float(value)
                metric["count"] += 1

    finalized: dict[str, Any] = {}
    for intent, record in sorted(stats.items(), key=lambda kv: (-int(kv[1]["count"]), kv[0]))[:limit]:
        metrics = {
            key: metric["sum"] / metric["count"]
            for key, metric in record["metrics"].items()
            if metric["count"]
        }
        score_count = int(record.get("score_count") or 0)
        finalized[intent] = {
            "count": int(record.get("count") or 0),
            "avg_score": float(record["score_sum"]) / score_count if score_count else None,
            "score_count": score_count,
            "metrics": metrics,
        }
    return finalized


def row_issue_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
    if validation.get("json_parse_ok") is False:
        labels.append("json_parse_fail")
    if validation.get("schema_valid_strict") is False:
        labels.append("strict_schema_fail")
    if validation.get("repair_needed") is True or int(validation.get("repair_attempts") or 0) > 0:
        labels.append("repair")
    if gen.get("error"):
        labels.append("gen_error")
    if row.get("fallback_generated") or validation.get("fallback_generated"):
        labels.append("fallback")
    score = metrics.get("overall_score")
    if isinstance(score, (int, float)) and score < 60:
        labels.append("low_score")
    markdown = metrics.get("markdown_leakage_rate")
    if isinstance(markdown, (int, float)) and markdown > 0:
        labels.append("markdown_leak")
    component_count = metrics.get("component_count")
    content_coverage = metrics.get("content_coverage")
    if (
        isinstance(component_count, (int, float))
        and isinstance(content_coverage, (int, float))
        and component_count < 12
        and content_coverage < 0.55
    ):
        labels.append("sparse_ir")
    return labels


def collect_issue_samples(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
        labels = row_issue_labels(row)
        score = metrics.get("overall_score")
        if not labels and not isinstance(score, (int, float)):
            continue
        if not labels and isinstance(score, (int, float)) and score >= 70:
            continue
        provider = str(gen.get("provider") or gen.get("llm_provider") or "").strip()
        model = str(gen.get("model") or "").strip()
        sample = {
            "row": index,
            "ui_id": str(row.get("ui_id") or "").strip(),
            "response_id": str(row.get("response_id") or "").strip(),
            "query_id": str(row.get("query_id") or "").strip(),
            "intent": str(row.get("intent") or row.get("intent_bucket") or "").strip(),
            "score": float(score) if isinstance(score, (int, float)) else None,
            "issues": labels,
            "model": "/".join(part for part in [provider, model] if part) or "unknown",
            "prompt_version": ir_version_for_row(row),
            "content_coverage": metrics.get("content_coverage") if isinstance(metrics.get("content_coverage"), (int, float)) else None,
            "component_count": metrics.get("component_count") if isinstance(metrics.get("component_count"), (int, float)) else None,
            "warnings": [
                str(item)[:120]
                for item in (validation.get("errors") or validation.get("warnings") or [])
                if str(item).strip()
            ][:2],
        }
        samples.append(sample)
    samples.sort(
        key=lambda item: (
            item["score"] if isinstance(item.get("score"), (int, float)) else 999.0,
            -len(item.get("issues") or []),
            item.get("row") or 0,
        )
    )
    return samples[:limit]


def collect_quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "sampled": 0,
        "json_parse_fail": 0,
        "strict_schema_fail": 0,
        "repair_needed": 0,
        "repair_attempted": 0,
        "gen_errors": 0,
        "fallback_generated": 0,
        "low_score": 0,
        "markdown_leakage": 0,
        "sparse_ir": 0,
        "warnings": [],
        "issue_samples": [],
    }
    warnings: dict[str, int] = {}
    for row in rows:
        summary["sampled"] += 1
        validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
        if validation.get("json_parse_ok") is False:
            summary["json_parse_fail"] += 1
        if validation.get("schema_valid_strict") is False:
            summary["strict_schema_fail"] += 1
        if validation.get("repair_needed") is True:
            summary["repair_needed"] += 1
        if int(validation.get("repair_attempts") or 0) > 0:
            summary["repair_attempted"] += 1
        if gen.get("error"):
            summary["gen_errors"] += 1
        if row.get("fallback_generated") or validation.get("fallback_generated"):
            summary["fallback_generated"] += 1
        score = metrics.get("overall_score")
        if isinstance(score, (int, float)) and score < 60:
            summary["low_score"] += 1
        markdown = metrics.get("markdown_leakage_rate")
        if isinstance(markdown, (int, float)) and markdown > 0:
            summary["markdown_leakage"] += 1
        component_count = metrics.get("component_count")
        content_coverage = metrics.get("content_coverage")
        if (
            isinstance(component_count, (int, float))
            and isinstance(content_coverage, (int, float))
            and component_count < 12
            and content_coverage < 0.55
        ):
            summary["sparse_ir"] += 1
        for error in validation.get("errors") or []:
            text = str(error).strip()
            if text:
                warnings[text[:140]] = warnings.get(text[:140], 0) + 1
        for warning in validation.get("warnings") or []:
            text = str(warning).strip()
            if text:
                warnings[text[:140]] = warnings.get(text[:140], 0) + 1
    summary["warnings"] = [
        {"message": message, "count": count}
        for message, count in sorted(warnings.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    ]
    summary["issue_samples"] = collect_issue_samples(rows)
    return summary


def ir_version_for_row(row: dict[str, Any]) -> str:
    gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
    candidates = [
        gen.get("prompt_version"),
        row.get("prompt_version"),
        row.get("ir_version"),
        gen.get("ir_version"),
        row.get("schema_version"),
        row.get("version"),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return "unknown"


def collect_ir_version_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        version = ir_version_for_row(row)
        counts[version] = counts.get(version, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def score_from_genui(genui_path: Path) -> float | None:
    values: list[float] = []
    with genui_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            value = metrics.get("overall_score")
            if isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def run_score(run_dir: Path) -> float | None:
    aggregate_path = run_dir / "aggregates.json"
    if aggregate_path.exists():
        try:
            aggregate = load_json(aggregate_path, {})
            for key in ("overall_score", "score", "aggregate_score"):
                value = aggregate.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        except Exception:
            pass
    genui_path = run_dir / "genui.jsonl"
    if genui_path.exists():
        return score_from_genui(genui_path)
    return None


def parse_day(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 100_000_000_000:
            timestamp /= 1000
        try:
            return datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text[:10]
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).astimezone(timezone.utc).date().isoformat()
    except Exception:
        return None


def row_day(row: dict[str, Any], fallback_day: str) -> str:
    candidates = [
        row.get("created_at"),
        row.get("generated_at"),
        row.get("updated_at"),
        row.get("timestamp"),
    ]
    gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
    candidates.extend([gen.get("created_at"), gen.get("timestamp")])
    for candidate in candidates:
        parsed = parse_day(candidate)
        if parsed:
            return parsed
    return fallback_day


def new_day_bucket(day: str) -> dict[str, Any]:
    return {
        "day": day,
        "queries": 0,
        "responses": 0,
        "genui": 0,
        "score_sum": 0.0,
        "score_count": 0,
    }


def merge_day_buckets(target: dict[str, dict[str, Any]], source: dict[str, dict[str, Any]]) -> None:
    for day, values in source.items():
        bucket = target.setdefault(day, new_day_bucket(day))
        bucket["queries"] += int(values.get("queries") or 0)
        bucket["responses"] += int(values.get("responses") or 0)
        bucket["genui"] += int(values.get("genui") or 0)
        bucket["score_sum"] += float(values.get("score_sum") or 0.0)
        bucket["score_count"] += int(values.get("score_count") or 0)


def jsonl_day_buckets(path: Path, kind: str, fallback_day: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return buckets
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                row = {}
            row = row if isinstance(row, dict) else {}
            day = row_day(row, fallback_day)
            bucket = buckets.setdefault(day, new_day_bucket(day))
            if kind == "queries":
                bucket["queries"] += 1
            elif kind == "responses":
                bucket["responses"] += 1
            elif kind == "genui":
                bucket["genui"] += 1
                metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                score = metrics.get("overall_score")
                if isinstance(score, (int, float)):
                    bucket["score_sum"] += float(score)
                    bucket["score_count"] += 1
    return buckets


def collect_ir_version_stats(path: Path, fallback_day: str) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return stats
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                row = {}
            row = row if isinstance(row, dict) else {}
            version = ir_version_for_row(row)
            record = stats.setdefault(
                version,
                {
                    "queries": 0,
                    "responses": 0,
                    "genui": 0,
                    "score_sum": 0.0,
                    "score_count": 0,
                    "days": {},
                    "_query_ids": set(),
                    "_response_ids": set(),
                },
            )
            query_id = str(row.get("query_id") or "").strip()
            response_id = str(row.get("response_id") or "").strip()
            if query_id:
                record["_query_ids"].add(query_id)
            if response_id:
                record["_response_ids"].add(response_id)
            record["genui"] += 1
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            score = metrics.get("overall_score")
            if isinstance(score, (int, float)):
                record["score_sum"] += float(score)
                record["score_count"] += 1
            day = row_day(row, fallback_day)
            day_bucket = record["days"].setdefault(day, new_day_bucket(day))
            day_bucket["queries"] += 1 if query_id or not record["_query_ids"] else 0
            day_bucket["responses"] += 1 if response_id or not record["_response_ids"] else 0
            day_bucket["genui"] += 1
            if isinstance(score, (int, float)):
                day_bucket["score_sum"] += float(score)
                day_bucket["score_count"] += 1

    finalized: dict[str, dict[str, Any]] = {}
    for version, record in stats.items():
        genui_count = int(record.get("genui") or 0)
        query_count = len(record.get("_query_ids") or []) or genui_count
        response_count = len(record.get("_response_ids") or []) or genui_count
        score_count = int(record.get("score_count") or 0)
        finalized[version] = {
            "queries": query_count,
            "responses": response_count,
            "genui": genui_count,
            "score_sum": float(record.get("score_sum") or 0.0),
            "score_count": score_count,
            "avg_score": float(record["score_sum"]) / score_count if score_count else None,
            "days": finalize_day_buckets(record.get("days") or {}),
        }
    return dict(sorted(finalized.items(), key=lambda kv: (-kv[1]["genui"], kv[0])))


def finalize_day_buckets(buckets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    days = []
    for day in sorted(buckets.keys(), reverse=True):
        bucket = buckets[day]
        score_count = int(bucket.get("score_count") or 0)
        avg_score = float(bucket["score_sum"]) / score_count if score_count else None
        days.append(
            {
                "day": day,
                "queries": int(bucket.get("queries") or 0),
                "responses": int(bucket.get("responses") or 0),
                "genui": int(bucket.get("genui") or 0),
                "score_sum": float(bucket.get("score_sum") or 0.0),
                "score_count": score_count,
                "avg_score": avg_score,
            }
        )
    return days


def aggregate_days(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        for day_record in record.get("days") or []:
            day = str(day_record.get("day") or "").strip()
            if not day:
                continue
            bucket = buckets.setdefault(day, new_day_bucket(day))
            bucket["queries"] += int(day_record.get("queries") or 0)
            bucket["responses"] += int(day_record.get("responses") or 0)
            bucket["genui"] += int(day_record.get("genui") or 0)
            score_count = int(day_record.get("score_count") or 0)
            if score_count:
                bucket["score_sum"] += float(day_record.get("score_sum") or 0.0)
                bucket["score_count"] += score_count
            elif isinstance(day_record.get("avg_score"), (int, float)):
                count = int(day_record.get("genui") or 0)
                bucket["score_sum"] += float(day_record["avg_score"]) * count
                bucket["score_count"] += count
    return finalize_day_buckets(buckets)


def aggregate_ir_versions(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for version, count in (record.get("ir_versions") or {}).items():
            counts[version] = counts.get(version, 0) + int(count or 0)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def backlog_summary(queries: int, responses: int, genui: int) -> dict[str, Any]:
    response_backlog = max(queries - responses, 0)
    ir_backlog = max(responses - genui, 0)
    completion_rate = (genui / queries) if queries else (1.0 if genui else 0.0)
    return {
        "response_backlog": response_backlog,
        "ir_backlog": ir_backlog,
        "completion_rate": completion_rate,
    }


def file_artifact(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    stat = path.stat()
    return {
        "label": label,
        "name": path.name,
        "path": str(path),
        "size": int(stat.st_size),
        "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def sample_files(root: Path, pattern: str, limit: int = 6) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    files = [path for path in root.rglob(pattern) if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    samples = []
    for path in files[:limit]:
        stat = path.stat()
        samples.append(
            {
                "name": path.name,
                "path": str(path),
                "size": int(stat.st_size),
                "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    return samples


def run_artifacts(run_dir: Path) -> dict[str, Any]:
    core = [
        artifact
        for artifact in (
            file_artifact(run_dir / "queries.jsonl", "queries"),
            file_artifact(run_dir / "responses.jsonl", "responses"),
            file_artifact(run_dir / "genui.jsonl", "IR"),
            file_artifact(run_dir / "aggregates.json", "aggregates"),
            file_artifact(run_dir / "run_manifest.json", "manifest"),
            file_artifact(run_dir / "run.log", "log"),
        )
        if artifact is not None
    ]
    screenshot_roots = [run_dir / name for name in ("android_device_rendered", "rendered", "rendered_lit")]
    screenshots: list[dict[str, Any]] = []
    for root in screenshot_roots:
        screenshots.extend(sample_files(root, "*.png", limit=3))
    screenshots.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return {
        "core": core,
        "asset_samples": sample_files(run_dir / "assets", "*", limit=6),
        "screenshot_samples": screenshots[:6],
    }


def aggregate_model_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        response_model = dominant_model(record.get("response_models") or {})
        ir_model = dominant_model(record.get("ir_models") or {})
        key = f"{response_model} -> {ir_model}"
        bucket = buckets.setdefault(
            key,
            {
                "response_model": response_model,
                "ir_model": ir_model,
                "runs": 0,
                "genui": 0,
                "score_sum": 0.0,
                "score_weight": 0,
                "metrics": {},
            },
        )
        weight = max(1, int(record.get("genui") or 0))
        bucket["runs"] += 1
        bucket["genui"] += int(record.get("genui") or 0)
        score = record.get("overall_score")
        if isinstance(score, (int, float)):
            bucket["score_sum"] += float(score) * weight
            bucket["score_weight"] += weight
        for metric, value in (record.get("metric_avgs") or {}).items():
            if not isinstance(value, (int, float)):
                continue
            metric_bucket = bucket["metrics"].setdefault(metric, {"sum": 0.0, "weight": 0})
            metric_bucket["sum"] += float(value) * weight
            metric_bucket["weight"] += weight

    comparisons = []
    for bucket in buckets.values():
        metrics = {
            key: data["sum"] / data["weight"]
            for key, data in bucket["metrics"].items()
            if data.get("weight")
        }
        comparisons.append(
            {
                "response_model": bucket["response_model"],
                "ir_model": bucket["ir_model"],
                "runs": bucket["runs"],
                "genui": bucket["genui"],
                "avg_score": bucket["score_sum"] / bucket["score_weight"] if bucket["score_weight"] else None,
                "metrics": metrics,
            }
        )
    return sorted(comparisons, key=lambda item: (-(item.get("genui") or 0), str(item.get("response_model"))))


def effective_dashboard_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = [
        {
            "id": "_local_checkout",
            "label": "Local checkout",
            "type": "local",
            "path": "data/runs",
            "mirror_local_in_place": True,
            "enabled": True,
        }
    ]
    sources.extend(source for source in config.get("sources") or [] if isinstance(source, dict))
    return sources


def safe_identity_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(os.path.expandvars(os.path.expanduser(text))).name


def source_config_summary(source: dict[str, Any]) -> dict[str, Any]:
    source_id = safe_source_id(str(source.get("id") or source_label(source)))
    summary = {
        "source_id": source_id,
        "source_label": source_label(source),
        "type": str(source.get("type") or "local"),
        "enabled": source_enabled(source),
        "path": str(source.get("path") or ""),
        "path_match": source_path_mode(source),
        "path_base": str(source.get("path_base") or ""),
        "mirror_local_in_place": bool(source.get("mirror_local_in_place", False)),
        "ask_password": bool(source.get("ask_password") or source.get("prompt_password")),
        "has_password": bool(ssh_password(source)),
    }
    for key in ("host", "user", "port", "ssh_backend", "proxy_jump"):
        value = source.get(key)
        if value not in (None, ""):
            summary[key] = value
    identity_label = safe_identity_label(source.get("identity_file"))
    if identity_label:
        summary["identity_file"] = identity_label
    proxy_command = str(source.get("proxy_command") or "").strip()
    if proxy_command:
        summary["has_proxy_command"] = True
        summary["proxy_command_preview"] = proxy_command[:96] + ("..." if len(proxy_command) > 96 else "")
    options = normalized_ssh_options(source) if str(source.get("type") or "local").lower() == "ssh" else []
    if options:
        summary["ssh_options"] = options
    return summary


def dashboard_config_summary(config: dict[str, Any], mirror_dir: Path) -> dict[str, Any]:
    configured_sources = [source for source in config.get("sources") or [] if isinstance(source, dict)]
    effective_sources = effective_dashboard_sources(config)
    return {
        "mirror_dir": str(mirror_dir),
        "include_globs": list(config.get("include_globs") or []),
        "exclude_globs": list(config.get("exclude_globs") or []),
        "configured_source_count": len(configured_sources),
        "effective_source_count": len(effective_sources),
        "enabled_source_count": len([source for source in effective_sources if source_enabled(source)]),
        "sources": [source_config_summary(source) for source in effective_sources],
    }


def latest_sync_result(latest_sync: dict[str, Any] | None, source_id: str) -> dict[str, Any] | None:
    if not isinstance(latest_sync, dict):
        return None
    for result in latest_sync.get("results") or []:
        if isinstance(result, dict) and result.get("source_id") == source_id:
            return result
    return None


def source_health_record(
    source: dict[str, Any],
    source_id: str,
    local_root: Path,
    source_runs: list[dict[str, Any]],
    latest_sync: dict[str, Any] | None,
) -> dict[str, Any]:
    sync_result = latest_sync_result(latest_sync, source_id)
    if sync_result:
        errors = sync_result.get("errors") or []
        error_count = int(sync_result.get("error_count") or 0)
        status = "error" if error_count else "ok"
        message = str(errors[0]) if errors else f"Last sync listed {int(sync_result.get('listed') or 0)} files"
        return {
            "status": status,
            "message": message,
            "last_synced_at": latest_sync.get("synced_at") if isinstance(latest_sync, dict) else None,
            "listed": int(sync_result.get("listed") or 0),
            "copied": int(sync_result.get("copied") or 0),
            "skipped": int(sync_result.get("skipped") or 0),
            "error_count": error_count,
        }
    if local_root.exists() and source_runs:
        return {
            "status": "ok",
            "message": f"Local mirror has {len(source_runs)} run(s)",
            "last_synced_at": None,
            "listed": None,
            "copied": None,
            "skipped": None,
            "error_count": 0,
        }
    if local_root.exists():
        return {
            "status": "empty",
            "message": "Path exists, but no dataset runs were found",
            "last_synced_at": None,
            "listed": None,
            "copied": None,
            "skipped": None,
            "error_count": 0,
        }
    return {
        "status": "unknown",
        "message": "No local mirror found; run sync or test connection",
        "last_synced_at": None,
        "listed": None,
        "copied": None,
        "skipped": None,
        "error_count": None,
    }


def local_source_probe(source: dict[str, Any]) -> dict[str, Any]:
    mode = source_path_mode(source)
    roots: list[Path]
    if mode == "literal":
        root = resolve_dataset_path(str(source.get("path") or ""), ROOT)
        roots = [root] if root.exists() else []
    elif mode == "glob":
        pattern = str(resolve_dataset_path(str(source.get("path") or ""), ROOT))
        roots = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]
    elif mode == "regex":
        roots = regex_local_roots(source)
    else:
        raise ValueError(f"Unsupported local path_match: {mode}")
    return {
        "root_count": len(roots),
        "sample_roots": [str(path) for path in sorted(roots, key=lambda p: p.as_posix())[:5]],
        "exists": bool(roots),
    }


def ssh_source_probe(source: dict[str, Any]) -> dict[str, Any]:
    remote_root = str(source.get("path") or "").rstrip("/")
    if not remote_root:
        raise ValueError(f"SSH source {source.get('id')} missing path")
    mode = source_path_mode(source)
    path_base = str(source.get("path_base") or "").rstrip("/")
    if mode == "regex" and not path_base:
        raise ValueError(f"SSH source {source.get('id')} uses path_match=regex but missing path_base")
    py = r"""
import os, sys, json, glob, re
root=sys.argv[1]
mode=sys.argv[2]
path_base=sys.argv[3]
def roots_for_mode():
    if mode == 'literal':
        return [root] if os.path.isdir(root) else []
    if mode == 'glob':
        return [p for p in sorted(glob.glob(root)) if os.path.isdir(p)]
    if mode == 'regex':
        compiled=re.compile(root)
        return [os.path.join(path_base, name) for name in sorted(os.listdir(path_base)) if os.path.isdir(os.path.join(path_base, name)) and compiled.search(name)]
    raise SystemExit('Unsupported path_match: ' + mode)
roots=roots_for_mode()
print(json.dumps({'exists': bool(roots), 'root_count': len(roots), 'sample_roots': roots[:5]}, separators=(',',':')))
"""
    remote_command = " ".join(
        [
            "python3",
            "-c",
            shell_quote(py),
            shell_quote(remote_root),
            shell_quote(mode),
            shell_quote(path_base),
        ]
    )
    timeout = int(source.get("test_timeout_sec", source.get("connect_timeout_sec", 30)))
    if use_paramiko_ssh(source):
        client = connect_paramiko(source)
        try:
            stdin, stdout, stderr = client.exec_command(remote_command, timeout=timeout)
            del stdin
            output = stdout.read().decode("utf-8", errors="replace")
            error = stderr.read().decode("utf-8", errors="replace")
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                raise RuntimeError(error.strip() or output.strip() or "ssh source probe failed")
        finally:
            client.close()
    else:
        result = run_command(ssh_base_command(source) + [remote_command], timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ssh source probe failed")
        output = result.stdout
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError("ssh source probe returned no JSON")


def command_source_probe(source: dict[str, Any], include_globs: list[str], exclude_globs: list[str]) -> dict[str, Any]:
    entries = list_command_files(source, include_globs, exclude_globs)
    return {
        "exists": bool(entries),
        "root_count": None,
        "file_count": len(entries),
        "sample_files": [entry.rel for entry in entries[:5]],
    }


def run_file_stats(run_dir: Path) -> dict[str, Any]:
    fallback_mtime = run_dir.stat().st_mtime
    stats = {
        "mtime": fallback_mtime,
        "total_bytes": 0,
        "file_count": 0,
        "asset_bytes": 0,
        "asset_files": 0,
        "screenshot_bytes": 0,
        "screenshot_files": 0,
    }
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            rel_parts = path.relative_to(run_dir).parts
        except OSError:
            continue
        stats["file_count"] += 1
        stats["total_bytes"] += int(stat.st_size)
        stats["mtime"] = max(float(stats["mtime"]), float(stat.st_mtime))
        top = rel_parts[0] if rel_parts else ""
        if top == "assets":
            stats["asset_files"] += 1
            stats["asset_bytes"] += int(stat.st_size)
        if top in {"android_device_rendered", "rendered", "rendered_lit"} and path.suffix.lower() == ".png":
            stats["screenshot_files"] += 1
            stats["screenshot_bytes"] += int(stat.st_size)
    return stats


def test_source_connection(
    config: dict[str, Any],
    mirror_dir: Path,
    requested_source_id: str,
) -> dict[str, Any]:
    del mirror_dir
    include_globs = list(config.get("include_globs") or [])
    exclude_globs = list(config.get("exclude_globs") or [])
    sources = effective_dashboard_sources(config)
    source = None
    for candidate in sources:
        candidate_id = safe_source_id(str(candidate.get("id") or source_label(candidate)))
        if candidate_id == requested_source_id:
            source = candidate
            break
    if source is None:
        raise ValueError(f"Unknown source_id: {requested_source_id}")
    source_id = safe_source_id(str(source.get("id") or source_label(source)))
    source_type = str(source.get("type") or "local").lower()
    started = datetime.now(timezone.utc)
    if not source_enabled(source):
        return {
            "source_id": source_id,
            "source_label": source_label(source),
            "source_type": source_type,
            "status": "disabled",
            "message": "Source is disabled",
            "checked_at": utc_now(),
            "elapsed_ms": 0,
        }
    try:
        if source_type == "local":
            probe = local_source_probe(source)
        elif source_type == "ssh":
            probe = ssh_source_probe(source)
        elif source_type == "command":
            probe = command_source_probe(source, include_globs, exclude_globs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        status = "ok" if probe.get("exists") or probe.get("file_count") else "empty"
        message = (
            f"Found {probe.get('root_count')} root(s)"
            if probe.get("root_count") is not None
            else f"Found {probe.get('file_count', 0)} file(s)"
        )
        return {
            "source_id": source_id,
            "source_label": source_label(source),
            "source_type": source_type,
            "status": status,
            "message": message,
            "probe": probe,
            "checked_at": utc_now(),
            "elapsed_ms": int((datetime.now(timezone.utc) - started).total_seconds() * 1000),
        }
    except Exception as exc:
        return {
            "source_id": source_id,
            "source_label": source_label(source),
            "source_type": source_type,
            "status": "error",
            "message": str(exc),
            "checked_at": utc_now(),
            "elapsed_ms": int((datetime.now(timezone.utc) - started).total_seconds() * 1000),
        }


def scan_run(source_id: str, source_label_text: str, run_dir: Path) -> dict[str, Any]:
    queries = count_jsonl(run_dir / "queries.jsonl")
    responses = count_jsonl(run_dir / "responses.jsonl")
    genui = count_jsonl(run_dir / "genui.jsonl")
    backlog = backlog_summary(queries, responses, genui)
    file_stats = run_file_stats(run_dir)
    assets = int(file_stats.get("asset_files") or 0)
    screenshots = int(file_stats.get("screenshot_files") or 0)
    missing_core_files = [
        name
        for name in ("queries.jsonl", "responses.jsonl", "genui.jsonl", "aggregates.json")
        if not (run_dir / name).exists()
    ]
    genui_rows = read_first_jsonl(run_dir / "genui.jsonl", limit=5000)
    response_rows = read_first_jsonl(run_dir / "responses.jsonl", limit=5000)
    query_rows = read_first_jsonl(run_dir / "queries.jsonl", limit=5000)
    intents: dict[str, int] = {}
    for row in query_rows + response_rows + genui_rows:
        intent = str(row.get("intent") or row.get("intent_bucket") or "").strip()
        if intent:
            intents[intent] = intents.get(intent, 0) + 1
    mtime = float(file_stats.get("mtime") or run_dir.stat().st_mtime)
    fallback_day = datetime.fromtimestamp(mtime, timezone.utc).date().isoformat()
    day_buckets: dict[str, dict[str, Any]] = {}
    for filename, kind in (
        ("queries.jsonl", "queries"),
        ("responses.jsonl", "responses"),
        ("genui.jsonl", "genui"),
    ):
        path = run_dir / filename
        file_day = (
            datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date().isoformat()
            if path.exists()
            else fallback_day
        )
        merge_day_buckets(day_buckets, jsonl_day_buckets(path, kind, file_day))
    ir_version_stats = collect_ir_version_stats(run_dir / "genui.jsonl", fallback_day)
    return {
        "source_id": source_id,
        "source_label": source_label_text,
        "run_id": run_dir.name,
        "path": str(run_dir),
        "queries": queries,
        "responses": responses,
        "genui": genui,
        **backlog,
        "assets": assets,
        "screenshots": screenshots,
        "asset_bytes": int(file_stats.get("asset_bytes") or 0),
        "screenshot_bytes": int(file_stats.get("screenshot_bytes") or 0),
        "total_bytes": int(file_stats.get("total_bytes") or 0),
        "file_count": int(file_stats.get("file_count") or 0),
        "missing_core_files": missing_core_files,
        "overall_score": run_score(run_dir),
        "updated_at": datetime.fromtimestamp(mtime, timezone.utc).isoformat(),
        "query_models": collect_model_counts(query_rows),
        "response_models": collect_model_counts(response_rows),
        "ir_models": collect_model_counts(genui_rows),
        "metric_avgs": collect_metric_avgs(genui_rows),
        "intent_quality": collect_intent_quality(genui_rows),
        "quality_summary": collect_quality_summary(genui_rows),
        "artifacts": run_artifacts(run_dir),
        "ir_versions": {version: stats["genui"] for version, stats in ir_version_stats.items()},
        "ir_version_stats": ir_version_stats,
        "intents": dict(sorted(intents.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
        "days": finalize_day_buckets(day_buckets),
    }


def scan_all(config: dict[str, Any], mirror_dir: Path) -> dict[str, Any]:
    sources = []
    runs = []
    latest_sync = load_json(mirror_dir / "last_sync.json", None)
    effective_sources = effective_dashboard_sources(config)

    seen_sources: set[str] = set()
    for source in effective_sources:
        if not source_enabled(source):
            continue
        source_id = safe_source_id(str(source.get("id") or source_label(source)))
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        local_root = source_local_root(source, mirror_dir)
        run_dirs = [p for p in local_root.iterdir() if p.is_dir()] if local_root.exists() else []
        source_runs = []
        for run_dir in sorted(run_dirs, key=lambda p: p.name):
            if not any((run_dir / name).exists() for name in ("queries.jsonl", "responses.jsonl", "genui.jsonl", "aggregates.json")):
                continue
            record = scan_run(source_id, source_label(source), run_dir)
            runs.append(record)
            source_runs.append(record)
        queries = sum(r["queries"] for r in source_runs)
        responses = sum(r["responses"] for r in source_runs)
        genui = sum(r["genui"] for r in source_runs)
        backlog = backlog_summary(queries, responses, genui)
        updated_values = [str(r.get("updated_at") or "") for r in source_runs if r.get("updated_at")]
        sources.append(
            {
                "source_id": source_id,
                "source_label": source_label(source),
                "type": source.get("type", "local"),
                "local_path": str(local_root),
                "run_count": len(source_runs),
                "queries": queries,
                "responses": responses,
                "genui": genui,
                **backlog,
                "assets": sum(r["assets"] for r in source_runs),
                "screenshots": sum(r["screenshots"] for r in source_runs),
                "asset_bytes": sum(int(r.get("asset_bytes") or 0) for r in source_runs),
                "screenshot_bytes": sum(int(r.get("screenshot_bytes") or 0) for r in source_runs),
                "total_bytes": sum(int(r.get("total_bytes") or 0) for r in source_runs),
                "file_count": sum(int(r.get("file_count") or 0) for r in source_runs),
                "missing_core_run_count": sum(1 for r in source_runs if r.get("missing_core_files")),
                "avg_score": (
                    sum(r["overall_score"] for r in source_runs if isinstance(r["overall_score"], (int, float)))
                    / max(1, len([r for r in source_runs if isinstance(r["overall_score"], (int, float))]))
                    if source_runs
                    else None
                ),
                "latest_run_updated_at": max(updated_values) if updated_values else None,
                "oldest_run_updated_at": min(updated_values) if updated_values else None,
                "days": aggregate_days(source_runs),
                "health": source_health_record(source, source_id, local_root, source_runs, latest_sync),
            }
        )

    total_queries = sum(r["queries"] for r in runs)
    total_responses = sum(r["responses"] for r in runs)
    total_genui = sum(r["genui"] for r in runs)
    totals = {
        "sources": len(sources),
        "runs": len(runs),
        "queries": total_queries,
        "responses": total_responses,
        "genui": total_genui,
        **backlog_summary(total_queries, total_responses, total_genui),
        "assets": sum(r["assets"] for r in runs),
        "screenshots": sum(r["screenshots"] for r in runs),
        "asset_bytes": sum(int(r.get("asset_bytes") or 0) for r in runs),
        "screenshot_bytes": sum(int(r.get("screenshot_bytes") or 0) for r in runs),
        "total_bytes": sum(int(r.get("total_bytes") or 0) for r in runs),
        "file_count": sum(int(r.get("file_count") or 0) for r in runs),
        "missing_core_run_count": sum(1 for r in runs if r.get("missing_core_files")),
        "latest_run_updated_at": max([str(r.get("updated_at") or "") for r in runs if r.get("updated_at")], default=None),
    }
    return {
        "generated_at": utc_now(),
        "totals": totals,
        "sources": sorted(sources, key=lambda s: s["source_label"].lower()),
        "runs": sorted(runs, key=lambda r: (r["source_label"].lower(), r["run_id"].lower())),
        "days": aggregate_days(runs),
        "ir_versions": aggregate_ir_versions(runs),
        "model_comparisons": aggregate_model_comparisons(runs),
        "last_sync": latest_sync,
        "config": dashboard_config_summary(config, mirror_dir),
        "config_note": f"Using {DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else DEFAULT_EXAMPLE_CONFIG}",
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GenUICraft Dataset Dashboard</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --ink: #17202a;
      --muted: #667085;
      --card: rgba(255,255,255,.82);
      --line: rgba(36,48,64,.14);
      --accent: #0f766e;
      --accent2: #c2410c;
      --good: #15803d;
      --warn: #b45309;
      --bad: #b91c1c;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 8%, rgba(15,118,110,.16), transparent 34rem),
        radial-gradient(circle at 85% 4%, rgba(194,65,12,.14), transparent 30rem),
        linear-gradient(135deg, #fbf8ef, var(--bg));
    }
    header { padding: 30px 34px 18px; }
    h1 { margin: 0; font-size: clamp(30px, 4vw, 54px); letter-spacing: -.04em; }
    .sub { color: var(--muted); margin: 8px 0 0; max-width: 980px; }
    .toolbar { display:flex; gap: 12px; align-items:center; flex-wrap: wrap; margin-top: 20px; }
    button, input, select {
      border: 1px solid var(--line);
      background: var(--card);
      color: var(--ink);
      border-radius: 14px;
      padding: 11px 13px;
      font-size: 14px;
    }
    button {
      cursor: pointer;
      background: var(--accent);
      color: white;
      border-color: transparent;
      font-weight: 750;
      box-shadow: 0 10px 24px rgba(15,118,110,.18);
    }
    .date-label { display:inline-flex; align-items:center; gap: 6px; color: var(--muted); font-size: 13px; }
    main { padding: 0 34px 34px; }
    .stats { display:grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap: 14px; margin: 16px 0 24px; }
    .stat, .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 16px 40px rgba(32,38,46,.08);
      backdrop-filter: blur(14px);
    }
    .stat { padding: 18px; }
    .stat .v { font-size: 30px; font-weight: 850; letter-spacing: -.03em; }
    .stat .k { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .grid { display:grid; grid-template-columns: minmax(280px, 420px) 1fr; gap: 18px; align-items:start; }
    .panel { padding: 18px; overflow:hidden; }
    h2 { margin: 0 0 12px; font-size: 20px; letter-spacing: -.02em; }
    table { width:100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align:left; border-bottom: 1px solid var(--line); padding: 10px 8px; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
    tr:hover td { background: rgba(15,118,110,.05); }
    .badge { display:inline-flex; align-items:center; border-radius: 999px; padding: 4px 8px; background: rgba(15,118,110,.10); color: #115e59; font-weight: 750; font-size: 12px; }
    .badge.ok { background: rgba(21,128,61,.12); color: var(--good); }
    .badge.empty, .badge.unknown { background: rgba(180,83,9,.12); color: var(--warn); }
    .badge.warn { background: rgba(180,83,9,.12); color: var(--warn); }
    .badge.error { background: rgba(185,28,28,.12); color: var(--bad); }
    .mini-btn { padding: 7px 10px; border-radius: 999px; font-size: 12px; box-shadow: none; }
    .ghost-btn { background: rgba(255,255,255,.66); color: var(--accent); border-color: rgba(15,118,110,.22); box-shadow: none; }
    .source-head { display:flex; align-items:flex-start; justify-content:space-between; gap: 10px; }
    .score { font-weight: 850; }
    .score.good { color: var(--good); }
    .score.warn { color: var(--warn); }
    .small { color: var(--muted); font-size: 12px; }
    .scroll { overflow:auto; max-height: calc(100vh - 260px); }
    .source-card { border:1px solid var(--line); border-radius:18px; padding: 14px; margin-bottom: 12px; background: rgba(255,255,255,.54); }
    .source-card strong { display:block; margin-bottom:6px; }
    .kv { display:grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 8px; margin-top: 10px; }
    .kv div { background: rgba(255,255,255,.55); border-radius: 12px; padding: 9px; }
    .wide-panel { margin-top: 18px; }
    .two-col-panels { display:grid; grid-template-columns: minmax(280px, 1fr) minmax(360px, 1.3fr); gap: 18px; align-items:start; }
    .detail-panel { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 14px; }
    .detail-grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); gap: 10px; margin-top: 10px; }
    .detail-box { background: rgba(255,255,255,.58); border: 1px solid var(--line); border-radius: 14px; padding: 10px; }
    .warning-list { display:grid; gap: 8px; }
    .warning-row { display:flex; justify-content:space-between; gap: 10px; border-bottom: 1px solid var(--line); padding: 8px 0; }
    .nowrap { white-space: nowrap; }
    .artifact-list { display:grid; gap: 6px; margin-top: 8px; }
    .artifact-row { display:grid; grid-template-columns: 90px 1fr auto; gap: 8px; align-items:center; padding: 7px 0; border-bottom: 1px solid var(--line); }
    .chart-wrap { min-height: 230px; }
    .chart-svg { width:100%; height:220px; overflow:visible; }
    .config-source { border: 1px solid var(--line); border-radius: 14px; padding: 10px; margin-top: 8px; background: rgba(255,255,255,.52); }
    .pill-row { display:flex; flex-wrap:wrap; gap: 6px; margin-top: 8px; }
    .dist-grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap: 16px; }
    .dist-row { display:grid; grid-template-columns: minmax(90px, 1fr) 2fr 70px; gap: 10px; align-items:center; padding: 7px 0; border-bottom: 1px solid var(--line); }
    .dist-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .pager { display:flex; align-items:center; justify-content:space-between; gap: 10px; flex-wrap:wrap; margin: 0 0 12px; }
    .pager-controls { display:flex; align-items:center; gap: 8px; flex-wrap:wrap; }
    button:disabled { opacity:.45; cursor:not-allowed; box-shadow:none; }
    .day-row { display:grid; grid-template-columns: 118px 1fr 92px; gap: 12px; align-items:center; padding: 10px 0; border-bottom: 1px solid var(--line); }
    .freshness-row { display:grid; grid-template-columns: minmax(160px, 1.4fr) 96px minmax(180px, 1fr); gap: 12px; align-items:center; padding: 10px 0; border-bottom: 1px solid var(--line); }
    .eta-row { display:grid; grid-template-columns: minmax(160px, 1.2fr) minmax(220px, 1.4fr) minmax(190px, 1fr); gap: 12px; align-items:center; padding: 10px 0; border-bottom: 1px solid var(--line); }
    .bar-track { height: 12px; border-radius: 999px; background: rgba(15,118,110,.10); overflow:hidden; margin: 6px 0; }
    .bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
    .day-counts { display:flex; gap: 10px; flex-wrap: wrap; }
    .day-counts span { color: var(--muted); font-size: 12px; }
    .day-table { margin-top: 12px; }
    .sync-panel { display:none; margin-top: 14px; padding: 14px; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,.62); max-width: 980px; }
    .sync-panel.active { display:block; }
    .sync-top { display:flex; justify-content:space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; }
    .sync-messages { margin-top: 8px; display:grid; gap: 3px; }
    .status { min-height: 20px; color: var(--muted); font-size: 13px; }
    @media (max-width: 980px) { .grid, .two-col-panels, .freshness-row, .eta-row { grid-template-columns: 1fr; } header, main { padding-left:18px; padding-right:18px; } }
  </style>
</head>
<body>
  <header>
    <h1>GenUICraft Dataset Dashboard</h1>
    <p class="sub">Sync generated dataset runs from local folders, SSH servers, or custom copy commands. The mirror only downloads files whose source size/mtime changed.</p>
    <div class="toolbar">
      <button id="syncBtn">Sync sources</button>
      <button id="refreshBtn">Refresh scan</button>
      <button class="ghost-btn" id="exportCsvBtn">Export CSV</button>
      <button class="ghost-btn" id="exportJsonBtn">Export JSON</button>
      <input id="filter" placeholder="Filter run/source/model..." />
      <select id="sourceFilter">
        <option value="">All sources</option>
      </select>
      <select id="scoreFilter">
        <option value="">All scores</option>
        <option value="80">Score >= 80</option>
        <option value="70">Score >= 70</option>
        <option value="60">Score >= 60</option>
      </select>
      <select id="issueFilter">
        <option value="">All run health</option>
        <option value="backlog">Has backlog</option>
        <option value="quality">Has quality issues</option>
        <option value="low_coverage">Low content coverage</option>
        <option value="low_media">Low media usage</option>
      </select>
      <select id="irVersionFilter">
        <option value="">All IR versions</option>
      </select>
      <select id="sortBy">
        <option value="updated_desc">Sort: newest</option>
        <option value="score_desc">Sort: score high</option>
        <option value="score_asc">Sort: score low</option>
        <option value="ir_desc">Sort: IR count</option>
        <option value="backlog_desc">Sort: backlog</option>
        <option value="quality_desc">Sort: quality issues</option>
        <option value="source_run">Sort: source/run</option>
      </select>
      <label class="date-label">From <input id="dateFrom" type="date" title="From date" /></label>
      <label class="date-label">To <input id="dateTo" type="date" title="To date" /></label>
      <label class="date-label">Auto refresh
        <select id="autoRefreshInterval" title="Auto refresh interval">
          <option value="0">Off</option>
          <option value="15">15s</option>
          <option value="30" selected>30s</option>
          <option value="60">60s</option>
          <option value="300">5m</option>
        </select>
      </label>
      <span class="status" id="status"></span>
    </div>
    <div class="sync-panel" id="syncPanel">
      <div class="sync-top">
        <b id="syncPhase">Sync idle</b>
        <span class="small" id="syncCounters"></span>
      </div>
      <div class="bar-track"><div class="bar-fill" id="syncBar" style="width:0%"></div></div>
      <div class="small" id="syncFile"></div>
      <div class="sync-messages small" id="syncMessages"></div>
    </div>
  </header>
  <main>
    <section class="stats" id="stats"></section>
    <section class="two-col-panels wide-panel">
      <section class="panel">
        <h2>Sync Configuration</h2>
        <div id="syncConfig"></div>
      </section>
      <section class="panel">
        <h2>Last Sync Results</h2>
        <div id="lastSyncResults"></div>
      </section>
    </section>
    <section class="panel wide-panel">
      <h2>Freshness</h2>
      <div id="freshness"></div>
    </section>
    <section class="grid">
      <aside class="panel">
        <h2>Sources</h2>
        <div id="sources"></div>
      </aside>
      <section class="panel">
        <h2>Runs</h2>
        <div class="pager">
          <span class="small" id="runPageInfo"></span>
          <div class="pager-controls">
            <label class="date-label">Rows
              <select id="pageSize">
                <option value="25">25</option>
                <option value="50" selected>50</option>
                <option value="100">100</option>
                <option value="250">250</option>
                <option value="500">500</option>
              </select>
            </label>
            <button class="mini-btn ghost-btn" id="prevPageBtn">Prev</button>
            <button class="mini-btn ghost-btn" id="nextPageBtn">Next</button>
          </div>
        </div>
        <div class="scroll">
          <table>
            <thead>
              <tr>
                <th>Source / Run</th>
                <th>Counts</th>
                <th>IR Score</th>
                <th>Models</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody id="runs"></tbody>
          </table>
        </div>
        <div id="runDetails" class="detail-panel"></div>
      </section>
    </section>
    <section class="two-col-panels wide-panel">
      <section class="panel">
        <h2>Backlog</h2>
        <div id="backlog"></div>
      </section>
      <section class="panel">
        <h2>Quality Alerts</h2>
        <div id="qualityAlerts"></div>
      </section>
    </section>
    <section class="panel wide-panel">
      <h2>Filtered Metrics Overview</h2>
      <div id="metricsOverview"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Intent Quality</h2>
      <div id="intentQuality"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Throughput And ETA</h2>
      <div id="throughputEta"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Storage And Artifacts</h2>
      <div id="storageArtifacts"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Model Comparison</h2>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Response -> IR</th>
              <th>Runs / IR</th>
              <th>Score</th>
              <th>Quality Signals</th>
            </tr>
          </thead>
          <tbody id="modelComparison"></tbody>
        </table>
      </div>
    </section>
    <section class="panel wide-panel">
      <h2>Filtered Distribution</h2>
      <div id="distribution" class="dist-grid"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Worst Sampled IR Records</h2>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Sample</th>
              <th>Run</th>
              <th>Score</th>
              <th>Issues</th>
              <th>Model / Prompt</th>
              <th>Warnings</th>
            </tr>
          </thead>
          <tbody id="worstSamples"></tbody>
        </table>
      </div>
    </section>
    <section class="panel wide-panel">
      <h2>Score And Volume Trend</h2>
      <div id="trend" class="chart-wrap"></div>
    </section>
    <section class="panel wide-panel">
      <h2 id="daysTitle">Day Wise Data</h2>
      <div id="days"></div>
    </section>
  </main>
  <script>
    let current = null;
    let syncPollTimer = null;
    let autoRefreshTimer = null;
    let summaryLoading = false;
    let sourceHealthOverrides = {};
    let selectedRunKey = null;
    let runPage = 1;
    const fmt = n => (n ?? 0).toLocaleString();
    const pct = n => n == null ? "n/a" : `${(Number(n) * 100).toFixed(1)}%`;
    const metricPct = n => n == null ? "n/a" : `${(Number(n) * 100).toFixed(0)}%`;
    const scoreClass = s => s == null ? "" : s >= 75 ? "good" : s >= 60 ? "warn" : "";
    const scoreText = s => s == null ? "n/a" : Number(s).toFixed(2);
    const dominantModel = obj => Object.entries(obj || {})[0]?.[0] || "unknown";
    const runKey = r => `${r.source_id}::${r.run_id}`;
    const timestampMs = value => {
      const ms = Date.parse(value || "");
      return Number.isFinite(ms) ? ms : null;
    };
    const ageHours = value => {
      const ms = timestampMs(value);
      if (ms == null) return null;
      return Math.max(0, (Date.now() - ms) / 36e5);
    };
    const ageText = value => {
      const hours = ageHours(value);
      if (hours == null) return "unknown";
      if (hours < 1) return `${Math.max(1, Math.round(hours * 60))}m ago`;
      if (hours < 48) return `${hours.toFixed(hours < 10 ? 1 : 0)}h ago`;
      return `${(hours / 24).toFixed(hours < 24 * 10 ? 1 : 0)}d ago`;
    };
    const freshnessClass = value => {
      const hours = ageHours(value);
      if (hours == null) return "unknown";
      if (hours > 24 * 30) return "error";
      if (hours > 24 * 7) return "warn";
      return "ok";
    };
    const freshnessLabel = value => {
      const hours = ageHours(value);
      if (hours == null) return "unknown";
      if (hours > 24 * 30) return "very stale";
      if (hours > 24 * 7) return "stale";
      return "fresh";
    };
    const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const bytesText = value => {
      const n = Number(value || 0);
      if (n >= 1024 * 1024 * 1024) return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
      if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
      if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
      return `${fmt(n)} B`;
    };
    const modelText = obj => {
      const entries = Object.entries(obj || {}).slice(0, 3);
      return entries.length ? entries.map(([k,v]) => `${k} (${v})`).join("<br>") : "<span class='small'>n/a</span>";
    };
    const versionText = obj => {
      const entries = Object.entries(obj || {}).slice(0, 3);
      return entries.length ? entries.map(([k,v]) => `${k} (${v})`).join("<br>") : "<span class='small'>n/a</span>";
    };
    function setStatus(text) { document.getElementById("status").textContent = text || ""; }
    function renderSyncStatus(s) {
      const panel = document.getElementById("syncPanel");
      const active = s && (s.running || (s.phase && s.phase !== "idle"));
      panel.classList.toggle("active", Boolean(active));
      if (!active) return;
      const total = Number(s.total || 0);
      const processed = Number(s.processed || 0);
      const width = total ? Math.max(3, Math.min(100, Math.round((processed / total) * 100))) : (s.running ? 8 : 100);
      const sourcePart = s.source_label ? `${s.source_label}` : "sources";
      const sourceIndex = s.total_sources ? `source ${s.source_index || 0}/${s.total_sources}` : "";
      document.getElementById("syncPhase").textContent = `${s.running ? "Syncing" : "Sync"} - ${s.phase || "status"} ${sourceIndex}`;
      document.getElementById("syncCounters").textContent = [
        sourcePart,
        `listed ${fmt(s.listed)}`,
        `processed ${fmt(processed)}/${fmt(total)}`,
        `copied ${fmt(s.copied)}`,
        `skipped ${fmt(s.skipped)}`,
        `errors ${fmt(s.error_count)}`,
      ].filter(Boolean).join(" | ");
      document.getElementById("syncBar").style.width = `${width}%`;
      document.getElementById("syncFile").textContent = s.current_file ? `Current: ${s.current_file}` : (s.message || "");
      document.getElementById("syncMessages").innerHTML = (s.messages || []).slice(-5).map(m => `<div>${m}</div>`).join("");
    }
    async function loadSyncStatus() {
      const res = await fetch("/api/sync/status");
      const status = await res.json();
      renderSyncStatus(status);
      if (!status.running && syncPollTimer) {
        clearInterval(syncPollTimer);
        syncPollTimer = null;
      }
      return status;
    }
    function startSyncPolling() {
      if (syncPollTimer) clearInterval(syncPollTimer);
      loadSyncStatus().catch(() => {});
      syncPollTimer = setInterval(() => loadSyncStatus().catch(() => {}), 1000);
    }
    async function loadSummary(options = {}) {
      if (summaryLoading) return;
      summaryLoading = true;
      const silent = Boolean(options.silent);
      if (!silent) setStatus("Loading...");
      try {
        const res = await fetch("/api/summary");
        current = await res.json();
        render();
        setStatus(`${silent ? "Auto refreshed" : "Loaded"} ${new Date(current.generated_at).toLocaleString()}`);
      } finally {
        summaryLoading = false;
      }
    }
    function startAutoRefresh() {
      if (autoRefreshTimer) {
        clearInterval(autoRefreshTimer);
        autoRefreshTimer = null;
      }
      const seconds = Number(document.getElementById("autoRefreshInterval").value || "0");
      if (!seconds) {
        setStatus(current ? `Loaded ${new Date(current.generated_at).toLocaleString()} | auto refresh off` : "Auto refresh off");
        return;
      }
      autoRefreshTimer = setInterval(() => {
        if (document.hidden) return;
        loadSummary({silent: true}).catch(e => setStatus(`Auto refresh failed: ${e.message}`));
      }, seconds * 1000);
      if (current) setStatus(`Loaded ${new Date(current.generated_at).toLocaleString()} | auto refresh ${seconds}s`);
    }
    async function syncSources() {
      setStatus("Syncing sources...");
      startSyncPolling();
      try {
        const res = await fetch("/api/sync", {method: "POST"});
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "sync failed");
        await loadSyncStatus().catch(() => {});
        await loadSummary();
        const copied = (payload.results || []).reduce((a,r) => a + (r.copied || 0), 0);
        setStatus(`Sync complete. Copied ${copied} changed files.`);
      } catch (error) {
        await loadSyncStatus().catch(() => {});
        throw error;
      }
    }
    async function testSource(sourceId) {
      setStatus(`Testing ${sourceId}...`);
      const res = await fetch("/api/source/test", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({source_id: sourceId}),
      });
      const payload = await res.json();
      if (!res.ok && !payload.status) throw new Error(payload.error || "source test failed");
      sourceHealthOverrides[sourceId] = payload;
      render();
      setStatus(`${payload.source_label || sourceId}: ${payload.status} - ${payload.message || ""}`);
    }
    function downloadBlob(filename, content, type) {
      const blob = new Blob([content], {type});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }
    function csvCell(value) {
      const text = String(value ?? "");
      return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    }
    function exportFiltered(format) {
      if (!current) return;
      const runs = filteredRuns();
      const sourceStats = filteredSourceStats(current.sources || [], runs);
      const payload = {
        exported_at: new Date().toISOString(),
        generated_at: current.generated_at,
        filters: filters(),
        totals: runTotals(runs),
        sources: sourceStats,
        runs,
      };
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      if (format === "json") {
        downloadBlob(`genuicraft_dataset_dashboard_${stamp}.json`, JSON.stringify(payload, null, 2), "application/json");
        setStatus(`Exported ${fmt(runs.length)} runs as JSON`);
        return;
      }
      const headers = [
        "source_id","source_label","run_id","queries","responses","genui","missing_responses","missing_ir",
        "overall_score","assets","screenshots","total_bytes","file_count","missing_core_files","updated_at","response_model","ir_model","ir_versions","path",
      ];
      const rows = runs.map(r => [
        r.source_id,
        r.source_label,
        r.run_id,
        r.queries,
        r.responses,
        r.genui,
        r.response_backlog,
        r.ir_backlog,
        r.display_score ?? r.overall_score,
        r.assets,
        r.screenshots,
        r.total_bytes,
        r.file_count,
        (r.missing_core_files || []).join("; "),
        r.updated_at,
        dominantModel(r.response_models),
        dominantModel(r.ir_models),
        Object.keys(r.ir_versions || {}).join("; "),
        r.path,
      ]);
      const csv = [headers, ...rows].map(row => row.map(csvCell).join(",")).join("\n");
      downloadBlob(`genuicraft_dataset_dashboard_${stamp}.csv`, csv, "text/csv");
      setStatus(`Exported ${fmt(runs.length)} runs as CSV`);
    }
    function renderStats(t) {
      const stats = [
        ["Sources", t.sources],
        ["Runs", t.runs],
        ["Queries", t.queries],
        ["Responses", t.responses],
        ["IR records", t.genui],
        ["Missing responses", t.response_backlog],
        ["Missing IR", t.ir_backlog],
        ["Pipeline complete", pct(t.completion_rate)],
        ["Assets", t.assets],
        ["Screenshots", t.screenshots],
        ["Storage", bytesText(t.total_bytes)],
        ["Files", t.file_count],
        ["Missing core runs", t.missing_core_run_count],
      ];
      document.getElementById("stats").innerHTML = stats.map(([k,v]) => `<div class="stat"><div class="v">${fmt(v)}</div><div class="k">${k}</div></div>`).join("");
    }
    function pills(items) {
      return `<div class="pill-row">${(items || []).map(item => `<span class="badge">${escapeHtml(item)}</span>`).join("") || "<span class='small'>none</span>"}</div>`;
    }
    function renderSyncConfig(config) {
      if (!config) {
        document.getElementById("syncConfig").innerHTML = "<span class='small'>No config summary available.</span>";
        return;
      }
      const sources = (config.sources || []).map(source => {
        const meta = [
          `${source.type}`,
          source.enabled ? "enabled" : "disabled",
          source.path_match ? `match ${source.path_match}` : "",
          source.host ? `host ${source.host}` : "",
          source.user ? `user ${source.user}` : "",
          source.port ? `port ${source.port}` : "",
          source.identity_file ? `key ${source.identity_file}` : "",
          source.proxy_jump ? `jump ${source.proxy_jump}` : "",
          source.has_proxy_command ? "proxy command" : "",
          source.has_password ? "password in config" : "",
          source.ask_password ? "prompt password" : "",
        ].filter(Boolean);
        return `
          <div class="config-source">
            <b>${escapeHtml(source.source_label)}</b>
            ${pills(meta)}
            <div class="small">path: ${escapeHtml(source.path || "n/a")}</div>
            ${source.path_base ? `<div class="small">path_base: ${escapeHtml(source.path_base)}</div>` : ""}
            ${source.proxy_command_preview ? `<div class="small">proxy: ${escapeHtml(source.proxy_command_preview)}</div>` : ""}
            ${(source.ssh_options || []).length ? `<div class="small">ssh options: ${escapeHtml((source.ssh_options || []).join(" | "))}</div>` : ""}
          </div>`;
      }).join("");
      document.getElementById("syncConfig").innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(config.effective_source_count)}</b><br><span class="small">effective sources</span></div>
          <div class="detail-box"><b>${fmt(config.enabled_source_count)}</b><br><span class="small">enabled sources</span></div>
          <div class="detail-box"><b>${fmt(config.configured_source_count)}</b><br><span class="small">configured sources</span></div>
        </div>
        <div class="small">mirror: ${escapeHtml(config.mirror_dir || "")}</div>
        <div class="small">include globs:</div>${pills(config.include_globs)}
        <div class="small">exclude globs:</div>${pills(config.exclude_globs)}
        ${sources}
      `;
    }
    function renderLastSyncResults(lastSync) {
      if (!lastSync || !(lastSync.results || []).length) {
        document.getElementById("lastSyncResults").innerHTML = "<span class='small'>No sync has been recorded yet.</span>";
        return;
      }
      const rows = (lastSync.results || []).map(result => {
        const errors = result.errors || [];
        const status = (result.error_count || 0) ? "error" : "ok";
        return `
          <tr>
            <td><b>${escapeHtml(result.source_label || result.source_id)}</b><br><span class="badge ${status}">${status}</span></td>
            <td>listed ${fmt(result.listed)}<br>copied ${fmt(result.copied)}<br>skipped ${fmt(result.skipped)}</td>
            <td>${fmt(result.error_count || 0)}<br><span class="small">${errors.length ? escapeHtml(errors[0]) : "no errors"}</span></td>
            <td><span class="small">${escapeHtml(result.mirror_path || "")}</span></td>
          </tr>`;
      }).join("");
      document.getElementById("lastSyncResults").innerHTML = `
        <div class="small">synced at: ${lastSync.synced_at ? new Date(lastSync.synced_at).toLocaleString() : "unknown"}</div>
        <div class="scroll">
          <table>
            <thead><tr><th>Source</th><th>Files</th><th>Errors</th><th>Mirror</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }
    function freshnessBadge(value) {
      return `<span class="badge ${freshnessClass(value)}">${freshnessLabel(value)}</span>`;
    }
    function renderFreshness(runs, sources, lastSync) {
      const target = document.getElementById("freshness");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const sortedRuns = [...runs].sort((a, b) => (timestampMs(b.updated_at) || 0) - (timestampMs(a.updated_at) || 0));
      const latestRun = sortedRuns[0];
      const staleRuns = runs.filter(r => (ageHours(r.updated_at) ?? Infinity) > 24 * 7);
      const veryStaleRuns = runs.filter(r => (ageHours(r.updated_at) ?? Infinity) > 24 * 30);
      const noRecentIrRuns = runs.filter(r => (r.responses || 0) > (r.genui || 0));
      const staleSources = [...sources]
        .filter(s => !s.latest_run_updated_at || (ageHours(s.latest_run_updated_at) ?? Infinity) > 24 * 7)
        .sort((a, b) => (ageHours(b.latest_run_updated_at) ?? Infinity) - (ageHours(a.latest_run_updated_at) ?? Infinity));
      const oldestRuns = [...runs]
        .sort((a, b) => (timestampMs(a.updated_at) || Infinity) - (timestampMs(b.updated_at) || Infinity))
        .slice(0, 8);
      const sourceRows = staleSources.slice(0, 8).map(s => `
        <div class="freshness-row">
          <div><b>${escapeHtml(s.source_label)}</b><br><span class="small">${escapeHtml(s.local_path || "")}</span></div>
          <div>${freshnessBadge(s.latest_run_updated_at)}<br><span class="small">${ageText(s.latest_run_updated_at)}</span></div>
          <div class="small">${fmt(s.run_count)} runs | Q ${fmt(s.queries)} R ${fmt(s.responses)} IR ${fmt(s.genui)} | missing IR ${fmt(s.ir_backlog)}</div>
        </div>`).join("");
      const runRows = oldestRuns.map(r => `
        <div class="freshness-row">
          <div><b>${escapeHtml(r.run_id)}</b><br><span class="small">${escapeHtml(r.source_label)}</span></div>
          <div>${freshnessBadge(r.updated_at)}<br><span class="small">${ageText(r.updated_at)}</span></div>
          <div class="small">Q ${fmt(r.queries)} R ${fmt(r.responses)} IR ${fmt(r.genui)} | missing R ${fmt(r.response_backlog)} | missing IR ${fmt(r.ir_backlog)}</div>
        </div>`).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${latestRun ? ageText(latestRun.updated_at) : "unknown"}</b><br><span class="small">latest run update</span></div>
          <div class="detail-box"><b>${lastSync?.synced_at ? ageText(lastSync.synced_at) : "unknown"}</b><br><span class="small">last source sync</span></div>
          <div class="detail-box"><b>${fmt(staleRuns.length)}</b><br><span class="small">runs older than 7d</span></div>
          <div class="detail-box"><b>${fmt(veryStaleRuns.length)}</b><br><span class="small">runs older than 30d</span></div>
          <div class="detail-box"><b>${fmt(staleSources.length)}</b><br><span class="small">sources stale/unknown</span></div>
          <div class="detail-box"><b>${fmt(noRecentIrRuns.length)}</b><br><span class="small">runs with IR backlog</span></div>
        </div>
        <h2>Stale Sources</h2>
        <div class="warning-list">${sourceRows || "<span class='small'>No stale sources in current filters.</span>"}</div>
        <h2>Oldest Matching Runs</h2>
        <div class="warning-list">${runRows || "<span class='small'>No run freshness data.</span>"}</div>
        <div class="small">Freshness follows current source, date, score, issue, text, and IR-version filters. Stale means no matching run update in 7 days; very stale means 30 days.</div>
      `;
    }
    function getPageSize() {
      return Math.max(1, Number(document.getElementById("pageSize").value || "50"));
    }
    function filters() {
      return {
        text: document.getElementById("filter").value.toLowerCase().trim(),
        minScore: Number(document.getElementById("scoreFilter").value || "0"),
        issue: document.getElementById("issueFilter").value,
        sourceId: document.getElementById("sourceFilter").value,
        irVersion: document.getElementById("irVersionFilter").value,
        sortBy: document.getElementById("sortBy").value,
        dateFrom: document.getElementById("dateFrom").value,
        dateTo: document.getElementById("dateTo").value,
      };
    }
    function qualityIssueCount(r) {
      const q = r.quality_summary || {};
      return ["json_parse_fail","strict_schema_fail","repair_needed","repair_attempted","gen_errors","fallback_generated","low_score","markdown_leakage","sparse_ir"]
        .reduce((sum, key) => sum + (q[key] || 0), 0);
    }
    function dateFilterActive(f) {
      return Boolean(f.dateFrom || f.dateTo);
    }
    function dayInRange(day, f) {
      if (!day) return false;
      if (f.dateFrom && day < f.dateFrom) return false;
      if (f.dateTo && day > f.dateTo) return false;
      return true;
    }
    function filteredRunDays(r, f) {
      const stats = f.irVersion ? (r.ir_version_stats || {})[f.irVersion] : null;
      const days = stats ? (stats.days || []) : (r.days || []);
      return days.filter(day => dayInRange(day.day, f));
    }
    function dayFilteredCounts(r, f) {
      const versionStats = f.irVersion ? (r.ir_version_stats || {})[f.irVersion] : null;
      if (!dateFilterActive(f)) {
        if (versionStats) {
          return {
            queries: versionStats.queries || 0,
            responses: versionStats.responses || 0,
            genui: versionStats.genui || 0,
            score_sum: versionStats.score_sum || 0,
            score_count: versionStats.score_count || 0,
          };
        }
        return {
          queries: r.queries || 0,
          responses: r.responses || 0,
          genui: r.genui || 0,
          score_sum: r.overall_score != null ? Number(r.overall_score) : 0,
          score_count: r.overall_score != null ? 1 : 0,
        };
      }
      return filteredRunDays(r, f).reduce((acc, day) => {
        acc.queries += day.queries || 0;
        acc.responses += day.responses || 0;
        acc.genui += day.genui || 0;
        acc.score_sum += day.score_sum || 0;
        acc.score_count += day.score_count || 0;
        return acc;
      }, {queries: 0, responses: 0, genui: 0, score_sum: 0, score_count: 0});
    }
    function filteredRunRecord(r, f) {
      const counts = dayFilteredCounts(r, f);
      const avg = counts.score_count ? counts.score_sum / counts.score_count : r.overall_score;
      return {
        ...r,
        queries: counts.queries,
        responses: counts.responses,
        genui: counts.genui,
        response_backlog: Math.max((counts.queries || 0) - (counts.responses || 0), 0),
        ir_backlog: Math.max((counts.responses || 0) - (counts.genui || 0), 0),
        completion_rate: counts.queries ? (counts.genui || 0) / counts.queries : ((counts.genui || 0) ? 1 : 0),
        display_score: avg,
      };
    }
    function runScoreForFilter(r, f) {
      if (!dateFilterActive(f)) return r.overall_score;
      const counts = dayFilteredCounts(r, f);
      return counts.score_count ? counts.score_sum / counts.score_count : r.overall_score;
    }
    function runMatches(r, f) {
      const hay = JSON.stringify([r.source_label, r.run_id, r.query_models, r.response_models, r.ir_models, r.ir_versions, r.intents]).toLowerCase();
      if (f.sourceId && r.source_id !== f.sourceId) return false;
      if (f.irVersion && !(r.ir_version_stats || {})[f.irVersion]) return false;
      if (f.text && !hay.includes(f.text)) return false;
      if (dateFilterActive(f) && !filteredRunDays(r, f).length) return false;
      if (f.issue) {
        const counts = dayFilteredCounts(r, f);
        const metrics = r.metric_avgs || {};
        if (f.issue === "backlog" && !Math.max((counts.queries || 0) - (counts.responses || 0), 0) && !Math.max((counts.responses || 0) - (counts.genui || 0), 0)) return false;
        if (f.issue === "quality" && !qualityIssueCount(r)) return false;
        if (f.issue === "low_coverage" && !(metrics.content_coverage != null && metrics.content_coverage < 0.65)) return false;
        if (f.issue === "low_media" && !((metrics.image_presence ?? 0) < 0.25 && (metrics.icon_presence ?? 0) < 0.25)) return false;
      }
      const score = runScoreForFilter(r, f);
      if (f.minScore && (score == null || score < f.minScore)) return false;
      return true;
    }
    function runBacklogTotal(r) {
      return (r.response_backlog || 0) + (r.ir_backlog || 0);
    }
    function sortRuns(runs, sortBy) {
      const sorted = [...runs];
      const scoreValue = r => r.display_score ?? r.overall_score ?? -1;
      const updatedValue = r => Date.parse(r.updated_at || "") || 0;
      const textValue = r => `${r.source_label || ""}/${r.run_id || ""}`.toLowerCase();
      sorted.sort((a, b) => {
        if (sortBy === "score_desc") return scoreValue(b) - scoreValue(a) || textValue(a).localeCompare(textValue(b));
        if (sortBy === "score_asc") return scoreValue(a) - scoreValue(b) || textValue(a).localeCompare(textValue(b));
        if (sortBy === "ir_desc") return (b.genui || 0) - (a.genui || 0) || textValue(a).localeCompare(textValue(b));
        if (sortBy === "backlog_desc") return runBacklogTotal(b) - runBacklogTotal(a) || textValue(a).localeCompare(textValue(b));
        if (sortBy === "quality_desc") return qualityIssueCount(b) - qualityIssueCount(a) || textValue(a).localeCompare(textValue(b));
        if (sortBy === "source_run") return textValue(a).localeCompare(textValue(b));
        return updatedValue(b) - updatedValue(a) || textValue(a).localeCompare(textValue(b));
      });
      return sorted;
    }
    function filteredRuns() {
      const f = filters();
      return sortRuns((current.runs || []).filter(r => runMatches(r, f)).map(r => filteredRunRecord(r, f)), f.sortBy);
    }
    function paginatedRuns(runs) {
      const size = getPageSize();
      const totalPages = Math.max(1, Math.ceil(runs.length / size));
      runPage = Math.min(Math.max(1, runPage), totalPages);
      const start = (runPage - 1) * size;
      const rows = runs.slice(start, start + size);
      return {
        rows,
        totalPages,
        start: rows.length ? start + 1 : 0,
        end: start + rows.length,
        total: runs.length,
      };
    }
    function resetPageAndRender() {
      runPage = 1;
      render();
    }
    function runTotals(runs) {
      const sourceIds = new Set(runs.map(r => r.source_id));
      return {
        sources: sourceIds.size,
        runs: runs.length,
        queries: runs.reduce((a,r) => a + (r.queries || 0), 0),
        responses: runs.reduce((a,r) => a + (r.responses || 0), 0),
        genui: runs.reduce((a,r) => a + (r.genui || 0), 0),
        response_backlog: runs.reduce((a,r) => a + (r.response_backlog || 0), 0),
        ir_backlog: runs.reduce((a,r) => a + (r.ir_backlog || 0), 0),
        completion_rate: runs.reduce((a,r) => a + (r.queries || 0), 0) ? runs.reduce((a,r) => a + (r.genui || 0), 0) / runs.reduce((a,r) => a + (r.queries || 0), 0) : 0,
        assets: runs.reduce((a,r) => a + (r.assets || 0), 0),
        screenshots: runs.reduce((a,r) => a + (r.screenshots || 0), 0),
        asset_bytes: runs.reduce((a,r) => a + (r.asset_bytes || 0), 0),
        screenshot_bytes: runs.reduce((a,r) => a + (r.screenshot_bytes || 0), 0),
        total_bytes: runs.reduce((a,r) => a + (r.total_bytes || 0), 0),
        file_count: runs.reduce((a,r) => a + (r.file_count || 0), 0),
        missing_core_run_count: runs.reduce((a,r) => a + ((r.missing_core_files || []).length ? 1 : 0), 0),
      };
    }
    function sortedDayBuckets(dayMap) {
      return [...dayMap.values()]
        .map(day => ({...day, avg_score: day.score_count ? day.score_sum / day.score_count : null}))
        .sort((a, b) => b.day.localeCompare(a.day));
    }
    function filteredSourceStats(sources, runs) {
      const emptySource = s => ({
        ...s,
        run_count: 0,
        queries: 0,
        responses: 0,
        genui: 0,
        response_backlog: 0,
        ir_backlog: 0,
        completion_rate: 0,
        assets: 0,
        screenshots: 0,
        asset_bytes: 0,
        screenshot_bytes: 0,
        total_bytes: 0,
        file_count: 0,
        missing_core_run_count: 0,
        avg_score: null,
        latest_run_updated_at: null,
        oldest_run_updated_at: null,
        _scoreSum: 0,
        _scoreCount: 0,
        _latestMs: null,
        _oldestMs: null,
        _dayMap: new Map(),
      });
      const bySource = new Map(sources.map(s => [s.source_id, emptySource(s)]));
      const f = filters();
      for (const r of runs) {
        if (!bySource.has(r.source_id)) {
          bySource.set(r.source_id, emptySource({source_id: r.source_id, source_label: r.source_label, type: "unknown", local_path: ""}));
        }
        const s = bySource.get(r.source_id);
        s.run_count += 1;
        s.queries += r.queries || 0;
        s.responses += r.responses || 0;
        s.genui += r.genui || 0;
        s.response_backlog += r.response_backlog || 0;
        s.ir_backlog += r.ir_backlog || 0;
        s.assets += r.assets || 0;
        s.screenshots += r.screenshots || 0;
        s.asset_bytes += r.asset_bytes || 0;
        s.screenshot_bytes += r.screenshot_bytes || 0;
        s.total_bytes += r.total_bytes || 0;
        s.file_count += r.file_count || 0;
        if ((r.missing_core_files || []).length) s.missing_core_run_count += 1;
        const score = r.display_score ?? r.overall_score;
        if (score != null) {
          s._scoreSum += Number(score);
          s._scoreCount += 1;
        }
        const updated = timestampMs(r.updated_at);
        if (updated != null) {
          if (s._latestMs == null || updated > s._latestMs) {
            s._latestMs = updated;
            s.latest_run_updated_at = r.updated_at;
          }
          if (s._oldestMs == null || updated < s._oldestMs) {
            s._oldestMs = updated;
            s.oldest_run_updated_at = r.updated_at;
          }
        }
        for (const day of filteredRunDays(r, f)) {
          if (!s._dayMap.has(day.day)) {
            s._dayMap.set(day.day, {day: day.day, queries: 0, responses: 0, genui: 0, score_sum: 0, score_count: 0, avg_score: null});
          }
          const target = s._dayMap.get(day.day);
          target.queries += day.queries || 0;
          target.responses += day.responses || 0;
          target.genui += day.genui || 0;
          target.score_sum += day.score_sum || 0;
          target.score_count += day.score_count || 0;
        }
      }
      const sourceId = document.getElementById("sourceFilter").value;
      const rows = [...bySource.values()].filter(s => sourceId ? s.source_id === sourceId : s.run_count > 0);
      return rows.map(s => ({...s, avg_score: s._scoreCount ? s._scoreSum / s._scoreCount : null, completion_rate: s.queries ? s.genui / s.queries : 0, days: sortedDayBuckets(s._dayMap)}));
    }
    function healthForSource(s) {
      return sourceHealthOverrides[s.source_id] || s.health || {status: "unknown", message: "Not tested"};
    }
    function renderHealth(h) {
      const status = h.status || "unknown";
      const checked = h.checked_at ? `<br><span class="small">checked ${new Date(h.checked_at).toLocaleString()}</span>` : "";
      const sync = h.last_synced_at ? `<br><span class="small">synced ${new Date(h.last_synced_at).toLocaleString()}</span>` : "";
      return `<span class="badge ${status}">${status}</span><div class="small">${h.message || ""}${checked}${sync}</div>`;
    }
    function renderSources(sources) {
      document.getElementById("sources").innerHTML = sources.map(s => `
        <div class="source-card">
          <div class="source-head">
            <div>
              <strong>${s.source_label}</strong>
              <span class="badge">${s.type}</span>
              ${renderHealth(healthForSource(s))}
            </div>
            <button class="mini-btn" onclick="testSource('${s.source_id}')">Test</button>
          </div>
          <div class="small">${s.local_path}</div>
          <div class="small">latest run: ${s.latest_run_updated_at ? `${ageText(s.latest_run_updated_at)} (${new Date(s.latest_run_updated_at).toLocaleString()})` : "unknown"}</div>
          <div class="kv">
            <div><b>${fmt(s.run_count)}</b><br><span class="small">runs</span></div>
            <div><b>${fmt(s.queries)}</b><br><span class="small">queries</span></div>
            <div><b>${fmt(s.responses)}</b><br><span class="small">responses</span></div>
            <div><b>${fmt(s.genui)}</b><br><span class="small">IR</span></div>
            <div><b>${fmt(s.response_backlog)}</b><br><span class="small">missing responses</span></div>
            <div><b>${fmt(s.ir_backlog)}</b><br><span class="small">missing IR</span></div>
            <div><b>${scoreText(s.avg_score)}</b><br><span class="small">avg score</span></div>
            <div><b>${pct(s.completion_rate)}</b><br><span class="small">complete</span></div>
            <div><b>${bytesText(s.total_bytes)}</b><br><span class="small">storage</span></div>
            <div><b>${fmt(s.file_count)}</b><br><span class="small">files</span></div>
          </div>
        </div>`).join("");
    }
    function renderSourceFilter(sources) {
      const select = document.getElementById("sourceFilter");
      const selected = select.value;
      const options = sources.map(s => `<option value="${s.source_id}">${s.source_label} (${fmt(s.run_count)})</option>`);
      select.innerHTML = `<option value="">All sources</option>${options.join("")}`;
      if ([...select.options].some(o => o.value === selected)) select.value = selected;
    }
    function renderIrVersionFilter(versions) {
      const select = document.getElementById("irVersionFilter");
      const selected = select.value;
      const options = Object.entries(versions || {}).map(([version, count]) => `<option value="${version}">${version} (${fmt(count)})</option>`);
      select.innerHTML = `<option value="">All IR versions</option>${options.join("")}`;
      if ([...select.options].some(o => o.value === selected)) select.value = selected;
    }
    function renderRuns(runs, pageInfo) {
      document.getElementById("runPageInfo").textContent = pageInfo.total
        ? `Showing ${fmt(pageInfo.start)}-${fmt(pageInfo.end)} of ${fmt(pageInfo.total)} filtered runs, page ${fmt(runPage)} of ${fmt(pageInfo.totalPages)}`
        : "No runs match current filters";
      document.getElementById("prevPageBtn").disabled = runPage <= 1;
      document.getElementById("nextPageBtn").disabled = runPage >= pageInfo.totalPages;
      document.getElementById("runs").innerHTML = runs.map(r => `
        <tr>
          <td>
            <b>${escapeHtml(r.run_id)}</b><br>
            <span class="small">${escapeHtml(r.source_label)}</span><br>
            <span class="small">${escapeHtml(r.path)}</span><br>
            <button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(r))}')">Details</button>
          </td>
          <td>
            Q ${fmt(r.queries)}<br>R ${fmt(r.responses)}<br>IR ${fmt(r.genui)}<br>
            <span class="small">missing R ${fmt(r.response_backlog)} | missing IR ${fmt(r.ir_backlog)}</span><br>
            <span class="small">assets ${fmt(r.assets)} | shots ${fmt(r.screenshots)}</span>
          </td>
          <td><span class="score ${scoreClass(r.display_score ?? r.overall_score)}">${scoreText(r.display_score ?? r.overall_score)}</span></td>
          <td>
            <span class="small">Stage 2</span><br>${modelText(r.response_models)}
            <br><span class="small">Stage 3</span><br>${modelText(r.ir_models)}
            <br><span class="small">IR version</span><br>${versionText(r.ir_versions)}
          </td>
          <td><span class="small">${new Date(r.updated_at).toLocaleString()}</span></td>
        </tr>`).join("");
    }
    function selectRun(key) {
      selectedRunKey = decodeURIComponent(key);
      render();
    }
    function copyText(value) {
      navigator.clipboard.writeText(decodeURIComponent(value)).then(
        () => setStatus("Path copied"),
        () => setStatus("Copy failed")
      );
    }
    function qualityBox(label, value) {
      return `<div class="detail-box"><b>${fmt(value)}</b><br><span class="small">${label}</span></div>`;
    }
    function renderArtifactRows(rows, label) {
      if (!rows || !rows.length) return `<span class="small">No ${label} found.</span>`;
      return `<div class="artifact-list">${rows.map(item => `
        <div class="artifact-row">
          <b>${escapeHtml(item.label || label)}</b>
          <span class="small">${escapeHtml(item.name)}<br>${escapeHtml(item.path)}</span>
          <button class="mini-btn ghost-btn" onclick="copyText('${encodeURIComponent(item.path)}')">${bytesText(item.size)}</button>
        </div>`).join("")}</div>`;
    }
    function renderRunDetails(runs) {
      const el = document.getElementById("runDetails");
      if (!runs.length) {
        selectedRunKey = null;
        el.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const selected = selectedRunKey ? runs.find(r => runKey(r) === selectedRunKey) : null;
      if (!selected) {
        el.innerHTML = "<span class='small'>Select Details on a run to inspect metrics, warnings, models, intents, and versions.</span>";
        return;
      }
      const q = selected.quality_summary || {};
      const m = selected.metric_avgs || {};
      const artifacts = selected.artifacts || {};
      const warnings = (q.warnings || []).map(w => `
        <div class="warning-row">
          <span>${escapeHtml(w.message)}</span>
          <b>${fmt(w.count)}</b>
        </div>`).join("");
      el.innerHTML = `
        <h2>${escapeHtml(selected.run_id)}</h2>
        <div class="small">${escapeHtml(selected.path)}</div>
        <div class="detail-grid">
          ${qualityBox("sampled IR rows", q.sampled || 0)}
          ${qualityBox("strict schema fail", q.strict_schema_fail || 0)}
          ${qualityBox("repair attempted", q.repair_attempted || 0)}
          ${qualityBox("generation errors", q.gen_errors || 0)}
          ${qualityBox("low score rows", q.low_score || 0)}
          ${qualityBox("markdown leakage rows", q.markdown_leakage || 0)}
          ${qualityBox("sparse IR rows", q.sparse_ir || 0)}
          <div class="detail-box"><b>${scoreText(selected.display_score ?? selected.overall_score)}</b><br><span class="small">filtered score</span></div>
          <div class="detail-box"><b>${bytesText(selected.total_bytes)}</b><br><span class="small">storage</span></div>
          <div class="detail-box"><b>${fmt(selected.file_count)}</b><br><span class="small">files</span></div>
          <div class="detail-box"><b>${(selected.missing_core_files || []).length ? escapeHtml(selected.missing_core_files.join(", ")) : "none"}</b><br><span class="small">missing core files</span></div>
        </div>
        <div class="detail-grid">
          <div class="detail-box"><b>Stage 2</b><br><span class="small">${modelText(selected.response_models)}</span></div>
          <div class="detail-box"><b>Stage 3</b><br><span class="small">${modelText(selected.ir_models)}</span></div>
          <div class="detail-box"><b>IR versions</b><br><span class="small">${versionText(selected.ir_versions)}</span></div>
          <div class="detail-box"><b>Intents</b><br><span class="small">${modelText(selected.intents)}</span></div>
        </div>
        <div class="detail-grid">
          <div class="detail-box"><b>${metricPct(m.content_coverage)}</b><br><span class="small">content coverage</span></div>
          <div class="detail-box"><b>${metricPct(m.intent_score)}</b><br><span class="small">intent score</span></div>
          <div class="detail-box"><b>${metricPct(m.section_heading_coverage)}</b><br><span class="small">heading coverage</span></div>
          <div class="detail-box"><b>${metricPct(m.table_cell_coverage)}</b><br><span class="small">table coverage</span></div>
          <div class="detail-box"><b>${metricPct(m.action_coverage)}</b><br><span class="small">action coverage</span></div>
          <div class="detail-box"><b>${metricPct(m.image_presence)}</b><br><span class="small">image presence</span></div>
        </div>
        <h2>Validation Warnings</h2>
        <div class="warning-list">${warnings || "<span class='small'>No sampled validation warnings.</span>"}</div>
        <h2>Artifacts</h2>
        <div class="detail-grid">
          <div class="detail-box"><b>Core files</b>${renderArtifactRows(artifacts.core || [], "file")}</div>
          <div class="detail-box"><b>Screenshot samples</b>${renderArtifactRows(artifacts.screenshot_samples || [], "screenshot")}</div>
          <div class="detail-box"><b>Asset samples</b>${renderArtifactRows(artifacts.asset_samples || [], "asset")}</div>
        </div>
      `;
    }
    function totalDayCount(day) {
      return (day.queries || 0) + (day.responses || 0) + (day.genui || 0);
    }
    function renderBacklog(sources) {
      const rows = [...sources]
        .map(s => ({
          ...s,
          total_backlog: (s.response_backlog || 0) + (s.ir_backlog || 0),
        }))
        .sort((a,b) => (b.total_backlog - a.total_backlog) || String(a.source_label).localeCompare(String(b.source_label)));
      if (!rows.length) {
        document.getElementById("backlog").innerHTML = "<span class='small'>No source data found for current filters.</span>";
        return;
      }
      const maxBacklog = Math.max(1, ...rows.map(r => r.total_backlog));
      document.getElementById("backlog").innerHTML = rows.map(row => {
        const width = Math.round((row.total_backlog / maxBacklog) * 100);
        return `
          <div class="day-row">
            <div><b>${row.source_label}</b><br><span class="small">${fmt(row.run_count)} runs</span></div>
            <div>
              <div class="bar-track"><div class="bar-fill" style="width:${Math.max(3, width)}%"></div></div>
              <div class="day-counts">
                <span>Q ${fmt(row.queries)}</span>
                <span>R ${fmt(row.responses)}</span>
                <span>IR ${fmt(row.genui)}</span>
                <span>missing R ${fmt(row.response_backlog)}</span>
                <span>missing IR ${fmt(row.ir_backlog)}</span>
              </div>
            </div>
            <div><b>${fmt(row.total_backlog)}</b><br><span class="small">backlog</span></div>
          </div>`;
      }).join("");
    }
    function dayMs(day) {
      const ms = Date.parse(`${day}T23:59:59Z`);
      return Number.isFinite(ms) ? ms : null;
    }
    function windowStats(days, windowDays = 7) {
      const cutoff = Date.now() - windowDays * 86400000;
      const rows = (days || []).filter(day => {
        const ms = dayMs(day.day);
        return ms != null && ms >= cutoff;
      });
      const totals = rows.reduce((acc, day) => {
        acc.queries += day.queries || 0;
        acc.responses += day.responses || 0;
        acc.genui += day.genui || 0;
        acc.score_sum += day.score_sum || 0;
        acc.score_count += day.score_count || 0;
        return acc;
      }, {queries: 0, responses: 0, genui: 0, score_sum: 0, score_count: 0});
      return {
        ...totals,
        active_days: rows.length,
        window_days: windowDays,
        response_per_day: totals.responses / windowDays,
        ir_per_day: totals.genui / windowDays,
        query_per_day: totals.queries / windowDays,
        avg_score: totals.score_count ? totals.score_sum / totals.score_count : null,
      };
    }
    function rateText(value) {
      const n = Number(value || 0);
      return n >= 10 ? n.toFixed(0) : n.toFixed(1);
    }
    function etaText(backlog, perDay) {
      const missing = Number(backlog || 0);
      const rate = Number(perDay || 0);
      if (missing <= 0) return "cleared";
      if (rate <= 0) return "no recent rate";
      const days = missing / rate;
      if (days < 1) return "<1d";
      if (days < 30) return `${days.toFixed(days < 10 ? 1 : 0)}d`;
      if (days < 365) return `${(days / 30).toFixed(1)}mo`;
      return ">1y";
    }
    function bottleneckText(source) {
      const responseBacklog = Number(source.response_backlog || 0);
      const irBacklog = Number(source.ir_backlog || 0);
      if (responseBacklog <= 0 && irBacklog <= 0) return "clear";
      if (responseBacklog >= irBacklog) return "Stage 2";
      return "Stage 3";
    }
    function renderThroughputEta(runs, sources) {
      const target = document.getElementById("throughputEta");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const days = aggregateDaysFromRuns(runs);
      if (!days.length) {
        target.innerHTML = "<span class='small'>No dated buckets found for current filters.</span>";
        return;
      }
      const totals = runTotals(runs);
      const stats = windowStats(days, 7);
      const latestDay = days[0]?.day || "unknown";
      const sourceRows = [...sources]
        .map(source => {
          const rate = windowStats(source.days || [], 7);
          return {
            ...source,
            rate,
            total_backlog: (source.response_backlog || 0) + (source.ir_backlog || 0),
            bottleneck: bottleneckText(source),
          };
        })
        .sort((a, b) => (b.total_backlog - a.total_backlog) || (b.rate.genui - a.rate.genui) || String(a.source_label).localeCompare(String(b.source_label)))
        .slice(0, 12);
      const rows = sourceRows.map(row => `
        <div class="eta-row">
          <div>
            <b>${escapeHtml(row.source_label)}</b><br>
            <span class="badge ${row.bottleneck === "clear" ? "ok" : "warn"}">${row.bottleneck}</span>
            <span class="small">${fmt(row.run_count)} runs</span>
          </div>
          <div>
            <div class="day-counts">
              <span>7d Q ${fmt(row.rate.queries)}</span>
              <span>R ${fmt(row.rate.responses)}</span>
              <span>IR ${fmt(row.rate.genui)}</span>
              <span>R/day ${rateText(row.rate.response_per_day)}</span>
              <span>IR/day ${rateText(row.rate.ir_per_day)}</span>
            </div>
            <div class="small">active dated buckets: ${fmt(row.rate.active_days)} | latest ${row.latest_run_updated_at ? ageText(row.latest_run_updated_at) : "unknown"}</div>
          </div>
          <div class="small">
            missing R ${fmt(row.response_backlog)} -> ${etaText(row.response_backlog, row.rate.response_per_day)}<br>
            missing IR ${fmt(row.ir_backlog)} -> ${etaText(row.ir_backlog, row.rate.ir_per_day)}
          </div>
        </div>`).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(stats.queries)}</b><br><span class="small">queries in last 7d</span></div>
          <div class="detail-box"><b>${fmt(stats.responses)}</b><br><span class="small">responses in last 7d</span></div>
          <div class="detail-box"><b>${fmt(stats.genui)}</b><br><span class="small">IR in last 7d</span></div>
          <div class="detail-box"><b>${rateText(stats.response_per_day)}</b><br><span class="small">responses/day</span></div>
          <div class="detail-box"><b>${rateText(stats.ir_per_day)}</b><br><span class="small">IR/day</span></div>
          <div class="detail-box"><b>${etaText(totals.response_backlog, stats.response_per_day)}</b><br><span class="small">response backlog ETA</span></div>
          <div class="detail-box"><b>${etaText(totals.ir_backlog, stats.ir_per_day)}</b><br><span class="small">IR backlog ETA</span></div>
          <div class="detail-box"><b>${latestDay}</b><br><span class="small">latest dated bucket</span></div>
        </div>
        <h2>Source Throughput</h2>
        <div class="warning-list">${rows || "<span class='small'>No source throughput rows.</span>"}</div>
        <div class="small">Rates use dated records from the last 7 calendar days under the active filters. ETA is backlog divided by recent response/IR generation rate; no recent rate means the pipeline appears stalled for that filtered slice.</div>
      `;
    }
    function artifactIssueLabels(r) {
      const issues = [];
      if ((r.missing_core_files || []).length) issues.push(`missing ${r.missing_core_files.join(", ")}`);
      if ((r.genui || 0) > 0 && !(r.screenshots || 0)) issues.push("no screenshots");
      if ((r.responses || 0) > 0 && !(r.assets || 0)) issues.push("no assets");
      return issues;
    }
    function renderStorageArtifacts(runs, sources) {
      const target = document.getElementById("storageArtifacts");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const totals = runTotals(runs);
      const artifactIssueRuns = runs.filter(r => artifactIssueLabels(r).length);
      const largestRuns = [...runs]
        .sort((a, b) => (b.total_bytes || 0) - (a.total_bytes || 0))
        .slice(0, 10);
      const sourceRows = [...sources]
        .sort((a, b) => (b.total_bytes || 0) - (a.total_bytes || 0))
        .slice(0, 8)
        .map(source => `
          <div class="freshness-row">
            <div><b>${escapeHtml(source.source_label)}</b><br><span class="small">${fmt(source.run_count)} runs | ${fmt(source.file_count)} files</span></div>
            <div><b>${bytesText(source.total_bytes)}</b><br><span class="small">storage</span></div>
            <div class="small">assets ${bytesText(source.asset_bytes)} | screenshots ${bytesText(source.screenshot_bytes)} | missing core runs ${fmt(source.missing_core_run_count)}</div>
          </div>`).join("");
      const largestRows = largestRuns.map(run => `
        <tr>
          <td><b>${escapeHtml(run.run_id)}</b><br><span class="small">${escapeHtml(run.source_label)}</span><br><button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(run))}')">Details</button></td>
          <td>${bytesText(run.total_bytes)}<br><span class="small">${fmt(run.file_count)} files</span></td>
          <td>${fmt(run.assets)}<br><span class="small">${bytesText(run.asset_bytes)}</span></td>
          <td>${fmt(run.screenshots)}<br><span class="small">${bytesText(run.screenshot_bytes)}</span></td>
          <td class="small">${artifactIssueLabels(run).map(escapeHtml).join("<br>") || "none"}</td>
        </tr>`).join("");
      const issueRows = artifactIssueRuns.slice(0, 10).map(run => `
        <div class="warning-row">
          <span><b>${escapeHtml(run.run_id)}</b><br><span class="small">${escapeHtml(run.source_label)} | ${escapeHtml(run.path)}</span></span>
          <span>${artifactIssueLabels(run).map(issue => `<span class="badge warn">${escapeHtml(issue)}</span>`).join(" ")}</span>
        </div>`).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${bytesText(totals.total_bytes)}</b><br><span class="small">filtered storage</span></div>
          <div class="detail-box"><b>${fmt(totals.file_count)}</b><br><span class="small">files</span></div>
          <div class="detail-box"><b>${bytesText(totals.asset_bytes)}</b><br><span class="small">asset bytes</span></div>
          <div class="detail-box"><b>${bytesText(totals.screenshot_bytes)}</b><br><span class="small">screenshot bytes</span></div>
          <div class="detail-box"><b>${fmt(totals.missing_core_run_count)}</b><br><span class="small">missing core runs</span></div>
          <div class="detail-box"><b>${fmt(artifactIssueRuns.length)}</b><br><span class="small">artifact issue runs</span></div>
        </div>
        <h2>Storage By Source</h2>
        <div class="warning-list">${sourceRows || "<span class='small'>No source storage rows.</span>"}</div>
        <h2>Largest Runs</h2>
        <div class="scroll">
          <table>
            <thead><tr><th>Run</th><th>Storage</th><th>Assets</th><th>Screenshots</th><th>Issues</th></tr></thead>
            <tbody>${largestRows}</tbody>
          </table>
        </div>
        <h2>Artifact Issues</h2>
        <div class="warning-list">${issueRows || "<span class='small'>No missing core, screenshot, or asset issues detected in current filters.</span>"}</div>
      `;
    }
    function aggregateQuality(runs) {
      const totals = {
        sampled: 0,
        json_parse_fail: 0,
        strict_schema_fail: 0,
        repair_needed: 0,
        repair_attempted: 0,
        gen_errors: 0,
        fallback_generated: 0,
        low_score: 0,
        markdown_leakage: 0,
        sparse_ir: 0,
        warnings: new Map(),
      };
      for (const r of runs) {
        const q = r.quality_summary || {};
        for (const key of ["sampled","json_parse_fail","strict_schema_fail","repair_needed","repair_attempted","gen_errors","fallback_generated","low_score","markdown_leakage","sparse_ir"]) {
          totals[key] += q[key] || 0;
        }
        for (const warning of q.warnings || []) {
          const message = warning.message || "warning";
          totals.warnings.set(message, (totals.warnings.get(message) || 0) + (warning.count || 0));
        }
      }
      totals.warning_rows = [...totals.warnings.entries()]
        .map(([message, count]) => ({message, count}))
        .sort((a,b) => (b.count - a.count) || a.message.localeCompare(b.message))
        .slice(0, 8);
      return totals;
    }
    function renderQualityAlerts(runs) {
      const q = aggregateQuality(runs);
      const issueCount = q.json_parse_fail + q.strict_schema_fail + q.gen_errors + q.fallback_generated + q.low_score + q.markdown_leakage + q.sparse_ir;
      const topWarnings = q.warning_rows.map(w => `
        <div class="warning-row">
          <span>${escapeHtml(w.message)}</span>
          <b>${fmt(w.count)}</b>
        </div>`).join("");
      document.getElementById("qualityAlerts").innerHTML = `
        <div class="detail-grid">
          ${qualityBox("sampled IR rows", q.sampled)}
          ${qualityBox("total issue signals", issueCount)}
          ${qualityBox("strict schema fail", q.strict_schema_fail)}
          ${qualityBox("generation errors", q.gen_errors)}
          ${qualityBox("repair attempted", q.repair_attempted)}
          ${qualityBox("fallback generated", q.fallback_generated)}
          ${qualityBox("low score rows", q.low_score)}
          ${qualityBox("markdown leakage", q.markdown_leakage)}
          ${qualityBox("sparse IR", q.sparse_ir)}
        </div>
        <h2>Top Warnings</h2>
        <div class="warning-list">${topWarnings || "<span class='small'>No sampled validation warnings.</span>"}</div>
        <div class="small">Diagnostics are aggregated from sampled run records and follow the current source/text/score/date/IR-version filters at run level.</div>
      `;
    }
    const overviewMetrics = [
      {key: "content_coverage", label: "Content coverage", kind: "pct"},
      {key: "intent_score", label: "Intent score", kind: "pct"},
      {key: "section_heading_coverage", label: "Heading coverage", kind: "pct"},
      {key: "table_cell_coverage", label: "Table coverage", kind: "pct"},
      {key: "action_coverage", label: "Action coverage", kind: "pct"},
      {key: "image_presence", label: "Image presence", kind: "pct"},
      {key: "icon_presence", label: "Icon presence", kind: "pct"},
      {key: "markdown_leakage_rate", label: "Markdown leakage", kind: "pct_low"},
      {key: "component_count", label: "Components", kind: "number"},
      {key: "max_tree_depth", label: "Max depth", kind: "number"},
    ];
    function metricDisplay(metric, value) {
      if (value == null) return "n/a";
      if (metric.kind === "pct" || metric.kind === "pct_low") return metricPct(value);
      return Number(value).toFixed(Number(value) >= 10 ? 1 : 2);
    }
    function aggregateMetricsOverview(runs) {
      const scoreBands = [
        {key: "excellent", label: ">= 80", runs: 0, genui: 0, className: "ok"},
        {key: "good", label: "70-79", runs: 0, genui: 0, className: "ok"},
        {key: "watch", label: "60-69", runs: 0, genui: 0, className: "warn"},
        {key: "poor", label: "< 60", runs: 0, genui: 0, className: "error"},
        {key: "unknown", label: "unknown", runs: 0, genui: 0, className: "unknown"},
      ];
      const bandForScore = score => {
        if (score == null || Number.isNaN(Number(score))) return scoreBands[4];
        const value = Number(score);
        if (value >= 80) return scoreBands[0];
        if (value >= 70) return scoreBands[1];
        if (value >= 60) return scoreBands[2];
        return scoreBands[3];
      };
      const metricTotals = new Map(overviewMetrics.map(metric => [metric.key, {sum: 0, weight: 0}]));
      let scoreSum = 0;
      let scoreWeight = 0;
      for (const run of runs) {
        const weight = Math.max(1, Number(run.genui || 0));
        const score = run.display_score ?? run.overall_score;
        const band = bandForScore(score);
        band.runs += 1;
        band.genui += Number(run.genui || 0);
        if (score != null && !Number.isNaN(Number(score))) {
          scoreSum += Number(score) * weight;
          scoreWeight += weight;
        }
        for (const metric of overviewMetrics) {
          const value = (run.metric_avgs || {})[metric.key];
          if (value == null || Number.isNaN(Number(value))) continue;
          const total = metricTotals.get(metric.key);
          total.sum += Number(value) * weight;
          total.weight += weight;
        }
      }
      return {
        weighted_score: scoreWeight ? scoreSum / scoreWeight : null,
        score_weight: scoreWeight,
        score_bands: scoreBands,
        metrics: overviewMetrics.map(metric => {
          const total = metricTotals.get(metric.key);
          return {...metric, value: total.weight ? total.sum / total.weight : null, weight: total.weight};
        }),
      };
    }
    function renderMetricsOverview(runs) {
      const target = document.getElementById("metricsOverview");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const overview = aggregateMetricsOverview(runs);
      const totals = runTotals(runs);
      const q = aggregateQuality(runs);
      const maxBand = Math.max(1, ...overview.score_bands.map(band => band.genui || band.runs));
      const scoreRows = overview.score_bands.map(band => {
        const width = Math.max(3, Math.round(((band.genui || band.runs) / maxBand) * 100));
        return `
          <div class="dist-row">
            <div><span class="badge ${band.className}">${band.label}</span></div>
            <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
            <div class="small">${fmt(band.runs)} runs<br>${fmt(band.genui)} IR</div>
          </div>`;
      }).join("");
      const metricRows = overview.metrics.map(metric => `
        <tr>
          <td><b>${metric.label}</b></td>
          <td>${metricDisplay(metric, metric.value)}</td>
          <td class="small">${fmt(metric.weight)} weighted IR</td>
        </tr>`).join("");
      const sampled = Math.max(1, q.sampled || 0);
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${scoreText(overview.weighted_score)}</b><br><span class="small">weighted avg score</span></div>
          <div class="detail-box"><b>${fmt(totals.genui)}</b><br><span class="small">filtered IR records</span></div>
          <div class="detail-box"><b>${fmt(totals.runs)}</b><br><span class="small">filtered runs</span></div>
          <div class="detail-box"><b>${metricPct(q.strict_schema_fail / sampled)}</b><br><span class="small">sampled strict fail rate</span></div>
          <div class="detail-box"><b>${metricPct(q.repair_attempted / sampled)}</b><br><span class="small">sampled repair rate</span></div>
          <div class="detail-box"><b>${metricPct(q.fallback_generated / sampled)}</b><br><span class="small">sampled fallback rate</span></div>
        </div>
        <div class="two-col-panels wide-panel">
          <div>
            <h2>Score Bands</h2>
            ${scoreRows}
          </div>
          <div>
            <h2>Core Metric Averages</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Metric</th><th>Average</th><th>Weight</th></tr></thead>
                <tbody>${metricRows}</tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="small">Averages are weighted by filtered IR count per run. Validation rates use sampled genui.jsonl rows already collected for quality alerts.</div>
      `;
    }
    const intentMetricKeys = [
      "content_coverage",
      "intent_score",
      "section_heading_coverage",
      "table_cell_coverage",
      "action_coverage",
      "image_presence",
      "icon_presence",
      "markdown_leakage_rate",
      "component_count",
    ];
    function aggregateIntentQuality(runs) {
      const byIntent = new Map();
      for (const run of runs) {
        for (const [intent, stats] of Object.entries(run.intent_quality || {})) {
          const count = Number(stats.count || stats.score_count || 0);
          if (!count) continue;
          if (!byIntent.has(intent)) {
            byIntent.set(intent, {intent, runs: 0, count: 0, scoreSum: 0, scoreWeight: 0, metrics: {}});
          }
          const target = byIntent.get(intent);
          target.runs += 1;
          target.count += count;
          if (stats.avg_score != null && !Number.isNaN(Number(stats.avg_score))) {
            target.scoreSum += Number(stats.avg_score) * count;
            target.scoreWeight += count;
          }
          for (const key of intentMetricKeys) {
            const value = (stats.metrics || {})[key];
            if (value == null || Number.isNaN(Number(value))) continue;
            const metric = target.metrics[key] || {sum: 0, weight: 0};
            metric.sum += Number(value) * count;
            metric.weight += count;
            target.metrics[key] = metric;
          }
        }
      }
      return [...byIntent.values()].map(row => {
        const metrics = {};
        for (const [key, value] of Object.entries(row.metrics)) {
          if (value.weight) metrics[key] = value.sum / value.weight;
        }
        return {
          ...row,
          avg_score: row.scoreWeight ? row.scoreSum / row.scoreWeight : null,
          metrics,
        };
      });
    }
    function renderIntentQuality(runs) {
      const target = document.getElementById("intentQuality");
      const rows = aggregateIntentQuality(runs);
      if (!rows.length) {
        target.innerHTML = "<span class='small'>No sampled intent quality data found for current filters.</span>";
        return;
      }
      const sortedByScore = [...rows].sort((a, b) => {
        const scoreA = a.avg_score == null ? 999 : a.avg_score;
        const scoreB = b.avg_score == null ? 999 : b.avg_score;
        return scoreA - scoreB || b.count - a.count || a.intent.localeCompare(b.intent);
      });
      const sortedByVolume = [...rows].sort((a, b) => b.count - a.count || a.intent.localeCompare(b.intent));
      const weakRows = sortedByScore.slice(0, 12).map(row => {
        const m = row.metrics || {};
        return `
          <tr>
            <td><b>${escapeHtml(row.intent)}</b><br><span class="small">${fmt(row.runs)} runs</span></td>
            <td>${fmt(row.count)}</td>
            <td><span class="score ${scoreClass(row.avg_score)}">${scoreText(row.avg_score)}</span></td>
            <td class="small">
              coverage ${metricPct(m.content_coverage)} | intent ${metricPct(m.intent_score)}<br>
              headings ${metricPct(m.section_heading_coverage)} | table ${metricPct(m.table_cell_coverage)}<br>
              actions ${metricPct(m.action_coverage)} | images ${metricPct(m.image_presence)} | icons ${metricPct(m.icon_presence)}
            </td>
          </tr>`;
      }).join("");
      const maxVolume = Math.max(1, ...sortedByVolume.map(row => row.count));
      const volumeRows = sortedByVolume.slice(0, 12).map(row => {
        const width = Math.max(3, Math.round((row.count / maxVolume) * 100));
        return `
          <div class="dist-row">
            <div class="dist-label" title="${escapeHtml(row.intent)}">${escapeHtml(row.intent)}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
            <div class="small">${fmt(row.count)} IR<br><span class="score ${scoreClass(row.avg_score)}">${scoreText(row.avg_score)}</span></div>
          </div>`;
      }).join("");
      const best = sortedByScore.filter(row => row.avg_score != null).slice(-1)[0];
      const weakest = sortedByScore.find(row => row.avg_score != null);
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(rows.length)}</b><br><span class="small">sampled intents</span></div>
          <div class="detail-box"><b>${fmt(rows.reduce((a, r) => a + r.count, 0))}</b><br><span class="small">sampled IR rows</span></div>
          <div class="detail-box"><b>${best ? escapeHtml(best.intent) : "n/a"}</b><br><span class="small">best intent ${best ? scoreText(best.avg_score) : ""}</span></div>
          <div class="detail-box"><b>${weakest ? escapeHtml(weakest.intent) : "n/a"}</b><br><span class="small">weakest intent ${weakest ? scoreText(weakest.avg_score) : ""}</span></div>
        </div>
        <div class="two-col-panels wide-panel">
          <div>
            <h2>Largest Intent Buckets</h2>
            ${volumeRows}
          </div>
          <div>
            <h2>Lowest Scoring Intents</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Intent</th><th>Sampled IR</th><th>Score</th><th>Signals</th></tr></thead>
                <tbody>${weakRows}</tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="small">Intent quality is computed from sampled genui.jsonl rows per run and then aggregated across the current filtered run set. It is best used to identify weak domains, not as an exact full-dataset per-intent metric.</div>
      `;
    }
    function aggregateIssueSamples(runs) {
      const rows = [];
      for (const run of runs) {
        const samples = ((run.quality_summary || {}).issue_samples || []);
        for (const sample of samples) {
          rows.push({
            ...sample,
            source_id: run.source_id,
            source_label: run.source_label,
            run_id: run.run_id,
            run_key: runKey(run),
          });
        }
      }
      return rows
        .sort((a, b) => {
          const scoreA = a.score == null ? 999 : Number(a.score);
          const scoreB = b.score == null ? 999 : Number(b.score);
          return scoreA - scoreB || ((b.issues || []).length - (a.issues || []).length) || String(a.run_id).localeCompare(String(b.run_id));
        })
        .slice(0, 40);
    }
    function renderWorstSamples(runs) {
      const samples = aggregateIssueSamples(runs);
      if (!samples.length) {
        document.getElementById("worstSamples").innerHTML = "<tr><td colspan='6'><span class='small'>No sampled low-score or failed records found for current filters.</span></td></tr>";
        return;
      }
      document.getElementById("worstSamples").innerHTML = samples.map(sample => `
        <tr>
          <td>
            <b>${escapeHtml(sample.ui_id || `row ${sample.row}`)}</b><br>
            <span class="small">${escapeHtml(sample.query_id || "")} ${escapeHtml(sample.response_id || "")}</span><br>
            <span class="small">${escapeHtml(sample.intent || "")}</span>
          </td>
          <td>
            <b>${escapeHtml(sample.run_id)}</b><br>
            <span class="small">${escapeHtml(sample.source_label)}</span><br>
            <button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(sample.run_key)}')">Run details</button>
          </td>
          <td class="nowrap">
            <span class="score ${scoreClass(sample.score)}">${scoreText(sample.score)}</span><br>
            <span class="small">coverage ${metricPct(sample.content_coverage)}</span><br>
            <span class="small">components ${fmt(sample.component_count)}</span>
          </td>
          <td>${(sample.issues || []).map(issue => `<span class="badge error">${escapeHtml(issue)}</span>`).join(" ") || "<span class='small'>low metric</span>"}</td>
          <td>
            <span class="small">${escapeHtml(sample.model || "unknown")}</span><br>
            <span class="small">${escapeHtml(sample.prompt_version || "unknown")}</span>
          </td>
          <td class="small">${(sample.warnings || []).map(escapeHtml).join("<br>") || "n/a"}</td>
        </tr>`).join("");
    }
    function addWeightedMetric(bucket, r, key, weight) {
      const value = (r.metric_avgs || {})[key];
      if (value == null || Number.isNaN(Number(value))) return;
      const target = bucket.metrics[key] || {sum: 0, weight: 0};
      target.sum += Number(value) * weight;
      target.weight += weight;
      bucket.metrics[key] = target;
    }
    function aggregateModelComparisonsFromRuns(runs) {
      const buckets = new Map();
      for (const r of runs) {
        const responseModel = dominantModel(r.response_models);
        const irModel = dominantModel(r.ir_models);
        const key = `${responseModel} -> ${irModel}`;
        if (!buckets.has(key)) {
          buckets.set(key, {responseModel, irModel, runs: 0, genui: 0, scoreSum: 0, scoreWeight: 0, metrics: {}});
        }
        const bucket = buckets.get(key);
        const weight = Math.max(1, r.genui || 0);
        bucket.runs += 1;
        bucket.genui += r.genui || 0;
        const score = r.display_score ?? r.overall_score;
        if (score != null) {
          bucket.scoreSum += Number(score) * weight;
          bucket.scoreWeight += weight;
        }
        for (const key of ["content_coverage", "intent_score", "section_heading_coverage", "table_cell_coverage", "action_coverage", "image_presence", "icon_presence", "markdown_leakage_rate"]) {
          addWeightedMetric(bucket, r, key, weight);
        }
      }
      return [...buckets.values()]
        .map(bucket => {
          const metrics = {};
          for (const [key, value] of Object.entries(bucket.metrics)) {
            if (value.weight) metrics[key] = value.sum / value.weight;
          }
          return {...bucket, metrics, avg_score: bucket.scoreWeight ? bucket.scoreSum / bucket.scoreWeight : null};
        })
        .sort((a,b) => (b.genui - a.genui) || String(a.responseModel).localeCompare(String(b.responseModel)));
    }
    function renderModelComparison(runs) {
      const rows = aggregateModelComparisonsFromRuns(runs);
      if (!rows.length) {
        document.getElementById("modelComparison").innerHTML = "<tr><td colspan='4'><span class='small'>No model data found.</span></td></tr>";
        return;
      }
      document.getElementById("modelComparison").innerHTML = rows.map(row => {
        const m = row.metrics || {};
        return `
          <tr>
            <td>
              <b>${row.responseModel}</b><br>
              <span class="small">to</span><br>
              <b>${row.irModel}</b>
            </td>
            <td>${fmt(row.runs)} runs<br><span class="small">${fmt(row.genui)} IR</span></td>
            <td><span class="score ${scoreClass(row.avg_score)}">${scoreText(row.avg_score)}</span></td>
            <td class="small">
              coverage ${metricPct(m.content_coverage)} | intent ${metricPct(m.intent_score)}<br>
              headings ${metricPct(m.section_heading_coverage)} | table ${metricPct(m.table_cell_coverage)}<br>
              actions ${metricPct(m.action_coverage)} | images ${metricPct(m.image_presence)} | markdown leak ${metricPct(m.markdown_leakage_rate)}
            </td>
          </tr>`;
      }).join("");
    }
    function addObjectCounts(target, obj) {
      for (const [key, value] of Object.entries(obj || {})) {
        target.set(key, (target.get(key) || 0) + Number(value || 0));
      }
    }
    function topCounts(map, limit = 8) {
      return [...map.entries()].sort((a,b) => (b[1] - a[1]) || a[0].localeCompare(b[0])).slice(0, limit);
    }
    function renderDistributionBlock(title, entries) {
      if (!entries.length) return `<div><h2>${title}</h2><span class="small">No data</span></div>`;
      const max = Math.max(1, ...entries.map(([, count]) => count));
      return `
        <div>
          <h2>${title}</h2>
          ${entries.map(([label, count]) => {
            const width = Math.max(3, Math.round((count / max) * 100));
            return `
              <div class="dist-row">
                <div class="dist-label" title="${escapeHtml(label)}">${escapeHtml(label)}</div>
                <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
                <b>${fmt(count)}</b>
              </div>`;
          }).join("")}
        </div>`;
    }
    function renderDistribution(runs) {
      const intents = new Map();
      const responseModels = new Map();
      const irModels = new Map();
      const irVersions = new Map();
      const sources = new Map();
      for (const run of runs) {
        addObjectCounts(intents, run.intents || {});
        addObjectCounts(responseModels, run.response_models || {});
        addObjectCounts(irModels, run.ir_models || {});
        addObjectCounts(irVersions, run.ir_versions || {});
        sources.set(run.source_label, (sources.get(run.source_label) || 0) + (run.genui || run.responses || run.queries || 1));
      }
      document.getElementById("distribution").innerHTML = [
        renderDistributionBlock("Intent Mix", topCounts(intents)),
        renderDistributionBlock("Stage 2 Models", topCounts(responseModels)),
        renderDistributionBlock("Stage 3 Models", topCounts(irModels)),
        renderDistributionBlock("IR Versions", topCounts(irVersions)),
        renderDistributionBlock("Source Volume", topCounts(sources)),
      ].join("");
    }
    function aggregateDaysFromRuns(runs) {
      const f = filters();
      const byDay = new Map();
      for (const run of runs) {
        const versionStats = f.irVersion ? (run.ir_version_stats || {})[f.irVersion] : null;
        const days = versionStats ? (versionStats.days || []) : (run.days || []);
        for (const day of days) {
          if (!dayInRange(day.day, f)) continue;
          if (!byDay.has(day.day)) {
            byDay.set(day.day, {day: day.day, queries: 0, responses: 0, genui: 0, score_sum: 0, score_count: 0, avg_score: null});
          }
          const target = byDay.get(day.day);
          target.queries += day.queries || 0;
          target.responses += day.responses || 0;
          target.genui += day.genui || 0;
          target.score_sum += day.score_sum || 0;
          target.score_count += day.score_count || 0;
        }
      }
      return [...byDay.values()]
        .map(day => ({...day, avg_score: day.score_count ? day.score_sum / day.score_count : null}))
        .sort((a,b) => b.day.localeCompare(a.day));
    }
    function svgPoint(x, y) {
      return `${Number(x).toFixed(1)},${Number(y).toFixed(1)}`;
    }
    function renderTrend(runs) {
      const days = aggregateDaysFromRuns(runs).slice().sort((a,b) => a.day.localeCompare(b.day));
      const target = document.getElementById("trend");
      if (days.length < 2) {
        target.innerHTML = "<span class='small'>Need at least two dated buckets to show a trend.</span>";
        return;
      }
      const visible = days.slice(-45);
      const width = 960;
      const height = 220;
      const pad = {left: 42, right: 22, top: 18, bottom: 38};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const maxTotal = Math.max(1, ...visible.map(totalDayCount));
      const scores = visible.map(d => d.avg_score).filter(v => v != null).map(Number);
      const minScore = Math.max(0, Math.min(...scores, 50));
      const maxScore = Math.min(100, Math.max(...scores, 100));
      const xFor = index => pad.left + (visible.length === 1 ? plotW / 2 : (index / (visible.length - 1)) * plotW);
      const yScore = score => pad.top + (1 - ((Number(score) - minScore) / Math.max(1, maxScore - minScore))) * plotH;
      const yCount = count => pad.top + (1 - (count / maxTotal)) * plotH;
      const bars = visible.map((day, index) => {
        const x = xFor(index);
        const y = yCount(totalDayCount(day));
        const barW = Math.max(4, Math.min(14, plotW / visible.length * .58));
        return `<rect x="${x - barW / 2}" y="${y}" width="${barW}" height="${pad.top + plotH - y}" rx="3" fill="rgba(15,118,110,.20)" />`;
      }).join("");
      const scorePoints = visible
        .map((day, index) => day.avg_score == null ? null : svgPoint(xFor(index), yScore(day.avg_score)))
        .filter(Boolean);
      const scoreCircles = visible.map((day, index) => day.avg_score == null ? "" : `
        <circle cx="${xFor(index)}" cy="${yScore(day.avg_score)}" r="4" fill="#c2410c">
          <title>${day.day}: score ${scoreText(day.avg_score)}, Q ${fmt(day.queries)}, R ${fmt(day.responses)}, IR ${fmt(day.genui)}</title>
        </circle>`).join("");
      const labels = visible.filter((_, index) => index === 0 || index === visible.length - 1 || index % Math.ceil(visible.length / 6) === 0)
        .map((day, index, arr) => {
          const actualIndex = visible.indexOf(day);
          return `<text x="${xFor(actualIndex)}" y="${height - 10}" text-anchor="middle" font-size="11" fill="#667085">${day.day.slice(5)}</text>`;
        }).join("");
      target.innerHTML = `
        <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Filtered score and volume trend">
          <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + plotH}" stroke="rgba(36,48,64,.18)" />
          <line x1="${pad.left}" y1="${pad.top + plotH}" x2="${width - pad.right}" y2="${pad.top + plotH}" stroke="rgba(36,48,64,.18)" />
          <text x="4" y="${pad.top + 6}" font-size="11" fill="#667085">score</text>
          <text x="4" y="${pad.top + plotH}" font-size="11" fill="#667085">${Math.round(minScore)}</text>
          ${bars}
          ${scorePoints.length ? `<polyline points="${scorePoints.join(" ")}" fill="none" stroke="#c2410c" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />` : ""}
          ${scoreCircles}
          ${labels}
        </svg>
        <div class="small">Bars show total Q+R+IR records per day. Orange line shows average IR score where metric data exists. Trend follows all active filters.</div>
      `;
    }
    function renderDays(runs) {
      const sourceId = document.getElementById("sourceFilter").value;
      const source = (current.sources || []).find(s => s.source_id === sourceId);
      const days = aggregateDaysFromRuns(runs);
      const label = source ? source.source_label : "Filtered sources";
      document.getElementById("daysTitle").textContent = `Day Wise Data - ${label}`;
      if (!days.length) {
        document.getElementById("days").innerHTML = "<span class='small'>No dated records found.</span>";
        return;
      }
      const maxTotal = Math.max(1, ...days.map(totalDayCount));
      const rows = days.slice(0, 60);
      const bars = rows.slice(0, 21).map(day => {
        const total = totalDayCount(day);
        const width = Math.max(4, Math.round((total / maxTotal) * 100));
        return `
          <div class="day-row">
            <div><b>${day.day}</b><br><span class="small">${fmt(total)} total</span></div>
            <div>
              <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
              <div class="day-counts">
                <span>Q ${fmt(day.queries)}</span>
                <span>R ${fmt(day.responses)}</span>
                <span>IR ${fmt(day.genui)}</span>
              </div>
            </div>
            <div><span class="score ${scoreClass(day.avg_score)}">${scoreText(day.avg_score)}</span><br><span class="small">avg IR score</span></div>
          </div>`;
      }).join("");
      const table = `
        <table class="day-table">
          <thead><tr><th>Day</th><th>Queries</th><th>Responses</th><th>IR</th><th>Avg IR Score</th></tr></thead>
          <tbody>
            ${rows.map(day => `<tr>
              <td>${day.day}</td>
              <td>${fmt(day.queries)}</td>
              <td>${fmt(day.responses)}</td>
              <td>${fmt(day.genui)}</td>
              <td><span class="score ${scoreClass(day.avg_score)}">${scoreText(day.avg_score)}</span></td>
            </tr>`).join("")}
          </tbody>
        </table>`;
      document.getElementById("days").innerHTML = `${bars}${table}`;
    }
    function render() {
      if (!current) return;
      renderSourceFilter(current.sources || []);
      renderIrVersionFilter(current.ir_versions || {});
      const runs = filteredRuns();
      const page = paginatedRuns(runs);
      renderStats(runTotals(runs));
      renderSyncConfig(current.config || {});
      renderLastSyncResults(current.last_sync);
      const sourceStats = filteredSourceStats(current.sources || [], runs);
      renderFreshness(runs, sourceStats, current.last_sync);
      renderSources(sourceStats);
      renderRuns(page.rows, page);
      renderRunDetails(runs);
      renderBacklog(sourceStats);
      renderQualityAlerts(runs);
      renderMetricsOverview(runs);
      renderIntentQuality(runs);
      renderThroughputEta(runs, sourceStats);
      renderStorageArtifacts(runs, sourceStats);
      renderWorstSamples(runs);
      renderModelComparison(runs);
      renderDistribution(runs);
      renderTrend(runs);
      renderDays(runs);
    }
    document.getElementById("syncBtn").onclick = () => syncSources().catch(e => setStatus(`Sync failed: ${e.message}`));
    document.getElementById("refreshBtn").onclick = () => loadSummary().catch(e => setStatus(`Refresh failed: ${e.message}`));
    document.getElementById("exportCsvBtn").onclick = () => exportFiltered("csv");
    document.getElementById("exportJsonBtn").onclick = () => exportFiltered("json");
    document.getElementById("filter").oninput = resetPageAndRender;
    document.getElementById("sourceFilter").onchange = resetPageAndRender;
    document.getElementById("irVersionFilter").onchange = resetPageAndRender;
    document.getElementById("scoreFilter").onchange = resetPageAndRender;
    document.getElementById("issueFilter").onchange = resetPageAndRender;
    document.getElementById("sortBy").onchange = resetPageAndRender;
    document.getElementById("dateFrom").onchange = resetPageAndRender;
    document.getElementById("dateTo").onchange = resetPageAndRender;
    document.getElementById("pageSize").onchange = resetPageAndRender;
    document.getElementById("prevPageBtn").onclick = () => { runPage = Math.max(1, runPage - 1); render(); };
    document.getElementById("nextPageBtn").onclick = () => { runPage += 1; render(); };
    document.getElementById("autoRefreshInterval").onchange = startAutoRefresh;
    loadSyncStatus().catch(() => {});
    loadSummary().then(startAutoRefresh).catch(e => setStatus(`Load failed: ${e.message}`));
  </script>
</body>
</html>
"""


class DashboardServer:
    def __init__(self, config_path: Path, mirror_dir: Path):
        self.config_path = config_path
        self.mirror_dir = mirror_dir
        self.lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.sync_status: dict[str, Any] = self.new_sync_status()

    def new_sync_status(self) -> dict[str, Any]:
        return {
            "running": False,
            "phase": "idle",
            "started_at": None,
            "updated_at": utc_now(),
            "source_id": "",
            "source_label": "",
            "source_type": "",
            "source_index": 0,
            "total_sources": 0,
            "current_file": "",
            "listed": 0,
            "total": 0,
            "processed": 0,
            "copied": 0,
            "skipped": 0,
            "error_count": 0,
            "message": "Idle",
            "messages": [],
        }

    def update_sync_status(self, update: dict[str, Any]) -> None:
        with self.status_lock:
            if update.get("phase") == "starting":
                self.sync_status = self.new_sync_status()
                self.sync_status["started_at"] = utc_now()
            status = dict(self.sync_status)
            status.update(update)
            status["updated_at"] = utc_now()
            message = str(update.get("message") or "").strip()
            messages = list(status.get("messages") or [])
            if message and (not messages or messages[-1] != message):
                messages.append(message)
            status["messages"] = messages[-12:]
            self.sync_status = status

    def status(self) -> dict[str, Any]:
        with self.status_lock:
            return json.loads(json.dumps(self.sync_status))

    def config(self) -> dict[str, Any]:
        config = load_config(self.config_path)
        mirror_raw = config.get("mirror_dir")
        if mirror_raw:
            self.mirror_dir = resolve_dataset_path(str(mirror_raw), ROOT)
        return config

    def summary(self) -> dict[str, Any]:
        return scan_all(self.config(), self.mirror_dir)

    def sync(self) -> dict[str, Any]:
        with self.lock:
            self.update_sync_status({"running": True, "phase": "starting", "message": "Starting sync"})
            try:
                return run_sync(self.config(), self.mirror_dir, progress=self.update_sync_status)
            except Exception as exc:
                self.update_sync_status({"running": False, "phase": "failed", "message": f"Sync failed: {exc}"})
                raise

    def test_source(self, source_id: str) -> dict[str, Any]:
        return test_source_connection(self.config(), self.mirror_dir, source_id)


def make_handler(server_state: DashboardServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def send_json(self, payload: Any, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"Invalid JSON body: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/summary":
                try:
                    self.send_json(server_state.summary())
                except Exception as exc:
                    self.send_json({"error": str(exc)}, status=500)
                return
            if parsed.path == "/api/sync/status":
                self.send_json(server_state.status())
                return
            if parsed.path == "/api/config":
                config = server_state.config()
                safe = json.loads(json.dumps(config))
                for source in safe.get("sources", []) or []:
                    for key in ("password", "token", "secret"):
                        if key in source:
                            source[key] = "***"
                self.send_json(safe)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/sync":
                try:
                    self.send_json(server_state.sync())
                except Exception as exc:
                    self.send_json({"error": str(exc)}, status=500)
                return
            if parsed.path == "/api/source/test":
                try:
                    payload = self.read_json_body()
                    source_id = str(payload.get("source_id") or "").strip()
                    if not source_id:
                        raise ValueError("source_id is required")
                    result = server_state.test_source(source_id)
                    self.send_json(result, status=200 if result.get("status") != "error" else 502)
                except Exception as exc:
                    self.send_json({"error": str(exc)}, status=400)
                return
            self.send_error(404)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2UI generated dataset sync dashboard")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Source config JSON path")
    parser.add_argument("--mirror-dir", default=str(DEFAULT_MIRROR_DIR), help="Local mirror directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sync-on-start", action="store_true", help="Sync enabled sources before serving")
    parser.add_argument("--sync-once", action="store_true", help="Run one sync and exit")
    parser.add_argument("--summary-once", action="store_true", help="Print summary JSON and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_dataset_path(args.config, ROOT)
    mirror_dir = resolve_dataset_path(args.mirror_dir, ROOT)
    state = DashboardServer(config_path=config_path, mirror_dir=mirror_dir)

    if args.sync_once:
        print(json.dumps(state.sync(), indent=2, ensure_ascii=False))
        return 0
    if args.summary_once:
        print(json.dumps(state.summary(), indent=2, ensure_ascii=False))
        return 0
    if args.sync_on_start:
        print(json.dumps(state.sync(), indent=2, ensure_ascii=False))

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Dataset dashboard: http://{args.host}:{args.port}")
    print(f"Config: {config_path}")
    print(f"Mirror: {mirror_dir}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
