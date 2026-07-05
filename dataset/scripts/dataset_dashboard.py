#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import getpass
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DEFAULT_MAX_PARALLEL_SOURCES = 10
PASSWORD_CACHE: dict[str, str] = {}
ProgressCallback = Callable[[dict[str, Any]], None]


class SyncStopped(RuntimeError):
    """Raised when the user requests dashboard sync cancellation."""


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


def scan_jsonl_id_fields(path: Path, fields: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.exists(),
        "rows": 0,
        "parse_errors": 0,
        "parse_error_lines": [],
        "missing": {field: 0 for field in fields},
        "ids": {field: set() for field in fields},
        "duplicate_counts": {field: 0 for field in fields},
        "duplicate_examples": {field: [] for field in fields},
    }
    if not path.exists():
        return result
    seen = {field: set() for field in fields}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            result["rows"] += 1
            try:
                row = json.loads(line)
            except Exception:
                result["parse_errors"] += 1
                if len(result["parse_error_lines"]) < 5:
                    result["parse_error_lines"].append(line_number)
                continue
            if not isinstance(row, dict):
                result["parse_errors"] += 1
                if len(result["parse_error_lines"]) < 5:
                    result["parse_error_lines"].append(line_number)
                continue
            for field in fields:
                value = str(row.get(field) or "").strip()
                if not value:
                    result["missing"][field] += 1
                    continue
                result["ids"][field].add(value)
                if value in seen[field]:
                    result["duplicate_counts"][field] += 1
                    if len(result["duplicate_examples"][field]) < 5 and value not in result["duplicate_examples"][field]:
                        result["duplicate_examples"][field].append(value)
                seen[field].add(value)
    return result


def collect_data_integrity_summary(run_dir: Path, queries: int, responses: int, genui: int) -> dict[str, Any]:
    query_scan = scan_jsonl_id_fields(run_dir / "queries.jsonl", ["query_id"])
    response_scan = scan_jsonl_id_fields(run_dir / "responses.jsonl", ["query_id", "response_id"])
    genui_scan = scan_jsonl_id_fields(run_dir / "genui.jsonl", ["query_id", "response_id", "ui_id"])

    query_ids = query_scan["ids"].get("query_id", set())
    response_query_ids = response_scan["ids"].get("query_id", set())
    response_ids = response_scan["ids"].get("response_id", set())
    genui_query_ids = genui_scan["ids"].get("query_id", set())
    genui_response_ids = genui_scan["ids"].get("response_id", set())

    orphan_links = {
        "responses_without_query": sorted(response_query_ids - query_ids)[:8] if query_ids else [],
        "genui_without_query": sorted(genui_query_ids - query_ids)[:8] if query_ids else [],
        "genui_without_response": sorted(genui_response_ids - response_ids)[:8] if response_ids else [],
    }
    orphan_counts = {
        "responses_without_query": len(response_query_ids - query_ids) if query_ids else 0,
        "genui_without_query": len(genui_query_ids - query_ids) if query_ids else 0,
        "genui_without_response": len(genui_response_ids - response_ids) if response_ids else 0,
    }

    count_warnings = []
    if responses > queries and queries:
        count_warnings.append({"type": "responses_gt_queries", "message": "responses count is greater than queries count"})
    if genui > responses and responses:
        count_warnings.append({"type": "ir_gt_responses", "message": "IR count is greater than responses count"})
    if responses and not queries:
        count_warnings.append({"type": "responses_without_queries_file", "message": "responses exist but queries are missing or empty"})
    if genui and not responses:
        count_warnings.append({"type": "ir_without_responses_file", "message": "IR exists but responses are missing or empty"})

    files = {
        "queries": {
            "exists": query_scan["exists"],
            "rows": query_scan["rows"],
            "parse_errors": query_scan["parse_errors"],
            "parse_error_lines": query_scan["parse_error_lines"],
        },
        "responses": {
            "exists": response_scan["exists"],
            "rows": response_scan["rows"],
            "parse_errors": response_scan["parse_errors"],
            "parse_error_lines": response_scan["parse_error_lines"],
        },
        "genui": {
            "exists": genui_scan["exists"],
            "rows": genui_scan["rows"],
            "parse_errors": genui_scan["parse_errors"],
            "parse_error_lines": genui_scan["parse_error_lines"],
        },
    }
    missing_ids: dict[str, int] = {}
    duplicate_ids: dict[str, dict[str, Any]] = {}
    for label, scan in (("queries", query_scan), ("responses", response_scan), ("genui", genui_scan)):
        for field, count in (scan.get("missing") or {}).items():
            if count:
                missing_ids[f"{label}.{field}"] = int(count)
        for field, count in (scan.get("duplicate_counts") or {}).items():
            if count:
                duplicate_ids[f"{label}.{field}"] = {
                    "count": int(count),
                    "examples": scan.get("duplicate_examples", {}).get(field, []),
                }

    parse_total = sum(int(entry["parse_errors"]) for entry in files.values())
    missing_total = sum(missing_ids.values())
    duplicate_total = sum(int(entry["count"]) for entry in duplicate_ids.values())
    orphan_total = sum(orphan_counts.values())
    warning_total = len(count_warnings)
    return {
        "files": files,
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "orphan_counts": orphan_counts,
        "orphan_examples": orphan_links,
        "count_warnings": count_warnings,
        "parse_error_count": parse_total,
        "missing_id_count": missing_total,
        "duplicate_id_count": duplicate_total,
        "orphan_link_count": orphan_total,
        "count_warning_count": warning_total,
        "total_issues": parse_total + missing_total + duplicate_total + orphan_total + warning_total,
    }


def safe_source_id(source_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in source_id.strip())
    return cleaned or "source"


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def safe_relative_path(raw: Any, *, context: str = "path") -> str:
    rel = str(raw or "").replace("\\", "/").strip()
    if not rel or rel == ".":
        raise RuntimeError(f"empty relative {context}")
    parts = PurePosixPath(rel).parts
    if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel) or ".." in parts:
        raise RuntimeError(f"unsafe relative {context}: {raw}")
    return rel


def validate_sync_entries(entries: list[FileEntry], *, source_name: str) -> list[FileEntry]:
    validated: list[FileEntry] = []
    seen: set[str] = set()
    for entry in entries:
        rel = safe_relative_path(entry.rel, context=f"path from {source_name}")
        if rel in seen:
            raise RuntimeError(f"duplicate relative path from {source_name}: {rel}")
        seen.add(rel)
        if rel == entry.rel:
            validated.append(entry)
        else:
            validated.append(FileEntry(rel=rel, size=entry.size, mtime=entry.mtime, source_path=entry.source_path))
    return validated


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
        rel = safe_relative_path(obj["rel"], context=f"SSH source {source.get('id')} output")
        entries.append(
            FileEntry(
                rel=rel,
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
        rel = safe_relative_path(obj["rel"], context=f"command source {source.get('id')} output")
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
    # OpenSSH scp uses SFTP by default on modern systems. In SFTP mode, shell
    # quoting becomes part of the remote filename and causes false "not found"
    # errors for paths that do exist. Keep quoting opt-in only for legacy scp.
    remote_path_arg = shell_quote(remote_path) if config_bool(source.get("scp_quote_remote_path"), False) else remote_path
    remote_spec = f"{target}:{remote_path_arg}"
    cmd = scp_base_command(source) + [remote_spec, str(dest)]
    result = run_command(cmd, timeout=int(source.get("copy_timeout_sec", 300)))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"scp failed for {entry.rel}")


def run_remote_ssh_command(
    source: dict[str, Any],
    remote_command: str,
    client: Any | None = None,
    timeout: int | None = None,
) -> str:
    command_timeout = timeout if timeout is not None else int(source.get("copy_timeout_sec", 300))
    if client is not None:
        stdin, stdout, stderr = client.exec_command(remote_command, timeout=command_timeout)
        del stdin
        output = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            raise RuntimeError(error.strip() or output.strip() or "ssh command failed")
        return output
    cmd = ssh_base_command(source) + [remote_command]
    result = run_command(cmd, timeout=command_timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ssh command failed")
    return result.stdout


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


def file_signature(entry: FileEntry) -> tuple[dict[str, Any], dict[str, Any]]:
    legacy_signature = {"size": entry.size, "mtime": round(entry.mtime, 6)}
    signature = {**legacy_signature, "source_path": entry.source_path or ""}
    return signature, legacy_signature


def is_entry_unchanged(previous: dict[str, Any], dest_root: Path, entry: FileEntry) -> bool:
    signature, legacy_signature = file_signature(entry)
    return previous.get(entry.rel) in (signature, legacy_signature) and (dest_root / entry.rel).exists()


def source_transfer_mode(source: dict[str, Any]) -> str:
    # Archive mode was removed: remote temp space is too unreliable for large runs.
    # Keep accepting older config keys, but all syncs now use direct changed-file copies.
    return "direct"


def source_priority_copy_globs(source: dict[str, Any]) -> list[str]:
    raw = source.get("priority_copy_globs", source.get("direct_first_globs"))
    if raw is None:
        return ["*.jsonl", "*.json"]
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def split_priority_entries(source: dict[str, Any], entries: list[FileEntry]) -> tuple[list[FileEntry], list[FileEntry]]:
    patterns = source_priority_copy_globs(source)
    if not patterns:
        return [], entries
    priority: list[FileEntry] = []
    remaining: list[FileEntry] = []
    for entry in entries:
        if match_filter_path(entry.rel, patterns):
            priority.append(entry)
        else:
            remaining.append(entry)
    return priority, remaining


def emit_progress(progress: ProgressCallback | None, **payload: Any) -> None:
    if progress is not None:
        progress(payload)


def sync_stop_requested(cancel_event: threading.Event | None) -> bool:
    return bool(cancel_event and cancel_event.is_set())


def raise_if_sync_stopped(cancel_event: threading.Event | None) -> None:
    if sync_stop_requested(cancel_event):
        raise SyncStopped("sync stopped by user")


def sync_source(
    source: dict[str, Any],
    mirror_dir: Path,
    include_globs: list[str],
    exclude_globs: list[str],
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
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
    entries = validate_sync_entries(entries, source_name=source_label(source))
    cancelled = sync_stop_requested(cancel_event)

    transfer_mode = source_transfer_mode(source)
    effective_transfer_mode = "direct"
    emit_progress(
        progress,
        phase="checking_changed_files",
        source_id=source_id,
        source_label=source_label(source),
        source_type=source_type,
        transfer_mode=transfer_mode,
        listed=len(entries),
        total=len(entries),
        processed=0,
        copied=0,
        skipped=0,
        error_count=0,
        message=f"Checking changed files from {source_label(source)}",
    )
    copied = 0
    skipped = 0
    errors: list[str] = []
    warnings: list[str] = []
    fallback_reason = ""
    fallback_at = ""
    current_files: dict[str, Any] = {}
    listed_signatures: dict[str, Any] = {}
    changed_entries: list[FileEntry] = []
    for entry in entries:
        if sync_stop_requested(cancel_event):
            cancelled = True
            break
        signature, _ = file_signature(entry)
        listed_signatures[entry.rel] = signature
        if is_entry_unchanged(previous, dest_root, entry):
            skipped += 1
            current_files[entry.rel] = signature
        else:
            changed_entries.append(entry)
    emit_progress(
        progress,
        phase="checking_changed_files",
        source_id=source_id,
        source_label=source_label(source),
        source_type=source_type,
        transfer_mode=transfer_mode,
        listed=len(entries),
        total=len(entries),
        processed=skipped,
        changed=len(changed_entries),
        copied=0,
        skipped=skipped,
        error_count=0,
        message=f"Found {len(changed_entries)} changed file(s), {skipped} unchanged file(s) from {source_label(source)}",
    )

    processed_changed = 0
    priority_entries, remaining_entries = split_priority_entries(source, changed_entries)
    ordered_groups: list[tuple[str, list[FileEntry]]] = [
        ("Copying priority JSON/JSONL", priority_entries),
        ("Copying remaining changed files", remaining_entries),
    ]

    def copy_entries_per_file(entries_to_copy: list[FileEntry], *, mode: str, label: str) -> None:
        nonlocal cancelled, copied, processed_changed
        if not entries_to_copy:
            return
        emit_progress(
            progress,
            phase="copying",
            source_id=source_id,
            source_label=source_label(source),
            source_type=source_type,
            transfer_mode=mode,
            listed=len(entries),
            total=len(entries),
            processed=skipped + processed_changed,
            copied=copied,
            skipped=skipped,
            error_count=len(errors),
            warning_count=len(warnings),
            changed=len(changed_entries),
            fallback_reason=fallback_reason,
            fallback_at=fallback_at,
            message=f"{label} {len(entries_to_copy)} file(s) from {source_label(source)}",
        )
        for entry in entries_to_copy:
            if sync_stop_requested(cancel_event):
                cancelled = True
                break
            try:
                emit_progress(
                    progress,
                    phase="copying",
                    source_id=source_id,
                    source_label=source_label(source),
                    source_type=source_type,
                    transfer_mode=mode,
                    current_file=entry.rel,
                    listed=len(entries),
                    total=len(entries),
                    processed=skipped + processed_changed,
                    copied=copied,
                    skipped=skipped,
                    error_count=len(errors),
                    warning_count=len(warnings),
                    changed=len(changed_entries),
                    fallback_reason=fallback_reason,
                    fallback_at=fallback_at,
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
                current_files[entry.rel] = listed_signatures[entry.rel]
            except Exception as exc:
                errors.append(f"{entry.rel}: {exc}")
                if previous.get(entry.rel) is not None and (dest_root / entry.rel).exists():
                    current_files[entry.rel] = previous[entry.rel]
            processed_changed += 1
            emit_progress(
                progress,
                phase="copying",
                source_id=source_id,
                source_label=source_label(source),
                source_type=source_type,
                transfer_mode=mode,
                current_file=entry.rel,
                listed=len(entries),
                total=len(entries),
                processed=skipped + processed_changed,
                copied=copied,
                skipped=skipped,
                error_count=len(errors),
                warning_count=len(warnings),
                changed=len(changed_entries),
                fallback_reason=fallback_reason,
                fallback_at=fallback_at,
                message=f"Processed {skipped + processed_changed}/{len(entries)} from {source_label(source)}",
            )

    for label, group_entries in ordered_groups:
        if cancelled or not group_entries:
            continue
        copy_entries_per_file(group_entries, mode=effective_transfer_mode, label=label)
        if sync_stop_requested(cancel_event):
            cancelled = True
            break
    if not cancelled and not changed_entries:
        emit_progress(
            progress,
            phase="copying",
            source_id=source_id,
            source_label=source_label(source),
            source_type=source_type,
            transfer_mode=effective_transfer_mode,
            current_file="",
            listed=len(entries),
            total=len(entries),
            processed=len(entries),
            copied=0,
            skipped=skipped,
            error_count=len(errors),
            warning_count=len(warnings),
            changed=0,
            fallback_reason=fallback_reason,
            fallback_at=fallback_at,
            message=f"No changed files for {source_label(source)}",
        )

    if sftp is not None:
        sftp.close()
    if client is not None:
        client.close()

    if cancelled:
        if not any("sync stopped by user" in warning for warning in warnings):
            warnings.append("sync stopped by user")
        emit_progress(
            progress,
            phase="source_cancelled",
            source_id=source_id,
            source_label=source_label(source),
            source_type=source_type,
            transfer_mode=effective_transfer_mode,
            current_file="",
            listed=len(entries),
            total=len(entries),
            processed=min(len(entries), skipped + copied),
            copied=copied,
            skipped=skipped,
            error_count=len(errors),
            warning_count=len(warnings),
            changed=len(changed_entries),
            fallback_reason=fallback_reason,
            fallback_at=fallback_at,
            cancelled=True,
            message=f"Stopped {source_label(source)} after copying {copied} file(s)",
        )
        return {
            "source_id": source_id,
            "source_label": source_label(source),
            "source_type": source_type,
            "transfer_mode": effective_transfer_mode,
            "listed": len(entries),
            "changed": len(changed_entries),
            "copied": copied,
            "skipped": skipped,
            "fallback_reason": fallback_reason,
            "fallback_at": fallback_at,
            "warnings": warnings[:50],
            "warning_count": len(warnings),
            "errors": errors[:50],
            "error_count": len(errors),
            "cancelled": True,
            "mirror_path": str(dest_root),
        }

    emit_progress(
        progress,
        phase="updating_manifest",
        source_id=source_id,
        source_label=source_label(source),
        source_type=source_type,
        transfer_mode=effective_transfer_mode,
        current_file="",
        listed=len(entries),
        total=len(entries),
        processed=len(entries),
        copied=copied,
        skipped=skipped,
        error_count=len(errors),
        warning_count=len(warnings),
        changed=len(changed_entries),
        fallback_reason=fallback_reason,
        fallback_at=fallback_at,
        message=f"Updating manifest for {source_label(source)}",
    )
    write_json(
        manifest_path,
        {
            "source_id": source_id,
            "source_label": source_label(source),
            "source_type": source_type,
            "transfer_mode": effective_transfer_mode,
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
        transfer_mode=effective_transfer_mode,
        current_file="",
        listed=len(entries),
        total=len(entries),
        processed=len(entries),
        copied=copied,
        skipped=skipped,
        error_count=len(errors),
        warning_count=len(warnings),
        changed=len(changed_entries),
        fallback_reason=fallback_reason,
        fallback_at=fallback_at,
        message=f"Finished {source_label(source)}: copied {copied}, skipped {skipped}, errors {len(errors)}",
    )
    return {
        "source_id": source_id,
        "source_label": source_label(source),
        "source_type": source_type,
        "transfer_mode": effective_transfer_mode,
        "listed": len(entries),
        "changed": len(changed_entries),
        "copied": copied,
        "skipped": skipped,
        "fallback_reason": fallback_reason,
        "fallback_at": fallback_at,
        "warnings": warnings[:50],
        "warning_count": len(warnings),
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
    if "max_parallel_sources" not in config:
        config["max_parallel_sources"] = DEFAULT_MAX_PARALLEL_SOURCES
    return config


def positive_int(value: Any, default: int, *, minimum: int = 1, maximum: int = 64) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def run_sync(
    config: dict[str, Any],
    mirror_dir: Path,
    progress: ProgressCallback | None = None,
    max_parallel_sources: int | None = None,
    cancel_event: threading.Event | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    if cancel_event is None:
        cancel_event = threading.Event()
    include_globs = list(config.get("include_globs") or [])
    exclude_globs = list(config.get("exclude_globs") or [])
    enabled_sources = [source for source in config.get("sources") or [] if isinstance(source, dict) and source_enabled(source)]
    if source_id:
        requested_source_id = safe_source_id(source_id)
        enabled_sources = [
            source
            for source in enabled_sources
            if safe_source_id(str(source.get("id") or source_label(source))) == requested_source_id
        ]
        if not enabled_sources:
            raise ValueError(f"No enabled source found for source_id={source_id}")
    configured_parallel = positive_int(
        max_parallel_sources if max_parallel_sources is not None else config.get("max_parallel_sources"),
        DEFAULT_MAX_PARALLEL_SOURCES,
    )
    worker_count = min(configured_parallel, max(1, len(enabled_sources)))
    results_by_index: list[dict[str, Any] | None] = [None] * len(enabled_sources)
    emit_progress(
        progress,
        running=True,
        phase="starting",
        total_sources=len(enabled_sources),
        source_index=0,
        completed_sources=0,
        max_parallel_sources=configured_parallel,
        stop_requested=False,
        message=f"Starting sync for {len(enabled_sources)} source(s) with up to {worker_count} parallel worker(s)",
    )
    for source_index, source in enumerate(enabled_sources, start=1):
        emit_progress(
            progress,
            running=True,
            phase="queued",
            total_sources=len(enabled_sources),
            source_index=source_index,
            completed_sources=0,
            max_parallel_sources=configured_parallel,
            source_id=safe_source_id(str(source.get("id") or source_label(source))),
            source_label=source_label(source),
            source_type=str(source.get("type") or "local").lower(),
            current_file="",
            listed=0,
            total=0,
            processed=0,
            copied=0,
            skipped=0,
            error_count=0,
            warning_count=0,
            message=f"Queued source {source_index}/{len(enabled_sources)}: {source_label(source)}",
        )

    def sync_one(source_index: int, source: dict[str, Any]) -> dict[str, Any]:
        source_id = safe_source_id(str(source.get("id") or source_label(source)))
        label = source_label(source)
        source_type = str(source.get("type") or "local").lower()

        def cancelled_result(message: str) -> dict[str, Any]:
            return {
                "source_id": source_id,
                "source_label": label,
                "source_type": source_type,
                "transfer_mode": "stopped",
                "listed": 0,
                "changed": 0,
                "copied": 0,
                "skipped": 0,
                "warnings": [message],
                "warning_count": 1,
                "errors": [],
                "error_count": 0,
                "cancelled": True,
                "mirror_path": str(source_local_root(source, mirror_dir)),
            }

        def source_progress(payload: dict[str, Any]) -> None:
            update = dict(payload)
            update.update(
                {
                    "running": True,
                    "total_sources": len(enabled_sources),
                    "source_index": source_index,
                    "max_parallel_sources": configured_parallel,
                    "stop_requested": cancel_event.is_set(),
                }
            )
            emit_progress(progress, **update)

        try:
            if cancel_event.is_set():
                emit_progress(
                    progress,
                    running=True,
                    phase="source_cancelled",
                    total_sources=len(enabled_sources),
                    source_index=source_index,
                    max_parallel_sources=configured_parallel,
                    source_id=source_id,
                    source_label=label,
                    source_type=source_type,
                    transfer_mode="stopped",
                    current_file="",
                    listed=0,
                    total=0,
                    processed=0,
                    copied=0,
                    skipped=0,
                    error_count=0,
                    warning_count=1,
                    cancelled=True,
                    stop_requested=True,
                    message=f"Skipped source after stop request: {label}",
                )
                return cancelled_result("sync stopped by user before source started")
            emit_progress(
                progress,
                running=True,
                phase="source_start",
                total_sources=len(enabled_sources),
                source_index=source_index,
                max_parallel_sources=configured_parallel,
                source_id=source_id,
                source_label=label,
                source_type=source_type,
                stop_requested=cancel_event.is_set(),
                message=f"Starting source {source_index}/{len(enabled_sources)}: {label}",
            )
            return sync_source(source, mirror_dir, include_globs, exclude_globs, progress=source_progress, cancel_event=cancel_event)
        except SyncStopped:
            emit_progress(
                progress,
                running=True,
                phase="source_cancelled",
                total_sources=len(enabled_sources),
                source_index=source_index,
                max_parallel_sources=configured_parallel,
                source_id=source_id,
                source_label=label,
                source_type=source_type,
                transfer_mode="stopped",
                current_file="",
                listed=0,
                total=0,
                processed=0,
                copied=0,
                skipped=0,
                error_count=0,
                warning_count=1,
                cancelled=True,
                stop_requested=True,
                message=f"Stopped source: {label}",
            )
            return cancelled_result("sync stopped by user")
        except Exception as exc:
            emit_progress(
                progress,
                running=True,
                phase="source_error",
                total_sources=len(enabled_sources),
                source_index=source_index,
                max_parallel_sources=configured_parallel,
                source_id=source_id,
                source_label=label,
                source_type=source_type,
                current_file="",
                listed=0,
                total=0,
                processed=0,
                copied=0,
                skipped=0,
                error_count=1,
                warning_count=0,
                stop_requested=cancel_event.is_set(),
                message=f"Source failed: {label}: {exc}",
            )
            return {
                "source_id": source_id,
                "source_label": label,
                "source_type": source_type,
                "listed": 0,
                "copied": 0,
                "skipped": 0,
                "errors": [str(exc)],
                "error_count": 1,
                "mirror_path": str(source_local_root(source, mirror_dir)),
            }

    if enabled_sources:
        completed_sources = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(sync_one, source_index, source): source_index
                for source_index, source in enumerate(enabled_sources, start=1)
            }
            for future in as_completed(future_to_index):
                source_index = future_to_index[future]
                result = future.result()
                results_by_index[source_index - 1] = result
                completed_sources += 1
                source_error_count = int(result.get("error_count") or 0)
                source_warning_count = int(result.get("warning_count") or 0)
                source_final_phase = "source_error" if source_error_count else "source_complete"
                result_source_label = str(result.get("source_label") or "")
                emit_progress(
                    progress,
                    running=True,
                    phase=source_final_phase,
                    total_sources=len(enabled_sources),
                    source_index=source_index,
                    completed_sources=completed_sources,
                    max_parallel_sources=configured_parallel,
                    source_id=result.get("source_id", ""),
                    source_label=result.get("source_label", ""),
                    source_type=result.get("source_type", ""),
                    transfer_mode=result.get("transfer_mode", ""),
                    listed=int(result.get("listed") or 0),
                    changed=int(result.get("changed") or 0),
                    copied=int(result.get("copied") or 0),
                    skipped=int(result.get("skipped") or 0),
                    error_count=source_error_count,
                    warning_count=source_warning_count,
                    stop_requested=cancel_event.is_set(),
                    cancelled=bool(result.get("cancelled")),
                    message=(
                        f"Completed {completed_sources}/{len(enabled_sources)} source(s); "
                        f"{result_source_label} had {source_error_count} error(s)"
                        if source_error_count
                        else f"Completed {completed_sources}/{len(enabled_sources)} source(s)"
                    ),
                )
    results = [result for result in results_by_index if result is not None]
    cancelled = cancel_event.is_set() or any(bool(result.get("cancelled")) for result in results)
    total_errors = sum(int(r.get("error_count") or 0) for r in results)
    total_warnings = sum(int(r.get("warning_count") or 0) for r in results)
    final_phase = "stopped" if cancelled else ("done_with_errors" if total_errors else "done")
    summary = {
        "synced_at": utc_now(),
        "max_parallel_sources": configured_parallel,
        "cancelled": cancelled,
        "error_count": total_errors,
        "warning_count": total_warnings,
        "stopped_at": utc_now() if cancelled else "",
        "results": results,
    }
    write_json(mirror_dir / "last_sync.json", summary)
    emit_progress(
        progress,
        running=False,
        phase=final_phase,
        total_sources=len(enabled_sources),
        source_index=len(enabled_sources),
        completed_sources=len(results),
        max_parallel_sources=configured_parallel,
        listed=sum(int(r.get("listed") or 0) for r in results),
        changed=sum(int(r.get("changed") or 0) for r in results),
        copied=sum(int(r.get("copied") or 0) for r in results),
        skipped=sum(int(r.get("skipped") or 0) for r in results),
        error_count=total_errors,
        warning_count=total_warnings,
        stop_requested=False,
        stopped_at=summary["stopped_at"],
        cancelled=cancelled,
        current_file="",
        message=(
            "Sync stopped by user"
            if cancelled
            else (f"Sync completed with {total_errors} error(s)" if total_errors else "Sync complete")
        ),
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


def prompt_version_for_row(row: dict[str, Any]) -> str:
    gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
    candidates = [
        gen.get("prompt_version"),
        row.get("prompt_version"),
        gen.get("prompt_id"),
        row.get("prompt_id"),
        gen.get("template_version"),
        row.get("template_version"),
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


def collect_prompt_versions(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        version = prompt_version_for_row(row)
        counts[version] = counts.get(version, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def collect_generation_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms_sum": 0.0,
        "latency_count": 0,
        "cost_usd_sum": 0.0,
        "cost_count": 0,
        "error_count": 0,
        "models": {},
    }
    for row in rows:
        gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
        provider = str(gen.get("provider") or gen.get("llm_provider") or "").strip()
        model = str(gen.get("model") or "").strip()
        model_key = "/".join(part for part in [provider, model] if part) or "unknown"
        model_stats = totals["models"].setdefault(
            model_key,
            {
                "count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms_sum": 0.0,
                "latency_count": 0,
                "cost_usd_sum": 0.0,
                "cost_count": 0,
                "error_count": 0,
            },
        )
        totals["count"] += 1
        model_stats["count"] += 1
        for target in (totals, model_stats):
            input_tokens = gen.get("input_tokens")
            if isinstance(input_tokens, (int, float)):
                target["input_tokens"] += int(input_tokens)
            output_tokens = gen.get("output_tokens")
            if isinstance(output_tokens, (int, float)):
                target["output_tokens"] += int(output_tokens)
            latency = gen.get("latency_ms")
            if isinstance(latency, (int, float)):
                target["latency_ms_sum"] += float(latency)
                target["latency_count"] += 1
            cost = gen.get("cost_usd")
            if isinstance(cost, (int, float)):
                target["cost_usd_sum"] += float(cost)
                target["cost_count"] += 1
            if gen.get("error"):
                target["error_count"] += 1

    def finalize(record: dict[str, Any]) -> dict[str, Any]:
        latency_count = int(record.get("latency_count") or 0)
        cost_count = int(record.get("cost_count") or 0)
        return {
            "count": int(record.get("count") or 0),
            "input_tokens": int(record.get("input_tokens") or 0),
            "output_tokens": int(record.get("output_tokens") or 0),
            "total_tokens": int(record.get("input_tokens") or 0) + int(record.get("output_tokens") or 0),
            "avg_latency_ms": float(record.get("latency_ms_sum") or 0.0) / latency_count if latency_count else None,
            "latency_count": latency_count,
            "cost_usd": float(record.get("cost_usd_sum") or 0.0) if cost_count else None,
            "cost_count": cost_count,
            "error_count": int(record.get("error_count") or 0),
        }

    models = {
        key: finalize(value)
        for key, value in sorted(
            totals["models"].items(),
            key=lambda kv: (-(int(kv[1].get("input_tokens") or 0) + int(kv[1].get("output_tokens") or 0)), kv[0]),
        )
    }
    finalized = finalize(totals)
    finalized["models"] = models
    return finalized


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


KNOWN_COMPONENT_TYPES = {
    "Stack",
    "Column",
    "Row",
    "Card",
    "Text",
    "Image",
    "Icon",
    "Button",
    "List",
    "Divider",
    "Tabs",
    "Tab",
    "Table",
    "Badge",
    "Chip",
    "Spacer",
    "Link",
    "Form",
    "Input",
    "TextField",
    "Select",
    "Checkbox",
    "Radio",
    "Switch",
    "Slider",
    "Modal",
    "CodeBlock",
    "ConsoleLog",
    "Formula",
    "Chart",
    "EmailPreview",
}


def increment_count(target: dict[str, int], key: Any, amount: int = 1) -> None:
    label = str(key or "unknown").strip() or "unknown"
    target[label] = target.get(label, 0) + amount


def top_dict(counts: dict[str, int], limit: int = 20) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit])


def genui_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("genui_json", "ir", "flat_spec", "payload"):
        value = row.get(key)
        if isinstance(value, dict) and isinstance(value.get("elements"), dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except Exception:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("elements"), dict):
                return parsed
    if isinstance(row.get("elements"), dict) and "root" in row:
        return row
    return None


def resolve_state_path(state: Any, path: Any) -> Any:
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    current = state
    for token in path.strip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current.get(token)
        elif isinstance(current, list) and token.isdigit():
            index = int(token)
            current = current[index] if 0 <= index < len(current) else None
        else:
            return None
    return current


def table_dimensions(props: dict[str, Any], state: Any) -> tuple[int | None, int | None]:
    columns = props.get("columns")
    rows = props.get("rows")
    if rows is None:
        rows = resolve_state_path(state, props.get("statePath"))
    column_count = len(columns) if isinstance(columns, list) else None
    row_count = len(rows) if isinstance(rows, list) else None
    return row_count, column_count


def collect_ir_structure_summary(rows: list[dict[str, Any]], limit: int = 24) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sampled": 0,
        "flat_spec_rows": 0,
        "component_count": 0,
        "avg_components": None,
        "component_types": {},
        "uncommon_component_types": {},
        "table_domains": {},
        "table_presentations": {},
        "action_types": {},
        "table_count": 0,
        "button_count": 0,
        "image_count": 0,
        "icon_count": 0,
        "chart_count": 0,
        "formula_count": 0,
        "code_count": 0,
        "email_preview_count": 0,
        "table_rows_total": 0,
        "table_rows_counted": 0,
        "table_columns_total": 0,
        "table_columns_counted": 0,
        "avg_table_rows": None,
        "avg_table_columns": None,
    }
    component_types: dict[str, int] = {}
    uncommon_types: dict[str, int] = {}
    table_domains: dict[str, int] = {}
    table_presentations: dict[str, int] = {}
    action_types: dict[str, int] = {}

    sampled_rows = rows if limit <= 0 else rows[:limit]
    for row in sampled_rows:
        summary["sampled"] += 1
        payload = genui_payload(row)
        if not payload:
            continue
        elements = payload.get("elements")
        if not isinstance(elements, dict):
            continue
        summary["flat_spec_rows"] += 1
        state = payload.get("state") if isinstance(payload.get("state"), (dict, list)) else {}
        summary["component_count"] += len(elements)
        for element in elements.values():
            if not isinstance(element, dict):
                continue
            type_name = str(element.get("type") or "unknown").strip() or "unknown"
            props = element.get("props") if isinstance(element.get("props"), dict) else {}
            increment_count(component_types, type_name)
            if type_name not in KNOWN_COMPONENT_TYPES:
                increment_count(uncommon_types, type_name)
            if type_name == "Table":
                summary["table_count"] += 1
                increment_count(table_domains, props.get("domain") or "generic")
                increment_count(table_presentations, props.get("preferredPresentation") or "auto")
                row_count, column_count = table_dimensions(props, state)
                if isinstance(row_count, int):
                    summary["table_rows_total"] += row_count
                    summary["table_rows_counted"] += 1
                if isinstance(column_count, int):
                    summary["table_columns_total"] += column_count
                    summary["table_columns_counted"] += 1
            elif type_name == "Button":
                summary["button_count"] += 1
            elif type_name == "Image":
                summary["image_count"] += 1
            elif type_name == "Icon":
                summary["icon_count"] += 1
            elif type_name == "Chart":
                summary["chart_count"] += 1
            elif type_name == "Formula":
                summary["formula_count"] += 1
            elif type_name in {"CodeBlock", "ConsoleLog"}:
                summary["code_count"] += 1
            elif type_name == "EmailPreview":
                summary["email_preview_count"] += 1
            on = element.get("on") if isinstance(element.get("on"), dict) else {}
            for event in on.values():
                if isinstance(event, dict):
                    increment_count(action_types, event.get("action") or "unknown")
            if isinstance(props.get("action"), str):
                increment_count(action_types, props.get("action"))

    flat_rows = int(summary["flat_spec_rows"] or 0)
    if flat_rows:
        summary["avg_components"] = float(summary["component_count"]) / flat_rows
    if summary["table_rows_counted"]:
        summary["avg_table_rows"] = float(summary["table_rows_total"]) / int(summary["table_rows_counted"])
    if summary["table_columns_counted"]:
        summary["avg_table_columns"] = float(summary["table_columns_total"]) / int(summary["table_columns_counted"])
    summary["component_types"] = top_dict(component_types, limit)
    summary["uncommon_component_types"] = top_dict(uncommon_types, limit)
    summary["table_domains"] = top_dict(table_domains, limit)
    summary["table_presentations"] = top_dict(table_presentations, limit)
    summary["action_types"] = top_dict(action_types, limit)
    return summary


MEDIA_COMPONENT_TYPES = {"Image", "Icon"}
MEDIA_PROP_KEYS = ("url", "src", "image", "source", "name", "icon", "uri", "path", "value")
MEDIA_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".html",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".webp",
}


def media_host(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return parsed.netloc.lower().removeprefix("www.")


def media_extension(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    parsed = urlparse(value)
    suffix = Path(parsed.path or value).suffix.lower()
    return suffix if suffix else "unknown"


def nested_media_strings(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, dict):
        found: list[str] = []
        for key in MEDIA_PROP_KEYS:
            if key in value:
                found.extend(nested_media_strings(value.get(key)))
        return found
    if isinstance(value, list):
        found: list[str] = []
        for item in value:
            found.extend(nested_media_strings(item))
        return found
    return []


def element_media_values(element_type: str, props: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in MEDIA_PROP_KEYS:
        if key in props:
            values.extend(nested_media_strings(props.get(key)))
    if element_type == "Icon" and not values:
        # Icon names may be symbolic rather than file paths; only count explicit media-looking values.
        for key in ("label", "alt"):
            values.extend(nested_media_strings(props.get(key)))
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def local_media_path(run_dir: Path, value: str) -> Path | None:
    raw = value.strip().strip("\"'")
    if not raw or media_host(raw):
        return None
    if raw.startswith("file://"):
        raw = raw[7:]
    normalized = raw.replace("\\", "/")
    while normalized.startswith("../"):
        normalized = normalized[3:]
    if "/assets/" in normalized:
        normalized = normalized[normalized.index("/assets/") + 1 :]
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("assets/"):
        return run_dir / PurePosixPath(normalized)
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return None


def media_ref_sample(record_id: str, ref_type: str, value: str) -> dict[str, str]:
    return {"id": record_id, "type": ref_type, "value": value[:240]}


def collect_media_asset_summary(
    run_dir: Path,
    response_rows: list[dict[str, Any]],
    genui_rows: list[dict[str, Any]],
    limit: int = 48,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sampled_response_rows": 0,
        "sampled_ir_rows": 0,
        "response_asset_records": 0,
        "response_asset_files_present": 0,
        "response_asset_files_missing": 0,
        "response_asset_bytes": 0,
        "ir_media_components": 0,
        "ir_image_components": 0,
        "ir_icon_components": 0,
        "ir_media_refs": 0,
        "ir_local_media_refs": 0,
        "ir_local_media_missing": 0,
        "ir_remote_media_refs": 0,
        "ir_symbolic_icon_refs": 0,
        "asset_file_extensions": {},
        "response_asset_hosts": {},
        "ir_remote_hosts": {},
        "missing_local_samples": [],
        "remote_ref_samples": [],
    }
    asset_exts: dict[str, int] = {}
    response_hosts: dict[str, int] = {}
    ir_hosts: dict[str, int] = {}
    missing_samples: list[dict[str, str]] = []
    remote_samples: list[dict[str, str]] = []

    for row in response_rows[:limit]:
        summary["sampled_response_rows"] += 1
        record_id = str(row.get("response_id") or row.get("query_id") or "response")
        for asset in row.get("assets") or []:
            if not isinstance(asset, dict):
                continue
            summary["response_asset_records"] += 1
            url = asset.get("url") or asset.get("source_url") or asset.get("source")
            host = media_host(url)
            if host:
                increment_count(response_hosts, host)
            increment_count(asset_exts, media_extension(str(asset.get("path") or url or "")))
            if isinstance(asset.get("bytes"), (int, float)):
                summary["response_asset_bytes"] += int(asset.get("bytes") or 0)
            path_value = str(asset.get("path") or "").strip()
            resolved = local_media_path(run_dir, path_value) if path_value else None
            if resolved and resolved.exists():
                summary["response_asset_files_present"] += 1
            elif path_value:
                summary["response_asset_files_missing"] += 1
                if len(missing_samples) < 8:
                    missing_samples.append(media_ref_sample(record_id, "response asset", path_value))

    for row in genui_rows[:limit]:
        summary["sampled_ir_rows"] += 1
        record_id = str(row.get("ui_id") or row.get("response_id") or row.get("query_id") or "IR")
        payload = genui_payload(row)
        elements = payload.get("elements") if isinstance(payload, dict) else None
        if not isinstance(elements, dict):
            continue
        for element in elements.values():
            if not isinstance(element, dict):
                continue
            element_type = str(element.get("type") or "").strip()
            if element_type not in MEDIA_COMPONENT_TYPES:
                continue
            summary["ir_media_components"] += 1
            if element_type == "Image":
                summary["ir_image_components"] += 1
            if element_type == "Icon":
                summary["ir_icon_components"] += 1
            props = element.get("props") if isinstance(element.get("props"), dict) else {}
            for value in element_media_values(element_type, props):
                host = media_host(value)
                resolved = local_media_path(run_dir, value)
                if host:
                    summary["ir_media_refs"] += 1
                    summary["ir_remote_media_refs"] += 1
                    increment_count(ir_hosts, host)
                    if len(remote_samples) < 8:
                        remote_samples.append(media_ref_sample(record_id, element_type, value))
                elif resolved:
                    summary["ir_media_refs"] += 1
                    summary["ir_local_media_refs"] += 1
                    increment_count(asset_exts, media_extension(value))
                    if not resolved.exists():
                        summary["ir_local_media_missing"] += 1
                        if len(missing_samples) < 8:
                            missing_samples.append(media_ref_sample(record_id, element_type, value))
                elif element_type == "Icon":
                    summary["ir_symbolic_icon_refs"] += 1

    summary["asset_file_extensions"] = top_dict(asset_exts, 12)
    summary["response_asset_hosts"] = top_dict(response_hosts, 12)
    summary["ir_remote_hosts"] = top_dict(ir_hosts, 12)
    summary["missing_local_samples"] = missing_samples
    summary["remote_ref_samples"] = remote_samples
    return summary


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


def collect_training_readiness_summary(rows: list[dict[str, Any]], responses: int, genui: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "paired_records": min(responses, genui),
        "sampled": 0,
        "json_parse_ok": 0,
        "strict_valid": 0,
        "score_ge_60": 0,
        "score_ge_70": 0,
        "score_ge_80": 0,
        "score_unknown": 0,
        "content_ge_65": 0,
        "repair_free": 0,
        "fallback_free": 0,
        "markdown_clean": 0,
        "gen_error_free": 0,
        "ready_score70": 0,
        "ready_score80": 0,
        "estimated_ready_score70": 0,
        "estimated_ready_score80": 0,
        "avg_output_tokens_json": None,
        "avg_component_count": None,
    }
    output_token_sum = 0.0
    output_token_count = 0
    component_sum = 0.0
    component_count = 0
    for row in rows:
        summary["sampled"] += 1
        validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
        json_ok = validation.get("json_parse_ok") is True
        strict_valid = validation.get("schema_valid_strict") is True
        repair_free = not (validation.get("repair_needed") is True or int(validation.get("repair_attempts") or 0) > 0)
        fallback_free = not (row.get("fallback_generated") or validation.get("fallback_generated"))
        gen_error_free = not bool(gen.get("error"))
        markdown = metrics.get("markdown_leakage_rate")
        markdown_clean = not (isinstance(markdown, (int, float)) and markdown > 0)
        score = metrics.get("overall_score")
        score_value = float(score) if isinstance(score, (int, float)) else None
        content_coverage = metrics.get("content_coverage")
        if json_ok:
            summary["json_parse_ok"] += 1
        if strict_valid:
            summary["strict_valid"] += 1
        if repair_free:
            summary["repair_free"] += 1
        if fallback_free:
            summary["fallback_free"] += 1
        if markdown_clean:
            summary["markdown_clean"] += 1
        if gen_error_free:
            summary["gen_error_free"] += 1
        if score_value is None:
            summary["score_unknown"] += 1
        else:
            if score_value >= 60:
                summary["score_ge_60"] += 1
            if score_value >= 70:
                summary["score_ge_70"] += 1
            if score_value >= 80:
                summary["score_ge_80"] += 1
        if isinstance(content_coverage, (int, float)) and content_coverage >= 0.65:
            summary["content_ge_65"] += 1
        output_tokens = metrics.get("output_tokens_json")
        if isinstance(output_tokens, (int, float)):
            output_token_sum += float(output_tokens)
            output_token_count += 1
        components = metrics.get("component_count")
        if isinstance(components, (int, float)):
            component_sum += float(components)
            component_count += 1
        base_ready = json_ok and strict_valid and gen_error_free and fallback_free and markdown_clean
        if base_ready and score_value is not None and score_value >= 70:
            summary["ready_score70"] += 1
        if base_ready and score_value is not None and score_value >= 80:
            summary["ready_score80"] += 1

    sampled = int(summary["sampled"] or 0)
    paired = int(summary["paired_records"] or 0)
    if sampled:
        summary["estimated_ready_score70"] = int(round(paired * int(summary["ready_score70"]) / sampled))
        summary["estimated_ready_score80"] = int(round(paired * int(summary["ready_score80"]) / sampled))
    if output_token_count:
        summary["avg_output_tokens_json"] = output_token_sum / output_token_count
    if component_count:
        summary["avg_component_count"] = component_sum / component_count
    return summary


def compact_text(value: Any, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def response_text_for_row(row: dict[str, Any]) -> str:
    for key in ("response_text", "text", "content", "answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def query_text_for_row(row: dict[str, Any]) -> str:
    for key in ("query_text", "query", "prompt", "user_query"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def normalize_duplicate_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def duplicate_bucket_summary(values: list[str], limit: int = 5) -> dict[str, Any]:
    counts: dict[str, int] = {}
    previews: dict[str, str] = {}
    for value in values:
        normalized = normalize_duplicate_text(value)
        if not normalized:
            continue
        key = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
        counts[key] = counts.get(key, 0) + 1
        previews.setdefault(key, compact_text(value, 220))
    duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
    duplicate_groups = sum(1 for count in counts.values() if count > 1)
    examples = [
        {"count": count, "preview": previews[key]}
        for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if count > 1
    ][:limit]
    sampled = sum(counts.values())
    return {
        "sampled": sampled,
        "unique": len(counts),
        "duplicate_rows": duplicate_rows,
        "duplicate_groups": duplicate_groups,
        "duplicate_rate": duplicate_rows / sampled if sampled else 0.0,
        "examples": examples,
    }


def ir_text_for_duplication(row: dict[str, Any]) -> str:
    payload = genui_payload(row)
    if payload:
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return str(payload)
    value = row.get("genui_json") or row.get("ir") or row.get("flat_spec") or row.get("payload")
    if isinstance(value, str):
        return value
    if value is not None:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return str(value)
    return ""


def collect_content_duplication_summary(
    query_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    genui_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    queries = duplicate_bucket_summary([query_text_for_row(row) for row in query_rows])
    responses = duplicate_bucket_summary([response_text_for_row(row) for row in response_rows])
    ir_payloads = duplicate_bucket_summary([ir_text_for_duplication(row) for row in genui_rows])
    total_duplicate_rows = (
        int(queries["duplicate_rows"])
        + int(responses["duplicate_rows"])
        + int(ir_payloads["duplicate_rows"])
    )
    return {
        "sample_limit": max(len(query_rows), len(response_rows), len(genui_rows)),
        "queries": queries,
        "responses": responses,
        "ir_payloads": ir_payloads,
        "total_duplicate_rows": total_duplicate_rows,
    }


def legacy_component_list(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("genui_json")
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except Exception:
            return []
    if not isinstance(value, list):
        return []
    components: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        update = item.get("updateComponents")
        if isinstance(update, dict) and isinstance(update.get("components"), list):
            components.extend(comp for comp in update["components"] if isinstance(comp, dict))
    return components


def summarize_ir_for_sample(row: dict[str, Any]) -> dict[str, Any]:
    payload = genui_payload(row)
    if payload:
        elements = payload.get("elements") if isinstance(payload.get("elements"), dict) else {}
        type_counts: dict[str, int] = {}
        table_count = 0
        image_count = 0
        icon_count = 0
        button_count = 0
        for element in elements.values():
            if not isinstance(element, dict):
                continue
            type_name = str(element.get("type") or "unknown").strip() or "unknown"
            increment_count(type_counts, type_name)
            if type_name == "Table":
                table_count += 1
            elif type_name == "Image":
                image_count += 1
            elif type_name == "Icon":
                icon_count += 1
            elif type_name == "Button":
                button_count += 1
        return {
            "format": "flat_spec",
            "root": str(payload.get("root") or ""),
            "component_count": len(elements),
            "component_types": top_dict(type_counts, 8),
            "table_count": table_count,
            "image_count": image_count,
            "icon_count": icon_count,
            "button_count": button_count,
        }
    components = legacy_component_list(row)
    type_counts: dict[str, int] = {}
    for component in components:
        type_name = str(component.get("type") or component.get("component") or "unknown").strip() or "unknown"
        increment_count(type_counts, type_name)
    return {
        "format": "legacy" if components else "unknown",
        "root": "",
        "component_count": len(components),
        "component_types": top_dict(type_counts, 8),
        "table_count": type_counts.get("Table", 0),
        "image_count": type_counts.get("Image", 0),
        "icon_count": type_counts.get("Icon", 0),
        "button_count": type_counts.get("Button", 0),
    }


def collect_record_samples(
    query_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    genui_rows: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    queries_by_id = {str(row.get("query_id") or ""): row for row in query_rows if row.get("query_id")}
    responses_by_id = {str(row.get("response_id") or ""): row for row in response_rows if row.get("response_id")}
    candidates: list[tuple[tuple[float, int, int], dict[str, Any]]] = []
    for index, row in enumerate(genui_rows, start=1):
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
        score = metrics.get("overall_score")
        score_value = float(score) if isinstance(score, (int, float)) else None
        issues = row_issue_labels(row)
        response_id = str(row.get("response_id") or "").strip()
        query_id = str(row.get("query_id") or "").strip()
        response = responses_by_id.get(response_id, {})
        query = queries_by_id.get(query_id or str(response.get("query_id") or ""), {})
        query_id = query_id or str(response.get("query_id") or "")
        provider = str(gen.get("provider") or gen.get("llm_provider") or "").strip()
        model = str(gen.get("model") or "").strip()
        sample = {
            "row": index,
            "ui_id": str(row.get("ui_id") or "").strip(),
            "response_id": response_id,
            "query_id": query_id,
            "intent": str(row.get("intent") or row.get("intent_bucket") or query.get("intent") or "").strip(),
            "score": score_value,
            "issues": issues,
            "model": "/".join(part for part in [provider, model] if part) or "unknown",
            "prompt_version": ir_version_for_row(row),
            "query_text": compact_text(query_text_for_row(query), 520),
            "response_preview": compact_text(response_text_for_row(response), 900),
            "ir_summary": summarize_ir_for_sample(row),
            "validation_errors": [
                str(item)[:180]
                for item in (validation.get("errors") or validation.get("warnings") or [])
                if str(item).strip()
            ][:3],
        }
        issue_rank = 0 if issues else 1
        score_rank = score_value if score_value is not None else 999.0
        candidates.append(((issue_rank, score_rank, index), sample))
    candidates.sort(key=lambda item: item[0])
    return [sample for _, sample in candidates[:limit]]


def ir_version_for_row(row: dict[str, Any]) -> str:
    return prompt_version_for_row(row)


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


def tail_text(path: Path, max_bytes: int = 200_000) -> str:
    if not path.exists() or path.stat().st_size <= 0:
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(max(0, size - max_bytes))
        data = handle.read(max_bytes)
    return data.decode("utf-8", errors="replace")


def run_log_files(run_dir: Path, limit: int = 8) -> list[Path]:
    candidates: dict[str, Path] = {}
    for pattern in ("*.log", "logs/*.log", "*/run.log"):
        for path in run_dir.glob(pattern):
            if path.is_file():
                candidates[path.resolve().as_posix()] = path
    return sorted(candidates.values(), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


LOG_ISSUE_RE = re.compile(
    r"\b(error|exception|traceback|failed|failure|fatal|timeout|timed out|http\s*[45]\d\d|rate limit|quota|oom|out of memory|killed|connection refused)\b",
    re.IGNORECASE,
)
LOG_PROGRESS_RE = re.compile(
    r"\b(stage\s*[1-5]|generated|created|completed|finished|saved|wrote|score|aggregate|progress|processed|batch|queries|responses|genui|ir)\b",
    re.IGNORECASE,
)


def summarize_run_logs(run_dir: Path) -> dict[str, Any]:
    files = run_log_files(run_dir)
    if not files:
        return {"present": False, "files": [], "issue_count": 0, "progress_count": 0}
    summaries = []
    issue_lines: list[dict[str, Any]] = []
    progress_lines: list[dict[str, Any]] = []
    latest_mtime = 0.0
    total_bytes = 0
    for path in files:
        stat = path.stat()
        latest_mtime = max(latest_mtime, stat.st_mtime)
        total_bytes += int(stat.st_size)
        text = tail_text(path)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        file_issue_count = 0
        file_progress_count = 0
        for line_number, line in enumerate(lines[-1000:], start=max(1, len(lines) - 999)):
            compact = line[-500:]
            if LOG_ISSUE_RE.search(line):
                file_issue_count += 1
                if len(issue_lines) < 12:
                    issue_lines.append({"file": path.name, "line": line_number, "text": compact})
            elif LOG_PROGRESS_RE.search(line):
                file_progress_count += 1
                if len(progress_lines) < 12:
                    progress_lines.append({"file": path.name, "line": line_number, "text": compact})
        summaries.append(
            {
                "name": path.name,
                "path": str(path),
                "size": int(stat.st_size),
                "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "issue_count_tail": file_issue_count,
                "progress_count_tail": file_progress_count,
                "last_lines": lines[-5:],
            }
        )
    return {
        "present": True,
        "files": summaries,
        "file_count": len(summaries),
        "total_bytes": total_bytes,
        "latest_updated_at": datetime.fromtimestamp(latest_mtime, timezone.utc).isoformat() if latest_mtime else None,
        "issue_count": sum(item["issue_count_tail"] for item in summaries),
        "progress_count": sum(item["progress_count_tail"] for item in summaries),
        "recent_issues": issue_lines,
        "recent_progress": progress_lines[:8],
    }


def summarize_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest = load_json(manifest_path, None)
    except Exception as exc:
        return {"present": False, "parse_error": str(exc), "path": str(manifest_path)}
    if not isinstance(manifest, dict):
        return {"present": False}

    repo = manifest.get("repo") if isinstance(manifest.get("repo"), dict) else {}
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    command = manifest.get("command") if isinstance(manifest.get("command"), dict) else {}
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    components = repo.get("components") if isinstance(repo.get("components"), dict) else {}
    compatibility = repo.get("compatibility") if isinstance(repo.get("compatibility"), dict) else {}

    settings = {
        key: value
        for key, value in manifest.items()
        if key.endswith("_settings") and isinstance(value, dict)
    }
    setting_summary: dict[str, Any] = {}
    for name, values in settings.items():
        useful = {}
        for key, value in values.items():
            if key in {
                "run_id",
                "target",
                "chunk_size",
                "model",
                "intent_count",
                "k_per_intent",
                "target_per_intent",
                "stage1_batch_size",
                "stage1_intent_batch_size",
                "stage2_response_batch_size",
                "stage2_query_batch_size",
                "stage3_batch_size",
                "rate_limit_qps",
                "call_sleep_seconds",
                "started_at",
                "max_extra_source",
                "replacement_batch_size",
            }:
                useful[key] = value
        setting_summary[name] = useful or {key: values[key] for key in list(values.keys())[:12]}

    argv = command.get("argv") if isinstance(command.get("argv"), list) else []
    return {
        "present": True,
        "manifest_version": manifest.get("manifest_version"),
        "generated_at": manifest.get("generated_at"),
        "stage": manifest.get("stage"),
        "repo": {
            "release_version": repo.get("release_version"),
            "git_commit": repo.get("git_commit"),
            "git_branch": repo.get("git_branch"),
            "git_dirty": repo.get("git_dirty"),
            "components": components,
            "compatibility": compatibility,
        },
        "model": {
            "name": model.get("name"),
            "provider": model.get("provider"),
            "model": model.get("model"),
        },
        "config": {
            "combined_sha256": config.get("combined_sha256"),
            "file_count": len(config.get("files") or {}) if isinstance(config.get("files"), dict) else 0,
            "files": list((config.get("files") or {}).keys())[:12] if isinstance(config.get("files"), dict) else [],
        },
        "paths": {key: paths.get(key) for key in ("run_dir", "queries_path", "responses_path", "genui_path", "aggregates_path") if paths.get(key)},
        "command": {
            "cwd": command.get("cwd"),
            "argv": argv[:24],
            "argv_truncated": len(argv) > 24,
        },
        "settings": setting_summary,
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
    configured_sources = [source for source in config.get("sources") or [] if isinstance(source, dict)]
    include_local_checkout = config_bool(config.get("include_local_checkout"), default=not configured_sources)
    sources: list[dict[str, Any]] = []
    if include_local_checkout:
        sources.append(
            {
                "id": "_local_checkout",
                "label": "Local checkout",
                "type": "local",
                "path": "data/runs",
                "mirror_local_in_place": True,
                "enabled": True,
            }
        )
    sources.extend(configured_sources)
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
        "transfer_mode": source_transfer_mode(source),
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
        "max_parallel_sources": positive_int(config.get("max_parallel_sources"), DEFAULT_MAX_PARALLEL_SOURCES),
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
        warnings = sync_result.get("warnings") or []
        error_count = int(sync_result.get("error_count") or 0)
        warning_count = int(sync_result.get("warning_count") or 0)
        fallback_reason = str(sync_result.get("fallback_reason") or "").strip()
        status = "error" if error_count else ("warn" if warning_count or fallback_reason else "ok")
        message = (
            str(errors[0])
            if errors
            else (fallback_reason or (str(warnings[0]) if warnings else f"Last sync listed {int(sync_result.get('listed') or 0)} files"))
        )
        return {
            "status": status,
            "message": message,
            "last_synced_at": latest_sync.get("synced_at") if isinstance(latest_sync, dict) else None,
            "listed": int(sync_result.get("listed") or 0),
            "copied": int(sync_result.get("copied") or 0),
            "skipped": int(sync_result.get("skipped") or 0),
            "warning_count": warning_count,
            "fallback_reason": fallback_reason,
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
        "query_prompt_versions": collect_prompt_versions(query_rows),
        "response_prompt_versions": collect_prompt_versions(response_rows),
        "ir_prompt_versions": collect_prompt_versions(genui_rows),
        "query_usage": collect_generation_usage(query_rows),
        "response_usage": collect_generation_usage(response_rows),
        "ir_usage": collect_generation_usage(genui_rows),
        "metric_avgs": collect_metric_avgs(genui_rows),
        "intent_quality": collect_intent_quality(genui_rows),
        "ir_structure": collect_ir_structure_summary(genui_rows),
        "media_health": collect_media_asset_summary(run_dir, response_rows, genui_rows),
        "quality_summary": collect_quality_summary(genui_rows),
        "training_readiness": collect_training_readiness_summary(genui_rows, responses, genui),
        "record_samples": collect_record_samples(query_rows, response_rows, genui_rows),
        "content_duplicates": collect_content_duplication_summary(query_rows, response_rows, genui_rows),
        "data_integrity": collect_data_integrity_summary(run_dir, queries, responses, genui),
        "run_logs": summarize_run_logs(run_dir),
        "artifacts": run_artifacts(run_dir),
        "manifest_summary": summarize_manifest(run_dir),
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
                "log_issue_run_count": sum(1 for r in source_runs if int((r.get("run_logs") or {}).get("issue_count") or 0)),
                "log_issue_count": sum(int((r.get("run_logs") or {}).get("issue_count") or 0) for r in source_runs),
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
        "log_issue_run_count": sum(1 for r in runs if int((r.get("run_logs") or {}).get("issue_count") or 0)),
        "log_issue_count": sum(int((r.get("run_logs") or {}).get("issue_count") or 0) for r in runs),
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
    .sample-list { display:grid; gap: 10px; margin-top: 10px; }
    .sample-card { border: 1px solid var(--line); border-radius: 16px; padding: 10px; background: rgba(255,255,255,.55); }
    .sample-card summary { cursor:pointer; font-weight: 800; }
    .sample-preview { white-space: pre-wrap; background: rgba(23,32,42,.04); border: 1px solid var(--line); border-radius: 12px; padding: 9px; margin: 8px 0; max-height: 180px; overflow:auto; }
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
    .sync-source-list { margin-top: 12px; display:grid; gap: 8px; }
    .sync-source-row { display:grid; grid-template-columns: minmax(150px, 1.1fr) minmax(180px, 1.4fr) minmax(150px, 1fr); gap: 10px; align-items:center; border-top: 1px solid var(--line); padding-top: 8px; }
    .sync-source-row .bar-track { margin: 3px 0; height: 8px; }
    .status { min-height: 20px; color: var(--muted); font-size: 13px; }
    html { scroll-behavior: smooth; }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(rgba(23,32,42,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(23,32,42,.035) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(to bottom, rgba(0,0,0,.72), transparent 72%);
    }
    header { padding: 24px 34px 18px; }
    .header-card {
      position: relative;
      overflow: visible;
      max-width: 1580px;
      margin: 0 auto;
      border: 1px solid rgba(255,255,255,.72);
      border-radius: 30px;
      padding: clamp(20px, 3vw, 34px);
      background:
        radial-gradient(circle at 8% 12%, rgba(15,118,110,.18), transparent 34%),
        radial-gradient(circle at 92% 0%, rgba(194,65,12,.16), transparent 32%),
        linear-gradient(135deg, rgba(255,255,255,.92), rgba(255,255,255,.68));
      box-shadow: 0 28px 80px rgba(32,38,46,.12);
      backdrop-filter: blur(18px);
    }
    .header-card::after {
      content: "";
      position: absolute;
      right: -90px;
      top: -120px;
      width: 310px;
      height: 310px;
      border-radius: 999px;
      background: conic-gradient(from 120deg, rgba(15,118,110,.22), rgba(194,65,12,.18), rgba(15,118,110,.05), rgba(15,118,110,.22));
      filter: blur(1px);
      opacity: .55;
      pointer-events: none;
      z-index: 0;
    }
    .hero-row {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    h1 { font-size: clamp(34px, 4.4vw, 66px); line-height: .95; max-width: 980px; }
    .sub { font-size: 15px; line-height: 1.55; max-width: 900px; }
    .toolbar {
      position: relative;
      z-index: 1;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 24px;
      padding: 14px;
      border: 1px solid rgba(36,48,64,.10);
      border-radius: 24px;
      background: rgba(255,255,255,.58);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.72);
    }
    .toolbar > * { flex: 0 1 auto; }
    .toolbar input:not([type="date"]), .toolbar select { min-width: 150px; }
    .toolbar #filter { flex: 1 1 320px; min-width: min(330px, 100%); }
    button, input, select { transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease, background .16s ease; }
    button:hover { transform: translateY(-1px); box-shadow: 0 14px 28px rgba(15,118,110,.22); }
    .ghost-btn:hover { background: rgba(15,118,110,.08); box-shadow: 0 10px 22px rgba(15,118,110,.08); }
    .danger-btn {
      background: #b91c1c;
      color: #fff;
      box-shadow: 0 10px 24px rgba(185,28,28,.18);
    }
    .danger-btn:hover { box-shadow: 0 14px 28px rgba(185,28,28,.22); }
    button:focus-visible, input:focus-visible, select:focus-visible, a:focus-visible, summary:focus-visible {
      outline: none;
      box-shadow: 0 0 0 4px rgba(15,118,110,.18);
      border-color: rgba(15,118,110,.42);
    }
    .quick-nav {
      position: sticky;
      top: 10px;
      z-index: 5;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 14px;
      padding: 10px;
      border: 1px solid rgba(36,48,64,.10);
      border-radius: 999px;
      background: rgba(255,255,255,.78);
      backdrop-filter: blur(14px);
      box-shadow: 0 14px 30px rgba(32,38,46,.09);
    }
    .quick-nav a {
      text-decoration: none;
      color: #115e59;
      font-size: 12px;
      font-weight: 850;
      padding: 8px 10px;
      border-radius: 999px;
    }
    .quick-nav a:hover { background: rgba(15,118,110,.10); }
    main {
      max-width: 1580px;
      margin: 0 auto;
      padding: 0 34px 42px;
    }
    .stats { grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 16px; margin: 18px 0 26px; }
    .stat, .panel { border-color: rgba(36,48,64,.10); box-shadow: 0 18px 50px rgba(32,38,46,.09); }
    .stat {
      position: relative;
      overflow: hidden;
      min-height: 86px;
      padding: 20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.94), rgba(255,255,255,.74));
    }
    .stat::before {
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 4px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      opacity: .85;
    }
    .stat .v { font-size: clamp(26px, 3vw, 38px); }
    .stat .k { font-weight: 750; text-transform: uppercase; letter-spacing: .04em; }
    .panel {
      padding: 20px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.88), rgba(255,255,255,.72));
    }
    .panel > h2 {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 14px;
    }
    .panel > h2::before {
      content: "";
      width: 8px;
      height: 24px;
      border-radius: 999px;
      background: linear-gradient(180deg, var(--accent), var(--accent2));
      flex: 0 0 auto;
    }
    .wide-panel { margin-top: 20px; }
    .grid { grid-template-columns: minmax(320px, 430px) minmax(0, 1fr); gap: 20px; margin-top: 20px; }
    .two-col-panels { gap: 20px; }
    .scroll {
      border: 1px solid rgba(36,48,64,.08);
      border-radius: 16px;
      background: rgba(255,255,255,.44);
    }
    .scroll table { min-width: 760px; }
    .scroll thead th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(250,247,239,.96);
      backdrop-filter: blur(12px);
    }
    th { color: #526070; font-weight: 900; }
    td { line-height: 1.42; }
    tbody tr:nth-child(2n) td { background: rgba(255,255,255,.24); }
    tr:hover td { background: rgba(15,118,110,.075); }
    .source-card, .sample-card, .config-source, .detail-box {
      background: rgba(255,255,255,.68);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.72);
    }
    .source-card {
      transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    }
    .source-card:hover {
      transform: translateY(-1px);
      border-color: rgba(15,118,110,.22);
      box-shadow: 0 12px 26px rgba(32,38,46,.08);
    }
    .badge { border: 1px solid rgba(15,118,110,.12); }
    .bar-track {
      background: rgba(15,118,110,.12);
      box-shadow: inset 0 1px 2px rgba(23,32,42,.08);
    }
    .bar-fill { box-shadow: 0 0 20px rgba(15,118,110,.24); }
    .sync-panel {
      position: relative;
      z-index: 1;
      max-width: none;
      margin-top: 14px;
      padding: 16px;
      background:
        linear-gradient(135deg, rgba(15,118,110,.10), rgba(255,255,255,.70)),
        rgba(255,255,255,.72);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.70);
    }
    .sync-source-row {
      border: 1px solid rgba(36,48,64,.10);
      border-radius: 16px;
      padding: 12px;
      background: rgba(255,255,255,.66);
    }
    .sync-messages div {
      padding: 6px 8px;
      border-radius: 10px;
      background: rgba(255,255,255,.58);
    }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after { transition: none !important; }
    }
    @media (max-width: 980px) {
      .hero-row, .grid, .two-col-panels, .freshness-row, .eta-row, .sync-source-row { grid-template-columns: 1fr; }
      .quick-nav { border-radius: 22px; }
      .toolbar input:not([type="date"]), .toolbar select, .toolbar button, .toolbar label { width: 100%; box-sizing: border-box; }
      header, main { padding-left:18px; padding-right:18px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-card">
      <div class="hero-row">
        <div>
          <h1>GenUICraft Dataset Dashboard</h1>
          <p class="sub">Monitor generated queries, responses, IR quality, source sync, training readiness, media health, and model regressions from one local mirror.</p>
        </div>
      </div>
      <div class="toolbar">
        <button id="syncBtn">Sync sources</button>
        <button class="danger-btn" id="stopSyncBtn" disabled>Stop Sync</button>
        <button id="refreshBtn">Refresh scan</button>
        <button class="ghost-btn" id="exportCsvBtn">Export CSV</button>
        <button class="ghost-btn" id="exportJsonBtn">Export JSON</button>
        <label class="date-label">Parallel sources
          <input id="maxParallelSources" type="number" min="1" max="64" value="10" title="Maximum sources to sync at once" style="width:76px" />
        </label>
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
          <option value="integrity">Data integrity issues</option>
          <option value="duplicates">Content duplicates</option>
          <option value="logs">Log issues</option>
          <option value="metric_risk">Metric risk</option>
          <option value="low_coverage">Low content coverage</option>
          <option value="low_media">Low media usage</option>
          <option value="media_refs">Broken media refs</option>
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
          <option value="integrity_desc">Sort: integrity issues</option>
          <option value="duplicates_desc">Sort: content duplicates</option>
          <option value="logs_desc">Sort: log issues</option>
          <option value="metric_risk_desc">Sort: metric risk</option>
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
      <nav class="quick-nav" aria-label="Dashboard sections">
        <a href="#stats">Overview</a>
        <a href="#syncConfig">Sync</a>
        <a href="#sources">Sources</a>
        <a href="#runsPanel">Runs</a>
        <a href="#metricsOverview">Metrics</a>
        <a href="#trainingReadiness">Training</a>
        <a href="#irStructure">IR Structure</a>
        <a href="#trend">Trends</a>
      </nav>
      <div class="sync-panel" id="syncPanel">
        <div class="sync-top">
          <b id="syncPhase">Sync idle</b>
          <span class="small" id="syncCounters"></span>
        </div>
        <div class="bar-track"><div class="bar-fill" id="syncBar" style="width:0%"></div></div>
        <div class="small" id="syncFile"></div>
        <div class="sync-source-list" id="syncSourceProgress"></div>
        <div class="sync-messages small" id="syncMessages"></div>
      </div>
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
    <section class="panel wide-panel">
      <h2>Action Items</h2>
      <div id="actionItems"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Run Logs</h2>
      <div id="runLogs"></div>
    </section>
    <section class="grid">
      <aside class="panel">
        <h2>Sources</h2>
        <div id="sources"></div>
      </aside>
      <section class="panel" id="runsPanel">
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
      <h2>Completion Funnel</h2>
      <div id="completionFunnel"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Data Integrity</h2>
      <div id="dataIntegrity"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Content Duplicates</h2>
      <div id="contentDuplicates"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Filtered Metrics Overview</h2>
      <div id="metricsOverview"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Metric Risk</h2>
      <div id="metricRisk"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Training Readiness</h2>
      <div id="trainingReadiness"></div>
    </section>
    <section class="panel wide-panel">
      <h2>IR Structure</h2>
      <div id="irStructure"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Media & Asset Health</h2>
      <div id="mediaHealth"></div>
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
      <h2>Prompt Provenance</h2>
      <div id="promptProvenance"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Regression Watch</h2>
      <div id="regressionWatch"></div>
    </section>
    <section class="panel wide-panel">
      <h2>IR Version Quality</h2>
      <div id="irVersionQuality"></div>
    </section>
    <section class="panel wide-panel">
      <h2>Token Cost Latency</h2>
      <div id="usagePanel"></div>
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
    let lastSyncSummaryRefreshMs = 0;
    let lastSyncCopiedForSummary = -1;
    let sourceHealthOverrides = {};
    let selectedRunKey = null;
    let runPage = 1;
    const syncSummaryRefreshIntervalMs = 3000;
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
    const compactNumber = value => {
      const n = Number(value || 0);
      if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
      if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
      if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
      return fmt(n);
    };
    const latencyText = value => {
      if (value == null || Number.isNaN(Number(value))) return "n/a";
      const ms = Number(value);
      if (ms >= 60000) return `${(ms / 60000).toFixed(1)}m`;
      if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
      return `${ms.toFixed(0)}ms`;
    };
    const costText = value => value == null || Number.isNaN(Number(value)) ? "n/a" : `$${Number(value).toFixed(Number(value) < 1 ? 4 : 2)}`;
    const modelText = obj => {
      const entries = Object.entries(obj || {}).slice(0, 3);
      return entries.length ? entries.map(([k,v]) => `${k} (${v})`).join("<br>") : "<span class='small'>n/a</span>";
    };
    const versionText = obj => {
      const entries = Object.entries(obj || {}).slice(0, 3);
      return entries.length ? entries.map(([k,v]) => `${k} (${v})`).join("<br>") : "<span class='small'>n/a</span>";
    };
    function setStatus(text) { document.getElementById("status").textContent = text || ""; }
    function syncPhaseLabel(phase) {
      return ({
        listing: "listing remote files",
        checking_changed_files: "checking changed files",
        updating_manifest: "updating manifest",
        copying: "copying files",
        stopping: "stopping",
        stopped: "stopped",
        source_cancelled: "stopped",
        source_done: "done",
        source_error: "source error",
        source_complete: "done",
        queued: "queued",
        done: "done",
        done_with_errors: "completed with errors",
      })[phase] || phase || "status";
    }
    function renderSyncStatus(s) {
      const panel = document.getElementById("syncPanel");
      const active = s && (s.running || (s.phase && s.phase !== "idle"));
      const syncBtn = document.getElementById("syncBtn");
      const stopBtn = document.getElementById("stopSyncBtn");
      if (syncBtn) syncBtn.disabled = Boolean(s?.running);
      if (stopBtn) {
        stopBtn.disabled = !Boolean(s?.running);
        stopBtn.textContent = s?.stop_requested ? "Stopping..." : "Stop Sync";
      }
      panel.classList.toggle("active", Boolean(active));
      if (!active) return;
      const total = Number(s.total || 0);
      const processed = Number(s.processed || 0);
      const width = total ? Math.max(3, Math.min(100, Math.round((processed / total) * 100))) : (s.running ? 8 : 100);
      const sourcePart = s.total_sources
        ? `${fmt(s.active_sources || 0)} active / ${fmt(s.completed_sources || 0)} complete`
        : (s.source_label ? `${s.source_label}` : "sources");
      const sourceIndex = s.total_sources ? `${fmt(s.completed_sources || 0)}/${fmt(s.total_sources)} complete` : "";
      const activeSources = s.running ? `, ${fmt(s.active_sources || 0)} active, max ${fmt(s.max_parallel_sources || 1)}` : "";
      document.getElementById("syncPhase").textContent = `${s.running ? "Syncing" : "Sync"} - ${syncPhaseLabel(s.phase)} ${sourceIndex}${activeSources}`;
      document.getElementById("syncCounters").textContent = [
        sourcePart,
        `listed ${fmt(s.listed)}`,
        `changed ${fmt(s.changed)}`,
        `processed ${fmt(processed)}/${fmt(total)}`,
        `copied ${fmt(s.copied)}`,
        `skipped ${fmt(s.skipped)}`,
        `warnings ${fmt(s.warning_count)}`,
        `errors ${fmt(s.error_count)}`,
      ].filter(Boolean).join(" | ");
      document.getElementById("syncBar").style.width = `${width}%`;
      document.getElementById("syncFile").textContent = s.current_file ? `Current: ${s.current_file}` : (s.message || "");
      const sourceRows = Object.values(s.sources || {})
        .sort((a, b) => Number(a.source_index || 0) - Number(b.source_index || 0))
        .map(source => {
          const sourceTotal = Number(source.total || source.listed || 0);
          const sourceProcessed = Number(source.processed || 0);
          const sourceWidth = sourceTotal ? Math.max(3, Math.min(100, Math.round((sourceProcessed / sourceTotal) * 100))) : (source.running ? 8 : (source.done ? 100 : 0));
          const phase = String(source.phase || "queued");
          const sourceId = String(source.source_id || "");
          const hasFallback = Boolean(source.fallback_reason);
          const badgeClass = phase === "source_error" || Number(source.error_count || 0) ? "error" : (hasFallback || Number(source.warning_count || 0) ? "warn" : (source.done ? "ok" : (phase === "queued" ? "unknown" : "warn")));
          const transferMode = source.transfer_mode ? `mode ${source.transfer_mode}` : "";
          const fallbackDetail = source.fallback_reason ? `<br><span class="badge warn">fallback reason</span><br><span class="small">${escapeHtml(source.fallback_reason)}</span>` : "";
          const resyncButton = source.done && sourceId && !s.running
            ? `<button class="ghost-btn source-resync-btn" data-source-id="${escapeHtml(sourceId)}" title="Sync only this source">Resync source</button>`
            : "";
          return `
            <div class="sync-source-row">
              <div><b>${escapeHtml(source.source_label || source.source_id || "source")}</b><br><span class="badge ${badgeClass}">${escapeHtml(syncPhaseLabel(phase))}</span><br><span class="small">${escapeHtml(transferMode)}</span>${fallbackDetail}</div>
              <div>
                <div class="bar-track"><div class="bar-fill" style="width:${sourceWidth}%"></div></div>
                <span class="small">${fmt(sourceProcessed)}/${fmt(sourceTotal)} files, listed ${fmt(source.listed)}, changed ${fmt(source.changed)}, copied ${fmt(source.copied)}, skipped ${fmt(source.skipped)}, warnings ${fmt(source.warning_count)}, errors ${fmt(source.error_count)}</span>
              </div>
              <div class="small">${escapeHtml(source.current_file || source.message || "")}${resyncButton ? `<div style="margin-top:8px">${resyncButton}</div>` : ""}</div>
            </div>`;
        }).join("");
      document.getElementById("syncSourceProgress").innerHTML = sourceRows;
      document.getElementById("syncMessages").innerHTML = (s.messages || []).slice(-5).map(m => `<div>${escapeHtml(m)}</div>`).join("");
    }
    async function loadSyncStatus() {
      const res = await fetch("/api/sync/status");
      const status = await res.json();
      renderSyncStatus(status);
      maybeRefreshSummaryDuringSync(status);
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
      const preserveStatus = Boolean(options.preserveStatus);
      if (!silent) setStatus("Loading...");
      try {
        const res = await fetch("/api/summary");
        current = await res.json();
        render();
        if (!preserveStatus) {
          setStatus(`${silent ? "Auto refreshed" : "Loaded"} ${new Date(current.generated_at).toLocaleString()}`);
        }
      } finally {
        summaryLoading = false;
      }
    }
    function maybeRefreshSummaryDuringSync(status) {
      if (!status || !status.running) {
        lastSyncSummaryRefreshMs = 0;
        lastSyncCopiedForSummary = -1;
        return;
      }
      const copied = Number(status.copied || 0);
      const now = Date.now();
      if (copied <= lastSyncCopiedForSummary) return;
      if (now - lastSyncSummaryRefreshMs < syncSummaryRefreshIntervalMs) return;
      if (summaryLoading) return;
      lastSyncCopiedForSummary = copied;
      lastSyncSummaryRefreshMs = now;
      loadSummary({silent: true, preserveStatus: true}).catch(e => setStatus(`Live count refresh failed: ${e.message}`));
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
    async function syncSources(sourceId = "") {
      const targetSource = String(sourceId || "");
      setStatus(targetSource ? `Syncing ${targetSource}...` : "Syncing sources...");
      startSyncPolling();
      try {
        const maxParallel = Number(document.getElementById("maxParallelSources").value || "10");
        const body = {max_parallel_sources: targetSource ? 1 : maxParallel};
        if (targetSource) body.source_id = targetSource;
        const res = await fetch("/api/sync", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body),
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "sync failed");
        await loadSyncStatus().catch(() => {});
        await loadSummary();
        const copied = (payload.results || []).reduce((a,r) => a + (r.copied || 0), 0);
        const scope = targetSource ? `Source ${targetSource}` : "Sync";
        setStatus(payload.cancelled ? `${scope} stopped. Copied ${copied} changed files before stopping.` : `${scope} complete. Copied ${copied} changed files.`);
      } catch (error) {
        await loadSyncStatus().catch(() => {});
        throw error;
      }
    }
    async function stopSync() {
      setStatus("Stopping sync...");
      const res = await fetch("/api/sync/stop", {method: "POST"});
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "stop sync failed");
      renderSyncStatus(payload);
      startSyncPolling();
      setStatus(payload.running ? "Stop requested. Active copy operation will finish first." : (payload.message || "Sync stopped"));
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
    function csvRatio(numerator, denominator) {
      const den = Number(denominator || 0);
      if (!den) return "";
      return (Number(numerator || 0) / den).toFixed(4);
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
        "query_to_response_rate","response_to_ir_rate","query_to_ir_rate",
        "overall_score","assets","screenshots","total_bytes","file_count","missing_core_files",
        "content_coverage_avg","intent_score_avg","section_heading_coverage_avg","table_cell_coverage_avg","action_coverage_avg","image_presence_avg","icon_presence_avg","markdown_leakage_rate_avg","component_count_avg",
        "integrity_issues","integrity_parse_errors","integrity_duplicate_ids","integrity_missing_ids","integrity_orphan_links",
        "duplicate_query_rows","duplicate_response_rows","duplicate_ir_rows","duplicate_query_rate","duplicate_response_rate","duplicate_ir_rate",
        "log_files","log_issues","log_latest_updated_at",
        "response_asset_records","missing_response_assets","ir_media_components","ir_local_media_missing","ir_remote_media_refs",
        "record_sample_count",
        "paired_records","estimated_ready_score70","estimated_ready_score80","sample_ready_score70","sample_strict_valid","estimated_ready70_rate","sample_strict_valid_rate",
        "stage2_tokens","stage2_avg_latency_ms","stage2_cost_usd","stage3_tokens","stage3_avg_latency_ms","stage3_cost_usd",
        "updated_at","response_model","ir_model","query_prompt_versions","response_prompt_versions","ir_prompt_versions","ir_versions","path",
      ];
      const rows = runs.map(r => {
        const readiness = r.training_readiness || {};
        return [
          r.source_id,
          r.source_label,
          r.run_id,
          r.queries,
          r.responses,
          r.genui,
          r.response_backlog,
          r.ir_backlog,
          csvRatio(r.responses, r.queries),
          csvRatio(r.genui, r.responses),
          csvRatio(r.genui, r.queries),
          r.display_score ?? r.overall_score,
          r.assets,
          r.screenshots,
          r.total_bytes,
          r.file_count,
          (r.missing_core_files || []).join("; "),
          r.metric_avgs?.content_coverage ?? "",
          r.metric_avgs?.intent_score ?? "",
          r.metric_avgs?.section_heading_coverage ?? "",
          r.metric_avgs?.table_cell_coverage ?? "",
          r.metric_avgs?.action_coverage ?? "",
          r.metric_avgs?.image_presence ?? "",
          r.metric_avgs?.icon_presence ?? "",
          r.metric_avgs?.markdown_leakage_rate ?? "",
          r.metric_avgs?.component_count ?? "",
          r.data_integrity?.total_issues || 0,
          r.data_integrity?.parse_error_count || 0,
          r.data_integrity?.duplicate_id_count || 0,
          r.data_integrity?.missing_id_count || 0,
          r.data_integrity?.orphan_link_count || 0,
          r.content_duplicates?.queries?.duplicate_rows || 0,
          r.content_duplicates?.responses?.duplicate_rows || 0,
          r.content_duplicates?.ir_payloads?.duplicate_rows || 0,
          r.content_duplicates?.queries?.duplicate_rate ?? "",
          r.content_duplicates?.responses?.duplicate_rate ?? "",
          r.content_duplicates?.ir_payloads?.duplicate_rate ?? "",
          r.run_logs?.file_count || 0,
          r.run_logs?.issue_count || 0,
          r.run_logs?.latest_updated_at || "",
          r.media_health?.response_asset_records || 0,
          r.media_health?.response_asset_files_missing || 0,
          r.media_health?.ir_media_components || 0,
          r.media_health?.ir_local_media_missing || 0,
          r.media_health?.ir_remote_media_refs || 0,
          (r.record_samples || []).length,
          readiness.paired_records || 0,
          readiness.estimated_ready_score70 || 0,
          readiness.estimated_ready_score80 || 0,
          readiness.ready_score70 || 0,
          readiness.strict_valid || 0,
          csvRatio(readiness.estimated_ready_score70, readiness.paired_records),
          csvRatio(readiness.strict_valid, readiness.sampled),
          r.response_usage?.total_tokens || 0,
          r.response_usage?.avg_latency_ms ?? "",
          r.response_usage?.cost_usd ?? "",
          r.ir_usage?.total_tokens || 0,
          r.ir_usage?.avg_latency_ms ?? "",
          r.ir_usage?.cost_usd ?? "",
          r.updated_at,
          dominantModel(r.response_models),
          dominantModel(r.ir_models),
          Object.keys(r.query_prompt_versions || {}).join("; "),
          Object.keys(r.response_prompt_versions || {}).join("; "),
          Object.keys(r.ir_prompt_versions || {}).join("; "),
          Object.keys(r.ir_versions || {}).join("; "),
          r.path,
        ];
      });
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
        ["Integrity issue runs", t.integrity_issue_runs],
        ["Integrity issues", t.integrity_issues],
        ["Log issue runs", t.log_issue_runs],
        ["Log issues", t.log_issues],
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
      const parallelInput = document.getElementById("maxParallelSources");
      if (parallelInput && document.activeElement !== parallelInput && !parallelInput.dataset.userEdited) {
        parallelInput.value = String(config.max_parallel_sources || 10);
      }
      const sources = (config.sources || []).map(source => {
        const meta = [
          `${source.type}`,
          source.enabled ? "enabled" : "disabled",
          source.transfer_mode ? `transfer ${source.transfer_mode}` : "",
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
          <div class="detail-box"><b>${fmt(config.max_parallel_sources || 10)}</b><br><span class="small">default parallel sources</span></div>
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
      const totalErrors = Number(lastSync.error_count || 0);
      const totalWarnings = Number(lastSync.warning_count || 0);
      const summaryBadge = totalErrors
        ? `<span class="badge error">completed with ${fmt(totalErrors)} error(s)</span>`
        : (totalWarnings ? `<span class="badge warn">completed with ${fmt(totalWarnings)} warning(s)</span>` : `<span class="badge ok">completed</span>`);
      const rows = (lastSync.results || []).map(result => {
        const errors = result.errors || [];
        const warnings = result.warnings || [];
        const fallbackReason = result.fallback_reason || "";
        const status = (result.error_count || 0) ? "error" : (fallbackReason || (result.warning_count || 0) ? "warn" : "ok");
        const issueLines = [
          fallbackReason ? `<span class="badge warn">fallback</span><br><span class="small">${escapeHtml(fallbackReason)}</span>` : "",
          warnings.length ? `<br><span class="small">${escapeHtml(warnings[0])}</span>` : "",
          errors.length ? `<br><span class="small">${escapeHtml(errors[0])}</span>` : "",
        ].filter(Boolean).join("");
        return `
          <tr>
            <td><b>${escapeHtml(result.source_label || result.source_id)}</b><br><span class="badge ${status}">${status}</span><br><span class="small">mode ${escapeHtml(result.transfer_mode || "n/a")}</span></td>
            <td>listed ${fmt(result.listed)}<br>changed ${fmt(result.changed)}<br>copied ${fmt(result.copied)}<br>skipped ${fmt(result.skipped)}</td>
            <td>warnings ${fmt(result.warning_count || 0)}<br>errors ${fmt(result.error_count || 0)}${issueLines || "<br><span class='small'>no warnings/errors</span>"}</td>
            <td><span class="small">${escapeHtml(result.mirror_path || "")}</span></td>
          </tr>`;
      }).join("");
      document.getElementById("lastSyncResults").innerHTML = `
        <div class="small">synced at: ${lastSync.synced_at ? new Date(lastSync.synced_at).toLocaleString() : "unknown"} ${summaryBadge}</div>
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
    function actionBadgeClass(severity) {
      if (severity >= 80) return "error";
      if (severity >= 50) return "warn";
      return "ok";
    }
    function actionSeverityText(severity) {
      if (severity >= 80) return "high";
      if (severity >= 50) return "medium";
      return "watch";
    }
    function addActionItem(items, item) {
      items.push({
        severity: Number(item.severity || 0),
        category: item.category || "Action",
        title: item.title || "Review item",
        detail: item.detail || "",
        run: item.run || null,
        source: item.source || null,
      });
    }
    function buildActionItems(runs, sources, lastSync) {
      const items = [];
      for (const result of lastSync?.results || []) {
        const errors = result.errors || [];
        if ((result.error_count || 0) > 0) {
          addActionItem(items, {
            severity: 95,
            category: "Sync",
            title: `${result.source_label || result.source_id} sync has ${fmt(result.error_count)} errors`,
            detail: errors[0] || "Inspect source connection and mirror path.",
            source: result.source_label || result.source_id,
          });
        }
      }
      for (const source of sources) {
        const backlog = (source.response_backlog || 0) + (source.ir_backlog || 0);
        if (backlog) {
          addActionItem(items, {
            severity: Math.min(92, 45 + Math.log10(backlog + 1) * 12),
            category: "Backlog",
            title: `${source.source_label} has ${fmt(backlog)} pending records`,
            detail: `missing responses ${fmt(source.response_backlog)} | missing IR ${fmt(source.ir_backlog)}`,
            source: source.source_label,
          });
        }
        const hours = ageHours(source.latest_run_updated_at);
        if (hours == null || hours > 24 * 7) {
          addActionItem(items, {
            severity: hours == null ? 72 : Math.min(90, hours > 24 * 30 ? 88 : 62),
            category: "Freshness",
            title: `${source.source_label} has stale or unknown updates`,
            detail: `latest matching run update: ${ageText(source.latest_run_updated_at)}`,
            source: source.source_label,
          });
        }
      }
      for (const run of runs) {
        const q = run.quality_summary || {};
        const integrity = integrityIssueCount(run);
        const logIssues = logIssueCount(run);
        const backlog = runBacklogTotal(run);
        const artifactIssues = artifactIssueLabels(run);
        const qualityCritical = (q.strict_schema_fail || 0) + (q.gen_errors || 0) + (q.fallback_generated || 0);
        if ((run.missing_core_files || []).length) {
          addActionItem(items, {
            severity: 98,
            category: "Files",
            title: `${run.run_id} is missing core files`,
            detail: (run.missing_core_files || []).join(", "),
            run,
            source: run.source_label,
          });
        }
        if (integrity) {
          addActionItem(items, {
            severity: Math.min(96, 55 + Math.log10(integrity + 1) * 12),
            category: "Integrity",
            title: `${run.run_id} has ${fmt(integrity)} data integrity signals`,
            detail: integrityIssueLabels(run.data_integrity || {}).join(" | ") || "Inspect ID/link integrity.",
            run,
            source: run.source_label,
          });
        }
        const contentDuplicates = contentDuplicateTotal(run);
        if (contentDuplicates) {
          addActionItem(items, {
            severity: Math.min(84, 42 + Math.log10(contentDuplicates + 1) * 12),
            category: "Duplicates",
            title: `${run.run_id} has ${fmt(contentDuplicates)} sampled duplicate content rows`,
            detail: "Inspect duplicate query, response, and IR payload examples.",
            run,
            source: run.source_label,
          });
        }
        if (logIssues) {
          const firstIssue = ((run.run_logs || {}).recent_issues || [])[0];
          addActionItem(items, {
            severity: Math.min(94, 58 + Math.log10(logIssues + 1) * 12),
            category: "Logs",
            title: `${run.run_id} has ${fmt(logIssues)} recent log issue lines`,
            detail: firstIssue ? `${firstIssue.file}: ${firstIssue.text}` : "Inspect run logs.",
            run,
            source: run.source_label,
          });
        }
        if (qualityCritical) {
          addActionItem(items, {
            severity: Math.min(94, 60 + Math.log10(qualityCritical + 1) * 12),
            category: "Quality",
            title: `${run.run_id} has ${fmt(qualityCritical)} critical sampled IR issues`,
            detail: `strict fail ${fmt(q.strict_schema_fail)} | gen errors ${fmt(q.gen_errors)} | fallback ${fmt(q.fallback_generated)}`,
            run,
            source: run.source_label,
          });
        } else if ((q.low_score || 0) || (q.markdown_leakage || 0) || (q.sparse_ir || 0)) {
          const issueCount = (q.low_score || 0) + (q.markdown_leakage || 0) + (q.sparse_ir || 0);
          addActionItem(items, {
            severity: Math.min(72, 40 + Math.log10(issueCount + 1) * 10),
            category: "Quality",
            title: `${run.run_id} has sampled quality warnings`,
            detail: `low score ${fmt(q.low_score)} | markdown ${fmt(q.markdown_leakage)} | sparse ${fmt(q.sparse_ir)}`,
            run,
            source: run.source_label,
          });
        }
        if (backlog) {
          addActionItem(items, {
            severity: Math.min(82, 38 + Math.log10(backlog + 1) * 10),
            category: "Backlog",
            title: `${run.run_id} has incomplete pipeline stages`,
            detail: `missing responses ${fmt(run.response_backlog)} | missing IR ${fmt(run.ir_backlog)}`,
            run,
            source: run.source_label,
          });
        }
        if (artifactIssues.length) {
          addActionItem(items, {
            severity: 48 + artifactIssues.length * 5,
            category: "Artifacts",
            title: `${run.run_id} has artifact gaps`,
            detail: artifactIssues.join(" | "),
            run,
            source: run.source_label,
          });
        }
        const mediaIssues = mediaIssueCount(run);
        if (mediaIssues) {
          addActionItem(items, {
            severity: Math.min(78, 44 + Math.log10(mediaIssues + 1) * 12),
            category: "Media",
            title: `${run.run_id} has unresolved media references`,
            detail: `missing local media/asset refs ${fmt(mediaIssues)}`,
            run,
            source: run.source_label,
          });
        }
      }
      return items
        .sort((a, b) => (b.severity - a.severity) || String(a.source || "").localeCompare(String(b.source || "")) || a.title.localeCompare(b.title))
        .slice(0, 16);
    }
    function renderActionItems(runs, sources, lastSync) {
      const target = document.getElementById("actionItems");
      if (!runs.length && !sources.length) {
        target.innerHTML = "<span class='small'>No source or run data found.</span>";
        return;
      }
      const items = buildActionItems(runs, sources, lastSync);
      if (!items.length) {
        target.innerHTML = "<span class='small'>No prioritized action items for the current filters.</span>";
        return;
      }
      const rows = items.map(item => `
        <div class="warning-row">
          <span>
            <span class="badge ${actionBadgeClass(item.severity)}">${actionSeverityText(item.severity)}</span>
            <span class="badge">${escapeHtml(item.category)}</span>
            <b>${escapeHtml(item.title)}</b><br>
            <span class="small">${escapeHtml(item.detail)}${item.source ? ` | ${escapeHtml(item.source)}` : ""}</span>
          </span>
          ${item.run ? `<button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(item.run))}')">Details</button>` : `<span class="small">${Math.round(item.severity)}</span>`}
        </div>`).join("");
      target.innerHTML = `
        <div class="warning-list">${rows}</div>
        <div class="small">Action items are derived from the filtered run set and combine sync errors, stale sources, backlog, integrity issues, sampled generation quality, media references, and artifact gaps.</div>`;
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
    function integrityIssueCount(r) {
      return Number((r.data_integrity || {}).total_issues || 0);
    }
    function logIssueCount(r) {
      return Number((r.run_logs || {}).issue_count || 0);
    }
    function mediaIssueCount(r) {
      const m = r.media_health || {};
      return Number(m.ir_local_media_missing || 0) + Number(m.response_asset_files_missing || 0);
    }
    function integrityIssueLabels(summary) {
      const s = summary || {};
      const labels = [];
      if (s.parse_error_count) labels.push(`parse errors ${fmt(s.parse_error_count)}`);
      if (s.duplicate_id_count) labels.push(`duplicate IDs ${fmt(s.duplicate_id_count)}`);
      if (s.missing_id_count) labels.push(`missing IDs ${fmt(s.missing_id_count)}`);
      if (s.orphan_link_count) labels.push(`broken links ${fmt(s.orphan_link_count)}`);
      if (s.count_warning_count) labels.push(`count warnings ${fmt(s.count_warning_count)}`);
      return labels;
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
      const hay = JSON.stringify([
        r.source_label,
        r.run_id,
        r.query_models,
        r.response_models,
        r.ir_models,
        r.query_prompt_versions,
        r.response_prompt_versions,
        r.ir_prompt_versions,
        r.ir_versions,
        r.intents,
      ]).toLowerCase();
      if (f.sourceId && r.source_id !== f.sourceId) return false;
      if (f.irVersion && !(r.ir_version_stats || {})[f.irVersion]) return false;
      if (f.text && !hay.includes(f.text)) return false;
      if (dateFilterActive(f) && !filteredRunDays(r, f).length) return false;
      if (f.issue) {
        const counts = dayFilteredCounts(r, f);
        const metrics = r.metric_avgs || {};
        if (f.issue === "backlog" && !Math.max((counts.queries || 0) - (counts.responses || 0), 0) && !Math.max((counts.responses || 0) - (counts.genui || 0), 0)) return false;
        if (f.issue === "quality" && !qualityIssueCount(r)) return false;
        if (f.issue === "integrity" && !integrityIssueCount(r)) return false;
        if (f.issue === "duplicates" && !contentDuplicateTotal(r)) return false;
        if (f.issue === "logs" && !logIssueCount(r)) return false;
        if (f.issue === "metric_risk" && !metricRiskForRun(r).length) return false;
        if (f.issue === "media_refs" && !mediaIssueCount(r)) return false;
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
        if (sortBy === "integrity_desc") return integrityIssueCount(b) - integrityIssueCount(a) || textValue(a).localeCompare(textValue(b));
        if (sortBy === "duplicates_desc") return contentDuplicateTotal(b) - contentDuplicateTotal(a) || textValue(a).localeCompare(textValue(b));
        if (sortBy === "logs_desc") return logIssueCount(b) - logIssueCount(a) || textValue(a).localeCompare(textValue(b));
        if (sortBy === "metric_risk_desc") return metricRiskScore(b) - metricRiskScore(a) || textValue(a).localeCompare(textValue(b));
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
        integrity_issue_runs: runs.reduce((a,r) => a + (integrityIssueCount(r) ? 1 : 0), 0),
        integrity_issues: runs.reduce((a,r) => a + integrityIssueCount(r), 0),
        log_issue_runs: runs.reduce((a,r) => a + (logIssueCount(r) ? 1 : 0), 0),
        log_issues: runs.reduce((a,r) => a + logIssueCount(r), 0),
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
    function renderObjectPairs(obj, emptyLabel = "none") {
      const entries = Object.entries(obj || {}).filter(([, value]) => value != null && value !== "");
      if (!entries.length) return `<span class="small">${emptyLabel}</span>`;
      return entries.map(([key, value]) => `<div class="small"><b>${escapeHtml(key)}</b>: ${escapeHtml(Array.isArray(value) ? value.join(", ") : value)}</div>`).join("");
    }
    function renderManifestSummary(manifest) {
      if (!manifest || !manifest.present) {
        if (manifest?.parse_error) {
          return `<span class="badge error">manifest parse error</span><div class="small">${escapeHtml(manifest.parse_error)}<br>${escapeHtml(manifest.path || "")}</div>`;
        }
        return "<span class='small'>No run_manifest.json found.</span>";
      }
      const repo = manifest.repo || {};
      const model = manifest.model || {};
      const config = manifest.config || {};
      const command = manifest.command || {};
      const argv = (command.argv || []).join(" ");
      const settings = Object.entries(manifest.settings || {}).map(([name, values]) => `
        <div class="detail-box">
          <b>${escapeHtml(name)}</b>
          ${renderObjectPairs(values)}
        </div>`).join("");
      return `
        <div class="detail-grid">
          <div class="detail-box"><b>${escapeHtml(manifest.stage ?? "unknown")}</b><br><span class="small">manifest stage</span></div>
          <div class="detail-box"><b>${escapeHtml(manifest.generated_at || "unknown")}</b><br><span class="small">generated at</span></div>
          <div class="detail-box"><b>${escapeHtml([model.provider, model.model].filter(Boolean).join("/") || model.name || "unknown")}</b><br><span class="small">manifest model</span></div>
          <div class="detail-box"><b>${escapeHtml(repo.git_branch || "unknown")}</b><br><span class="small">git branch</span></div>
          <div class="detail-box"><b>${escapeHtml((repo.git_commit || "").slice(0, 12) || "unknown")}</b><br><span class="small">git commit ${repo.git_dirty ? "(dirty)" : ""}</span></div>
          <div class="detail-box"><b>${escapeHtml(config.combined_sha256 ? config.combined_sha256.slice(0, 12) : "unknown")}</b><br><span class="small">config hash | ${fmt(config.file_count || 0)} files</span></div>
        </div>
        <div class="detail-grid">
          <div class="detail-box"><b>Command</b><br><span class="small">${escapeHtml(argv || "n/a")}${command.argv_truncated ? " ..." : ""}</span></div>
          <div class="detail-box"><b>Command cwd</b><br><span class="small">${escapeHtml(command.cwd || "n/a")}</span></div>
          <div class="detail-box"><b>Config files</b><br><span class="small">${escapeHtml((config.files || []).join(" | ") || "n/a")}</span></div>
        </div>
        ${settings ? `<div class="detail-grid">${settings}</div>` : ""}
      `;
    }
    function renderIntegrityDetails(summary) {
      const s = summary || {};
      const duplicateRows = Object.entries(s.duplicate_ids || {}).map(([field, info]) =>
        `<div class="warning-row"><span>duplicate ${escapeHtml(field)} ${escapeHtml((info.examples || []).join(", "))}</span><b>${fmt(info.count || 0)}</b></div>`
      ).join("");
      const missingRows = Object.entries(s.missing_ids || {}).map(([field, count]) =>
        `<div class="warning-row"><span>missing ${escapeHtml(field)}</span><b>${fmt(count)}</b></div>`
      ).join("");
      const orphanRows = Object.entries(s.orphan_counts || {}).filter(([, count]) => count).map(([field, count]) => {
        const examples = (s.orphan_examples || {})[field] || [];
        return `<div class="warning-row"><span>${escapeHtml(field)} ${escapeHtml(examples.join(", "))}</span><b>${fmt(count)}</b></div>`;
      }).join("");
      const parseRows = Object.entries(s.files || {}).filter(([, file]) => file.parse_errors).map(([name, file]) =>
        `<div class="warning-row"><span>${escapeHtml(name)} parse errors at lines ${escapeHtml((file.parse_error_lines || []).join(", ") || "n/a")}</span><b>${fmt(file.parse_errors)}</b></div>`
      ).join("");
      const warningRows = (s.count_warnings || []).map(w =>
        `<div class="warning-row"><span>${escapeHtml(w.message || w.type)}</span><b>1</b></div>`
      ).join("");
      const rows = [parseRows, duplicateRows, missingRows, orphanRows, warningRows].filter(Boolean).join("");
      return `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(s.total_issues || 0)}</b><br><span class="small">integrity issues</span></div>
          <div class="detail-box"><b>${fmt(s.parse_error_count || 0)}</b><br><span class="small">JSONL parse errors</span></div>
          <div class="detail-box"><b>${fmt(s.duplicate_id_count || 0)}</b><br><span class="small">duplicate IDs</span></div>
          <div class="detail-box"><b>${fmt(s.missing_id_count || 0)}</b><br><span class="small">missing IDs</span></div>
          <div class="detail-box"><b>${fmt(s.orphan_link_count || 0)}</b><br><span class="small">broken stage links</span></div>
          <div class="detail-box"><b>${fmt(s.count_warning_count || 0)}</b><br><span class="small">count warnings</span></div>
        </div>
        <div class="warning-list">${rows || "<span class='small'>No data integrity issues found.</span>"}</div>`;
    }
    function contentDuplicateTotal(run) {
      const d = run.content_duplicates || {};
      return Number(d.total_duplicate_rows || 0);
    }
    function duplicateStageSummary(summary, key) {
      const data = (summary || {})[key] || {};
      return {
        sampled: Number(data.sampled || 0),
        unique: Number(data.unique || 0),
        duplicateRows: Number(data.duplicate_rows || 0),
        duplicateGroups: Number(data.duplicate_groups || 0),
        duplicateRate: data.duplicate_rate == null ? null : Number(data.duplicate_rate),
        examples: data.examples || [],
      };
    }
    function renderDuplicateExamples(summary, key, label) {
      const data = duplicateStageSummary(summary, key);
      const rows = data.examples.map(example => `
        <div class="warning-row">
          <span><b>${escapeHtml(label)}</b><br><span class="small">${escapeHtml(example.preview || "")}</span></span>
          <span class="badge warn">${fmt(example.count || 0)}x</span>
        </div>`).join("");
      return `
        <div class="detail-box">
          <b>${escapeHtml(label)}</b><br>
          <span class="small">${fmt(data.duplicateRows)} duplicate rows | ${fmt(data.duplicateGroups)} groups | ${pct(data.duplicateRate)} rate | ${fmt(data.unique)} unique / ${fmt(data.sampled)} sampled</span>
        </div>
        ${rows}`;
    }
    function renderContentDuplicateDetails(summary) {
      const s = summary || {};
      const total = Number(s.total_duplicate_rows || 0);
      return `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(total)}</b><br><span class="small">sampled duplicate content rows</span></div>
          <div class="detail-box"><b>${fmt(duplicateStageSummary(s, "queries").duplicateRows)}</b><br><span class="small">duplicate query rows</span></div>
          <div class="detail-box"><b>${fmt(duplicateStageSummary(s, "responses").duplicateRows)}</b><br><span class="small">duplicate response rows</span></div>
          <div class="detail-box"><b>${fmt(duplicateStageSummary(s, "ir_payloads").duplicateRows)}</b><br><span class="small">duplicate IR rows</span></div>
        </div>
        <div class="warning-list">
          ${renderDuplicateExamples(s, "queries", "Queries")}
          ${renderDuplicateExamples(s, "responses", "Responses")}
          ${renderDuplicateExamples(s, "ir_payloads", "IR Payloads")}
        </div>`;
    }
    function renderRunLogDetails(logs) {
      const l = logs || {};
      if (!l.present) return "<span class='small'>No run log files found.</span>";
      const fileRows = (l.files || []).map(file => `
        <div class="artifact-row">
          <b>${escapeHtml(file.name)}</b>
          <span class="small">${escapeHtml(file.path)}<br>updated ${file.updated_at ? new Date(file.updated_at).toLocaleString() : "unknown"}</span>
          <button class="mini-btn ghost-btn" onclick="copyText('${encodeURIComponent(file.path)}')">${bytesText(file.size)}</button>
        </div>`).join("");
      const issueRows = (l.recent_issues || []).map(issue => `
        <div class="warning-row">
          <span><b>${escapeHtml(issue.file)}:${fmt(issue.line)}</b><br><span class="small">${escapeHtml(issue.text)}</span></span>
          <span class="badge error">issue</span>
        </div>`).join("");
      const progressRows = (l.recent_progress || []).map(item => `
        <div class="warning-row">
          <span><b>${escapeHtml(item.file)}:${fmt(item.line)}</b><br><span class="small">${escapeHtml(item.text)}</span></span>
          <span class="badge ok">progress</span>
        </div>`).join("");
      return `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(l.file_count || 0)}</b><br><span class="small">log files</span></div>
          <div class="detail-box"><b>${fmt(l.issue_count || 0)}</b><br><span class="small">issue lines in tail</span></div>
          <div class="detail-box"><b>${fmt(l.progress_count || 0)}</b><br><span class="small">progress lines in tail</span></div>
          <div class="detail-box"><b>${l.latest_updated_at ? ageText(l.latest_updated_at) : "unknown"}</b><br><span class="small">latest log update</span></div>
        </div>
        <div class="artifact-list">${fileRows || "<span class='small'>No log file metadata.</span>"}</div>
        <h2>Recent Log Issues</h2>
        <div class="warning-list">${issueRows || "<span class='small'>No issue lines found in log tails.</span>"}</div>
        <h2>Recent Log Progress</h2>
        <div class="warning-list">${progressRows || "<span class='small'>No progress lines found in log tails.</span>"}</div>`;
    }
    function sortedObjectEntries(obj, limit = 8) {
      return Object.entries(obj || {})
        .map(([label, count]) => [String(label), Number(count || 0)])
        .filter(([, count]) => count > 0)
        .sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]))
        .slice(0, limit);
    }
    function renderRunIrStructureDetails(structure) {
      const s = structure || {};
      if (!s.sampled) return "<span class='small'>No sampled IR structure data found.</span>";
      const uncommonRows = sortedObjectEntries(s.uncommon_component_types || {}, 8).map(([type, count]) => `
        <div class="warning-row">
          <span>${escapeHtml(type)}</span>
          <b>${fmt(count)}</b>
        </div>`).join("");
      return `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(s.flat_spec_rows || 0)}</b><br><span class="small">sampled flat-spec rows</span></div>
          <div class="detail-box"><b>${Number(s.avg_components || 0).toFixed(1)}</b><br><span class="small">avg components</span></div>
          <div class="detail-box"><b>${fmt(s.component_count || 0)}</b><br><span class="small">sampled components</span></div>
          <div class="detail-box"><b>${fmt(s.table_count || 0)}</b><br><span class="small">tables</span></div>
          <div class="detail-box"><b>${fmt(s.button_count || 0)}</b><br><span class="small">buttons</span></div>
          <div class="detail-box"><b>${fmt(s.image_count || 0)}</b><br><span class="small">images</span></div>
          <div class="detail-box"><b>${fmt(s.icon_count || 0)}</b><br><span class="small">icons</span></div>
          <div class="detail-box"><b>${Number(s.avg_table_rows || 0).toFixed(1)}</b><br><span class="small">avg table rows</span></div>
          <div class="detail-box"><b>${Number(s.avg_table_columns || 0).toFixed(1)}</b><br><span class="small">avg table columns</span></div>
        </div>
        <div class="dist-grid">
          ${renderStructureCountBlock("Component Types", sortedObjectEntries(s.component_types || {}, 10))}
          ${renderStructureCountBlock("Table Domains", sortedObjectEntries(s.table_domains || {}, 8))}
          ${renderStructureCountBlock("Table Presentations", sortedObjectEntries(s.table_presentations || {}, 8))}
          ${renderStructureCountBlock("Action Types", sortedObjectEntries(s.action_types || {}, 8))}
        </div>
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(s.chart_count || 0)}</b><br><span class="small">charts</span></div>
          <div class="detail-box"><b>${fmt(s.formula_count || 0)}</b><br><span class="small">formula elements</span></div>
          <div class="detail-box"><b>${fmt(s.code_count || 0)}</b><br><span class="small">code/console blocks</span></div>
          <div class="detail-box"><b>${fmt(s.email_preview_count || 0)}</b><br><span class="small">email previews</span></div>
        </div>
        <h2>Uncommon Component Types</h2>
        <div class="warning-list">${uncommonRows || "<span class='small'>No uncommon component types in sampled IR.</span>"}</div>`;
    }
    function renderRunMediaHealthDetails(media) {
      const m = media || {};
      if (!m.sampled_response_rows && !m.sampled_ir_rows) {
        return "<span class='small'>No sampled media health data found.</span>";
      }
      return `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(m.response_asset_records || 0)}</b><br><span class="small">response asset records</span></div>
          <div class="detail-box"><b>${fmt(m.response_asset_files_present || 0)}</b><br><span class="small">asset files present</span></div>
          <div class="detail-box"><b>${fmt(m.response_asset_files_missing || 0)}</b><br><span class="small">missing response assets</span></div>
          <div class="detail-box"><b>${bytesText(m.response_asset_bytes || 0)}</b><br><span class="small">declared asset bytes</span></div>
          <div class="detail-box"><b>${fmt(m.ir_media_components || 0)}</b><br><span class="small">IR media components</span></div>
          <div class="detail-box"><b>${fmt(m.ir_image_components || 0)}</b><br><span class="small">IR images</span></div>
          <div class="detail-box"><b>${fmt(m.ir_icon_components || 0)}</b><br><span class="small">IR icons</span></div>
          <div class="detail-box"><b>${fmt(m.ir_local_media_missing || 0)}</b><br><span class="small">missing local IR refs</span></div>
          <div class="detail-box"><b>${fmt(m.ir_remote_media_refs || 0)}</b><br><span class="small">remote refs left in IR</span></div>
        </div>
        <div class="dist-grid">
          ${renderStructureCountBlock("Response Asset Hosts", sortedObjectEntries(m.response_asset_hosts || {}, 8))}
          ${renderStructureCountBlock("IR Remote Hosts", sortedObjectEntries(m.ir_remote_hosts || {}, 8))}
          ${renderStructureCountBlock("Asset Extensions", sortedObjectEntries(m.asset_file_extensions || {}, 8))}
        </div>
        <h2>Missing Local Media Samples</h2>
        <div class="warning-list">${mediaSampleRows(m.missing_local_samples || [], "No missing local media references found in sampled rows.")}</div>
        <h2>Remote IR Media Samples</h2>
        <div class="warning-list">${mediaSampleRows(m.remote_ref_samples || [], "No remote media references found in sampled IR rows.")}</div>`;
    }
    function renderComponentTypeBadges(componentTypes) {
      const entries = Object.entries(componentTypes || {}).slice(0, 8);
      return entries.map(([type, count]) => `<span class="badge">${escapeHtml(type)} ${fmt(count)}</span>`).join(" ") || "<span class='small'>no components</span>";
    }
    function renderRunRecordSamples(samples) {
      if (!(samples || []).length) {
        return "<span class='small'>No sampled records found for this run.</span>";
      }
      const rows = samples.map(sample => {
        const ir = sample.ir_summary || {};
        const issueBadges = (sample.issues || []).map(issue => `<span class="badge error">${escapeHtml(issue)}</span>`).join(" ") || "<span class='badge ok'>sample</span>";
        const validation = (sample.validation_errors || []).map(escapeHtml).join("<br>");
        return `
          <details class="sample-card">
            <summary>
              ${escapeHtml(sample.ui_id || `row ${sample.row}`)}
              <span class="score ${scoreClass(sample.score)}">${scoreText(sample.score)}</span>
              <span class="small">${escapeHtml(sample.intent || "")}</span>
            </summary>
            <div class="pill-row">
              ${issueBadges}
              <span class="badge">${escapeHtml(sample.prompt_version || "unknown prompt")}</span>
              <span class="badge">${escapeHtml(sample.model || "unknown model")}</span>
              <span class="badge">${escapeHtml(ir.format || "unknown IR")}</span>
            </div>
            <div class="detail-grid">
              <div class="detail-box"><b>${fmt(ir.component_count || 0)}</b><br><span class="small">components</span></div>
              <div class="detail-box"><b>${fmt(ir.table_count || 0)}</b><br><span class="small">tables</span></div>
              <div class="detail-box"><b>${fmt(ir.image_count || 0)}</b><br><span class="small">images</span></div>
              <div class="detail-box"><b>${fmt(ir.icon_count || 0)}</b><br><span class="small">icons</span></div>
              <div class="detail-box"><b>${fmt(ir.button_count || 0)}</b><br><span class="small">buttons</span></div>
            </div>
            <div class="small"><b>IDs:</b> ${escapeHtml(sample.query_id || "")} | ${escapeHtml(sample.response_id || "")}</div>
            <div class="small"><b>Component types:</b> ${renderComponentTypeBadges(ir.component_types || {})}</div>
            <h2>Query</h2>
            <div class="sample-preview">${escapeHtml(sample.query_text || "n/a")}</div>
            <h2>Response Preview</h2>
            <div class="sample-preview">${escapeHtml(sample.response_preview || "n/a")}</div>
            ${validation ? `<h2>Validation</h2><div class="sample-preview">${validation}</div>` : ""}
          </details>`;
      }).join("");
      return `<div class="sample-list">${rows}</div>`;
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
      const manifest = selected.manifest_summary || {};
      const integrity = selected.data_integrity || {};
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
        <h2>Prompt Provenance</h2>
        ${renderPromptVersionDetails(selected)}
        <div class="detail-grid">
          <div class="detail-box"><b>${metricPct(m.content_coverage)}</b><br><span class="small">content coverage</span></div>
          <div class="detail-box"><b>${metricPct(m.intent_score)}</b><br><span class="small">intent score</span></div>
          <div class="detail-box"><b>${metricPct(m.section_heading_coverage)}</b><br><span class="small">heading coverage</span></div>
          <div class="detail-box"><b>${metricPct(m.table_cell_coverage)}</b><br><span class="small">table coverage</span></div>
          <div class="detail-box"><b>${metricPct(m.action_coverage)}</b><br><span class="small">action coverage</span></div>
          <div class="detail-box"><b>${metricPct(m.image_presence)}</b><br><span class="small">image presence</span></div>
        </div>
        <h2>Metric Risk</h2>
        <div class="warning-list">${metricRiskBadges(selected, 8) || "<span class='small'>No sampled metric risks for this run.</span>"}</div>
        <h2>Validation Warnings</h2>
        <div class="warning-list">${warnings || "<span class='small'>No sampled validation warnings.</span>"}</div>
        <h2>Training Readiness</h2>
        ${renderTrainingReadinessDetails(selected.training_readiness || {})}
        <h2>IR Structure</h2>
        ${renderRunIrStructureDetails(selected.ir_structure || {})}
        <h2>Media & Asset Health</h2>
        ${renderRunMediaHealthDetails(selected.media_health || {})}
        <h2>Sample Records</h2>
        ${renderRunRecordSamples(selected.record_samples || [])}
        <h2>Data Integrity</h2>
        ${renderIntegrityDetails(integrity)}
        <h2>Content Duplicates</h2>
        ${renderContentDuplicateDetails(selected.content_duplicates || {})}
        <h2>Run Logs</h2>
        ${renderRunLogDetails(selected.run_logs || {})}
        <h2>Run Provenance</h2>
        ${renderManifestSummary(manifest)}
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
    function ratioOrNull(numerator, denominator) {
      const den = Number(denominator || 0);
      return den ? Number(numerator || 0) / den : null;
    }
    function estimatedFromSample(total, sampleHitCount, sampleCount) {
      const rate = ratioOrNull(sampleHitCount, sampleCount);
      return rate == null ? null : Math.round(Number(total || 0) * rate);
    }
    function renderFunnelStep(step) {
      const width = Math.max(3, Math.min(100, Math.round(Number(step.rate || 0) * 100)));
      return `
        <div class="day-row">
          <div>
            <b>${escapeHtml(step.label)}</b><br>
            <span class="small">${escapeHtml(step.detail || "")}</span>
          </div>
          <div>
            <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
            <div class="day-counts">
              <span>${fmt(step.count == null ? 0 : step.count)} records</span>
              <span>${pct(step.rate)} of queries</span>
              ${step.stageRate == null ? "" : `<span>${escapeHtml(step.stageLabel || "stage")} ${pct(step.stageRate)}</span>`}
            </div>
          </div>
          <div><b>${pct(step.rate)}</b><br><span class="small">cumulative</span></div>
        </div>`;
    }
    function sourceFunnelRows(runs, sources) {
      const bySource = new Map();
      for (const run of runs) {
        if (!bySource.has(run.source_id)) bySource.set(run.source_id, []);
        bySource.get(run.source_id).push(run);
      }
      return sources.map(source => {
        const sourceRuns = bySource.get(source.source_id) || [];
        const readiness = aggregateTrainingReadiness(sourceRuns);
        const ready70Rate = ratioOrNull(readiness.estimated_ready_score70, readiness.paired_records);
        return {
          ...source,
          readiness,
          queryToResponse: ratioOrNull(source.responses, source.queries),
          responseToIr: ratioOrNull(source.genui, source.responses),
          queryToIr: ratioOrNull(source.genui, source.queries),
          ready70Rate,
          total_backlog: (source.response_backlog || 0) + (source.ir_backlog || 0),
        };
      }).sort((a, b) =>
        (b.total_backlog - a.total_backlog)
        || ((a.queryToIr ?? 1) - (b.queryToIr ?? 1))
        || ((a.ready70Rate ?? 1) - (b.ready70Rate ?? 1))
        || String(a.source_label).localeCompare(String(b.source_label))
      );
    }
    function renderCompletionFunnel(runs, sources) {
      const target = document.getElementById("completionFunnel");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const totals = runTotals(runs);
      const readiness = aggregateTrainingReadiness(runs);
      const strictEstimate = estimatedFromSample(totals.genui, readiness.strict_valid, readiness.sampled);
      const steps = [
        {
          label: "Stage 1 Queries",
          count: totals.queries,
          rate: totals.queries ? 1 : null,
          detail: "accepted query records",
        },
        {
          label: "Stage 2 Responses",
          count: totals.responses,
          rate: ratioOrNull(totals.responses, totals.queries),
          stageRate: ratioOrNull(totals.responses, totals.queries),
          stageLabel: "Q->R",
          detail: `missing responses ${fmt(totals.response_backlog)}`,
        },
        {
          label: "Stage 3 IR",
          count: totals.genui,
          rate: ratioOrNull(totals.genui, totals.queries),
          stageRate: ratioOrNull(totals.genui, totals.responses),
          stageLabel: "R->IR",
          detail: `missing IR ${fmt(totals.ir_backlog)}`,
        },
        {
          label: "Strict Valid IR (estimated)",
          count: strictEstimate,
          rate: ratioOrNull(strictEstimate, totals.queries),
          stageRate: ratioOrNull(readiness.strict_valid, readiness.sampled),
          stageLabel: "sample strict",
          detail: `${fmt(readiness.strict_valid)} / ${fmt(readiness.sampled)} sampled rows passed strict schema`,
        },
        {
          label: "Ready >=70 (estimated)",
          count: readiness.estimated_ready_score70,
          rate: ratioOrNull(readiness.estimated_ready_score70, totals.queries),
          stageRate: ratioOrNull(readiness.ready_score70, readiness.sampled),
          stageLabel: "sample ready",
          detail: "strict, no fallback/error/markdown leakage, score >=70",
        },
        {
          label: "Ready >=80 (estimated)",
          count: readiness.estimated_ready_score80,
          rate: ratioOrNull(readiness.estimated_ready_score80, totals.queries),
          stageRate: ratioOrNull(readiness.ready_score80, readiness.sampled),
          stageLabel: "sample ready",
          detail: "same gate with score >=80",
        },
      ];
      const sourceRows = sourceFunnelRows(runs, sources).slice(0, 12).map(row => `
        <tr>
          <td><b>${escapeHtml(row.source_label)}</b><br><span class="small">${fmt(row.run_count)} runs</span></td>
          <td>${fmt(row.queries)} / ${fmt(row.responses)} / ${fmt(row.genui)}</td>
          <td>${pct(row.queryToResponse)}<br><span class="small">Q->R</span></td>
          <td>${pct(row.responseToIr)}<br><span class="small">R->IR</span></td>
          <td>${pct(row.queryToIr)}<br><span class="small">Q->IR</span></td>
          <td>${fmt(row.readiness.estimated_ready_score70 || 0)}<br><span class="small">${pct(row.ready70Rate)} of paired</span></td>
          <td><span class="score ${scoreClass(row.avg_score)}">${scoreText(row.avg_score)}</span></td>
        </tr>`).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${pct(ratioOrNull(totals.responses, totals.queries))}</b><br><span class="small">query -> response</span></div>
          <div class="detail-box"><b>${pct(ratioOrNull(totals.genui, totals.responses))}</b><br><span class="small">response -> IR</span></div>
          <div class="detail-box"><b>${pct(ratioOrNull(totals.genui, totals.queries))}</b><br><span class="small">query -> IR</span></div>
          <div class="detail-box"><b>${fmt(strictEstimate || 0)}</b><br><span class="small">estimated strict valid IR</span></div>
          <div class="detail-box"><b>${fmt(readiness.estimated_ready_score70 || 0)}</b><br><span class="small">estimated ready >=70</span></div>
          <div class="detail-box"><b>${fmt(readiness.estimated_ready_score80 || 0)}</b><br><span class="small">estimated ready >=80</span></div>
        </div>
        <div class="warning-list">${steps.map(renderFunnelStep).join("")}</div>
        <h2>Source Funnel</h2>
        <div class="scroll">
          <table>
            <thead><tr><th>Source</th><th>Q / R / IR</th><th>Q->R</th><th>R->IR</th><th>Q->IR</th><th>Ready >=70</th><th>Score</th></tr></thead>
            <tbody>${sourceRows || "<tr><td colspan='7'><span class='small'>No source rows.</span></td></tr>"}</tbody>
          </table>
        </div>
        <div class="small">Strict-valid and ready counts are sampled estimates from the same Stage 3 readiness gates used by the Training Readiness panel. Query/response/IR conversion uses exact filtered counts.</div>`;
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
    function renderDataIntegrity(runs) {
      const target = document.getElementById("dataIntegrity");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const totals = runs.reduce((acc, run) => {
        const s = run.data_integrity || {};
        acc.total += s.total_issues || 0;
        acc.parse += s.parse_error_count || 0;
        acc.duplicates += s.duplicate_id_count || 0;
        acc.missing += s.missing_id_count || 0;
        acc.orphans += s.orphan_link_count || 0;
        acc.warnings += s.count_warning_count || 0;
        if ((s.total_issues || 0) > 0) acc.affected += 1;
        return acc;
      }, {total: 0, parse: 0, duplicates: 0, missing: 0, orphans: 0, warnings: 0, affected: 0});
      const issueRows = runs
        .filter(run => integrityIssueCount(run))
        .sort((a, b) => integrityIssueCount(b) - integrityIssueCount(a) || String(a.run_id).localeCompare(String(b.run_id)))
        .slice(0, 12)
        .map(run => {
          const labels = integrityIssueLabels(run.data_integrity || {});
          return `
            <tr>
              <td><b>${escapeHtml(run.run_id)}</b><br><span class="small">${escapeHtml(run.source_label)}</span><br><button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(run))}')">Details</button></td>
              <td><b>${fmt(integrityIssueCount(run))}</b></td>
              <td>${labels.map(label => `<span class="badge warn">${escapeHtml(label)}</span>`).join(" ")}</td>
              <td class="small">Q ${fmt(run.queries)} | R ${fmt(run.responses)} | IR ${fmt(run.genui)}</td>
            </tr>`;
        }).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(totals.affected)}</b><br><span class="small">affected runs</span></div>
          <div class="detail-box"><b>${fmt(totals.total)}</b><br><span class="small">total issue signals</span></div>
          <div class="detail-box"><b>${fmt(totals.parse)}</b><br><span class="small">JSONL parse errors</span></div>
          <div class="detail-box"><b>${fmt(totals.duplicates)}</b><br><span class="small">duplicate IDs</span></div>
          <div class="detail-box"><b>${fmt(totals.missing)}</b><br><span class="small">missing IDs</span></div>
          <div class="detail-box"><b>${fmt(totals.orphans)}</b><br><span class="small">broken stage links</span></div>
          <div class="detail-box"><b>${fmt(totals.warnings)}</b><br><span class="small">count warnings</span></div>
        </div>
        <div class="scroll">
          <table>
            <thead><tr><th>Run</th><th>Issues</th><th>Categories</th><th>Counts</th></tr></thead>
            <tbody>${issueRows || "<tr><td colspan='4'><span class='small'>No data integrity issues found in current filters.</span></td></tr>"}</tbody>
          </table>
        </div>
        <div class="small">Integrity checks scan core JSONL IDs and links: query_id, response_id, ui_id, duplicate IDs, parse errors, and response/IR records that cannot be matched to upstream stages.</div>`;
    }
    function aggregateContentDuplicates(runs) {
      const totals = {
        affectedRuns: 0,
        duplicateRows: 0,
        queries: {sampled: 0, unique: 0, duplicateRows: 0, duplicateGroups: 0},
        responses: {sampled: 0, unique: 0, duplicateRows: 0, duplicateGroups: 0},
        irPayloads: {sampled: 0, unique: 0, duplicateRows: 0, duplicateGroups: 0},
      };
      for (const run of runs) {
        const summary = run.content_duplicates || {};
        const total = contentDuplicateTotal(run);
        if (total) totals.affectedRuns += 1;
        totals.duplicateRows += total;
        for (const [field, key] of [["queries", "queries"], ["responses", "responses"], ["irPayloads", "ir_payloads"]]) {
          const data = duplicateStageSummary(summary, key);
          totals[field].sampled += data.sampled;
          totals[field].unique += data.unique;
          totals[field].duplicateRows += data.duplicateRows;
          totals[field].duplicateGroups += data.duplicateGroups;
        }
      }
      return totals;
    }
    function duplicateMetricRows(totals) {
      const rows = [
        ["Queries", totals.queries],
        ["Responses", totals.responses],
        ["IR payloads", totals.irPayloads],
      ];
      return rows.map(([label, data]) => `
        <tr>
          <td><b>${label}</b></td>
          <td>${fmt(data.duplicateRows)}</td>
          <td>${fmt(data.duplicateGroups)}</td>
          <td>${pct(ratioOrNull(data.duplicateRows, data.sampled))}</td>
          <td class="small">${fmt(data.unique)} unique / ${fmt(data.sampled)} sampled</td>
        </tr>`).join("");
    }
    function renderContentDuplicates(runs) {
      const target = document.getElementById("contentDuplicates");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const totals = aggregateContentDuplicates(runs);
      const affectedRows = [...runs]
        .filter(run => contentDuplicateTotal(run))
        .sort((a, b) => contentDuplicateTotal(b) - contentDuplicateTotal(a) || String(a.run_id).localeCompare(String(b.run_id)))
        .slice(0, 12)
        .map(run => {
          const d = run.content_duplicates || {};
          return `
            <tr>
              <td><b>${escapeHtml(run.run_id)}</b><br><span class="small">${escapeHtml(run.source_label)}</span><br><button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(run))}')">Details</button></td>
              <td>${fmt(contentDuplicateTotal(run))}</td>
              <td>${fmt(duplicateStageSummary(d, "queries").duplicateRows)}<br><span class="small">${pct(duplicateStageSummary(d, "queries").duplicateRate)}</span></td>
              <td>${fmt(duplicateStageSummary(d, "responses").duplicateRows)}<br><span class="small">${pct(duplicateStageSummary(d, "responses").duplicateRate)}</span></td>
              <td>${fmt(duplicateStageSummary(d, "ir_payloads").duplicateRows)}<br><span class="small">${pct(duplicateStageSummary(d, "ir_payloads").duplicateRate)}</span></td>
              <td><span class="score ${scoreClass(run.display_score ?? run.overall_score)}">${scoreText(run.display_score ?? run.overall_score)}</span></td>
            </tr>`;
        }).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(totals.affectedRuns)}</b><br><span class="small">runs with sampled duplicates</span></div>
          <div class="detail-box"><b>${fmt(totals.duplicateRows)}</b><br><span class="small">duplicate content rows</span></div>
          <div class="detail-box"><b>${fmt(totals.queries.duplicateRows)}</b><br><span class="small">duplicate queries</span></div>
          <div class="detail-box"><b>${fmt(totals.responses.duplicateRows)}</b><br><span class="small">duplicate responses</span></div>
          <div class="detail-box"><b>${fmt(totals.irPayloads.duplicateRows)}</b><br><span class="small">duplicate IR payloads</span></div>
        </div>
        <div class="two-col-panels wide-panel">
          <div>
            <h2>Duplicate Rates</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Stage</th><th>Duplicate Rows</th><th>Groups</th><th>Rate</th><th>Sample</th></tr></thead>
                <tbody>${duplicateMetricRows(totals)}</tbody>
              </table>
            </div>
          </div>
          <div>
            <h2>Top Duplicate Runs</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Run</th><th>Total</th><th>Query</th><th>Response</th><th>IR</th><th>Score</th></tr></thead>
                <tbody>${affectedRows || "<tr><td colspan='6'><span class='small'>No sampled duplicate content found.</span></td></tr>"}</tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="small">Duplicate detection uses normalized exact text/payload matching over sampled JSONL rows already loaded for the dashboard. It catches repeated content, not semantic near-duplicates.</div>`;
    }
    function renderRunLogs(runs) {
      const target = document.getElementById("runLogs");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const withLogs = runs.filter(run => (run.run_logs || {}).present);
      const issueRuns = withLogs.filter(run => logIssueCount(run));
      const staleLogRuns = withLogs.filter(run => {
        const updated = (run.run_logs || {}).latest_updated_at;
        return updated && (ageHours(updated) ?? 0) > 24;
      });
      const issueRows = issueRuns
        .sort((a, b) => logIssueCount(b) - logIssueCount(a) || String(a.run_id).localeCompare(String(b.run_id)))
        .slice(0, 12)
        .map(run => {
          const logs = run.run_logs || {};
          const firstIssue = (logs.recent_issues || [])[0];
          return `
            <tr>
              <td><b>${escapeHtml(run.run_id)}</b><br><span class="small">${escapeHtml(run.source_label)}</span><br><button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(run))}')">Details</button></td>
              <td>${fmt(logs.file_count || 0)} files<br><span class="small">${bytesText(logs.total_bytes || 0)}</span></td>
              <td><span class="badge error">${fmt(logs.issue_count || 0)} issues</span><br><span class="small">${fmt(logs.progress_count || 0)} progress lines</span></td>
              <td class="small">${logs.latest_updated_at ? `${ageText(logs.latest_updated_at)} (${new Date(logs.latest_updated_at).toLocaleString()})` : "unknown"}</td>
              <td class="small">${firstIssue ? `${escapeHtml(firstIssue.file)}:${fmt(firstIssue.line)} ${escapeHtml(firstIssue.text)}` : "n/a"}</td>
            </tr>`;
        }).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(withLogs.length)}</b><br><span class="small">runs with logs</span></div>
          <div class="detail-box"><b>${fmt(issueRuns.length)}</b><br><span class="small">runs with log issues</span></div>
          <div class="detail-box"><b>${fmt(runs.reduce((sum, run) => sum + logIssueCount(run), 0))}</b><br><span class="small">issue lines in log tails</span></div>
          <div class="detail-box"><b>${fmt(withLogs.reduce((sum, run) => sum + Number((run.run_logs || {}).progress_count || 0), 0))}</b><br><span class="small">progress lines in log tails</span></div>
          <div class="detail-box"><b>${fmt(staleLogRuns.length)}</b><br><span class="small">log files stale >24h</span></div>
        </div>
        <div class="scroll">
          <table>
            <thead><tr><th>Run</th><th>Logs</th><th>Tail Signals</th><th>Latest Log</th><th>Recent Issue</th></tr></thead>
            <tbody>${issueRows || "<tr><td colspan='5'><span class='small'>No issue lines found in current run log tails.</span></td></tr>"}</tbody>
          </table>
        </div>
        <div class="small">Log scanning reads bounded tails from run log files only. Counts are operational signals, not full historical log totals.</div>`;
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
    const metricRiskRules = [
      {key: "content_coverage", label: "Content coverage", kind: "pct", direction: "low", threshold: 0.65, weight: 1.5},
      {key: "intent_score", label: "Intent score", kind: "pct", direction: "low", threshold: 0.70, weight: 1.5},
      {key: "section_heading_coverage", label: "Heading coverage", kind: "pct", direction: "low", threshold: 0.50, weight: 1.1},
      {key: "table_cell_coverage", label: "Table coverage", kind: "pct", direction: "low", threshold: 0.40, weight: 1.0},
      {key: "action_coverage", label: "Action coverage", kind: "pct", direction: "low", threshold: 0.50, weight: 1.0},
      {key: "image_presence", label: "Image presence", kind: "pct", direction: "low", threshold: 0.25, weight: 0.8},
      {key: "icon_presence", label: "Icon presence", kind: "pct", direction: "low", threshold: 0.25, weight: 0.8},
      {key: "markdown_leakage_rate", label: "Markdown leakage", kind: "pct_low", direction: "high", threshold: 0.01, weight: 1.0},
      {key: "component_count", label: "Components", kind: "number", direction: "low", threshold: 12, weight: 0.8},
    ];
    function metricDisplay(metric, value) {
      if (value == null) return "n/a";
      if (metric.kind === "pct" || metric.kind === "pct_low") return metricPct(value);
      return Number(value).toFixed(Number(value) >= 10 ? 1 : 2);
    }
    function metricRiskGap(rule, value) {
      if (value == null || Number.isNaN(Number(value))) return null;
      const actual = Number(value);
      const threshold = Number(rule.threshold);
      if (rule.direction === "high") {
        if (actual <= threshold) return null;
        return Math.max(0.1, actual - threshold) * Number(rule.weight || 1);
      }
      if (actual >= threshold) return null;
      const base = threshold ? (threshold - actual) / Math.abs(threshold) : threshold - actual;
      return Math.max(0.1, base) * Number(rule.weight || 1);
    }
    function metricRiskForRun(run) {
      const metrics = run.metric_avgs || {};
      const failures = [];
      for (const rule of metricRiskRules) {
        const value = metrics[rule.key];
        const gap = metricRiskGap(rule, value);
        if (gap == null) continue;
        failures.push({...rule, value, gap});
      }
      failures.sort((a, b) => b.gap - a.gap || a.label.localeCompare(b.label));
      return failures;
    }
    function metricRiskScore(run) {
      return metricRiskForRun(run).reduce((sum, item) => sum + item.gap, 0);
    }
    function metricRiskBadges(run, limit = 4) {
      const failures = metricRiskForRun(run).slice(0, limit);
      return failures.map(item => `<span class="badge warn">${escapeHtml(item.label)} ${metricDisplay(item, item.value)}</span>`).join(" ");
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
    function aggregateMetricRisk(runs) {
      const metricBuckets = new Map(metricRiskRules.map(rule => [
        rule.key,
        {...rule, affectedRuns: 0, affectedIr: 0, valueSum: 0, valueWeight: 0, worstRun: null, worstValue: null, worstGap: 0},
      ]));
      const riskyRuns = [];
      for (const run of runs) {
        const failures = metricRiskForRun(run);
        if (!failures.length) continue;
        const riskScore = metricRiskScore(run);
        riskyRuns.push({run, failures, riskScore});
        const weight = Math.max(1, Number(run.genui || 0));
        for (const failure of failures) {
          const bucket = metricBuckets.get(failure.key);
          bucket.affectedRuns += 1;
          bucket.affectedIr += Number(run.genui || 0);
          bucket.valueSum += Number(failure.value) * weight;
          bucket.valueWeight += weight;
          if (bucket.worstRun == null || failure.gap > bucket.worstGap) {
            bucket.worstRun = run;
            bucket.worstValue = failure.value;
            bucket.worstGap = failure.gap;
          }
        }
      }
      return {
        affectedRuns: riskyRuns.length,
        riskSignals: riskyRuns.reduce((sum, row) => sum + row.failures.length, 0),
        metricRows: [...metricBuckets.values()]
          .filter(row => row.affectedRuns)
          .map(row => ({...row, avgValue: row.valueWeight ? row.valueSum / row.valueWeight : null}))
          .sort((a, b) => b.affectedRuns - a.affectedRuns || b.affectedIr - a.affectedIr || a.label.localeCompare(b.label)),
        riskyRuns: riskyRuns
          .sort((a, b) => b.riskScore - a.riskScore || String(a.run.run_id).localeCompare(String(b.run.run_id)))
          .slice(0, 12),
      };
    }
    function renderMetricRisk(runs) {
      const target = document.getElementById("metricRisk");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const risk = aggregateMetricRisk(runs);
      const topMetric = risk.metricRows[0];
      const metricRows = risk.metricRows.map(row => `
        <tr>
          <td><b>${escapeHtml(row.label)}</b><br><span class="small">${row.direction === "high" ? ">" : "<"} ${metricDisplay(row, row.threshold)}</span></td>
          <td>${fmt(row.affectedRuns)}</td>
          <td>${fmt(row.affectedIr)}</td>
          <td>${metricDisplay(row, row.avgValue)}</td>
          <td>${metricDisplay(row, row.worstValue)}<br><span class="small">${row.worstRun ? escapeHtml(row.worstRun.run_id) : "n/a"}</span></td>
        </tr>`).join("");
      const runRows = risk.riskyRuns.map(row => `
        <tr>
          <td><b>${escapeHtml(row.run.run_id)}</b><br><span class="small">${escapeHtml(row.run.source_label)}</span><br><button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(row.run))}')">Details</button></td>
          <td><span class="score ${scoreClass(row.run.display_score ?? row.run.overall_score)}">${scoreText(row.run.display_score ?? row.run.overall_score)}</span></td>
          <td>${fmt(row.failures.length)}<br><span class="small">risk ${row.riskScore.toFixed(2)}</span></td>
          <td>${metricRiskBadges(row.run, 6) || "<span class='small'>none</span>"}</td>
          <td class="small">Q/R/IR ${fmt(row.run.queries)} / ${fmt(row.run.responses)} / ${fmt(row.run.genui)}</td>
        </tr>`).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(risk.affectedRuns)}</b><br><span class="small">runs with metric risk</span></div>
          <div class="detail-box"><b>${fmt(risk.riskSignals)}</b><br><span class="small">metric risk signals</span></div>
          <div class="detail-box"><b>${topMetric ? escapeHtml(topMetric.label) : "none"}</b><br><span class="small">most common risk</span></div>
          <div class="detail-box"><b>${topMetric ? fmt(topMetric.affectedRuns) : "0"}</b><br><span class="small">runs affected by top risk</span></div>
        </div>
        <div class="two-col-panels wide-panel">
          <div>
            <h2>Risk By Metric</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Metric</th><th>Runs</th><th>IR</th><th>Avg Failing Value</th><th>Worst</th></tr></thead>
                <tbody>${metricRows || "<tr><td colspan='5'><span class='small'>No metric risks found.</span></td></tr>"}</tbody>
              </table>
            </div>
          </div>
          <div>
            <h2>Highest Risk Runs</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Run</th><th>Score</th><th>Risks</th><th>Signals</th><th>Volume</th></tr></thead>
                <tbody>${runRows || "<tr><td colspan='5'><span class='small'>No metric risks found.</span></td></tr>"}</tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="small">Metric risk uses heuristic thresholds over per-run sampled metric averages. It complements score bands by showing the likely failing quality dimension.</div>
      `;
    }
    function aggregateTrainingReadiness(runs) {
      const totals = {
        paired_records: 0,
        sampled: 0,
        json_parse_ok: 0,
        strict_valid: 0,
        score_ge_60: 0,
        score_ge_70: 0,
        score_ge_80: 0,
        score_unknown: 0,
        content_ge_65: 0,
        repair_free: 0,
        fallback_free: 0,
        markdown_clean: 0,
        gen_error_free: 0,
        ready_score70: 0,
        ready_score80: 0,
        estimated_ready_score70: 0,
        estimated_ready_score80: 0,
        tokenSum: 0,
        tokenWeight: 0,
        componentSum: 0,
        componentWeight: 0,
      };
      const runRows = [];
      for (const run of runs) {
        const t = run.training_readiness || {};
        for (const key of [
          "paired_records","sampled","json_parse_ok","strict_valid","score_ge_60","score_ge_70","score_ge_80",
          "score_unknown","content_ge_65","repair_free","fallback_free","markdown_clean","gen_error_free",
          "ready_score70","ready_score80","estimated_ready_score70","estimated_ready_score80",
        ]) {
          totals[key] += Number(t[key] || 0);
        }
        if (t.avg_output_tokens_json != null && Number(t.sampled || 0)) {
          totals.tokenSum += Number(t.avg_output_tokens_json) * Number(t.sampled || 0);
          totals.tokenWeight += Number(t.sampled || 0);
        }
        if (t.avg_component_count != null && Number(t.sampled || 0)) {
          totals.componentSum += Number(t.avg_component_count) * Number(t.sampled || 0);
          totals.componentWeight += Number(t.sampled || 0);
        }
        const sampled = Number(t.sampled || 0);
        const readyRate = sampled ? Number(t.ready_score70 || 0) / sampled : null;
        const strictRate = sampled ? Number(t.strict_valid || 0) / sampled : null;
        if (sampled || run.genui) {
          runRows.push({
            run,
            sampled,
            paired: Number(t.paired_records || 0),
            ready70: Number(t.ready_score70 || 0),
            estimatedReady70: Number(t.estimated_ready_score70 || 0),
            readyRate,
            strictRate,
            avgScore: run.display_score ?? run.overall_score,
          });
        }
      }
      totals.avg_output_tokens_json = totals.tokenWeight ? totals.tokenSum / totals.tokenWeight : null;
      totals.avg_component_count = totals.componentWeight ? totals.componentSum / totals.componentWeight : null;
      totals.weak_runs = runRows
        .filter(row => row.sampled && row.paired)
        .sort((a, b) => (a.readyRate - b.readyRate) || (a.strictRate - b.strictRate) || String(a.run.run_id).localeCompare(String(b.run.run_id)))
        .slice(0, 12);
      totals.best_runs = runRows
        .filter(row => row.sampled && row.paired)
        .sort((a, b) => (b.estimatedReady70 - a.estimatedReady70) || (b.readyRate - a.readyRate) || String(a.run.run_id).localeCompare(String(b.run.run_id)))
        .slice(0, 8);
      return totals;
    }
    function renderTrainingReadinessDetails(t) {
      const sampled = Math.max(1, Number(t.sampled || 0));
      return `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(t.paired_records || 0)}</b><br><span class="small">paired response/IR records</span></div>
          <div class="detail-box"><b>${fmt(t.estimated_ready_score70 || 0)}</b><br><span class="small">estimated ready >=70</span></div>
          <div class="detail-box"><b>${fmt(t.estimated_ready_score80 || 0)}</b><br><span class="small">estimated ready >=80</span></div>
          <div class="detail-box"><b>${metricPct((t.ready_score70 || 0) / sampled)}</b><br><span class="small">sample ready >=70 rate</span></div>
          <div class="detail-box"><b>${metricPct((t.strict_valid || 0) / sampled)}</b><br><span class="small">strict schema pass</span></div>
          <div class="detail-box"><b>${metricPct((t.json_parse_ok || 0) / sampled)}</b><br><span class="small">JSON parse ok</span></div>
          <div class="detail-box"><b>${metricPct((t.repair_free || 0) / sampled)}</b><br><span class="small">repair-free</span></div>
          <div class="detail-box"><b>${metricPct((t.fallback_free || 0) / sampled)}</b><br><span class="small">fallback-free</span></div>
          <div class="detail-box"><b>${metricPct((t.markdown_clean || 0) / sampled)}</b><br><span class="small">markdown-clean</span></div>
          <div class="detail-box"><b>${Number(t.avg_output_tokens_json || 0).toFixed(0)}</b><br><span class="small">avg JSON output tokens</span></div>
          <div class="detail-box"><b>${Number(t.avg_component_count || 0).toFixed(1)}</b><br><span class="small">avg components</span></div>
        </div>`;
    }
    function renderTrainingReadiness(runs) {
      const target = document.getElementById("trainingReadiness");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const t = aggregateTrainingReadiness(runs);
      const weakRows = t.weak_runs.map(row => `
        <tr>
          <td><b>${escapeHtml(row.run.run_id)}</b><br><span class="small">${escapeHtml(row.run.source_label)}</span><br><button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(row.run))}')">Details</button></td>
          <td>${fmt(row.paired)} paired<br><span class="small">${fmt(row.sampled)} sampled</span></td>
          <td>${metricPct(row.readyRate)}<br><span class="small">est ${fmt(row.estimatedReady70)} ready</span></td>
          <td>${metricPct(row.strictRate)}</td>
          <td><span class="score ${scoreClass(row.avgScore)}">${scoreText(row.avgScore)}</span></td>
        </tr>`).join("");
      const bestRows = t.best_runs.map(row => `
        <tr>
          <td><b>${escapeHtml(row.run.run_id)}</b><br><span class="small">${escapeHtml(row.run.source_label)}</span><br><button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(row.run))}')">Details</button></td>
          <td>${fmt(row.estimatedReady70)} ready >=70<br><span class="small">${fmt(row.paired)} paired</span></td>
          <td>${metricPct(row.readyRate)}</td>
          <td><span class="score ${scoreClass(row.avgScore)}">${scoreText(row.avgScore)}</span></td>
        </tr>`).join("");
      target.innerHTML = `
        ${renderTrainingReadinessDetails(t)}
        <div class="two-col-panels wide-panel">
          <div>
            <h2>Weakest Training Slices</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Run</th><th>Records</th><th>Ready >=70</th><th>Strict</th><th>Score</th></tr></thead>
                <tbody>${weakRows || "<tr><td colspan='5'><span class='small'>No sampled paired records found.</span></td></tr>"}</tbody>
              </table>
            </div>
          </div>
          <div>
            <h2>Largest Ready Runs</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Run</th><th>Estimated Ready</th><th>Rate</th><th>Score</th></tr></thead>
                <tbody>${bestRows || "<tr><td colspan='4'><span class='small'>No sampled paired records found.</span></td></tr>"}</tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="small">Training readiness is a sampled estimate. A record is counted ready when Stage 3 JSON parses, strict schema passes, no generation error/fallback/markdown leakage is observed, and overall score meets the threshold.</div>`;
    }
    function addStructureCounts(target, obj) {
      for (const [key, value] of Object.entries(obj || {})) {
        target.set(key, (target.get(key) || 0) + Number(value || 0));
      }
    }
    function aggregateIrStructure(runs) {
      const componentTypes = new Map();
      const uncommonTypes = new Map();
      const tableDomains = new Map();
      const tablePresentations = new Map();
      const actionTypes = new Map();
      const totals = {
        sampled: 0,
        flat_spec_rows: 0,
        component_count: 0,
        table_count: 0,
        button_count: 0,
        image_count: 0,
        icon_count: 0,
        chart_count: 0,
        formula_count: 0,
        code_count: 0,
        email_preview_count: 0,
        table_rows_total: 0,
        table_rows_counted: 0,
        table_columns_total: 0,
        table_columns_counted: 0,
      };
      for (const run of runs) {
        const s = run.ir_structure || {};
        for (const key of Object.keys(totals)) {
          totals[key] += Number(s[key] || 0);
        }
        addStructureCounts(componentTypes, s.component_types);
        addStructureCounts(uncommonTypes, s.uncommon_component_types);
        addStructureCounts(tableDomains, s.table_domains);
        addStructureCounts(tablePresentations, s.table_presentations);
        addStructureCounts(actionTypes, s.action_types);
      }
      return {
        ...totals,
        avg_components: totals.flat_spec_rows ? totals.component_count / totals.flat_spec_rows : null,
        avg_table_rows: totals.table_rows_counted ? totals.table_rows_total / totals.table_rows_counted : null,
        avg_table_columns: totals.table_columns_counted ? totals.table_columns_total / totals.table_columns_counted : null,
        component_types: topCounts(componentTypes, 12),
        uncommon_component_types: topCounts(uncommonTypes, 8),
        table_domains: topCounts(tableDomains, 10),
        table_presentations: topCounts(tablePresentations, 8),
        action_types: topCounts(actionTypes, 10),
      };
    }
    function renderStructureCountBlock(title, entries) {
      return renderDistributionBlock(title, entries.length ? entries : [["none", 0]]);
    }
    function renderIrStructure(runs) {
      const target = document.getElementById("irStructure");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const s = aggregateIrStructure(runs);
      const uncommonRows = s.uncommon_component_types.map(([type, count]) => `
        <div class="warning-row">
          <span>${escapeHtml(type)}</span>
          <b>${fmt(count)}</b>
        </div>`).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(s.flat_spec_rows)}</b><br><span class="small">sampled flat-spec rows</span></div>
          <div class="detail-box"><b>${Number(s.avg_components || 0).toFixed(1)}</b><br><span class="small">avg components</span></div>
          <div class="detail-box"><b>${fmt(s.component_count)}</b><br><span class="small">sampled components</span></div>
          <div class="detail-box"><b>${fmt(s.table_count)}</b><br><span class="small">tables</span></div>
          <div class="detail-box"><b>${fmt(s.button_count)}</b><br><span class="small">buttons</span></div>
          <div class="detail-box"><b>${fmt(s.image_count)}</b><br><span class="small">images</span></div>
          <div class="detail-box"><b>${fmt(s.icon_count)}</b><br><span class="small">icons</span></div>
          <div class="detail-box"><b>${Number(s.avg_table_rows || 0).toFixed(1)}</b><br><span class="small">avg table rows</span></div>
          <div class="detail-box"><b>${Number(s.avg_table_columns || 0).toFixed(1)}</b><br><span class="small">avg table columns</span></div>
        </div>
        <div class="dist-grid">
          ${renderStructureCountBlock("Component Types", s.component_types)}
          ${renderStructureCountBlock("Table Domains", s.table_domains)}
          ${renderStructureCountBlock("Table Presentations", s.table_presentations)}
          ${renderStructureCountBlock("Action Types", s.action_types)}
        </div>
        <h2>Special Components</h2>
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(s.chart_count)}</b><br><span class="small">charts</span></div>
          <div class="detail-box"><b>${fmt(s.formula_count)}</b><br><span class="small">formula elements</span></div>
          <div class="detail-box"><b>${fmt(s.code_count)}</b><br><span class="small">code/console blocks</span></div>
          <div class="detail-box"><b>${fmt(s.email_preview_count)}</b><br><span class="small">email previews</span></div>
          <div class="detail-box"><b>${fmt(s.uncommon_component_types.reduce((sum, [, count]) => sum + Number(count || 0), 0))}</b><br><span class="small">uncommon components</span></div>
        </div>
        <h2>Uncommon Component Types</h2>
        <div class="warning-list">${uncommonRows || "<span class='small'>No uncommon component types in sampled IR.</span>"}</div>
        <div class="small">IR structure is computed from sampled genui.jsonl records per run. It helps verify prompt output and renderer coverage by showing actual component, table-domain, and action usage.</div>`;
    }
    function aggregateMediaHealth(runs) {
      const responseHosts = new Map();
      const irHosts = new Map();
      const extensions = new Map();
      const totals = {
        sampled_response_rows: 0,
        sampled_ir_rows: 0,
        response_asset_records: 0,
        response_asset_files_present: 0,
        response_asset_files_missing: 0,
        response_asset_bytes: 0,
        ir_media_components: 0,
        ir_image_components: 0,
        ir_icon_components: 0,
        ir_media_refs: 0,
        ir_local_media_refs: 0,
        ir_local_media_missing: 0,
        ir_remote_media_refs: 0,
        ir_symbolic_icon_refs: 0,
        missing_local_samples: [],
        remote_ref_samples: [],
        affected_runs: 0,
      };
      for (const run of runs) {
        const m = run.media_health || {};
        const issueCount = Number(m.ir_local_media_missing || 0) + Number(m.response_asset_files_missing || 0);
        if (issueCount) totals.affected_runs += 1;
        for (const key of Object.keys(totals)) {
          if (Array.isArray(totals[key])) continue;
          if (key === "affected_runs") continue;
          totals[key] += Number(m[key] || 0);
        }
        addStructureCounts(responseHosts, m.response_asset_hosts);
        addStructureCounts(irHosts, m.ir_remote_hosts);
        addStructureCounts(extensions, m.asset_file_extensions);
        for (const sample of m.missing_local_samples || []) {
          if (totals.missing_local_samples.length < 10) {
            totals.missing_local_samples.push({...sample, run_id: run.run_id, source_label: run.source_label});
          }
        }
        for (const sample of m.remote_ref_samples || []) {
          if (totals.remote_ref_samples.length < 10) {
            totals.remote_ref_samples.push({...sample, run_id: run.run_id, source_label: run.source_label});
          }
        }
      }
      return {
        ...totals,
        response_asset_hosts: topCounts(responseHosts, 10),
        ir_remote_hosts: topCounts(irHosts, 10),
        asset_file_extensions: topCounts(extensions, 10),
      };
    }
    function mediaSampleRows(samples, emptyText) {
      return (samples || []).map(sample => `
        <div class="warning-row">
          <span>
            <b>${escapeHtml(sample.run_id || sample.id || "sample")}</b>
            ${sample.source_label ? `<span class="small"> | ${escapeHtml(sample.source_label)}</span>` : ""}<br>
            <span class="small">${escapeHtml(sample.type || "media")} ${sample.id ? `from ${escapeHtml(sample.id)}` : ""}: ${escapeHtml(sample.value || "")}</span>
          </span>
        </div>`).join("") || `<span class='small'>${emptyText}</span>`;
    }
    function renderMediaHealth(runs) {
      const target = document.getElementById("mediaHealth");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const s = aggregateMediaHealth(runs);
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(s.response_asset_records)}</b><br><span class="small">response asset records</span></div>
          <div class="detail-box"><b>${fmt(s.response_asset_files_present)}</b><br><span class="small">asset files present</span></div>
          <div class="detail-box"><b>${fmt(s.response_asset_files_missing)}</b><br><span class="small">missing response assets</span></div>
          <div class="detail-box"><b>${bytesText(s.response_asset_bytes)}</b><br><span class="small">declared response asset bytes</span></div>
          <div class="detail-box"><b>${fmt(s.ir_media_components)}</b><br><span class="small">IR media components</span></div>
          <div class="detail-box"><b>${fmt(s.ir_image_components)}</b><br><span class="small">IR images</span></div>
          <div class="detail-box"><b>${fmt(s.ir_icon_components)}</b><br><span class="small">IR icons</span></div>
          <div class="detail-box"><b>${fmt(s.ir_local_media_missing)}</b><br><span class="small">missing local IR refs</span></div>
          <div class="detail-box"><b>${fmt(s.ir_remote_media_refs)}</b><br><span class="small">remote refs left in IR</span></div>
          <div class="detail-box"><b>${fmt(s.affected_runs)}</b><br><span class="small">affected runs</span></div>
        </div>
        <div class="dist-grid">
          ${renderStructureCountBlock("Response Asset Hosts", s.response_asset_hosts)}
          ${renderStructureCountBlock("IR Remote Hosts", s.ir_remote_hosts)}
          ${renderStructureCountBlock("Asset Extensions", s.asset_file_extensions)}
        </div>
        <h2>Missing Local Media Samples</h2>
        <div class="warning-list">${mediaSampleRows(s.missing_local_samples, "No missing local media references found in sampled rows.")}</div>
        <h2>Remote IR Media Samples</h2>
        <div class="warning-list">${mediaSampleRows(s.remote_ref_samples, "No remote media references found in sampled IR rows.")}</div>
        <div class="small">Media health is sampled from response asset metadata and flat-spec Image/Icon props. Missing local refs indicate copied assets are incomplete or IR points at the wrong path; remote refs indicate assets were not localized before IR/rendering.</div>`;
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
    const promptStageDefs = [
      {key: "query_prompt_versions", label: "Stage 1 Query", countKey: "queries"},
      {key: "response_prompt_versions", label: "Stage 2 Response", countKey: "responses"},
      {key: "ir_prompt_versions", label: "Stage 3 IR", countKey: "genui"},
    ];
    function dominantPromptVersion(counts) {
      const entries = Object.entries(counts || {});
      return entries.length ? entries[0][0] : "unknown";
    }
    function promptVersionCount(counts) {
      return Object.keys(counts || {}).length;
    }
    function renderPromptVersionDetails(run) {
      const boxes = promptStageDefs.map(stage => {
        const counts = run[stage.key] || {};
        const total = Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0);
        const versionCount = promptVersionCount(counts);
        return `<div class="detail-box"><b>${escapeHtml(dominantPromptVersion(counts))}</b><br><span class="small">${escapeHtml(stage.label)} | ${fmt(total)} rows | ${fmt(versionCount)} version${versionCount === 1 ? "" : "s"}</span></div>`;
      }).join("");
      return `
        <div class="detail-grid">${boxes}</div>
        <div class="dist-grid">
          ${renderStructureCountBlock("Stage 1 Prompt Versions", sortedObjectEntries(run.query_prompt_versions || {}, 8))}
          ${renderStructureCountBlock("Stage 2 Prompt Versions", sortedObjectEntries(run.response_prompt_versions || {}, 8))}
          ${renderStructureCountBlock("Stage 3 Prompt Versions", sortedObjectEntries(run.ir_prompt_versions || run.ir_versions || {}, 8))}
        </div>`;
    }
    function aggregatePromptProvenance(runs) {
      const stageMaps = {
        query_prompt_versions: new Map(),
        response_prompt_versions: new Map(),
        ir_prompt_versions: new Map(),
      };
      const mixedRuns = [];
      const triples = new Map();
      for (const run of runs) {
        const triple = [];
        for (const stage of promptStageDefs) {
          const counts = run[stage.key] || {};
          addStructureCounts(stageMaps[stage.key], counts);
          const versionCount = promptVersionCount(counts);
          const dominant = dominantPromptVersion(counts);
          triple.push(dominant);
          if (versionCount > 1) {
            mixedRuns.push({run, stage: stage.label, versionCount, dominant, rows: Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0)});
          }
        }
        const key = triple.join(" -> ");
        if (!triples.has(key)) {
          triples.set(key, {key, runs: 0, genui: 0, scoreSum: 0, scoreWeight: 0});
        }
        const bucket = triples.get(key);
        const weight = Math.max(1, run.genui || 0);
        bucket.runs += 1;
        bucket.genui += run.genui || 0;
        const score = run.display_score ?? run.overall_score;
        if (score != null) {
          bucket.scoreSum += Number(score) * weight;
          bucket.scoreWeight += weight;
        }
      }
      return {
        stage_versions: {
          query_prompt_versions: topCounts(stageMaps.query_prompt_versions, 12),
          response_prompt_versions: topCounts(stageMaps.response_prompt_versions, 12),
          ir_prompt_versions: topCounts(stageMaps.ir_prompt_versions, 12),
        },
        mixed_runs: mixedRuns.sort((a, b) => (b.versionCount - a.versionCount) || String(a.run.run_id).localeCompare(String(b.run.run_id))).slice(0, 12),
        triples: [...triples.values()]
          .map(bucket => ({...bucket, avg_score: bucket.scoreWeight ? bucket.scoreSum / bucket.scoreWeight : null}))
          .sort((a, b) => (b.genui - a.genui) || a.key.localeCompare(b.key))
          .slice(0, 12),
      };
    }
    function renderPromptProvenance(runs) {
      const target = document.getElementById("promptProvenance");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const data = aggregatePromptProvenance(runs);
      const mixedRows = data.mixed_runs.map(item => `
        <tr>
          <td><b>${escapeHtml(item.run.run_id)}</b><br><span class="small">${escapeHtml(item.run.source_label)}</span><br><button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(item.run))}')">Details</button></td>
          <td>${escapeHtml(item.stage)}</td>
          <td>${fmt(item.versionCount)}</td>
          <td>${escapeHtml(item.dominant)}</td>
          <td>${fmt(item.rows)}</td>
        </tr>`).join("");
      const tripleRows = data.triples.map(item => `
        <tr>
          <td><span class="small">${escapeHtml(item.key)}</span></td>
          <td>${fmt(item.runs)}</td>
          <td>${fmt(item.genui)}</td>
          <td><span class="score ${scoreClass(item.avg_score)}">${scoreText(item.avg_score)}</span></td>
        </tr>`).join("");
      target.innerHTML = `
        <div class="dist-grid">
          ${renderStructureCountBlock("Stage 1 Query Prompts", data.stage_versions.query_prompt_versions)}
          ${renderStructureCountBlock("Stage 2 Response Prompts", data.stage_versions.response_prompt_versions)}
          ${renderStructureCountBlock("Stage 3 IR Prompts", data.stage_versions.ir_prompt_versions)}
        </div>
        <div class="two-col-panels wide-panel">
          <div>
            <h2>Prompt Combinations</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Stage 1 -> Stage 2 -> Stage 3</th><th>Runs</th><th>IR</th><th>Score</th></tr></thead>
                <tbody>${tripleRows || "<tr><td colspan='4'><span class='small'>No prompt combinations found.</span></td></tr>"}</tbody>
              </table>
            </div>
          </div>
          <div>
            <h2>Mixed Prompt Runs</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Run</th><th>Stage</th><th>Versions</th><th>Dominant</th><th>Rows</th></tr></thead>
                <tbody>${mixedRows || "<tr><td colspan='5'><span class='small'>No mixed prompt-version runs in current filters.</span></td></tr>"}</tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="small">Prompt provenance uses gen.prompt_version and related prompt/version fields from each stage. It helps identify mixed runs and compare full Stage 1 -> Stage 2 -> Stage 3 prompt lineages.</div>`;
    }
    function regressionGroupKey(run) {
      return [
        run.source_id || "",
        dominantModel(run.response_models),
        dominantModel(run.ir_models),
        dominantPromptVersion(run.ir_prompt_versions || run.ir_versions || {}),
      ].join("||");
    }
    function regressionGroupLabel(run) {
      return [
        run.source_label || "unknown source",
        `${dominantModel(run.response_models)} -> ${dominantModel(run.ir_models)}`,
        dominantPromptVersion(run.ir_prompt_versions || run.ir_versions || {}),
      ].join(" | ");
    }
    function scoreDeltaText(delta) {
      if (delta == null || Number.isNaN(Number(delta))) return "n/a";
      const value = Number(delta);
      return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
    }
    function deltaBadge(delta) {
      if (delta == null || Number.isNaN(Number(delta))) return "<span class='small'>n/a</span>";
      const value = Number(delta);
      const klass = value < -5 ? "error" : value < -2 ? "warn" : value >= 2 ? "ok" : "";
      return `<span class="badge ${klass}">${scoreDeltaText(value)}</span>`;
    }
    function buildRegressionWatch(runs) {
      const byGroup = new Map();
      for (const run of runs) {
        if ((run.genui || 0) <= 0) continue;
        const score = run.display_score ?? run.overall_score;
        if (score == null || Number.isNaN(Number(score))) continue;
        const key = regressionGroupKey(run);
        if (!byGroup.has(key)) byGroup.set(key, []);
        byGroup.get(key).push(run);
      }
      const comparisons = [];
      for (const groupRuns of byGroup.values()) {
        groupRuns.sort((a, b) => (timestampMs(b.updated_at) || 0) - (timestampMs(a.updated_at) || 0));
        if (groupRuns.length < 2) continue;
        const latest = groupRuns[0];
        const previous = groupRuns[1];
        const latestScore = Number(latest.display_score ?? latest.overall_score);
        const previousScore = Number(previous.display_score ?? previous.overall_score);
        comparisons.push({
          latest,
          previous,
          groupLabel: regressionGroupLabel(latest),
          latestScore,
          previousScore,
          delta: latestScore - previousScore,
          irDelta: Number(latest.genui || 0) - Number(previous.genui || 0),
        });
      }
      const regressions = comparisons
        .filter(item => item.delta < -1)
        .sort((a, b) => a.delta - b.delta || (timestampMs(b.latest.updated_at) || 0) - (timestampMs(a.latest.updated_at) || 0))
        .slice(0, 12);
      const improvements = comparisons
        .filter(item => item.delta > 1)
        .sort((a, b) => b.delta - a.delta || (timestampMs(b.latest.updated_at) || 0) - (timestampMs(a.latest.updated_at) || 0))
        .slice(0, 8);
      const latestWeak = [...runs]
        .filter(run => (run.genui || 0) > 0 && (run.display_score ?? run.overall_score) != null && Number(run.display_score ?? run.overall_score) < 75)
        .sort((a, b) => (timestampMs(b.updated_at) || 0) - (timestampMs(a.updated_at) || 0))
        .slice(0, 12);
      return {comparisons, regressions, improvements, latestWeak};
    }
    function renderRegressionRows(rows) {
      return rows.map(item => `
        <tr>
          <td>
            <b>${escapeHtml(item.latest.run_id)}</b><br>
            <span class="small">${escapeHtml(item.groupLabel)}</span><br>
            <button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(item.latest))}')">Latest</button>
            <button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(item.previous))}')">Previous</button>
          </td>
          <td><span class="score ${scoreClass(item.latestScore)}">${scoreText(item.latestScore)}</span><br><span class="small">${ageText(item.latest.updated_at)}</span></td>
          <td><span class="score ${scoreClass(item.previousScore)}">${scoreText(item.previousScore)}</span><br><span class="small">${ageText(item.previous.updated_at)}</span></td>
          <td>${deltaBadge(item.delta)}<br><span class="small">IR ${scoreDeltaText(item.irDelta)}</span></td>
          <td class="small">latest Q/R/IR ${fmt(item.latest.queries)} / ${fmt(item.latest.responses)} / ${fmt(item.latest.genui)}<br>prev Q/R/IR ${fmt(item.previous.queries)} / ${fmt(item.previous.responses)} / ${fmt(item.previous.genui)}</td>
        </tr>`).join("");
    }
    function renderLatestWeakRows(rows) {
      return rows.map(run => {
        const score = run.display_score ?? run.overall_score;
        const issues = [
          (run.response_backlog || 0) ? `missing R ${fmt(run.response_backlog)}` : "",
          (run.ir_backlog || 0) ? `missing IR ${fmt(run.ir_backlog)}` : "",
          qualityIssueCount(run) ? `quality ${fmt(qualityIssueCount(run))}` : "",
          integrityIssueCount(run) ? `integrity ${fmt(integrityIssueCount(run))}` : "",
          logIssueCount(run) ? `logs ${fmt(logIssueCount(run))}` : "",
        ].filter(Boolean);
        return `
          <tr>
            <td><b>${escapeHtml(run.run_id)}</b><br><span class="small">${escapeHtml(run.source_label)}</span><br><button class="mini-btn ghost-btn" onclick="selectRun('${encodeURIComponent(runKey(run))}')">Details</button></td>
            <td><span class="score ${scoreClass(score)}">${scoreText(score)}</span><br><span class="small">${ageText(run.updated_at)}</span></td>
            <td>${fmt(run.queries)} / ${fmt(run.responses)} / ${fmt(run.genui)}</td>
            <td class="small">${issues.join("<br>") || "low score"}</td>
          </tr>`;
      }).join("");
    }
    function renderRegressionWatch(runs) {
      const target = document.getElementById("regressionWatch");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const data = buildRegressionWatch(runs);
      const worstDrop = data.regressions[0];
      const bestGain = data.improvements[0];
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(data.comparisons.length)}</b><br><span class="small">comparable source/model/prompt groups</span></div>
          <div class="detail-box"><b>${fmt(data.regressions.length)}</b><br><span class="small">latest runs down >1 score point</span></div>
          <div class="detail-box"><b>${worstDrop ? scoreDeltaText(worstDrop.delta) : "n/a"}</b><br><span class="small">largest latest drop</span></div>
          <div class="detail-box"><b>${bestGain ? scoreDeltaText(bestGain.delta) : "n/a"}</b><br><span class="small">largest latest gain</span></div>
          <div class="detail-box"><b>${fmt(data.latestWeak.length)}</b><br><span class="small">recent sampled runs below 75</span></div>
        </div>
        <div class="two-col-panels wide-panel">
          <div>
            <h2>Latest Regressions</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Run / Lineage</th><th>Latest</th><th>Previous</th><th>Delta</th><th>Volume</th></tr></thead>
                <tbody>${renderRegressionRows(data.regressions) || "<tr><td colspan='5'><span class='small'>No latest score drops found for comparable groups.</span></td></tr>"}</tbody>
              </table>
            </div>
          </div>
          <div>
            <h2>Recent Weak Runs</h2>
            <div class="scroll">
              <table>
                <thead><tr><th>Run</th><th>Score</th><th>Q / R / IR</th><th>Signals</th></tr></thead>
                <tbody>${renderLatestWeakRows(data.latestWeak) || "<tr><td colspan='4'><span class='small'>No recent weak runs under the current filters.</span></td></tr>"}</tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="small">Comparisons group runs by source, dominant Stage 2 model, dominant Stage 3 model, and dominant Stage 3 prompt version. This is a lightweight regression signal; use run details to inspect exact samples.</div>`;
    }
    function addMapCount(map, key, count = 1) {
      const label = String(key || "unknown");
      map.set(label, (map.get(label) || 0) + Number(count || 0));
    }
    function aggregateIrVersionQuality(runs) {
      const f = filters();
      const buckets = new Map();
      for (const run of runs) {
        for (const [version, stats] of Object.entries(run.ir_version_stats || {})) {
          if (f.irVersion && version !== f.irVersion) continue;
          const dayRows = (stats.days || []).filter(day => !dateFilterActive(f) || dayInRange(day.day, f));
          const counts = dateFilterActive(f)
            ? dayRows.reduce((acc, day) => {
                acc.queries += day.queries || 0;
                acc.responses += day.responses || 0;
                acc.genui += day.genui || 0;
                acc.score_sum += day.score_sum || 0;
                acc.score_count += day.score_count || 0;
                return acc;
              }, {queries: 0, responses: 0, genui: 0, score_sum: 0, score_count: 0})
            : {
                queries: stats.queries || 0,
                responses: stats.responses || 0,
                genui: stats.genui || 0,
                score_sum: stats.score_sum || 0,
                score_count: stats.score_count || 0,
              };
          if (!counts.queries && !counts.responses && !counts.genui) continue;
          if (!buckets.has(version)) {
            buckets.set(version, {
              version,
              runs: 0,
              sourceLabels: new Set(),
              queries: 0,
              responses: 0,
              genui: 0,
              score_sum: 0,
              score_count: 0,
              first_day: null,
              latest_day: null,
              irModels: new Map(),
            });
          }
          const bucket = buckets.get(version);
          bucket.runs += 1;
          bucket.sourceLabels.add(run.source_label || run.source_id || "unknown");
          bucket.queries += counts.queries || 0;
          bucket.responses += counts.responses || 0;
          bucket.genui += counts.genui || 0;
          bucket.score_sum += counts.score_sum || 0;
          bucket.score_count += counts.score_count || 0;
          addMapCount(bucket.irModels, dominantModel(run.ir_models), counts.genui || 1);
          for (const day of dayRows) {
            if (!day.day) continue;
            bucket.first_day = bucket.first_day == null || day.day < bucket.first_day ? day.day : bucket.first_day;
            bucket.latest_day = bucket.latest_day == null || day.day > bucket.latest_day ? day.day : bucket.latest_day;
          }
        }
      }
      return [...buckets.values()]
        .map(bucket => ({
          ...bucket,
          source_count: bucket.sourceLabels.size,
          avg_score: bucket.score_count ? bucket.score_sum / bucket.score_count : null,
          top_models: topCounts(bucket.irModels, 3),
        }))
        .sort((a,b) => (b.genui - a.genui) || (b.avg_score ?? -1) - (a.avg_score ?? -1) || a.version.localeCompare(b.version));
    }
    function renderIrVersionQuality(runs) {
      const target = document.getElementById("irVersionQuality");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const rows = aggregateIrVersionQuality(runs);
      if (!rows.length) {
        target.innerHTML = "<span class='small'>No IR version data found for the current filters.</span>";
        return;
      }
      const scored = rows.filter(row => row.avg_score != null);
      const best = scored.slice().sort((a,b) => (b.avg_score - a.avg_score) || (b.genui - a.genui))[0];
      const weakest = scored.slice().sort((a,b) => (a.avg_score - b.avg_score) || (b.genui - a.genui))[0];
      const largest = rows.slice().sort((a,b) => b.genui - a.genui)[0];
      const totalIr = rows.reduce((sum, row) => sum + (row.genui || 0), 0);
      const tableRows = rows.map(row => `
        <tr>
          <td><b>${escapeHtml(row.version)}</b><br><span class="small">${fmt(row.runs)} runs | ${fmt(row.source_count)} sources</span></td>
          <td>${fmt(row.queries)} Q<br>${fmt(row.responses)} R<br>${fmt(row.genui)} IR</td>
          <td><span class="score ${scoreClass(row.avg_score)}">${scoreText(row.avg_score)}</span><br><span class="small">${fmt(row.score_count)} scored</span></td>
          <td class="small">${row.first_day || "n/a"}<br>to ${row.latest_day || "n/a"}</td>
          <td class="small">${row.top_models.map(([model, count]) => `${escapeHtml(model)} (${fmt(count)})`).join("<br>") || "n/a"}</td>
        </tr>`).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${fmt(rows.length)}</b><br><span class="small">IR prompt versions</span></div>
          <div class="detail-box"><b>${fmt(totalIr)}</b><br><span class="small">filtered IR rows</span></div>
          <div class="detail-box"><b>${best ? escapeHtml(best.version) : "n/a"}</b><br><span class="small">best score ${best ? scoreText(best.avg_score) : ""}</span></div>
          <div class="detail-box"><b>${weakest ? escapeHtml(weakest.version) : "n/a"}</b><br><span class="small">weakest score ${weakest ? scoreText(weakest.avg_score) : ""}</span></div>
          <div class="detail-box"><b>${largest ? escapeHtml(largest.version) : "n/a"}</b><br><span class="small">largest ${largest ? fmt(largest.genui) : "0"} IR</span></div>
        </div>
        <div class="scroll">
          <table>
            <thead><tr><th>IR Version</th><th>Counts</th><th>Avg Score</th><th>Date Span</th><th>Stage 3 Models</th></tr></thead>
            <tbody>${tableRows}</tbody>
          </table>
        </div>
        <div class="small">Version quality follows the current source/text/score/date filters and uses stored per-version score sums from genui.jsonl.</div>`;
    }
    function emptyUsageBucket(label = "") {
      return {
        label,
        count: 0,
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        latency_ms_sum: 0,
        latency_count: 0,
        cost_usd_sum: 0,
        cost_count: 0,
        error_count: 0,
      };
    }
    function addUsage(target, usage) {
      if (!usage) return;
      target.count += Number(usage.count || 0);
      target.input_tokens += Number(usage.input_tokens || 0);
      target.output_tokens += Number(usage.output_tokens || 0);
      target.total_tokens += Number(usage.total_tokens || 0);
      target.error_count += Number(usage.error_count || 0);
      if (usage.avg_latency_ms != null && Number(usage.latency_count || 0)) {
        target.latency_ms_sum += Number(usage.avg_latency_ms) * Number(usage.latency_count || 0);
        target.latency_count += Number(usage.latency_count || 0);
      }
      if (usage.cost_usd != null && Number(usage.cost_count || 0)) {
        target.cost_usd_sum += Number(usage.cost_usd);
        target.cost_count += Number(usage.cost_count || 0);
      }
    }
    function finalizeUsage(bucket) {
      return {
        ...bucket,
        avg_latency_ms: bucket.latency_count ? bucket.latency_ms_sum / bucket.latency_count : null,
        cost_usd: bucket.cost_count ? bucket.cost_usd_sum : null,
      };
    }
    function aggregateUsage(runs) {
      const stages = {
        queries: emptyUsageBucket("Stage 1 Queries"),
        responses: emptyUsageBucket("Stage 2 Responses"),
        ir: emptyUsageBucket("Stage 3 IR"),
      };
      const models = new Map();
      for (const run of runs) {
        for (const [stageKey, usageKey, stageLabel] of [
          ["queries", "query_usage", "Stage 1"],
          ["responses", "response_usage", "Stage 2"],
          ["ir", "ir_usage", "Stage 3"],
        ]) {
          const usage = run[usageKey] || {};
          addUsage(stages[stageKey], usage);
          for (const [model, modelUsage] of Object.entries(usage.models || {})) {
            const key = `${stageLabel}::${model}`;
            if (!models.has(key)) {
              models.set(key, {...emptyUsageBucket(`${stageLabel} ${model}`), stage: stageLabel, model});
            }
            addUsage(models.get(key), modelUsage);
          }
        }
      }
      return {
        stages: Object.values(stages).map(finalizeUsage),
        models: [...models.values()].map(finalizeUsage).sort((a, b) => (b.total_tokens - a.total_tokens) || a.label.localeCompare(b.label)),
      };
    }
    function renderUsagePanel(runs) {
      const target = document.getElementById("usagePanel");
      if (!runs.length) {
        target.innerHTML = "<span class='small'>No runs match current filters.</span>";
        return;
      }
      const usage = aggregateUsage(runs);
      const total = finalizeUsage(usage.stages.reduce((acc, stage) => {
        addUsage(acc, stage);
        return acc;
      }, emptyUsageBucket("Total")));
      const stageRows = usage.stages.map(stage => `
        <tr>
          <td><b>${stage.label}</b></td>
          <td>${fmt(stage.count)}</td>
          <td>${compactNumber(stage.input_tokens)} in<br>${compactNumber(stage.output_tokens)} out</td>
          <td>${compactNumber(stage.total_tokens)}</td>
          <td>${latencyText(stage.avg_latency_ms)}<br><span class="small">${fmt(stage.latency_count)} timed</span></td>
          <td>${costText(stage.cost_usd)}<br><span class="small">${fmt(stage.cost_count)} costed</span></td>
          <td>${fmt(stage.error_count)}</td>
        </tr>`).join("");
      const modelRows = usage.models.slice(0, 18).map(row => `
        <tr>
          <td><span class="badge">${row.stage}</span><br><b>${escapeHtml(row.model)}</b></td>
          <td>${fmt(row.count)}</td>
          <td>${compactNumber(row.input_tokens)} / ${compactNumber(row.output_tokens)}</td>
          <td>${compactNumber(row.total_tokens)}</td>
          <td>${latencyText(row.avg_latency_ms)}</td>
          <td>${costText(row.cost_usd)}</td>
          <td>${fmt(row.error_count)}</td>
        </tr>`).join("");
      target.innerHTML = `
        <div class="detail-grid">
          <div class="detail-box"><b>${compactNumber(total.total_tokens)}</b><br><span class="small">sampled total tokens</span></div>
          <div class="detail-box"><b>${compactNumber(total.input_tokens)}</b><br><span class="small">input tokens</span></div>
          <div class="detail-box"><b>${compactNumber(total.output_tokens)}</b><br><span class="small">output tokens</span></div>
          <div class="detail-box"><b>${latencyText(total.avg_latency_ms)}</b><br><span class="small">avg latency</span></div>
          <div class="detail-box"><b>${costText(total.cost_usd)}</b><br><span class="small">known cost</span></div>
          <div class="detail-box"><b>${fmt(total.error_count)}</b><br><span class="small">generation errors</span></div>
        </div>
        <h2>By Stage</h2>
        <div class="scroll">
          <table>
            <thead><tr><th>Stage</th><th>Rows</th><th>Input / Output</th><th>Total Tokens</th><th>Latency</th><th>Cost</th><th>Errors</th></tr></thead>
            <tbody>${stageRows}</tbody>
          </table>
        </div>
        <h2>Top Stage/Model Usage</h2>
        <div class="scroll">
          <table>
            <thead><tr><th>Model</th><th>Rows</th><th>Input / Output</th><th>Total Tokens</th><th>Latency</th><th>Cost</th><th>Errors</th></tr></thead>
            <tbody>${modelRows || "<tr><td colspan='7'><span class='small'>No usage metadata found.</span></td></tr>"}</tbody>
          </table>
        </div>
        <div class="small">Usage is aggregated from available gen metadata in sampled JSONL rows. Missing token/cost fields are shown as zero or n/a rather than estimated.</div>
      `;
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
      renderActionItems(runs, sourceStats, current.last_sync);
      renderRunLogs(runs);
      renderSources(sourceStats);
      renderRuns(page.rows, page);
      renderRunDetails(runs);
      renderBacklog(sourceStats);
      renderQualityAlerts(runs);
      renderCompletionFunnel(runs, sourceStats);
      renderDataIntegrity(runs);
      renderContentDuplicates(runs);
      renderMetricsOverview(runs);
      renderMetricRisk(runs);
      renderTrainingReadiness(runs);
      renderIrStructure(runs);
      renderMediaHealth(runs);
      renderIntentQuality(runs);
      renderThroughputEta(runs, sourceStats);
      renderStorageArtifacts(runs, sourceStats);
      renderWorstSamples(runs);
      renderModelComparison(runs);
      renderPromptProvenance(runs);
      renderRegressionWatch(runs);
      renderIrVersionQuality(runs);
      renderUsagePanel(runs);
      renderDistribution(runs);
      renderTrend(runs);
      renderDays(runs);
    }
    document.getElementById("syncBtn").onclick = () => syncSources().catch(e => setStatus(`Sync failed: ${e.message}`));
    document.getElementById("stopSyncBtn").onclick = () => stopSync().catch(e => setStatus(`Stop failed: ${e.message}`));
    document.getElementById("syncSourceProgress").addEventListener("click", event => {
      const button = event.target.closest(".source-resync-btn");
      if (!button) return;
      const sourceId = button.getAttribute("data-source-id") || "";
      if (!sourceId) return;
      syncSources(sourceId).catch(e => setStatus(`Source sync failed: ${e.message}`));
    });
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
    document.getElementById("maxParallelSources").onchange = event => { event.target.dataset.userEdited = "1"; };
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
        self.stop_event = threading.Event()
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
            "completed_sources": 0,
            "active_sources": 0,
            "max_parallel_sources": DEFAULT_MAX_PARALLEL_SOURCES,
            "current_file": "",
            "listed": 0,
            "total": 0,
            "processed": 0,
            "changed": 0,
            "copied": 0,
            "skipped": 0,
            "error_count": 0,
            "warning_count": 0,
            "stop_requested": False,
            "cancelled": False,
            "stopped_at": "",
            "message": "Idle",
            "messages": [],
            "sources": {},
        }

    def update_sync_status(self, update: dict[str, Any]) -> None:
        with self.status_lock:
            if update.get("phase") == "starting":
                self.sync_status = self.new_sync_status()
                self.sync_status["started_at"] = utc_now()
            status = dict(self.sync_status)
            status.update(update)
            status["updated_at"] = utc_now()
            source_id = str(update.get("source_id") or "").strip()
            if source_id:
                sources = dict(status.get("sources") or {})
                source_status = dict(sources.get(source_id) or {})
                source_status.update({k: v for k, v in update.items() if k not in {"sources", "messages"}})
                source_status["updated_at"] = status["updated_at"]
                phase = str(source_status.get("phase") or "")
                source_status["done"] = phase in {"source_done", "source_error", "source_complete", "source_cancelled"}
                source_status["running"] = bool(status.get("running")) and not bool(source_status.get("done"))
                sources[source_id] = source_status
                status["sources"] = sources
                source_values = list(sources.values())
                if source_values:
                    status["listed"] = sum(int(item.get("listed") or 0) for item in source_values)
                    status["total"] = sum(int(item.get("total") or item.get("listed") or 0) for item in source_values)
                    status["processed"] = sum(int(item.get("processed") or 0) for item in source_values)
                    status["changed"] = sum(int(item.get("changed") or 0) for item in source_values)
                    status["copied"] = sum(int(item.get("copied") or 0) for item in source_values)
                    status["skipped"] = sum(int(item.get("skipped") or 0) for item in source_values)
                    status["error_count"] = sum(int(item.get("error_count") or 0) for item in source_values)
                    status["warning_count"] = sum(int(item.get("warning_count") or 0) for item in source_values)
                    status["completed_sources"] = sum(1 for item in source_values if item.get("done"))
                    status["active_sources"] = sum(
                        1
                        for item in source_values
                        if not item.get("done") and str(item.get("phase") or "") not in {"queued", ""}
                    )
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

    def sync(self, max_parallel_sources: int | None = None, source_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            self.stop_event.clear()
            message = f"Starting sync for {source_id}" if source_id else "Starting sync"
            self.update_sync_status({"running": True, "phase": "starting", "message": message, "target_source_id": source_id or ""})
            try:
                return run_sync(
                    self.config(),
                    self.mirror_dir,
                    progress=self.update_sync_status,
                    max_parallel_sources=max_parallel_sources,
                    cancel_event=self.stop_event,
                    source_id=source_id,
                )
            except Exception as exc:
                self.stop_event.clear()
                self.update_sync_status({"running": False, "phase": "failed", "stop_requested": False, "message": f"Sync failed: {exc}"})
                raise

    def stop_sync(self) -> dict[str, Any]:
        status = self.status()
        if not status.get("running"):
            self.stop_event.clear()
            self.update_sync_status(
                {
                    "running": False,
                    "phase": "idle",
                    "stop_requested": False,
                    "message": "No active sync to stop",
                }
            )
            return self.status()
        self.stop_event.set()
        self.update_sync_status(
            {
                "running": True,
                "phase": "stopping",
                "stop_requested": True,
                "message": "Stop requested. Waiting for active copy operation to finish.",
            }
        )
        return self.status()

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
                    payload = self.read_json_body()
                    max_parallel_raw = payload.get("max_parallel_sources")
                    max_parallel_sources = (
                        positive_int(max_parallel_raw, DEFAULT_MAX_PARALLEL_SOURCES)
                        if max_parallel_raw not in (None, "")
                        else None
                    )
                    source_id = str(payload.get("source_id") or "").strip() or None
                    self.send_json(server_state.sync(max_parallel_sources=max_parallel_sources, source_id=source_id))
                except Exception as exc:
                    self.send_json({"error": str(exc)}, status=500)
                return
            if parsed.path == "/api/sync/stop":
                try:
                    self.send_json(server_state.stop_sync())
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


def emit_json_stdout(payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    try:
        sys.stdout.buffer.write(text.encode("utf-8"))
    except AttributeError:
        sys.stdout.write(text)


def main() -> int:
    args = parse_args()
    config_path = resolve_dataset_path(args.config, ROOT)
    mirror_dir = resolve_dataset_path(args.mirror_dir, ROOT)
    state = DashboardServer(config_path=config_path, mirror_dir=mirror_dir)

    if args.sync_once:
        emit_json_stdout(state.sync())
        return 0
    if args.summary_once:
        emit_json_stdout(state.summary())
        return 0
    if args.sync_on_start:
        emit_json_stdout(state.sync())

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
