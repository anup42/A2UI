#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dataset_dashboard.sources.json"
DEFAULT_EXAMPLE_CONFIG = ROOT / "configs" / "dataset_dashboard.sources.example.json"
DEFAULT_MIRROR_DIR = ROOT / "data" / "dashboard_mirror"
MANIFEST_NAME = "sync_manifest.json"


@dataclass(frozen=True)
class FileEntry:
    rel: str
    size: int
    mtime: float


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


def should_include(rel: str, include_globs: list[str], exclude_globs: list[str]) -> bool:
    rel = rel.replace("\\", "/")
    if exclude_globs and match_any(rel, exclude_globs):
        return False
    return not include_globs or match_any(rel, include_globs)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def ssh_password(source: dict[str, Any]) -> str:
    return str(source.get("password") or "").strip()


def ssh_transport_mode(source: dict[str, Any], copy: bool = False) -> str:
    if not ssh_password(source):
        return "openssh"
    if shutil.which("sshpass"):
        return "sshpass"
    if copy:
        if shutil.which("pscp"):
            return "putty"
    elif shutil.which("plink"):
        return "putty"
    need = "sshpass or PuTTY pscp" if copy else "sshpass or PuTTY plink"
    raise RuntimeError(
        f"SSH source {source.get('id')} uses password auth, but {need} was not found. "
        "Install one of those tools or use identity_file key auth."
    )


def ssh_target(source: dict[str, Any]) -> str:
    host = str(source.get("host") or "").strip()
    if not host:
        raise ValueError(f"SSH source {source.get('id')} missing host")
    user = str(source.get("user") or "").strip()
    return f"{user}@{host}" if user else host


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


def source_enabled(source: dict[str, Any]) -> bool:
    return bool(source.get("enabled", True))


def source_label(source: dict[str, Any]) -> str:
    return str(source.get("label") or source.get("id") or "source")


def source_local_root(source: dict[str, Any], mirror_dir: Path) -> Path:
    source_id = safe_source_id(str(source.get("id") or source_label(source)))
    if source.get("type") == "local" and source.get("mirror_local_in_place", False):
        return resolve_dataset_path(str(source.get("path") or ""), ROOT)
    return mirror_dir / source_id


def list_local_files(root: Path, include_globs: list[str], exclude_globs: list[str]) -> list[FileEntry]:
    if not root.exists():
        return []
    entries: list[FileEntry] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = rel_posix(path, root)
        if not should_include(rel, include_globs, exclude_globs):
            continue
        stat = path.stat()
        entries.append(FileEntry(rel=rel, size=stat.st_size, mtime=stat.st_mtime))
    return entries


def ssh_base_command(source: dict[str, Any]) -> list[str]:
    target = ssh_target(source)
    mode = ssh_transport_mode(source, copy=False)
    password = ssh_password(source)
    if mode == "sshpass":
        cmd = ["sshpass", "-p", password, "ssh"]
    elif mode == "putty":
        cmd = ["plink", "-batch", "-pw", password]
    else:
        cmd = ["ssh"]
    port = source.get("port")
    if port and mode == "putty":
        cmd += ["-P", str(port)]
    elif port:
        cmd += ["-p", str(port)]
    identity = str(source.get("identity_file") or "").strip()
    if identity:
        cmd += ["-i", os.path.expandvars(os.path.expanduser(identity))]
    for option in source.get("ssh_options", []) or []:
        cmd += ["-o", str(option)]
    cmd.append(target)
    return cmd


def scp_base_command(source: dict[str, Any]) -> list[str]:
    mode = ssh_transport_mode(source, copy=True)
    password = ssh_password(source)
    if mode == "sshpass":
        cmd = ["sshpass", "-p", password, "scp", "-p"]
    elif mode == "putty":
        cmd = ["pscp", "-batch", "-pw", password, "-p"]
    else:
        cmd = ["scp", "-p"]
    port = source.get("port")
    if port:
        cmd += ["-P", str(port)]
    identity = str(source.get("identity_file") or "").strip()
    if identity:
        cmd += ["-i", os.path.expandvars(os.path.expanduser(identity))]
    for option in source.get("ssh_options", []) or []:
        cmd += ["-o", str(option)]
    return cmd


def list_ssh_files(source: dict[str, Any], include_globs: list[str], exclude_globs: list[str]) -> list[FileEntry]:
    remote_root = str(source.get("path") or "").rstrip("/")
    if not remote_root:
        raise ValueError(f"SSH source {source.get('id')} missing path")
    py = r"""
import os, sys, json, fnmatch
root=sys.argv[1]
include=json.loads(sys.argv[2])
exclude=json.loads(sys.argv[3])
def match_any(rel, patterns):
    return any(fnmatch.fnmatch(rel, p) for p in patterns)
for base, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in {'.git','__pycache__'}]
    for name in files:
        path=os.path.join(base,name)
        rel=os.path.relpath(path, root).replace(os.sep, '/')
        if exclude and match_any(rel, exclude):
            continue
        if include and not match_any(rel, include):
            continue
        try:
            st=os.stat(path)
        except OSError:
            continue
        print(json.dumps({'rel': rel, 'size': st.st_size, 'mtime': st.st_mtime}, separators=(',',':')))
"""
    cmd = ssh_base_command(source) + [
        "python3",
        "-c",
        py,
        remote_root,
        json.dumps(include_globs),
        json.dumps(exclude_globs),
    ]
    result = run_command(cmd, timeout=int(source.get("list_timeout_sec", 300)))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ssh list failed")
    entries: list[FileEntry] = []
    for line in result.stdout.splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        entries.append(FileEntry(rel=str(obj["rel"]), size=int(obj["size"]), mtime=float(obj["mtime"])))
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


def copy_local_file(source_root: Path, dest_root: Path, rel: str) -> None:
    src = source_root / rel
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_ssh_file(source: dict[str, Any], dest_root: Path, rel: str) -> None:
    remote_root = str(source.get("path") or "").rstrip("/")
    target = ssh_target(source)
    remote_path = str(PurePosixPath(remote_root) / PurePosixPath(rel))
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if ssh_transport_mode(source, copy=True) == "putty":
        remote_spec = f"{target}:{remote_path}"
    else:
        remote_spec = f"{target}:{shell_quote(remote_path)}"
    cmd = scp_base_command(source) + [remote_spec, str(dest)]
    result = run_command(cmd, timeout=int(source.get("copy_timeout_sec", 300)))
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"scp failed for {rel}")


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


def sync_source(
    source: dict[str, Any],
    mirror_dir: Path,
    include_globs: list[str],
    exclude_globs: list[str],
) -> dict[str, Any]:
    source_id = safe_source_id(str(source.get("id") or source_label(source)))
    source_type = str(source.get("type") or "local").lower()
    dest_root = source_local_root(source, mirror_dir)
    manifest_path = mirror_dir / source_id / f".{MANIFEST_NAME}"
    manifest = load_json(manifest_path, {"files": {}})
    previous: dict[str, Any] = manifest.get("files", {}) if isinstance(manifest, dict) else {}

    if source_type == "local":
        source_root = resolve_dataset_path(str(source.get("path") or ""), ROOT)
        entries = list_local_files(source_root, include_globs, exclude_globs)
    elif source_type == "ssh":
        source_root = None
        entries = list_ssh_files(source, include_globs, exclude_globs)
    elif source_type == "command":
        source_root = None
        entries = list_command_files(source, include_globs, exclude_globs)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    copied = 0
    skipped = 0
    errors: list[str] = []
    current_files: dict[str, Any] = {}
    for entry in entries:
        signature = {"size": entry.size, "mtime": round(entry.mtime, 6)}
        current_files[entry.rel] = signature
        if previous.get(entry.rel) == signature and (dest_root / entry.rel).exists():
            skipped += 1
            continue
        try:
            if source_type == "local":
                assert source_root is not None
                copy_local_file(source_root, dest_root, entry.rel)
            elif source_type == "ssh":
                copy_ssh_file(source, dest_root, entry.rel)
            else:
                copy_command_file(source, dest_root, entry.rel)
            copied += 1
        except Exception as exc:
            errors.append(f"{entry.rel}: {exc}")

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


def run_sync(config: dict[str, Any], mirror_dir: Path) -> dict[str, Any]:
    include_globs = list(config.get("include_globs") or [])
    exclude_globs = list(config.get("exclude_globs") or [])
    results = []
    for source in config.get("sources") or []:
        if not isinstance(source, dict) or not source_enabled(source):
            continue
        try:
            results.append(sync_source(source, mirror_dir, include_globs, exclude_globs))
        except Exception as exc:
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


def scan_run(source_id: str, source_label_text: str, run_dir: Path) -> dict[str, Any]:
    queries = count_jsonl(run_dir / "queries.jsonl")
    responses = count_jsonl(run_dir / "responses.jsonl")
    genui = count_jsonl(run_dir / "genui.jsonl")
    assets = len([p for p in (run_dir / "assets").rglob("*") if p.is_file()]) if (run_dir / "assets").exists() else 0
    screenshots = 0
    for folder in ("android_device_rendered", "rendered", "rendered_lit"):
        path = run_dir / folder
        if path.exists():
            screenshots += len(list(path.rglob("*.png")))
    genui_rows = read_first_jsonl(run_dir / "genui.jsonl", limit=5000)
    response_rows = read_first_jsonl(run_dir / "responses.jsonl", limit=5000)
    query_rows = read_first_jsonl(run_dir / "queries.jsonl", limit=5000)
    intents: dict[str, int] = {}
    for row in query_rows + response_rows + genui_rows:
        intent = str(row.get("intent") or row.get("intent_bucket") or "").strip()
        if intent:
            intents[intent] = intents.get(intent, 0) + 1
    mtime = max((p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()), default=run_dir.stat().st_mtime)
    return {
        "source_id": source_id,
        "source_label": source_label_text,
        "run_id": run_dir.name,
        "path": str(run_dir),
        "queries": queries,
        "responses": responses,
        "genui": genui,
        "assets": assets,
        "screenshots": screenshots,
        "overall_score": run_score(run_dir),
        "updated_at": datetime.fromtimestamp(mtime, timezone.utc).isoformat(),
        "query_models": collect_model_counts(query_rows),
        "response_models": collect_model_counts(response_rows),
        "ir_models": collect_model_counts(genui_rows),
        "intents": dict(sorted(intents.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
    }


def scan_all(config: dict[str, Any], mirror_dir: Path) -> dict[str, Any]:
    sources = []
    runs = []
    # Always show this checkout even if no config exists.
    effective_sources: list[dict[str, Any]] = [
        {
            "id": "_local_checkout",
            "label": "Local checkout",
            "type": "local",
            "path": "data/runs",
            "mirror_local_in_place": True,
            "enabled": True,
        }
    ]
    effective_sources.extend(source for source in config.get("sources") or [] if isinstance(source, dict))

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
        sources.append(
            {
                "source_id": source_id,
                "source_label": source_label(source),
                "type": source.get("type", "local"),
                "local_path": str(local_root),
                "run_count": len(source_runs),
                "queries": sum(r["queries"] for r in source_runs),
                "responses": sum(r["responses"] for r in source_runs),
                "genui": sum(r["genui"] for r in source_runs),
                "assets": sum(r["assets"] for r in source_runs),
                "screenshots": sum(r["screenshots"] for r in source_runs),
                "avg_score": (
                    sum(r["overall_score"] for r in source_runs if isinstance(r["overall_score"], (int, float)))
                    / max(1, len([r for r in source_runs if isinstance(r["overall_score"], (int, float))]))
                    if source_runs
                    else None
                ),
            }
        )

    totals = {
        "sources": len(sources),
        "runs": len(runs),
        "queries": sum(r["queries"] for r in runs),
        "responses": sum(r["responses"] for r in runs),
        "genui": sum(r["genui"] for r in runs),
        "assets": sum(r["assets"] for r in runs),
        "screenshots": sum(r["screenshots"] for r in runs),
    }
    latest_sync = load_json(mirror_dir / "last_sync.json", None)
    return {
        "generated_at": utc_now(),
        "totals": totals,
        "sources": sorted(sources, key=lambda s: s["source_label"].lower()),
        "runs": sorted(runs, key=lambda r: (r["source_label"].lower(), r["run_id"].lower())),
        "last_sync": latest_sync,
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
    .score { font-weight: 850; }
    .score.good { color: var(--good); }
    .score.warn { color: var(--warn); }
    .small { color: var(--muted); font-size: 12px; }
    .scroll { overflow:auto; max-height: calc(100vh - 260px); }
    .source-card { border:1px solid var(--line); border-radius:18px; padding: 14px; margin-bottom: 12px; background: rgba(255,255,255,.54); }
    .source-card strong { display:block; margin-bottom:6px; }
    .kv { display:grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 8px; margin-top: 10px; }
    .kv div { background: rgba(255,255,255,.55); border-radius: 12px; padding: 9px; }
    .status { min-height: 20px; color: var(--muted); font-size: 13px; }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } header, main { padding-left:18px; padding-right:18px; } }
  </style>
</head>
<body>
  <header>
    <h1>GenUICraft Dataset Dashboard</h1>
    <p class="sub">Sync generated dataset runs from local folders, SSH servers, or custom copy commands. The mirror only downloads files whose source size/mtime changed.</p>
    <div class="toolbar">
      <button id="syncBtn">Sync sources</button>
      <button id="refreshBtn">Refresh scan</button>
      <input id="filter" placeholder="Filter run/source/model..." />
      <select id="scoreFilter">
        <option value="">All scores</option>
        <option value="80">Score >= 80</option>
        <option value="70">Score >= 70</option>
        <option value="60">Score >= 60</option>
      </select>
      <span class="status" id="status"></span>
    </div>
  </header>
  <main>
    <section class="stats" id="stats"></section>
    <section class="grid">
      <aside class="panel">
        <h2>Sources</h2>
        <div id="sources"></div>
      </aside>
      <section class="panel">
        <h2>Runs</h2>
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
      </section>
    </section>
  </main>
  <script>
    let current = null;
    const fmt = n => (n ?? 0).toLocaleString();
    const scoreClass = s => s == null ? "" : s >= 75 ? "good" : s >= 60 ? "warn" : "";
    const scoreText = s => s == null ? "n/a" : Number(s).toFixed(2);
    const modelText = obj => {
      const entries = Object.entries(obj || {}).slice(0, 3);
      return entries.length ? entries.map(([k,v]) => `${k} (${v})`).join("<br>") : "<span class='small'>n/a</span>";
    };
    function setStatus(text) { document.getElementById("status").textContent = text || ""; }
    async function loadSummary() {
      setStatus("Loading...");
      const res = await fetch("/api/summary");
      current = await res.json();
      render();
      setStatus(`Loaded ${new Date(current.generated_at).toLocaleString()}`);
    }
    async function syncSources() {
      setStatus("Syncing sources...");
      const res = await fetch("/api/sync", {method: "POST"});
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "sync failed");
      await loadSummary();
      const copied = (payload.results || []).reduce((a,r) => a + (r.copied || 0), 0);
      setStatus(`Sync complete. Copied ${copied} changed files.`);
    }
    function renderStats(t) {
      const stats = [
        ["Sources", t.sources],
        ["Runs", t.runs],
        ["Queries", t.queries],
        ["Responses", t.responses],
        ["IR records", t.genui],
        ["Assets", t.assets],
        ["Screenshots", t.screenshots],
      ];
      document.getElementById("stats").innerHTML = stats.map(([k,v]) => `<div class="stat"><div class="v">${fmt(v)}</div><div class="k">${k}</div></div>`).join("");
    }
    function renderSources(sources) {
      document.getElementById("sources").innerHTML = sources.map(s => `
        <div class="source-card">
          <strong>${s.source_label}</strong>
          <span class="badge">${s.type}</span>
          <div class="small">${s.local_path}</div>
          <div class="kv">
            <div><b>${fmt(s.run_count)}</b><br><span class="small">runs</span></div>
            <div><b>${fmt(s.genui)}</b><br><span class="small">IR</span></div>
            <div><b>${scoreText(s.avg_score)}</b><br><span class="small">avg score</span></div>
            <div><b>${fmt(s.screenshots)}</b><br><span class="small">screenshots</span></div>
          </div>
        </div>`).join("");
    }
    function renderRuns(runs) {
      const q = document.getElementById("filter").value.toLowerCase().trim();
      const minScore = Number(document.getElementById("scoreFilter").value || "0");
      const filtered = runs.filter(r => {
        const hay = JSON.stringify([r.source_label, r.run_id, r.query_models, r.response_models, r.ir_models, r.intents]).toLowerCase();
        if (q && !hay.includes(q)) return false;
        if (minScore && (r.overall_score == null || r.overall_score < minScore)) return false;
        return true;
      });
      document.getElementById("runs").innerHTML = filtered.map(r => `
        <tr>
          <td><b>${r.run_id}</b><br><span class="small">${r.source_label}</span><br><span class="small">${r.path}</span></td>
          <td>
            Q ${fmt(r.queries)}<br>R ${fmt(r.responses)}<br>IR ${fmt(r.genui)}<br>
            <span class="small">assets ${fmt(r.assets)} | shots ${fmt(r.screenshots)}</span>
          </td>
          <td><span class="score ${scoreClass(r.overall_score)}">${scoreText(r.overall_score)}</span></td>
          <td>
            <span class="small">Stage 2</span><br>${modelText(r.response_models)}
            <br><span class="small">Stage 3</span><br>${modelText(r.ir_models)}
          </td>
          <td><span class="small">${new Date(r.updated_at).toLocaleString()}</span></td>
        </tr>`).join("");
    }
    function render() {
      if (!current) return;
      renderStats(current.totals || {});
      renderSources(current.sources || []);
      renderRuns(current.runs || []);
    }
    document.getElementById("syncBtn").onclick = () => syncSources().catch(e => setStatus(`Sync failed: ${e.message}`));
    document.getElementById("refreshBtn").onclick = () => loadSummary().catch(e => setStatus(`Refresh failed: ${e.message}`));
    document.getElementById("filter").oninput = render;
    document.getElementById("scoreFilter").onchange = render;
    loadSummary().catch(e => setStatus(`Load failed: ${e.message}`));
  </script>
</body>
</html>
"""


class DashboardServer:
    def __init__(self, config_path: Path, mirror_dir: Path):
        self.config_path = config_path
        self.mirror_dir = mirror_dir
        self.lock = threading.Lock()

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
            return run_sync(self.config(), self.mirror_dir)


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
