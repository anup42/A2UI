import argparse
import json
import mimetypes
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_RUNS_DIR = ROOT / "data" / "runs"
DEFAULT_RENDERER_DIR = ROOT / "renderer"


def _safe_path(base: Path, rel: str) -> Path:
    rel_path = Path(rel.lstrip("/")).resolve()
    base_resolved = base.resolve()
    try:
        candidate = (base_resolved / rel_path).resolve()
    except Exception:
        candidate = base_resolved
    if base_resolved == candidate or base_resolved in candidate.parents:
        return candidate
    raise ValueError("invalid path")


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                yield {"_parse_error": str(exc), "_raw": line}


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _resolve_genui_jsonl(run_dir: Path) -> Path:
    """Return the GenUICraft JSONL path for a run (supports legacy filenames)."""
    genui = run_dir / "genui.jsonl"
    if genui.exists():
        return genui
    legacy = run_dir / "a2ui.jsonl"
    if legacy.exists():
        return legacy
    return genui


def _slice_jsonl(path: Path, offset: int, limit: int, search: str | None, field: str | None):
    results = []
    if not path.exists():
        return results
    query = search.lower() if search else None
    index = 0
    for row in _iter_jsonl(path):
        if query:
            haystack = ""
            if field and isinstance(row, dict):
                value = row.get(field, "")
                haystack = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            else:
                haystack = json.dumps(row, ensure_ascii=False)
            if query not in haystack.lower():
                continue
        if index < offset:
            index += 1
            continue
        results.append(row)
        index += 1
        if len(results) >= limit:
            break
    return results


class DatasetHandler(BaseHTTPRequestHandler):
    server_version = "DatasetViz/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        mime, _ = mimetypes.guess_type(str(path))
        content_type = mime or "application/octet-stream"
        body = path.read_bytes()
        self._send(200, body, content_type)

    def _handle_api_runs(self, runs_dir: Path) -> None:
        runs = []
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                queries = run_dir / "queries.jsonl"
                responses = run_dir / "responses.jsonl"
                genui = _resolve_genui_jsonl(run_dir)
                aggregates = run_dir / "aggregates.json"
                render = run_dir / "render.jsonl"
                runs.append(
                    {
                        "run_id": run_dir.name,
                        "queries": _count_lines(queries),
                        "responses": _count_lines(responses),
                        "genui": _count_lines(genui),
                        "has_aggregates": aggregates.exists(),
                        "has_render": render.exists(),
                        "updated_at": datetime.utcfromtimestamp(run_dir.stat().st_mtime).isoformat() + "Z",
                    }
                )
        self._send_json({"runs": runs})

    def _handle_api_summary(self, run_dir: Path) -> None:
        payload = {
            "run_id": run_dir.name,
            "queries": _count_lines(run_dir / "queries.jsonl"),
            "responses": _count_lines(run_dir / "responses.jsonl"),
            "genui": _count_lines(_resolve_genui_jsonl(run_dir)),
        }
        aggregates = run_dir / "aggregates.json"
        if aggregates.exists():
            try:
                payload["aggregates"] = json.loads(aggregates.read_text(encoding="utf-8"))
            except Exception:
                payload["aggregates"] = None
        self._send_json(payload)

    def _handle_api_jsonl(self, run_dir: Path, name: str, query: dict) -> None:
        if name == "a2ui":
            name = "genui"
        path = run_dir / f"{name}.jsonl"
        offset = int(query.get("offset", ["0"])[0])
        limit = int(query.get("limit", ["50"])[0])
        search = query.get("search", [""])[0].strip() or None
        field = query.get("field", [""])[0].strip() or None
        items = _slice_jsonl(path, offset, limit, search, field)
        total = _count_lines(path)
        self._send_json({"items": items, "total": total, "offset": offset, "limit": limit})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        runs_dir = self.server.runs_dir
        renderer_dir = self.server.renderer_dir

        if path in ("/", ""):
            self._serve_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            self._serve_file(_safe_path(STATIC_DIR, rel))
            return
        if path.startswith("/renderer/"):
            rel = path[len("/renderer/") :]
            self._serve_file(_safe_path(renderer_dir, rel))
            return
        if path.startswith("/runs/"):
            rel = path[len("/runs/") :]
            self._serve_file(_safe_path(runs_dir, rel))
            return
        if path == "/api/runs":
            self._handle_api_runs(runs_dir)
            return
        if path.startswith("/api/run/"):
            parts = path.split("/")
            if len(parts) < 4:
                self._send_json({"error": "invalid path"}, status=400)
                return
            run_id = parts[3]
            run_dir = runs_dir / run_id
            if not run_dir.exists():
                self._send_json({"error": "run not found"}, status=404)
                return
            if len(parts) == 4:
                self._handle_api_summary(run_dir)
                return
            if parts[4] == "summary":
                self._handle_api_summary(run_dir)
                return
            if parts[4] in ("queries", "responses", "genui", "a2ui"):
                self._handle_api_jsonl(run_dir, parts[4], query)
                return
        self._send_json({"error": "not found"}, status=404)


class DatasetServer(ThreadingHTTPServer):
    def __init__(self, host: str, port: int, runs_dir: Path, renderer_dir: Path):
        super().__init__((host, port), DatasetHandler)
        self.runs_dir = runs_dir
        self.renderer_dir = renderer_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--runs-dir", type=str, default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--renderer-dir", type=str, default=str(DEFAULT_RENDERER_DIR))
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    renderer_dir = Path(args.renderer_dir).resolve()
    server = DatasetServer(args.host, args.port, runs_dir, renderer_dir)
    print(f"Dataset visualizer running on http://{args.host}:{args.port}")
    print(f"Runs dir: {runs_dir}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
