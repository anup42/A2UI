"""Loopback-only GenUI dashboard; archive reads never alter sealed evidence."""
from __future__ import annotations

import hmac
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from pipeline import chatgpt_stage3
from pipeline.genui_codex_client import CodexClient, validate_auth_location
from pipeline.v11_regeneration_io import checked, digest, file_hash, safe_path

ASSETS = Path(__file__).resolve().parents[2] / "dashboard/genui"


def _read(directory, name, *, sealed=False, rows=False):
    path = safe_path(directory / name)
    if directory not in path.parents:
        raise ValueError("Evidence must stay inside the selected run directory")
    if not path.exists():
        return [] if rows else {}
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Dashboard evidence file exceeds 128 MiB")
    if sealed:
        return checked(path)
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()] if rows else json.loads(text)


class Evidence:
    def __init__(self, directory, rescore_dir=None):
        self.directory = safe_path(directory)
        self.rescore = safe_path(rescore_dir) if rescore_dir else None
        self._cache, self._lock = {}, threading.RLock()

    def read(self, name, **kwargs):
        path = safe_path(self.directory / name)
        stamp = (path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None
        key = (name, tuple(sorted(kwargs.items())))
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] == stamp:
                return cached[1]
            value = _read(self.directory, name, **kwargs)
            if stamp and name not in {"contract.json", "collection.json", "chatgpt_collection.json"}:
                contract = self.read("contract.json", sealed=True)
                expected = dict(contract.get("files", {}))
                for receipt_name in ("collection.json", "chatgpt_collection.json"):
                    receipt = self.read(receipt_name, sealed=True)
                    if receipt:
                        if receipt.get("contract_sha256") != digest(contract):
                            raise ValueError("Collection belongs to a different prepared run")
                        expected.update(receipt.get("files", {}))
                if name in expected and file_hash(path) != expected[name]:
                    raise ValueError("Dashboard evidence hash changed: " + name)
            self._cache[key] = (stamp, value)
            return value

    def samples(self):
        selected = self.read("selected.jsonl", rows=True)
        contract = self.read("contract.json", sealed=True)
        mapping = {value: key for key, value in contract.get("bindings", {}).items()}
        assessments = {r["custom_id"]: r for r in self.read("assessments.jsonl", rows=True)}
        attempts = self.read("generation.json", sealed=True).get("attempts", {})
        results = []
        for i, entry in enumerate(selected):
            identifier = mapping.get(entry["response_id"])
            assessment = assessments.get(identifier)
            if assessment is not None:
                status = "accepted" if assessment.get("eligible") else "held"
            else:
                status = attempts.get(identifier, {}).get("status", "pending")
            results.append({"index": i, "source_id": entry["source_id"], "custom_id": identifier,
                            "category": entry.get("selection_bucket", "random"), "status": status,
                            "reasons": assessment.get("reasons", []) if assessment else
                                ([attempts[identifier]["diagnostic"]] if attempts.get(identifier, {}).get("diagnostic") else [])})
        return results

    def detail(self, index):
        selected = self.read("selected.jsonl", rows=True)
        if not 0 <= index < len(selected):
            raise ValueError("Unknown sample index")
        entry, item = selected[index], self.samples()[index]
        row = entry["row"]
        from pipeline.v11_regeneration_guard import source_text
        source = source_text(row)
        identifier = item["custom_id"]
        output = ""
        contract = self.read("contract.json", sealed=True)
        if contract.get("config", {}).get("transport") == "chatgpt":
            if identifier and identifier in contract["bindings"]:
                output = self.read("chatgpt_records/" + identifier + ".json", sealed=True).get("text", "")
        else:
            for record in self.read("output_file_id.jsonl", rows=True):
                if record.get("custom_id") == identifier:
                    body = (record.get("response") or {}).get("body") or {}
                    output = "\n".join(c.get("text", "") for message in body.get("output", [])
                                       if message.get("type") == "message"
                                       for c in message.get("content", []) if c.get("type") == "output_text")
                    break
        return {**item, "source": source, "output": output}

    def snapshot(self):
        contract = self.read("contract.json", sealed=True)
        config = contract.get("config", {})
        report, progress = self.read("report.json"), self.read("progress.json")
        remote = self.read("remote_status.json", sealed=True)
        generation = self.read("generation.json", sealed=True)
        samples = self.samples()
        categories = {}
        for item in samples:
            category = categories.setdefault(item["category"], {"total": 0, "accepted": 0, "held": 0})
            category["total"] += 1
            if item["status"] in {"accepted", "held"}:
                category[item["status"]] += 1
        transport = config.get("transport", "batch")
        status = ("validated" if report else progress.get("status") or generation.get("status")
                  or remote.get("status") or ("prepared" if contract else "not_prepared"))
        counts = remote.get("request_counts") or {}
        completed = progress.get("completed", counts.get("completed", report.get("results_received", 0)))
        usage = self.read("usage.json") or progress.get("usage", {})
        result = {"brand": "GenUI", "name": self.directory.name, "path": str(self.directory),
            "transport": transport, "config": config, "status": status, "requested": len(samples),
            "completed": completed, "provider_failed": counts.get("failed", 0),
            "uncertain": progress.get("uncertain", 0), "skipped": progress.get("skipped", 0),
            "accepted": report.get("accepted"), "held": report.get("rejected_or_missing"),
            "categories": categories, "samples": samples, "usage": usage,
            "estimate": self.read("estimate.json"), "rejection_reasons": report.get("rejection_reasons", {}),
            "selection": self.read("selection_report.json"), "updated_at": progress.get("updated_at"),
            "started_at": progress.get("started_at"), "batch_id": remote.get("id"),
            "source_counts": contract.get("counts", {}),
            "sealed": bool(self.read("chatgpt_collection.json" if transport == "chatgpt" else "collection.json", sealed=True)),
            "active_custom_id": progress.get("active_custom_id"),
            "local_snapshot_at": time.time(), "rescore": None}
        if self.rescore:
            receipt = _read(self.rescore, "rescore.json", sealed=True)
            summary = _read(self.rescore, "summary.json")
            if (receipt.get("contract_sha256") != digest(contract)
                    or receipt.get("files", {}).get("summary.json") != file_hash(self.rescore / "summary.json")):
                raise ValueError("Offline reassessment does not match this pilot")
            result["rescore"] = summary.get("overall")
        return result


class Controller:
    def __init__(self, directory, auth_home=None, rescore_dir=None, client_factory=CodexClient):
        validate_auth_location(directory, auth_home)
        self.evidence = Evidence(directory, rescore_dir)
        self.auth_home, self.client_factory = auth_home, client_factory
        self.token = secrets.token_urlsafe(32)
        self.client, self.worker, self.job, self.error = None, None, None, None
        self._lock, self._auth_at, self._auth = threading.Lock(), 0, {}

    def _client(self):
        with self._lock:
            if self.client is None:
                self.client = self.client_factory(self.auth_home)
            return self.client

    def auth(self):
        if self.client is None:
            return {"signed_in": False, "checked": False}
        if time.monotonic() - self._auth_at > 5:
            try:
                self._auth = {**self.client.account(), "checked": True}
                if self._auth["signed_in"]:
                    self._auth["limits"] = self.client.limits()
            except (RuntimeError, OSError, ValueError):
                self._auth = {"signed_in": False, "checked": True, "error": "Codex connection unavailable"}
            self._auth_at = time.monotonic()
        return self._auth

    def snapshot(self):
        return {**self.evidence.snapshot(), "auth": self.auth(), "csrf": self.token,
                "job": self.job, "error": self.error, "busy": bool(self.worker and self.worker.is_alive())}

    def action(self, name, body):
        if name in {"auth/check", "auth/login", "auth/device"}:
            client = self._client()
            self._auth_at = 0
            if name == "auth/check":
                return self.auth()
            if self.worker and self.worker.is_alive():
                raise ValueError("Wait for generation to stop before changing sign-in")
            return client.login(device=name == "auth/device")
        if name not in {"run", "collect"}:
            raise ValueError("Unknown GenUI action")
        if body:
            raise ValueError("Generation settings are bound by offline preparation")
        with self._lock:
            if self.worker and self.worker.is_alive():
                raise ValueError("GenUI already has an active operation")
            chatgpt_stage3.load(self.evidence.directory)  # Fail before creating any worker.
            if name == "run" and (self.client is None or not self.client.account()["signed_in"]):
                raise ValueError("Check your GenUI ChatGPT sign-in first")
            self.error, self.job = None, name
            self.worker = threading.Thread(target=self._work, args=(name,), daemon=True)
            self.worker.start()
        return {"status": "started", "operation": name}

    def _work(self, name):
        try:
            if name == "run":
                result = chatgpt_stage3.run(self.evidence.directory, client=self.client)
                if result["status"] != "generated":
                    return
            self.job = "collect"
            chatgpt_stage3.collect(self.evidence.directory)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            self.error = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "Operation failed; inspect local run evidence"
        finally:
            self.job = None

    def close(self):
        if self.client:
            self.client.close()


def make_server(controller, port=8767):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No prompts, OAuth URLs, codes or account details in access logs.

        def _allowed(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return (self.headers.get("Host") in hosts
                    and self.headers.get("Sec-Fetch-Site") != "cross-site"
                    and self.headers.get("Origin", "http://" + self.headers.get("Host", ""))
                    in {"http://" + host for host in hosts})

        def _send(self, status, data, content_type="application/json; charset=utf-8"):
            content = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if not self._allowed():
                return self._send(403, {"error": "Local GenUI origin required"})
            route = urlsplit(self.path)
            try:
                if route.path == "/api/state":
                    return self._send(200, controller.snapshot())
                if route.path == "/api/sample":
                    index = int(parse_qs(route.query).get("index", [""])[0])
                    return self._send(200, controller.evidence.detail(index))
                files = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
                if route.path not in files:
                    return self._send(404, {"error": "Not found"})
                name, mime = files[route.path]
                return self._send(200, (ASSETS / name).read_bytes(), mime + "; charset=utf-8")
            except (ValueError, OSError, KeyError, TypeError) as exc:
                return self._send(409, {"error": str(exc)})

        def do_POST(self):
            if not self._allowed() or not hmac.compare_digest(self.headers.get("X-GenUI-Token", ""), controller.token):
                return self._send(403, {"error": "GenUI session token required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected bounded JSON request")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict) or not self.path.startswith("/api/"):
                    raise ValueError("Invalid request")
                result = controller.action(self.path.removeprefix("/api/"), body)
                return self._send(200, result)
            except (ValueError, OSError, RuntimeError) as exc:
                return self._send(409, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server
