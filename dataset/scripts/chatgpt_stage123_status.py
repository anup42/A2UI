"""Serve the three-stage ChatGPT run's progress on loopback."""

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

COUNTS = ("target", "queries", "responses", "generated", "accepted", "held", "workers")


def snapshot(directory):
    try:
        raw = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"stage": "Waiting for the first progress update"}
    if not isinstance(raw, dict):
        raise TypeError("Expected a progress object")
    result = {key: raw[key] for key in COUNTS if type(raw.get(key)) is int and raw[key] >= 0}
    for key in ("stage", "detail"):
        if isinstance(raw.get(key), str):
            result[key] = raw[key]
    return result


def dashboard(state):
    def number(key):
        return format(state[key], ",") if key in state else "–"

    stage = html.escape(state.get("stage", "Waiting for progress"))
    detail = html.escape(state.get("detail", ""))
    target = state.get("target") or 0
    stages = []
    for key, label in (("queries", "Stage 1 · Queries"), ("responses", "Stage 2 · Responses"),
                       ("generated", "Stage 3 · A2UI Express outputs")):
        bar = f'<progress aria-label="{label}" max="{target}" value="{state[key]}"></progress>' if target and key in state else ""
        stages.append(f'<section><h2>{label}</h2><strong>{number(key)}</strong> / {number("target")}{bar}</section>')
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="5">
<title>GenUI generation progress</title><style>
:root{{font-family:Inter,system-ui,sans-serif;color:#173c39;background:#f6f8f6}}
main{{max-width:60rem;margin:8vh auto;padding:0 1rem}}
h1{{font-size:clamp(2rem,5vw,3rem);letter-spacing:-.05em;margin:0}}
h2{{font-size:1rem;margin:0 0 1rem}}section{{background:white;border:1px solid #d5dfd8;padding:1.25rem;border-radius:.5rem}}
article{{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));margin:2rem 0}}
progress{{display:block;width:100%;height:.6rem;accent-color:#177660;margin-top:1rem}}
strong{{font-size:1.6rem}}p{{line-height:1.5;overflow-wrap:anywhere}}dl{{display:flex;flex-wrap:wrap;gap:2rem}}
dt{{font-size:.875rem}}dd{{font-size:1.35rem;margin:.25rem 0;font-weight:600}}
</style><main><h1>GenUI generation progress</h1><p>Current stage · {stage}</p>
<article>{''.join(stages)}</article><dl>
<div><dt>Accepted for training</dt><dd>{number("accepted")}</dd></div>
<div><dt>Held for review</dt><dd>{number("held")}</dd></div>
<div><dt>Workers</dt><dd>{number("workers")}</dd></div></dl><p>{detail}</p></main></html>"""


def make_server(directory, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if (self.headers.get("Host") not in hosts or self.headers.get("Sec-Fetch-Site") == "cross-site"
                    or self.headers.get("Origin") not in {None, *("http://" + host for host in hosts)}):
                self.send_error(403)
                return
            route = urlsplit(self.path).path
            if route not in {"/", "/api/state"}:
                self.send_error(404)
                return
            try:
                data = snapshot(directory)
                status = 200
            except (OSError, TypeError, ValueError):
                data = {"stage": "Unable to read the current progress file"}
                status = 503
            if route == "/":
                body = dashboard(data).encode("utf-8")
                mime = "text/html"
            else:
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                mime = "application/json"
            self.send_response(status)
            self.send_header("Content-Type", f"{mime}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8784)
    args = parser.parse_args()
    with make_server(args.output_dir.resolve(), args.port) as server:
        print(f"GenUI dashboard at http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
