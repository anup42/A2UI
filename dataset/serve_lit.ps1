param(
  [int]$Port = 8008,
  [string]$Root = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  throw "Python is required. Install Python 3 and ensure 'python' is on PATH."
}

$resolvedRoot = (Resolve-Path $Root).Path

@'
import argparse
import http.server
import mimetypes
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8008)
parser.add_argument("--root", type=str, required=True)
args = parser.parse_args()

root = str(pathlib.Path(args.root).resolve())

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/wasm", ".wasm")

class LitHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = dict(http.server.SimpleHTTPRequestHandler.extensions_map)
    extensions_map.update(
        {
            ".js": "application/javascript",
            ".mjs": "application/javascript",
            ".json": "application/json",
            ".css": "text/css",
            ".svg": "image/svg+xml",
            ".wasm": "application/wasm",
        }
    )

    def __init__(self, *handler_args, **handler_kwargs):
        super().__init__(*handler_args, directory=root, **handler_kwargs)

    def log_message(self, fmt, *log_args):
        print(f"[serve_lit] {fmt % log_args}")

server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), LitHandler)
print(f"[serve_lit] Serving {root} at http://127.0.0.1:{args.port}")
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
    print("[serve_lit] Stopped")
'@ | python - --port $Port --root "$resolvedRoot"
