param(
  [int]$Port = 8008,
  [string]$Root = $PSScriptRoot,
  [ValidateSet("lit", "visualizer")]
  [string]$Mode = "lit",
  [string]$OpenPath = "",
  [switch]$NoOpenBrowser
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  throw "Python is required. Install Python 3 and ensure 'python' is on PATH."
}

$resolvedRoot = (Resolve-Path $Root).Path

function Get-ServeUrl {
  param(
    [int]$PortValue,
    [string]$PathValue,
    [string]$ModeValue
  )
  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    if ($ModeValue -eq "visualizer") {
      return "http://127.0.0.1:$PortValue/"
    }
    return "http://127.0.0.1:$PortValue/data/runs/"
  }
  $trimmed = $PathValue.TrimStart("/")
  return "http://127.0.0.1:$PortValue/$trimmed"
}

$openUrl = Get-ServeUrl -PortValue $Port -PathValue $OpenPath -ModeValue $Mode

if ($Mode -eq "visualizer") {
  $visualizerApp = Join-Path $resolvedRoot "visualizer/app.py"
  if (-not (Test-Path $visualizerApp)) {
    throw "Visualizer app not found: $visualizerApp"
  }
  $runsDir = Join-Path $resolvedRoot "data/runs"
  $rendererDir = Join-Path $resolvedRoot "renderer"
  Write-Host "[serve_lit] Starting visualizer at $openUrl"
  if (-not $NoOpenBrowser) {
    Start-Process $openUrl
  }
  & python $visualizerApp --host 127.0.0.1 --port $Port --runs-dir $runsDir --renderer-dir $rendererDir
  exit $LASTEXITCODE
}

Write-Host "[serve_lit] Starting lit static server at $openUrl"
if (-not $NoOpenBrowser) {
  Start-Process $openUrl
}

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

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

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
