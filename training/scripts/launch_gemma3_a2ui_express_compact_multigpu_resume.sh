#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. Activate the GPU host's environment first.
# Print-only by default; pass --execute to start. No saved host/user/GPU paths.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/launch_review_training.py" "$@"
