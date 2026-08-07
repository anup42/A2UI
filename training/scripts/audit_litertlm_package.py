"""CLI wrapper for the read-only LiteRT-LM package/graph inspector."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.export.litertlm_inspector import main


if __name__ == "__main__":
    raise SystemExit(main())
