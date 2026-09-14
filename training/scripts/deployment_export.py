"""Isolated-environment entry point for current checkpoint deployment export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.pipeline.deployment_export import main

if __name__ == "__main__":
    main()
