from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.common.logging import configure_logging
from ir_training.data.build_pairs import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare response-to-IR SFT data from a dataset run.")
    parser.add_argument("--config", required=True, help="Path to training dataset YAML config.")
    args = parser.parse_args()
    configure_logging()
    config_path = Path(args.config).resolve()
    manifest = prepare_dataset(load_yaml(config_path), config_path=config_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
