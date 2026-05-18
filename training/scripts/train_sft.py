from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.common.logging import configure_logging
from ir_training.train.sft import train_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a response-to-IR SFT adapter.")
    parser.add_argument("--config", required=True, help="Path to model training YAML config.")
    args = parser.parse_args()
    configure_logging()
    config_path = Path(args.config).resolve()
    result = train_sft(load_yaml(config_path), config_path=config_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
