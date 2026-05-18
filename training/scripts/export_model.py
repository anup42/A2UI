from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.export.litertlm import export_litertlm_package
from ir_training.export.merge_lora import merge_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained IR model.")
    parser.add_argument("--config", required=True, help="Path to export YAML config.")
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA adapter into HF model before packaging.")
    args = parser.parse_args()
    config = load_yaml(Path(args.config).resolve())
    source = config.get("source") if isinstance(config.get("source"), dict) else {}
    if args.merge_lora:
        merge_lora_adapter(
            base_model_id=str(source.get("base_model_id") or ""),
            adapter_dir=source.get("adapter_dir") or "",
            output_dir=source.get("merged_model_dir") or "",
        )
    manifest = export_litertlm_package(config)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
