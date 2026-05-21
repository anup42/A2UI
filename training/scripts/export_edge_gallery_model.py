from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.export.edge_gallery import export_edge_gallery_model
from ir_training.export.merge_lora import merge_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained IR model as a Google AI Edge Gallery .litertlm model.")
    parser.add_argument("--config", required=True, help="Path to Edge Gallery export YAML config.")
    parser.add_argument("--merge-lora", action="store_true", help="Merge the configured LoRA adapter into an HF model before exporting.")
    parser.add_argument("--dry-run", action="store_true", help="Write the export command/manifest without running litert-torch.")
    parser.add_argument("--model-source", help="Override model source passed to litert-torch export_hf.")
    parser.add_argument("--output-dir", help="Override run.output_dir in the export config.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    if args.output_dir:
        config.setdefault("run", {})["output_dir"] = args.output_dir

    source = config.get("source") if isinstance(config.get("source"), dict) else {}
    if args.merge_lora:
        merged_dir = merge_lora_adapter(
            base_model_id=str(source.get("base_model_id") or ""),
            adapter_dir=source.get("adapter_dir") or "",
            output_dir=source.get("merged_model_dir") or "",
        )
        config.setdefault("source", {})["merged_model_dir"] = str(merged_dir)

    manifest = export_edge_gallery_model(
        config,
        dry_run=args.dry_run,
        model_source_override=args.model_source,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
