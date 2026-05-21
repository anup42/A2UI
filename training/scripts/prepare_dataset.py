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
    parser.add_argument("--source-run-dir", help="Override run.source_run_dir for one completed dataset run.")
    parser.add_argument("--source-genui-dir", help="Override run.source_genui_dir for a folder of Stage 3 JSONL files.")
    parser.add_argument("--source-glob", help="Glob under --source-genui-dir, default **/genui.jsonl.")
    parser.add_argument("--output-dir", help="Override run.output_dir.")
    parser.add_argument("--seed", type=int, help="Override run.seed.")
    args = parser.parse_args()
    configure_logging()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    run_cfg = config.setdefault("run", {})
    if args.source_run_dir:
        run_cfg.pop("source_genui_dir", None)
        run_cfg["source_run_dir"] = str(Path(args.source_run_dir).resolve())
    if args.source_genui_dir:
        run_cfg.pop("source_run_dir", None)
        run_cfg["source_genui_dir"] = str(Path(args.source_genui_dir).resolve())
    if args.source_glob:
        run_cfg["source_glob"] = args.source_glob
    if args.output_dir:
        run_cfg["output_dir"] = str(Path(args.output_dir).resolve())
    if args.seed is not None:
        run_cfg["seed"] = args.seed
    manifest = prepare_dataset(config, config_path=config_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
