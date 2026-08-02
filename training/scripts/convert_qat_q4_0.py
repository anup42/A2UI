from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.export.q4_0 import build_q4_0_conversion_plan, execute_q4_0_conversion


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or execute matched Gemma 4 target/assistant GGUF Q4_0 conversion.")
    parser.add_argument(
        "--config",
        default="training/configs/export/gemma4_e2b_qat_q4_0.yaml",
    )
    parser.add_argument("--llama-cpp-dir", help="Override conversion.llama_cpp_dir.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Download/load inputs and run external conversion. Without this flag only commands are printed.",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacement of existing final Q4_0 output files.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config).resolve())
    if not args.execute:
        plan = build_q4_0_conversion_plan(config, llama_cpp_dir_override=args.llama_cpp_dir)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print("Plan only: no checkpoint was downloaded and no conversion or quantization was run.")
        return
    result = execute_q4_0_conversion(
        config,
        llama_cpp_dir_override=args.llama_cpp_dir,
        force=args.force,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
