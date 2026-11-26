"""Export a trained 270M/E2B checkpoint or evaluate the export on a named phone."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.export.qualcomm_npu import export_checkpoint, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    export = actions.add_parser(
        "export", help="Compile a target-only Qualcomm SM8850 LiteRT-LM"
    )
    export.add_argument("--profile", choices=("270m", "e2b"), required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument(
        "--training-config",
        type=Path,
        help="Use a relocated copy of the bound training config",
    )
    export.add_argument("--cache-length", type=int, default=8192)
    export.add_argument("--output-dir", type=Path, required=True)
    test = actions.add_parser(
        "bixby50", help="Greedy native Android NPU evaluation, MTP off"
    )
    test.add_argument("--export-dir", type=Path, required=True)
    test.add_argument("--runtime-dir", type=Path, required=True)
    test.add_argument("--serial", required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    test.add_argument("--adb", default="adb")
    args = parser.parse_args()
    if args.action == "export":
        config = args.training_config
        if config is None:
            receipt, field = (
                ("training_metadata.json", "config_path")
                if args.profile == "270m"
                else ("qat_mtp_merge_metadata.json", "training_config")
            )
            config = Path(read_json(args.checkpoint / receipt)[field])
        result = export_checkpoint(
            args.profile, args.checkpoint, config, args.output_dir, args.cache_length
        )
    else:
        from ir_training.eval.qualcomm_npu import test_on_device

        result = test_on_device(
            args.export_dir, args.runtime_dir, args.serial, args.output_dir, args.adb
        )
    print(result)


if __name__ == "__main__":
    main()
