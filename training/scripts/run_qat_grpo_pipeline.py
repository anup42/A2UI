"""Plan/run isolated E2B or 270M QAT LoRA GRPO and canonical export/native QA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.pipeline.qat_grpo import QATGRPOOptions, run_pipeline, worker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-stage", help=argparse.SUPPRESS)
    parser.add_argument("--plan-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--family", choices=("e2b", "270m"))
    for name in ("model-dir", "sft-config", "sft-checkpoint", "output-dir", "official-litertlm", "exporter-python",
                 "input-dir", "prepared-input-dir", "source-safetensors"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--devices", default="auto")
    for name, default in (("max-steps", 200), ("num-generations", 4), ("microbatch", 1), ("effective-batch", 32),
                          ("max-seq-length", 4096), ("max-input-tokens", 5120), ("max-new-tokens", 2048),
                          ("golden-every-steps", 50), ("seed", 42), ("prepare-workers", 0)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--tensorboard-root", default="/tensorboard")
    parser.add_argument("--stage-timeout-seconds", type=float, default=172800)
    parser.add_argument("--generation-timeout-seconds", type=float, default=7200)
    parser.add_argument("--progress-seconds", type=float, default=30)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial")
    parser.add_argument("--defer-native-eval", action="store_true",
                        help="Export on the GPU host; leave native quality explicitly pending for the Android host.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.worker_stage:
        if args.plan_file is None:
            parser.error("worker stage requires plan file")
        worker(args.plan_file, args.worker_stage)
        return 0
    for name in ("family", "model_dir", "sft_config", "sft_checkpoint", "output_dir", "official_litertlm", "exporter_python"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    values = vars(args).copy()
    execute = values.pop("execute")
    values.pop("worker_stage")
    values.pop("plan_file")
    result = run_pipeline(QATGRPOOptions(**values), execute=execute)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
