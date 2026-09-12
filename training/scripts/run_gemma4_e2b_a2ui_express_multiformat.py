#!/usr/bin/env python3
"""Plan or explicitly run the Gemma 4 E2B A2UI Express multi-format pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml  # noqa: E402
from ir_training.train.gpu_profile import visible_launch_profile
from ir_training.pipeline.gemma4_e2b_multiformat import (  # noqa: E402
    STAGE_ORDER,
    Gemma4E2BMultiformatPipelineError,
    run_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="training/configs/pipelines/gemma4_e2b_a2ui_express_multiformat.yaml",
    )
    parser.add_argument("--run-id", help="Unique run ID; overrides pipeline.run_id.")
    parser.add_argument("--source-safetensors", help="Exact official packed model.safetensors.")
    parser.add_argument("--base-litertlm", help="Exact official released Gemma 4 E2B .litertlm.")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--gpu-ids", help="Comma-separated CUDA indices passed to the portable trainer.")
    parser.add_argument(
        "--runner-config",
        help="Real external LiteRT-LM JSONL runner config; overrides the checked-in example.",
    )
    mtp = parser.add_mutually_exclusive_group()
    mtp.add_argument("--mtp", dest="mtp_enabled", action="store_true")
    mtp.add_argument("--no-mtp", dest="mtp_enabled", action="store_false")
    parser.set_defaults(mtp_enabled=None)
    parser.add_argument(
        "--execute-stage",
        action="append",
        choices=STAGE_ORDER,
        default=[],
        help="Execute one explicit stage; repeat in dependency order.",
    )
    parser.add_argument(
        "--execute-all",
        action="store_true",
        help="Execute every configured stage in the checked-in dependency order.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    try:
        num_gpus, gpu_ids = args.num_gpus, args.gpu_ids
        if args.execute_all or "training" in args.execute_stage:
            host_gpu_profile = visible_launch_profile(model="e2b", num_gpus=num_gpus, launch_ids=gpu_ids)
            num_gpus, gpu_ids = host_gpu_profile["world_size"], host_gpu_profile["cuda_visible_devices"]
        plan = run_pipeline(
            load_yaml(config_path),
            config_path=config_path,
            stages=args.execute_stage,
            execute_all=args.execute_all,
            run_id_override=args.run_id,
            source_safetensors_override=args.source_safetensors,
            base_litertlm_override=args.base_litertlm,
            num_gpus_override=num_gpus,
            gpu_ids_override=gpu_ids,
            mtp_enabled_override=args.mtp_enabled,
            runner_config_override=args.runner_config,
        )
    except (OSError, ValueError, Gemma4E2BMultiformatPipelineError) as exc:
        print(f"Gemma 4 E2B multi-format pipeline failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not args.execute_stage and not args.execute_all:
        print(
            "Plan only: no dataset write, training, model load, conversion, "
            "evaluation, or TensorBoard write was run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
