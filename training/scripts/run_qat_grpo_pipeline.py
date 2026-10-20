"""Plan/run isolated E2B or 270M QAT LoRA GRPO and canonical export/native QA."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset" / "src"))

from ir_training.pipeline.qat_grpo import QATGRPOOptions, run_pipeline, worker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-stage", help=argparse.SUPPRESS)
    parser.add_argument("--plan-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rerun-from", type=Path,
                        help="Reuse original SFT identities/options from this GRPO run; prepare raw data afresh.")
    parser.add_argument("--continue-run", type=Path,
                        help="Verify and continue post-training stages only; never optimizer resume.")
    parser.add_argument("--family", choices=("e2b", "270m"))
    parser.add_argument("--objective", choices=("grpo", "grpo_sft", "dpo"),
                        help="Default grpo; opt in to supervised rehearsal or offline QAT DPO.")
    parser.add_argument("--sft-rehearsal-weight", type=float,
                        help="Positive supervised-loss weight; grpo_sft only (default 0.1).")
    parser.add_argument("--dpo-preferences", type=Path,
                        help="Offline source-bound chosen/rejected JSONL; required for dpo.")
    parser.add_argument("--dpo-beta", type=float,
                        help="Positive DPO temperature; dpo only (default 0.1).")
    parser.add_argument("--dpo-learning-rate", type=float,
                        help="DPO-only LR override, including with --rerun-from; otherwise inherit.")
    parser.add_argument("--dpo-max-steps", type=int,
                        help="DPO-only optimizer update cap (>=20); otherwise inherit.")
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--dpo-warmup-steps", type=int,
                        help="DPO-only warmup steps, including 0; replace inherited SFT warmup.")
    warmup.add_argument("--dpo-warmup-ratio", type=float,
                        help="DPO-only warmup fraction in [0,1); resolve to ceil(max_steps * ratio).")
    parser.add_argument("--dpo-golden-every-steps", type=int,
                        help="DPO-only evaluation/checkpoint cadence (>=20); otherwise inherit.")
    for name in ("model-dir", "sft-config", "sft-checkpoint", "output-dir", "official-litertlm", "exporter-python",
                 "input-dir", "prepared-input-dir", "source-safetensors"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--rollout-microbatch", type=int, default=1,
                        help="Generation candidates per rank/call; does not change the optimizer batch.")
    parser.add_argument("--rollout-qat-weight-cache", action="store_true",
                        help="Opt-in eval-only fake-quant weight cache; costs additional GPU memory.")
    for name, default in (("max-steps", 200), ("num-generations", 4), ("microbatch", 1), ("effective-batch", 32),
                          ("max-seq-length", 6144), ("max-input-tokens", 5120), ("max-new-tokens", 2048),
                          ("golden-every-steps", 50), ("validation-max-rows", 32),
                          ("early-stopping-patience", 3), ("audit-every-steps", 25), ("seed", 42), ("prepare-workers", 0)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--min-retained-fraction", type=float, default=0.5,
                        help="Fail preparation below this train/val retention fraction; never truncate facts.")
    parser.add_argument("--quality-min-delta", type=float, default=0.5,
                        help="Minimum unique-source Golden32 gain in 0..100 points for promotion.")
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
    if args.continue_run:
        from ir_training.pipeline.qat_grpo import continue_pipeline
        supplied = list(sys.argv[1:] if argv is None else argv)
        allowed = {"--continue-run", "--execute"}
        if any(token.split("=", 1)[0] not in allowed for token in supplied if token.startswith("--")):
            parser.error("--continue-run accepts only --execute; the saved plan is immutable")
        print(json.dumps(continue_pipeline(args.continue_run, execute=args.execute), indent=2))
        return 0
    if args.rerun_from:
        from ir_training.train.grpo_continuation import options_from_previous_run
        if args.output_dir is None:
            parser.error("--rerun-from requires a fresh --output-dir")
        supplied = list(sys.argv[1:] if argv is None else argv)
        allowed = {"--rerun-from", "--output-dir", "--input-dir", "--defer-native-eval", "--execute",
                   "--objective", "--sft-rehearsal-weight", "--dpo-preferences", "--dpo-beta",
                   "--dpo-learning-rate", "--dpo-max-steps", "--dpo-warmup-steps",
                   "--dpo-warmup-ratio", "--dpo-golden-every-steps"}
        if any(token.split("=", 1)[0] not in allowed for token in supplied if token.startswith("--")):
            parser.error("Rerun preserves original hyperparameters except explicit DPO overrides; "
                         "use a new explicit plan for other changes")
        options = options_from_previous_run(args.rerun_from, args.output_dir, args.input_dir)
        if args.objective is not None and args.objective != options.objective:
            # Switching objective is a fresh branch from original SFT, not
            # continuation of the previous objective or its stale options.
            options = replace(options, objective=args.objective, sft_rehearsal_weight=None,
                              dpo_preferences=None, dpo_beta=None, dpo_learning_rate=None,
                              dpo_max_steps=None, dpo_warmup_steps=None, dpo_warmup_ratio=None,
                              dpo_golden_every_steps=None)
        changes = {name: getattr(args, name) for name in (
            "sft_rehearsal_weight", "dpo_preferences", "dpo_beta", "dpo_learning_rate",
            "dpo_max_steps", "dpo_warmup_steps", "dpo_warmup_ratio", "dpo_golden_every_steps",
        ) if getattr(args, name) is not None}
        if args.dpo_warmup_steps is not None:
            changes["dpo_warmup_ratio"] = None
        elif args.dpo_warmup_ratio is not None:
            changes["dpo_warmup_steps"] = None
        options = replace(options, **changes)
        if args.defer_native_eval:
            options = replace(options, defer_native_eval=True)
        print(json.dumps(run_pipeline(options, execute=args.execute), indent=2))
        return 0
    for name in ("family", "model_dir", "sft_config", "sft_checkpoint", "output_dir", "official_litertlm", "exporter_python"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    values = vars(args).copy()
    execute = values.pop("execute")
    values.pop("worker_stage")
    values.pop("plan_file")
    values.pop("rerun_from")
    values.pop("continue_run")
    values["objective"] = values["objective"] or "grpo"
    result = run_pipeline(QATGRPOOptions(**values), execute=execute)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
