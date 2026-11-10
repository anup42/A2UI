#!/usr/bin/env python3
"""Train Gemma 3 270M with full QAT, score Golden/Bixby, then export W8 LiteRT-LM."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / "src"))
from export_gemma3_270m_full_qat_w8 import validate_270m_model
from ir_training.common.bounded_command import run_bounded_command
from ir_training.pipeline.deployment_export import _python, file_sha256
from ir_training.pipeline.deployment_recovery import deployment_lock
from ir_training.pipeline.golden_training import (
    GoldenTrainingOptions,
    build_plan,
    run_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="Local dense 270M SFT checkpoint with tokenizer")
    parser.add_argument("--input-dir", required=True, type=Path, help="Extracted dataset directory with train.jsonl and val.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="Fresh run directory")
    parser.add_argument("--exporter-python", required=True, type=Path, help="Absolute Python in the pinned CPU exporter environment")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--golden-every-steps", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=5120)
    parser.add_argument("--max-input-tokens", type=int, help="Defaults to the 270M supervised sequence limit")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--cache-length", type=int, default=8192)
    parser.add_argument("--tensorboard-root")
    parser.add_argument("--continue-run", action="store_true", help="Verify and reuse completed training/evaluation stages")
    parser.add_argument("--execute", action="store_true", help="Run the pipeline. Default: read-only plan")
    args = parser.parse_args(argv)

    try:
        exporter_python = _python(args.exporter_python)
        output = args.output_dir.expanduser().resolve()
        model = args.model_dir.expanduser().resolve()
        validate_270m_model(model)
        max_input_tokens = args.max_seq_length if args.max_input_tokens is None else args.max_input_tokens
        if args.cache_length < max_input_tokens + args.max_new_tokens:
            raise ValueError("Export cache must cover the evaluation prompt plus generation budget.")
        options = GoldenTrainingOptions(
            profile="270m", qat=True, model_dir=model, input_dir=args.input_dir, output_dir=output,
            devices=args.devices, epochs=args.epochs, learning_rate=args.learning_rate, steps=args.steps,
            eval_steps=args.eval_steps, golden_every_steps=args.golden_every_steps,
            max_seq_length=args.max_seq_length, max_input_tokens=max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            tensorboard_root=args.tensorboard_root or os.environ.get("A2UI_TENSORBOARD_ROOT") or f"{output}_tensorboard",
        )
        training = build_plan(options)
        export_dir = output / "litertlm_w8"
        artifact = export_dir / "model.litertlm"
        export_report = export_dir / "w8_export_report.json"
        probe = [
            exporter_python, "-u", str(SCRIPTS / "deployment_export.py"), "probe",
            "--profile", "270m", "--variants", "w8", "--model-dir", str(model),
            "--cache-length", str(args.cache_length),
        ]
        export = [
            exporter_python, "-u", str(SCRIPTS / "export_gemma3_270m_full_qat_w8.py"),
            "--run-dir", str(output), "--output-dir", str(export_dir),
            "--cache-length", str(args.cache_length), "--execute",
        ]
        if not args.execute:
            print(json.dumps({
                "status": "plan_only", "training": training,
                "stages": ["exporter_preflight", *training["stages"], "export_w8"],
                "exporter_preflight_command": [*probe, "--report", "<temporary preflight report>"],
                "export_command": export, "artifact": str(artifact), "export_report": str(export_report),
            }, indent=2))
            return 0

        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        with deployment_lock(output):
            if output.exists() and not args.continue_run:
                raise FileExistsError(f"Choose a fresh run directory: {output}")
            if args.continue_run and not (output / "pipeline_manifest.json").is_file():
                raise FileNotFoundError("--continue-run requires an existing Golden training manifest.")
            if export_dir.exists():
                raise FileExistsError(f"W8 export already exists: {export_dir}")
            with tempfile.TemporaryDirectory(prefix="a2ui-270m-probe-") as scratch:
                scratch = Path(scratch)
                preflight_path = scratch / "preflight.json"
                run_bounded_command([*probe, "--report", str(preflight_path)], scratch / "probe.log", environment,
                                    timeout_seconds=300)
                preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
                if (preflight.get("status") != "passed" or preflight.get("profile") != "270m"
                        or preflight.get("variants") != ["w8"]):
                    raise ValueError("Exporter preflight did not verify the 270M W8 converter.")
            state = run_pipeline(options, execute=True, continue_run=args.continue_run)
            if state.get("status") != "complete":
                raise RuntimeError("Golden training and evaluation did not complete.")
            run_bounded_command(export, output / "logs/export_w8.log", environment)
            report = json.loads(export_report.read_text(encoding="utf-8"))
            if (report.get("status") != "exported" or report.get("artifact") != str(artifact)
                    or report.get("artifact_sha256") != file_sha256(artifact)
                    or report.get("physical_weight_precision", {}).get("verified") is not True):
                raise ValueError("W8 exporter did not publish the expected audited artifact.")
        print(json.dumps({
            "status": "complete", "output_dir": str(output),
            "evaluation_scorecard": str(output / "evaluation_scorecard.json"),
            "artifact": str(artifact), "export_report": str(export_report),
        }, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, ImportError, KeyError) as exc:
        print(f"270M full-QAT pipeline failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
