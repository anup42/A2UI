#!/usr/bin/env python3
"""Export the Golden-selected Gemma 3 270M full QAT checkpoint as W8 LiteRT-LM."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.export.litertlm_inspector import inspect_litertlm
from ir_training.pipeline.deployment_export import (
    export_kwargs,
    file_sha256,
    inspect_variant_precision,
    probe_exporter,
)
from ir_training.pipeline.deployment_recovery import deployment_lock
from ir_training.train.resume_contract import (
    _verify_source_inventory,
    resolve_export_training_lineage,
)

COHORTS = {"golden32": 32, "golden35": 35, "bixby50": 50}
RECIPE = "dynamic_wi8_afp32"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def validate_270m_model(model_dir: Path) -> None:
    config = _json(model_dir / "config.json")
    text = config.get("text_config", config)
    if (config.get("model_type") not in {"gemma3_text", "gemma3"}
            or not isinstance(text, dict)
            or (text.get("hidden_size"), text.get("num_hidden_layers")) != (640, 18)):
        raise ValueError(f"Expected a local Gemma 3 270M checkpoint: {model_dir}")


def _selected_checkpoint(run: Path) -> tuple[Path, dict, dict[Path, str]]:
    manifest_path = run / "pipeline_manifest.json"
    bound = {manifest_path: file_sha256(manifest_path)}
    manifest = _json(manifest_path)
    plan = manifest.get("plan") or {}
    options = plan.get("options") or {}
    if (manifest.get("status") != "complete"
            or plan.get("workflow") != "shared_prompt_golden_bixby_training_v2"
            or options.get("profile") != "270m" or options.get("qat") is not True):
        raise ValueError("Expected a completed 270M full-QAT Golden training run.")
    completed = manifest["completed"]

    def bind(stage: str, path: Path) -> None:
        if path.resolve() != path or not path.is_relative_to(run):
            raise ValueError(f"Stage {stage} evidence escapes the run: {path}")
        digest = (completed.get(stage) or {}).get("files", {}).get(str(path))
        if not digest or file_sha256(path) != digest:
            raise ValueError(f"Stage {stage} evidence is missing or changed: {path}")
        bound[path] = digest

    checkpoint = run / "fit/training/best_golden_checkpoint"
    config_path = run / "fit/training_config.yaml"
    bind("configure", config_path)
    bind("configure", config_path.with_name("preparation_report.json"))
    bind("training", checkpoint / "training_metadata.json")
    lineage = resolve_export_training_lineage(config_path, checkpoint)
    config, metadata = lineage["config"], lineage["metadata"]
    qat = config.get("qat") or {}
    expected_qat = {
        "enabled": True, "profile": "gemma3_270m_wi8_afp32_full_finetune",
        "final_quantization": RECIPE, "weight_bits": 8, "activation_bits": 32,
        "quantizer": "ste_ai_edge", "quantize_embeddings": True,
    }
    if (config["training"]["method"] != "full_finetune_qat"
            or any(qat.get(key) != value for key, value in expected_qat.items())
            or metadata.get("checkpoint_kind") != "full_model"
            or metadata.get("checkpoint_role") != "best_golden"
            or type(metadata.get("checkpoint_step")) is not int
            or metadata["checkpoint_step"] <= 0):
        raise ValueError("Expected the Golden-selected full-model W8 QAT checkpoint.")
    validate_270m_model(checkpoint)
    _verify_source_inventory(checkpoint, metadata)

    scorecard_path = run / "evaluation_scorecard.json"
    bind("scorecard", scorecard_path)
    scorecard = _json(scorecard_path)
    if scorecard.get("status") != "complete" or scorecard.get("profile") != "270m":
        raise ValueError("Expected a complete 270M evaluation scorecard.")
    for cohort, rows in COHORTS.items():
        stage = f"best_{cohort}"
        files = (completed.get(stage) or {}).get("files", {})
        results = [Path(path) for path in files if Path(path).name == "evaluation_result.json"]
        if len(results) != 1 or not results[0].is_relative_to(run / "evaluations" / stage):
            raise ValueError(f"Missing selected-checkpoint evaluation for {cohort}.")
        bind(stage, results[0])
        evaluation = _json(results[0])
        if (evaluation != scorecard.get("evaluations", {}).get(stage)
                or evaluation.get("row_count") != rows or evaluation.get("qat_applied") is not True
                or Path(evaluation.get("checkpoint", "")).resolve() != checkpoint):
            raise ValueError(f"{cohort} lacks a complete QAT-on evaluation of the selected checkpoint.")
    return checkpoint, lineage, bound


def export_w8(run_dir: Path, output_dir: Path, *, cache_length: int = 8192) -> dict:
    run, out = (path.expanduser().resolve() for path in (run_dir, output_dir))
    checkpoint, lineage, bound = _selected_checkpoint(run)
    config, metadata = lineage["config"], lineage["metadata"]
    golden = config["golden_eval"]
    if cache_length < golden["max_input_tokens"] + golden["max_new_tokens"]:
        raise ValueError("Export cache must cover the saved prompt plus generation budget.")
    protected = [run / "fit", run / "prepared"]
    for section, key in (("model", "model_source"), ("run", "dataset_dir")):
        value = config.get(section, {}).get(key)
        if value:
            protected.append(Path(value).resolve())
    if run.is_relative_to(out) or any(out.is_relative_to(path) or path.is_relative_to(out) for path in protected):
        raise ValueError("Export output must be separate from the checkpoint, data, and training inputs.")
    with deployment_lock(out):
        if out.exists():
            raise FileExistsError(f"Choose a fresh W8 export directory: {out}")
        preflight = probe_exporter(profile="270m", model_dir=checkpoint, cache_length=cache_length,
                                   selected_variants=("w8",))
        from litert_torch.generative.export_hf.export import export

        # Keep failed conversion evidence in staging and publish only audited results.
        staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
        print(f"Converting selected checkpoint in {staging}", flush=True)
        kwargs = export_kwargs("270m", "w8", checkpoint, staging, cache_length)
        export(**kwargs)
        packages = list(staging.rglob("*.litertlm"))
        if len(packages) != 1 or packages[0].stat().st_size < 8:
            raise RuntimeError("Expected exactly one nonempty LiteRT-LM package.")
        artifact = staging / "model.litertlm"
        if packages[0] != artifact:
            packages[0].rename(artifact)
        package = inspect_litertlm(artifact, include_hashes=False, include_graph_details=True)
        precision = inspect_variant_precision(package, "270m", "w8")
        _verify_source_inventory(checkpoint, metadata)
        if any(file_sha256(path) != digest for path, digest in bound.items()):
            raise ValueError("Training or evaluation evidence changed during conversion.")
        report = {
            "status": "exported", "profile": "270m", "variant": "w8", "recipe": RECIPE,
            "artifact": str(out / artifact.name), "artifact_sha256": file_sha256(artifact),
            "source_checkpoint": str(checkpoint), "checkpoint_step": metadata["checkpoint_step"],
            "training_config_sha256": lineage["training_config_sha256"],
            "training_metadata_sha256": bound[checkpoint / "training_metadata.json"],
            "export_settings": kwargs, "exporter_preflight": preflight, "physical_weight_precision": precision,
        }
        (staging / "w8_export_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staging.rename(out)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed 270M full-QAT Golden training run")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory, defaults to <run-dir>/litertlm_w8")
    parser.add_argument("--cache-length", type=int, default=8192)
    parser.add_argument("--execute", action="store_true", help="Perform CPU conversion after verifying the source")
    args = parser.parse_args(argv)
    out = args.output_dir or args.run_dir / "litertlm_w8"
    try:
        if args.execute:
            os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            result = export_w8(args.run_dir, out, cache_length=args.cache_length)
        else:
            result = {"status": "plan_only", "run_dir": str(args.run_dir), "output_dir": str(out),
                      "recipe": RECIPE, "cache_length": args.cache_length}
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, ImportError, KeyError) as exc:
        print(f"270M W8 export failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
