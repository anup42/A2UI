from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml, resolve_path, training_root
from ir_training.export.merge_lora import merge_lora_adapter
from ir_training.qat_mtp.workflow import summarize_issues, validate_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or merge a LoRA adapter into the QAT-derived Gemma 4 E2B target.")
    parser.add_argument(
        "--config",
        default="training/configs/models/gemma4_e2b_ir_qat_lora.yaml",
        help="Recommended target-only training config.",
    )
    parser.add_argument("--adapter-dir", help="Override qat_mtp.merge_adapter_dir.")
    parser.add_argument("--output-dir", help="Override qat_mtp.merged_model_dir.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Load the base and adapter and perform the merge. Without this flag only a plan is printed.",
    )
    parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty merged-model directory.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config).resolve())
    validation = summarize_issues(validate_training_config(config))
    if not validation["ok"]:
        raise SystemExit(json.dumps({"validation": validation}, indent=2, ensure_ascii=False))
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    workflow = config.get("qat_mtp") if isinstance(config.get("qat_mtp"), dict) else {}
    adapter_dir = args.adapter_dir or workflow.get("merge_adapter_dir")
    output_dir = args.output_dir or workflow.get("merged_model_dir")
    base = training_root()
    plan = {
        "base_model_id": model.get("model_id"),
        "base_model_source": str(
            resolve_path(model.get("model_source"), base)
        )
        if model.get("model_source")
        else model.get("model_id"),
        "mobile_training_seed_manifest": str(
            resolve_path(model.get("mobile_training_seed_manifest"), base)
        )
        if model.get("mobile_training_seed_manifest")
        else None,
        "adapter_dir": str(resolve_path(adapter_dir, base)),
        "merged_model_dir": str(resolve_path(output_dir, base)),
        "model_loader": model.get("model_loader", "auto_causal_lm"),
        "dtype": model.get("dtype", "bfloat16"),
        "continued_qat_performed": False,
        "packed_int4_output": False,
        "assistant_modified": False,
        "requires_post_merge_quantization": True,
        "validation": validation,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print("Plan only: no model was downloaded or loaded and no adapter was merged.")
        return

    merged_output_path = Path(plan["merged_model_dir"])
    destination_has_content = merged_output_path.exists() and (
        not merged_output_path.is_dir() or any(merged_output_path.iterdir())
    )
    if destination_has_content and not args.force:
        raise SystemExit(
            f"Refusing to write into non-empty merged-model directory without --force: {merged_output_path}"
        )

    merged_dir = merge_lora_adapter(
        base_model_id=str(model.get("model_id") or ""),
        adapter_dir=str(adapter_dir or ""),
        output_dir=str(output_dir or ""),
        model_loader=str(model.get("model_loader") or "auto_causal_lm"),
        dtype=str(model.get("dtype") or "bfloat16"),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        processor_model_id=str(model.get("model_id") or ""),
        training_config_path=Path(args.config).resolve(),
        base_model_source=model.get("model_source"),
        mobile_training_seed_manifest=model.get(
            "mobile_training_seed_manifest"
        ),
    )
    plan["merged_model_dir"] = str(merged_dir)
    plan["executed"] = True
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
