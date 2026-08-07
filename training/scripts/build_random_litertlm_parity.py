"""Build a random-weight LiteRT-LM candidate and compare it with an artifact.

This is an opt-in conversion audit, not a training command.  The default mode
prints a plan and never imports Transformers, allocates a model, or invokes
``litert-torch``.  ``--execute`` creates a randomly initialized model from the
source model's config, exports it with the requested LiteRT recipe, and writes
separate graph/quantization/weight comparison results.

A random model can prove that an exporter produced the same graph topology and
quantization *layout*.  It cannot prove equal learned weights, calibration
scales, tokenizer metadata, or Google's private training recipe.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.export.litertlm_inspector import (  # noqa: E402
    LiteRTLMInspectionError,
    inspect_and_compare,
)


class RandomParityError(RuntimeError):
    """Raised when the explicit random export audit cannot be completed."""


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    random_model_dir = output_dir / "random_hf"
    litert_output_dir = output_dir / "litertlm"
    command = [
        args.converter,
        "export_hf",
        f"--model={random_model_dir}",
        f"--output_dir={litert_output_dir}",
        f"--quantization_recipe={args.quantization_recipe}",
        *args.extra_flag,
    ]
    return {
        "execute": bool(args.execute),
        "training_executed": False,
        "random_initialization": True,
        "model_source": str(args.model_source),
        "official_artifact": str(Path(args.official_artifact).expanduser().resolve()),
        "output_dir": str(output_dir),
        "random_model_dir": str(random_model_dir),
        "litert_output_dir": str(litert_output_dir),
        "quantization_recipe": args.quantization_recipe,
        "seed": args.seed,
        "command": command,
        "comparison_levels": [
            "section layout and metadata",
            "embedded graph topology",
            "quantization layout (dtype/scale-count/axis)",
            "quantization values (scales/zero-points)",
            "weight bytes",
        ],
    }


def _load_random_model(source: str, destination: Path, seed: int) -> None:
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - conversion environment only
        raise RandomParityError(
            "--execute requires torch and transformers. Install the training/export environment first."
        ) from exc

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        config = AutoConfig.from_pretrained(source, trust_remote_code=False)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    except Exception as exc:  # pragma: no cover - architecture/tool version dependent
        raise RandomParityError(
            "Could not instantiate a random AutoModelForCausalLM from the source config. "
            "For multimodal Gemma 4, use the same model class/configuration that the pinned "
            "LiteRT exporter expects and provide its resulting random HF directory to the converter."
        ) from exc

    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(destination, safe_serialization=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        tokenizer.save_pretrained(destination)
    except Exception as exc:  # pragma: no cover - tokenizer is exporter/version dependent
        (destination / "tokenizer_warning.txt").write_text(
            "Tokenizer was not copied automatically. The LiteRT exporter may require tokenizer files.\n"
            f"Source: {source}\nReason: {exc}\n",
            encoding="utf-8",
        )


def _run_converter(plan: dict[str, Any], log_path: Path) -> list[Path]:
    command = [str(item) for item in plan["command"]]
    executable = shutil.which(command[0])
    if executable is None:
        raise RandomParityError(
            f"LiteRT converter {command[0]!r} is not on PATH. Install training/requirements-edge-export.txt."
        )
    command[0] = executable
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command:\n" + " ".join(command) + "\n\n")
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(process.stdout or "")
    if process.returncode != 0:
        raise RandomParityError(f"LiteRT conversion failed with exit code {process.returncode}; see {log_path}.")
    candidates = sorted(Path(plan["litert_output_dir"]).rglob("*.litertlm"))
    if not candidates:
        raise RandomParityError(f"Converter produced no .litertlm file under {plan['litert_output_dir']}.")
    return candidates


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _plan(args)
    output_dir = Path(plan["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise RandomParityError(
            f"Output directory is non-empty: {output_dir}. Choose a new path or pass --force; no files were removed."
        )
    if Path(plan["random_model_dir"]).exists() or Path(plan["litert_output_dir"]).exists():
        raise RandomParityError(
            "The planned random_hf or litertlm subdirectory already exists. Choose a new output directory; "
            "the parity audit never overwrites model/export outputs."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "random_litertlm_parity_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.execute:
        return plan

    _load_random_model(str(args.model_source), Path(plan["random_model_dir"]), args.seed)
    candidates = _run_converter(plan, output_dir / "litert_torch_export.log")
    official_artifact = Path(plan["official_artifact"])
    if not official_artifact.is_file():
        raise RandomParityError(f"Official artifact does not exist: {official_artifact}")

    candidate_reports = []
    for candidate in candidates:
        try:
            report = inspect_and_compare(
                official_artifact,
                candidate,
                include_hashes=args.include_hashes,
                inspect_tflite=not args.no_tflite,
                include_graph_details=args.graph_details,
            )
        except (OSError, LiteRTLMInspectionError) as exc:
            raise RandomParityError(f"Could not inspect {candidate}: {exc}") from exc
        comparison = report["comparison"]
        candidate_reports.append(
            {
                "candidate": str(candidate),
                "comparison": comparison,
                "same_observable_network": bool(
                    comparison.get("section_layout_match")
                    and comparison.get("graph_structure_match")
                    and comparison.get("quantization_layout_match")
                ),
                "learned_weight_equivalence": False,
                "report": report,
            }
        )

    result = {
        **plan,
        "candidates": candidate_reports,
        "decision": {
            "same_observable_network_for_any_candidate": any(
                item["same_observable_network"] for item in candidate_reports
            ),
            "exact_official_artifact_reproduction": False,
            "reason": (
                "Random initialization makes learned-weight equivalence false by construction. "
                "Even a graph/layout match does not reveal private calibration or QAT training details."
            ),
        },
    }
    (output_dir / "random_litertlm_parity_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or execute random-weight LiteRT-LM graph parity audit.")
    parser.add_argument("--model-source", required=True, help="HF model id or local model directory used only for config/tokenizer.")
    parser.add_argument("--official-artifact", required=True, help="Official .litertlm artifact to compare against.")
    parser.add_argument("--output-dir", required=True, help="New/empty directory for the random model, export, and report.")
    parser.add_argument("--quantization-recipe", required=True, help="Exact litert-torch recipe to audit (for example dynamic_wi8_afp32).")
    parser.add_argument("--converter", default="litert-torch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-flag", action="append", default=[], help="Additional flag passed to litert-torch; repeat as needed.")
    parser.add_argument("--include-hashes", action="store_true", help="Hash all sections; reads full official/candidate artifacts.")
    parser.add_argument("--no-tflite", action="store_true", help="Skip embedded TFLite graph inspection.")
    parser.add_argument("--graph-details", action="store_true", help="Include full graph details in the report.")
    parser.add_argument("--execute", action="store_true", help="Instantiate/save random weights and run the converter.")
    parser.add_argument("--force", action="store_true", help="Allow an existing empty/non-empty output directory; never deletes files.")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, RandomParityError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
