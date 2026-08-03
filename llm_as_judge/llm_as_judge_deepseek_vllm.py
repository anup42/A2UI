from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


# Fill these in directly if you prefer editing the script instead of passing CLI args.
MODEL_PATH = os.environ.get("GENUI_JUDGE_MODEL", "")
INPUT_IMAGE = r""
INPUT_DIR = r""
OUTPUT_JSON = r"ui_scores.json"
TASK_DESCRIPTION = r"Is this UI professional and aesthetically pleasing?"

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.50
MAX_MODEL_LEN = 4096
DTYPE = "bfloat16"
TRUST_REMOTE_CODE = True

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 900

DIMENSIONS: dict[str, str] = {
    "visual_appeal": "How polished, attractive, and modern the UI looks.",
    "layout_hierarchy": "How well the layout guides attention and organizes information.",
    "readability": "How clear, legible, and easy the text/content is to scan.",
    "consistency": "How consistent the spacing, alignment, styling, and component usage are.",
    "usability": "How obvious the interactions, navigation, and actions appear to be.",
    "task_alignment": (
        "How well the UI supports the intended task. If no task is provided, infer the"
        " likely main task from the visible screen and judge against that."
    ),
    "professionalism": "How professional and polished the overall design appears.",
    "production_readiness": "How production-ready and complete the UI appears.",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score generated UI screenshots with a local DeepSeek-VL2-Tiny model via vLLM."
    )
    parser.add_argument("--model-path", default=MODEL_PATH or None, help="Local model directory.")
    parser.add_argument("--image", default=INPUT_IMAGE or None, help="Path to one UI image.")
    parser.add_argument(
        "--image-dir",
        default=INPUT_DIR or None,
        help="Directory of UI images to score. Supported: png/jpg/jpeg/webp/bmp",
    )
    parser.add_argument("--output", default=OUTPUT_JSON, help="Output JSON path.")
    parser.add_argument(
        "--task-description",
        default=TASK_DESCRIPTION or None,
        help="Optional task context for task_alignment scoring.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=TENSOR_PARALLEL_SIZE,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=GPU_MEMORY_UTILIZATION,
        help="vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=MAX_MODEL_LEN,
        help="vLLM max model length.",
    )
    parser.add_argument("--dtype", default=DTYPE, help="vLLM dtype, e.g. float16 or bfloat16.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=TRUST_REMOTE_CODE,
        help="Pass trust_remote_code=True to vLLM if your local model requires it.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature. 0.0 is recommended for stable scoring.",
    )
    parser.add_argument("--top-p", type=float, default=TOP_P, help="Sampling top_p.")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Max output tokens.")
    return parser.parse_args()


def resolve_image_paths(single_image: str | None, image_dir: str | None) -> list[Path]:
    paths: list[Path] = []

    if single_image:
        path = Path(single_image).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        paths.append(path)

    if image_dir:
        directory = Path(image_dir).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Image directory not found: {directory}")
        dir_paths = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths.extend(dir_paths)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)

    if not deduped:
        raise SystemExit("Provide --image or --image-dir, or fill INPUT_IMAGE / INPUT_DIR in the script.")

    return deduped


def build_schema() -> dict[str, Any]:
    dimension_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(DIMENSIONS.keys()),
        "properties": {
            key: {
                "type": "object",
                "additionalProperties": False,
                "required": ["score", "reason"],
                "properties": {
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "reason": {"type": "string"},
                },
            }
            for key in DIMENSIONS
        },
    }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["dimensions", "overall_score", "overall_reason"],
        "properties": {
            "dimensions": dimension_schema,
            "overall_score": {"type": "number", "minimum": 1, "maximum": 5},
            "overall_reason": {"type": "string"},
        },
    }


def build_ui_judge_messages(image_path: str, user_question: str):
    judge_prompt = f"""
Evaluate this UI screenshot.
Question: {user_question}

Score each dimension on a scale of 1 to 5:
- very bad: 1
- bad: 2
- average: 3
- above average/good: 4
- very good: 5

ONLY use whole numbers for each dimension [Do NOT use decimals e.g. 3.5]

Dimensions to evaluate:
visual_appeal, layout_hierarchy, readability,
consistency, usability, task_alignment,
professionalism, production_readiness.

Return ONLY JSON:
{{
 "overall_score": 0,
 "dimension_scores": {{
   "visual_appeal": 0,
   "layout_hierarchy": 0,
   "readability": 0,
   "consistency": 0,
   "usability": 0,
   "task_alignment": 0,
   "professionalism": 0,
   "production_readiness": 0
 }},
 "verdict": "",
 "strengths": [],
 "weaknesses": [],
 "reason": ""
}}
""".strip()

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"file://{os.path.abspath(image_path)}"
                    }
                },
                {
                    "type": "text",
                    "text": judge_prompt
                }
            ],
        }
    ]


def load_image(path: Path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Missing dependency: Pillow. Install it with `pip install pillow`.") from exc

    with Image.open(path) as image:
        return image.convert("RGB")


def build_llm(args: argparse.Namespace):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError("Missing dependency: vLLM. Install it in your inference environment.") from exc

    if not args.model_path:
        raise SystemExit("Set MODEL_PATH in the script or pass --model-path.")

    return LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
        allowed_local_media_path=str(
            Path(args.image_dir).expanduser().resolve()
            if args.image_dir
            else Path(args.image).expanduser().resolve().parent
            if args.image
            else Path.cwd().resolve()
        ),
    )


def build_sampling_params(args: argparse.Namespace):
    first_error: Exception | None = None

    try:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        structured_outputs = StructuredOutputsParams(json=build_schema())
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            structured_outputs=structured_outputs,
        )
    except Exception as exc:
        first_error = exc

    try:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        guided_decoding = GuidedDecodingParams(json=build_schema())
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            guided_decoding=guided_decoding,
        )
    except Exception as exc:
        raise ImportError(
            "Could not import vLLM structured output helpers. Upgrade vLLM to a version with "
            "`StructuredOutputsParams` or `GuidedDecodingParams`."
        ) from (first_error or exc)


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        return json.loads(text[start : end + 1])


def validate_result(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Model output is not a JSON object.")

    dimensions = data.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("Missing `dimensions` object in model output.")

    cleaned_dimensions: dict[str, dict[str, Any]] = {}
    for key in DIMENSIONS:
        item = dimensions.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"Missing dimension entry: {key}")

        score = item.get("score")
        reason = item.get("reason")
        if not isinstance(score, int):
            raise ValueError(f"`{key}.score` must be an integer.")
        if score < 1 or score > 5:
            raise ValueError(f"`{key}.score` must be between 1 and 5.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"`{key}.reason` must be a non-empty string.")

        cleaned_dimensions[key] = {
            "score": score,
            "reason": reason.strip(),
        }

    overall_score = float(data.get("overall_score"))
    if overall_score < 1 or overall_score > 5:
        raise ValueError("`overall_score` must be between 1 and 5.")

    overall_reason = data.get("overall_reason")
    if not isinstance(overall_reason, str) or not overall_reason.strip():
        raise ValueError("`overall_reason` must be a non-empty string.")

    average_score = round(
        sum(item["score"] for item in cleaned_dimensions.values()) / len(cleaned_dimensions),
        2,
    )

    return {
        "dimensions": cleaned_dimensions,
        "overall_score": round(overall_score, 2),
        "overall_reason": overall_reason.strip(),
        "overall_score_from_dimensions": average_score,
    }


def score_image(llm, sampling_params, prompt: str, image_path: Path) -> dict[str, Any]:
    request = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": load_image(image_path),
        },
    }
    outputs = llm.generate([request], sampling_params=sampling_params, use_tqdm=False)
    raw_text = outputs[0].outputs[0].text
    parsed = validate_result(parse_json_response(raw_text))
    return {
        "image_path": str(image_path),
        "result": parsed,
    }


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_folder(
    folder_path: str,
    task_description: str,
    llm,
    sampling_params,
    output_dir: str,
):
    """Process a folder of images using the new scoring approach."""
    # Convert folder_path to Path and resolve it
    folder = Path(folder_path).expanduser().resolve()

    # Find all image files in the folder
    image_paths = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"\nNo image files found in folder: {folder_path}")
        return

    print("\n" + "=" * 100)
    print(f"Processing folder: {folder_path}")
    print(f"Found {len(image_paths)} image files")

    # Build prompt once for all images in the folder
    prompt = build_prompt(task_description)

    results: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Scoring {image_path.name}")
        try:
            result = score_image(llm, sampling_params, prompt, image_path)
            overall = result["result"]["overall_score"]
            avg_score = result["result"]["overall_score_from_dimensions"]
            print(f"  overall_score={overall} average_dimension_score={avg_score}")
            results.append(result)
        except Exception as exc:
            print(f"  failed: {exc}")
            results.append(
                {
                    "image_path": str(image_path),
                    "error": str(exc),
                }
            )

    # Create output directory if it doesn't exist
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results to JSON file
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_path": str(Path(MODEL_PATH).expanduser().resolve()) if MODEL_PATH else None,
        "task_description": task_description,
        "scoring_scale": "1-5",
        "dimensions_definition": DIMENSIONS,
        "results": results,
    }

    # Save to a file named after the folder
    folder_name = folder.name
    json_output_path = output_path / f"{folder_name}_ui_scores.json"
    write_output(json_output_path, payload)
    print(f"\nSaved JSON to: {json_output_path}")


def main() -> None:
    args = parse_args()

    # If specific image or directory is provided via CLI, use that
    # Otherwise, check if folders were provided as positional args (old format)
    if args.image or args.image_dir:
        image_paths = resolve_image_paths(args.image, args.image_dir)
        prompt = build_prompt(args.task_description)

        llm = build_llm(args)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=512,
            top_p=1.0,
            top_k=-1,
            stop_token_ids=[],
        )

        results: list[dict[str, Any]] = []
        for index, image_path in enumerate(image_paths, start=1):
            print(f"[{index}/{len(image_paths)}] Scoring {image_path.name}")
            try:
                result = score_image(llm, sampling_params, prompt, image_path)
                overall = result["result"]["overall_score"]
                avg_score = result["result"]["overall_score_from_dimensions"]
                print(f"  overall_score={overall} average_dimension_score={avg_score}")
                results.append(result)
            except Exception as exc:
                print(f"  failed: {exc}")
                results.append(
                    {
                        "image_path": str(image_path),
                        "error": str(exc),
                    }
                )

        output_path = Path(args.output).expanduser().resolve()
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_path": str(Path(args.model_path).expanduser().resolve()) if args.model_path else None,
            "task_description": args.task_description,
            "scoring_scale": "1-5",
            "dimensions_definition": DIMENSIONS,
            "results": results,
        }
        write_output(output_path, payload)
        print(f"Saved JSON to: {output_path}")
    else:
        # Handle the old format with positional folder arguments
        import sys
        if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
            # Positional arguments are folder paths
            folder_paths = [arg for arg in sys.argv[1:] if not arg.startswith('--')]

            # Use DeepSeek model with parameters similar to InternLM
            model_args = argparse.Namespace(
                model_path=MODEL_PATH,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                max_model_len=MAX_MODEL_LEN,
                dtype=DTYPE,
                trust_remote_code=TRUST_REMOTE_CODE,
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
            )

            llm = build_llm(model_args)
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=512,
                top_p=1.0,
                top_k=-1,
                stop_token_ids=[],
            )

            for folder_path in folder_paths:
                process_folder(
                    folder_path=folder_path,
                    task_description=TASK_DESCRIPTION,
                    llm=llm,
                    sampling_params=sampling_params,
                    output_dir=folder_path,
                )
        else:
            print("Please provide either --image/--image-dir options or folder paths as arguments.")


if __name__ == "__main__":
    main()
