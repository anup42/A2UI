# -*- coding: utf-8 -*-
"""
Few-shot UI Judge with example images in prompt.
Includes one good, one bad, and one average UI example to guide scoring.
"""
import os
import json
import re
import argparse
from typing import Any, Dict, List, Optional

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

DEFAULT_MODEL_PATH = os.environ.get("GENUI_JUDGE_MODEL", "")

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Example images for few-shot prompting (relative paths)
FEWSHOT_EXAMPLES = {
    "bad": {
        "image_path": "bad_prompt.png",
        "scores": {
            "visual_appeal": 1,
            "visual_hierarchy": 1,
            "readability": 1,
            "color_harmony": 1
        },
        "feedback": {
            "visual_appeal": "Extremely cluttered, overlapping elements, no spacing, visually chaotic.",
            "visual_hierarchy": "No clear focus, everything competes for attention equally.",
            "readability": "Hard to read due to excessive colors, fonts, and noise.",
            "color_harmony": "Clashing, overly saturated colors with no consistency."
        },
        "verdict": "Poor UI with major design flaws.",
        "reason": "Extremely cluttered with no visual organization, clashing colors, and unreadable text."
    },
    "average": {
        "image_path": "average_prompt.png",
        "scores": {
            "visual_appeal": 3,
            "visual_hierarchy": 3,
            "readability": 3,
            "color_harmony": 3
        },
        "feedback": {
            "visual_appeal": "Clean but very basic, lacks refinement or visual polish.",
            "visual_hierarchy": "Structure exists but weak emphasis; important elements not strongly highlighted.",
            "readability": "Readable but not optimized (font sizes/spacing feel standard, not refined).",
            "color_harmony": "Neutral and safe colors, but not well-designed or visually engaging."
        },
        "verdict": "Average UI that functions but lacks polish.",
        "reason": "Functional and usable but lacks refinement in all dimensions."
    },
    "good": {
        "image_path": "good_prompt.png",
        "scores": {
            "visual_appeal": 5,
            "visual_hierarchy": 5,
            "readability": 5,
            "color_harmony": 5
        },
        "feedback": {
            "visual_appeal": "Clean, well-spaced, polished and visually balanced.",
            "visual_hierarchy": "Clear focus and strong prioritization; attention flows naturally.",
            "readability": "Excellent clarity with proper font sizes, spacing, and contrast.",
            "color_harmony": "Cohesive, consistent, and aesthetically pleasing palette."
        },
        "verdict": "Excellent UI with professional design quality.",
        "reason": "Clean, polished design with clear hierarchy, excellent readability, and harmonious colors."
    }
}


# ------------------------------------------------------------
# Few-shot multimodal prompt builder
# ------------------------------------------------------------
def build_fewshot_ui_judge_messages(image_path: str, user_question: str, processor):
    """Build few-shot prompt with example images and their scores."""

    # Build the system instruction
    system_instruction = """You are an expert UI quality evaluator. Score UI screenshots on 4 dimensions from 1 to 5.

Scoring Scale:
- 1: Very Bad - Major issues, severely lacking
- 2: Bad - Noticeable issues, below average
- 3: Average - Acceptable but unimpressive
- 4: Good - Well executed with minor issues
- 5: Very Good - Excellent, nearly flawless

Dimensions:
1. visual_appeal: How aesthetically pleasing and visually polished the screen looks overall.
2. visual_hierarchy: How clearly the design shows what is most important and guides user attention.
3. readability: How easy the text is to read considering font size, contrast, and spacing.
4. color_harmony: How well the colors work together and support the interface.

ONLY use whole numbers (1, 2, 3, 4, or 5) - NO decimals.

Here are some examples:"""

    messages = []

    # Add few-shot examples
    for example_type, example in FEWSHOT_EXAMPLES.items():
        example_image_path = example["image_path"]
        if os.path.exists(example_image_path):
            example_json = {
                "dimension_scores": example["scores"],
                "dimension_feedback": example["feedback"],
                "verdict": example["verdict"],
                "reason": example["reason"]
            }

            example_message = f"""Example ({example_type.upper()}):
{json.dumps(example_json, indent=2)}"""

            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example_message},
                ],
            })

    # Add the actual evaluation request
    eval_request = f"""Now evaluate this UI screenshot.

Question: {user_question}

Return ONLY JSON:
{{
  "dimension_scores": {{
    "visual_appeal": <score 1-5>,
    "visual_hierarchy": <score 1-5>,
    "readability": <score 1-5>,
    "color_harmony": <score 1-5>
  }},
  "dimension_feedback": {{
    "visual_appeal": "<one line explanation>",
    "visual_hierarchy": "<one line explanation>",
    "readability": "<one line explanation>",
    "color_harmony": "<one line explanation>"
  }},
  "verdict": "<overall verdict in one sentence>",
  "reason": "<brief explanation of overall assessment>"
}}"""

    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": eval_request},
        ],
    })

    return messages


# ------------------------------------------------------------
# Helper: convert chat messages -> vLLM input with multiple images
# ------------------------------------------------------------
def prepare_fewshot_inputs_for_vllm(
    image_path: str,
    messages: List[Dict[str, Any]],
    processor,
):
    """Prepare inputs for vLLM with few-shot example images."""

    # Build prompt with all images
    prompt_parts = []
    images = []

    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            for item in content:
                if item["type"] == "image":
                    # This is a placeholder - we'll load images separately
                    pass
                elif item["type"] == "text":
                    prompt_parts.append(item["text"])

    # Load all images (few-shot examples + target image)
    for example_type, example in FEWSHOT_EXAMPLES.items():
        example_path = example["image_path"]
        if os.path.exists(example_path):
            images.append(Image.open(example_path).convert("RGB"))

    # Load target image
    images.append(Image.open(image_path).convert("RGB"))

    # Build the full prompt using processor's chat template
    full_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Debug: verify images are loaded
    print(f"[DEBUG] Loaded {len(images)} images for few-shot prompt")
    print(f"[DEBUG] Prompt length: {len(full_prompt)} chars")

    return {
        "prompt": full_prompt,
        "multi_modal_data": {
            "image": images
        },
    }


# ------------------------------------------------------------
# Main prompt builder with few-shot examples as text
# ------------------------------------------------------------
def build_ui_judge_messages_with_fewshot(image_path: str, user_question: str):
    """Build prompt with few-shot examples included as text descriptions."""

    judge_prompt = f"""You are an expert UI quality evaluator. Score this UI screenshot on 4 dimensions from 1 to 5.

Scoring Scale:
- 1: Very Bad - Major issues, severely lacking
- 2: Bad - Noticeable issues, below average
- 3: Average - Acceptable but unimpressive
- 4: Good - Well executed with minor issues
- 5: Very Good - Excellent, nearly flawless

Dimensions:
1. visual_appeal: How aesthetically pleasing and visually polished the screen looks overall.
2. visual_hierarchy: How clearly the design shows what is most important and guides user attention.
3. readability: How easy the text is to read considering font size, contrast, and spacing.
4. color_harmony: How well the colors work together and support the interface.

ONLY use whole numbers (1, 2, 3, 4, or 5) - NO decimals.

=== FEW-SHOT EXAMPLES ===

EXAMPLE 1 - GOOD UI (Score: 5/5 on all dimensions):
Visual Appeal: Clean, modern design with excellent visual polish and balanced layout.
Visual Hierarchy: Clear focal points with well-defined primary and secondary elements.
Readability: Text is crisp with excellent contrast and comfortable spacing.
Color Harmony: Cohesive color palette that enhances the user experience.
Verdict: Excellent UI with professional design quality.

EXAMPLE 2 - BAD UI (Score: 1/5 on all dimensions):
Visual Appeal: Cluttered, messy layout with no visual polish.
Visual Hierarchy: No clear structure; all elements compete for attention.
Readability: Text is hard to read due to poor contrast and sizing.
Color Harmony: Colors clash and create visual discomfort.
Verdict: Poor UI with major design flaws.

EXAMPLE 3 - AVERAGE UI (Score: 3/5 on all dimensions):
Visual Appeal: Acceptable design but lacks visual refinement.
Visual Hierarchy: Basic structure exists but emphasis could be clearer.
Readability: Text is readable but could benefit from better spacing.
Color Harmony: Colors are acceptable but not particularly refined.
Verdict: Average UI that functions but lacks polish.

=== NOW EVALUATE THIS UI ===

Question: {user_question}

Return ONLY JSON:
{{
  "dimension_scores": {{
    "visual_appeal": <score 1-5>,
    "visual_hierarchy": <score 1-5>,
    "readability": <score 1-5>,
    "color_harmony": <score 1-5>
  }},
  "dimension_feedback": {{
    "visual_appeal": "<one line explanation>",
    "visual_hierarchy": "<one line explanation>",
    "readability": "<one line explanation>",
    "color_harmony": "<one line explanation>"
  }},
  "verdict": "<overall verdict in one sentence>",
  "reason": "<brief explanation of overall assessment>"
}}"""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": judge_prompt},
            ],
        }
    ]


# ------------------------------------------------------------
# Helper: convert simple chat messages -> vLLM input
# ------------------------------------------------------------
def prepare_inputs_for_vllm(
    image_path: str,
    messages: List[Dict[str, Any]],
    processor,
):
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image = Image.open(image_path).convert("RGB")

    return {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    }


# ------------------------------------------------------------
# JSON extraction helper
# ------------------------------------------------------------
def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError("JSON not found in model output")


# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------
def get_pngs_from_folder(folder_path: str) -> List[str]:
    if not os.path.isdir(folder_path):
        return []

    return [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(".png")
    ]


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def compute_weighted_overall_score(dimension_scores: Dict[str, Any]) -> float:
    """Calculate weighted overall score: 0.25 weight for each of 4 dimensions."""
    if not isinstance(dimension_scores, dict):
        return 0.0
    dimension_keys = ["visual_appeal", "visual_hierarchy", "readability", "color_harmony"]
    total = sum(safe_float(dimension_scores.get(k, 0.0)) for k in dimension_keys)
    return round(0.25 * total, 4)


def compute_folder_summary(image_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_results = [
        r for r in image_results
        if "overall_score" in r
    ]

    num_images = len(image_results)
    num_valid = len(valid_results)

    dimension_keys = [
        "visual_appeal",
        "visual_hierarchy",
        "readability",
        "color_harmony",
    ]

    if num_valid == 0:
        return {
            "num_images": num_images,
            "num_valid_scores": 0,
            "average_dimension_scores": {k: None for k in dimension_keys},
            "average_overall_score": None,
        }

    avg_dimension_scores = {}
    for key in dimension_keys:
        vals = [safe_float(r.get("dimension_scores", {}).get(key, 0.0)) for r in valid_results]
        avg_dimension_scores[key] = round(sum(vals) / num_valid, 4)

    avg_overall = sum(safe_float(r.get("overall_score", 0.0)) for r in valid_results) / num_valid

    return {
        "num_images": num_images,
        "num_valid_scores": num_valid,
        "average_dimension_scores": avg_dimension_scores,
        "average_overall_score": round(avg_overall, 4),
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Main folder processor
# ------------------------------------------------------------
def process_folder(
    folder_path: str,
    user_prompt: str,
    processor,
    llm,
    sampling_params,
    output_dir: str,
    use_fewshot: bool = True,
    n_iterations: int = 1,
):
    image_list = get_pngs_from_folder(folder_path)

    if not image_list:
        print(f"\nNo PNG files found in folder: {folder_path}")
        return

    print("\n" + "=" * 100)
    print(f"Processing folder: {folder_path}")
    print(f"Found {len(image_list)} PNG files")
    print(f"Few-shot mode: {use_fewshot}")
    print(f"Iterations per image: {n_iterations}")

    # Store all iteration results per image
    all_iteration_results = {img: [] for img in image_list}

    # Run n_iterations
    for iteration in range(n_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*50}")

        inputs = []
        for img in image_list:
            # Use actual few-shot images
            messages = build_fewshot_ui_judge_messages(img, user_prompt, processor)
            vllm_input = prepare_fewshot_inputs_for_vllm(
                image_path=img,
                messages=messages,
                processor=processor,
            )
            inputs.append(vllm_input)

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for img, output in zip(image_list, outputs):
            generated_text = output.outputs[0].text
            print(f"\n--- Image: {img} (Iteration {iteration + 1}) ---")

            try:
                parsed = extract_json(generated_text)
                parsed["iteration"] = iteration + 1
                parsed["raw_output"] = generated_text
                parsed["overall_score"] = compute_weighted_overall_score(parsed.get("dimension_scores", {}))
                all_iteration_results[img].append(parsed)
                print(f"Scores: {parsed.get('dimension_scores', {})}")
            except Exception as e:
                print(f"JSON parsing failed: {e}")
                all_iteration_results[img].append({
                    "iteration": iteration + 1,
                    "error": str(e),
                    "raw_output": generated_text,
                })

    # Average results across iterations
    image_results: List[Dict[str, Any]] = []
    dimension_keys = ["visual_appeal", "visual_hierarchy", "readability", "color_harmony"]

    for img in image_list:
        iterations_data = all_iteration_results[img]
        valid_iterations = [r for r in iterations_data if "error" not in r]

        if not valid_iterations:
            image_results.append({
                "image": os.path.basename(img),
                "image_path": img,
                "error": "All iterations failed",
            })
            continue

        # Average dimension scores
        avg_dimension_scores = {}
        for key in dimension_keys:
            scores = [safe_float(r.get("dimension_scores", {}).get(key, 0.0)) for r in valid_iterations]
            avg_dimension_scores[key] = round(sum(scores) / len(scores), 4)

        avg_overall = compute_weighted_overall_score(avg_dimension_scores)

        result = {
            "image": os.path.basename(img),
            "image_path": img,
            "dimension_scores": avg_dimension_scores,
            "overall_score": avg_overall,
            "n_iterations": n_iterations,
            "n_valid_iterations": len(valid_iterations),
            "all_iterations": iterations_data,
        }

        print(f"\n=== AVERAGED RESULT for {img} ===")
        print(f"Avg Scores: {avg_dimension_scores}")
        print(f"Avg Overall: {avg_overall}")

        image_results.append(result)

    summary = compute_folder_summary(image_results)

    folder_name = os.path.basename(os.path.normpath(folder_path))
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, f"{folder_name}_image_scores_fewshot.jsonl")
    summary_json_path = os.path.join(output_dir, f"{folder_name}_summary_fewshot.json")

    write_jsonl(jsonl_path, image_results)

    summary_payload = {
        "summary": summary,
        "folder_path": folder_path,
        "folder_name": folder_name,
        "prompt": user_prompt,
        "image_scores_jsonl": jsonl_path,
        "fewshot_mode": use_fewshot,
        "n_iterations": n_iterations,
        "temperature": temperature,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved image-wise results to: {jsonl_path}")
    print(f"Saved folder summary to: {summary_json_path}")


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folders",
        nargs="+",
        help="One or more folder paths containing PNG images",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH or None,
        required=not bool(DEFAULT_MODEL_PATH),
        help="Local model directory (or set GENUI_JUDGE_MODEL).",
    )
    parser.add_argument(
        "--prompt",
        default="Is this UI professional and aesthetically pleasing?",
        help="Prompt used for judging all images",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory where outputs will be stored. Defaults to each folder itself.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--no-fewshot",
        action="store_true",
        help="Disable few-shot prompting (use plain prompt)",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=1,
        help="Number of iterations to run per image (results will be averaged)",
    )

    args = parser.parse_args()

    # Few-shot is enabled by default, disable with --no-fewshot
    use_fewshot = not args.no_fewshot
    n_iterations = args.n_iterations

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Need 4 images: 3 few-shot examples + 1 target image
    llm = LLM(
        model=args.model_path,
        runner="generate",
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 4},
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Use temperature=0.7 when n_iterations > 1, otherwise temperature=0
    temperature = 0.7 if n_iterations > 1 else 0

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=512,
        top_k=-1,
        seed=None,
    )

    print(f"\nSampling params: temperature={temperature}, n_iterations={n_iterations}")

    for folder_path in args.folders:
        out_dir = args.output_dir or folder_path
        process_folder(
            folder_path=folder_path,
            user_prompt=args.prompt,
            processor=processor,
            llm=llm,
            sampling_params=sampling_params,
            output_dir=out_dir,
            use_fewshot=use_fewshot,
            n_iterations=n_iterations,
        )
