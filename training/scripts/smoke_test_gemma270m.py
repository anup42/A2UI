from __future__ import annotations

import argparse
import json
from pathlib import Path


SAMPLE_RESPONSE = """Weather comparison for Bengaluru
Today is warm with a high of 29 C and a low of 21 C. Rain chance is 35 percent.
Tomorrow is cloudy with a high of 27 C and a low of 20 C. Rain chance is 55 percent."""
IR_PREFIX = "Given an agent response you have to generate a structured intermediate representation. "


def extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "root" in value and "elements" in value:
            return value
    raise ValueError("Model output did not contain a JSON object")


def validate_flat_spec(value: dict) -> None:
    root = value.get("root")
    if not isinstance(root, str) or not root:
        raise ValueError("IR root must be a non-empty string")
    elements = value.get("elements")
    if not isinstance(elements, dict) or root not in elements:
        raise ValueError("IR must contain an element matching root")
    for element_id, element in elements.items():
        if not isinstance(element, dict):
            raise ValueError(f"Element {element_id} is not an object")
        if not isinstance(element.get("props"), dict) or not isinstance(element.get("children"), list):
            raise ValueError(f"Element {element_id} must have object props and array children")
        for child in element["children"]:
            if child not in elements:
                raise ValueError(f"Element {element_id} references missing child {child}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU smoke test for the merged Gemma 270M IR model.")
    parser.add_argument("--model", required=True, help="Merged Hugging Face model directory.")
    parser.add_argument("--prompt-template", required=True, help="Gemma Stage 3 prompt template.")
    parser.add_argument("--response", default=SAMPLE_RESPONSE)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--raw-response-only",
        action="store_true",
        help="Match the response_text-only SFT input (including its IR prefix).",
    )
    parser.add_argument("--training-prefix", default=IR_PREFIX)
    parser.add_argument("--output", help="Optional path for the generated IR JSON.")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for this smoke test")

    model_path = Path(args.model).resolve()
    template = Path(args.prompt_template).read_text(encoding="utf-8")
    stage3_prompt = (
        f"{args.training_prefix}{args.response}"
        if args.raw_response_only
        else template.replace("{response_text}", args.response)
    )
    # This is the exact plain-text boundary used by the 270M SFT corpus.
    # Do not add a trailing space after Assistant:; the model emits it with the
    # opening JSON token.
    prompt = f"User: {stage3_prompt.strip()}\nAssistant:"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    if args.output:
        raw_output_path = Path(args.output).resolve().with_suffix(".raw.txt")
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(output + "\n", encoding="utf-8")
    try:
        ir = extract_json(output)
    except ValueError:
        print(json.dumps({
            "gpu": torch.cuda.get_device_name(0),
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "output_tokens": int(generated.shape[1] - encoded["input_ids"].shape[1]),
            "raw_output": output,
        }, indent=2, ensure_ascii=False))
        raise
    validate_flat_spec(ir)

    rendered = json.dumps(ir, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "input_tokens": int(encoded["input_ids"].shape[1]),
        "output_tokens": int(generated.shape[1] - encoded["input_ids"].shape[1]),
        "element_count": len(ir["elements"]),
        "ir": ir,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
