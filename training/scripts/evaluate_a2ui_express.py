from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset" / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Gemma A2UI Express model with training-identical prompts.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--print-output", action="store_true")
    args = parser.parse_args()

    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from pipeline.ir_formats import decode_to_flat_spec, semantic_hash  # type: ignore

    rows = []
    with Path(args.split).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.max_rows:
                break
    if not rows:
        raise SystemExit("No evaluation rows found")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    predictions: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        messages = list(row.get("messages") or [])
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        elapsed = time.perf_counter() - started
        generated = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True).strip()

        envelope_ok = generated.startswith("<a2ui>") and generated.endswith("</a2ui>")
        decode_ok = False
        semantic_match = False
        decode_error = None
        try:
            decoded = decode_to_flat_spec(generated, format_hint="a2ui_express_v1")
            decode_ok = True
            semantic_match = semantic_hash(decoded.flat_spec) == str(row.get("semantic_hash") or "")
        except Exception as exc:  # quality report should retain malformed outputs
            decode_error = str(exc)

        predictions.append(
            {
                "id": row.get("id"),
                "generated_text": generated,
                "expected_text": row.get("completion"),
                "exact_match": generated == str(row.get("completion") or "").strip(),
                "envelope_ok": envelope_ok,
                "express_decode_ok": decode_ok,
                "semantic_match": semantic_match,
                "decode_error": decode_error,
                "input_tokens": input_length,
                "output_tokens": int(generated_ids.shape[-1] - input_length),
                "latency_seconds": elapsed,
            }
        )
        if args.print_output:
            print(f"--- generated row {index} ---\n{generated}\n--- end row {index} ---", flush=True)
        print(f"row={index}/{len(rows)} decode_ok={decode_ok} semantic_match={semantic_match} output_tokens={predictions[-1]['output_tokens']}", flush=True)

    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    summary = {
        "model_dir": str(Path(args.model_dir)),
        "split": str(Path(args.split)),
        "sample_count": len(predictions),
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "training_prompt_compatible": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "envelope_rate": sum(item["envelope_ok"] for item in predictions) / len(predictions),
        "express_decode_rate": sum(item["express_decode_ok"] for item in predictions) / len(predictions),
        "semantic_match_rate": sum(item["semantic_match"] for item in predictions) / len(predictions),
        "exact_match_rate": sum(item["exact_match"] for item in predictions) / len(predictions),
        "avg_input_tokens": sum(item["input_tokens"] for item in predictions) / len(predictions),
        "avg_output_tokens": sum(item["output_tokens"] for item in predictions) / len(predictions),
        "avg_latency_seconds": sum(item["latency_seconds"] for item in predictions) / len(predictions),
        "predictions": str(predictions_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
