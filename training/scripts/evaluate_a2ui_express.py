from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--config", help="Training YAML for tokenizer loader and exact chat-template kwargs. Otherwise use saved training_metadata.json when present.")
    parser.add_argument("--model-loader", choices=("auto_causal_lm", "auto_multimodal_lm"), help="Override the saved training loader")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--print-output", action="store_true")
    args = parser.parse_args()
    args.model_dir = str(Path(args.model_dir).expanduser().resolve())
    if not Path(args.model_dir).is_dir():
        raise FileNotFoundError(f"Missing full HF model directory: {args.model_dir}")
    if (Path(args.model_dir) / "adapter_config.json").is_file():
        raise ValueError("For PEFT adapters use evaluate_checkpoint_on_golden.py with the original --config.")

    import torch  # type: ignore
    from pipeline.ir_formats import validate_express_completion  # type: ignore
    from ir_training.train.sft import _tokenize_completion_only_row  # type: ignore
    from ir_training.models.hf_loading import load_hf_model
    from ir_training.models.registry import create_adapter
    from ir_training.common.config import load_yaml
    from ir_training.eval.generate import build_prediction_record, _extract_user_text
    from ir_training.eval.compare_to_baseline import evaluate_predictions
    from ir_training.generation_policy import build_stopping_criteria, preserve_generation_eos, generation_diagnostics, stop_express_completion

    rows = []
    with Path(args.split).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.max_rows:
                break
    if not rows:
        raise SystemExit("No evaluation rows found")

    if args.config:
        model_config = dict(load_yaml(Path(args.config).resolve()).get("model") or {})
    else:
        metadata_path = Path(args.model_dir) / "training_metadata.json"
        saved = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        model_config = dict(saved.get("model") or {})
    model_config.update(model_id=args.model_dir, model_source=args.model_dir, tokenizer_source=args.model_dir)
    model_config.setdefault("family", "gemma")
    adapter = create_adapter(model_config)
    tokenizer = adapter.load_tokenizer()
    model = load_hf_model(args.model_dir, {"model_loader": args.model_loader or model_config.get("model_loader", "auto_causal_lm"),
        "dtype": "bfloat16", "device_map": {"": 0}, "attn_implementation": "sdpa"})
    model.eval()
    device = next(model.parameters()).device
    eos_ids = preserve_generation_eos(model, tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    predictions: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        _extract_user_text(row)
        prompt = adapter.format_example(row, tokenizer=tokenizer, include_assistant=False)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        if input_length > args.max_input_tokens:
            raise ValueError(f"Evaluation row {index} exceeds max-input-tokens; source truncation is forbidden.")
        started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_ids,
                stop_strings=None,
                stopping_criteria=build_stopping_criteria(tokenizer, input_length),
            )
        elapsed = time.perf_counter() - started
        raw_generated = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
        generated = stop_express_completion(raw_generated).strip()
        runtime = generation_diagnostics(tokenizer, generated_ids[0][input_length:],
            eos_token_ids=eos_ids, max_new_tokens=args.max_new_tokens,
            prompt_text=prompt, input_ids=inputs["input_ids"][0])

        full_text = adapter.format_example(row, tokenizer=tokenizer, include_assistant=True)
        completion_text = (
            full_text[len(prompt) :]
            if prompt and full_text.startswith(prompt)
            else str(row.get("completion") or "")
        )
        teacher_forced = _tokenize_completion_only_row(
            tokenizer=tokenizer,
            prompt_text=prompt,
            completion_text=completion_text,
            full_text=full_text,
            max_seq_length=args.max_input_tokens,
        )
        teacher_input_ids = torch.tensor(
            [teacher_forced["input_ids"]], dtype=torch.long, device=device
        )
        teacher_labels = torch.tensor(
            [teacher_forced["labels"]], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            teacher_logits = model(input_ids=teacher_input_ids).logits
            shifted_logits = teacher_logits[:, :-1, :].float()
            shifted_labels = teacher_labels[:, 1:]
            valid_labels = shifted_labels.ne(-100)
            teacher_forced_tokens = int(valid_labels.sum().item())
            if teacher_forced_tokens <= 0:
                raise ValueError(f"Evaluation row {index} has no completion labels.")
            teacher_forced_loss = float(
                torch.nn.functional.cross_entropy(
                    shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                    shifted_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="mean",
                ).item()
            )
        if not math.isfinite(teacher_forced_loss):
            raise ValueError(
                f"Evaluation row {index} produced non-finite teacher-forced loss "
                f"{teacher_forced_loss}."
            )

        envelope_ok = generated.startswith("<a2ui>") and generated.endswith("</a2ui>")
        decode_ok = False
        semantic_match = False
        decode_error = None
        try:
            decoded = validate_express_completion(generated)
            decode_ok = decoded.raw_valid
            decode_error = None if decoded.raw_valid else "; ".join(decoded.errors)
            semantic_match = decoded.raw_valid and decoded.semantic_hash == str(row.get("semantic_hash") or "")
        except Exception as exc:  # quality report should retain malformed outputs
            decode_error = str(exc)

        predictions.append(
            {
                **build_prediction_record(row, raw_generated, runtime=runtime),
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
                "teacher_forced_loss": teacher_forced_loss,
                "teacher_forced_tokens": teacher_forced_tokens,
            }
        )
        if args.print_output:
            print(f"--- generated row {index} ---\n{generated}\n--- end row {index} ---", flush=True)
        print(
            f"row={index}/{len(rows)} decode_ok={decode_ok} "
            f"semantic_match={semantic_match} "
            f"teacher_forced_loss={teacher_forced_loss:.6f} "
            f"output_tokens={predictions[-1]['output_tokens']}",
            flush=True,
        )

    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    scores = evaluate_predictions(predictions_path, output_dir=output_dir, metric_version="v5_4")

    summary = {
        "model_dir": str(Path(args.model_dir)),
        "generation_reward_v5_4": scores.get("generation_reward_v5_4"),
        "effective_eos_token_ids": eos_ids,
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
        "teacher_forced_loss": (
            sum(
                item["teacher_forced_loss"] * item["teacher_forced_tokens"]
                for item in predictions
            )
            / sum(item["teacher_forced_tokens"] for item in predictions)
        ),
        "teacher_forced_tokens": sum(
            item["teacher_forced_tokens"] for item in predictions
        ),
        "predictions": str(predictions_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
