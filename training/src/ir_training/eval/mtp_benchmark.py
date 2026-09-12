from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from ir_training.common.config import resolve_path, training_root
from ir_training.common.jsonl import write_jsonl
from ir_training.eval.generate import build_prediction_record
from ir_training.eval.compare_to_baseline import evaluate_predictions
from ir_training.eval.golden_set import load_fixed_golden_rows
from ir_training.models.hf_loading import load_hf_model
from ir_training.qat_mtp.workflow import summarize_issues, validate_benchmark_config


def build_mtp_benchmark_plan(config: dict[str, Any]) -> dict[str, Any]:
    base = training_root()
    run_cfg = _section(config, "run")
    source_cfg = _section(config, "source")
    data_cfg = _section(config, "data")
    benchmark_cfg = _section(config, "benchmark")
    issues = validate_benchmark_config(config)
    return {
        "run_id": run_cfg.get("id"),
        "output_dir": str(resolve_path(run_cfg.get("output_dir", "outputs/eval/gemma4_e2b_qat_mtp_reference"), base)),
        "split_path": str(resolve_path(data_cfg.get("split_path") or "outputs/datasets/golden35_stage3_eval/all.jsonl", base)),
        "golden_contract": _golden_row_contract(data_cfg),
        "max_new_tokens": int(benchmark_cfg.get("max_new_tokens", 2048)),
        "target_model": _resolve_model_source(source_cfg.get("target_model_id"), base, local_hint=True),
        "target_base_model": source_cfg.get("target_base_model_id"),
        "assistant_model": source_cfg.get("assistant_model_id"),
        "processor_model": source_cfg.get("processor_model_id"),
        "mode": benchmark_cfg.get("mode"),
        "packed_int4": bool(benchmark_cfg.get("packed_int4", False)),
        "executes_training": False,
        "downloads_or_loads_models_when_executed": True,
        "acceptance_rate_available": False,
        "acceptance_rate_note": (
            "Transformers generate() does not expose accepted-draft counts through this stable wrapper; "
            "use final runtime instrumentation for acceptance-rate gating."
        ),
        "final_runtime_validation_required": bool(benchmark_cfg.get("final_runtime_validation_required", False)),
        "validation": summarize_issues(issues),
    }


def run_mtp_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Compare greedy target-only and target+assistant decoding.

    This is deliberately a floating-point Transformers reference check. It
    detects target drift and measures gross speculative-decoding benefit, but
    it is not evidence for final Q4_0/LiteRT-LM device performance.
    """

    validation = summarize_issues(validate_benchmark_config(config))
    if not validation["ok"]:
        raise ValueError(f"Invalid QAT/MTP benchmark config: {json.dumps(validation, ensure_ascii=False)}")

    base = training_root()
    run_cfg = _section(config, "run")
    source_cfg = _section(config, "source")
    data_cfg = _section(config, "data")
    benchmark_cfg = _section(config, "benchmark")
    evaluation_cfg = _section(config, "evaluation")

    output_dir = resolve_path(run_cfg.get("output_dir", "outputs/eval/gemma4_e2b_qat_mtp_reference"), base)
    split_path = resolve_path(data_cfg.get("split_path") or "outputs/datasets/golden35_stage3_eval/all.jsonl", base)
    # Validate the configured cohort before importing the runtime or loading models.
    # Custom diagnostic sets remain supported when no fixed-size contract is set.
    rows = load_fixed_golden_rows(split_path, **_golden_row_contract(data_cfg))
    if not rows:
        raise ValueError(f"Benchmark split contains no readable JSON objects: {split_path}")
    target_source = _resolve_model_source(source_cfg.get("target_model_id"), base, local_hint=True)
    if _looks_local(str(source_cfg.get("target_model_id") or "")) and not Path(target_source).exists():
        raise FileNotFoundError(
            f"Missing merged target model: {target_source}. Run merge_qat_lora.py --execute after training completes."
        )
    assistant_source = str(source_cfg.get("assistant_model_id") or "")
    processor_source = str(source_cfg.get("processor_model_id") or source_cfg.get("target_base_model_id") or "")

    try:
        import torch  # type: ignore
        from transformers import AutoProcessor  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install training/requirements-gemma4-qat.txt before executing the benchmark.") from exc

    model_config = {
        "model_loader": source_cfg.get("model_loader", "auto_causal_lm"),
        "dtype": source_cfg.get("dtype", "bfloat16"),
        "device_map": source_cfg.get("device_map", "auto"),
        "trust_remote_code": bool(source_cfg.get("trust_remote_code", False)),
        "load_in_4bit": False,
    }
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=bool(source_cfg.get("trust_remote_code", False)),
    )
    target_model = load_hf_model(target_source, model_config)
    assistant_model = load_hf_model(assistant_source, model_config)
    target_model.eval()
    assistant_model.eval()
    assistant_model.generation_config.num_assistant_tokens = int(benchmark_cfg.get("num_assistant_tokens", 4))
    assistant_model.generation_config.num_assistant_tokens_schedule = str(
        benchmark_cfg.get("num_assistant_tokens_schedule", "heuristic")
    )

    max_input_tokens = int(benchmark_cfg.get("max_input_tokens", 4096))
    max_new_tokens = int(benchmark_cfg.get("max_new_tokens", 2048))
    repeats = max(1, int(benchmark_cfg.get("repeats", 1)))
    do_sample = bool(benchmark_cfg.get("do_sample", False))
    if do_sample:
        raise ValueError("The reference comparison requires greedy decoding (benchmark.do_sample=false).")

    warmup_rows = min(len(rows), max(0, int(benchmark_cfg.get("warmup_rows", 1))))
    for row in rows[:warmup_rows]:
        inputs = _prepare_inputs(processor, row, target_model, max_input_tokens)
        _generate_once(
            torch=torch,
            target_model=target_model,
            assistant_model=None,
            inputs=inputs,
            max_new_tokens=min(max_new_tokens, 32),
        )
        _generate_once(
            torch=torch,
            target_model=target_model,
            assistant_model=assistant_model,
            inputs=inputs,
            max_new_tokens=min(max_new_tokens, 32),
        )

    target_predictions: list[dict[str, Any]] = []
    mtp_predictions: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for row in rows:
        inputs = _prepare_inputs(processor, row, target_model, max_input_tokens)
        input_length = int(inputs["input_ids"].shape[-1])
        target_outputs = [
            _generate_once(
                torch=torch,
                target_model=target_model,
                assistant_model=None,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
            )
            for _ in range(repeats)
        ]
        mtp_outputs = [
            _generate_once(
                torch=torch,
                target_model=target_model,
                assistant_model=assistant_model,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
            )
            for _ in range(repeats)
        ]
        target_tokens = target_outputs[0]["tokens"][0][input_length:]
        mtp_tokens = mtp_outputs[0]["tokens"][0][input_length:]
        target_text = processor.decode(target_tokens, skip_special_tokens=True)
        mtp_text = processor.decode(mtp_tokens, skip_special_tokens=True)
        target_latency = statistics.median(float(item["elapsed_seconds"]) for item in target_outputs)
        mtp_latency = statistics.median(float(item["elapsed_seconds"]) for item in mtp_outputs)
        target_token_count = int(target_tokens.shape[-1])
        mtp_token_count = int(mtp_tokens.shape[-1])
        # Repeats measure latency only: score one prediction per source and mode.
        target_predictions.append(_prediction_common(row, target_text))
        mtp_predictions.append(_prediction_common(row, mtp_text))
        timing_rows.append(
            {
                "id": row.get("id"),
                "target_latency_seconds": target_latency,
                "mtp_latency_seconds": mtp_latency,
                "target_output_tokens": target_token_count,
                "mtp_output_tokens": mtp_token_count,
                "target_tokens_per_second": target_token_count / target_latency if target_latency > 0 else None,
                "mtp_tokens_per_second": mtp_token_count / mtp_latency if mtp_latency > 0 else None,
                "latency_speedup": target_latency / mtp_latency if mtp_latency > 0 else None,
                "greedy_token_ids_equal": bool(torch.equal(target_tokens, mtp_tokens)),
                "decoded_text_equal": target_text == mtp_text,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    target_predictions_path = output_dir / "target_only_predictions.jsonl"
    mtp_predictions_path = output_dir / "target_plus_mtp_predictions.jsonl"
    timings_path = output_dir / "timings.jsonl"
    write_jsonl(target_predictions_path, target_predictions)
    write_jsonl(mtp_predictions_path, mtp_predictions)
    write_jsonl(timings_path, timing_rows)

    target_metrics = None
    mtp_metrics = None
    if bool(evaluation_cfg.get("enabled", True)):
        weights_value = evaluation_cfg.get("weights_config")
        weights_path = resolve_path(weights_value, base) if weights_value else None
        target_metrics = evaluate_predictions(
            target_predictions_path,
            output_dir=output_dir / "target_only",
            weights_config_path=weights_path,
        )
        mtp_metrics = evaluate_predictions(
            mtp_predictions_path,
            output_dir=output_dir / "target_plus_mtp",
            weights_config_path=weights_path,
        )

    target_total = sum(float(row["target_latency_seconds"]) for row in timing_rows)
    mtp_total = sum(float(row["mtp_latency_seconds"]) for row in timing_rows)
    summary = {
        "run_id": run_cfg.get("id"),
        "sample_count": len(timing_rows),
        "target_model": target_source,
        "target_base_model": source_cfg.get("target_base_model_id"),
        "assistant_model": assistant_source,
        "benchmark_mode": "transformers_reference",
        "packed_int4": False,
        "continued_qat_performed": False,
        "assistant_trained_or_modified": False,
        "greedy_equivalence_rate": sum(bool(row["greedy_token_ids_equal"]) for row in timing_rows) / len(timing_rows),
        "target_total_seconds": target_total,
        "mtp_total_seconds": mtp_total,
        "aggregate_latency_speedup": target_total / mtp_total if mtp_total > 0 else None,
        "target_metrics": target_metrics,
        "mtp_metrics": mtp_metrics,
        "acceptance_rate": None,
        "acceptance_rate_note": (
            "Not exposed by this stable Transformers wrapper; collect accepted draft length/rejection counts "
            "from the final deployment runtime."
        ),
        "final_runtime_validation_required": True,
        "artifacts": {
            "target_predictions": str(target_predictions_path),
            "mtp_predictions": str(mtp_predictions_path),
            "timings": str(timings_path),
        },
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _generate_once(*, torch: Any, target_model: Any, assistant_model: Any | None, inputs: Any, max_new_tokens: int) -> dict[str, Any]:
    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    tokenizer_config = getattr(target_model, "generation_config", None)
    pad_token_id = getattr(tokenizer_config, "pad_token_id", None)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    if assistant_model is not None:
        generate_kwargs["assistant_model"] = assistant_model
    _synchronize(torch)
    start = time.perf_counter()
    with torch.no_grad():
        tokens = target_model.generate(**generate_kwargs)
    _synchronize(torch)
    return {"tokens": tokens.detach().cpu(), "elapsed_seconds": time.perf_counter() - start}


def _prepare_inputs(processor: Any, row: dict[str, Any], target_model: Any, max_input_tokens: int):
    messages = [dict(item) for item in (row.get("messages") or []) if isinstance(item, dict)]
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    return inputs.to(_model_input_device(target_model))


def _prediction_common(row: dict[str, Any], generated_text: str = "") -> dict[str, Any]:
    return build_prediction_record(row, generated_text)


def _model_input_device(model: Any):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = iter(model.parameters())
    return next(parameters).device


def _synchronize(torch: Any) -> None:
    if bool(torch.cuda.is_available()):
        torch.cuda.synchronize()


def _resolve_model_source(value: Any, base: Path, *, local_hint: bool) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    if Path(text).is_absolute() or (local_hint and _looks_local(text)):
        return str(resolve_path(text, base))
    return text


def _looks_local(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return normalized.startswith(("runs/", "outputs/", "checkpoints/", "./", "../"))


def _golden_row_contract(data_cfg: dict[str, Any]) -> dict[str, Any]:
    default_split = not data_cfg.get("split_path")
    required_rows = data_cfg.get("required_rows", 35 if default_split else None)
    return {
        "max_rows": int(data_cfg.get("max_rows", 35)),
        "required_rows": int(required_rows) if required_rows is not None else None,
        "require_exact_rows": bool(data_cfg.get("require_exact_rows", default_split)),
        "require_unique_rows": bool(data_cfg.get("require_unique_rows", default_split)),
    }


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}
