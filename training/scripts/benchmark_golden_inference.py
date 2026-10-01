"""Measure exact cached/uncached QAT decoding on the same loaded checkpoint.

This diagnostic uses the real prepared prompt, quantizers, EOS IDs and serving
stop policy. It records isolated short-probe parity and optional full completion
latency separately; it never shortens an evaluation cohort for quality scoring.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "dataset" / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-job", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-tokens", type=int, default=64)
    parser.add_argument("--full-case", action="store_true")
    parser.add_argument("--reference-predictions", type=Path, help="Validate full-case tokens against an existing baseline.")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile-activations", action="store_true", help="Measure precise fused SRQ; token parity is mandatory.")
    args = parser.parse_args()
    if args.probe_tokens < 1:
        parser.error("--probe-tokens must be positive")
    import torch
    from peft import PeftModel
    from ir_training.models.registry import create_adapter
    from ir_training.eval.generate import build_prediction_record, place_model_for_generation
    from ir_training.generation_policy import build_stopping_criteria, generation_cache_scope, generation_diagnostics, preserve_generation_eos
    from ir_training.qat.fake_quant import prepare_qat_model
    from ir_training.qat.full_model_contract import full_parameter_autocast
    from ir_training.train.cuda_runtime import sdpa_policy

    args.output_dir.mkdir(parents=True, exist_ok=True)
    job = json.loads(args.worker_job.read_text(encoding="utf-8"))
    config = copy.deepcopy(job["config"])
    config["model"].update(device_map={"": 0}, inference_device="auto")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    started = time.perf_counter()
    adapter = create_adapter(config["model"])
    tokenizer = adapter.load_tokenizer()
    tokenizer_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model = adapter.load_model()
    base_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if job.get("adapter_checkpoint"):
        model = PeftModel.from_pretrained(model, job["adapter_checkpoint"], is_trainable=False)
    place_model_for_generation(model, config["model"])
    adapter_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    controller = prepare_qat_model(model, config)
    model.eval()
    torch.cuda.synchronize()
    qat_setup_seconds = time.perf_counter() - started
    row = json.loads(next(line for line in Path(job["split_path"]).read_text(encoding="utf-8").splitlines() if line.strip()))
    prompt = adapter.format_example(row, tokenizer=tokenizer, include_assistant=False)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    width = inputs["input_ids"].shape[-1]
    eos = preserve_generation_eos(model, tokenizer)

    class Timer:
        def __init__(self):
            self.times = []
            self.prompt = True
        def put(self, value):
            if self.prompt:
                self.prompt = False
                return
            self.times.append(time.perf_counter())
        def end(self):
            pass

    results = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "checkpoint": job.get("adapter_checkpoint"),
        "case_id": row.get("id"),
        "input_tokens": width,
        "load_seconds": {"tokenizer": tokenizer_seconds, "base_model": base_load_seconds,
                         "adapter": adapter_load_seconds, "qat_setup": qat_setup_seconds},
        "probes": [],
    }

    def save():
        (args.output_dir / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    def run(label, count, cached, stopped=False):
        timer = Timer()
        kwargs = dict(max_new_tokens=count, do_sample=False, use_cache=True,
                      eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                      stop_strings=None, streamer=timer)
        if stopped:
            kwargs["stopping_criteria"] = build_stopping_criteria(tokenizer, width)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.inference_mode(), full_parameter_autocast(model):
            output = model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        ids = output[0, width:].tolist()
        measured = dict(label=label, cache_enabled=cached, generated_tokens=len(ids),
                        generation_seconds=elapsed, tokens_per_second=len(ids) / elapsed,
                        first_token_seconds=timer.times[0] - begin if timer.times else None,
                        decode_tokens_per_second=(len(timer.times) - 1) / (timer.times[-1] - timer.times[0]) if len(timer.times) > 1 else None,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        token_ids=ids)
        if cached:
            measured["cache_stats"] = dict(controller.inference_cache_stats)
        results["probes"].append(measured)
        save()
        print(json.dumps({k: v for k, v in measured.items() if k != "token_ids"}), flush=True)
        if stopped:
            runtime = generation_diagnostics(tokenizer, ids, eos_token_ids=eos, max_new_tokens=count,
                prompt_text=prompt, input_ids=inputs["input_ids"][0])
            runtime.update(generation_seconds=elapsed, output_tokens_per_second=len(ids)/elapsed,
                           inference_device=str(model.device), use_cache=True, qat_inference_weight_cache=cached)
            record = build_prediction_record(row, tokenizer.decode(ids, skip_special_tokens=True), runtime=runtime)
            (args.output_dir / (label + ".jsonl")).write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            if args.reference_predictions:
                references = [json.loads(line) for line in args.reference_predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
                matches = [item for item in references if item.get("id") == row.get("id")]
                if len(matches) != 1:
                    raise ValueError("Full-case reference must contain exactly one matching case ID.")
                reference = matches[0]
                parity = {
                    "input_tokens_match": reference["runtime"]["input_token_ids_sha256"] == runtime["input_token_ids_sha256"],
                    "output_tokens_match": reference["runtime"]["generated_token_ids_sha256"] == runtime["generated_token_ids_sha256"],
                    "output_text_match": reference["raw_generated_text"] == record["raw_generated_text"],
                }
                results["full_case_parity"] = parity
                save()
                if not all(parity.values()):
                    raise RuntimeError("Full-case baseline parity failed; do not deploy this optimization.")
        return ids

    try:
        with sdpa_policy(config), generation_cache_scope(model):
            with controller.inference_cache(enabled=False):
                run("uncached_warmup", 8, False)
                baseline = run("uncached_probe", args.probe_tokens, False)
                if args.profile:
                    from torch.profiler import profile, ProfilerActivity
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        run("uncached_profile", 8, False)
                    (args.output_dir / "uncached_profile.txt").write_text(prof.key_averages().table(sort_by="self_device_time_total", row_limit=35))
            with controller.inference_cache(enabled=True, compile_srq=False):
                run("cached_first_use", 8, True)
                candidate = run("cached_probe", args.probe_tokens, True)
                results["probe_token_exact_match"] = baseline == candidate
                save()
                if baseline != candidate:
                    raise RuntimeError("Cached and uncached generated token IDs differ; do not deploy.")
                if args.profile:
                    from torch.profiler import profile, ProfilerActivity
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        run("cached_profile", 8, True)
                    (args.output_dir / "cached_profile_cpu.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
                    (args.output_dir / "cached_profile_device.txt").write_text(prof.key_averages().table(sort_by="self_device_time_total", row_limit=40))
                if args.full_case and not args.compile_activations:
                    run("cached_full_case", job.get("max_new_tokens", 2048), True, stopped=True)
            if args.compile_activations:
                with controller.inference_cache(enabled=True, compile_srq=True):
                    run("compiled_first_use", 8, True)
                    compiled_ids = run("compiled_probe", args.probe_tokens, True)
                    results["compiled_probe_token_exact_match"] = baseline == compiled_ids
                    save()
                    if baseline != compiled_ids:
                        raise RuntimeError("Compiled activation token IDs differ; do not deploy.")
                    if args.profile:
                        from torch.profiler import profile, ProfilerActivity
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                            run("compiled_profile", 8, True)
                        (args.output_dir / "compiled_profile_cpu.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
                        (args.output_dir / "compiled_profile_device.txt").write_text(prof.key_averages().table(sort_by="self_device_time_total", row_limit=40))
                    if args.full_case:
                        run("compiled_full_case", job.get("max_new_tokens", 2048), True, stopped=True)
    finally:
        controller.restore()
        save()


if __name__ == "__main__":
    main()
