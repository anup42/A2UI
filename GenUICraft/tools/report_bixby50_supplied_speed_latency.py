"""Calculate explicitly modelled latency using user-supplied decode rates."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "GenUICraft/validation/20260929_bixby50_full_gpu_fp32"
OUT = RUN / "supplied_speed_latency_estimate"
SPEEDS = {"litert_mtp_off": 35.0, "litert_mtp_on": 52.0}
SCORES = {"checkpoint_fp32_reference": 85.90, "litert_mtp_off": 83.26, "litert_mtp_on": 80.89}


def mean(rows, key):
    return statistics.mean(row[key] for row in rows)


def aggregate(rows):
    cold = [row for row in rows if row["engine_initialized_for_request"]]
    warm = mean(rows, "estimated_warm_conversion_s")
    initialization = mean(cold, "measured_engine_initialization_s") if cold else None
    return {
        "n": len(rows),
        "input_tokens_total": sum(row["input_tokens"] for row in rows),
        "input_tokens_mean": mean(rows, "input_tokens"),
        "output_tokens_total": sum(row["output_tokens"] for row in rows),
        "output_tokens_mean": mean(rows, "output_tokens"),
        "decode_speed_supplied_tps": rows[0]["decode_speed_supplied_tps"],
        "estimated_decode_s": mean(rows, "estimated_decode_s"),
        "measured_prefill_s": mean(rows, "measured_prefill_s"),
        "measured_provider_residual_s": mean(rows, "measured_provider_residual_s"),
        "measured_converter_and_recording_s": mean(rows, "measured_converter_and_recording_s"),
        "measured_other_warm_latency_s": mean(rows, "measured_other_warm_latency_s"),
        "estimated_warm_conversion_s": warm,
        "cold_start_samples": len(cold),
        "measured_cold_initialization_mean_s": initialization,
        "estimated_cold_conversion_s": warm + initialization if initialization is not None else None,
        "estimated_conversion_with_observed_batch_restart_frequency_s": mean(rows, "estimated_same_restart_frequency_s"),
        "recorded_provider_call_mean_s": mean(rows, "recorded_provider_call_s"),
        "recorded_converter_elapsed_mean_s": mean(rows, "recorded_converter_elapsed_s"),
    }


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def main():
    source = RUN / "model_inputs.jsonl"
    inputs = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    assert len(inputs) == 100
    rows = []
    for item in inputs:
        runtime, result = item["runtime"], item["provider_result"]
        initialized = runtime["engineInitializedForRequest"]
        initialization = runtime.get("engineInitializationSeconds") or 0.0
        assert initialized or initialization == 0
        input_tokens, output_tokens = runtime["inputTokens"], runtime["actualOutputTokens"]
        prefill = input_tokens / runtime["prefillTokensPerSecond"]
        recorded_decode = output_tokens / runtime["decodeTokensPerSecond"]
        call = runtime["providerCallMs"] / 1000
        converter = result["elapsedMs"] / 1000
        provider_residual = call - prefill - recorded_decode - initialization
        converter_and_recording = converter - call
        other = prefill + provider_residual + converter_and_recording
        estimated_decode = output_tokens / SPEEDS[item["model"]]
        row = {
            "model": item["model"], "id": item["id"], "device_model": item["device_model"],
            "generation_origin": item["generation_origin"], "finish_reason": runtime["finishReason"],
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "decode_speed_supplied_tps": SPEEDS[item["model"]],
            "estimated_decode_s": estimated_decode, "measured_prefill_s": prefill,
            "measured_provider_residual_s": provider_residual,
            "measured_converter_and_recording_s": converter_and_recording,
            "measured_other_warm_latency_s": other,
            "engine_initialized_for_request": initialized,
            "measured_engine_initialization_s": initialization,
            "estimated_warm_conversion_s": estimated_decode + other,
            "estimated_same_restart_frequency_s": estimated_decode + other + initialization,
            "recorded_provider_call_s": call, "recorded_converter_elapsed_s": converter,
        }
        assert abs(prefill + recorded_decode + initialization + provider_residual + converter_and_recording - converter) < 1e-9
        rows.append(row)
    populations = {}
    for model in SPEEDS:
        subset = [row for row in rows if row["model"] == model]
        assert len(subset) == 50 and len({row["id"] for row in subset}) == 50
        populations[model] = {
            "all_50": aggregate(subset),
            "excluding_repetition_stops": aggregate([row for row in subset if row["finish_reason"] != "REPETITION_LIMIT"]),
        }
    data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "supplied_comparison_scores": SCORES,
        "score_provenance": "85.90 is the user-supplied original FP32 checkpoint reference. Native 83.26/80.89 match the rounded prior full50 repaired scores.",
        "supplied_decode_speeds_tps": SPEEDS,
        "latency_status": "ESTIMATED using supplied decode rates; not newly measured latency at those rates",
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pooled_output_tokens_mean_over_100_attempts": mean(rows, "output_tokens"),
        "populations": populations,
        "exclusions": ["Perplexity answer generation/network", "UI rendering and screenshot capture", "test harness waits"],
        "assumptions": [
            "Prefill and non-decode overhead remain as recorded in the mixed-device September28/29 runs.",
            "Decode time is native generated output tokens divided by the user-supplied 35 or 52 tokens/second.",
            "Warm estimates remove engine loading; cold estimates add mean measured initialization among cold calls.",
            "TTFT overlaps prefill and first-token decode; native initialization phase totals overlap wall initialization. Neither is added again.",
            "Converter residual includes prompt construction, compile/repair, attribution and benchmark recording I/O; it is not pure repair duration.",
            "All 50-case means include guard-stopped attempts and conversion failures; they do not imply every attempt produces a rendered answer.",
        ],
    }
    off, on = (populations[model]["all_50"] for model in SPEEDS)
    completed_off, completed_on = (populations[model]["excluding_repetition_stops"] for model in SPEEDS)
    comparison = table(["Model / mode", "Comparison score /100", "Decode speed tokens/s", "Average generated tokens"], [
        ["Original FP32 checkpoint (supplied reference)", "85.90", "Not supplied", "Not estimated here"],
        ["LiteRT MTP off, after SDK repair", "83.26", "35", f"{off['output_tokens_mean']:.2f}"],
        ["LiteRT MTP on, after SDK repair", "80.89", "52", f"{on['output_tokens_mean']:.2f}"],
    ])
    components = [
        ("Generated tokens / supplied decode speed", "estimated_decode_s"),
        ("Prefill (input tokens / recorded native prefill rate)", "measured_prefill_s"),
        ("Other provider/dispatch and output recording", "measured_provider_residual_s"),
        ("Converter + compile/repair + benchmark recording", "measured_converter_and_recording_s"),
        ("Estimated warm conversion total", "estimated_warm_conversion_s"),
        ("Additional cold engine initialization", "measured_cold_initialization_mean_s"),
        ("Estimated cold conversion total", "estimated_cold_conversion_s"),
    ]
    body = "\n".join([
        "# Bixby50: supplied comparison scores and decode-speed latency estimates", "",
        comparison, "",
        "The FP32 checkpoint score 85.90 and decode speeds 35/52 tokens/s are supplied comparison values. The native repaired scores 83.26/80.89 match the completed full 50-case report. The checkpoint differences are 2.64 points without MTP and 5.01 points with MTP.", "",
        "## Average tokens across all 50 cases per mode", "",
        f"The average model input is **{off['input_tokens_mean']:.2f} tokens** in either mode, including the model prompt. Total generated tokens are **{off['output_tokens_total']:,} off** and **{on['output_tokens_total']:,} on**. Dividing each by 50 gives **{off['output_tokens_mean']:.2f} off** and **{on['output_tokens_mean']:.2f} on**. Pooling all 100 attempts gives **{data['pooled_output_tokens_mean_over_100_attempts']:.2f} generated tokens** per attempt. Repair adds no model-generated tokens.", "",
        "## Estimated latency in seconds", "",
        table(["Component", "MTP off", "MTP on"], [[label, f"{off[key]:.2f}", f"{on[key]:.2f}"] for label, key in components]), "",
        "Formula: `warm conversion = output tokens / supplied decode tokens per second + measured prefill + measured remaining conversion overhead`. Cold conversion adds the measured cold-start initialization mean. The endpoint is conversion completion or failure; Android render/display time is outside this estimate.", "",
        f"At the benchmark's actual batch restart frequency ({off['cold_start_samples']} cold calls off and {on['cold_start_samples']} on, out of 50 each), the hybrid mean estimates would be **{off['estimated_conversion_with_observed_batch_restart_frequency_s']:.2f}s off** and **{on['estimated_conversion_with_observed_batch_restart_frequency_s']:.2f}s on**. That restart frequency is a test-harness property, not a recommended app behavior.", "",
        "## Effect of repetition-stopped outputs", "",
        "The full 50 includes 6 guard-stopped outputs without MTP and 7 with MTP. Some stopped outputs were repaired successfully, while 9 attempts across both modes had no recoverable UI content. Short stopped generations lower the token and latency averages. Excluding REPETITION_LIMIT attempts gives:", "",
        table(["Population", "Count", "Average output tokens", "Decode estimate seconds", "Warm conversion estimate seconds"], [
            ["MTP off, excluding repetition stops", completed_off["n"], f"{completed_off['output_tokens_mean']:.2f}", f"{completed_off['estimated_decode_s']:.2f}", f"{completed_off['estimated_warm_conversion_s']:.2f}"],
            ["MTP on, excluding repetition stops", completed_on["n"], f"{completed_on['output_tokens_mean']:.2f}", f"{completed_on['estimated_decode_s']:.2f}", f"{completed_on['estimated_warm_conversion_s']:.2f}"],
        ]), "",
        "The secondary population contains runs labelled COMPLETED by the app; it does not guarantee natural EOS, perfect content, or visual success.", "",
        "## Measurement limits and timing definitions", "",
        "These are estimates, not new measurements at 35/52 tokens/s. Prefill, initialization and residual overhead come from the existing two-device run with mixed temperatures and generation dates. Holding those costs fixed while replacing decode speed is an explicit assumption.", "",
        "Do not add time-to-first-token to prefill: it already includes prefill and first-token work. Do not also add the native initialization phase total: its phase sums can overlap the wall initialization timer. The converter overhead includes timing/benchmark file writes, so it is not an isolated repair measurement. `caseElapsedMs` includes renderer setup, screenshot work and artificial test waits and is intentionally excluded.", "",
        "No decode speed was supplied for the original FP32 checkpoint, so its latency is not inferred. V5.4 is a UI-representation score, not factual-answer accuracy.", "",
        "[Summary JSON](summary.json) · [Per-case latency calculations](per_case_latency.csv) · [Full 50-case raw/repaired report](../full_native_repaired_report/REPORT.md)", "",
        f"Input manifest SHA-256: `{data['source_sha256']}`.", "",
    ])
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "summary.json").exists(), "Preserve existing estimates"
    (OUT / "summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (OUT / "REPORT.md").write_text(body, encoding="utf-8")
    with (OUT / "per_case_latency.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"report": str(OUT / "REPORT.md"), "averages": populations}, indent=2))


if __name__ == "__main__":
    main()
