"""Verify the matched CPU/GPU experiment and save raw checkpoint comparisons."""
import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-predictions", type=Path, required=True)
    args = parser.parse_args()
    pred_bytes = args.checkpoint_predictions.read_bytes()
    predictions = {r["id"]: r for r in map(json.loads, pred_bytes.decode("utf-8").splitlines())}
    configs = {b: read(ROOT / b / "run_config.json") for b in ("cpu", "gpu")}
    results = {b: {r["id"]: r for r in read(ROOT / b / "results.json")} for b in configs}
    assert configs["cpu"]["cases"] == configs["gpu"]["cases"] == ["BXP-001", "BXP-003"]
    for key in ("corpus", "prompt", "model", "device"):
        assert configs["cpu"][key] == configs["gpu"][key], key
    for backend, config in configs.items():
        assert read(ROOT / backend / "status.json")["state"] == "complete"
        assert config["runtime"]["accelerator"] == backend.upper()
        assert config["runtime"]["mtpEnabled"] is False
        assert config["runtime"]["temperature"] == 0
        assert config["runtime"]["sourceFallbackEnabled"] is False
    assert {k: v for k, v in configs["cpu"]["runtime"].items() if k != "accelerator"} == {
        k: v for k, v in configs["gpu"]["runtime"].items() if k != "accelerator"
    }
    rows = []
    checkpoint = ROOT / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    sections = []
    for case in configs["cpu"]["cases"]:
        prediction = predictions[case]
        cpu, gpu = results["cpu"][case], results["gpu"][case]
        assert cpu["renderedPromptSha256"] == gpu["renderedPromptSha256"] == prediction["runtime"]["prompt_sha256"]
        assert cpu["metrics"]["inputTokens"] == gpu["metrics"]["inputTokens"] == prediction["runtime"]["input_tokens"]
        source = read(ROOT / "cpu" / case / "source.json")
        assert source == read(ROOT / "gpu" / case / "source.json")
        assert source["text"] == prediction["response_text"]
        raw_checkpoint = prediction["raw_generated_text"]
        (checkpoint / f"{case}.express").write_text(raw_checkpoint, encoding="utf-8")
        row = {"id": case, "prompt_sha256": cpu["renderedPromptSha256"], "input_tokens": cpu["metrics"]["inputTokens"],
               "checkpoint": {"raw_strict_valid": prediction["metrics"]["schema_valid_strict"], "output_tokens": prediction["runtime"]["output_tokens"], "raw_sha256": hashlib.sha256(raw_checkpoint.encode()).hexdigest(), "generation_reward_v5_4": prediction["metrics"]["generation_reward_v5_4"]}}
        panels = [f'<article><h3>Supplied CUDA checkpoint</h3><p>{row["checkpoint"]["output_tokens"]} tokens; raw strict valid: {row["checkpoint"]["raw_strict_valid"]}</p><pre>{html.escape(raw_checkpoint)}</pre></article>']
        for backend, result in (("cpu", cpu), ("gpu", gpu)):
            raw_path = ROOT / backend / case / "output.express"
            raw = raw_path.read_text(encoding="utf-8")
            scored = next(r for r in map(json.loads, (ROOT / backend / "scored_predictions.jsonl").read_text(encoding="utf-8").splitlines()) if r["id"] == case)
            row[backend] = {"raw_strict_valid": result["rawStrictValid"], "python_raw_strict_valid": scored["metrics"]["schema_valid_strict"],
                "repair": result["repairKind"], "runtime": result["runtime"], "seconds": result["elapsedMs"] / 1000,
                "output_tokens": result["metrics"]["outputTokens"], "decode_tokens_per_second": result["metrics"]["decodeTokensPerSecond"],
                "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "generation_reward_v5_4": scored["metrics"]["generation_reward_v5_4"]}
            assert result["rawStrictValid"] == row[backend]["python_raw_strict_valid"]
            panels.append(f'<article><h3>LiteRT {backend.upper()} · MTP off</h3><p>{row[backend]["seconds"]:.2f}s; {row[backend]["output_tokens"]} tokens; raw valid: {result["rawStrictValid"]}; repair: {html.escape(result["repairKind"])}</p><pre>{html.escape(raw)}</pre><details><summary>Rendered screenshot</summary><img src="{backend}/{case}/screen.png" alt="{backend} {case} screenshot"></details></article>')
        previous_gpu = ROOT.parent / "20260928_r64_qat_reexport/device/flip8_r64_qat_mtp_off_20260928_r1" / case / "output.express"
        row["gpu_exactly_repeats_previous_mtp_off_output"] = previous_gpu.read_bytes() == (ROOT / "gpu" / case / "output.express").read_bytes()
        rows.append(row)
        sections.append(f'<section><h2>{case}</h2><div class="grid">{"".join(panels)}</div></section>')
    summary = {"model_sha256": read(ROOT / "device_identity.json")["model"]["sha256"],
               "checkpoint_prediction_file_sha256": hashlib.sha256(pred_bytes).hexdigest(),
               "matched_source_rendered_prompt_input_count_runtime_and_device": True,
               "native_token_ids_compared": False, "cases": rows}
    (ROOT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rank-64 accuracy gap: checkpoint, CPU, GPU</title><style>body{font:16px system-ui;margin:24px;background:#fff;color:#182230}h1{font-size:28px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}article{border:1px solid #cdd6e1;border-radius:12px;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace;max-height:650px;overflow:auto;background:#f4f6f9;padding:12px}img{width:100%;height:auto}section{margin:32px 0}summary{cursor:pointer;padding:12px}@media(max-width:1000px){.grid{grid-template-columns:1fr}}</style><h1>Same quantized model: CPU vs GPU</h1><p>Fresh two-case experiment, MTP off. CPU raw validity: 2/2; GPU raw validity: 0/2. Model SHA-256: de60d19c…e63e62. Text below is before repair. Screenshots show the subsequent rendering; CPU required no repair, GPU did.</p><p>Supplied CUDA predictions are reference artifacts, not freshly rerun here. CPU still has citation/graph issues. This test isolates a backend-dependent regression; it does not identify the exact GPU kernel.</p>' + ''.join(sections) + '</html>'
    (ROOT / "checkpoint_cpu_gpu.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
