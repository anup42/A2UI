"""Verify and present the completed native precision/sampler controls."""
from pathlib import Path
import hashlib
import html
import json

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "20260928_r64_accuracy_gap"
RUNS = {
    "gpu_default_r2": "GPU FP16 + GPU sampler",
    "gpu_cpu_sampler": "GPU FP16 + CPU sampler / FP32 logits",
    "gpu_fp32": "GPU FP32 + GPU sampler",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    rows = []
    sections = []
    for case in ("BXP-001", "BXP-003"):
        panels = []
        baseline_ids = (ROOT / "gpu_default_r2" / f"{case}.input_token_ids.txt").read_bytes()
        old_gpu = (PRIOR / "gpu" / case / "output.express").read_text(encoding="utf-8").strip()
        old_cpu = (PRIOR / "cpu" / case / "output.express").read_text(encoding="utf-8").strip()
        checkpoint = (PRIOR / "checkpoint" / f"{case}.express").read_text(encoding="utf-8").strip()
        for run, label in RUNS.items():
            folder = ROOT / run
            metrics = read(folder / f"{case}.metrics.json")
            scored = read(folder / f"{case}.scored.json")
            manifest = read(folder / "run_manifest.json")
            raw = (folder / f"{case}.raw.txt").read_text(encoding="utf-8")
            assert metrics["status"] == "ok"
            assert manifest["model_sha256"] == read(ROOT / "runtime_manifest.json")["model_sha256"]
            assert metrics["backend_requested"] == "gpu" and metrics["speculative_decoding"] is False
            assert (folder / f"{case}.input_token_ids.txt").read_bytes() == baseline_ids
            assert metrics["prefill_token_counts"] == [3174 if case == "BXP-001" else 3394]
            assert metrics["force_f32_activations"] == (run == "gpu_fp32")
            if run == "gpu_cpu_sampler":
                log = (folder / f"{case}.native.log").read_text(encoding="utf-8")
                assert "sampler_after=CPU(3) ABI_guards=passed" in log
                assert "session_sampler=CPU(3) ABI_guards=passed" in log
                assert metrics["sampler_backend_requested"] == "cpu"
            row = {"case": case, "run": run, "label": label,
                   "raw_valid": scored["metrics"]["schema_valid_strict"],
                   "reward_v5_4": scored["metrics"]["generation_reward_v5_4"],
                   "output_tokens": sum(metrics["decode_token_counts"]),
                   "prefill_ms": metrics["prefill_wall_ms"], "decode_ms": metrics["decode_wall_ms"],
                   "generation_ms": metrics["generation_wall_ms"], "engine_create_ms": metrics["engine_create_ms"],
                   "decode_tokens_per_second": metrics["decode_tokens_per_s"][0],
                   "matches_prior_jni_gpu_trimmed": raw.strip() == old_gpu,
                   "matches_prior_jni_cpu_trimmed": raw.strip() == old_cpu,
                   "matches_checkpoint_trimmed": raw.strip() == checkpoint,
                   "input_tokenizer_ids_sha256": hashlib.sha256(baseline_ids).hexdigest(),
                   "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                   "probe_sha256": manifest["probe_sha256"]}
            rows.append(row)
            panels.append(f'<article><h3>{html.escape(label)}</h3><p class="{"pass" if row["raw_valid"] else "fail"}">Raw valid: {row["raw_valid"]} · score {row["reward_v5_4"]:.2f}</p><p>{row["output_tokens"]} output tokens · {row["decode_tokens_per_second"]:.2f} decode tokens/s</p><pre>{html.escape(raw)}</pre></article>')
        sections.append(f'<section><h2>{case}</h2><div class="grid">{"".join(panels)}</div><details><summary>Supplied checkpoint raw reference</summary><pre>{html.escape(checkpoint)}</pre></details></section>')
    for row in rows:
        assert row["raw_valid"] == (row["run"] == "gpu_fp32")
        if row["run"] != "gpu_fp32":
            assert row["matches_prior_jni_gpu_trimmed"]
    result = {"case_count": 2, "native_run_count": 6, "native_input_tokenizer_ids_match": True,
              "no_repair_or_fallback": True, "results": rows}
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU precision controls</title><style>body{font:16px system-ui;background:white;color:#16202e;margin:28px}h1{font-size:30px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}article{border:1px solid #d0d7df;border-radius:12px;padding:16px}pre{font:13px/1.55 monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:14px;max-height:650px;overflow:auto}.pass{color:#136d3a;font-weight:700}.fail{color:#a12626;font-weight:700}section{margin:32px 0}summary{cursor:pointer;padding:14px}@media(max-width:1000px){.grid{grid-template-columns:1fr}}</style><h1>GPU FP16 is the failing execution setting</h1><p>Same rank-64 W4 model, same tokenized input, GPU execution, MTP off. Switching sampling to CPU does not change the failures; switching GPU computation to FP32 restores valid output for both tested cases.</p><p><b>All text below is actual raw model output, before repair.</b> This is a two-case diagnosis, not a full Bixby50 result. Timing is from individual runs, without thermal control.</p>' + ''.join(sections) + '</html>'
    (ROOT / "raw_comparison.html").write_text(page, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
