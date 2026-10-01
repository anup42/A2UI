"""Read-only metadata audit and synthetic QAT/merge rounding reproduction.

Run from any directory with Python, NumPy and PyTorch installed. This does not
load model weights into a framework, train, export, or operate an Android device.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "training/src"), str(REPO / "training/scripts")]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def function_ast(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def rounding_reproduction():
    import numpy as np
    import torch
    import build_gemma4_retained_scale_litertlm as exporter
    from ir_training.qat.fake_quant import _effective_lora_weight, fake_quantize_ste

    # Rank one is enough to demonstrate the addition rule, independent of rank.
    # Dtypes match the supplied export telemetry: BF16 base, FP32 adapter.
    base = torch.ones((1, 4), dtype=torch.bfloat16)
    lora_a = np.full((1, 4), -0.498, dtype=np.float32)
    lora_b = np.ones((1, 1), dtype=np.float32)
    delta = torch.from_numpy(lora_b) @ torch.from_numpy(lora_a)
    module = SimpleNamespace(base_layer=SimpleNamespace(weight=base), get_delta_weight=lambda _: delta)
    qat = _effective_lora_weight(module, ("default",))
    merged = base.clone()
    merged.add_(delta)  # Same operation as the exporter reconstruction gate.
    scales = np.ones((1, 1), dtype=np.float32)
    findings = []
    for bits in (2, 4):
        raw, _ = exporter._quantize_projection(merged.float().numpy(), scales, bits=bits, working_set_bytes=1048576)
        qat_raw, _ = exporter._quantize_projection(qat.float().numpy(), scales, bits=bits, working_set_bytes=1048576)
        parity = exporter._base_lora_projection_parity(
            merged.float().numpy(), base.float().numpy(), lora_a, lora_b, scales,
            candidate_dtype="BF16", base_dtype="BF16", adapter_dtype="F32",
            scaling=1.0, bits=bits, candidate_raw=raw, working_set_bytes=1048576,
        )
        actual_qat = fake_quantize_ste(qat, bits=bits, per_channel=True, scale_override=torch.from_numpy(scales),
                                     quantizer="ste_ai_edge", ste_gradient="clipped")
        assert qat_raw != raw
        assert parity["numerical_exact"] and parity["retained_scale_code_exact"]
        findings.append({
            "bits": bits, "qat_effective_weight": qat.float().tolist(),
            "export_effective_weight": merged.float().tolist(), "actual_qat_dequantized": actual_qat.float().tolist(),
            "qat_packed_hex": qat_raw.hex(), "export_packed_hex": raw.hex(),
            "export_gate_numerical_exact": parity["numerical_exact"],
            "export_gate_retained_scale_code_exact": parity["retained_scale_code_exact"],
            "qat_and_export_codes_equal": qat_raw == raw,
        })
    return {"kind": "synthetic_cpu_counterexample_not_actual_checkpoint_measurement", "torch_version": torch.__version__,
            "base_dtype": "BF16", "adapter_dtype": "F32", "base_value": 1.0,
            "delta_value_float32": delta[0, 0].item(), "retained_scale": 1.0, "results": findings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path(r"C:\Users\anupk\Downloads\r64_lora_litert_metadata"))
    parser.add_argument("--model", type=Path, default=Path(r"C:\Users\anupk\Downloads\trained_model_data\gemma4_e2b_a2ui_mobile.litertlm"))
    parser.add_argument("--predictions", type=Path, default=Path(r"C:\Users\anupk\Downloads\trained_model_data\e2b_runD_predictions"))
    args = parser.parse_args()
    paths = {"training": args.metadata / "best_golden_checkpoint.training_metadata.json",
             "merge": args.metadata / "qat_mtp_merge_metadata.json",
             "export": args.metadata / "gemma4_retained_scale_code_only_report.json"}
    t, m, e = (load(paths[k]) for k in ("training", "merge", "export"))
    hashes = {k: sha256(v) for k, v in paths.items()}
    model_hash = sha256(args.model)
    training_adapters = {x["path"]: x["sha256"] for x in t["checkpoint_adapter_files"]}
    merge_adapters = {x["path"]: x["sha256"] for x in m["adapter_files"]}
    export_adapters = {x["path"]: x["sha256"] for x in e["adapter_identity"]["files"]}
    checks = {
        "training_metadata_hash_matches_merge": hashes["training"] == m["training_run_metadata"]["sha256"],
        "training_metadata_hash_matches_export": hashes["training"] == e["adapter_identity"]["training_metadata_sha256"],
        "merge_metadata_hash_matches_export": hashes["merge"] == e["merged_checkpoint_identity"]["metadata_sha256"],
        "local_model_hash_matches_export": model_hash == e["output_sha256"],
        "local_model_size_matches_export": args.model.stat().st_size == e["output_inspection"]["file_size"],
        "training_config_hashes_agree": t["training_config_sha256"] == m["training_config_sha256"] == e["training_config"]["sha256"],
        "adapter_hashes_agree": all(training_adapters.get(k) == v == export_adapters.get(k) for k, v in merge_adapters.items()),
        "seed_hashes_agree": t["mobile_training_seed"]["manifest_sha256"] == m["mobile_training_seed"]["manifest_sha256"] == e["mobile_training_seed_identity"]["manifest_sha256"],
        "qparams_hashes_agree": t["qat"]["retained_qparams"]["contract_sha256"] == m["mobile_qparams"]["contract_sha256"] == e["mobile_qparams_identity"]["contract_sha256"],
        "scale_storage_hashes_agree": t["qat"]["retained_qparams"]["scale_storage_sha256"] == m["mobile_qparams"]["scale_storage_sha256"] == e["mobile_qparams_identity"]["scale_storage_sha256"],
        "lora_rank_and_alpha_agree": t["lora"]["r"] == e["adapter_mapping"]["rank"] == 64 and t["lora"]["alpha"] == e["adapter_mapping"]["alpha"] == 64,
        "checkpoint_step_matches_selection": t["checkpoint_step"] == t["best_golden_eval"]["step"] == 9000,
    }
    section = e["target_section"]
    with args.model.open("rb") as stream:
        stream.seek(section["begin_offset"])
        remaining = section["size"]
        digest = hashlib.sha256()
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise EOFError("Model target section truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    checks["local_target_section_hash_matches_export"] = digest.hexdigest() == e["candidate_target_sha256"]

    code_checks = []
    for rel, name in [("training/src/ir_training/qat/fake_quant.py", "_effective_lora_weight"),
                      ("training/scripts/build_gemma4_retained_scale_litertlm.py", "_base_lora_projection_parity")]:
        recorded = subprocess.run(["git", "show", f"{t['git_commit']}:{rel}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
        current = (REPO / rel).read_text(encoding="utf-8")
        code_checks.append({"file": rel, "function": name,
                            "same_ast_at_recorded_training_commit": function_ast(recorded, name) == function_ast(current, name)})

    numeric = t["numeric_preflight"]
    quant = e["trained_quantization"]
    parities = {x["hf_weight_key"]: x for x in quant["base_lora_parity"]["telemetry"]}
    projection_rows = []
    for x in quant["telemetry"]:
        count = x["clipped_low_count"] + x["clipped_high_count"]
        projection_rows.append({"weight": x["hf_weight_key"], "bits": x["bits"], "values": x["value_count"],
                               "clipped_values": count, "clipped_fraction": count / x["value_count"],
                               "codes_changed": x["differs_from_official_codes"],
                               "delta_to_base_l2_ratio": parities[x["hf_weight_key"]]["delta_to_base_l2_ratio"]})
    bit_groups = {}
    for bits in sorted({r["bits"] for r in projection_rows}):
        subset = [r for r in projection_rows if r["bits"] == bits]
        bit_groups[str(bits)] = {"projections": len(subset), "values": sum(r["values"] for r in subset),
                                "clipped_values": sum(r["clipped_values"] for r in subset)}

    golden_path = args.predictions / "r64_golden32_scored.jsonl"
    golden = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in golden:
        grouped[row["source_id"]].append(row["metrics"]["generation_reward_v5_4"])
    reward = sum(sum(v) / len(v) for v in grouped.values()) / len(grouped)
    checks["supplied_golden32_metric_equals_selected_metric"] = reward == t["best_golden_eval"]["metric_value"]
    assert all(checks.values()), checks
    assert all(c["same_ast_at_recorded_training_commit"] for c in code_checks)

    result = {
        "scope": "metadata_binding_audit_and_synthetic_cpu_reproduction_no_fresh_model_inference",
        "metadata_directory": str(args.metadata), "input_hashes": hashes, "local_model_sha256": model_hash,
        "checks": checks, "recorded_run_id": t["run_id"], "recorded_training_commit": t["git_commit"],
        "source_verification": code_checks, "lora": t["lora"], "selected_checkpoint": t["best_golden_eval"],
        "train_max_seq_length": t["training"]["max_seq_length"], "token_cache_splits": t["token_cache"]["splits"],
        "golden_max_input_tokens": t["golden_eval"]["max_input_tokens"], "golden_max_new_tokens": t["golden_eval"]["max_new_tokens"],
        "native_runtime_numeric_parity_verified": t["qat"]["numeric_contract"]["native_runtime_numeric_parity_verified"],
        "native_kv_cache_simulated": t["qat"]["native_kv_cache_simulated"],
        "preflight": {"policy": numeric["policy"], "cross_mode_comparison": numeric["cross_mode_comparison"],
                      "qat_off_loss": numeric["baseline"]["completion_loss"], "qat_on_loss": numeric["qat_on"]["completion_loss"],
                      "top1_match": numeric["top1_probe_match_fraction"], "diagnostic_min_top1_match": numeric["min_top1_probe_match"],
                      "greedy_common_prefix": numeric["greedy_generation"]["baseline_qat_min_common_prefix_tokens"],
                      "diagnostic_min_common_prefix": numeric["greedy_generation"]["min_baseline_qat_greedy_prefix_tokens"],
                      "greedy_probe_rows": numeric["greedy_generation"]["qat_on"]["rows_checked"],
                      "greedy_probe_tokens": numeric["greedy_generation"]["qat_on"]["max_new_tokens"],
                      "mandatory_checks": numeric["mandatory_checks"],
                      "greedy_mandatory_checks": numeric["greedy_generation"]["mandatory_checks"],
                      "saturation": numeric["saturation_telemetry"]["summary"]},
        "supplied_export_gates": e["gates"], "all_supplied_export_gates_pass": all(e["gates"].values()),
        "projection_telemetry": {"by_bits": bit_groups, "total_values": quant["total_values"],
                                "clipped_fraction": (quant["total_clipped_low"] + quant["total_clipped_high"]) / quant["total_values"],
                                "changed_projections": quant["changed_from_official_count"],
                                "unchanged_projection_names": [r["weight"] for r in projection_rows if not r["codes_changed"]],
                                "top_clipped_projections": sorted(projection_rows, key=lambda r:r["clipped_fraction"], reverse=True)[:8]},
        "supplied_golden32": {"sha256": sha256(golden_path), "rows": len(golden), "unique_sources": len(grouped),
                              "unique_source_reward": reward, "note": "Score equality supports consistency, not a per-prediction adapter hash or QAT-mode receipt."},
        "rounding_reproduction": rounding_reproduction(),
        "missing_runtime_evidence": ["Exact evaluation result/manifest binding prediction hashes to adapter hash and qat_applied",
                                     "Same input-token-ID and teacher-forced logits comparison across QAT checkpoint, native CPU and GPU",
                                     "Late-training activation saturation telemetry", "Actual adapter and base weights for code-difference census"],
    }
    (HERE / "audit_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (HERE / "projection_telemetry.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(projection_rows[0]))
        writer.writeheader()
        writer.writerows(projection_rows)
    print(json.dumps({"binding_checks": len(checks), "all_pass": all(checks.values()),
                      "export_gates": len(e["gates"]), "reproduced_rounding_mismatch_bits": [2, 4],
                      "output": str(HERE / "audit_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
