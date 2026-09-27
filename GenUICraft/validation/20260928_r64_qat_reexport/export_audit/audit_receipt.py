"""Audit the supplied receipt and live device hashes, without rerunning export."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEVICE_FILES = {
    "source_model": "/sdcard/gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm",
    "app_model": "/sdcard/Android/data/com.samsung.genuicraft/files/sdk_models/gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm",
    "receipt": "/sdcard/gemma4_retained_scale_code_only_report_r64_qat_compatible.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="R3GL203AKSF")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = ROOT / "device_export_report.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    previous_path = args.metadata_dir / "gemma4_retained_scale_code_only_report.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    device_hashes = {}
    for name, path in DEVICE_FILES.items():
        output = subprocess.check_output(
            ["adb", "-s", args.serial, "shell", "sha256sum", path], text=True
        ).strip()
        value, returned_path = output.split(None, 1)
        assert returned_path.strip() == path and len(value) == 64
        device_hashes[name] = {"path": path, "sha256": value}

    qat = receipt["trained_quantization"]["qat_weight_reconstruction"]
    rows = qat["telemetry"]
    quant_rows = receipt["trained_quantization"]["telemetry"]
    quant_by_key = {row["hf_weight_key"]: row for row in quant_rows}
    previous_by_key = {row["hf_weight_key"]: row for row in previous["trained_quantization"]["telemetry"]}
    changed = sum(row["changed_code_count_from_merged_checkpoint"] for row in rows)
    total = sum(row["value_count"] for row in quant_rows)
    clipped = sum(row["clipped_low_count"] + row["clipped_high_count"] for row in quant_rows)
    checks = {
        "source_model_matches_receipt_hash": device_hashes["source_model"]["sha256"] == receipt["output_sha256"],
        "app_model_matches_receipt_hash": device_hashes["app_model"]["sha256"] == receipt["output_sha256"],
        "pulled_receipt_matches_device": sha(receipt_path) == device_hashes["receipt"]["sha256"],
        "different_from_previous_binary": receipt["output_sha256"] != previous["output_sha256"],
        "correct_mode": receipt["mode"] == "retained_scale_qat_compatible_weights_v1",
        "correct_arithmetic": receipt["weight_arithmetic"] == "qat_bf16_delta_before_add_v1",
        "execution_recorded_successful": receipt["executed"] is True and receipt["passed"] is True,
        "all_28_recorded_gates_true": len(receipt["gates"]) == 28 and all(v is True for v in receipt["gates"].values()),
        "all_plan_checks_true": all(v is True for v in receipt["checks"].values()),
        "all_package_checks_true": all(v is True for v in receipt["package_checks"].values()),
        "205_unique_qat_projections": len(rows) == len({r["hf_weight_key"] for r in rows}) == 205,
        "205_unique_quantization_projections": len(quant_rows) == len(quant_by_key) == 205,
        "same_projection_key_sets": set(quant_by_key) == {r["hf_weight_key"] for r in rows},
        "all_effective_weight_checks_true": all(r["effective_weight_matches_qat_helper"] is True for r in rows),
        "all_code_checks_true": all(r["code_exact"] is True for r in rows),
        "all_dequantized_weight_checks_true": all(r["dequantized_weight_forward_exact"] is True for r in rows),
        "reference_export_packed_hashes_agree": all(r["reference_code_sha256"] == r["export_code_sha256"] == quant_by_key[r["hf_weight_key"]]["packed_sha256"] for r in rows),
        "original_merged_hashes_match_previous_receipt": all(r["original_merged_code_sha256"] == previous_by_key[r["hf_weight_key"]]["packed_sha256"] for r in rows),
        "histogram_totals_match_value_counts": all(sum(r["code_histogram"].values()) == r["value_count"] for r in quant_rows),
        "changed_code_total_consistent": changed == qat["changed_code_count_from_merged_checkpoint"],
        "weight_value_total_consistent": total == receipt["trained_quantization"]["total_values"],
        "clipping_total_consistent": clipped == receipt["trained_quantization"]["total_clipped_low"] + receipt["trained_quantization"]["total_clipped_high"],
        "native_parity_explicitly_unverified": receipt["native_inference_parity_verified"] is False and qat["native_inference_parity_verified"] is False and all(r["native_inference_parity_verified"] is False for r in rows),
        "training_metadata_hash_matches_local_file": receipt["adapter_identity"]["training_metadata_sha256"] == sha(args.metadata_dir / "best_golden_checkpoint.training_metadata.json"),
        "merge_metadata_hash_matches_local_file": receipt["merged_checkpoint_identity"]["metadata_sha256"] == sha(args.metadata_dir / "qat_mtp_merge_metadata.json"),
    }
    identities = ["adapter_identity", "resolved_training_config_identity", "mobile_training_seed_identity", "mobile_qparams_identity", "merged_checkpoint_identity"]
    for key in identities:
        checks[key + "_unchanged_from_previous_receipt"] = receipt[key] == previous[key]
    summary = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_serial": args.serial,
        "scope": "Live file hashes and independent receipt consistency; exporter tensor gates not rerun.",
        "receipt_sha256": sha(receipt_path),
        "previous_receipt_sha256": sha(previous_path),
        "device_hashes": device_hashes,
        "checks": checks,
        "all_consistency_checks_passed": all(checks.values()),
        "reported_export_gates": receipt["gates"],
        "model_identity": {key: receipt[key] for key in identities},
        "adapter_rank": receipt["adapter_mapping"]["rank"],
        "adapter_alpha": receipt["adapter_mapping"]["alpha"],
        "adapter_scaling": receipt["adapter_mapping"]["scaling"],
        "qat_reconstruction": {
            "projections": len(rows), "projections_with_changed_codes": sum(r["changed_code_count_from_merged_checkpoint"] > 0 for r in rows),
            "changed_codes": changed, "adapted_weight_values": total,
            "changed_percent": changed / total * 100,
            "clipped_values": clipped, "clipped_percent": clipped / total * 100,
            "reconstruction_devices": sorted({r["reconstruction_device"] for r in rows}),
            "matmul_dtypes": sorted({r["delta_matmul_dtype"] for r in rows}),
            "autocast_settings": sorted({r["autocast_enabled"] for r in rows}),
        },
        "native_inference_parity_verified": False,
    }
    (ROOT / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks_passed": sum(checks.values()), "checks_total": len(checks), "failed_checks": [k for k,v in checks.items() if not v], "qat_reconstruction": summary["qat_reconstruction"]}, indent=2))
    assert all(checks.values()), "Receipt/device consistency check failed"


if __name__ == "__main__":
    main()
