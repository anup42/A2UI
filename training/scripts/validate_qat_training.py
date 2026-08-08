from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.config import load_yaml
from ir_training.qat.workflow import validate_qat_config
from ir_training.qat_mtp.workflow import summarize_issues

DEFAULT_CONFIGS = (
    "training/configs/models/gemma4_e2b_mobile_seed_ir_qat_sft.yaml",
    "training/configs/models/gemma3_270m_ir_qat_sft.yaml",
    "training/configs/models/functiongemma_270m_ir_qat_sft.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statically validate true-QAT Gemma SFT profiles.")
    parser.add_argument(
        "--config",
        action="append",
        help="Training YAML path; repeat to validate multiple profiles (defaults to all supported profiles).",
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument("--warnings-as-errors", action="store_true")
    args = parser.parse_args()

    config_values = args.config or list(DEFAULT_CONFIGS)
    results: list[dict[str, object]] = []
    for config_value in config_values:
        config_path = _resolve_path(config_value)
        config = load_yaml(config_path)
        result = summarize_issues(validate_qat_config(config))
        result["config"] = str(config_path)
        results.append(result)

    report: dict[str, object] = {
        "profiles": results,
        "ok": all(bool(result["ok"]) for result in results),
        "executes_training": False,
        "loads_models": False,
    }
    if args.warnings_as_errors and any(int(result["warning_count"]) for result in results):
        report["ok"] = False
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)
    if not report["ok"]:
        raise SystemExit(2)


def _resolve_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (ROOT.parent / candidate).resolve()


if __name__ == "__main__":
    main()
