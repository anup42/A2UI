from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.qat.mobile_schema import (
    audit_module_names,
    compare_to_public_schema,
    load_mobile_quant_schema,
    resolve_schema_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a Gemma 4 mobile checkpoint config against Google's public "
            "wNa8o8 module schema. This never loads model weights."
        )
    )
    parser.add_argument(
        "--official-config",
        help="Path to the official checkpoint config.json or its containing model directory.",
    )
    parser.add_argument(
        "--schema",
        default="training/configs/quantization/gemma4_e2b_mobile_public_schema.yaml",
        help="Local copy of the public schema YAML.",
    )
    parser.add_argument(
        "--module-list",
        help="Optional JSON array or newline-delimited named_modules list to audit bit assignments.",
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Exit 2 when an observed config differs.")
    args = parser.parse_args()

    schema = load_mobile_quant_schema(resolve_schema_path(args.schema))
    observed_config: dict[str, Any] | None = None
    schema_issues: list[dict[str, str]] = []
    if args.official_config:
        config_path = _config_path(Path(args.official_config))
        observed_config = json.loads(config_path.read_text(encoding="utf-8"))
        schema_issues = [issue.to_dict() for issue in compare_to_public_schema(observed_config, schema)]

    module_audit = None
    if args.module_list:
        module_names = _load_module_names(Path(args.module_list))
        module_audit = audit_module_names(module_names, schema)

    report: dict[str, Any] = {
        "schema_path": str(resolve_schema_path(args.schema)),
        "schema": schema.to_dict(),
        "official_config": str(_config_path(Path(args.official_config))) if args.official_config else None,
        "schema_issues": schema_issues,
        "public_schema_match": not schema_issues if args.official_config else None,
        "module_audit": module_audit,
        "executes_training": False,
        "loads_weights": False,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)
    if args.strict and schema_issues:
        raise SystemExit(2)


def _config_path(value: Path) -> Path:
    path = value if value.is_absolute() else (ROOT.parent / value)
    if path.is_dir():
        path = path / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Official config not found: {path}")
    return path.resolve()


def _load_module_names(path: Path) -> list[str]:
    path = path if path.is_absolute() else (ROOT.parent / path)
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(value, list):
        raise ValueError("--module-list JSON must be an array of module names.")
    return [str(item) for item in value]


if __name__ == "__main__":
    main()
