from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.metrics import count_tokens
from pipeline.toon_convert import encode_toon, roundtrip_ok


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def upgrade_genui_jsonl(genui_jsonl_path: Path) -> dict[str, Any]:
    """Rewrite `toon` field in-place using the spec TOON encoder.

    This is safe to run multiple times: it always re-encodes from `genui_json`.
    """
    if not genui_jsonl_path.exists():
        raise FileNotFoundError(str(genui_jsonl_path))

    rows = _iter_jsonl(genui_jsonl_path)

    updated = 0
    for row in rows:
        genui_json = row.get("genui_json")
        if genui_json is None:
            genui_json = row.get("a2ui_json")
        if genui_json is None:
            continue
        toon = encode_toon(genui_json)
        row["toon"] = toon

        validation = row.get("validation")
        if isinstance(validation, dict):
            validation["toon_roundtrip_ok"] = roundtrip_ok(genui_json, toon)

        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            metrics["output_tokens_toon"] = count_tokens(toon)

        updated += 1

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = genui_jsonl_path.with_suffix(genui_jsonl_path.suffix + f".bak_toon_{ts}")
    shutil.copy2(genui_jsonl_path, backup_path)

    tmp_path = genui_jsonl_path.with_suffix(genui_jsonl_path.suffix + ".tmp")
    _write_jsonl(tmp_path, rows)
    tmp_path.replace(genui_jsonl_path)

    return {"updated": updated, "backup": str(backup_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade TOON strings in genui.jsonl to spec format.")
    parser.add_argument(
        "--genui_jsonl",
        required=True,
        type=Path,
        help="Path to a run's genui.jsonl (e.g. dataset/data/runs/gemini_3/genui.jsonl).",
    )
    args = parser.parse_args()

    result = upgrade_genui_jsonl(args.genui_jsonl)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
