#!/usr/bin/env python3
"""Build labeled viewport contact sheets for a captured Android golden run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--per-sheet", type=int, default=8)
    args = parser.parse_args()

    run_dir = REPO_ROOT / "dataset" / "data" / "runs" / args.run_id
    captures_dir = run_dir / "android_device_rendered"
    output_dir = run_dir / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {row["case_id"]: row for row in read_jsonl(run_dir / "scenarios.jsonl")}
    genui = read_jsonl(run_dir / "genui.jsonl")
    captures = {row["ui_id"]: row for row in read_jsonl(captures_dir / "capture_manifest.jsonl")}
    font = ImageFont.load_default(size=24)
    panel_width = 560
    image_height = 1160
    label_height = 100
    columns = 2

    records: list[tuple[str, Path]] = []
    for row in genui:
        case_id = row["gen"]["case_id"]
        capture = captures[row["ui_id"]]
        screenshot = captures_dir / capture["viewport_screenshot"]
        if not screenshot.exists():
            raise FileNotFoundError(screenshot)
        records.append((case_id, screenshot))

    for sheet_index, start in enumerate(range(0, len(records), args.per_sheet), start=1):
        chunk = records[start : start + args.per_sheet]
        rows = (len(chunk) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * panel_width, rows * (image_height + label_height)), "#eceff4")
        draw = ImageDraw.Draw(sheet)
        for offset, (case_id, screenshot) in enumerate(chunk):
            column = offset % columns
            row_index = offset // columns
            left = column * panel_width
            top = row_index * (image_height + label_height)
            with Image.open(screenshot) as source:
                image = source.convert("RGB")
                image.thumbnail((panel_width, image_height), Image.Resampling.LANCZOS)
                image_left = left + (panel_width - image.width) // 2
                image_top = top + (image_height - image.height) // 2
                sheet.paste(image, (image_left, image_top))
            scenario = scenarios[case_id]
            label = f"{case_id}  {scenario['scenario_slug']}  {scenario['variant']}"
            draw.rectangle((left, top + image_height, left + panel_width, top + image_height + label_height), fill="#ffffff")
            draw.text((left + 14, top + image_height + 18), label, fill="#111827", font=font)
        output_path = output_dir / f"viewport_contact_{sheet_index:02d}.png"
        sheet.save(output_path, optimize=True)
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
