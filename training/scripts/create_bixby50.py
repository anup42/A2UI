"""Reproduce the frozen Bixby50 source-only holdout without generating targets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.common.config import repo_root
from ir_training.data.bixby50 import SOURCE_RESPONSES_PATH, build_bixby50, serialize_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo_root() / SOURCE_RESPONSES_PATH)
    parser.add_argument("--output-dir", type=Path, required=True, help="New destination; existing folders are refused")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    raw = args.source.read_bytes()
    rows, manifest = build_bixby50(
        [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()],
        source_responses_sha256=hashlib.sha256(raw).hexdigest(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "bixby50.jsonl").write_bytes(serialize_rows(rows))
    with (args.output_dir / "benchmark_manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"rows": len(rows), "reference_available": False,
                      "output_sha256": manifest["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
