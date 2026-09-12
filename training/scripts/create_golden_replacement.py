"""Create an explicit 32-occurrence / 31-case Golden revision in a new folder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.data.golden_replacement import build_replacement, read_rows_strict, serialize_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--failed-identity", default="q_012053")
    parser.add_argument("--donor-identity")
    parser.add_argument("--benchmark-id", default="golden32_archive_repeat_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    actual = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if actual != args.source_sha256:
        raise ValueError("Original source differs from the supplied SHA-256")
    rows, manifest = build_replacement(read_rows_strict(args.source), original_source_sha256=actual,
                                      failed_identity=args.failed_identity, donor_identity=args.donor_identity,
                                      benchmark_id=args.benchmark_id)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "golden32.jsonl").write_bytes(serialize_rows(rows))
    with (args.output_dir / "benchmark_manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
