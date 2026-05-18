from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.export.manifest import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a single Android-importable model package archive.")
    parser.add_argument("--export-dir", required=True, help="Directory containing model_manifest.json and model files.")
    parser.add_argument("--output", help="Output archive path. Defaults beside export dir.")
    args = parser.parse_args()
    export_dir = Path(args.export_dir).resolve()
    manifest_path = export_dir / "model_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing model_manifest.json: {manifest_path}")
    output = Path(args.output).resolve() if args.output else export_dir.with_suffix(".zip")
    if output.exists():
        output.unlink()
    archive_base = output.with_suffix("")
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(export_dir))
    checksum = sha256_file(archive_path)
    package_info = {"archive": archive_path, "sha256": checksum, "bytes": Path(archive_path).stat().st_size}
    print(json.dumps(package_info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
