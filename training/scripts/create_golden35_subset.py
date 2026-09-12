"""Reproduce the approved Golden35 subset from pinned historical source files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ir_training.common.config import repo_root
from ir_training.data.build_pairs import prepare_dataset
from ir_training.data.golden_replacement import read_rows_strict, serialize_rows
from ir_training.data.golden35_subset import SOURCE_GENUI_SHA256, SOURCE_RESPONSES_SHA256, SOURCE_RUN, build_golden35_subset


def source_config(output_dir: Path) -> dict:
    return {
        "run": {"id": "golden35_v1_source_materialization", "seed": 42,
                "source_run_dir": str(repo_root() / SOURCE_RUN), "source_genui_sha256": SOURCE_GENUI_SHA256,
                "source_responses_sha256": SOURCE_RESPONSES_SHA256, "output_dir": str(output_dir),
                "target_format": "a2ui_express_v1", "prompt_version": "genui_gen_mobile_a2ui_express_v1",
                "schema_path": "../dataset/schema/genuicraft_a2ui_express_profile_v1.json", "system_prompt": ""},
        "filters": {"require_strict_express": True, "max_input_chars": 60000, "max_output_chars": 60000, "deduplicate": True},
        "url_preprocessing": {"enabled": True},
        "split": {"train": 1.0, "val": 0.0, "test": 0.0, "stratify_by": "intent_bucket"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New destination; existing folders are refused")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = repo_root() / SOURCE_RUN / "genui.jsonl"
    responses = source.with_name("responses.jsonl")
    with tempfile.TemporaryDirectory(prefix="a2ui-golden35-materialization-") as directory:
        temporary = Path(directory) / "prepared"
        prepare_dataset(source_config(temporary))
        rows, manifest = build_golden35_subset(read_rows_strict(temporary / "all.jsonl"), read_rows_strict(source),
            source_genui_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_responses_sha256=hashlib.sha256(responses.read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "golden35.jsonl").write_bytes(serialize_rows(rows))
    with (args.output_dir / "benchmark_manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"rows": 35, "unique_sources": 35, "output_sha256": manifest["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
