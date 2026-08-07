"""Compose a compatible fine-tuned target section with the default MTP assistant.

This command is intentionally separate from ``export_edge_gallery_model.py``.
The latter can invoke the public LiteRT Torch exporter for a standalone model;
this command only packages a target section whose topology/layout has already
been validated against an official Gemma 4 mobile package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.export.litertlm_mtp import compose_with_default_mtp


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replace one compatible target TFLite section in an official "
            ".litertlm while preserving its default MTP drafter section."
        )
    )
    parser.add_argument("--base-litertlm", required=True, help="Official package containing the default MTP section.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-litertlm", help="Candidate package containing the fine-tuned target section.")
    target.add_argument("--target-section", help="Raw candidate target TFLite FlatBuffer (TFL3).")
    parser.add_argument("--output-litertlm", required=True, help="Output composed .litertlm path.")
    parser.add_argument("--target-model-type", default="tf_lite_prefill_decode")
    parser.add_argument("--mtp-model-type", default="tf_lite_mtp_drafter")
    parser.add_argument(
        "--allow-missing-graph-inspection",
        action="store_true",
        help="Do not fail only because generated TFLite Python bindings are unavailable (not recommended).",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing output file.")
    parser.add_argument("--manifest", help="Optional JSON manifest path.")
    args = parser.parse_args()

    try:
        manifest = compose_with_default_mtp(
            base_litertlm=args.base_litertlm,
            target_litertlm=args.target_litertlm,
            target_section=args.target_section,
            output_litertlm=args.output_litertlm,
            target_model_type=args.target_model_type,
            mtp_model_type=args.mtp_model_type,
            require_graph_compatibility=not args.allow_missing_graph_inspection,
            force=args.force,
            manifest_path=args.manifest,
        )
    except Exception as exc:
        print(f"LiteRT-LM MTP composition failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
