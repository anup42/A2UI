"""Bounded in-memory legacy-fallback signature diagnostic, never label export.

Exact semantic matches are signatures, not historical generator provenance.
Nonmatches do not prove a sample was not produced by another fallback version.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "dataset/src")]
from ir_training.data.express_preparation import TASK_PREFIX, serialize_checked
from ir_training.data.ir_targets import materialize_completion_targets
from ir_training.data.url_preprocess import preprocess_training_urls
from pipeline.flat_spec_contract import build_fallback_flat_spec


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--extra", action="append", default=[])
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    regular_count = len(samples)
    if args.extra:
        if args.source is None or args.inventory is None:
            raise ValueError("Extra cases require --source and --inventory")
        connection = sqlite3.connect(args.inventory.resolve().as_uri() + "?mode=ro", uri=True)
        for identity in args.extra:
            split, index = identity.split(":")
            line = int(index)
            offset, length, digest = connection.execute("SELECT byte_offset,byte_length,target_sha256 FROM rows WHERE split=? AND line=?", (split,line)).fetchone()
            with (args.source / f"{split}.jsonl").open("rb") as stream:
                bom = stream.read(4)
                encoding = "utf-16-le" if bom.startswith(b"\xff\xfe") else "utf-16-be" if bom.startswith(b"\xfe\xff") else "utf-8-sig"
                stream.seek(offset)
                row = json.loads(stream.read(length).decode(encoding).lstrip("\ufeff"))
            task, target = row["messages"][-2]["content"], row["messages"][-1]["content"]
            if not task.startswith(TASK_PREFIX) or sha(target) != digest:
                raise ValueError("Extra sample differs from bound source/target inventory")
            samples.append({"split":split,"line_1based":line,"response":task[len(TASK_PREFIX):],"target":target})
        connection.close()
    results = []
    for sample in samples:
        expected = serialize_checked(sample["target"],"root-first")
        diagnostic = {"split":sample["split"],"line_1based":sample["line_1based"],
                      "source_sha256":sha(sample["response"]),"target_sha256":sha(sample["target"]),
                      "reference_semantic_sha256":expected.semantic_sha256,
                      "reference_nodes":len(expected.graph["elements"]),
                      "source_heading":sample["response"].splitlines()[0] if sample["response"] else "",
                      "exact_semantic_match":False,"fallback_express_compatible":False}
        try:
            graph = build_fallback_flat_spec(sample["response"])
            diagnostic["fallback_nodes"] = len(graph["elements"])
            masked = preprocess_training_urls(sample["response"],graph,enabled=True)
            candidate = materialize_completion_targets(masked.canonical_graph)["a2ui_express_v1"]
            checked = serialize_checked(candidate,"root-first")
            diagnostic.update(fallback_express_compatible=True,
                              fallback_semantic_sha256=checked.semantic_sha256,
                              exact_semantic_match=expected.semantic_sha256==checked.semantic_sha256,
                              result="exact_semantic_match" if expected.semantic_sha256==checked.semantic_sha256 else "compatible_but_different")
        except (ValueError,TypeError,RecursionError) as exc:
            diagnostic.update(result="fallback_not_current_express_compatible",
                              reason=getattr(exc,"reason",type(exc).__name__),detail=str(exc)[:400])
        results.append(diagnostic)
    source_files = [Path(__file__),ROOT.parent / "dataset/src/pipeline/flat_spec_contract.py",
                    ROOT.parent / "dataset/src/pipeline/stage4_render.py",ROOT / "src/ir_training/data/url_preprocess.py",
                    ROOT / "src/ir_training/data/express_preparation.py"]
    report = {
        "scope":"Bounded diagnostic on saved qualitative samples, plus explicitly requested cases; no full-corpus origin inference",
        "saved_sample_count":regular_count,"extra_sample_count":len(samples)-regular_count,
        "sample_count":len(samples),"result_counts":dict(Counter(row["result"] for row in results)),
        "split_results":{split:dict(Counter(row["result"] for row in results if row["split"]==split)) for split in ("train","val")},
        "method":"Unmodified build_fallback_flat_spec(response), then normal URL preprocessing, Express materialization, strict root-first validation and semantic-hash comparison to the existing reference; all candidate values exist only in memory.",
        "limits":["Exact matches demonstrate a current fallback signature, not proof of which historical process produced the file.",
                  "A failed comparison does not disprove fallback ancestry: versions, URL maps and postprocessing can differ.",
                  "The legacy helper emits some Table properties not accepted by current Express; those failures are retained, not patched for matching.",
                  "No regenerated/fixed training label or Golden target was exported."],
        "sources_sha256":{str(path.relative_to(ROOT.parent)).replace("\\","/"):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "samples_sha256":hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "source_modified":False,"training_targets_exported":False,"model_inference":False,"cases":results,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"samples":len(samples),"results":report["result_counts"]}))


if __name__ == "__main__":
    main()
