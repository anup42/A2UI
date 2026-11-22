"""Generate and archive one shard of strict Stage 3 labels for fresh Stage 2 sources."""

from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline import chatgpt_stage3 as genui
from pipeline import openai_stage3_batch as pilot
from pipeline import openai_stage3_batch_runner as runner
from pipeline.source_quality import assess_source_quality
from pipeline.v11_regeneration import _exclusive
from pipeline.v11_regeneration_io import checked, digest, file_hash, safe_path, save


def _screen(rows):
    excluded_ids, excluded_hashes, _ = pilot.existing._benchmark_exclusions()
    known_holds = pilot.fidelity_v2._holds()
    selected, held, seen = [], [], set()
    for row in rows:
        identifier = pilot.existing._row_id(row)
        metadata = row.get("metadata") or {}
        query = pilot.query_for(row, fresh_sources=True)
        if {"messages", "completion", "completion_targets", "a2ui_express"} & row.keys():
            raise ValueError("Fresh Stage 2 sources must contain no previous targets")
        if metadata.get("assigned_split", "train") != "train":
            raise ValueError("Only fresh training sources can be labeled")
        reasons = list(pilot.guard.assess_source(row)["reasons"])
        reasons.extend(pilot.fidelity_v2.assess_source(row, catalog=known_holds)["reasons"])
        if not reasons:
            hashes = pilot.source_hashes(row)
            if pilot.source_aliases(row) & excluded_ids or hashes & excluded_hashes:
                reasons.append("heldout_overlap")
            if hashes & seen:
                reasons.append("duplicate_source")
            quality = assess_source_quality(query, pilot.existing._generation_source(row))
            if quality["training_eligibility"] == "exclude":
                reasons.extend(item["code"] for item in quality["findings"] if item["severity"] == "error")
            if not reasons:
                seen.update(hashes)
        if reasons:
            held.append({"id": identifier, "reasons": sorted(set(reasons))})
        else:
            selected.append(row)
    return selected, held


def _archive(output, work, paths):
    """Verify each compressed member against its sealed input before publishing."""
    pending = output / "evidence.tar.gz.pending"
    with tarfile.open(pending, "w:gz", compresslevel=6) as archive:
        for name in sorted(paths):
            source = safe_path(work / name)
            if work not in source.parents:
                raise ValueError("Audit member escaped shard storage")
            archive.add(source, arcname=name, recursive=False)
    observed = {}
    with tarfile.open(pending, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in paths or member.name in observed:
                raise ValueError("Unexpected archive member")
            with archive.extractfile(member) as stream:
                observed[member.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != paths:
        raise ValueError("Compressed shard evidence differs from original files")
    archive_path = output / "evidence.tar.gz"
    pending.replace(archive_path)
    return archive_path


def label_shard(rows, output_dir, client, *, workers=8, model="gpt-6.2-sol", reasoning_effort="medium", seed=123):
    """Resume one source-only shard. The caller supplies fresh, globally distinct training sources."""
    rows = list(rows)
    genui.worker_count(workers)
    if not 1 <= len(rows) <= 1000 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("A shard requires 1..1000 uniquely identified sources")
    output = safe_path(Path(output_dir))
    output.mkdir(parents=True, exist_ok=True)
    binding = {"sources_sha256": digest(rows), "requested": len(rows), "seed": seed,
               "model": model, "reasoning_effort": reasoning_effort, "profile": "fidelity_v2",
               "wrapper_sha256": file_hash(Path(__file__))}
    with _exclusive(output):
        work = safe_path(output / "work")
        contract = output / "shard.json"
        receipt = output / "result.json"
        if contract.exists():
            if checked(contract) != binding:
                raise ValueError("Shard inputs, settings or wrapper changed on resume")
        elif any(path.name != "regeneration.lock" for path in output.iterdir()):
            raise ValueError("Fresh shard output contains unrelated files")
        else:
            save(contract, binding)
        if receipt.exists():
            report = checked(receipt)
            if report["shard_sha256"] != digest(binding) or file_hash(output / "evidence.tar.gz") != report["archive_sha256"]:
                raise ValueError("Completed Stage 3 shard evidence changed")
            return report
        work.mkdir(exist_ok=True)
        source = work / "input"
        source.mkdir(exist_ok=True)
        eligible, held = _screen(rows)
        runner.write_rows(source / "sources.jsonl", rows)
        runner.write_rows(source / "source_holds.jsonl", held)
        runner.write_rows(source / "train.jsonl", eligible)
        runner.write_rows(source / "val.jsonl", [])
        paths = {f"input/{path.name}": file_hash(path) for path in source.iterdir()}
        accepted = 0
        if eligible:
            prepared = work / "p"
            if not prepared.exists():
                with tempfile.TemporaryDirectory(dir=work, prefix="s-") as temporary:
                    staging = Path(temporary) / "p"
                    pilot.prepare(source, staging, samples=len(eligible), seed=seed, model=model,
                                  reasoning_effort=reasoning_effort, generation_workers=workers,
                                  transport="chatgpt", pilot_profile="fidelity_v2", fresh_sources=True)
                    staging.rename(prepared)
            pilot_contract = genui.load(prepared)
            selected = runner.read_rows(prepared / "selected.jsonl")
            if {digest(item["row"]) for item in selected} != {digest(row) for row in eligible}:
                raise ValueError("Prepared shard does not match source lineage")
            status = genui.run(prepared, client=client, auth_home=getattr(client, "home", None), workers=workers)
            if status["status"] not in {"generated", "already_collected"}:
                raise RuntimeError(f"Stage 3 paused with saved progress: {status.get('reason', status['status'])}")
            report = genui.collect(prepared)
            accepted = report["accepted"]
            collection = checked(prepared / "chatgpt_collection.json")
            files = {**pilot_contract["files"], **collection["files"]}
            for name in ("contract.json", "chatgpt_collection.json"):
                files[name] = file_hash(prepared / name)
            paths.update({f"p/{name}": sha for name, sha in files.items()})
        archive = _archive(output, work, paths)
        result = {"requested": len(rows), "dispatched": len(eligible), "source_held": len(held),
                  "accepted": accepted, "held": len(rows) - accepted, "archive": str(archive),
                  "archive_sha256": file_hash(archive), "shard_sha256": digest(binding)}
        save(receipt, result)
        # This private work directory is created only after the shard contract is bound.
        shutil.rmtree(work)
        return result
