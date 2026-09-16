"""CPU-only binding checks shared by review preparation and final evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from ir_training.common.config import load_yaml
from ir_training.common.progress import Progress, fingerprint_file, log
from ir_training.data.audit_filter import (
    _identity_keys,
    _source_hashes,
    load_reserved_cohorts,
)
from ir_training.data.golden_replacement import source_identity_record
from ir_training.eval.golden_set import (
    benchmark_contract_for_split,
    load_fixed_golden_rows,
)
from ir_training.train.prepared_binding import value_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_preparation_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    validation = manifest.get("validation") or {}
    if not all(validation.get(key) is True for key in ("strict_express", "wire_schema", "semantic_roundtrip")) or validation.get("root_reachability") != 1.0:
        raise ValueError("Preparation manifest lacks strict target validation.")
    if manifest.get("scaffold_count") != 1:
        raise ValueError("Evaluation requires one shared prompt scaffold.")
    if file_sha256(directory / "prompt_scaffolds.json") != manifest.get("prompt_scaffolds_sha256"):
        raise ValueError("Saved prompt scaffold changed after preparation.")
    if manifest.get("shared_prompt") is not None:
        from ir_training.data.shared_prompt import validate_shared_prompt_snapshot

        contract = validate_shared_prompt_snapshot(manifest["shared_prompt"])
        for name in ("shared_prompt", "inference_prompt", "source_prompt_scaffolds"):
            if file_sha256(directory / f"{name}.json") != manifest.get(f"{name}_sha256"):
                raise ValueError(f"Saved {name} changed after preparation.")
        saved = json.loads((directory / "shared_prompt.json").read_text(encoding="utf-8"))
        if saved != contract:
            raise ValueError("Saved shared prompt differs from its preparation manifest.")
        scaffolds = json.loads((directory / "prompt_scaffolds.json").read_text(encoding="utf-8"))
        if scaffolds != [{"sha256": contract["scaffold_sha256"], **contract["scaffold"]}]:
            raise ValueError("Saved prompt scaffold differs from the frozen shared prompt contract.")
        inference = json.loads((directory / "inference_prompt.json").read_text(encoding="utf-8"))
        expected = {"version": contract["version"], "contract_sha256": contract["contract_sha256"], **contract["scaffold"]}
        if not isinstance(inference, dict) or any(inference.get(key) != value for key, value in expected.items()):
            raise ValueError("Saved inference prompt differs from the frozen shared prompt contract.")
    return manifest


def verify_golden_preparation(
    dataset: Path, split: Path, *, required_rows: int, max_sequence: int,
    max_prompt: int, expected_kind: str | None = None,
) -> dict[str, Any]:
    """Bind a complete Golden cohort to checked training prompt and tokenization."""
    training = checked_preparation_manifest(dataset)
    evaluation = checked_preparation_manifest(split.parent)
    if training["prompt_scaffolds_sha256"] != evaluation["prompt_scaffolds_sha256"]:
        raise ValueError("Training and Golden prompt scaffolds differ.")
    if training.get("shared_prompt") != evaluation.get("shared_prompt"):
        raise ValueError("Training and Golden shared prompt contracts differ.")
    expected = training.get("tokenizer") or {}
    actual = evaluation.get("tokenizer") or {}
    if expected.get("max_seq_length") != max_sequence or not expected.get("vocabulary_sha256"):
        raise ValueError("Prepare training with the selected tokenizer and sequence limit first.")
    for key in ("vocabulary_sha256", "chat_template_sha256", "chat_template_kwargs", "max_seq_length"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"Train and Golden tokenizer/chat-template fingerprints differ: {key}.")
    entry = (evaluation.get("splits") or {}).get(split.stem) or {}
    digest = file_sha256(split)
    if entry.get("output_sha256") != digest:
        raise ValueError("Golden file must match a checked preparation manifest.")
    if entry.get("quarantined_rows", 0):
        raise ValueError(f"Golden preparation quarantined rows; preserve all {required_rows} cases.")
    maxima = entry.get("max_accepted_token_lengths") or {}
    if int(maxima.get("prompt_tokens", max_prompt + 1)) > max_prompt:
        raise ValueError("Golden prompt exceeds inference context. Reprepare with --max-input-tokens.")
    rows = load_fixed_golden_rows(split, required_rows=required_rows, require_exact_rows=True, require_unique_rows=True)
    benchmark = benchmark_contract_for_split(split, rows)
    if expected_kind is not None and (benchmark or {}).get("benchmark_kind") != expected_kind:
        raise ValueError(f"Golden{required_rows} requires a hash-bound {expected_kind} benchmark manifest.")
    return {
        "split_path": str(split.resolve()), "required_rows": required_rows,
        "split_sha256": digest, "manifest_sha256": file_sha256(split.parent / "manifest.json"),
        "prompt_scaffolds_sha256": evaluation["prompt_scaffolds_sha256"],
        "shared_prompt_sha256": evaluation.get("shared_prompt_sha256"),
        "benchmark_id": (benchmark or {}).get("benchmark_id"),
        "benchmark_kind": (benchmark or {}).get("benchmark_kind"),
        "tokenizer": actual,
    }


def verify_reserved_train_validation(dataset: Path, goldens: list[Path]) -> None:
    """Reserve all source IDs, raw/masked texts, and omitted reference sources."""
    reserved = load_reserved_cohorts(goldens)
    for name in ("train", "val"):
        with Progress(f"Check reserved sources in {name}") as progress:
            with (dataset / f"{name}.jsonl").open(encoding="utf-8-sig") as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{name}:{number}: expected a JSON object")
                    if _identity_keys(source_identity_record(row)) & reserved["identities"] or _source_hashes(row) & reserved["responses"]:
                        raise ValueError(f"Reserved Golden source overlaps {name}; excluded cases and raw/URL-masked variants must not enter train/val.")
                    progress.advance()


def _verify_reserved_cached(dataset: Path, goldens: list[Path], hashes: dict[str, str], cache: Path) -> None:
    """Reuse only the JSON/source scan, never skip current-byte hash checks.

    The caller has freshly hashed train/val against their bound manifest. The
    receipt also binds reserved *contents* (including omitted sources) and the
    transitive preprocessing implementation, so changed cohorts/code miss.
    """
    from ir_training.common.cache_store import assert_no_links, cache_lock, digest
    from ir_training.common.config import repo_root
    from ir_training.pipeline.preparation_cache import _implementation

    reserved = load_reserved_cohorts(goldens)
    binding = {"version": 1, "splits": hashes,
               "identities": sorted(reserved["identities"]), "responses": sorted(reserved["responses"]),
               "implementation": _implementation(repo_root())}
    key = digest(binding)
    with cache_lock(cache, key):
        receipt = cache / f"{key}.json"
        assert_no_links(receipt)
        if receipt.is_file():
            try:
                saved = json.loads(receipt.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                saved = None
            if saved == {"binding": binding, "verified": True}:
                log("Evaluation validation cache HIT: train/val bytes and all reserved sources verified; reuse source-overlap scan")
                return
        log("Evaluation validation cache MISS: checking train/val source overlap")
        verify_reserved_train_validation(dataset, goldens)
        # Never publish a scan receipt if inputs changed while being scanned.
        if any(file_sha256(dataset / f"{name}.jsonl") != value for name, value in hashes.items()):
            raise ValueError("Prepared data changed during the source-overlap scan")
        _write_scan_receipt(receipt, {"binding": binding, "verified": True})


def _write_scan_receipt(path: Path, value: dict) -> None:
    # Keep this tiny writer local: importing the training orchestrator here
    # makes its entire training graph part of preprocessing cache identity.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                          prefix=".scan-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def verify_loaded_evaluation_tokenizer(tokenizer: Any, model_config: dict[str, Any], contract: dict[str, Any]) -> None:
    actual = {
        "vocabulary_sha256": value_sha256(tokenizer.get_vocab()),
        "chat_template_sha256": value_sha256(getattr(tokenizer, "chat_template", None)),
        "chat_template_kwargs": model_config.get("chat_template_kwargs") or {},
    }
    for key, value in actual.items():
        if value != contract["tokenizer"].get(key):
            raise ValueError(f"Evaluation tokenizer binding mismatch before model loading: {key}.")


def verify_evaluation_prepared_contract(
    config_path: Path, split: Path, *, required_rows: int, max_input_tokens: int,
) -> dict[str, Any]:
    """Fail before model loading if run, data, prompt, or cohort bindings drift."""
    config = load_yaml(config_path)
    report = json.loads((config_path.parent / "preparation_report.json").read_text(encoding="utf-8"))
    if report.get("training_config_sha256") != file_sha256(config_path):
        raise ValueError("Training config changed after preparation; final evaluation requires the original bound run plan.")
    model_dir = Path(config["model"]["model_source"]).resolve()
    model_files = report.get("model_files")
    if not isinstance(model_files, dict) or not model_files:
        raise ValueError("Final evaluation requires the source model/tokenizer file bindings from run preparation.")
    with Progress("Verify bound source model/tokenizer bytes", unit="stage"):
        for name, digest in model_files.items():
            path = model_dir / name
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f"Source model/tokenizer bundle changed after preparation: {name}.")
    dataset = Path(config["run"]["dataset_dir"]).resolve()
    manifest = checked_preparation_manifest(dataset)
    if file_sha256(dataset / "manifest.json") != report.get("dataset_manifest_sha256"):
        raise ValueError("Prepared training manifest changed after run preparation.")
    hashes = {}
    for name in ("train", "val"):
        hashes[name] = fingerprint_file(dataset / f"{name}.jsonl")["sha256"]
        if hashes[name] != (manifest.get("splits") or {}).get(name, {}).get("output_sha256"):
            raise ValueError(f"Prepared {name} hash differs from its manifest.")
    plans = config.get("final_evaluation_datasets")
    if not isinstance(plans, dict) or plans != report.get("final_evaluation_datasets"):
        raise ValueError("Prepared-contract evaluation requires matching final_evaluation_datasets in config and preparation report.")
    bound = [entry for entry in plans.values() if Path(entry["split_path"]).resolve() == split.resolve()]
    if len(bound) != 1 or bound[0].get("required_rows") != required_rows:
        raise ValueError("Requested evaluation split/count is not bound to the saved training run.")
    expected = bound[0]
    actual = verify_golden_preparation(
        dataset, split, required_rows=required_rows,
        max_sequence=int(config["training"]["max_seq_length"]), max_prompt=max_input_tokens,
        expected_kind=expected.get("benchmark_kind"),
    )
    for key, value in actual.items():
        if value != expected.get(key):
            raise ValueError(f"Final evaluation preparation binding changed: {key}.")
    if (config["model"].get("chat_template_kwargs") or {}) != actual["tokenizer"].get("chat_template_kwargs", {}):
        raise ValueError("Evaluation config changed the prepared tokenizer chat-template options.")
    _verify_reserved_cached(dataset, [Path(entry["split_path"]) for entry in plans.values()], hashes,
                            config_path.parent / "evaluation_validation_cache")
    return actual
