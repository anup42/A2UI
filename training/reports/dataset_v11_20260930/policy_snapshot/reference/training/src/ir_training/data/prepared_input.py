"""Import a portable, frozen preparation without filtering or retokenizing rows.

Only local tokenizer files are loaded. Absolute producer paths in manifests are
provenance, never inputs. Semantic bundles must carry flat, hash-bound evidence.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from ir_training.common.config import repo_root
from ir_training.data.augmentation import _rows
from ir_training.data.golden_replacement import read_rows_strict, source_identity_record
from ir_training.eval.golden_set import benchmark_contract_for_split
from ir_training.eval.prepared_contract import checked_preparation_manifest, file_sha256
from ir_training.train.prepared_binding import verify_tokenizer_binding

SPLITS = ("train", "val", "golden32", "golden35", "bixby50")
REQUIRED_FILES = {
    "manifest.json", "prompt_scaffolds.json", "shared_prompt.json",
    "inference_prompt.json", "source_prompt_scaffolds.json",
    *(f"{name}.jsonl" for name in SPLITS),
}
SEMANTIC_EVIDENCE = {
    "source_manifest_sha256": "augmentation_source_manifest.json",
    "generation_manifest_sha256": "augmentation_generation_manifest.json",
    "donors_sha256": "augmentation_donors.jsonl",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")  # noqa: TRY004 - malformed artifact schema
    return value


def _no_links(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError(f"Prepared input paths must not traverse links: {candidate}")


def prepared_input_files(directory: Path) -> list[Path]:
    """List every regular, flat bundle file, including its portable evidence."""
    directory = Path(directory).absolute()
    _no_links(directory)
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError("Prepared input must be a directory")
    files = sorted(directory.iterdir())
    for path in files:
        _no_links(path)
        if not path.is_file():
            raise ValueError(f"Prepared input must contain regular files only: {path.name}")
    missing = REQUIRED_FILES - {path.name for path in files}
    if missing:
        raise ValueError(f"Prepared input missing required files: {sorted(missing)}")
    return files


def _golden_bindings(directory: Path, plan: dict) -> dict:
    """Compare current pinned cohorts, not arbitrary count-compatible holdouts."""
    from ir_training.data.express_preparation import serialize_checked

    bindings = {}
    for name in SPLITS[2:]:
        pin = plan.get("goldens", {}).get(name) or {}
        source = Path(pin.get("path", ""))
        evidence = Path(pin.get("benchmark_manifest_path", ""))
        if (not source.is_file() or file_sha256(source) != pin.get("sha256")
                or not evidence.is_file() or file_sha256(evidence) != pin.get("benchmark_manifest_sha256")):
            raise ValueError(f"Pinned {name} source/benchmark hash mismatch")
        expected_rows = read_rows_strict(source)
        actual_rows = read_rows_strict(directory / f"{name}.jsonl")
        expected_contract = benchmark_contract_for_split(source, expected_rows)
        actual_contract = benchmark_contract_for_split(directory / f"{name}.jsonl", actual_rows)
        if expected_contract != _read(evidence) or actual_contract != expected_contract:
            raise ValueError(f"Prepared {name} benchmark lineage differs from the pinned cohort")
        expected = [source_identity_record(row) for row in expected_rows]
        actual = [source_identity_record(row) for row in actual_rows]
        if actual != expected or len(actual) != pin.get("rows"):
            raise ValueError(f"Prepared {name} canonical source membership differs from the pinned cohort")
        if name != "bixby50":
            for original, prepared in zip(expected_rows, actual_rows, strict=True):
                if serialize_checked(original["completion"], "root-first").semantic_sha256 != serialize_checked(prepared["completion"], "root-first").semantic_sha256:
                    raise ValueError(f"Prepared {name} target differs from the pinned reference")
        bindings[name] = {"source_sha256": pin["sha256"],
                          "benchmark_manifest_sha256": pin["benchmark_manifest_sha256"],
                          "canonical_membership_sha256": _digest(actual), "rows": len(actual)}
    return bindings


def _augmentation_binding(directory: Path, manifest: dict, hashes: dict) -> dict | None:
    declared = manifest.get("augmentation")
    binding_name = "augmentation_reference_bindings.json"
    evidence_names = {*SEMANTIC_EVIDENCE.values(), "augmentation_accepted_genui.jsonl", "augmentation.json", binding_name}
    if declared is None:
        if evidence_names & hashes.keys() or manifest.get("augmentation_sha256") or any(
            (row.get("metadata") or {}).get("augmentation") for _, _, row in _rows(directory / "train.jsonl")
        ):
            raise ValueError("Undeclared augmentation rows/evidence in prepared input")
        return None
    if not isinstance(declared, dict) or declared.get("kind") != "stage3_semantic_generation":
        raise ValueError("Prepared import currently supports only sealed semantic augmentation or plain preparations")
    if hashes.get("augmentation.json") != manifest.get("augmentation_sha256"):
        raise ValueError("Prepared augmentation report hash mismatch")
    report = _read(directory / "augmentation.json")
    if report != declared:
        raise ValueError("Prepared augmentation report differs from manifest")
    for key, name in SEMANTIC_EVIDENCE.items():
        if name not in hashes or hashes[name] != report.get(key):
            raise ValueError(f"Unsealed semantic augmentation: missing or changed {name}")
    original = _read(directory / SEMANTIC_EVIDENCE["source_manifest_sha256"])
    generation = _read(directory / SEMANTIC_EVIDENCE["generation_manifest_sha256"])
    accepted_name = "augmentation_accepted_genui.jsonl"
    if (accepted_name not in hashes or hashes[accepted_name] != generation.get("accepted_genui_sha256")
            or generation.get("status") != "completed" or generation.get("donors_sha256") != report["donors_sha256"]
            or generation.get("teacher_model") != report.get("teacher_model") or generation.get("seed") != report.get("seed")):
        raise ValueError("Semantic generation evidence hash/status/teacher/seed mismatch")
    if original.get("augmentation") or original.get("shared_prompt") != manifest.get("shared_prompt") or original.get("tokenizer") != manifest.get("tokenizer"):
        raise ValueError("Semantic augmentation base preparation contract mismatch")
    splits = manifest["splits"]
    if splits["train"].get("unaugmented_preparation") != (original.get("splits") or {}).get("train"):
        raise ValueError("Semantic augmentation unaugmented preparation mismatch")
    for name in SPLITS[1:]:
        digest = hashes[f"{name}.jsonl"]
        if (report.get("untouched_split_sha256", {}).get(name) != digest
                or (original.get("splits", {}).get(name) or {}).get("output_sha256") != digest):
            raise ValueError(f"Semantic augmentation changed reserved split: {name}")
    accepted = read_rows_strict(directory / accepted_name)
    by_hash = {_digest(row): row for row in accepted}
    donors = read_rows_strict(directory / "augmentation_donors.jsonl")
    by_donor = {row.get("donor_id"): row for row in donors}
    if len(by_donor) != len(donors) or any(row.get("split") != "train" for row in donors):
        raise ValueError("Semantic evidence contains duplicate or non-training donors")
    reference_bindings = None
    binding_hash = report.get("reference_bindings_sha256")
    if binding_hash is not None:
        if (hashes.get(binding_name) != binding_hash
                or generation.get("reference_bindings_sha256") != binding_hash):
            raise ValueError("Semantic reference binding evidence missing or changed")
        bindings = _read(directory / binding_name)
        reference_bindings = bindings.get("bindings")
        if (bindings.get("version") != 1 or bindings.get("donors_sha256") != report["donors_sha256"]
                or not isinstance(reference_bindings, dict) or set(reference_bindings) != set(by_donor)):
            raise ValueError("Semantic reference binding donor inventory mismatch")
    elif binding_name in hashes or generation.get("reference_bindings_sha256") is not None:
        raise ValueError("Undeclared semantic reference binding evidence")
    if (len(accepted) != generation.get("accepted_rows") or len(accepted) != report.get("stage3_accepted_rows")
            or generation.get("attempted_rows") != report.get("attempted_rows")
            or not len(accepted) <= generation.get("attempted_rows", -1) <= len(donors)):
        raise ValueError("Semantic augmentation generation count mismatch")
    original_count = report.get("original_rows")
    added_count = report.get("added_rows")
    if (type(original_count) is not int or type(added_count) is not int or added_count < 1
            or original_count != original["splits"]["train"].get("accepted_rows")
            or original_count + added_count != splits["train"].get("accepted_rows")
            or hashes["train.jsonl"] != report.get("output_train_sha256")):
        raise ValueError("Semantic augmentation output count/hash mismatch")
    # Producer appends candidates after byte-preserved originals. The original
    # final newline may be added, so compare its hash with and without that LF.
    prefix_digest, prefix_without_lf = hashlib.sha256(), None
    augmented, found_donors = [], set()
    original_tokens, added_tokens = 0, 0
    row_count = 0
    for row_count, (_, raw, row) in enumerate(_rows(directory / "train.jsonl"), 1):
        if row_count <= original_count:
            if row_count == original_count:
                prefix_without_lf = prefix_digest.copy()
                prefix_without_lf.update(raw.removesuffix(b"\n"))
            prefix_digest.update(raw)
            tokens = ((row.get("metadata") or {}).get("express_preparation") or {}).get("tokenization") or {}
            count = tokens.get("sequence_tokens")
            if type(count) is not int or not 0 < count <= manifest["tokenizer"]["max_seq_length"]:
                raise ValueError("Semantic original training token evidence invalid")
            original_tokens += count
            if (row.get("metadata") or {}).get("augmentation"):
                raise ValueError("Semantic augmentation must not recursively augment original rows")
            identity = source_identity_record(row)
            donor = by_donor.get(identity["row_id"])
            if donor is not None:
                if donor.get("source_group_id") != identity["source_id"] or donor.get("response_text") != row.get("response_text"):
                    raise ValueError("Semantic donor differs from original training source")
                if reference_bindings is not None:
                    from ir_training.data.reference_binding import (
                        _reference_map,
                        reference_tokens,
                    )
                    source = row["response_text"]
                    mapping = _reference_map(row)
                    expected_binding = {
                        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                        "reference_map": {token: mapping[token] for token in sorted(reference_tokens(source)) if token in mapping},
                    }
                    if reference_bindings[identity["row_id"]] != expected_binding:
                        raise ValueError("Semantic reference binding differs from original training source")
                found_donors.add(identity["row_id"])
        else:
            if row_count > original_count + added_count:
                raise ValueError("Semantic augmentation output count mismatch")
            augmented.append(row)
    if row_count != original_count + added_count or found_donors != set(by_donor):
        raise ValueError("Semantic augmentation output count or original donor membership mismatch")
    original_hash = original["splits"]["train"].get("output_sha256")
    if original_hash not in {prefix_digest.hexdigest(), prefix_without_lf.hexdigest() if prefix_without_lf else None}:
        raise ValueError("Semantic augmentation original training prefix changed")
    used = set()
    used_donors = set()
    for row in augmented:
        provenance = (row.get("metadata") or {}).get("augmentation") or {}
        record_hash = provenance.get("stage3_record_sha256")
        candidate = by_hash.get(record_hash)
        if candidate is None or record_hash in used:
            raise ValueError("Semantic prepared row missing unique Stage 3 evidence")
        used.add(record_hash)
        original_provenance = candidate.get("augmentation") or {}
        if any(provenance.get(key) != value for key, value in original_provenance.items()):
            raise ValueError("Semantic prepared row provenance differs from Stage 3")
        donor = by_donor.get(original_provenance.get("donor_id"))
        if (not donor or donor.get("source_group_id") != original_provenance.get("source_group_id")
                or hashlib.sha256(donor["response_text"].encode("utf-8")).hexdigest() != original_provenance.get("donor_source_sha256")):
            raise ValueError("Semantic prepared row source/donor lineage mismatch")
        if donor["donor_id"] in used_donors:
            raise ValueError("Semantic augmentation reused a donor family")
        used_donors.add(donor["donor_id"])
        tokens = ((row.get("metadata") or {}).get("express_preparation") or {}).get("tokenization") or {}
        count = tokens.get("sequence_tokens")
        if type(count) is not int or not 0 < count <= manifest["tokenizer"]["max_seq_length"]:
            raise ValueError("Semantic added training token evidence invalid")
        added_tokens += count
        if report.get("reference_normalization_version") is not None:
            from ir_training.data.semantic_reference_normalization import (
                VERSION as NORMALIZATION_VERSION,
            )
            from ir_training.data.semantic_reference_normalization import (
                verify_normalized_semantic_candidate,
            )
            if report["reference_normalization_version"] != NORMALIZATION_VERSION:
                raise ValueError("Unknown semantic reference normalization version")
            verify_normalized_semantic_candidate(row, candidate)
        else:
            # Previously sealed bundles retain their original raw-source contract.
            from ir_training.data.express_preparation import serialize_checked
            if (row.get("response_text") != candidate.get("response_text")
                    or serialize_checked(row["completion"], "root-first").semantic_sha256 != serialize_checked(candidate["a2ui_express"], "root-first").semantic_sha256):
                raise ValueError("Semantic prepared source/target differs from accepted Stage 3 evidence")
        from ir_training.data.express_preparation import _api
        _api()
        from pipeline.training_augmentation import admission_errors
        if admission_errors(candidate, candidate["response_text"]):
            raise ValueError("Semantic Stage 3 evidence is not admitted")
    categories = Counter((row["metadata"]["augmentation"])["category"] for row in augmented)
    if dict(categories) != report.get("category_counts"):
        raise ValueError("Semantic augmentation category count mismatch")
    if (original_tokens != report.get("original_tokens") or added_tokens != report.get("added_tokens")
            or type(report.get("extra_token_budget")) is not int or added_tokens > report["extra_token_budget"]):
        raise ValueError("Semantic augmentation token budget evidence mismatch")
    return report


def validate_prepared_input(plan: dict, *, tokenizer_loader=None) -> dict:
    """Read-only validation shared by bundle publication and import."""
    options = plan["options"]
    source = Path(options["prepared_input_dir"]).absolute()
    files = prepared_input_files(source)
    source = source.resolve()
    hashes = {path.name: file_sha256(path) for path in files}
    manifest = checked_preparation_manifest(source)
    if not plan.get("shared_prompt") or manifest.get("shared_prompt") != plan["shared_prompt"]:
        raise ValueError("Prepared input shared prompt differs from the current plan")
    tokenizer_contract = manifest.get("tokenizer") or {}
    for key in ("max_seq_length", "max_input_tokens"):
        if tokenizer_contract.get(key) != options[key]:
            raise ValueError(f"Prepared input tokenizer limit mismatch: {key}")
    if tokenizer_contract.get("chat_template_kwargs") != {"enable_thinking": False}:
        raise ValueError("Prepared input must use the non-thinking student chat template")
    if tokenizer_loader is None:
        from ir_training.pipeline.golden_training import _load_tokenizer
        tokenizer_loader = _load_tokenizer
    tokenizer = tokenizer_loader(Path(options["model_dir"]), options["profile"])
    tokenizer_evidence = verify_tokenizer_binding(source, tokenizer,
        {"chat_template_kwargs": {"enable_thinking": False}}, required=True)
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(tokenizer, key, None) != tokenizer_contract.get(key):
            raise ValueError(f"Prepared input special token mismatch: {key}")
    for name, entry in manifest.get("splits", {}).items():
        if not isinstance(name, str) or name not in {Path(path).stem for path in hashes if path.endswith(".jsonl")}:
            raise ValueError("Prepared split must name a local bundle JSONL file")
        if hashes.get(f"{name}.jsonl") != entry.get("output_sha256"):
            raise ValueError(f"Prepared split hash mismatch: {name}")
    sys.path.insert(0, str(repo_root() / "training/scripts"))
    from prepare_review_training import verify_prepared
    validation = verify_prepared(source, source / "golden32.jsonl", golden35=source / "golden35.jsonl",
        bixby50=source / "bixby50.jsonl", max_sequence=options["max_seq_length"], max_prompt=options["max_input_tokens"])
    goldens = _golden_bindings(source, plan)
    augmentation = _augmentation_binding(source, manifest, hashes)
    current = {path.name: file_sha256(path) for path in prepared_input_files(source)}
    if current != hashes:
        raise ValueError("Prepared input changed while validating")
    return {"mode": "reuse_verified_prepared_input", "source_directory": str(source), "source_files": hashes,
            "tokenizer": tokenizer_evidence, "validation": validation, "goldens": goldens,
            "augmentation": augmentation, "retokenized": False, "generation_executed": False,
            "student_weights_loaded": False}


def adopt_prepared_input(plan: dict, *, tokenizer_loader=None) -> dict:
    """Check and copy a complete bundle; never load weights or contact a teacher."""
    options = plan["options"]
    source = Path(options["prepared_input_dir"]).absolute()
    output = Path(options["output_dir"]).absolute()
    _no_links(source)
    _no_links(output)
    source, output = source.resolve(), output.resolve()
    destination = output / "prepared"
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Prepared input and output must not overlap")
    if destination.exists() or (output / "data_audit.json").exists():
        raise FileExistsError("Prepared import requires fresh prepared/ and data_audit.json outputs")
    report = validate_prepared_input(plan, tokenizer_loader=tokenizer_loader)
    hashes = report["source_files"]
    files = prepared_input_files(source)
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepared-import-", dir=output))
    for path in files:
        shutil.copyfile(path, temporary / path.name)
    copied = {path.name: file_sha256(path) for path in prepared_input_files(temporary)}
    current = {path.name: file_sha256(path) for path in prepared_input_files(source)}
    if current != hashes or copied != hashes:
        raise ValueError("Prepared input changed while importing; no prepared bundle published")
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)
    report.update(destination_directory=str(destination), destination_files=copied)
    (output / "data_audit.json").write_text(_json(report) + "\n", encoding="utf-8")
    return report


def adopt_prepared_augmentation(plan: dict, *, tokenizer_loader=None) -> dict:
    """Adopt a sealed combined bundle only when its original base matches this run.

    The caller has already prepared raw inputs into output/prepared. Copy the
    complete bundle to output/augmented, never append it to those original rows.
    No teacher, generation command, student weights or retokenization is used.
    """
    options = plan["options"]
    if options.get("augmentation") != "semantic" or not options.get("augmentation_dir"):
        raise ValueError("Offline semantic augmentation requires --augmentation and --augmentation-dir")
    source = Path(options["augmentation_dir"]).absolute()
    output = Path(options["output_dir"]).absolute()
    _no_links(source)
    _no_links(output)
    source, output = source.resolve(), output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Augmentation input and output must not overlap")
    base, destination, receipt = output / "prepared", output / "augmented", output / "augmentation_import.json"
    for path in (base, destination, receipt):
        _no_links(path)
    if destination.exists() or receipt.exists():
        raise FileExistsError("Augmentation import requires fresh augmented/ and augmentation_import.json outputs")
    # Require a completed fresh preparation before loading even the tokenizer.
    prepared_input_files(base)
    source_files = prepared_input_files(source)
    if tokenizer_loader is None:
        from ir_training.pipeline.golden_training import _load_tokenizer
        tokenizer_loader = _load_tokenizer
    loaded_tokenizer = None

    def once(model_dir, profile):
        nonlocal loaded_tokenizer
        if loaded_tokenizer is None:
            loaded_tokenizer = tokenizer_loader(model_dir, profile)
        return loaded_tokenizer

    def validation_plan(directory):
        return {**plan, "options": {**options, "prepared_input_dir": str(directory)}}

    base_report = validate_prepared_input(validation_plan(base), tokenizer_loader=once)
    if base_report["augmentation"] is not None:
        raise ValueError("Fresh base must not already contain augmentation")
    source_report = validate_prepared_input(validation_plan(source), tokenizer_loader=once)
    if source_report["augmentation"] is None:
        raise ValueError("--augmentation-dir requires a sealed semantic bundle, not a plain preparation")
    base_hashes, source_hashes = base_report["source_files"], source_report["source_files"]
    original = _read(source / "augmentation_source_manifest.json")
    current = checked_preparation_manifest(base)
    for name in SPLITS:
        if base_hashes[f"{name}.jsonl"] != (original.get("splits", {}).get(name) or {}).get("output_sha256"):
            raise ValueError(f"Fresh base {name} differs from the augmentation's original preparation")
    if current.get("shared_prompt") != original.get("shared_prompt"):
        raise ValueError("Fresh base shared prompt differs from the augmentation's original preparation")
    # Ignore producer-local paths, never tokenizer behavior or limits.
    tokenizer_keys = ("class", "vocabulary_sha256", "chat_template_sha256", "chat_template_kwargs",
                      "bos_token_id", "eos_token_id", "pad_token_id", "add_special_tokens",
                      "max_seq_length", "max_input_tokens")
    for key in tokenizer_keys:
        if (current.get("tokenizer") or {}).get(key) != (original.get("tokenizer") or {}).get(key):
            raise ValueError(f"Fresh base tokenizer differs from augmentation: {key}")
    temporary = Path(tempfile.mkdtemp(prefix=".augmentation-import-", dir=output))
    for path in source_files:
        shutil.copyfile(path, temporary / path.name)
    copied = {path.name: file_sha256(path) for path in prepared_input_files(temporary)}
    current_source = {path.name: file_sha256(path) for path in prepared_input_files(source)}
    current_base = {path.name: file_sha256(path) for path in prepared_input_files(base)}
    if copied != source_hashes or current_source != source_hashes or current_base != base_hashes:
        raise ValueError("Augmentation source or fresh base changed while importing; no bundle published")
    for path in (destination, receipt):
        _no_links(path)
        if path.exists():
            raise FileExistsError(path)
    temporary.rename(destination)
    report = {
        **source_report, "mode": "reuse_verified_prepared_augmentation",
        "destination_directory": str(destination), "destination_files": copied,
        "base_directory": str(base), "base_files": base_hashes,
        "base_validation": base_report, "combined_bundle_copied": True,
        "original_rows_appended": False,
        "artifact_paths": sorted({str(path) for path in [
            *(source / name for name in source_hashes), *(base / name for name in base_hashes),
            *(destination / name for name in copied), receipt,
        ]}),
    }
    receipt.write_text(_json(report) + "\n", encoding="utf-8")
    return report
