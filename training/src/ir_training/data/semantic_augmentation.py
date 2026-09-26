"""Freeze train-only Muse augmentations before loading the student model.

Generation belongs to dataset/ and runs in a bounded subprocess. This module
owns donor isolation, actual student-tokenizer admission and immutable prepared
split publication. It never repairs targets or generates inside a DataLoader.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from ir_training.common.bounded_command import run_bounded_command
from ir_training.common.config import repo_root
from ir_training.common.progress import log
from ir_training.data.audit_filter import (
    _identity_keys,
    _source_hashes,
    audit_and_filter_rows,
    load_reserved_cohorts,
)
from ir_training.data.augmentation import (
    _checked_row,
    _rows,
    validate_augmentation_options,
)
from ir_training.data.express_preparation import TASK_PREFIX, prepare_splits
from ir_training.data.golden_replacement import source_identity_record
from ir_training.eval.prepared_contract import checked_preparation_manifest, file_sha256

VERSION = "muse-semantic-startup-v1"
DEFAULT_TEACHER = "muse_glimmer_30b_sglang_reasoning_dflash"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(_json(value) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_json(row) + "\n")


def validate_semantic_options(options: dict[str, Any]) -> None:
    """Validate without reading data, importing a teacher or contacting servers."""
    limit = options.get("augmentation_max_samples", 500)
    if type(limit) is not int or not 1 <= limit <= 100000:
        raise ValueError("augmentation_max_samples must be an integer in [1, 100000]")
    timeout = options.get("augmentation_timeout_seconds", 7200)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("augmentation_timeout_seconds must be positive and finite")
    teacher = options.get("augmentation_teacher_model", DEFAULT_TEACHER)
    if not isinstance(teacher, str) or not teacher.strip():
        raise ValueError("augmentation_teacher_model must be an explicit registered model name")
    interpreter = options.get("augmentation_python")
    if interpreter is not None and (not isinstance(interpreter, (str, Path)) or not str(interpreter).strip()):
        raise ValueError("augmentation_python must identify a Python interpreter")
    validate_augmentation_options(
        max_extra_fraction=options.get("augmentation_max_extra_fraction", .10),
        max_family_copies=options.get("augmentation_max_family_repeats", 2),
    )
    if options.get("augmentation_max_extra_fraction", .10) <= 0:
        raise ValueError("Semantic augmentation requires a positive extra fraction")
    if options.get("augmentation_max_family_repeats", 2) < 2:
        raise ValueError("Semantic augmentation requires a family exposure cap of at least 2")


def _select_donors(source: Path, manifest: dict, reserved: dict, options: dict):
    """Transitive source families; keep only scalar identities in the census."""
    parents, row_ids, group_ids, blocked = [], [], [], []
    owners: dict[str, int] = {}
    source_hashes: set[str] = set()
    original_ids: set[str] = set()
    original_tokens = 0

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for _, _, row in _rows(source / "train.jsonl"):
        _checked_row(row)
        identity = source_identity_record(row)
        if not identity["source_id"] or not identity["row_id"] or identity["row_id"] in original_ids:
            raise ValueError("Semantic augmentation requires unique row IDs and source identities")
        keys, hashes = _identity_keys(identity), _source_hashes(row)
        index = len(parents)
        parents.append(index)
        row_ids.append(identity["row_id"])
        group_ids.append(identity["source_id"])
        blocked.append(bool(keys & reserved["identities"] or hashes & reserved["responses"]))
        original_ids.add(identity["row_id"])
        source_hashes.update(hashes)
        original_tokens += row["metadata"]["express_preparation"]["tokenization"]["sequence_tokens"]
        for key in sorted({"id:" + key for key in keys} | {"text:" + value for value in hashes}):
            if key in owners:
                parents[find(index)] = find(owners[key])
            else:
                owners[key] = index
    if len(parents) != manifest["splits"]["train"]["accepted_rows"] or not parents:
        raise ValueError("Prepared training row count differs from its manifest")
    families = defaultdict(list)
    for index in range(len(parents)):
        families[find(index)].append(index)
    seed = options.get("seed", 42)
    eligible = []
    for members in families.values():
        # Exactly one new variant per chosen family, never recursively sampled.
        if any(blocked[index] for index in members) or len(members) >= options.get("augmentation_max_family_repeats", 2):
            continue
        donor = min(members, key=lambda index: _sha(f"{seed}:{row_ids[index]}"))
        eligible.append(donor)
    eligible.sort(key=lambda index: _sha(f"{seed}:{row_ids[index]}"))
    fraction = options.get("augmentation_max_extra_fraction", .10)
    cap = min(options.get("augmentation_max_samples", 500), math.floor(len(parents) * fraction), len(eligible))
    if cap < 1:
        raise ValueError("No eligible train-only donor families within augmentation budget; no teacher calls made")
    selected = set(eligible[:cap])
    donors = []
    for index, (_, _, row) in enumerate(_rows(source / "train.jsonl")):
        if index in selected:
            donors.append({"donor_id": row_ids[index], "split": "train",
                           "source_group_id": group_ids[index], "response_text": row["response_text"]})
    # Stable seed order rather than source-file order determines category assignment.
    donors.sort(key=lambda row: _sha(f"{seed}:{row['donor_id']}"))
    return donors, original_ids, source_hashes, original_tokens, len(families)


def _candidate_rows(generated: Path, donors: list[dict], originals: set[str], source_hashes: set[str], options: dict, *, reserved: dict):
    manifest_path = generated / "manifest.json"
    if generated.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Missing regular generation manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_path = generated / "accepted_genui.jsonl"
    if accepted_path.is_symlink() or not accepted_path.is_file():
        raise ValueError("Missing regular accepted_genui.jsonl augmentation output")
    if manifest.get("accepted_genui_sha256") != file_sha256(accepted_path):
        raise ValueError("Generated augmentation candidate hash differs from manifest")
    candidates = [row for _, _, row in _rows(accepted_path)]
    if type(manifest.get("accepted_rows")) is not int or manifest["accepted_rows"] != len(candidates):
        raise ValueError("Generated augmentation candidate count differs from manifest")
    attempts = manifest.get("attempted_rows")
    if type(attempts) is not int or not len(candidates) <= attempts <= len(donors):
        raise ValueError("Generated augmentation exceeded the bounded donor/attempt budget")
    if not candidates:
        raise ValueError("No accepted semantic augmentation candidates; training was not started")
    donors_by_id = {row["donor_id"]: row for row in donors}
    used_donors, seen_ids, seen_sources = set(), set(originals), set(source_hashes)
    rows = []
    # Import the dataset-owned category contract, without constructing a model.
    from ir_training.data.express_preparation import _api
    _api()
    from pipeline.training_augmentation import CATEGORIES
    from pipeline.training_augmentation import VERSION as DATASET_VERSION
    categories = set(CATEGORIES)
    if (manifest.get("version") != DATASET_VERSION or manifest.get("status") != "completed"
            or manifest.get("teacher_model") != options.get("augmentation_teacher_model", DEFAULT_TEACHER)
            or manifest.get("seed") != options.get("seed", 42)):
        raise ValueError("Generated augmentation manifest version/status/teacher/seed binding mismatch")
    for candidate in candidates:
        provenance = candidate.get("augmentation") or {}
        donor = donors_by_id.get(provenance.get("donor_id"))
        if donor is None or donor["donor_id"] in used_donors:
            raise ValueError("Unknown or repeated augmentation donor; family cap cannot be verified")
        source = candidate.get("response_text")
        completion = candidate.get("a2ui_express")
        if not isinstance(source, str) or not source.strip() or not isinstance(completion, str) or not completion.strip():
            raise ValueError("Augmentation candidate requires complete source and Express completion")
        if (provenance.get("version") != DATASET_VERSION
                or provenance.get("source_group_id") != donor["source_group_id"]
                or provenance.get("donor_source_sha256") != _sha(donor["response_text"])
                or provenance.get("source_sha256") != _sha(source)
                or provenance.get("seed") != options.get("seed", 42)
                or provenance.get("teacher_model") != options.get("augmentation_teacher_model", DEFAULT_TEACHER)
                or provenance.get("category") not in categories):
            raise ValueError("Augmentation source lineage/category/teacher binding mismatch")
        prompt_digest = provenance.get("teacher_prompt_sha256")
        if not isinstance(prompt_digest, str) or len(prompt_digest) != 64 or any(c not in "0123456789abcdef" for c in prompt_digest):
            raise ValueError("Augmentation has no teacher-prompt hash binding")
        if prompt_digest != manifest.get("teacher_prompt_sha256"):
            raise ValueError("Augmentation teacher-prompt hash differs from generation manifest")
        acceptance, validation = candidate.get("training_acceptance") or {}, candidate.get("validation") or {}
        if (candidate.get("record_status") != "accepted" or acceptance.get("eligible") is not True
                or acceptance.get("blocking_reasons") != [] or acceptance.get("review_reasons") != []
                or validation.get("schema_valid_strict") is not True or validation.get("standard_a2ui_valid") is not True
                or (candidate.get("gen") or {}).get("error")
                or (candidate.get("gen") or {}).get("completion_complete") is not True
                or candidate.get("source_format") != "a2ui_express_v1"
                or candidate.get("completion") != completion):
            raise ValueError("Augmentation candidate is not fully admitted by Stage 3")
        # Check the producer's identities before replacing them with local row IDs.
        raw_identities = _identity_keys(source_identity_record(candidate))
        raw_identities.update(str(candidate[key]) for key in ("ui_id", "row_id") if candidate.get(key))
        if raw_identities & reserved["identities"]:
            raise ValueError("Augmentation candidate has a reserved heldout source identity")
        hashes = _source_hashes({"response_text": source})
        if hashes & seen_sources:
            raise ValueError("Augmentation source duplicates existing training data or another candidate")
        seen_sources.update(hashes)
        identity = "semantic-augmentation:" + _sha(_json([donor["donor_id"], provenance, completion]))
        if identity in seen_ids:
            raise ValueError("Augmentation row identity collision")
        seen_ids.add(identity)
        used_donors.add(donor["donor_id"])
        rows.append({"id": identity, "row_id": identity, "source_id": donor["source_group_id"],
                     "query_id": identity + ":query", "response_id": identity + ":response", "response_text": source,
                     "messages": [{"role": "user", "content": TASK_PREFIX + source},
                                  {"role": "assistant", "content": completion}],
                     "prompt": "", "completion": completion, "target_format": "a2ui_express_v1",
                     "metadata": {"augmentation": {**deepcopy(provenance), "kind": VERSION,
                                                   "stage3_ui_id": candidate.get("ui_id"),
                                                   "stage3_record_sha256": _sha(_json(candidate))}}})
    return rows, manifest


def augment_training_at_startup(plan: dict, *, tokenizer_loader=None, command_runner=None) -> dict:
    """Run once before student loading, then publish a new frozen preparation.

    Failure leaves audits in semantic_augmentation/ but never publishes an
    incomplete augmented/ directory. A retry uses a fresh run output directory.
    """
    options = plan["options"]
    if options.get("augmentation") != "semantic":
        raise ValueError("Semantic startup helper requires augmentation=semantic")
    validate_semantic_options(options)
    output = Path(options["output_dir"]).resolve()
    source, destination, work = output / "prepared", output / "augmented", output / "semantic_augmentation"
    if destination.exists() or work.exists():
        raise FileExistsError("Augmentation requires a fresh work and output directory")
    if source.is_symlink():
        raise ValueError("Prepared augmentation source must not be a symlink")
    manifest = checked_preparation_manifest(source)
    if manifest.get("augmentation"):
        raise ValueError("Do not recursively augment a prepared augmentation")
    if manifest.get("shared_prompt") != plan.get("shared_prompt") or not manifest.get("shared_prompt"):
        raise ValueError("Semantic augmentation requires the frozen shared production prompt")
    files = sorted(source.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("Prepared augmentation source must contain regular files only")
    hashes = {path.name: file_sha256(path) for path in files}
    splits = manifest.get("splits") or {}
    if not {"train", "val"} <= set(splits):
        raise ValueError("Semantic augmentation requires separate train/val splits")
    for name, entry in splits.items():
        if hashes.get(f"{name}.jsonl") != entry.get("output_sha256"):
            raise ValueError(f"Prepared {name} hash differs from manifest")
    tokenizer_contract = manifest.get("tokenizer") or {}
    if (not tokenizer_contract.get("vocabulary_sha256")
            or tokenizer_contract.get("max_seq_length") != options["max_seq_length"]
            or tokenizer_contract.get("max_input_tokens") != options["max_input_tokens"]):
        raise ValueError("Prepared tokenizer/length contract differs from startup options")
    # Includes omitted/replaced Golden source identities embedded in manifests.
    reserved = load_reserved_cohorts([source / f"{name}.jsonl" for name in splits if name != "train"])
    donors, original_ids, source_hashes, original_tokens, families = _select_donors(source, manifest, reserved, options)
    work.mkdir(parents=True)
    donor_path, generated = work / "donors.jsonl", work / "generated"
    _write_rows(donor_path, donors)
    command = [str(options.get("augmentation_python") or sys.executable),
               str(repo_root() / "dataset/scripts/generate_training_augmentations.py"),
               "--donors", str(donor_path), "--output-dir", str(generated),
               "--teacher-model", options.get("augmentation_teacher_model", DEFAULT_TEACHER),
               "--max-new-samples", str(len(donors)), "--seed", str(options.get("seed", 42))]
    log(f"Semantic augmentation: at most {len(donors)} train-only Muse variants before model loading")
    (command_runner or run_bounded_command)(command, log=work / "generation.log", environment=dict(os.environ),
                                            timeout_seconds=options.get("augmentation_timeout_seconds", 7200),
                                            progress_seconds=options.get("progress_seconds", 10), emit_heartbeat=False)
    candidates, generation = _candidate_rows(generated, donors, original_ids, source_hashes, options, reserved=reserved)
    if generation.get("donors_sha256") != file_sha256(donor_path):
        raise ValueError("Generated augmentation donor-file hash differs from this run")
    accepted, quarantine, filtering = audit_and_filter_rows(candidates, reserved=reserved, require_source_identities=True)
    _write_rows(work / "filtered_candidates.jsonl", accepted)
    _write_rows(work / "candidate_quarantine.jsonl", quarantine)
    _write(work / "filtering.json", filtering)
    if not accepted:
        raise ValueError("No semantic candidates survived reserved-source/strict validation")
    if tokenizer_loader is None:
        from ir_training.pipeline.golden_training import _load_tokenizer
        tokenizer_loader = _load_tokenizer
    tokenizer = tokenizer_loader(Path(options["model_dir"]), options.get("profile", "e2b"))
    candidate_prepared = work / "candidate_prepared"
    prepared = prepare_splits({"train": work / "filtered_candidates.jsonl"}, candidate_prepared,
                              tokenizer=tokenizer, ordering="root-first", shared_prompt=plan["shared_prompt"],
                              max_seq_length=options["max_seq_length"], max_input_tokens=options["max_input_tokens"],
                              chat_template_kwargs=tokenizer_contract.get("chat_template_kwargs") or {},
                              workers=1)
    for key in ("vocabulary_sha256", "chat_template_sha256", "chat_template_kwargs", "max_seq_length", "max_input_tokens"):
        if prepared["tokenizer"].get(key) != tokenizer_contract.get(key):
            raise ValueError(f"Augmentation tokenizer differs from prepared base: {key}")
    token_budget = math.floor(original_tokens * options.get("augmentation_max_extra_fraction", .10))
    added, added_tokens, budget_rejected = [], 0, []
    for _, _, row in _rows(candidate_prepared / "train.jsonl"):
        tokens = row["metadata"]["express_preparation"]["tokenization"]["sequence_tokens"]
        if added_tokens + tokens > token_budget:
            budget_rejected.append({"id": row["id"], "reason": "augmentation_token_budget_exceeded", "tokens": tokens})
        else:
            added.append(row)
            added_tokens += tokens
    _write_rows(work / "token_budget_quarantine.jsonl", budget_rejected)
    if not added:
        raise ValueError("No semantic candidates fit the augmentation token budget; training was not started")
    report = {"version": VERSION, "kind": "stage3_semantic_generation", "seed": options.get("seed", 42),
              "source_manifest_sha256": hashes["manifest.json"], "generation_manifest_sha256": file_sha256(generated / "manifest.json"),
              "donors_sha256": file_sha256(donor_path), "source_families": families, "selected_families": len(donors),
              "attempted_rows": generation["attempted_rows"], "stage3_accepted_rows": generation["accepted_rows"],
              "original_rows": splits["train"]["accepted_rows"], "added_rows": len(added),
              "original_tokens": original_tokens, "added_tokens": added_tokens, "extra_token_budget": token_budget,
              "actual_extra_token_fraction": added_tokens / original_tokens,
              "category_counts": dict(Counter(row["metadata"]["augmentation"]["category"] for row in added)),
              "candidate_filtering": filtering, "candidate_tokenization": prepared["splits"]["train"],
              "token_budget_rejected_rows": len(budget_rejected),
              "untouched_split_sha256": {name: hashes[f"{name}.jsonl"] for name in splits if name != "train"},
              "teacher_model": options.get("augmentation_teacher_model", DEFAULT_TEACHER),
              "recipe": {key: options.get(key) for key in options if key.startswith("augmentation")},
              "quality_claim": "Synthetic candidates passed automatic checks; real accuracy and rendered quality remain unmeasured.",
              "comparison_policy": "Extra training tokens change an epoch budget; compare matched token/step budgets, not equal epochs."}
    bundle = work / "bundle"
    bundle.mkdir()
    for path in files:
        if path.name not in {"train.jsonl", "manifest.json"}:
            shutil.copyfile(path, bundle / path.name)
    with (bundle / "train.jsonl").open("wb") as stream:
        for _, raw, _ in _rows(source / "train.jsonl"):
            stream.write(raw if raw.endswith(b"\n") else raw + b"\n")
        for row in added:
            stream.write((_json(row) + "\n").encode("utf-8"))
    report["output_train_sha256"] = file_sha256(bundle / "train.jsonl")
    _write(bundle / "augmentation.json", report)
    merged = deepcopy(manifest)
    merged["augmentation"] = report
    merged["augmentation_sha256"] = file_sha256(bundle / "augmentation.json")
    merged.setdefault("implementation_sha256", {})["training/src/ir_training/data/semantic_augmentation.py"] = file_sha256(Path(__file__))
    train = merged["splits"]["train"]
    train["unaugmented_preparation"] = deepcopy(splits["train"])
    train.update(input_rows=splits["train"]["accepted_rows"] + len(added),
                 accepted_rows=splits["train"]["accepted_rows"] + len(added), quarantined_rows=0, quarantine_reasons={},
                 source_path=str(source / "train.jsonl"), source_sha256=hashes["train.jsonl"],
                 output_sha256=report["output_train_sha256"])
    membership = hashlib.sha256()
    for index, (_, _, row) in enumerate(_rows(bundle / "train.jsonl"), 1):
        membership.update(f"{index}:{_sha(_json(row))}\n".encode())
    train["source_rows_sha256"] = train["accepted_source_rows_sha256"] = membership.hexdigest()
    from ir_training.data.express_preparation import _api
    _, _, _, references = _api()
    for row in added:
        elements = row["canonical_graph"]["elements"].values()
        for element in elements:
            name = element["type"]
            train["component_counts"][name] = train["component_counts"].get(name, 0) + 1
            for edge in references(element):
                key = edge.reference_kind
                train["renderer_reference_counts"][key] = train["renderer_reference_counts"].get(key, 0) + 1
        scaffold = row["metadata"]["express_preparation"]["scaffold_sha256"]
        train["scaffold_counts"][scaffold] = train["scaffold_counts"].get(scaffold, 0) + 1
        for key, value in row["metadata"]["express_preparation"]["tokenization"].items():
            if key.endswith("_tokens"):
                train["max_accepted_token_lengths"][key] = max(train["max_accepted_token_lengths"].get(key, 0), value)
    _write(bundle / "manifest.json", merged)
    checked_preparation_manifest(bundle)
    # Recheck source integrity after the potentially long teacher process.
    if any(file_sha256(source / name) != digest for name, digest in hashes.items()):
        raise ValueError("Prepared source changed during semantic augmentation; no output published")
    if any(file_sha256(bundle / path.name) != hashes[path.name] for path in files if path.name not in {"train.jsonl", "manifest.json"}):
        raise ValueError("Semantic augmentation changed an immutable evaluation/contract artifact")
    if destination.exists():
        raise FileExistsError(destination)
    bundle.rename(destination)
    log(f"Semantic augmentation frozen: {len(added)} new train rows, {added_tokens} added tokens; holdouts unchanged")
    return {**report, "artifact_paths": [str(path.resolve()) for root in (work, destination) for path in sorted(root.rglob("*")) if path.is_file()]}
