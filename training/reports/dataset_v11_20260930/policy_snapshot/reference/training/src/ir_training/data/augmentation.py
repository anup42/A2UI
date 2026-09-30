"""Opt-in, train-only rare-component exposure; never synthesize supervision.

This is bounded resampling, not new semantic data. It consumes an already
validated and token-bound preparation, retains every original occurrence, and
only gives selected existing source families additional training exposure.
Validation and Golden files are copied unchanged into a fresh directory.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from ir_training.common.progress import Progress, log
from ir_training.data.audit_filter import _identity_keys, _source_hashes
from ir_training.data.golden_replacement import source_identity_record
from ir_training.eval.prepared_contract import checked_preparation_manifest, file_sha256


AUGMENTATION_VERSION = "rare-component-resampling-v1"
# These common structural elements must not cause a whole corpus to be repeated.
EXCLUDED_COMPONENTS = frozenset({
    "Stack", "Column", "Row", "Box", "Card", "Text", "Table", "Icon",
    "Button", "Divider", "Spacer", "ScrollView", "Container",
})


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_augmentation_options(*, max_extra_fraction: float = 0.10,
                                  max_family_copies: int = 2, rare_frequency: float = 0.03,
                                  min_component_families: int = 5) -> None:
    if isinstance(max_extra_fraction, bool) or not math.isfinite(max_extra_fraction) or not 0 <= max_extra_fraction <= 0.5:
        raise ValueError("augmentation max_extra_fraction must be between 0 and 0.5")
    if isinstance(max_family_copies, bool) or not isinstance(max_family_copies, int) or not 1 <= max_family_copies <= 5:
        raise ValueError("augmentation max_family_copies must be an integer from 1 to 5 (total family exposures)")
    if isinstance(rare_frequency, bool) or not math.isfinite(rare_frequency) or not 0 < rare_frequency <= 0.10:
        raise ValueError("augmentation rare_frequency must be greater than 0 and at most 0.10")
    if isinstance(min_component_families, bool) or not isinstance(min_component_families, int) or min_component_families < 2:
        raise ValueError("augmentation min_component_families must be an integer of at least 2")


def _rows(path: Path, digest=None):
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if digest is not None:
                digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: malformed prepared JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: prepared row must be an object")
            yield line_number, raw, row


def _checked_row(row: dict) -> tuple[str, ...]:
    metadata = row.get("metadata") or {}
    evidence = metadata.get("express_preparation") or {}
    if metadata.get("augmentation"):
        raise ValueError("Do not recursively augment an already resampled training row")
    if evidence.get("completion_sha256") != _sha(str(row.get("completion", ""))) or evidence.get("prompt_sha256") != _sha(str(row.get("prompt", ""))):
        raise ValueError("Augmentation requires unchanged, hash-bound prepared prompts and completions")
    tokens = evidence.get("tokenization") or {}
    if not isinstance(tokens.get("sequence_tokens"), int) or tokens["sequence_tokens"] < 1:
        raise ValueError("Augmentation requires the model-tokenized, fully prepared training split")
    graph = row.get("canonical_graph") or {}
    elements = graph.get("elements") or {}
    if not isinstance(elements, dict) or not elements or any(not isinstance(item, dict) or not isinstance(item.get("type"), str) for item in elements.values()):
        raise ValueError("Augmentation requires the checked canonical graph from preparation")
    return tuple(item["type"] for item in elements.values())


def augment_prepared_training(
    source_dir: Path, output_dir: Path, *, seed: int = 42,
    max_extra_fraction: float = 0.10, max_family_copies: int = 2,
    rare_frequency: float = 0.03, min_component_families: int = 5,
    progress_seconds: float = 10,
) -> dict[str, Any]:
    """Create a fresh preparation containing bounded extra train occurrences.

    ``max_family_copies`` is an absolute cap including original occurrences.
    Larger pre-existing multi-target families are retained but never repeated.
    Family membership is transitive over source/query/response IDs and raw or
    URL-normalized source text. Rarity is measured only on training families;
    neither validation nor Golden component frequencies influence selection.
    """
    validate_augmentation_options(max_extra_fraction=max_extra_fraction, max_family_copies=max_family_copies,
                                  rare_frequency=rare_frequency, min_component_families=min_component_families)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("augmentation seed must be an integer")
    if not math.isfinite(progress_seconds) or progress_seconds <= 0:
        raise ValueError("augmentation progress_seconds must be positive")
    source, destination = Path(source_dir).resolve(), Path(output_dir).resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Augmentation output must be separate from the original prepared directory")
    if destination.exists():
        raise FileExistsError(f"Augmentation output already exists: {destination}")
    manifest = checked_preparation_manifest(source)
    if manifest.get("augmentation"):
        raise ValueError("Do not recursively augment an already augmented preparation")
    if not (manifest.get("tokenizer") or {}).get("vocabulary_sha256"):
        raise ValueError("Augmentation requires the model-tokenized preparation manifest")
    source_manifest_sha256 = file_sha256(source / "manifest.json")
    files = sorted(source.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in files):
        raise ValueError("Prepared augmentation input must contain regular files only, without symlinks or directories")
    splits = manifest.get("splits") or {}
    if not {"train", "val"} <= set(splits):
        raise ValueError("Augmentation requires separate checked train and val splits")
    for name in splits:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Unsafe split name in preparation manifest")
    # The parent workflow also verifies benchmark reservation. This module can
    # only copy training rows and never draws donors from any evaluation split.
    with Progress("Augmentation verify prepared files", unit="stage", interval=progress_seconds):
        original_hashes = {path.name: file_sha256(path) for path in files}
        for name, entry in splits.items():
            if original_hashes.get(f"{name}.jsonl") != entry.get("output_sha256"):
                raise ValueError(f"Prepared {name} hash differs from its manifest")

    parent: list[int] = []
    owners: dict[str, int] = {}
    occurrence_ids: set[str] = set()
    component_types: list[tuple[str, ...]] = []
    row_keys: list[str] = []
    before_presence: Counter[str] = Counter()

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    with Progress("Augmentation training-family census", total=splits["train"]["accepted_rows"], interval=progress_seconds) as progress:
        for _, _, row in _rows(source / "train.jsonl"):
            index = len(parent)
            parent.append(index)
            types = _checked_row(row)
            identity = source_identity_record(row)
            if not identity["source_id"] or not identity["row_id"] or identity["row_id"] in occurrence_ids:
                raise ValueError("Prepared augmentation input requires source IDs and unique occurrence IDs")
            occurrence_ids.add(identity["row_id"])
            keys = {"id:" + item for item in _identity_keys(identity)} | {"text:" + item for item in _source_hashes(row)}
            for key in sorted(keys):
                if key in owners:
                    parent[find(index)] = find(owners[key])
                else:
                    owners[key] = index
            component_types.append(types)
            row_keys.append(identity["row_id"])
            before_presence.update(set(types))
            progress.advance()
    if len(parent) != splits["train"]["accepted_rows"] or not parent:
        raise ValueError("Prepared training row count differs from its manifest")
    families: dict[int, list[int]] = defaultdict(list)
    for index in range(len(parent)):
        families[find(index)].append(index)
    family_components = {key: set().union(*(set(component_types[index]) for index in members)) for key, members in families.items()}
    family_counts: Counter[str] = Counter(component for values in family_components.values() for component in values)
    eligible_components = {component for component, count in family_counts.items()
                           if component not in EXCLUDED_COMPONENTS and min_component_families <= count <= rare_frequency * len(families)}
    # Keep only small scalar metadata in RAM, not multi-gigabyte training rows.
    ranked = []
    for family, members in families.items():
        useful = family_components[family] & eligible_components
        if useful and len(members) < max_family_copies:
            stable_family = min(row_keys[index] for index in members)
            ranked.append((min(family_counts[key] for key in useful), _sha(f"{seed}:{stable_family}"), family))
    ranked.sort()
    extra_limit = math.floor(len(parent) * max_extra_fraction)
    repeats: Counter[int] = Counter()
    selected_families: set[int] = set()
    selected_count = 0
    # Round-robin caps prevent a rare singleton from taking several repeats
    # before other eligible families have had one additional exposure.
    for round_index in range(1, max_family_copies):
        for _, _, family in ranked:
            members = families[family]
            if len(members) + round_index > max_family_copies or selected_count >= extra_limit:
                continue
            donors = sorted((index for index in members if set(component_types[index]) & eligible_components),
                            key=lambda index: _sha(f"{seed}:{row_keys[index]}"))
            donor = donors[(round_index - 1) % len(donors)]
            repeats[donor] += 1
            selected_count += 1
            selected_families.add(family)
    added_rows = sum(repeats.values())
    log(f"Augmentation: {len(parent):,} original rows; {len(families):,} source families; "
        f"{len(eligible_components)} eligible rare components; adding {added_rows:,} exact training copies "
        f"from {len(selected_families):,} families (budget {extra_limit:,}); validation/Goldens unchanged")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.augmenting-", dir=destination.parent))
    try:
        with Progress("Augmentation copy unchanged evaluation and contracts", unit="stage", interval=progress_seconds):
            for path in files:
                if path.name not in {"train.jsonl", "manifest.json"}:
                    shutil.copyfile(path, temporary / path.name)
                    if file_sha256(temporary / path.name) != original_hashes[path.name]:
                        raise ValueError(f"Prepared source changed while copying {path.name}")
        after_presence = before_presence.copy()
        added_types: Counter[str] = Counter()
        added_references: Counter[str] = Counter()
        membership_hash = hashlib.sha256()
        output_hash = hashlib.sha256()
        original_train_hash = hashlib.sha256()
        output_index = 0
        selections = []
        if added_rows:
            from ir_training.data.express_preparation import _api
            _, _, _, iter_renderer_references = _api()
        with Progress("Augmentation write training occurrences", total=len(parent) + added_rows, interval=progress_seconds) as progress, (temporary / "train.jsonl").open("wb") as output:
            for index, (_, raw, row) in enumerate(_rows(source / "train.jsonl", original_train_hash)):
                # Preserve every original record byte-for-byte; append newline
                # only when the original final row did not have one.
                original = raw if raw.endswith(b"\n") else raw + b"\n"
                output.write(original)
                output_hash.update(original)
                output_index += 1
                membership_hash.update(f"{output_index}:{_sha(_json(row))}\n".encode("utf-8"))
                progress.advance()
                for copy_index in range(1, repeats[index] + 1):
                    augmented = deepcopy(row)
                    new_id = "augmentation:" + _sha(f"{AUGMENTATION_VERSION}:{seed}:{row_keys[index]}:{copy_index}")
                    if new_id in occurrence_ids:
                        raise ValueError("Augmentation occurrence ID collides with an existing row")
                    occurrence_ids.add(new_id)
                    augmented["id"] = new_id
                    if "row_id" in augmented:
                        augmented["row_id"] = new_id
                    augmented["metadata"]["augmentation"] = {
                        "version": AUGMENTATION_VERSION, "kind": "exact_training_resample",
                        "original_occurrence_id": row_keys[index], "copy_index": copy_index,
                        "seed": seed, "content_changed": False, "split": "train",
                    }
                    encoded = (_json(augmented) + "\n").encode("utf-8")
                    output.write(encoded)
                    output_hash.update(encoded)
                    output_index += 1
                    membership_hash.update(f"{output_index}:{_sha(_json(augmented))}\n".encode("utf-8"))
                    after_presence.update(set(component_types[index]))
                    added_types.update(component_types[index])
                    added_references.update(edge.reference_kind for element in row["canonical_graph"]["elements"].values() for edge in iter_renderer_references(element))
                    selections.append({"original_occurrence_id": row_keys[index], "occurrence_id": new_id,
                                       "eligible_components": sorted(set(component_types[index]) & eligible_components)})
                    progress.advance()
        if original_train_hash.hexdigest() != original_hashes["train.jsonl"] or file_sha256(source / "manifest.json") != source_manifest_sha256:
            raise ValueError("Prepared source changed during augmentation; no output published")
        # Recheck even copied contracts against the current input: never publish
        # a result assembled from multiple revisions of its original directory.
        with Progress("Augmentation final source integrity", unit="stage", interval=progress_seconds):
            if any(file_sha256(source / name) != digest for name, digest in original_hashes.items() if name != "train.jsonl"):
                raise ValueError("Prepared source changed during augmentation; no output published")
        report = {
            "version": AUGMENTATION_VERSION, "kind": "rare_component_resampling", "seed": seed,
            "source_dir": str(source), "source_manifest_sha256": source_manifest_sha256,
            "source_train_sha256": original_hashes["train.jsonl"], "output_train_sha256": output_hash.hexdigest(),
            "original_rows": len(parent), "added_rows": added_rows, "output_rows": output_index,
            "source_families": len(families), "resampled_families": len(selected_families),
            "max_extra_fraction": max_extra_fraction, "extra_row_budget": extra_limit,
            "actual_extra_fraction": added_rows / len(parent), "max_family_copies": max_family_copies,
            "family_cap_policy": "Maximum total occurrences including originals; already larger families retained without repeats",
            "rare_frequency": rare_frequency, "min_component_families": min_component_families,
            "excluded_components": sorted(EXCLUDED_COMPONENTS), "eligible_components": sorted(eligible_components),
            "component_exposure": {name: {"original_rows": before_presence[name], "output_rows": after_presence[name],
                                            "source_families": family_counts[name], "eligible": name in eligible_components}
                                   for name in sorted(before_presence)},
            "untouched_split_sha256": {name: original_hashes[f"{name}.jsonl"] for name in splits if name != "train"},
            "synthetic_targets_created": 0, "source_texts_changed": 0, "independent_examples_added": 0,
            "selection_policy": "Training-family frequency only, smallest eligible frequency first, seeded tie ordering; no evaluation frequencies or scores",
            "quality_claim": "Exposure hypothesis only. Compare against a same-step baseline on validation and Golden32; Golden35 remains held out.",
            "selections": selections,
        }
        _write_json(temporary / "augmentation.json", report)
        augmented_manifest = deepcopy(manifest)
        augmented_manifest["augmentation"] = {key: value for key, value in report.items() if key != "selections"}
        augmented_manifest["augmentation_sha256"] = file_sha256(temporary / "augmentation.json")
        augmented_manifest.setdefault("implementation_sha256", {})["training/src/ir_training/data/augmentation.py"] = file_sha256(Path(__file__))
        augmented_manifest["ab_comparison"] = "Resampling changes training occurrences, not independent examples; compare matched optimizer-step budgets."
        train = augmented_manifest["splits"]["train"]
        train["unaugmented_preparation"] = deepcopy(manifest["splits"]["train"])
        train.update(input_rows=output_index, accepted_rows=output_index, quarantined_rows=0, quarantine_reasons={},
                     source_path=str(source / "train.jsonl"), source_sha256=original_hashes["train.jsonl"],
                     output_sha256=output_hash.hexdigest(), source_rows_sha256=membership_hash.hexdigest(),
                     accepted_source_rows_sha256=membership_hash.hexdigest())
        train["component_counts"] = dict(Counter(train.get("component_counts") or {}) + added_types)
        train["renderer_reference_counts"] = dict(Counter(train.get("renderer_reference_counts") or {}) + added_references)
        for scaffold in train.get("scaffold_counts", {}):
            train["scaffold_counts"][scaffold] += added_rows
        _write_json(temporary / "manifest.json", augmented_manifest)
        checked_preparation_manifest(temporary)
        if destination.exists():
            raise FileExistsError(f"Augmentation output appeared while preparing: {destination}")
        temporary.rename(destination)
        return report
    finally:
        if temporary.exists():
            if temporary.is_symlink() or temporary.resolve().parent != destination.parent or not temporary.name.startswith(f".{destination.name}.augmenting-"):
                raise ValueError("Refusing unsafe augmentation temporary-directory cleanup")
            shutil.rmtree(temporary)
