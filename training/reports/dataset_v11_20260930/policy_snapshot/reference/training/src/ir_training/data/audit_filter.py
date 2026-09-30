"""Strict, source-preserving training filters with explicit reserved cohorts."""
from __future__ import annotations

from collections import Counter
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ir_training.common.parallel import ordered_bounded_map

from ir_training.data.express_preparation import PreparationError, _token_lengths, prepare_row
from ir_training.data.golden_replacement import read_rows_strict, response_text, source_identity_record
from ir_training.data.url_preprocess import preprocess_training_urls
from ir_training.train.prepared_binding import value_sha256


def _identity_keys(record: Mapping[str, Any]) -> set[str]:
    return {str(record.get(key)) for key in ("source_id", "query_id", "response_id", "row_id") if record.get(key)}


def _source_hashes(row: Mapping[str, Any]) -> set[str]:
    raw = response_text(row)
    masked = preprocess_training_urls(raw, {}, enabled=True).response_text
    return {hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest() for text in (raw, masked)}


def load_reserved_cohorts(paths: Iterable[Path], manifests: Iterable[Path] = ()) -> dict[str, Any]:
    """Reserve valid and failed original identities, including replaced cases."""
    identities: set[str] = set()
    responses: set[str] = set()
    evidence = []
    embedded_contracts = []
    manifest_paths = {Path(path).resolve() for path in manifests}
    for value in paths:
        path = Path(value)
        rows = read_rows_strict(path)
        if not rows:
            raise ValueError(f"Reserved cohort is empty: {path}")
        for row in rows:
            record = source_identity_record(row)
            identities.update(_identity_keys(record))
            responses.update(_source_hashes(row))
        evidence.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows)})
        adjacent = path.parent / "benchmark_manifest.json"
        if adjacent.is_file():
            manifest_paths.add(adjacent.resolve())
        elif (path.parent / "manifest.json").is_file():
            from ir_training.eval.golden_set import benchmark_contract_for_split

            contract = benchmark_contract_for_split(path, rows)
            if contract:
                embedded_contracts.append((path.parent / "manifest.json", contract))
    contracts = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(manifest_paths)]
    for path, manifest in [*contracts, *embedded_contracts]:
        for record in manifest.get("excluded_sources", []):
            identities.update(_identity_keys(record))
            digest = record.get("response_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"Excluded source has no normalized response hash: {path}")
            responses.add(digest)
            for variant in record.get("response_sha256s", []):
                if not isinstance(variant, str) or len(variant) != 64 or any(char not in "0123456789abcdef" for char in variant):
                    raise ValueError(f"Excluded source has an invalid normalized response hash: {path}")
                responses.add(variant)
        evidence.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "excluded_sources": len(manifest.get("excluded_sources", []))})
    return {"identities": identities, "responses": responses, "evidence": evidence}


def _filter_candidate(row, *, reserved, require_source_identities):
    """Validate an independent row; global deduplication stays in the parent."""
    identity, missing_identity = None, False
    try:
        try:
            identity = source_identity_record(row)
        except ValueError as exc:
            raise PreparationError("missing_source_response", str(exc)) from exc
        missing_identity = not identity["source_id"]
        if _identity_keys(identity) & reserved["identities"] or _source_hashes(row) & reserved["responses"]:
            raise PreparationError("reserved_evaluation_source", "Source occurs in a reserved evaluation cohort or its replaced-source exclusions")
        if require_source_identities and missing_identity:
            raise PreparationError("missing_source_identity", "Recover original Stage 2/3 identity before training")
        _, target, _ = prepare_row(row, "root-first")
        return row, identity, missing_identity, target.semantic_sha256, target.component_types, None
    except PreparationError as exc:
        return row, identity, missing_identity, None, (), (exc.reason, str(exc))


def audit_and_filter_rows(
    rows: Iterable[dict[str, Any]], *, reserved: Mapping[str, Any] | None = None,
    tokenizer: Any | None = None, max_seq_length: int | None = None,
    max_input_tokens: int | None = None, chat_template_kwargs: Mapping[str, Any] | None = None,
    require_source_identities: bool = False,
    workers: int = 1, progress: Any = None,
    accepted_sink: Callable | None = None, quarantine_sink: Callable | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep original accepted rows; quarantine failures without target synthesis.

    Same-source/different-target supervision is reported, not silently removed;
    exact source/semantic-target duplicates are removed. All reserved sources
    are checked before strict validation, so an invalid reference is reserved.
    """
    for limit in (max_seq_length, max_input_tokens):
        if limit is not None and (tokenizer is None or not isinstance(limit, int) or limit <= 0):
            raise ValueError("Token limits require an explicit tokenizer and positive integers")
    template_kwargs = dict(chat_template_kwargs or {})
    if {"tokenize", "add_generation_prompt"} & set(template_kwargs):
        raise ValueError("chat_template_kwargs cannot override tokenize/add_generation_prompt")
    reserved = reserved or {"identities": set(), "responses": set(), "evidence": []}
    accepted, quarantine = [], []
    reasons, source_intents, accepted_intents, components = Counter(), Counter(), Counter(), Counter()
    seen_pairs = set()
    source_targets: dict[str, set[str]] = {}
    missing_identity_rows = 0
    maxima: dict[str, int] = {}
    accepted_count, quarantine_count = 0, 0
    candidates = ordered_bounded_map(
        partial(_filter_candidate, reserved=reserved, require_source_identities=require_source_identities),
        rows, workers=workers,
    )
    for index, (row, identity, missing_identity, semantic_sha256, component_types, error) in enumerate(candidates, 1):
        if progress is not None:
            progress.advance()
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        intent = str(row.get("intent_bucket") or metadata.get("intent_bucket") or "unknown")
        source_intents[intent] += 1
        try:
            missing_identity_rows += int(missing_identity)
            if error is not None:
                raise PreparationError(*error)
            pair = (identity["response_sha256"], semantic_sha256)
            if pair in seen_pairs:
                raise PreparationError("duplicate_source_target", "Identical normalized source and semantic target already accepted")
            if tokenizer is not None:
                lengths = _token_lengths(row, tokenizer, template_kwargs)
                if max_seq_length is not None and lengths["sequence_tokens"] > max_seq_length:
                    raise PreparationError("sequence_too_long", f"{lengths['sequence_tokens']} > {max_seq_length}; complete row quarantined")
                if max_input_tokens is not None and lengths["prompt_tokens"] > max_input_tokens:
                    raise PreparationError("prompt_too_long", f"{lengths['prompt_tokens']} > {max_input_tokens}; complete row quarantined")
                for key, value in lengths.items():
                    if key.endswith("_tokens"):
                        maxima[key] = max(maxima.get(key, 0), value)
            seen_pairs.add(pair)
            source_targets.setdefault(identity["response_sha256"], set()).add(semantic_sha256)
            (accepted_sink or accepted.append)(row)
            accepted_count += 1
            accepted_intents[intent] += 1
            components.update(component_types)
        except PreparationError as exc:
            reasons[exc.reason] += 1
            (quarantine_sink or quarantine.append)({"source_line": index, "reason": exc.reason, "detail": str(exc), "row": row})
            quarantine_count += 1
    report = {
        "schema_version": 1, "input_rows": accepted_count + quarantine_count, "accepted_rows": accepted_count,
        "quarantined_rows": quarantine_count, "quarantine_reasons": dict(reasons),
        "source_intent_counts": dict(source_intents), "accepted_intent_counts": dict(accepted_intents),
        "accepted_component_counts": dict(components), "missing_source_identity_rows": missing_identity_rows,
        "same_source_multiple_target_count": sum(len(targets) > 1 for targets in source_targets.values()),
        "tokenizer_checked": tokenizer is not None, "max_accepted_token_lengths": maxima,
        "tokenizer": None if tokenizer is None else {
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "vocabulary_sha256": value_sha256(tokenizer.get_vocab()),
            "chat_template_sha256": value_sha256(getattr(tokenizer, "chat_template", None)),
            "chat_template_kwargs": template_kwargs, "max_seq_length": max_seq_length,
            "max_input_tokens": max_input_tokens,
        },
        "reserved_cohorts": reserved.get("evidence", []),
        "retained_source_rows_unchanged": True, "synthetic_targets_created": 0,
        "augmentation_candidates": [{"intent": intent, "original_rows": count, "accepted_rows": accepted_intents[intent],
                                     "recommendation": "Regenerate quarantined Stage 3 targets from authoritative source responses; review before inclusion"}
                                    for intent, count in sorted(source_intents.items()) if accepted_intents[intent] < count],
    }
    return accepted, quarantine, report
