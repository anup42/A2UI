from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ir_training.common.config import repo_root, resolve_path, training_root
from ir_training.common.git import current_commit
from ir_training.common.jsonl import load_by_key, read_jsonl, write_jsonl
from ir_training.data.chat_templates import build_messages, build_prompt
from ir_training.data.filters import ExpressValidator, row_passes_express_filters
from ir_training.data.ir_targets import (
    A2UI_EXPRESS_V1,
    canonical_graph_from_source,
    materialize_completion_targets,
    resolve_target_formats,
    semantic_hash,
    serialize_completion,
)
from ir_training.data.legacy_targets import FLAT_SPEC_V1, canonical_graph_from_legacy_source
from ir_training.data.splits import stratified_split
from ir_training.data.url_preprocess import preprocess_training_urls


def prepare_dataset(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    run_cfg = config.get("run") if isinstance(config.get("run"), dict) else {}
    filter_cfg = config.get("filters") if isinstance(config.get("filters"), dict) else {}
    split_cfg = config.get("split") if isinstance(config.get("split"), dict) else {}
    url_cfg = config.get("url_preprocessing") if isinstance(config.get("url_preprocessing"), dict) else {}

    base = training_root()
    output_dir = resolve_path(run_cfg.get("output_dir", "outputs/datasets/dataset_v1_stage3"), base)
    genui_paths = _collect_stage3_genui_paths(run_cfg, base)
    response_lookups = {path: _load_response_lookup(path) for path in genui_paths}
    # Legacy graph validation is performed only while importing a legacy row;
    # the active target is validated as native Express text after encoding.
    validator = ExpressValidator()
    max_input_chars = int(filter_cfg.get("max_input_chars", 60000))
    max_output_chars = int(filter_cfg.get("max_output_chars", 60000))
    deduplicate = bool(filter_cfg.get("deduplicate", True))
    system_prompt = str(run_cfg.get("system_prompt") or "").strip()
    url_preprocessing_enabled = bool(url_cfg.get("enabled", True))

    accepted: list[dict[str, Any]] = []
    # Keep source-level records separate from serialized completion targets.
    # Split assignment is deliberately performed on this list before an
    # Express target is materialized, so target-format suffixes can never
    # influence train/validation/test membership.
    prepared_sources: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for genui_path in genui_paths:
        source_key = _stable_source_key(genui_path, repo_root())
        responses_by_id = response_lookups[genui_path]
        genui_rows = sorted(
            enumerate(read_jsonl(genui_path), start=1),
            key=lambda pair: (
                str(pair[1].get("response_id") or ""),
                str(pair[1].get("ui_id") or ""),
                pair[0],
            ),
        )
        for row_index, genui in genui_rows:
            response_id = str(genui.get("response_id") or f"{genui_path.stem}:{row_index}")
            response = responses_by_id.get(response_id, {})
            if genui.get("record_status") == "format_rejected":
                rejected.append(
                    {
                        "response_id": response_id,
                        "source_path": source_key,
                        "reason": "format_rejected",
                        "source_format": genui.get("source_format"),
                    }
                )
                continue
            source_format = genui.get("source_format")
            native_payload = _source_payload(genui)
            if source_format is None:
                source_format = (
                    A2UI_EXPRESS_V1
                    if isinstance(native_payload, str) and native_payload.strip().startswith("<a2ui>")
                    else FLAT_SPEC_V1
                )
            response_text = str(genui.get("response_text") or response.get("response_text") or "")
            if not response_text.strip():
                rejected.append(
                    {
                        "response_id": response_id,
                        "source_path": source_key,
                        "reason": "missing_response_text",
                    }
                )
                continue
            try:
                if str(source_format).strip().lower() == A2UI_EXPRESS_V1:
                    canonical = canonical_graph_from_source(native_payload, source_format=source_format)
                else:
                    canonical = canonical_graph_from_legacy_source(native_payload, source_format=source_format)
            except (TypeError, ValueError) as exc:
                rejected.append(
                    {
                        "response_id": response_id,
                        "source_path": source_key,
                        "reason": f"ir_decode_error:{exc}",
                        "source_format": genui.get("source_format"),
                    }
                )
                continue
            assets = genui.get("assets") or response.get("assets") or []
            # Process the response, canonical graph, and asset metadata in one
            # registry so no raw URL/local path is exposed to the model and
            # every placeholder has one deterministic restoration entry.
            url_processed = preprocess_training_urls(
                response_text,
                {"graph": canonical, "assets": assets},
                enabled=url_preprocessing_enabled,
            )
            processed_bundle = url_processed.canonical_graph
            processed_graph = (
                processed_bundle.get("graph")
                if isinstance(processed_bundle, dict)
                else canonical
            )
            masked_assets = (
                processed_bundle.get("assets")
                if isinstance(processed_bundle, dict)
                else assets
            )
            selected_target_formats = resolve_target_formats(run_cfg, genui)
            query_id = genui.get("query_id") or response.get("query_id")
            intent = genui.get("intent") or response.get("intent")
            tags = genui.get("tags") or response.get("tags") or []
            intent_bucket = genui.get("intent_bucket") or response.get("intent_bucket") or intent
            expected_ui_contract = (
                genui.get("expected_ui_contract")
                or response.get("expected_ui_contract")
            )
            expected_ui_contract_v5_4 = (
                genui.get("expected_ui_contract_v5_4")
                or response.get("expected_ui_contract_v5_4")
            )
            expected_ui_contract_v5_4_source = (
                genui.get("expected_ui_contract_v5_4_source")
                or response.get("expected_ui_contract_v5_4_source")
            )
            row_id = genui.get("ui_id") or f"u_{response_id}"
            source_id = str(
                genui.get("source_id")
                or response.get("source_id")
                or query_id
                or response_id
            )
            response_generation = _generation_metadata(response.get("gen"))
            ir_generation = _generation_metadata(genui.get("gen"))
            prepared_sources.append(
                {
                    "source_key": source_key,
                    "source_row_index": row_index,
                    "row_id": row_id,
                    "response_id": response_id,
                    "query_id": query_id,
                    "source_id": source_id,
                    "source_format": source_format,
                    "response_text": url_processed.response_text,
                    "processed_graph": processed_graph,
                    "masked_assets": masked_assets,
                    "url_map": url_processed.url_map,
                    "url_metrics": url_processed.metrics,
                    "intent": intent,
                    "intent_bucket": intent_bucket,
                    "tags": tags,
                    "expected_ui_contract": expected_ui_contract,
                    "expected_ui_contract_v5_4": expected_ui_contract_v5_4,
                    "expected_ui_contract_v5_4_source": expected_ui_contract_v5_4_source,
                    "response_generation": response_generation,
                    "ir_generation": ir_generation,
                    "created_at": genui.get("created_at") or response.get("created_at"),
                    "target_formats": selected_target_formats,
                }
            )

    # Assign immutable source groups before encoding or validating any active
    # completion target. This remains true even if a future config adds more
    # derived candidates or target views.
    source_splits = stratified_split(
        prepared_sources,
        train_ratio=float(split_cfg.get("train", 0.96)),
        val_ratio=float(split_cfg.get("val", 0.02)),
        test_ratio=float(split_cfg.get("test", 0.02)),
        stratify_key=str(split_cfg.get("stratify_by", "intent_bucket")),
        seed=int(run_cfg.get("seed", 42)),
    )
    split_by_source_object = {
        id(row): split_name
        for split_name, rows in source_splits.items()
        for row in rows
    }
    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for source in prepared_sources:
        split_name = split_by_source_object.get(id(source), "train")
        try:
            completion_targets = materialize_completion_targets(source["processed_graph"])
            for target_format in source["target_formats"]:
                target_payload = completion_targets[target_format]
                completion = serialize_completion(target_payload, target_format)
                # Validate the serialized target, never the legacy graph, as
                # the production training contract.
                completion_check = row_passes_express_filters(
                    response_text=source["response_text"],
                    completion=completion,
                    validator=validator,
                    max_input_chars=max_input_chars,
                    max_output_chars=max_output_chars,
                )
                if not completion_check.valid:
                    rejected.append(
                        {
                            "response_id": source["response_id"],
                            "source_path": source["source_key"],
                            "reason": completion_check.reason or "express_invalid",
                            "source_format": source["source_format"],
                            "assigned_split": split_name,
                        }
                    )
                    continue

                dedupe_key = hashlib.sha256(
                    (target_format + "\n" + source["response_text"] + "\n" + completion).encode("utf-8")
                ).hexdigest()
                if deduplicate and dedupe_key in seen_hashes:
                    rejected.append(
                        {
                            "response_id": source["response_id"],
                            "source_path": source["source_key"],
                            "reason": "duplicate_pair",
                            "target_format": target_format,
                            "assigned_split": split_name,
                        }
                    )
                    continue
                seen_hashes.add(dedupe_key)
                prompt = build_prompt(
                    system_prompt,
                    source["response_text"],
                    target_format=target_format,
                )
                accepted_row = {
                    "id": f"{source['source_key']}:{source['row_id']}:{target_format}",
                    "response_id": source["response_id"],
                    "messages": build_messages(
                        system_prompt,
                        source["response_text"],
                        target_payload,
                        target_format=target_format,
                    ),
                    "prompt": prompt,
                    "completion": completion,
                    "a2ui_express": completion,
                    "completion_targets": completion_targets,
                    "canonical_graph": source["processed_graph"],
                    "target_format": target_format,
                    "source_format": source["source_format"],
                    "semantic_hash": semantic_hash(source["processed_graph"]),
                    "source_id": source["source_id"],
                    "response_text": source["response_text"],
                    "intent_bucket": source["intent_bucket"],
                    "assets": source["masked_assets"],
                    "expected_ui_contract": source["expected_ui_contract"],
                    "expected_ui_contract_v5_4": source["expected_ui_contract_v5_4"],
                    "expected_ui_contract_v5_4_source": source["expected_ui_contract_v5_4_source"],
                    "source_model_family": str(
                        source["ir_generation"].get("model")
                        or source["response_generation"].get("model")
                        or "unknown"
                    ),
                    "source_created_at": source["created_at"],
                    "metadata": {
                        "query_id": source["query_id"],
                        "ui_id": source["row_id"],
                        "intent": source["intent"],
                        "intent_bucket": source["intent_bucket"],
                        "tags": source["tags"],
                        "prompt_version": run_cfg.get("prompt_version"),
                        "schema_path": run_cfg.get("schema_path"),
                        "source_path": source["source_key"],
                        "source_row_index": source["source_row_index"],
                        "source_id": source["source_id"],
                        "assigned_split": split_name,
                        "target_format": target_format,
                        "source_format": source["source_format"],
                        "input_chars": len(source["response_text"]),
                        "output_chars": len(completion),
                        "response_generation": source["response_generation"],
                        "ir_generation": source["ir_generation"],
                        "source_generation": source["ir_generation"],
                        "url_preprocessing": {
                            "enabled": url_preprocessing_enabled,
                            "url_map": source["url_map"],
                            "metrics": source["url_metrics"],
                        },
                    },
                }
                accepted.append(accepted_row)
                split_rows[split_name].append(accepted_row)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "response_id": source["response_id"],
                    "source_path": source["source_key"],
                    "reason": f"express_materialization_error:{exc}",
                    "source_format": source["source_format"],
                    "assigned_split": split_name,
                }
            )

    required_accepted_rows = filter_cfg.get("required_accepted_rows")
    if required_accepted_rows is not None:
        required_count = int(required_accepted_rows)
        if required_count < 1:
            raise ValueError("filters.required_accepted_rows must be at least 1")
        require_exact_accepted_rows = bool(filter_cfg.get("require_exact_accepted_rows", False))
        count_invalid = (
            len(accepted) != required_count
            if require_exact_accepted_rows
            else len(accepted) < required_count
        )
        if count_invalid:
            comparator = "exactly" if require_exact_accepted_rows else "at least"
            raise ValueError(
                f"Prepared dataset requires {comparator} {required_count} accepted rows, "
                f"but materialization produced {len(accepted)}."
            )
    if bool(filter_cfg.get("require_unique_source_ids", False)):
        source_ids = [str(row.get("source_id") or "").strip() for row in accepted]
        if any(not value for value in source_ids) or len(set(source_ids)) != len(source_ids):
            raise ValueError("Prepared dataset requires one unique source_id per accepted row.")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_count = write_jsonl(output_dir / "all.jsonl", accepted)
    counts = {name: write_jsonl(output_dir / f"{name}.jsonl", rows) for name, rows in split_rows.items()}
    rejected_count = write_jsonl(output_dir / "rejected.jsonl", rejected)
    manifest = {
        "run_id": run_cfg.get("id", output_dir.name),
        "source_run_dir": str(run_cfg.get("source_run_dir", "")),
        "source_genui_dir": str(run_cfg.get("source_genui_dir", "")),
        "source_genui_paths": [str(path) for path in genui_paths],
        "output_dir": str(output_dir),
        "prompt_version": run_cfg.get("prompt_version"),
        "schema_path": run_cfg.get("schema_path"),
        "git_commit": current_commit(repo_root()),
        "counts": {**counts, "all": all_count, "accepted": len(accepted), "rejected": rejected_count},
        "model_counts": _model_counts_by_generation(accepted),
        "filters": filter_cfg,
        "split": split_cfg,
        "split_assignment_stage": "source_group_before_target_materialization",
        "source_group_count": len({str(row["source_id"]) for row in prepared_sources}),
        "url_preprocessing": {"enabled": url_preprocessing_enabled},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if config_path is not None:
        manifest["config_path"] = str(config_path)
    return manifest


def _collect_stage3_genui_paths(run_cfg: dict[str, Any], base: Path) -> list[Path]:
    explicit_files = run_cfg.get("source_genui_files")
    if isinstance(explicit_files, list) and explicit_files:
        paths = sorted({resolve_path(str(path), base) for path in explicit_files})
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing source_genui_files: {', '.join(str(path) for path in missing)}")
        return paths

    source_genui_dir = run_cfg.get("source_genui_dir")
    if source_genui_dir:
        source_dir = resolve_path(str(source_genui_dir), base)
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing source_genui_dir: {source_dir}")
        glob_pattern = str(run_cfg.get("source_glob") or "**/genui.jsonl")
        paths = sorted(path for path in source_dir.glob(glob_pattern) if path.is_file())
        if not paths and glob_pattern != "*.jsonl":
            paths = sorted(path for path in source_dir.glob("*.jsonl") if path.is_file())
        if not paths:
            raise FileNotFoundError(f"No Stage 3 JSONL files found under: {source_dir}")
        return paths

    source_run_dir = resolve_path(run_cfg.get("source_run_dir", "../dataset/data/runs/dataset_v1"), base)
    genui_path = source_run_dir / "genui.jsonl"
    if not genui_path.exists():
        raise FileNotFoundError(f"Missing genui.jsonl: {genui_path}")
    return [genui_path]


def _load_response_lookup(genui_path: Path) -> dict[str, dict[str, Any]]:
    responses_path = genui_path.parent / "responses.jsonl"
    if not responses_path.exists():
        return {}
    return load_by_key(responses_path, "response_id")


def _source_payload(genui: Mapping[str, Any]) -> Any:
    """Select the source completion without making a legacy graph active.

    Express records carry their raw model text.  Historical records carry a
    FlatSpec graph under ``genui_json``/``a2ui_json`` and cross the explicit
    migration boundary in ``legacy_targets.canonical_graph_from_legacy_source``.
    """
    for key in ("a2ui_express", "model_completion_raw", "completion"):
        value = genui.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("genui_json", "a2ui_json", "canonical_graph"):
        value = genui.get(key)
        if value is not None:
            return value
    return None


def _stable_source_key(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _generation_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("provider", "model", "prompt_version"):
        raw = value.get(key)
        if raw is not None:
            out[key] = raw
    return out


def _model_counts_by_generation(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        "response_generation": {},
        "ir_generation": {},
        "response_to_ir_generation": {},
    }
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        response_label = _generation_label(metadata.get("response_generation"))
        ir_label = _generation_label(metadata.get("ir_generation") or metadata.get("source_generation"))
        counts["response_generation"][response_label] = counts["response_generation"].get(response_label, 0) + 1
        counts["ir_generation"][ir_label] = counts["ir_generation"].get(ir_label, 0) + 1
        pair_label = f"{response_label} -> {ir_label}"
        counts["response_to_ir_generation"][pair_label] = counts["response_to_ir_generation"].get(pair_label, 0) + 1
    return {section: dict(sorted(section_counts.items())) for section, section_counts in counts.items()}


def _generation_label(value: Any) -> str:
    if not isinstance(value, dict):
        return "unknown/unknown"
    provider = str(value.get("provider") or "unknown").strip() or "unknown"
    model = str(value.get("model") or "unknown").strip() or "unknown"
    return f"{provider}/{model}"
