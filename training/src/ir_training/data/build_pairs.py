from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ir_training.common.config import repo_root, resolve_path, training_root
from ir_training.common.git import current_commit
from ir_training.common.jsonl import load_by_key, read_jsonl, write_jsonl
from ir_training.data.chat_templates import build_messages, build_prompt, minify_json
from ir_training.data.filters import FlatSpecValidator, row_passes_basic_filters
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
    validator = FlatSpecValidator(require_strict=bool(filter_cfg.get("require_strict_flat_spec", True)))
    max_input_chars = int(filter_cfg.get("max_input_chars", 60000))
    max_output_chars = int(filter_cfg.get("max_output_chars", 60000))
    deduplicate = bool(filter_cfg.get("deduplicate", True))
    system_prompt = str(run_cfg.get("system_prompt") or "").strip()
    url_preprocessing_enabled = bool(url_cfg.get("enabled", True))

    accepted: list[dict[str, Any]] = []
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
            genui_json = genui.get("genui_json") if genui.get("genui_json") is not None else genui.get("a2ui_json")
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
            validation = row_passes_basic_filters(
                response_text=response_text,
                genui_json=genui_json,
                validator=validator,
                max_input_chars=max_input_chars,
                max_output_chars=max_output_chars,
            )
            if not validation.valid:
                rejected.append({"response_id": response_id, "source_path": source_key, "reason": validation.reason})
                continue
            url_processed = preprocess_training_urls(
                response_text,
                genui_json,
                enabled=url_preprocessing_enabled,
            )
            completion = minify_json(url_processed.genui_json)
            dedupe_key = hashlib.sha256((url_processed.response_text + "\n" + completion).encode("utf-8")).hexdigest()
            if deduplicate and dedupe_key in seen_hashes:
                rejected.append({"response_id": response_id, "source_path": source_key, "reason": "duplicate_pair"})
                continue
            seen_hashes.add(dedupe_key)

            query_id = genui.get("query_id") or response.get("query_id")
            intent = genui.get("intent") or response.get("intent")
            tags = genui.get("tags") or response.get("tags") or []
            intent_bucket = genui.get("intent_bucket") or response.get("intent_bucket") or intent
            assets = genui.get("assets") or response.get("assets") or []
            expected_ui_contract = (
                genui.get("expected_ui_contract")
                or response.get("expected_ui_contract")
            )
            row_id = genui.get("ui_id") or f"u_{response_id}"
            prompt = build_prompt(system_prompt, url_processed.response_text)
            response_generation = _generation_metadata(response.get("gen"))
            ir_generation = _generation_metadata(genui.get("gen"))
            accepted.append(
                {
                    "id": f"{source_key}:{row_id}",
                    "response_id": response_id,
                    "messages": build_messages(system_prompt, url_processed.response_text, url_processed.genui_json),
                    "prompt": prompt,
                    "completion": completion,
                    # Preserve candidate-independent reward columns for an SFT -> GRPO handoff.
                    "source_id": str(query_id or response_id),
                    "response_text": url_processed.response_text,
                    "intent_bucket": intent_bucket,
                    "assets": assets,
                    "expected_ui_contract": expected_ui_contract,
                    "source_model_family": str(
                        ir_generation.get("model")
                        or response_generation.get("model")
                        or "unknown"
                    ),
                    "source_created_at": genui.get("created_at") or response.get("created_at"),
                    "metadata": {
                        "query_id": query_id,
                        "ui_id": genui.get("ui_id"),
                        "intent": intent,
                        "intent_bucket": intent_bucket,
                        "tags": tags,
                        "prompt_version": run_cfg.get("prompt_version"),
                        "schema_path": run_cfg.get("schema_path"),
                        "source_path": source_key,
                        "source_row_index": row_index,
                        "input_chars": len(url_processed.response_text),
                        "output_chars": len(completion),
                        "response_generation": response_generation,
                        "ir_generation": ir_generation,
                        "source_generation": ir_generation,
                        "url_preprocessing": {
                            "enabled": url_preprocessing_enabled,
                            "url_map": url_processed.url_map,
                            "metrics": url_processed.metrics,
                        },
                    },
                }
            )

    splits = stratified_split(
        accepted,
        train_ratio=float(split_cfg.get("train", 0.96)),
        val_ratio=float(split_cfg.get("val", 0.02)),
        test_ratio=float(split_cfg.get("test", 0.02)),
        stratify_key=str(split_cfg.get("stratify_by", "intent_bucket")),
        seed=int(run_cfg.get("seed", 42)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_count = write_jsonl(output_dir / "all.jsonl", accepted)
    counts = {name: write_jsonl(output_dir / f"{name}.jsonl", rows) for name, rows in splits.items()}
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
