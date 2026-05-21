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


def prepare_dataset(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    run_cfg = config.get("run") if isinstance(config.get("run"), dict) else {}
    filter_cfg = config.get("filters") if isinstance(config.get("filters"), dict) else {}
    split_cfg = config.get("split") if isinstance(config.get("split"), dict) else {}

    base = training_root()
    output_dir = resolve_path(run_cfg.get("output_dir", "outputs/datasets/dataset_v1_stage3"), base)
    genui_paths = _collect_stage3_genui_paths(run_cfg, base)
    response_lookups = {path: _load_response_lookup(path) for path in genui_paths}
    validator = FlatSpecValidator(require_strict=bool(filter_cfg.get("require_strict_flat_spec", True)))
    max_input_chars = int(filter_cfg.get("max_input_chars", 60000))
    max_output_chars = int(filter_cfg.get("max_output_chars", 60000))
    deduplicate = bool(filter_cfg.get("deduplicate", True))
    system_prompt = str(run_cfg.get("system_prompt") or "").strip()

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
            completion = minify_json(genui_json)
            dedupe_key = hashlib.sha256((response_text + "\n" + completion).encode("utf-8")).hexdigest()
            if deduplicate and dedupe_key in seen_hashes:
                rejected.append({"response_id": response_id, "source_path": source_key, "reason": "duplicate_pair"})
                continue
            seen_hashes.add(dedupe_key)

            query_id = genui.get("query_id") or response.get("query_id")
            intent = genui.get("intent") or response.get("intent")
            tags = genui.get("tags") or response.get("tags") or []
            intent_bucket = genui.get("intent_bucket") or response.get("intent_bucket") or intent
            row_id = genui.get("ui_id") or f"u_{response_id}"
            prompt = build_prompt(system_prompt, response_text)
            accepted.append(
                {
                    "id": f"{source_key}:{row_id}",
                    "response_id": response_id,
                    "messages": build_messages(system_prompt, response_text, genui_json),
                    "prompt": prompt,
                    "completion": completion,
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
                        "input_chars": len(response_text),
                        "output_chars": len(completion),
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
        "filters": filter_cfg,
        "split": split_cfg,
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
