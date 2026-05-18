from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ir_training.common.config import repo_root, resolve_path, training_root
from ir_training.common.git import current_commit
from ir_training.common.jsonl import load_by_key, write_jsonl
from ir_training.data.chat_templates import build_messages, build_prompt, minify_json
from ir_training.data.filters import FlatSpecValidator, row_passes_basic_filters
from ir_training.data.splits import stratified_split


def prepare_dataset(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    run_cfg = config.get("run") if isinstance(config.get("run"), dict) else {}
    filter_cfg = config.get("filters") if isinstance(config.get("filters"), dict) else {}
    split_cfg = config.get("split") if isinstance(config.get("split"), dict) else {}

    base = training_root()
    source_run_dir = resolve_path(run_cfg.get("source_run_dir", "../dataset/data/runs/dataset_v1"), base)
    output_dir = resolve_path(run_cfg.get("output_dir", "outputs/datasets/dataset_v1_stage3"), base)
    responses_path = source_run_dir / "responses.jsonl"
    genui_path = source_run_dir / "genui.jsonl"
    if not responses_path.exists():
        raise FileNotFoundError(f"Missing responses.jsonl: {responses_path}")
    if not genui_path.exists():
        raise FileNotFoundError(f"Missing genui.jsonl: {genui_path}")

    responses_by_id = load_by_key(responses_path, "response_id")
    genui_by_response_id = load_by_key(genui_path, "response_id")
    validator = FlatSpecValidator(require_strict=bool(filter_cfg.get("require_strict_flat_spec", True)))
    max_input_chars = int(filter_cfg.get("max_input_chars", 60000))
    max_output_chars = int(filter_cfg.get("max_output_chars", 60000))
    deduplicate = bool(filter_cfg.get("deduplicate", True))
    system_prompt = str(run_cfg.get("system_prompt") or "").strip()

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for response_id, response in sorted(responses_by_id.items()):
        genui = genui_by_response_id.get(response_id)
        if genui is None:
            rejected.append({"response_id": response_id, "reason": "missing_genui_record"})
            continue
        genui_json = genui.get("genui_json") if genui.get("genui_json") is not None else genui.get("a2ui_json")
        response_text = str(response.get("response_text") or genui.get("response_text") or "")
        validation = row_passes_basic_filters(
            response_text=response_text,
            genui_json=genui_json,
            validator=validator,
            max_input_chars=max_input_chars,
            max_output_chars=max_output_chars,
        )
        if not validation.valid:
            rejected.append({"response_id": response_id, "reason": validation.reason})
            continue
        completion = minify_json(genui_json)
        dedupe_key = hashlib.sha256((response_text + "\n" + completion).encode("utf-8")).hexdigest()
        if deduplicate and dedupe_key in seen_hashes:
            rejected.append({"response_id": response_id, "reason": "duplicate_pair"})
            continue
        seen_hashes.add(dedupe_key)

        query_id = response.get("query_id") or genui.get("query_id")
        intent = response.get("intent") or genui.get("intent")
        tags = response.get("tags") or genui.get("tags") or []
        intent_bucket = response.get("intent_bucket") or genui.get("intent_bucket") or intent
        row_id = genui.get("ui_id") or f"u_{response_id}"
        prompt = build_prompt(system_prompt, response_text)
        accepted.append(
            {
                "id": row_id,
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
    counts = {name: write_jsonl(output_dir / f"{name}.jsonl", rows) for name, rows in splits.items()}
    rejected_count = write_jsonl(output_dir / "rejected.jsonl", rejected)
    manifest = {
        "run_id": run_cfg.get("id", output_dir.name),
        "source_run_dir": str(source_run_dir),
        "responses_path": str(responses_path),
        "genui_path": str(genui_path),
        "output_dir": str(output_dir),
        "prompt_version": run_cfg.get("prompt_version"),
        "schema_path": run_cfg.get("schema_path"),
        "git_commit": current_commit(repo_root()),
        "counts": {**counts, "accepted": len(accepted), "rejected": rejected_count},
        "filters": filter_cfg,
        "split": split_cfg,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if config_path is not None:
        manifest["config_path"] = str(config_path)
    return manifest
