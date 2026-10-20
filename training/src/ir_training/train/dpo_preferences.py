"""Offline, train-only preference mining from source-bound GRPO audit batches.

No generations, target repairs, held-out scoring, or changes to admission rules.
The caller must first verify the prepared run using ``load_preference_run``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ir_training.train.qat_posttrain_data import (
    _digest,
    _require_supervised_target,
    _reward_config,
    _score_targets,
    _text_hash,
    _training_index,
    prepare_dpo_pairs,
    tokenize_supervised_target,
)

AUDIT_NAME = re.compile(r"grpo_periodic_rewards\.rank([0-9]+)\.jsonl")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _verify_files(files: Mapping[str, str]) -> None:
    for path, expected in files.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Input changed during preference preparation: {path}; use a stopped run")


def build_preferences(audit_paths: Iterable[Path], train_rows: Iterable[Mapping[str, Any]],
                      tokenizer: Any, output_dir: str | Path, *, limits: Mapping[str, int],
                      reward_config: Any = None, forbidden_rows: Iterable[Mapping[str, Any]] = (),
                      min_reward_gap: float = 0.01, min_pairs: int = 32,
                      evidence: Mapping[str, Any] | None = None) -> dict:
    """Publish one best-faithful/hard-negative pair per verified train source.

    Invalid candidates are counted, not repaired. Provenance/mode/prompt errors
    fail the build. Final publication also requires unchanged canonical DPO
    admission. The fresh output directory retains a failure report on errors,
    but never a usable ``preferences.jsonl`` from a failed build.
    """
    if (isinstance(min_reward_gap, bool) or not isinstance(min_reward_gap, (int, float))
            or not math.isfinite(min_reward_gap) or not 0 < min_reward_gap <= 2):
        raise ValueError("min_reward_gap must be finite and in (0, 2]")
    if type(min_pairs) is not int or min_pairs < 1:
        raise ValueError("min_pairs must be a positive integer")
    expected_limits = {"max_prompt_length", "max_completion_length", "max_sequence_length"}
    if set(limits) != expected_limits or any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError("Expected three positive integer token limits")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "policy": "grpo_train_audit_to_dpo_v1", "status": "building",
              "pairs": 0, "min_pairs": min_pairs, "min_reward_gap": min_reward_gap,
              "token_limits": dict(limits), "selection": "one_best_faithful_chosen_and_hard_negative_per_source",
              "audit_scope": "saved periodic first-local-batch samples, not a full-run census",
              "counts": {}, "audit_files": {}, "run_evidence": dict(evidence or {}),
              "builder_sha256": file_sha256(Path(__file__))}
    counts: Counter = Counter()
    try:
        # Iterables are indexed once; both selection and final admission see the
        # identical source context, including validation exclusion.
        forbidden = list(forbidden_rows)
        indexed, train_binding = _training_index(train_rows, forbidden)
        report["train_split_binding_sha256"] = train_binding
        config = _reward_config(reward_config)
        inputs = dict((evidence or {}).get("files", {}))
        _verify_files(inputs)
        paths = sorted({Path(path).resolve(strict=True) for path in audit_paths}, key=str)
        if not paths:
            raise ValueError("No periodic GRPO reward audits; saved coverage is insufficient")
        groups: dict[str, dict[str, dict]] = {}
        for path in paths:
            name = AUDIT_NAME.fullmatch(path.name)
            if name is None:
                raise ValueError(f"Only periodic GRPO reward audits are supported: {path}")
            rank = int(name[1])
            before = file_sha256(path)
            if str(path) in inputs and inputs[str(path)] != before:
                raise ValueError(f"Audit changed after run verification: {path}; use a stopped run")
            inputs[str(path)] = before
            report["audit_files"][str(path)] = before
            with path.open(encoding="utf-8-sig") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    context = f"{path.name}:{line_number}"
                    try:
                        record = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                    except ValueError as exc:
                        raise ValueError(f"{context}: malformed audit JSON: {exc}") from exc
                    counts["audit_records"] += 1
                    if not isinstance(record, dict) or record.get("mode") not in {"train", "eval"}:
                        raise ValueError(f"{context}: missing/unknown train/eval mode")
                    if record["mode"] == "eval":
                        counts["excluded_eval_records"] += 1
                        continue
                    source_id = record.get("source_id")
                    if not isinstance(source_id, str) or source_id not in indexed:
                        raise ValueError(f"{context}: source is not in the verified train split")
                    source = indexed[source_id]
                    if record.get("prompt_sha256") != source["prompt_sha256"]:
                        raise ValueError(f"{context}: prompt_sha256 mismatch for {source_id}")
                    for key in ("step", "rank", "local_index"):
                        if type(record.get(key)) is not int or record[key] < 0:
                            raise ValueError(f"{context}: invalid audit {key}")
                    if record["rank"] != rank:
                        raise ValueError(f"{context}: audit rank does not match filename")
                    raw = record.get("raw_completion")
                    if not isinstance(raw, str) or not raw.strip():
                        raise ValueError(f"{context}: raw_completion must be nonempty text")
                    reason = record.get("termination_reason")
                    if reason in {"max_new_tokens", "unknown"}:
                        counts["excluded_unterminated"] += 1
                        continue
                    if reason not in {"eos_token", "closing_sentinel"}:
                        raise ValueError(f"{context}: missing/unknown termination_reason")
                    candidates = groups.setdefault(source_id, {})
                    digest = _text_hash(raw)
                    if digest in candidates:
                        if candidates[digest]["text"] != raw:
                            raise ValueError("Completion hash collision")
                        candidates[digest]["occurrences"] += 1
                        counts["duplicate_candidates"] += 1
                        continue
                    candidates[digest] = {"text": raw, "sha256": digest, "occurrences": 1,
                        "origin": {"path": str(path), "line": line_number,
                                   **{k: record[k] for k in ("step", "rank", "local_index", "termination_reason")}}}
            if file_sha256(path) != before:
                raise ValueError(f"Audit changed while reading: {path}; use a stopped run")
        pairs, provenance = [], []
        diagnostics_path = output / "candidate_diagnostics.jsonl"
        with diagnostics_path.open("x", encoding="utf-8", newline="\n") as diagnostics:
            for source_id in sorted(groups):
                row = indexed[source_id]
                candidates = []
                for digest, candidate in sorted(groups[source_id].items()):
                    raw = candidate["text"]
                    counts["unique_candidates"] += 1
                    diagnostic = {"source_id": source_id, "completion_sha256": digest, **candidate["origin"]}
                    try:
                        tokens = tokenize_supervised_target(tokenizer, row["prompt"], raw, **limits)
                    except ValueError as exc:
                        counts["excluded_token_budget_or_eos"] += 1
                        diagnostic.update(eligible=False, reason=str(exc))
                    else:
                        score = _score_targets(row, [raw], config)[0]
                        if not math.isfinite(score.reward):
                            raise ValueError(f"{source_id}: non-finite current reward")
                        chosen_ok, failure = True, None
                        try:
                            _require_supervised_target(row, raw, score)
                        except ValueError as exc:
                            chosen_ok, failure = False, str(exc)
                            counts["not_eligible_as_chosen"] += 1
                        candidate.update(reward=score.reward, chosen_ok=chosen_ok,
                                         token_sha256=_digest(tokens["input_ids"]),
                                         reasons=list(score.reasons), canonical_caps=list(score.canonical_caps))
                        candidates.append(candidate)
                        diagnostic.update(eligible=True, chosen_eligible=chosen_ok, chosen_failure=failure,
                                          reward=score.reward, reasons=list(score.reasons),
                                          canonical_caps=list(score.canonical_caps))
                    diagnostics.write(json.dumps(diagnostic, ensure_ascii=False, allow_nan=False) + "\n")
                # Highest lower-scoring negative is more informative than always
                # selecting a broken-syntax minimum. Stable hash tie-breakers.
                candidates.sort(key=lambda item: (-item["reward"], item["sha256"]))
                chosen = next((item for item in candidates if item["chosen_ok"]), None)
                if chosen is None:
                    counts["sources_without_faithful_chosen"] += 1
                    continue
                rejected = next((item for item in candidates
                    if chosen["reward"] - item["reward"] >= min_reward_gap
                    and chosen["token_sha256"] != item["token_sha256"]
                    and chosen["text"].strip() != item["text"].strip()), None)
                if rejected is None:
                    counts["sources_without_positive_gap"] += 1
                    continue
                pair_id = _digest([source_id, chosen["text"], rejected["text"]])
                pair = {"pair_id": pair_id, "source_id": source_id,
                        "chosen": chosen["text"], "rejected": rejected["text"],
                        "prompt_sha256": row["prompt_sha256"],
                        "response_text_sha256": row["response_text_sha256"]}
                if row.get("source_context_sha256"):
                    pair["source_context_sha256"] = row["source_context_sha256"]
                pairs.append(pair)
                provenance.append({"pair_id": pair_id, "source_id": source_id,
                    "reward_gap": chosen["reward"] - rejected["reward"],
                    **{label: {key: item[key] for key in ("sha256", "reward", "origin", "occurrences",
                                                        "reasons", "canonical_caps")}
                       for label, item in (("chosen", chosen), ("rejected", rejected))}})
        report["pairs"] = len(pairs)
        if len(pairs) < min_pairs:
            raise ValueError(f"Insufficient faithful preference pairs: {len(pairs)} < {min_pairs}. "
                             "Saved audits may lack coverage; do not relax fidelity or use held-out outputs.")
        pending = output / "preferences.pending.jsonl"
        _jsonl(pending, pairs)
        admitted, manifest = prepare_dpo_pairs(pending, indexed.values(), tokenizer, **limits,
            reward_config=config, forbidden_rows=forbidden, min_reward_gap=min_reward_gap)
        if len(admitted) != len(pairs):
            raise ValueError("Final admission changed the selected pair count")
        _verify_files(inputs)
        if file_sha256(pending) != manifest["pairs_sha256"]:
            raise ValueError("Preference candidate file changed during admission")
        final_path = output / "preferences.jsonl"
        manifest["pairs_path"] = str(final_path)
        _jsonl(output / "preference_provenance.jsonl", provenance)
        _json(output / "preference_admission.json", manifest)
        report.update(status="passed", preferences_path=str(final_path),
                      preferences_sha256=manifest["pairs_sha256"], counts=dict(counts))
        _json(output / "preference_build_report.json", report)
        # Publish the usable file last; all provenance and checks precede it.
        pending.rename(final_path)
        return report
    except Exception as exc:
        report.update(status="failed", counts=dict(counts), error=f"{type(exc).__name__}: {exc}")
        _json(output / "preference_build_report.json", report)
        raise
