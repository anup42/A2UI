"""Fail-closed, offline supervision for opt-in QAT post-training methods.

The caller first verifies the existing prepared training contract (including
reserved Golden/Bixby cohorts). These helpers accept only that train split;
they never load/generate alternative sources, repair targets, or silently drop
bad examples. URL restoration is for validation only, never a target rewrite.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

ADMISSION_VERSION = "qat_posttrain_source_fidelity_v1"
REHEARSAL_TOKEN_CACHE_SIZE = 64


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _StreamingRowsDigest:
    """Hash the canonical JSON array without materializing it or its rows."""

    def __init__(self) -> None:
        self.count = 0
        self._digest = hashlib.sha256(b"[")

    def add(self, row: Mapping[str, Any]) -> None:
        if self.count:
            self._digest.update(b",")
        self._digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode("utf-8"))
        self.count += 1

    def hexdigest(self) -> str:
        result = self._digest.copy()
        result.update(b"]")
        return result.hexdigest()


class RehearsalBank(Mapping[str, dict]):
    """Read-only prompt lookup with at most 64 resident tokenized examples.

    All rows were fully admitted at construction. Only compact text/bindings
    remain resident; lazy tokenization must reproduce each admitted token hash.
    Callers must not mutate the returned token records. Index lookup avoids
    retaining another full copy of prompts in the training integration.
    """

    def __init__(self, entries: list[dict], tokenizer: Any, limits: Mapping[str, int]) -> None:
        self._entries = entries
        self._tokenizer = tokenizer
        self._limits = dict(limits)
        self.prompt_to_index = {entry["prompt"]: index for index, entry in enumerate(entries)}
        self._cache: OrderedDict[int, dict] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self.prompt_to_index)

    def __getitem__(self, prompt: str) -> dict:
        return self.tokenized(self.prompt_to_index[prompt])

    def tokenized(self, index: int) -> dict:
        if type(index) is not int or not 0 <= index < len(self._entries):
            raise IndexError("Rehearsal row index is outside the admitted bank")
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        entry = self._entries[index]
        tokens = tokenize_supervised_target(self._tokenizer, entry["prompt"], entry["completion"], **self._limits)
        result = {**tokens, "source_id": entry["source_id"], "prompt_sha256": entry["prompt_sha256"],
                  "completion_sha256": entry["completion_sha256"]}
        if _digest(result) != entry["token_record_sha256"]:
            raise ValueError(f"{entry['source_id']}: rehearsal tokenization changed after strict admission")
        self._cache[index] = result
        if len(self._cache) > REHEARSAL_TOKEN_CACHE_SIZE:
            self._cache.popitem(last=False)
        return result


def _required_text(row: Mapping[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {key} must be a nonempty string")
    return value


def _training_index(train_rows: Iterable[Mapping[str, Any]],
                    forbidden_rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, dict], str]:
    forbidden: set[tuple[str, str]] = set()
    for row in forbidden_rows:
        for key in ("source_id", "response_text", "prompt"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                forbidden.add((key, value.strip()))
    indexed: dict[str, dict] = {}
    prompts: set[str] = set()
    bindings = _StreamingRowsDigest()
    for offset, original in enumerate(train_rows):
        if not isinstance(original, Mapping):
            raise TypeError(f"Train row {offset}: expected an object")
        row = dict(original)
        source_id = _required_text(row, "source_id", f"Train row {offset}")
        source = _required_text(row, "response_text", source_id)
        prompt = _required_text(row, "prompt", source_id)
        _required_text(row, "completion", source_id)
        if source_id in indexed or prompt in prompts:
            raise ValueError(f"{source_id}: duplicate/ambiguous train source or prompt")
        for key in ("source_id", "response_text", "prompt"):
            if (key, row[key].strip()) in forbidden:
                raise ValueError(f"{source_id}: train/held-out overlap in {key}")
        actual_source_hash = _text_hash(source)
        if row.get("response_text_sha256") not in (None, actual_source_hash):
            raise ValueError(f"{source_id}: response_text_sha256 does not bind the train source")
        row["response_text_sha256"] = actual_source_hash
        row["prompt_sha256"] = _text_hash(prompt)
        indexed[source_id] = row
        prompts.add(prompt)
        bindings.add({"source_id": source_id, "response_text_sha256": actual_source_hash,
                      "prompt_sha256": row["prompt_sha256"],
                      "source_context_sha256": row.get("source_context_sha256"),
                      "completion_sha256": _text_hash(row["completion"]),
                      "reward_context_sha256": _digest({key: row.get(key) for key in
                          ("url_map", "intent_bucket", "assets", "expected_ui_contract")})})
    if not indexed:
        raise ValueError("QAT post-training requires a nonempty verified train split")
    return indexed, bindings.hexdigest()


def _reward_config(config: Any) -> Any:
    from pipeline.genui_quality.config_v5_4 import (
        RewardConfigV54,
        coerce_reward_config_v5_4,
        load_reward_config_v5_4,
    )
    if isinstance(config, (str, Path)):
        config = load_reward_config_v5_4(config)
    elif isinstance(config, Mapping):
        config = RewardConfigV54.from_mapping(config)
    resolved = replace(coerce_reward_config_v5_4(config), grpo_entity_extraction=True)
    if resolved.unsupported_external_additions_policy != "penalize":
        raise ValueError("QAT post-training requires unsupported_external_additions_policy=penalize")
    return resolved


def _score_targets(row: Mapping[str, Any], completions: list[str], config: Any) -> list[Any]:
    from pipeline.genui_quality._v5_4 import (
        prepare_source_context_v5_4,
        score_completion_group_v5_4,
    )

    from ir_training.data.url_preprocess import restore_url_placeholders
    from ir_training.train.qat_grpo_reward import map_qat_reward
    prepared = prepare_source_context_v5_4(
        row["response_text"], intent=row.get("intent_bucket"), assets=row.get("assets"),
        expected_ui_contract=row.get("expected_ui_contract"), config=config,
    )
    restored = [restore_url_placeholders(completion, row.get("url_map") or {})
                for completion in completions]
    return [map_qat_reward(result) for result in score_completion_group_v5_4(
        restored, prepared, generation_mode=True, active_express=True)]


def _require_supervised_target(row: Mapping[str, Any], completion: str, score: Any) -> None:
    from pipeline.ir_formats import compile_express_to_wire, validate_express_completion

    from ir_training.data.url_preprocess import restore_url_placeholders
    source_id = row["source_id"]
    # No serving-stop cleanup or repair is allowed for a supervised label.
    restored = restore_url_placeholders(completion, row.get("url_map") or {})
    validation = validate_express_completion(restored)
    if not validation.raw_valid or validation.canonical_graph is None:
        raise ValueError(f"{source_id}: supervised target is not raw-valid A2UI Express")
    try:
        compile_express_to_wire(validation.canonical_graph)
    except Exception as exc:
        raise ValueError(f"{source_id}: supervised target cannot compile to native wire: {exc}") from exc
    if not score.strict_valid or score.reasons or score.canonical_caps:
        reasons = ", ".join(score.reasons) or "strict validity/canonical cap"
        raise ValueError(f"{source_id}: supervised target fails full source fidelity: {reasons}")
    _require_prompt_visible_references(row, completion)


def _require_prompt_visible_references(row: Mapping[str, Any], completion: str) -> None:
    """Do not teach restored/hidden reference values from a masked input.

    Decode the *raw* target's graph strings to catch JSON/Express escapes. This
    is admission only: labels are never rewritten. Rejected DPO candidates are
    deliberately not checked here because bad references can be useful negatives.
    """
    from pipeline.ir_formats import validate_express_completion

    from ir_training.data.url_preprocess import _PLACEHOLDER_RE
    source_id = row["source_id"]
    prompt = row["prompt"]
    url_map = row.get("url_map") or {}
    validation = validate_express_completion(completion)
    if not validation.raw_valid or validation.canonical_graph is None:
        raise ValueError(f"{source_id}: supervised raw target cannot preserve its reference bindings")

    def strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    values = tuple(strings(validation.canonical_graph))
    for text in values:
        for token in _PLACEHOLDER_RE.findall(text):
            entry = url_map.get(token)
            resolved = entry.get("url") if isinstance(entry, Mapping) else entry
            if not isinstance(resolved, str) or not resolved or token not in prompt:
                raise ValueError(f"{source_id}: supervised placeholder {token} must be bound and prompt-visible")
    for entry in url_map.values():
        url = entry.get("url") if isinstance(entry, Mapping) else entry
        if isinstance(url, str) and url and any(url in value for value in values) and url not in prompt:
            raise ValueError(f"{source_id}: supervised literal reference is hidden by the masked prompt")


def tokenize_supervised_target(tokenizer: Any, prompt: str, completion: str, *,
                               max_prompt_length: int, max_completion_length: int,
                               max_sequence_length: int) -> dict[str, Any]:
    """Encode exact rendered text, mask prompts, and include one EOS label.

    Refuse prompt/completion boundary merges, truncation and embedded EOS. A
    rendered chat prompt already owns its BOS/turn markers; none are injected.
    """
    for name, value in (("max_prompt_length", max_prompt_length),
                        ("max_completion_length", max_completion_length),
                        ("max_sequence_length", max_sequence_length)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(completion, str) or not completion.strip():
        raise ValueError("Supervision requires nonempty prompt and completion")
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or eos < 0:
        raise ValueError("Supervision requires exactly one tokenizer EOS token id")

    def encode(text: str) -> list[int]:
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        ids = encoded["input_ids"]
        if not isinstance(ids, (list, tuple)) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Tokenizer must return a flat list of token ids")
        return list(ids)

    prompt_ids = encode(prompt)
    full_ids = encode(prompt + completion)
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer changes the exact prompt/completion boundary")
    completion_ids = full_ids[len(prompt_ids):]
    if not completion_ids or eos in completion_ids:
        raise ValueError("Completion must have nonempty labels without embedded EOS tokens")
    completion_ids.append(eos)
    full_ids.append(eos)
    if len(prompt_ids) > max_prompt_length:
        raise ValueError(f"Prompt exceeds budget: {len(prompt_ids)} > {max_prompt_length}; no truncation")
    if len(completion_ids) > max_completion_length:
        raise ValueError(f"Completion exceeds budget: {len(completion_ids)} > {max_completion_length}; no truncation")
    if len(full_ids) > max_sequence_length:
        raise ValueError(f"Sequence exceeds budget: {len(full_ids)} > {max_sequence_length}; no truncation")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prompt_ids) + completion_ids,
            "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids),
            "prompt_input_ids": prompt_ids, "completion_tokens": len(completion_ids)}


def _manifest(kind: str, rows: list[dict] | _StreamingRowsDigest, train_binding: str, config: Any,
              limits: Mapping[str, int]) -> dict[str, Any]:
    from pipeline.genui_quality.identity_v5_4 import (
        metric_fingerprint_v5_4,
        reward_pipeline_fingerprint_v5_4,
    )
    fingerprint = metric_fingerprint_v5_4(config)
    rows_count = rows.count if isinstance(rows, _StreamingRowsDigest) else len(rows)
    rows_hash = rows.hexdigest() if isinstance(rows, _StreamingRowsDigest) else _digest(rows)
    return {"schema_version": 1, "admission_policy": ADMISSION_VERSION, "kind": kind,
            "scope": "verified train split only; no regeneration, repair or truncation",
            "rows": rows_count, "admitted_rows_sha256": rows_hash,
            "train_split_binding_sha256": train_binding,
            "admission_implementation_sha256": _text_hash(Path(__file__).read_text(encoding="utf-8")),
            "reward_mapping_sha256": _reward_mapping_hash(),
            "metric_fingerprint": fingerprint,
            "reward_pipeline_fingerprint": reward_pipeline_fingerprint_v5_4(fingerprint),
            "token_limits": dict(limits)}


def _reward_mapping_hash() -> str:
    from ir_training.train import qat_grpo_reward
    return _text_hash(Path(qat_grpo_reward.__file__).read_text(encoding="utf-8"))


def verify_admission_manifest(manifest: Any, *, kind: str, reward_config: Any = None,
                              expected_limits: Mapping[str, int] | None = None) -> dict:
    """Verify saved supervision evidence before selection/merge/export.

    Artifact/file hashes and prepared-split provenance remain the caller's
    responsibility. This CPU-only check prevents stale, missing or relabeled
    admission evidence from passing the existing checkpoint contract.
    """
    from pipeline.genui_quality.identity_v5_4 import (
        metric_fingerprint_v5_4,
        reward_pipeline_fingerprint_v5_4,
    )
    if kind not in {"grpo_sft_rehearsal", "qat_dpo"}:
        raise ValueError(f"Unsupported post-training admission kind: {kind}")
    if not isinstance(manifest, Mapping):
        raise ValueError("Post-training checkpoint lacks an admission manifest")  # noqa: TRY004
    if (type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1
            or manifest.get("admission_policy") != ADMISSION_VERSION or manifest.get("kind") != kind):
        raise ValueError("Post-training admission schema, policy or method does not match")
    if type(manifest.get("rows")) is not int or manifest["rows"] <= 0:
        raise ValueError("Post-training admission rows must be a positive integer")
    digest_keys = ("admitted_rows_sha256", "train_split_binding_sha256", "admission_implementation_sha256",
                   "reward_mapping_sha256", "metric_fingerprint", "reward_pipeline_fingerprint")
    for key in digest_keys:
        if not isinstance(manifest.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", manifest[key]) is None:
            raise ValueError(f"Post-training admission has invalid digest: {key}")
    if manifest["admission_implementation_sha256"] != _text_hash(Path(__file__).read_text(encoding="utf-8")):
        raise ValueError("Post-training admission implementation has changed")
    if manifest["reward_mapping_sha256"] != _reward_mapping_hash():
        raise ValueError("Post-training admission reward mapping has changed")
    fingerprint = metric_fingerprint_v5_4(_reward_config(reward_config))
    if manifest["metric_fingerprint"] != fingerprint or manifest["reward_pipeline_fingerprint"] != reward_pipeline_fingerprint_v5_4(fingerprint):
        raise ValueError("Post-training admission scorer fingerprints do not match")
    limits = manifest.get("token_limits")
    keys = {"max_prompt_length", "max_completion_length", "max_sequence_length"}
    if (not isinstance(limits, Mapping) or set(limits) != keys
            or any(type(value) is not int or value <= 0 for value in limits.values())):
        raise ValueError("Post-training admission token limits are invalid")
    if expected_limits is not None and (not isinstance(expected_limits, Mapping)
            or set(expected_limits) != keys
            or any(type(value) is not int or value <= 0 for value in expected_limits.values())
            or dict(limits) != dict(expected_limits)):
        raise ValueError("Post-training admission token limits do not match the checkpoint configuration")
    if kind == "qat_dpo":
        if not isinstance(manifest.get("pairs_sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", manifest["pairs_sha256"]) is None:
            raise ValueError("DPO admission lacks the exact pair-file hash")
        count, gap = manifest.get("unique_train_sources"), manifest.get("min_reward_gap")
        if type(count) is not int or not 0 < count <= manifest["rows"]:
            raise ValueError("DPO admission source count is invalid")
        if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap) or not 0 < gap <= 2:
            raise ValueError("DPO admission reward gap is invalid")
    return dict(manifest)


def build_rehearsal_bank(train_rows: Iterable[Mapping[str, Any]], tokenizer: Any, *,
                         max_prompt_length: int, max_completion_length: int,
                         max_sequence_length: int, reward_config: Any = None,
                         forbidden_rows: Iterable[Mapping[str, Any]] = ()) -> tuple[RehearsalBank, dict]:
    """Validate the existing SFT targets; key by the exact rollout prompt."""
    indexed, train_binding = _training_index(train_rows, forbidden_rows)
    config = _reward_config(reward_config)
    limits = {"max_prompt_length": max_prompt_length, "max_completion_length": max_completion_length,
              "max_sequence_length": max_sequence_length}
    entries = []
    admitted = _StreamingRowsDigest()
    for row in indexed.values():
        target = row["completion"]
        score = _score_targets(row, [target], config)[0]
        _require_supervised_target(row, target, score)
        try:
            tokens = tokenize_supervised_target(tokenizer, row["prompt"], target, **limits)
        except ValueError as exc:
            raise ValueError(f"{row['source_id']}: {exc}") from exc
        record = {**tokens, "source_id": row["source_id"], "prompt_sha256": row["prompt_sha256"],
                  "completion_sha256": _text_hash(target)}
        admitted.add(record)
        entries.append({"source_id": row["source_id"], "prompt": row["prompt"], "completion": target,
                        "prompt_sha256": row["prompt_sha256"], "completion_sha256": record["completion_sha256"],
                        "token_record_sha256": _digest(record)})
    return RehearsalBank(entries, tokenizer, limits), _manifest(
        "grpo_sft_rehearsal", admitted, train_binding, config, limits)


def prepare_dpo_pairs(pair_path: str | Path, train_rows: Iterable[Mapping[str, Any]], tokenizer: Any, *,
                      max_prompt_length: int, max_completion_length: int,
                      max_sequence_length: int, reward_config: Any = None,
                      forbidden_rows: Iterable[Mapping[str, Any]] = (),
                      min_reward_gap: float = 1e-6) -> tuple[list[dict], dict]:
    """Admit strict offline pairs joined to the verified train split.

    JSONL objects require ``source_id``, ``chosen``, ``rejected``. Optional
    ``pair_id``, ``response_text_sha256``, ``source_context_sha256`` and
    ``prompt_sha256`` bind upstream audit evidence. Pair files cannot override
    source text, prompts, contracts, URL bindings or train/evaluation splits.
    """
    if isinstance(min_reward_gap, bool) or not isinstance(min_reward_gap, (int, float)) or not math.isfinite(min_reward_gap) or min_reward_gap <= 0:
        raise ValueError("min_reward_gap must be finite and strictly positive")
    indexed, train_binding = _training_index(train_rows, forbidden_rows)
    config = _reward_config(reward_config)
    limits = {"max_prompt_length": max_prompt_length, "max_completion_length": max_completion_length,
              "max_sequence_length": max_sequence_length}
    path = Path(pair_path)
    raw_bytes = path.read_bytes()
    allowed = {"source_id", "chosen", "rejected", "pair_id", "response_text_sha256",
               "source_context_sha256", "prompt_sha256"}
    admitted, seen_pairs, seen_ids = [], set(), set()

    def unique_object(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field {key}")
            result[key] = value
        return result

    for line_number, line in enumerate(raw_bytes.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        context = f"DPO line {line_number}"
        try:
            pair = json.loads(line, object_pairs_hook=unique_object)
        except ValueError as exc:
            raise ValueError(f"{context}: malformed JSON") from exc
        if not isinstance(pair, Mapping) or set(pair) - allowed:
            raise ValueError(f"{context}: expected an object containing only documented pair fields")
        source_id = _required_text(pair, "source_id", context)
        context += f" source_id={source_id}"
        if source_id not in indexed:
            raise ValueError(f"{context}: source is not in the verified train split")
        row = indexed[source_id]
        for key in ("response_text_sha256", "source_context_sha256", "prompt_sha256"):
            if key in pair and (not pair[key] or pair[key] != row.get(key)):
                raise ValueError(f"{context}: mismatched {key}")
        chosen = _required_text(pair, "chosen", context)
        rejected = _required_text(pair, "rejected", context)
        if chosen.strip() == rejected.strip():
            raise ValueError(f"{context}: identical/ambiguous chosen and rejected")
        unordered_binding = _digest([source_id, sorted([chosen.strip(), rejected.strip()])])
        if unordered_binding in seen_pairs:
            raise ValueError(f"{context}: duplicate or reversed preference pair")
        pair_id = _required_text(pair, "pair_id", context) if "pair_id" in pair else _digest([source_id, chosen, rejected])
        if pair_id in seen_ids:
            raise ValueError(f"{context}: duplicate pair_id")
        scores = _score_targets(row, [chosen, rejected], config)
        _require_supervised_target(row, chosen, scores[0])
        gap = scores[0].reward - scores[1].reward
        if not math.isfinite(gap) or gap < min_reward_gap:
            raise ValueError(f"{context}: no grounded positive preference gap ({gap} < {min_reward_gap})")
        result = {"source_id": source_id, "pair_id": pair_id, "prompt": row["prompt"],
                  "prompt_sha256": row["prompt_sha256"], "chosen": chosen, "rejected": rejected,
                  "chosen_reward": scores[0].reward, "rejected_reward": scores[1].reward,
                  "reward_gap": gap, "response_text_sha256": row["response_text_sha256"],
                  "source_context_sha256": row.get("source_context_sha256")}
        for name, target in (("chosen", chosen), ("rejected", rejected)):
            try:
                tokens = tokenize_supervised_target(tokenizer, row["prompt"], target, **limits)
            except ValueError as exc:
                raise ValueError(f"{context} {name}: {exc}") from exc
            result.update({f"{name}_{key}": value for key, value in tokens.items() if key != "prompt_input_ids"})
            result["prompt_input_ids"] = tokens["prompt_input_ids"]
        if result["chosen_input_ids"] == result["rejected_input_ids"]:
            raise ValueError(f"{context}: chosen and rejected tokenize identically; no learnable preference")
        admitted.append(result)
        seen_pairs.add(unordered_binding)
        seen_ids.add(pair_id)
    if not admitted:
        raise ValueError("DPO pair file contains no preference pairs")
    manifest = _manifest("qat_dpo", admitted, train_binding, config, limits)
    manifest.update({"pairs_path": str(path.resolve()), "pairs_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                     "min_reward_gap": float(min_reward_gap),
                     "unique_train_sources": len({row["source_id"] for row in admitted})})
    return admitted, manifest
